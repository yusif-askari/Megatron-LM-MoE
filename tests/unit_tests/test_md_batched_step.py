"""MDDecoupling batched step (MD_BATCHED_STEP) against the per-matrix step.

Optimizer settings of the ladder runs (Muon + Nesterov, polar_express 5 steps, fp32 matmul precision
"medium", shape_up, row hypersphere with fan_in radius, rowcol softplus gains, GLU fc1 split). Parameters:
the per-GPU set of the 1.5B or 5B rung (SIZES) - per MoE layer 64 local experts (fc1 GLU [2 expert, latent],
fc2 [latent, expert]), a shared expert, the latent projections each way and a router (per-matrix path); the
dense layer-0 MLP - as fp32 tensors with the attributes Megatron sets. TP size 1.
1. Same parameters and gradients through both paths for 3 steps: parameters, momentum and gains compared
   (relative difference per matrix, max over matrices; bitwise-equal fraction).
2. Time of optimizer.step() (median of 10, after 3 warm-up steps) and kernels per step (torch.profiler).
Run: python test_md_batched_step.py out.json [1.5b|5b]   (one GPU; default 1.5b)."""
import copy
import json
import statistics
import sys
import time

import torch
import torch.distributed as dist

from megatron.core.optimizer.md_decoupling import MDDecoupling
from megatron.core.process_groups_config import ProcessGroupCollection

dev = torch.device("cuda", 0)


# per-GPU parameter shapes (EP 4: 64 local experts per MoE layer)
SIZES = {
    "1.5b": dict(layers=10, moe_layers=9, hidden=768, latent=384, expert=448, shared=448, ffn=1920),
    "5b": dict(layers=14, moe_layers=13, hidden=1280, latent=640, expert=704, shared=704, ffn=3200),
}


def make_params(g, size):
    z = SIZES[size]
    L, h, lat, e, sh, ffn = z["moe_layers"], z["hidden"], z["latent"], z["expert"], z["shared"], z["ffn"]
    col = dict(partition_dim=0, tensor_model_parallel=True)
    row = dict(partition_dim=1, tensor_model_parallel=True, is_out_proj=True)
    spec = [   # (shape, attrs, count)
        ((2 * e, lat), dict(col, glu_split_dim=0, expert_tp=True), 64 * L),
        ((lat, e), dict(row, expert_tp=True), 64 * L),
        ((2 * sh, h), dict(col, glu_split_dim=0), L),
        ((h, sh), dict(row), L),
        ((2 * ffn, h), dict(col, glu_split_dim=0), z["layers"] - L),
        ((h, ffn), dict(row), z["layers"] - L),
        ((lat, h), dict(partition_dim=0, tensor_model_parallel=False), L),
        ((h, lat), dict(partition_dim=0, tensor_model_parallel=False), L),
        ((256, h), dict(is_router=True), L),
    ]
    ps = []
    for shape, attrs, n in spec:
        for _ in range(n):
            p = torch.nn.Parameter(torch.randn(*shape, device=dev, generator=g) / shape[1] ** 0.5)
            for k, v in attrs.items():
                setattr(p, k, v)
            ps.append(p)
    return ps


def clone_params(ps):
    out = []
    for p in ps:
        q = torch.nn.Parameter(p.detach().clone())
        for k, v in p.__dict__.items():
            setattr(q, k, v)
        out.append(q)
    return out


def make_opt(ps, pgc, batched, hidden):
    opt = MDDecoupling(
        [{"params": ps, "lr": 1e-2, "max_lr": 1e-2, "min_lr": 1e-3}],
        hypersphere_mode="row", hypersphere_embedding_mode="row", hypersphere_router_mode="row",
        hypersphere_radius_mode="fan_in", hidden_size=hidden, use_orthogonal_updates=True,
        momentum_beta=0.95, use_nesterov=True, split_fc1=True, fp32_matmul_prec="medium",
        coefficient_type="polar_express", num_ns_steps=5, scale_mode="shape_up", router_scale_mode="none",
        pg_collection=pgc, tp_mode="duplicated", hypersphere_gains_mode="rowcol",
        hypersphere_gains_mode_router="rowcol", hypersphere_gains_mode_embedding="none",
        gains_lr=1e-3, gain_parametrization="softplus")
    opt.batched_step = batched
    return opt


def set_grads(ps, g, scale=1e-3):
    for p in ps:
        p.grad = torch.randn(p.shape, device=dev, generator=g) * scale


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def main():
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29541", rank=0, world_size=1)
    g1 = dist.new_group([0])
    pgc = ProcessGroupCollection(tp=g1, expt_tp=g1)
    g = torch.Generator(device=dev).manual_seed(0)
    size = sys.argv[2] if len(sys.argv) > 2 else "1.5b"
    base = make_params(g, size)
    res = {"size": size}
    pa, pb = clone_params(base), clone_params(base)
    hid = SIZES[size]["hidden"]
    oa, ob = make_opt(pa, pgc, False, hid), make_opt(pb, pgc, True, hid)
    ok = True
    for step in range(3):
        gs = torch.Generator(device=dev).manual_seed(100 + step)
        set_grads(pa, gs)
        for p, q in zip(pa, pb):
            q.grad = p.grad.clone()
        oa.step(); ob.step()                                                       # noqa: E702
        cmp = {"param": [], "exp_avg": [], "row_gain": [], "col_gain": []}
        eq = {k: [] for k in cmp}
        for p, q in zip(pa, pb):
            sa, sb = oa.state[p], ob.state[q]
            for k, x, y in (("param", p, q), ("exp_avg", sa["exp_avg"], sb["exp_avg"]),
                            ("row_gain", sa["row_gain"], sb["row_gain"]), ("col_gain", sa["col_gain"], sb["col_gain"])):
                cmp[k].append(rel(y, x))
                eq[k].append((x == y).float().mean().item())
        r = {k: {"max rel diff": max(v), "bitwise-equal fraction": round(statistics.mean(eq[k]), 4)} for k, v in cmp.items()}
        res[f"step {step + 1}"] = r
        print(f"step {step + 1}", json.dumps(r), flush=True)
        ok = ok and all(v["max rel diff"] < 2e-2 for k, v in r.items() if k == "param") \
            and all(v["max rel diff"] < 1e-5 for k, v in r.items() if k == "exp_avg")
    # how many matrices were batched
    batches, rest = ob._batch_plan(ob.param_groups[0], [p for p in pb if p.grad is not None])
    res["plan"] = {"batches": len(batches), "batched matrices": sum(len(b) for b in batches), "per-matrix": len(rest)}
    print("plan", res["plan"], flush=True)

    # timing and kernel count
    t = {}
    for name, opt, ps in (("per-matrix", oa, pa), ("batched", ob, pb)):
        for _ in range(3):
            opt.step()
        torch.cuda.synchronize()
        ts = []
        for _ in range(10):
            t0 = time.perf_counter(); opt.step(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)  # noqa
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            opt.step(); torch.cuda.synchronize()                                   # noqa: E702
        nk = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
        gpu_ms = sum(e.device_time for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA) / 1e3
        t[name] = {"step_ms (median of 10)": round(statistics.median(ts), 1), "min_ms": round(min(ts), 1),
                   "kernels per step": nk, "GPU kernel time ms": round(gpu_ms, 1),
                   "peak memory GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        print(name, json.dumps(t[name]), flush=True)
    t["speedup"] = round(t["per-matrix"]["step_ms (median of 10)"] / t["batched"]["step_ms (median of 10)"], 2)
    res["time"] = t
    print("speedup", t["speedup"])
    print("PASS" if ok else "FAIL")
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "test_md_batched_step.json", "w"), indent=1)
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
