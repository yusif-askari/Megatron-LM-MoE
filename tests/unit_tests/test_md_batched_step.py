# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MDDecoupling batched step (MD_BATCHED_STEP=1) against the per-matrix step.

The parameter set is a small copy of what one rank of a latent-MoE model holds: several local experts
per layer (fc1 with the GLU split, fc2), shared experts, latent projections each way, and a router
(which always takes the per-matrix path). Optimizer settings as in the md_decoupling recipe: Muon with
Nesterov momentum, polar-express coefficients, row hypersphere with fan-in radius, row/column gains.
"""

import pytest
import torch

from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.md_decoupling import MDDecoupling
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_EMERGING_OPTIMIZERS,
    reason="CUDA and emerging_optimizers are required for MDDecoupling orthogonal updates",
)

HIDDEN, LATENT, EXPERT = 32, 16, 24
EXPERTS, LAYERS = 6, 2


@pytest.fixture(scope="module", autouse=True)
def process_groups():
    """The per-matrix step of tensor-parallel matrices goes through their process group (size 1 here)."""
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _make_params():
    gen = torch.Generator(device="cuda").manual_seed(0)
    col = dict(partition_dim=0, tensor_model_parallel=True)
    row = dict(partition_dim=1, tensor_model_parallel=True, is_out_proj=True)
    spec = [  # (shape, attributes, count)
        ((2 * EXPERT, LATENT), dict(col, glu_split_dim=0, expert_tp=True), EXPERTS * LAYERS),
        ((LATENT, EXPERT), dict(row, expert_tp=True), EXPERTS * LAYERS),
        ((2 * EXPERT, HIDDEN), dict(col, glu_split_dim=0), LAYERS),
        ((HIDDEN, EXPERT), dict(row), LAYERS),
        ((LATENT, HIDDEN), dict(partition_dim=0, tensor_model_parallel=False), LAYERS),
        ((HIDDEN, LATENT), dict(partition_dim=0, tensor_model_parallel=False), LAYERS),
        ((8, HIDDEN), dict(is_router=True), LAYERS),  # routers: per-matrix path
        ((4 * HIDDEN, HIDDEN), dict(col, glu_split_dim=0), 1),  # the only matrix of its shape
    ]
    params = []
    for shape, attrs, count in spec:
        for _ in range(count):
            p = torch.nn.Parameter(
                torch.randn(*shape, device="cuda", generator=gen) / shape[1] ** 0.5
            )
            for name, value in attrs.items():
                setattr(p, name, value)
            params.append(p)
    return params


def _clone_params(params):
    out = []
    for p in params:
        q = torch.nn.Parameter(p.detach().clone())
        for name, value in p.__dict__.items():
            setattr(q, name, value)
        out.append(q)
    return out


def _make_optimizer(params, batched, **overrides):
    kwargs = dict(
        hypersphere_mode="row",
        hypersphere_embedding_mode="row",
        hypersphere_router_mode="row",
        hypersphere_radius_mode="fan_in",
        hidden_size=HIDDEN,
        use_orthogonal_updates=True,
        momentum_beta=0.95,
        use_nesterov=True,
        split_fc1=True,
        fp32_matmul_prec="medium",
        coefficient_type="polar_express",
        num_ns_steps=5,
        scale_mode="shape_up",
        router_scale_mode="none",
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        tp_mode="duplicated",
        hypersphere_gains_mode="rowcol",
        hypersphere_gains_mode_router="rowcol",
        hypersphere_gains_mode_embedding="none",
        gains_lr=1e-3,
        gain_parametrization="softplus",
    )
    kwargs.update(overrides)
    opt = MDDecoupling([{"params": params, "lr": 1e-2, "max_lr": 1e-2, "min_lr": 1e-3}], **kwargs)
    opt.batched_step = batched
    return opt


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _step_both(per_matrix, batched, params_a, params_b, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for p, q in zip(params_a, params_b):
        p.grad = torch.randn(p.shape, device="cuda", generator=gen) * 1e-3
        q.grad = p.grad.clone()
    per_matrix.step()
    batched.step()


def test_batch_plan_groups_same_shaped_matrices():
    params = _make_params()
    opt = _make_optimizer(params, batched=True)
    for p in params:
        p.grad = torch.zeros_like(p)
        opt._maybe_init_gain_state(p)
    batches, rest = opt._batch_plan(opt.param_groups[0], params)

    # expert fc1, expert fc2, shared fc1, shared fc2, latent projections down and up
    assert sorted(len(b) for b in batches) == sorted(
        [EXPERTS * LAYERS, EXPERTS * LAYERS, LAYERS, LAYERS, LAYERS, LAYERS]
    )
    for batch in batches:
        assert len({(tuple(p.shape), getattr(p, "glu_split_dim", None)) for p in batch}) == 1
    # the routers and the matrix that is alone in its shape keep the per-matrix path
    assert len(rest) == LAYERS + 1
    assert len(rest) + sum(len(b) for b in batches) == len(params)


def test_batch_max_splits_a_stack():
    params = _make_params()
    opt = _make_optimizer(params, batched=True)
    opt.batch_max = 5
    for p in params:
        p.grad = torch.zeros_like(p)
        opt._maybe_init_gain_state(p)
    batches, _ = opt._batch_plan(opt.param_groups[0], params)
    assert max(len(b) for b in batches) == 5
    assert sum(len(b) for b in batches) == 4 * EXPERTS + 4 * LAYERS  # nothing dropped


def test_batched_step_matches_per_matrix_step():
    base = _make_params()
    params_a, params_b = _clone_params(base), _clone_params(base)
    per_matrix = _make_optimizer(params_a, batched=False)
    batched = _make_optimizer(params_b, batched=True)

    for step in range(3):
        _step_both(per_matrix, batched, params_a, params_b, seed=100 + step)
        for p, q in zip(params_a, params_b):
            sa, sb = per_matrix.state[p], batched.state[q]
            assert sa.keys() == sb.keys()
            # the momentum is elementwise arithmetic: identical
            torch.testing.assert_close(sb["exp_avg"], sa["exp_avg"], rtol=1e-6, atol=0.0)
            # parameters and gains go through the Newton-Schulz GEMMs, which are batched GEMMs on the
            # batched path: equal up to GEMM rounding
            assert _rel(q, p) < 2e-3, (step, tuple(p.shape))
            for name in ("row_gain", "col_gain"):
                if name in sa:
                    assert _rel(sb[name], sa[name]) < 2e-3, (step, name, tuple(p.shape))
    assert all(torch.isfinite(q).all() for q in params_b)


def test_unsupported_configuration_keeps_the_per_matrix_path():
    """Without row/column gains nothing is batched, and the step equals the per-matrix step."""
    base = _make_params()
    params_a, params_b = _clone_params(base), _clone_params(base)
    overrides = dict(hypersphere_gains_mode="none", hypersphere_gains_mode_router="none")
    per_matrix = _make_optimizer(params_a, batched=False, **overrides)
    batched = _make_optimizer(params_b, batched=True, **overrides)

    for p in params_b:
        p.grad = torch.zeros_like(p)
        batched._maybe_init_gain_state(p)
    batches, rest = batched._batch_plan(batched.param_groups[0], params_b)
    assert not batches and len(rest) == len(params_b)

    _step_both(per_matrix, batched, params_a, params_b, seed=7)
    for p, q in zip(params_a, params_b):
        torch.testing.assert_close(q, p, rtol=0.0, atol=0.0)


def test_matrices_with_a_gain_pull_keep_the_per_matrix_path():
    """The batched step has no gain pull, so matrices tagged for it are not batched."""
    params = _make_params()
    experts_fc2 = [p for p in params if tuple(p.shape) == (LATENT, EXPERT)]
    for p in experts_fc2[:3]:
        p.md_rowcol_gain_pull_coeff = 0.1
    opt = _make_optimizer(params, batched=True)
    for p in params:
        p.grad = torch.zeros_like(p)
        opt._maybe_init_gain_state(p)
    batches, rest = opt._batch_plan(opt.param_groups[0], params)
    batched_ids = {id(p) for batch in batches for p in batch}
    assert all(id(p) not in batched_ids for p in experts_fc2[:3])
    assert all(id(p) in batched_ids for p in experts_fc2[3:])
    assert sum(1 for p in rest if id(p) in {id(q) for q in experts_fc2[:3]}) == 3
