# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused output layer + cross entropy (MEGATRON_FUSED_LINEAR_CE=1) against the unfused path."""

import pytest
import torch

from megatron.core.fusions import fused_linear_cross_entropy as flce

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused path is a Triton kernel"
)


@pytest.fixture(autouse=True)
def _restore_rng_state():
    """The tests seed the global generators; restore them for later tests in the same process."""
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        yield


@pytest.fixture
def small_chunks(monkeypatch):
    """Chunks of 256 rows and the mask * c check in the backward."""
    monkeypatch.setattr(flce, "CHUNK", 256)
    monkeypatch.setattr(flce, "_CHECK", True)


def _operands(dtype, n=700, v=5003, h=64, logit_std=4.0):
    """n is not a multiple of the chunk size and v not a multiple of the kernel's block."""
    torch.manual_seed(0)
    hidden = torch.randn(n, h, device="cuda").to(dtype)
    weight = (torch.randn(v, h, device="cuda") * logit_std / h**0.5).to(dtype)
    target = torch.randint(0, v, (n,), device="cuda")
    # first and last vocabulary entry, and the first entry of the kernel's second block
    target[:3] = torch.tensor([0, v - 1, min(4096, v - 1)], device="cuda")
    mask = (torch.rand(n, device="cuda") > 0.1).float()
    return hidden, weight, target, mask


def _reference(hidden, weight, target):
    logits = torch.matmul(hidden, weight.t())
    return torch.nn.functional.cross_entropy(logits.float(), target, reduction="none")


def _fused(hidden, weight, target, mask):
    return flce._FusedLinearCrossEntropy.apply(hidden, weight, target, mask)


def _loss_and_grads(fn, hidden, weight, mask, attrs=None):
    """Loss per token and the gradients the training loss sum(loss * mask) / sum(mask) gives."""
    hh = hidden.detach().clone().requires_grad_()
    ww = weight.detach().clone().requires_grad_()
    for name, value in (attrs or {}).items():
        setattr(ww, name, value)
    loss = fn(hh, ww)
    (loss * mask).sum().div(mask.sum()).backward()
    return loss.detach(), hh.grad, ww


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.bfloat16, 2e-2)])
def test_matches_unfused_path(small_chunks, dtype, tol):
    hidden, weight, target, mask = _operands(dtype)
    loss_ref, dh_ref, w_ref = _loss_and_grads(
        lambda a, b: _reference(a, b, target), hidden, weight, mask
    )
    loss, dh, w = _loss_and_grads(lambda a, b: _fused(a, b, target, mask), hidden, weight, mask)

    assert loss.dtype == torch.float32 and loss.shape == loss_ref.shape
    assert dh.dtype == dtype and w.grad.dtype == dtype
    # the losses come from the same stored logits: fp32 arithmetic in a different summation order
    torch.testing.assert_close(loss, loss_ref, rtol=1e-4, atol=1e-4)
    assert _rel(dh, dh_ref) < tol
    assert _rel(w.grad, w_ref.grad) < tol
    # masked tokens get no gradient
    assert dh[mask == 0].abs().max().item() == 0.0


@pytest.mark.parametrize("zero_out_wgrad", [False, True])
def test_weight_gradient_goes_into_main_grad(small_chunks, zero_out_wgrad):
    """With main_grad present (DistributedDataParallel) the gradient is added there in fp32."""
    hidden, weight, target, mask = _operands(torch.bfloat16)
    _, _, w_ref = _loss_and_grads(lambda a, b: _reference(a, b, target), hidden, weight, mask)
    start = torch.full(weight.shape, 0.5, dtype=torch.float32, device="cuda")
    attrs = dict(
        main_grad=start.clone(), grad_added_to_main_grad=False, zero_out_wgrad=zero_out_wgrad
    )
    _, _, w = _loss_and_grads(lambda a, b: _fused(a, b, target, mask), hidden, weight, mask, attrs)

    assert w.grad_added_to_main_grad is True
    assert _rel(w.main_grad - start, w_ref.grad) < 2e-2
    if zero_out_wgrad:
        # weights shared with the embedding: DistributedDataParallel adds .grad to main_grad as well
        assert w.grad.abs().max().item() == 0.0


def test_check_rejects_a_loss_that_is_not_mask_times_scalar(small_chunks):
    hidden, weight, target, mask = _operands(torch.float32, n=300)
    hh = hidden.clone().requires_grad_()
    ww = weight.clone().requires_grad_()
    loss = _fused(hh, ww, target, mask)
    token_weights = torch.linspace(0.5, 1.5, loss.numel(), device="cuda")
    with pytest.raises(AssertionError, match="mask \\* c"):
        (loss * mask * token_weights).sum().backward()


def test_shapes_of_the_model_interface(small_chunks):
    """fused_linear_cross_entropy takes hidden [s, b, H] and labels / loss_mask [b, s]."""
    s, b = 37, 3
    hidden, weight, target, mask = _operands(torch.float32, n=s * b, v=1000)
    loss_ref = _reference(hidden, weight, target).view(s, b).transpose(0, 1)
    loss = flce.fused_linear_cross_entropy(
        hidden.view(s, b, -1),
        weight,
        target.view(s, b).transpose(0, 1),
        mask.view(s, b).transpose(0, 1),
    )
    assert loss.shape == (b, s)
    torch.testing.assert_close(loss, loss_ref, rtol=1e-4, atol=1e-4)


class TestGPTModelUsesTheFusedPath:
    """GPTModel returns the same loss and gradients with MEGATRON_FUSED_LINEAR_CE=1."""

    def setup_method(self, method):
        pytest.importorskip("transformer_engine")
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_config import TransformerConfig
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            # no dropout: the two compared forward passes must compute the same hidden states
            hidden_dropout=0.0,
            attention_dropout=0.0,
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl='native',
        )
        self.model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=300,
            max_sequence_length=16,
        ).cuda()
        self.model.train()

    def teardown_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.destroy_model_parallel()

    def _loss_and_grads(self):
        torch.manual_seed(1)
        b, s = 3, 16
        tokens = torch.randint(0, 300, (b, s), device="cuda")
        labels = torch.randint(0, 300, (b, s), device="cuda")
        position_ids = torch.arange(s, device="cuda").repeat(b, 1)
        loss_mask = (torch.rand(b, s, device="cuda") > 0.2).float()
        self.model.zero_grad()
        loss = self.model(tokens, position_ids, None, labels=labels, loss_mask=loss_mask)
        (loss * loss_mask).sum().div(loss_mask.sum()).backward()
        grads = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None}
        return loss.detach(), grads

    def test_same_loss_and_gradients(self, monkeypatch):
        calls = []
        original = flce.fused_linear_cross_entropy
        monkeypatch.setattr(
            flce, "fused_linear_cross_entropy", lambda *a, **k: calls.append(1) or original(*a, **k)
        )
        monkeypatch.setattr(flce, "_CHECK", True)

        monkeypatch.delenv("MEGATRON_FUSED_LINEAR_CE", raising=False)
        loss_ref, grads_ref = self._loss_and_grads()
        assert not calls

        monkeypatch.setenv("MEGATRON_FUSED_LINEAR_CE", "1")
        loss, grads = self._loss_and_grads()
        assert len(calls) == 1

        torch.testing.assert_close(loss, loss_ref, rtol=1e-4, atol=1e-4)
        assert grads.keys() == grads_ref.keys()
        for name, grad_ref in grads_ref.items():
            assert _rel(grads[name], grad_ref) < 1e-3, name

    def test_not_used_in_eval_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(flce, "fused_linear_cross_entropy", lambda *a, **k: calls.append(1))
        monkeypatch.setenv("MEGATRON_FUSED_LINEAR_CE", "1")
        self.model.eval()
        with torch.no_grad():
            self._eval_forward()
        assert not calls

    def _eval_forward(self):
        b, s = 2, 16
        tokens = torch.randint(0, 300, (b, s), device="cuda")
        position_ids = torch.arange(s, device="cuda").repeat(b, 1)
        labels = torch.randint(0, 300, (b, s), device="cuda")
        return self.model(
            tokens, position_ids, None, labels=labels, loss_mask=torch.ones(b, s, device="cuda")
        )
