"""
S4D layer with a rate knob (2026-09-09). Written BEFORE the layer.

Pinned, in the order the spike needs them:
1. Rate equivariance (the property everything rests on). For a
   block-constant input with blocks of k steps, the state and the output of
   the layer run at Delta on the fine series coincide, at the block ends, with
   the layer run at k * Delta on the decimated series - EXACTLY (zero-order
   hold is exact on piecewise-constant inputs). On a smooth input that is not
   block-constant, the discrepancy grows with k. Same statement for the gated
   block without depthwise convolution (pointwise ops commute with
   decimation), which is why the block's conv is off by default.
2. FFT convolution == step recurrence at L = 1024 (float64, 1e-8; float32,
   1e-3), real and complex state.
3. Stability by construction: |A_bar| < 1 for every rate in [1/64, 64] and a
   finite gradient at L = 1024 (the failure of the March 2025 scan).
4. A per-item delta_scale tensor [B] equals the per-item scalar loop.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timessm.ssm import S4DLayer  # noqa: E402
from timessm.block import GatedSSMBlock  # noqa: E402


def _layer(d_model=8, d_state=8, real=False, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    layer = S4DLayer(d_model, d_state=d_state, real=real, dt_min=1e-2, dt_max=1e-1)
    return layer.to(dtype)


def _block_input(B, L, H, k, dtype, seed=1):
    """Piecewise-constant input: blocks of k identical steps."""
    torch.manual_seed(seed)
    coarse = torch.randn(B, L // k, H, dtype=dtype)
    fine = coarse.repeat_interleave(k, dim=1)
    return fine, coarse


# ------------------------------------------------------------- 1. rate
@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("k", [2, 4, 8])
def test_rate_equivariance_block_constant_input(real, k):
    layer = _layer(real=real)
    B, L, H = 3, 256, 8
    fine, coarse = _block_input(B, L, H, k, torch.float64)
    y_fine = layer(fine, delta_scale=1.0)
    y_coarse = layer(coarse, delta_scale=float(k))
    # output at the end of every block == coarse output at that block
    assert torch.allclose(y_fine[:, k - 1::k], y_coarse, atol=1e-8, rtol=1e-8)
    # the same through the recurrence: state after k fine steps == one coarse step
    st_f = layer.init_state(B, dtype=torch.float64)
    st_c = layer.init_state(B, dtype=torch.float64)
    for t in range(k):
        _, st_f = layer.step(fine[:, t], st_f, delta_scale=1.0)
    _, st_c = layer.step(coarse[:, 0], st_c, delta_scale=float(k))
    assert torch.allclose(st_f, st_c, atol=1e-8, rtol=1e-8)


def test_rate_discrepancy_grows_with_k_on_a_smooth_input():
    layer = _layer()
    B, L, H = 2, 512, 8
    t = torch.arange(L, dtype=torch.float64)
    x = torch.sin(2 * torch.pi * t / 64)[None, :, None].repeat(B, 1, H)
    y_fine = layer(x, delta_scale=1.0)
    errs = []
    for k in (2, 4, 8):
        y_coarse = layer(x[:, k - 1::k], delta_scale=float(k))
        errs.append((y_fine[:, k - 1::k] - y_coarse).abs().mean().item())
    assert errs[0] < errs[1] < errs[2]
    assert errs[0] > 1e-6                       # not block-constant: not exact


@pytest.mark.parametrize("k", [2, 4])
def test_gated_block_without_conv_is_rate_equivariant(k):
    torch.manual_seed(0)
    block = GatedSSMBlock(d_model=8, d_state=8, expand=2, d_conv=0, dropout=0.0).double()
    B, L, H = 2, 128, 8
    fine, coarse = _block_input(B, L, H, k, torch.float64)
    y_fine = block(fine, delta_scale=1.0)
    y_coarse = block(coarse, delta_scale=float(k))
    assert torch.allclose(y_fine[:, k - 1::k], y_coarse, atol=1e-8, rtol=1e-8)
    conv = GatedSSMBlock(d_model=8, d_state=8, expand=2, d_conv=4, dropout=0.0).double()
    yc_fine = conv(fine, delta_scale=1.0)
    yc_coarse = conv(coarse, delta_scale=float(k))
    # a fixed-lag depthwise conv is a filter in STEPS, not physical time
    assert not torch.allclose(yc_fine[:, k - 1::k], yc_coarse, atol=1e-6)


# ------------------------------------------------------------- 2. fft == step
@pytest.mark.parametrize("real", [False, True])
def test_fft_matches_recurrence_1024(real):
    layer = _layer(real=real)
    B, L, H = 2, 1024, 8
    torch.manual_seed(2)
    u = torch.randn(B, L, H, dtype=torch.float64)
    y = layer(u, delta_scale=1.0)
    st = layer.init_state(B, dtype=torch.float64)
    ys = []
    for t in range(L):
        yt, st = layer.step(u[:, t], st, delta_scale=1.0)
        ys.append(yt)
    ys = torch.stack(ys, dim=1)
    assert torch.allclose(y, ys, atol=1e-8, rtol=1e-8)
    lf = _layer(real=real, dtype=torch.float32)
    y32 = lf(u.float(), delta_scale=1.0)
    assert torch.allclose(y32.double(), y, atol=1e-3, rtol=1e-3)


# ------------------------------------------------------------- 3. stability
def test_stable_by_construction_and_finite_gradient_1024():
    layer = _layer(d_model=16, d_state=32, dtype=torch.float32)
    with torch.no_grad():
        layer.log_a_re.uniform_(-6, 4)          # wild real parts
    for s in (1 / 64, 1.0, 64.0):
        a_bar = layer.discretize(torch.tensor([s]))[0]
        assert (a_bar.abs() < 1.0).all()
    torch.manual_seed(3)
    u = torch.randn(4, 1024, 16, requires_grad=True)
    y = layer(u, delta_scale=1.0)
    assert torch.isfinite(y).all()
    y.pow(2).mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert torch.isfinite(u.grad).all()


# ------------------------------------------------------------- 4. per-item
def test_per_item_delta_scale_equals_loop():
    layer = _layer(dtype=torch.float32)
    torch.manual_seed(4)
    u = torch.randn(5, 64, 8)
    scales = torch.tensor([0.25, 1.0, 1.0, 4.0, 1 / 48])
    y = layer(u, delta_scale=scales)
    for b in range(5):
        yb = layer(u[b:b + 1], delta_scale=float(scales[b]))
        assert torch.allclose(y[b:b + 1], yb, atol=1e-6, rtol=1e-5)
    assert torch.allclose(layer(u), layer(u, delta_scale=1.0))
