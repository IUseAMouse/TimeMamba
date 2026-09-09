"""
SSMForecaster contract (2026-09-09).

5. forecast: the JEPATST keys, shapes [B, n, Q] for n in {8, 128, 256, 900}
   (free horizon, no rollout), quantiles increasing along Q, denormalized fan
   = exact inverse of the normalized one; FinetuneModule on the model:
   finite loss, backward reaches the blocks and the head, optimizer groups
   'encoder' / 'decoder'; size in [2M, 4M] at the spike's dimensions.
6. w changes the fan, w = 1 is the identity, per-item w == per-item loop;
   selective_readout composes; the multi-rate module draws w in train only
   and logs its witnesses.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timessm.model import SSMForecaster  # noqa: E402
from timessm.training import SSMFinetuneModule  # noqa: E402

KEYS = ("forecast", "forecast_denorm", "quantiles", "quantiles_denorm", "quantile_levels")


def _small(**kw):
    torch.manual_seed(0)
    args = dict(input_length=128, prediction_length=32, d_model=16, n_layers=2,
                d_state=8, expand=2, dropout=0.0, quantile_hidden_dim=32)
    args.update(kw)
    return SSMForecaster(**args).eval()


def _ctx(B=3, L=128, seed=1):
    torch.manual_seed(seed)
    t = torch.arange(L).float()
    base = torch.sin(2 * torch.pi * t / 24)[None, :, None]
    return 50.0 + 10.0 * base + torch.randn(B, L, 1) + 5.0 * torch.arange(B)[:, None, None]


# ------------------------------------------------------------- 5. contract
@pytest.mark.parametrize("n", [8, 128, 256, 900])
def test_forecast_contract_any_horizon(n):
    m = _small()
    x = _ctx()
    with torch.no_grad():
        out = m.forecast(x, n=n, return_representations=True)
    for k in KEYS:
        assert k in out
    Q = len(out["quantile_levels"])
    assert out["quantiles"].shape == (3, n, Q)
    assert out["quantiles_denorm"].shape == (3, n, Q)
    assert out["forecast"].shape == (3, n, 1)
    assert out["forecast_denorm"].shape == (3, n, 1)
    assert (out["quantiles"].diff(dim=-1) >= 0).all()
    assert out["context_embeddings"].shape == (3, 128, 16)
    assert out["future_representations"].shape == (3, n, 16)
    assert out["context_norm"].shape == x.shape
    # denormalization is the exact inverse of the normalization chain
    q = out["quantiles"]
    back = m.robust_scaler.inverse(m.revin.denormalize_target_space(q))
    assert torch.allclose(back, out["quantiles_denorm"], atol=1e-5, rtol=1e-5)
    assert torch.isfinite(out["quantiles_denorm"]).all()
    # the denormalized fan lives near the context's level
    assert (out["forecast_denorm"].mean() - x.mean()).abs() < 3 * x.std()


def test_finetune_module_trains_the_model():
    m = _small()
    mod = SSMFinetuneModule(
        m, finetune_mode="full_finetune", loss_type="huber", learning_rate=1e-3,
        encoder_lr_multiplier=1.0, lr_scheduler="constant",
        delta_scales=[0.5, 2.0], p_delta_scale=1.0,
    )
    mod.train()
    x = _ctx(B=4)
    y = _ctx(B=4, L=32, seed=2)
    loss, results, target = mod._forward_and_loss(x, y)
    assert torch.isfinite(loss)
    assert mod._last_delta_scale in (0.5, 2.0)          # drawn in train
    loss.backward()
    assert m.blocks[0].ssm.log_dt.grad is not None
    assert m.blocks[0].ssm.C_re.grad is not None
    assert m.decoder.decoder.mlp[0].weight.grad is not None
    assert m.future_token.grad is not None
    assert m.patching.projection.weight.grad is not None
    opt = mod.configure_optimizers()
    names = [g["name"] for g in opt.param_groups]
    assert names == ["encoder", "decoder"]
    n_dec = sum(p.numel() for g in opt.param_groups if g["name"] == "decoder" for p in g["params"])
    assert n_dec == sum(p.numel() for p in m.decoder.parameters())
    mod.eval()
    loss_eval, _, _ = mod._forward_and_loss(x, y)
    assert mod._last_delta_scale == 1.0                  # never drawn in eval


def test_spike_size_between_2m_and_4m():
    m = SSMForecaster(d_model=192, n_layers=6, d_state=32, expand=2,
                      quantile_hidden_dim=1536)
    n = m.count_parameters()
    assert 2_000_000 <= n <= 4_000_000, n
    parts = m.get_num_params()
    assert parts["decoder"] < 0.4 * parts["total"]


def test_linear_probe_and_pretrain_refusals():
    m = _small()
    with pytest.raises(ValueError):
        m.set_pretrain_mode(True)
    with pytest.raises(NotImplementedError):
        m.forward_pretrain(None, None)
    m.freeze_encoder(); m.freeze_patching()
    assert not any(p.requires_grad for p in m.blocks.parameters())
    assert all(p.requires_grad for p in m.decoder.parameters())
    m.unfreeze_encoder(); m.unfreeze_patching()
    assert all(p.requires_grad for p in m.blocks.parameters())
    assert m.predictor.w_film is None and m.rate_knob == "delta"


# ------------------------------------------------------------- 6. the knob
def test_w_changes_the_fan_and_w1_is_identity():
    m = _small()
    x = _ctx()
    with torch.no_grad():
        base = m.forecast(x, n=32)["quantiles"]
        one = m.forecast(x, n=32, w=1.0)["quantiles"]
        vec1 = m.forecast(x, n=32, w=torch.ones(3))["quantiles"]
        half = m.forecast(x, n=32, w=0.5)["quantiles"]
        mixed = m.forecast(x, n=32, w=torch.tensor([0.5, 1.0, 0.25]))["quantiles"]
        q_third = m.forecast(x[2:3], n=32, w=0.25)["quantiles"]
    assert torch.allclose(base, one) and torch.allclose(base, vec1)
    assert not torch.allclose(base, half, atol=1e-4)
    assert torch.allclose(mixed[0:1], half[0:1], atol=1e-5, rtol=1e-5)
    assert torch.allclose(mixed[1:2], base[1:2], atol=1e-5, rtol=1e-5)
    assert torch.allclose(mixed[2:3], q_third, atol=1e-5, rtol=1e-5)


def test_selective_readout_and_conv_compose():
    for kw in (dict(selective_readout=True), dict(d_conv=4), dict(real=True)):
        m = _small(**kw)
        with torch.no_grad():
            out = m.forecast(_ctx(), n=16)
        assert out["quantiles"].shape == (3, 16, 9)
        assert torch.isfinite(out["quantiles_denorm"]).all()
    assert _small(selective_readout=True).blocks[0].readout_gate is not None
    assert _small().blocks[0].readout_gate is None and _small().blocks[0].conv is None


def test_multirate_module_refuses_bad_config():
    m = _small()
    with pytest.raises(ValueError):
        SSMFinetuneModule(m, finetune_mode="full_finetune", p_delta_scale=0.5)
    with pytest.raises(ValueError):
        SSMFinetuneModule(m, finetune_mode="full_finetune", delta_scales=[0.0, 1.0],
                          p_delta_scale=0.5)
