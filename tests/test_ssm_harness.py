"""
7. The TimeJEPA GIFT harness runs on a (small, untrained) SSMForecaster in
   every mode the spike will use: off, flip, mix + pool (decimation stack),
   backtest, and ratein=delta (the knob); the knob path passes w = 1/K and
   native-length contexts; delta is refused on a model without rate_knob;
   the checkpoint round trip through TimeJEPA's loader (builder + core
   prefixes) rebuilds the same numbers.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TIMEJEPA = ROOT.parent / "TimeJEPA"
sys.path.insert(0, str(TIMEJEPA / "src"))
sys.path.insert(0, str(TIMEJEPA / "scripts"))

from timessm.model import SSMForecaster, build_from_config  # noqa: E402

pytest.importorskip("evaluate_gift")


def _model():
    torch.manual_seed(0)
    return SSMForecaster(input_length=256, prediction_length=32, d_model=16, n_layers=2,
                         d_state=8, quantile_hidden_dim=32, dropout=0.0).eval()


def _synthetic(rng, n_series=6, length=700):
    out = []
    for _ in range(n_series):
        level, amp = rng.uniform(5, 50), rng.uniform(1, 5)
        t = np.arange(length)
        y = level + amp * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.2 * amp, size=length)
        out.append(y.astype(np.float32))
    return out


@pytest.fixture
def harness(monkeypatch):
    import evaluate_gift as EG
    series = _synthetic(np.random.default_rng(3))
    monkeypatch.setattr(EG.gift, "load_series", lambda root, cfg: series)
    monkeypatch.setattr(EG.gift, "prediction_length", lambda cfg: 40)
    monkeypatch.setattr(EG.gift, "num_windows", lambda cfg, n: 2)
    monkeypatch.setattr(EG.gift, "seasonality", lambda f: 24)
    return EG


def _run(EG, model, **kw):
    return EG.evaluate_config(model, "stub/H/short", Path("."), torch.device("cpu"),
                              batch_size=4, **kw)


def test_harness_modes_run_on_the_ssm(harness):
    EG = harness
    m = _model()
    off = _run(EG, m)
    assert np.isfinite(off["model"]["CRPS"]) and np.isfinite(off["model"]["MASE"])
    flip = _run(EG, m, tta_flip=True)
    assert np.isfinite(flip["model"]["CRPS"])
    mix = _run(EG, m, ratein_mode="mix", ratein_pool=True)
    assert "mix" in mix["ratein"] and mix["ratein"]["backtest"]["knob"] == "decimation"
    bt = _run(EG, m, ratein_mode="backtest", ratein_pool=True)
    assert bt["ratein"]["backtest"]["knob"] == "decimation"
    calls = []
    orig = m.forecast

    def spy(batch, n=None, w=None, **kw):
        calls.append((batch.shape[1], n, None if w is None else float(w.reshape(-1)[0])))
        return orig(batch, n=n, w=w, **kw)

    m.forecast = spy
    delta = _run(EG, m, ratein_mode="delta", ratein_pool=True)
    d = delta["ratein"]["backtest"]
    assert d["knob"] == "delta" and set(d["ratios"]) >= {"2", "4"}   # k > 1 only
    assert all(L == 256 for L, _, _ in calls)                 # never decimated
    assert all(n == 40 for _, n, _ in calls)                  # native horizon
    ws = {w for _, _, w in calls}
    assert None in ws and any(w is not None and w < 1 for w in ws)
    if d["K"] > 1:
        assert all(w == pytest.approx(1.0 / d["K"]) for _, _, w in calls[-3:])
    assert np.isfinite(delta["model"]["CRPS"])


def test_delta_refused_without_knob_and_jepa_layers_refused():
    from evaluate_gift import check_model_flags
    m = _model()
    check_model_flags(m, "delta", False, None, None)
    m.rate_knob = None
    with pytest.raises(ValueError):
        check_model_flags(m, "delta", False, None, None)
    with pytest.raises(ValueError):
        check_model_flags(_model(), "off", False, None, {"params": "norm"})


def test_checkpoint_round_trip_through_timejepa_loader(tmp_path):
    from omegaconf import OmegaConf
    from timejepa.evaluation import loading
    cfg = OmegaConf.load(ROOT / "configs" / "ssm_mini_v3_eval.yaml")
    base = OmegaConf.load(ROOT / "configs" / "ssm_mini_v3.yaml")
    cfg = OmegaConf.merge(base, cfg)
    cfg.model.seq_length, cfg.model.prediction_length = 256, 32
    cfg.model.ssm.d_model, cfg.model.ssm.n_layers, cfg.model.ssm.d_state = 16, 2, 8
    cfg.model.decoder.quantile_hidden_dim = 32
    m = build_from_config(cfg).eval()
    assert m.name == "timessm_mini_v3_zs"
    ck = tmp_path / "epoch00.ckpt"
    torch.save({"state_dict": {"model." + k: v for k, v in m.state_dict().items()}}, ck)
    m2 = loading.create_model_from_config(cfg)
    assert isinstance(m2, SSMForecaster)
    m2 = loading.load_checkpoint(m2, str(ck), torch.device("cpu"))
    x = 50 + torch.randn(2, 256, 1)
    with torch.no_grad():
        a = m.forecast(x, n=16)["quantiles_denorm"]
        b = m2.forecast(x, n=16)["quantiles_denorm"]
    assert torch.allclose(a, b)
    # a different width is a core mismatch: refused, not scored
    cfg.model.ssm.d_state = 4
    with pytest.raises(RuntimeError):
        loading.load_checkpoint(loading.create_model_from_config(cfg), str(ck),
                                torch.device("cpu"))
