# TimeSSM

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small (about 3M parameters) linear time-invariant state space forecaster
with a **rate knob**, trained and evaluated with the
[TimeJEPA](../TimeJEPA) stack. Branch `timessm`; the March 2025 Mamba code
is kept under `src/models/` and described in
[docs/README_mamba_2025.md](docs/README_mamba_2025.md).

## The idea

TimeJEPA's most solid result is RateIN: choosing, per dataset, a decimation
factor k of the context by a causal backtest is worth up to 2.4 CRPS points
of oracle gain. Decimation loses information (k - 1 points out of k) and the
fan has to be re-interpolated to the native grid.

A continuous-time LTI state space model owns that invariance exactly.
Sampling every Delta with a zero-order hold gives A_bar = exp(Delta A):
running the series decimated by k at Delta is the same computation as
running the full series at Delta / k. So the knob is the model's sampling
interval, not the data's: RateIN's selector chooses k as before, and the
model is run at w = 1 / k on the full context with the fan at the native
horizon. No decimation, no re-interpolation.

This is why the layer is S4D (diagonal, time-invariant) and not Mamba: an
input-dependent Delta breaks the exact transfer. Selectivity is offered as
an ablation in the readout only (`selective_readout`).

## Layout

```
src/timessm/ssm.py        S4DLayer: ZOH discretization, FFT convolution, step recurrence, delta_scale
src/timessm/block.py      GatedSSMBlock (norm, gate, S4D, optional readout gate; no conv by default)
src/timessm/model.py      SSMForecaster: RobustScale + RevIN + per-step tokens + blocks + quantile head,
                          autonomous rollout through a learned future token, forecast(x, n, w)
src/timessm/training.py   SSMFinetuneModule: TimeJEPA's FinetuneModule + multi-rate training on Delta
scripts/train_ssm.py      Hydra entry point (TimeJEPA datamodule, checkpointing, W&B)
scripts/eval_ssm.sh       GIFT-Eval through TimeJEPA's harness (+ratein=delta is the knob)
configs/ssm_mini_v3.yaml  the spike's recipe (champion's recipe, copied), ssm_mini_v3_eval.yaml
tests/                    layer (rate equivariance first), model contract, harness
docs/EXPERIMENTAL_LOG.md  registry: predictions before runs, results after (French)
docs/RUNBOOK.md           setup, train, eval commands
```

## Setup

TimeJEPA must be checked out next to this repo (`../TimeJEPA`): it is a
dependency, installed editable through `[tool.uv.sources]`.

```bash
uv sync                    # the SSM stack (torch >= 2.8, pytorch-lightning, timejepa)
uv sync --extra legacy     # also the 2025 Mamba tests (lightning, mlflow)
uv run pytest -q           # 29 tests, about 5 minutes on CPU
```

## Train and evaluate

```bash
python scripts/train_ssm.py --config-name ssm_mini_v3 wandb.run_name=ssm-mini-v3
scripts/eval_ssm.sh checkpoints/timessm_mini_v3_zs/pretrain_False/<ckpt> \
    +tta_flip=true +ratein=mix +ratein_pool=true      # TimeJEPA's official stack (decimation)
scripts/eval_ssm.sh <ckpt> +ratein=delta +ratein_pool=true   # the knob
scripts/eval_ssm.sh <ckpt> +ratein=backtest +ratein_pool=true # decimation, same selector
```

## Rate equivariance, tested

`tests/test_ssm_layer.py` pins the property before anything else: on a
block-constant input (blocks of k steps) the layer at Delta on the fine
series and the layer at k Delta on the decimated series agree at the block
ends to 1e-8 (float64), state and output, through the gated block too. A
depthwise convolution breaks it (a filter in steps, not in physical time),
which is why the block ships without one.
