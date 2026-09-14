#!/usr/bin/env bash
# Everything that must be green BEFORE a paid multi-GPU run (2026-09-14).
# ~10-15 minutes on any pod. Stops at the first failure.
#
#   scripts/preflight.sh                       # config ssm_mid_v3, 8 GPUs if present
#   CONFIG=ssm_mini_v3 scripts/preflight.sh    # another config
#   DEVICES=2 scripts/preflight.sh             # fewer GPUs for the smoke
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-ssm_mid_v3}
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
DEVICES=${DEVICES:-$NGPU}
echo "== preflight: config $CONFIG, $NGPU GPU visible, smoke on $DEVICES"

echo "== 1. tests (TimeMamba + TimeJEPA corpus/harness)"
python -m pytest -q tests/test_ssm_layer.py tests/test_ssm_model.py -x
(cd ../TimeJEPA && python -m pytest -q tests/test_ratein_delta.py tests/test_schedule_fraction.py -x)

echo "== 2. corpus audit (106 files, 15.85 B observations)"
(cd ../TimeJEPA && bash scripts/build_corpus_v3.sh --check)

echo "== 3. one training step at the target size (memory and speed, 1 GPU)"
CUDA_VISIBLE_DEVICES=0 python scripts/profile_step.py --config "$CONFIG" --batch "${BATCH:-$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load(\"configs/$CONFIG.yaml\").data.batch_size)")}" --no-profile

echo "== 4. DDP smoke: 30 optimizer steps on $DEVICES GPU, real corpus, checkpoint written and reloadable"
S=/tmp/preflight_$$
WANDB_MODE=offline python scripts/train_ssm.py --config-name "$CONFIG" \
  trainer.devices="$DEVICES" trainer.limit_train_batches=90 trainer.limit_val_batches=4 \
  trainer.val_check_interval=45 training.schedule_fraction=1.0 data.checkpoint_dir="$S/ckpt" \
  data.output_dir="$S/lightning" wandb.run_name=preflight 2>&1 | tail -5
CK=$(ls -t "$S"/ckpt/*/pretrain_False/epoch00_valloss*.ckpt | head -1)
echo "   checkpoint: $CK"
CUDA_VISIBLE_DEVICES=0 python - "$CK" "$CONFIG" <<'PY'
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "../TimeJEPA/src")
import torch
import os
from hydra import initialize_config_dir, compose
from timejepa.evaluation.loading import load_checkpoint
from timessm.model import build_from_config
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs")):
    cfg = compose(config_name=sys.argv[2])
m = load_checkpoint(build_from_config(cfg), sys.argv[1], torch.device("cuda"))
out = m.forecast(50 + torch.randn(2, 1024, 1, device="cuda"), n=256)
assert torch.isfinite(out["quantiles_denorm"]).all()
print("   reload + forecast OK:", tuple(out["quantiles_denorm"].shape))
PY
rm -rf "$S"
echo "== preflight OK - launch the real run"
