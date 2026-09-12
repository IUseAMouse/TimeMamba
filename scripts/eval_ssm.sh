#!/usr/bin/env bash
# Evaluate a TimeSSM checkpoint on GIFT-Eval with TimeJEPA's harness, from
# the TimeMamba repo (results in ./evaluation/<model.name>/<ckpt>/gift<tag>/).
#
#   scripts/eval_ssm.sh <checkpoint> [hydra flags...]
#   scripts/eval_ssm.sh checkpoints/timessm_mini_v3_zs/pretrain_False/epoch00_valloss0.4321.ckpt \
#       +tta_flip=true +ratein=mix +ratein_pool=true        # the official stack
#   scripts/eval_ssm.sh <ckpt> +ratein=delta                 # the Delta knob (P-SSM.2)
#   scripts/eval_ssm.sh <ckpt> +ratein=backtest              # decimation, same selector
#
# TIMEJEPA=<path> overrides the sibling checkout. The GIFT data are read from
# TimeJEPA (+gift_data_dir), the config from this repo (absolute config path).
set -eu
if [ $# -lt 1 ]; then
  echo "usage: $0 <checkpoint> [hydra flags...]" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")/.." && pwd)
TIMEJEPA=${TIMEJEPA:-"$HERE/../TimeJEPA"}
CK="$1"; shift
PYTHONUNBUFFERED=1 python "$TIMEJEPA/scripts/evaluate_gift.py" \
  --config-path "$HERE/configs" --config-name ssm_mini_v3_eval \
  "+checkpoint_path=$CK" "+gift_data_dir=$TIMEJEPA/data/gift_eval" "$@"
