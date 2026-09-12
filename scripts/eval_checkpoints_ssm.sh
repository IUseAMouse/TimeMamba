#!/usr/bin/env bash
# Evaluate every checkpoint of a TimeSSM run on GIFT-Eval, one after the other,
# with the official stack (flip + RateIN mix + pool) unless STACK says
# otherwise, and keep a digest. Mirror of TimeJEPA/scripts/eval_checkpoints.sh
# on top of scripts/eval_ssm.sh.
#
#   scripts/eval_checkpoints_ssm.sh <checkpoint_dir> [extra hydra flags...]
#   scripts/eval_checkpoints_ssm.sh checkpoints/timessm_mini_v3_zs/pretrain_False
#   STACK="+ratein=delta +ratein_pool=true" scripts/eval_checkpoints_ssm.sh <dir>   # the knob
#   STACK="+ratein=backtest +ratein_pool=true" scripts/eval_checkpoints_ssm.sh <dir>
#   scripts/eval_checkpoints_ssm.sh <dir> +gift_batch_size=8                       # while a run holds the GPUs
#
# Checkpoints are taken in creation order (oldest first), last.ckpt excluded.
# evaluate_gift caches per checkpoint stem and per flag set, so re-running
# only evaluates what is new. Digest: logs/eval_<run>[_<tag>].log, full logs
# in logs/eval_<run>_<stem>[_<tag>].log. Selection doctrine: the GIFT eval
# picks the champion, never val_loss.
set -u
if [ $# -lt 1 ]; then
  echo "usage: $0 <checkpoint_dir> [extra hydra flags...]" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")/.." && pwd)
DIR="$1"; shift
EXTRA=("$@")
STACK_DEFAULT="+tta_flip=true +ratein=mix +ratein_pool=true"
STACK_STR=${STACK:-$STACK_DEFAULT}
read -r -a STACK_ARR <<< "$STACK_STR"
TAG=""
[ "$STACK_STR" != "$STACK_DEFAULT" ] && TAG="_$(echo "$STACK_STR" | tr -d '+' | tr ' =' '_-')"
RUN=$(basename "$(dirname "$DIR")")
[ "$RUN" = "checkpoints" ] && RUN=$(basename "$DIR")
mkdir -p "$HERE/logs"
DIGEST="$HERE/logs/eval_${RUN}${TAG}.log"
echo "== $(date '+%F %T') run=$RUN flags: ${STACK_ARR[*]} ${EXTRA[*]:-}" | tee -a "$DIGEST"

mapfile -t CKPTS < <(ls -tr "$DIR"/*.ckpt 2>/dev/null | grep -v '/last[^/]*\.ckpt$')
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "no checkpoint in $DIR" | tee -a "$DIGEST"; exit 1
fi
declare -a ROWS=()
for CK in "${CKPTS[@]}"; do
  STEM=$(basename "$CK" .ckpt)
  LOG="$HERE/logs/eval_${RUN}_${STEM}${TAG}.log"
  echo "-- $STEM ($(date '+%T'))" | tee -a "$DIGEST"
  bash "$HERE/scripts/eval_ssm.sh" "$CK" "${STACK_ARR[@]}" "${EXTRA[@]}" > "$LOG" 2>&1
  RC=$?
  grep -E "vs_official_seasonal_naive|vs_local_seasonal_naive|coverage \(mean|RateIN:|Results:" "$LOG" \
    | sed -E 's/.*INFO\] - //' | tee -a "$DIGEST"
  [ $RC -ne 0 ] && echo "   exit $RC (see $LOG)" | tee -a "$DIGEST"
  CACHED=$(grep -c "already done" "$LOG"); FAILED=$(grep -c "FAILED:" "$LOG")
  COMPUTED=$(grep -cE "\[[0-9]+/[0-9]+\] [^ ]+: MASE" "$LOG")
  echo "   configs: $CACHED cached, $COMPUTED computed, $FAILED failed" | tee -a "$DIGEST"
  LINE=$(grep "vs_official_seasonal_naive" "$LOG" | tail -1 | sed -E 's/.*MASE ratio ([0-9.]+) \| CRPS ratio ([0-9.]+).*/\1 \2/')
  COV=$(grep "coverage (mean" "$LOG" | tail -1 | sed -E 's/.*-> ([0-9.]+).*/\1/')
  NCFG=$(grep "coverage (mean" "$LOG" | tail -1 | sed -E 's/.*mean over ([0-9]+) configs.*/\1/')
  ROWS+=("$STEM ${LINE:-nan nan} ${COV:-nan} ${NCFG:-0}")
done
{
  echo "== table run=$RUN flags: ${STACK_ARR[*]} ($(date '+%F %T'))"
  printf "%-36s %8s %8s %8s %6s\n" checkpoint MASE CRPS cov80 n_cfg
  for r in "${ROWS[@]}"; do
    set -- $r
    FLAG=""; [ "$5" != "97" ] && FLAG="*"
    printf "%-36s %8s %8s %8s %5s%s\n" "$1" "$2" "$3" "$4" "$5" "$FLAG"
  done
  echo "reference TimeJEPA head8 champion stack: 0.7842 0.5340 0.756 (5%: 0.5585, scratch S4-c 5%: 0.5506)"
  echo "* fewer than 97 configs: NOT comparable to the reference nor across rows (compare_subset.py)"
} | tee -a "$DIGEST"
