#!/usr/bin/env bash
# Train the pouring policy on the in-house GPU and bring checkpoints back.
#   GPU_HOST=user@gpubox GPU_DIR=~/lerobot_pour ./train_remote.sh probe
#   GPU_HOST=... GPU_DIR=... ./train_remote.sh validate     # 20k steps
#   GPU_HOST=... GPU_DIR=... ./train_remote.sh full         # 100k steps
#   GPU_HOST=... GPU_DIR=... ./train_remote.sh fetch        # checkpoints only
#   RESUME=1 GPU_HOST=... GPU_DIR=... ./train_remote.sh full   # resume from last checkpoint
set -euo pipefail
: "${GPU_HOST:?set GPU_HOST, e.g. user@gpubox}"
: "${GPU_DIR:?set GPU_DIR, e.g. ~/lerobot_pour}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="${1:-probe}"
JOB=pour_v1
DATASET=so101_pour_bottle_v1
REMOTE_PY="${REMOTE_PY:-~/lerobot-venv/bin/python}"

trap 'echo "== stage \"$STAGE\" failed" >&2' ERR

case "$STAGE" in
  probe)    STEPS=200;    SAVE=200 ;;
  validate) STEPS=20000;  SAVE=10000 ;;
  full)     STEPS=100000; SAVE=20000 ;;
  fetch)    STEPS=0;      SAVE=0 ;;
  *) echo "Usage: $0 {probe|validate|full|fetch}" >&2; exit 1 ;;
esac

if [ "$STAGE" != "fetch" ]; then
  echo "== syncing dataset to $GPU_HOST"
  ssh "$GPU_HOST" "mkdir -p \"$GPU_DIR/datasets\""
  rsync -a -s --info=progress2 "$HERE/datasets/$DATASET/" "$GPU_HOST:$GPU_DIR/datasets/$DATASET/"

  echo "== training ($STAGE, $STEPS steps)"
  START=$(date +%s)
  ssh "$GPU_HOST" "cd \"$GPU_DIR\" && \"$REMOTE_PY\" -m lerobot.scripts.lerobot_train \
    --policy.type=act \
    --policy.device=cuda \
    --dataset.repo_id=local/$DATASET \
    --dataset.root=\"$GPU_DIR/datasets/$DATASET\" \
    --output_dir=\"$GPU_DIR/outputs/train/$JOB\" \
    --job_name=$JOB \
    --steps=$STEPS \
    --save_freq=$SAVE \
    --batch_size=8 \
    ${RESUME:+--resume=true}"
  echo "== $STAGE finished in $(( $(date +%s) - START ))s"
fi

echo "== fetching checkpoints"
mkdir -p "$HERE/outputs/train/$JOB"
rsync -a -s --info=progress2 "$GPU_HOST:$GPU_DIR/outputs/train/$JOB/" "$HERE/outputs/train/$JOB/"
echo "local checkpoints:"
ls "$HERE/outputs/train/$JOB/checkpoints"
