#!/usr/bin/env bash
# Train the pouring policy on the in-house GPU and bring checkpoints back.
# GPU_DIR is expanded by YOUR local shell before it ever reaches the remote
# host (bash expands a leading "~" in a plain assignment), so GPU_DIR must be
# an absolute remote path, e.g. /home/user/lerobot_pour, not ~/lerobot_pour.
#   GPU_HOST=user@gpubox GPU_DIR=/home/user/lerobot_pour ./train_remote.sh probe
#   GPU_HOST=... GPU_DIR=... ./train_remote.sh validate     # 20k steps
#   GPU_HOST=... GPU_DIR=... ./train_remote.sh full         # 100k steps
#   GPU_HOST=... GPU_DIR=... ./train_remote.sh fetch        # checkpoints only
#   RESUME=1 GPU_HOST=... GPU_DIR=... ./train_remote.sh full   # resume from last checkpoint
set -euo pipefail
: "${GPU_HOST:?set GPU_HOST, e.g. user@gpubox}"
: "${GPU_DIR:?set GPU_DIR, e.g. /home/user/lerobot_pour (absolute remote path)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="${1:-probe}"
JOB=pour_v1
DATASET=so101_pour_bottle_v1
# The remote command below is built as one string by THIS (local) shell, then
# sent to ssh for the REMOTE shell to parse and expand. Anything left
# unescaped here (like $GPU_DIR, $STEPS) is expanded locally before sending;
# anything escaped (like \"$REMOTE_PY\") is sent literally and expanded
# remotely instead. A literal "~" never expands inside quotes in either
# shell, so the default below uses remote $HOME (quoted, so it still expands
# remotely) rather than "~", and still works when overridden with an
# absolute path or a path containing spaces.
REMOTE_PY=${REMOTE_PY:-'$HOME/lerobot-venv/bin/python'}

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
