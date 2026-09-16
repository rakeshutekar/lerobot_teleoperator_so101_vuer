#!/usr/bin/env bash
# Train the pouring policy on the in-house GPU and bring checkpoints back.
# GPU_DIR is expanded by YOUR local shell before it ever reaches the remote
# host (bash expands a leading "~" in a plain assignment), so GPU_DIR must be
# an absolute remote path, e.g. /home/user/lerobot_pour, not ~/lerobot_pour.
#   GPU_HOST=user@gpubox GPU_DIR=/home/user/lerobot_pour ./train_remote.sh probe
#   GPU_HOST=... GPU_DIR=... ./train_remote.sh validate     # 20k steps, starts pour_v1
#   GPU_HOST=... GPU_DIR=... ./train_remote.sh full         # 100k steps, continues pour_v1
#   GPU_HOST=... GPU_DIR=... ./train_remote.sh fetch        # checkpoints only
#   ./train_remote.sh local                                 # Apple GPU fallback, no GPU host
#   RESUME=0 GPU_HOST=... GPU_DIR=... ./train_remote.sh full   # force a fresh run
#   RESUME=1 GPU_HOST=... GPU_DIR=... ./train_remote.sh full   # force a resume
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="${1:-probe}"
JOB=pour_v1                 # the run that gets deployed
PROBE_JOB=pour_v1_probe     # throwaway: a 200-step checkpoint must never be deployable
LOCAL_JOB=pour_v1_local     # Apple GPU fallback, kept apart from the GPU host's run
DATASET=so101_pour_bottle_v1
# The remote command below is built as one string by THIS (local) shell, then
# sent to ssh for the REMOTE shell to parse and expand. Anything left
# unescaped here (like $GPU_DIR, $STEPS) is expanded locally before sending;
# anything escaped (like \"$REMOTE_PY\") is sent literally and expanded
# remotely instead. A literal "~" never expands inside quotes in either
# shell, so the default below uses remote $HOME (quoted, so it still expands
# remotely) rather than "~", and still works when overridden with an
# absolute path or a path containing spaces. A REMOTE_PY override must obey
# the same rule: write '$HOME/...' or an absolute path, never '~/...'.
REMOTE_PY=${REMOTE_PY:-'$HOME/lerobot-venv/bin/python'}
LOCAL_PY=${LOCAL_PY:-$HERE/../robot-arm/.venv/bin/python}

trap 'echo "== stage \"$STAGE\" failed" >&2' ERR

case "$STAGE" in
  probe)    STEPS=200;    SAVE=200;   OUT_JOB=$PROBE_JOB ;;
  validate) STEPS=20000;  SAVE=10000; OUT_JOB=$JOB ;;
  full)     STEPS=100000; SAVE=20000; OUT_JOB=$JOB ;;
  local)    STEPS=2000;   SAVE=1000;  OUT_JOB=$LOCAL_JOB ;;
  fetch)    STEPS=0;      SAVE=0;     OUT_JOB=$JOB ;;
  *) echo "Usage: $0 {probe|validate|full|local|fetch}" >&2; exit 1 ;;
esac

# --- Apple GPU fallback: everything stays on this machine, no ssh, no rsync ---
if [ "$STAGE" = "local" ]; then
  echo "== training locally on the Apple GPU ($STEPS steps)"
  START=$(date +%s)
  "$LOCAL_PY" -m lerobot.scripts.lerobot_train \
    --policy.type=act \
    --policy.device=mps \
    --policy.push_to_hub=false \
    --dataset.repo_id=local/$DATASET \
    --dataset.root="$HERE/datasets/$DATASET" \
    --output_dir="$HERE/outputs/train/$LOCAL_JOB" \
    --job_name=$LOCAL_JOB \
    --steps=$STEPS \
    --save_freq=$SAVE \
    --batch_size=8
  echo "== local finished in $(( $(date +%s) - START ))s"
  echo "local checkpoints:"
  ls "$HERE/outputs/train/$LOCAL_JOB/checkpoints" || echo "(none written)"
  exit 0
fi

: "${GPU_HOST:?set GPU_HOST, e.g. user@gpubox}"
: "${GPU_DIR:?set GPU_DIR, e.g. /home/user/lerobot_pour (absolute remote path)}"

# `full` continues the run `validate` started, so the operator never has to remember a
# flag: the remote output directory's existence decides it. LeRobot refuses to write into
# an existing directory unless it is resuming, so guessing wrong stops the stage either way.
resume_wanted() {
  [ "$STAGE" = "full" ] || return 1
  case "${RESUME:-auto}" in
    auto) ;;
    0|false|no) return 1 ;;
    *) return 0 ;;
  esac
  set +e
  ssh "$GPU_HOST" "test -d \"$GPU_DIR/outputs/train/$JOB\""
  local status=$?
  set -e
  case $status in
    0) return 0 ;;
    1) return 1 ;;
    # ssh itself failed (255 is its connection error); do not read that as "no run yet".
    *) echo "cannot ask $GPU_HOST whether $JOB exists (ssh exit $status)" >&2; exit "$status" ;;
  esac
}

if [ "$STAGE" != "fetch" ]; then
  echo "== syncing dataset to $GPU_HOST"
  ssh "$GPU_HOST" "mkdir -p \"$GPU_DIR/datasets\""
  rsync -a -s --info=progress2 "$HERE/datasets/$DATASET/" "$GPU_HOST:$GPU_DIR/datasets/$DATASET/"

  if resume_wanted; then
    echo "== resuming the existing $JOB run"
    # On resume LeRobot takes the whole configuration from the checkpoint (output_dir
    # included) and applies only the CLI flags given here on top of it.
    TRAIN_ARGS="--config_path=\"$GPU_DIR/outputs/train/$JOB/checkpoints/last/pretrained_model/train_config.json\" \
    --resume=true \
    --steps=$STEPS \
    --save_freq=$SAVE"
  else
    # push_to_hub defaults to true, and LeRobot then refuses to start without a repo_id.
    TRAIN_ARGS="--policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=local/$DATASET \
    --dataset.root=\"$GPU_DIR/datasets/$DATASET\" \
    --output_dir=\"$GPU_DIR/outputs/train/$OUT_JOB\" \
    --job_name=$OUT_JOB \
    --steps=$STEPS \
    --save_freq=$SAVE \
    --batch_size=8"
  fi

  echo "== training ($STAGE, $STEPS steps, output $OUT_JOB)"
  START=$(date +%s)
  ssh "$GPU_HOST" "cd \"$GPU_DIR\" && \"$REMOTE_PY\" -m lerobot.scripts.lerobot_train $TRAIN_ARGS"
  echo "== $STAGE finished in $(( $(date +%s) - START ))s"
fi

if [ "$STAGE" = "probe" ]; then
  echo "== probe measured only; its checkpoint stays on $GPU_HOST and is never deployed"
  exit 0
fi

echo "== fetching checkpoints"
mkdir -p "$HERE/outputs/train/$JOB"
rsync -a -s --info=progress2 "$GPU_HOST:$GPU_DIR/outputs/train/$JOB/" "$HERE/outputs/train/$JOB/"
echo "local checkpoints:"
ls "$HERE/outputs/train/$JOB/checkpoints" || echo "(none fetched)"
