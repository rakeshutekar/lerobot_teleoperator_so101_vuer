#!/usr/bin/env bash
# Leader-arm recording, the other half of the Quest-vs-leader A/B. See record_leader.py.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARM_REPO="$HERE/../robot-arm"
: "${FOLLOWER_PORT:?set FOLLOWER_PORT=/dev/cu.usbmodem... (Righty)}"
: "${LEADER_PORT:?set LEADER_PORT=/dev/cu.usbmodem... (leader arm)}"
cd "$HERE"
"$ARM_REPO/.venv/bin/python" "$HERE/preflight_cameras.py" || { echo "camera pre-flight failed; not recording" >&2; exit 1; }
exec "$ARM_REPO/.venv/bin/python" "$HERE/record_leader.py" "$@"
