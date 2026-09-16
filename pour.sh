#!/usr/bin/env bash
# Run the trained pouring policy on the arm. Keep estop_so101.py ready.
set -euo pipefail
shopt -s nullglob
HERE="$(cd "$(dirname "$0")" && pwd)"
ARM_REPO="$HERE/../robot-arm"
ports=(/dev/cu.usbmodem*)
if [ -z "${PORT:-}" ]; then
  case ${#ports[@]} in
    0) echo "No /dev/cu.usbmodem* found; plug in the follower arm." >&2; exit 1 ;;
    1) PORT="${ports[0]}" ;;
    *) echo "Several serial devices found (${ports[*]}); set PORT=<follower arm port>." >&2; exit 1 ;;
  esac
fi
cd "$HERE"
exec "$ARM_REPO/.venv/bin/python" "$HERE/pour.py" --port "$PORT" "$@"
