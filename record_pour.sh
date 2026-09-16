#!/usr/bin/env bash
# Record pouring demonstrations from the Quest. See record_pour.py for the keys.
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
echo "Arm port: $PORT   Mac IP: $(ipconfig getifaddr en0)"
echo "Quest URL: https://$(ipconfig getifaddr en0):8012/?ws=wss://$(ipconfig getifaddr en0):8012"
cd "$HERE"
exec "$ARM_REPO/.venv/bin/python" "$HERE/record_pour.py" --robot.port="$PORT" "$@"
