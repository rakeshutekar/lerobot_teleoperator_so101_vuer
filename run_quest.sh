#!/usr/bin/env bash
# Teleoperate the real SO-101 from a Meta Quest 3 (Vuer/WebXR, controllers or hand tracking).
# Usage:  ./run_quest.sh [extra lerobot-teleoperate args]      (PORT=/dev/cu.usbmodemXXXX if several)
# Quest URL:  https://<mac-ip>:8012/?ws=wss://<mac-ip>:8012   (Mac IP: ipconfig getifaddr en0)
# Controls:   hold GRIP to move the arm, TRIGGER closes the gripper. Hands: pinch LEFT hand to move,
#             right thumb-index distance opens/closes the gripper. After Ctrl-C: python park.py
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
exec "$ARM_REPO/.venv/bin/python" "$HERE/run_quest.py" \
  --robot.type=so101_follower \
  --robot.port="$PORT" \
  --robot.id=so101_follower \
  --robot.calibration_dir="$HERE/calibration" \
  --robot.max_relative_target=3 \
  --robot.disable_torque_on_disconnect=false \
  --teleop.type=so101_vuer \
  --teleop.id=quest \
  --teleop.vuer_cert="$HERE/cert.pem" \
  --teleop.vuer_key="$HERE/key.pem" \
  --teleop.mjcf_path="$ARM_REPO/sim/so101/scene.xml" \
  --fps=50 \
  "$@"
