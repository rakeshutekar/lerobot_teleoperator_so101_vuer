#!/usr/bin/env python3
"""Record demonstrations from the LEADER ARM instead of the Quest — the other half of the A/B.

Same follower safety envelope as record_pour.py (gains re-applied after LeRobot's own
configure(), non-interactive connect, 3 deg/tick cap, torque held at exit). No teleop
seeding hook: the leader arm's own pose is the command, so there is nothing to seed.

Two serial devices are present for this, so both ports are explicit:

    FOLLOWER_PORT=/dev/cu.usbmodem5B3D0423741 LEADER_PORT=/dev/cu.usbmodem5B3D0425321 \\
        ./record_leader.sh --dataset.num_episodes=10

Writes to datasets/so101_pour_bottle_leader_v1 by default so it never collides with the
Quest dataset. Keys during recording are LeRobot's: right arrow ends the episode, left
arrow re-records it, escape stops.

Compare the two datasets before choosing which to scale to 50:
  - per-joint action jerk (second difference of action over time), lower is smoother
  - episode length variance
  - how many episodes you re-recorded
"""
import logging
import os
import sys
from pathlib import Path

from pipeline.cameras import build_cameras_arg, load_camera_map
from pipeline.hooks import install_gains, install_noninteractive_connect

HERE = Path(__file__).resolve().parent
ARM_REPO = HERE.parent / "robot-arm"
logger = logging.getLogger(__name__)

DATASET = os.environ.get("DATASET", "so101_pour_bottle_leader_v1")
TASK = os.environ.get("TASK", "pour from the bottle into the cup")

DEFAULT_ARGS = [
    "--robot.type=so101_follower",
    "--robot.id=so101_follower",
    f"--robot.calibration_dir={HERE / 'calibration'}",
    "--robot.max_relative_target=3",
    "--robot.disable_torque_on_disconnect=false",
    "--teleop.type=so101_leader",
    "--teleop.id=so101_leader",
    f"--teleop.calibration_dir={ARM_REPO / 'calibration'}",
    f"--dataset.repo_id=local/{DATASET}",
    f"--dataset.root={HERE / 'datasets' / DATASET}",
    f"--dataset.single_task={TASK}",
    "--dataset.fps=30",
    "--dataset.num_episodes=10",
    "--dataset.episode_time_s=40",
    "--dataset.reset_time_s=15",
    "--dataset.push_to_hub=false",
    "--fps=30",
]


def build_command(follower_port: str, leader_port: str, camera_map_path: Path) -> list[str]:
    cameras = build_cameras_arg(load_camera_map(camera_map_path))
    return [
        *DEFAULT_ARGS,
        f"--robot.port={follower_port}",
        f"--teleop.port={leader_port}",
        f"--robot.cameras={cameras}",
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argv = sys.argv[1:]
    printing = "--print-command" in argv
    argv = [a for a in argv if a != "--print-command"]

    follower = os.environ.get("FOLLOWER_PORT")
    leader = os.environ.get("LEADER_PORT")
    if not follower or not leader:
        print("set FOLLOWER_PORT and LEADER_PORT (two /dev/cu.usbmodem* devices)", file=sys.stderr)
        raise SystemExit(2)
    if follower == leader:
        print("FOLLOWER_PORT and LEADER_PORT are the same device", file=sys.stderr)
        raise SystemExit(2)

    try:
        command = build_command(follower, leader, Path.cwd() / "cameras.json")
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    if printing:
        print(" ".join(command))
        return

    install_noninteractive_connect()
    install_gains()

    from lerobot.scripts import lerobot_record as lr

    sys.argv = [sys.argv[0], *command, *argv]
    lr.register_third_party_plugins()
    lr.main()


if __name__ == "__main__":
    main()
