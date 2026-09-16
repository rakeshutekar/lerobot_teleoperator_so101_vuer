#!/usr/bin/env python3
"""Run the trained pouring policy on the real SO-101, then park it.

    ./pour.sh                      # newest checkpoint
    ./pour.sh --checkpoint 040000  # a specific one, for comparison

Safety: 3 deg/tick cap, a hard duration limit, torque held at exit, and park.py
lowers the arm afterwards. Keep the workspace clear and estop_so101.py ready.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

from pipeline.cameras import build_cameras_arg, load_camera_map
from pipeline.checkpoints import resolve_checkpoint
from run_quest import apply_gains

HERE = Path(__file__).resolve().parent
ARM_REPO = HERE.parent / "robot-arm"
TASK = "pour from the bottle into the cup"
JOB = "pour_v1"
logger = logging.getLogger(__name__)


def build_command(checkpoint: Path, port: str, cameras: str, duration: int) -> list[str]:
    """Assemble the lerobot-rollout CLI args for one base-mode deployment run."""
    return [
        str(ARM_REPO / ".venv/bin/lerobot-rollout"),
        "--strategy.type=base",
        f"--policy.path={checkpoint}",
        "--robot.type=so101_follower",
        f"--robot.port={port}",
        "--robot.id=so101_follower",
        f"--robot.calibration_dir={HERE / 'calibration'}",
        "--robot.max_relative_target=3",
        "--robot.disable_torque_on_disconnect=false",
        f"--robot.cameras={cameras}",
        f"--task={TASK}",
        f"--duration={duration}",
    ]


def apply_gains_before_rollout(port: str) -> None:
    """Connect once, apply the arm's servo gains, then disconnect with torque held.

    lerobot-rollout's own robot.connect() has no hook for this, so we do it as a
    separate pre-step using the identical robot config the rollout command uses.
    disable_torque_on_disconnect=False means the disconnect below leaves torque on.
    """
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    robot = SOFollower(
        SOFollowerRobotConfig(
            port=port,
            id="so101_follower",
            calibration_dir=HERE / "calibration",
            max_relative_target=3,
            disable_torque_on_disconnect=False,
        )
    )
    robot.connect()
    try:
        apply_gains(robot)
    finally:
        robot.disconnect()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="step number, e.g. 040000")
    parser.add_argument("--port", default="/dev/cu.usbmodem")
    parser.add_argument("--duration", type=int, default=60, help="hard time limit, seconds")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()

    try:
        checkpoint = resolve_checkpoint(Path.cwd() / "outputs/train" / JOB, args.checkpoint)
        cameras = build_cameras_arg(load_camera_map(Path.cwd() / "cameras.json"))
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error

    command = build_command(checkpoint, args.port, cameras, args.duration)
    if args.print_command:
        print(" ".join(command))
        return

    logger.info("policy: %s", checkpoint)
    apply_gains_before_rollout(args.port)
    try:
        subprocess.run(command, check=False)
    finally:
        logger.info("parking")
        subprocess.run([str(ARM_REPO / ".venv/bin/python"), str(HERE / "park.py")], check=False)


if __name__ == "__main__":
    main()
