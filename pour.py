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


def run_subprocess(command: list[str], label: str) -> int:
    """Run a subprocess and return its exit code; a failure to launch it counts as failure."""
    try:
        result = subprocess.run(command, check=False)
        return result.returncode
    except OSError as error:
        logger.error("failed to launch %s: %s", label, error)
        return 1


def park_arm(park_python: str, park_script: Path) -> int:
    """Run park.py and, if it fails, say plainly that the arm may still be raised."""
    returncode = run_subprocess([park_python, str(park_script)], "park.py")
    if returncode != 0:
        logger.error(
            "park.py failed (exit %d); the arm may still be raised and under torque. "
            "Run estop_so101.py in the robot-arm repo to release torque, then retry park.py.",
            returncode,
        )
    return returncode


def run_rollout_and_park(command: list[str], park_python: str, park_script: Path) -> int:
    """Run the rollout, always park afterwards, and fail loudly if either step failed."""
    rollout_returncode = 1
    try:
        rollout_returncode = run_subprocess(command, "lerobot-rollout")
    finally:
        logger.info("parking")
        park_returncode = park_arm(park_python, park_script)
    return park_returncode if park_returncode != 0 else rollout_returncode


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
    exit_code = run_rollout_and_park(
        command, str(ARM_REPO / ".venv/bin/python"), HERE / "park.py"
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
