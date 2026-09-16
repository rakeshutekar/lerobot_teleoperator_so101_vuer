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
from collections.abc import Callable
from pathlib import Path

from pipeline.cameras import build_cameras_arg, load_camera_map
from pipeline.checkpoints import resolve_checkpoint
from pipeline.hooks import install_gains, install_noninteractive_connect

HERE = Path(__file__).resolve().parent
ARM_REPO = HERE.parent / "robot-arm"
TASK = "pour from the bottle into the cup"
JOB = "pour_v1"
DEFAULT_FPS = 30
logger = logging.getLogger(__name__)


def build_command(checkpoint: Path, port: str, cameras: str, duration: int, fps: int) -> list[str]:
    """Assemble the lerobot-rollout CLI args for one base-mode deployment run."""
    return [
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
        f"--fps={fps}",
    ]


def run_rollout(command: list[str]) -> int:
    """Run lerobot-rollout in THIS process and return an exit code.

    In-process is what lets the gain and connect hooks reach the robot LeRobot builds,
    and it keeps an interrupt inside our own try/finally, so parking still happens
    instead of the rollout being killed out from under us.
    """
    from lerobot.scripts import lerobot_rollout

    sys.argv = [sys.argv[0], *command]
    try:
        lerobot_rollout.main()
    except KeyboardInterrupt:
        logger.info("rollout interrupted")
        return 130
    except SystemExit as error:
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        logger.error("rollout exited: %s", error.code)
        return 1
    except Exception:
        logger.exception("rollout failed")
        return 1
    return 0


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


def run_rollout_and_park(rollout: Callable[[], int], park_python: str, park_script: Path) -> int:
    """Run the rollout, always park afterwards, and fail loudly if either step failed."""
    rollout_returncode = 1
    try:
        rollout_returncode = rollout()
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
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS,
        help=f"control rate; must match the rate the policy was recorded at (default {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--print-command", action="store_true",
        help="print the rollout arguments and exit, touching no hardware",
    )
    args = parser.parse_args()

    try:
        checkpoint = resolve_checkpoint(Path.cwd() / "outputs/train" / JOB, args.checkpoint)
        cameras = build_cameras_arg(load_camera_map(Path.cwd() / "cameras.json"))
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error

    command = build_command(checkpoint, args.port, cameras, args.duration, args.fps)
    if args.print_command:
        print(" ".join(command))
        return

    logger.info("policy: %s", checkpoint)
    install_noninteractive_connect()
    install_gains()
    exit_code = run_rollout_and_park(
        lambda: run_rollout(command), str(ARM_REPO / ".venv/bin/python"), HERE / "park.py"
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
