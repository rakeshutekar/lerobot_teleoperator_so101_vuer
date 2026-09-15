#!/usr/bin/env python3
"""Record pouring demonstrations: lerobot-record driven by the Quest teleoperator.

Same safety envelope as run_quest.py (gains, 3 deg/tick cap, torque held at exit).
Keys during recording: right arrow ends the episode, left arrow re-records it,
escape stops the session.

    ./record_pour.sh                       # 50 episodes
    ./record_pour.sh --dataset.num_episodes=5 --resume   # add five more later
"""
import logging
import sys
from pathlib import Path

from pipeline.cameras import build_cameras_arg, load_camera_map
from run_quest import apply_gains

HERE = Path(__file__).resolve().parent
ARM_REPO = HERE.parent / "robot-arm"
logger = logging.getLogger(__name__)

DEFAULT_ARGS = [
    "--robot.type=so101_follower",
    "--robot.id=so101_follower",
    f"--robot.calibration_dir={HERE / 'calibration'}",
    "--robot.max_relative_target=3",
    "--robot.disable_torque_on_disconnect=false",
    "--teleop.type=so101_vuer",
    "--teleop.id=quest",
    f"--teleop.vuer_cert={HERE / 'cert.pem'}",
    f"--teleop.vuer_key={HERE / 'key.pem'}",
    f"--teleop.mjcf_path={ARM_REPO / 'sim/so101/scene.xml'}",
    "--dataset.repo_id=local/so101_pour_bottle_v1",
    f"--dataset.root={HERE / 'datasets/so101_pour_bottle_v1'}",
    '--dataset.single_task=pour from the bottle into the cup',
    "--dataset.fps=30",
    "--dataset.num_episodes=50",
    "--dataset.episode_time_s=40",
    "--dataset.reset_time_s=15",
    "--dataset.push_to_hub=false",
    "--fps=30",
]


def build_command(port: str, camera_map_path: Path) -> list[str]:
    """Assemble the lerobot-record CLI args, cameras included, for one port."""
    cameras = build_cameras_arg(load_camera_map(camera_map_path))
    return [*DEFAULT_ARGS, f"--robot.port={port}", f"--robot.cameras={cameras}"]


def apply_gains_before_recording(port: str) -> None:
    """Connect once, apply the arm's servo gains, then disconnect with torque held.

    lerobot-record's own robot.connect() has no hook for this, so we do it as a
    separate pre-step using the identical robot config the recording command uses.
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
    argv = sys.argv[1:]
    printing = "--print-command" in argv
    argv = [a for a in argv if a != "--print-command"]
    port = next((a.split("=", 1)[1] for a in argv if a.startswith("--robot.port=")), "/dev/cu.usbmodem")
    try:
        command = build_command(port, Path.cwd() / "cameras.json")
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    if printing:
        print(" ".join(command))
        return

    apply_gains_before_recording(port)

    from lerobot.scripts import lerobot_record as lr

    sys.argv = [sys.argv[0], *command, *argv]
    lr.register_third_party_plugins()
    lr.main()


if __name__ == "__main__":
    main()
