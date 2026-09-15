#!/usr/bin/env python3
"""lerobot-teleoperate for the Quest teleop, plus two things this SO-101 needs.

Identical CLI to `lerobot-teleoperate` (same --robot.* / --teleop.* / --fps flags). After connect:
  1. apply_gains(): arm P=32 / base P=16 / Acceleration=30, mirroring robot-arm/teach_pick_place.py
     and dance_real.py; LeRobot's default P=16 makes a gravity-loaded elbow creep on small steps.
  2. seed the teleop with the arm's measured pose, so it holds still until you clutch in.
"""
import logging

from lerobot.scripts import lerobot_teleoperate as lt

ARM_P, BASE_P, ACCEL = 32, 16, 30
logger = logging.getLogger(__name__)


def apply_gains(robot) -> None:
    for joint in robot.bus.motors:
        if joint == "gripper":
            continue
        robot.bus.write("P_Coefficient", joint, BASE_P if joint == "shoulder_pan" else ARM_P, num_retry=5)
        robot.bus.write("Acceleration", joint, ACCEL, num_retry=5)
    logger.info("gains: arm P=%d, base P=%d, accel=%d", ARM_P, BASE_P, ACCEL)


@lt.parser.wrap()
def main(cfg: lt.TeleoperateConfig) -> None:
    lt.init_logging()
    teleop = lt.make_teleoperator_from_config(cfg.teleop)
    robot = lt.make_robot_from_config(cfg.robot)
    teleop_proc, robot_act_proc, robot_obs_proc = lt.make_default_processors()
    teleop.connect()
    robot.connect()
    try:
        apply_gains(robot)
        teleop.seed_from_robot(robot.get_observation())
        lt.teleop_loop(
            teleop=teleop, robot=robot, fps=cfg.fps, display_data=cfg.display_data,
            display_mode=cfg.display_mode, duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_proc, robot_action_processor=robot_act_proc,
            robot_observation_processor=robot_obs_proc,
            display_compressed_images=cfg.display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    lt.register_third_party_plugins()
    main()
