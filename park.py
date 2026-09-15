#!/usr/bin/env python3
"""Glide the SO-101 slowly to its folded rest pose, then release torque.

Run this after a Quest teleop session (run_quest.sh leaves torque ON at exit so a
raised arm does not drop). Rest pose = what the arm read while resting on 2026-09-14.
    PORT=/dev/cu.usbmodemXXXX python park.py
"""
import glob, os, sys, time
from pathlib import Path
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

HERE = Path(__file__).resolve().parent
REST = {"shoulder_pan.pos": 2.5, "shoulder_lift.pos": -76.5, "elbow_flex.pos": 97.4,
        "wrist_flex.pos": 17.3, "wrist_roll.pos": 3.3, "gripper.pos": 5.0}
STEP_DEG, HZ = 1.0, 30          # 30 deg/s cap, gentle

def main():
    port = os.environ.get("PORT") or (glob.glob("/dev/cu.usbmodem*") or [None])[0]
    if not port:
        sys.exit("No /dev/cu.usbmodem* found; set PORT=...")
    robot = SOFollower(SOFollowerRobotConfig(port=port, id="so101_follower",
                                             calibration_dir=HERE / "calibration",
                                             max_relative_target=STEP_DEG))
    robot.connect(calibrate=False)
    try:
        for _ in range(int(20 * HZ)):                      # up to 20 s
            obs = robot.get_observation()
            if all(abs(obs[k] - v) < 1.5 for k, v in REST.items() if k != "gripper.pos"):
                break
            robot.send_action(REST)
            time.sleep(1.0 / HZ)
        print("parked at", {k: round(v, 1) for k, v in robot.get_observation().items()})
    finally:
        robot.bus.disable_torque()
        robot.disconnect()
        print("torque released")

if __name__ == "__main__":
    main()
