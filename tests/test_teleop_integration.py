"""get_action end to end, without the Vuer server or hardware: XR samples in, joint degrees out."""
import numpy as np
import pytest

from lerobot_teleoperator_so101_vuer import So101VuerTeleop, So101VuerTeleopConfig
from lerobot_teleoperator_so101_vuer.ik import ArmIK
from lerobot_teleoperator_so101_vuer.xr_input import XRSample

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
HZ = 60


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def teleop(scene_path, tmp_path, clock):
    t = So101VuerTeleop(So101VuerTeleopConfig(id="test", calibration_dir=tmp_path, mjcf_path=str(scene_path), motion_scale=0.5))
    t._clock = clock
    t._init_kinematics()
    return t


@pytest.fixture
def ik(scene_path):
    return ArmIK(scene_path)


def tip_of(action, ik):
    return ik.fk(np.radians([action[f"{j}.pos"] for j in ARM])).tip


def feed(teleop, clock, pose, clutch, seconds=1.0, gripper_open=0.5, source="controller", act=False):
    """Stream samples at HZ; with act=True the control loop runs alongside, as on the real arm."""
    action = None
    for _ in range(int(seconds * HZ)):
        clock.now += 1.0 / HZ
        teleop._on_sample(XRSample(pose=pose, clutch=clutch, gripper_open=gripper_open, source=source), clock.now)
        if act:
            action = teleop.get_action()
    return action


def settle(teleop, calls=15):
    action = None
    for _ in range(calls):
        action = teleop.get_action()
    return action


def vr(z):
    T = np.eye(4)
    T[:3, 3] = [0.0, 1.0, z]
    return T


def test_before_any_input_it_holds_the_seeded_robot_pose(teleop):
    seeded = {"shoulder_pan.pos": 2.5, "shoulder_lift.pos": -76.5, "elbow_flex.pos": 97.4,
              "wrist_flex.pos": 17.3, "wrist_roll.pos": 3.3, "gripper.pos": 4.8}
    teleop.seed_from_robot(seeded)
    assert teleop.get_action() == pytest.approx(seeded)


def test_clutched_forward_hand_motion_moves_tip_forward_half_as_far(teleop, clock, ik):
    feed(teleop, clock, vr(0.0), clutch=True)
    tip0 = tip_of(settle(teleop, 1), ik)
    feed(teleop, clock, vr(-0.10), clutch=True)
    tip1 = tip_of(settle(teleop), ik)
    np.testing.assert_allclose(tip1 - tip0, [0.05, 0.0, 0.0], atol=5e-3)


def test_released_clutch_freezes_arm_while_gripper_still_follows(teleop, clock, ik):
    feed(teleop, clock, vr(0.0), clutch=True)
    before = settle(teleop)
    feed(teleop, clock, vr(-0.30), clutch=False, gripper_open=0.2)
    after = settle(teleop)
    np.testing.assert_allclose(tip_of(after, ik), tip_of(before, ik), atol=1e-9)
    assert np.isclose(after["gripper.pos"], 20.0, atol=1.0)


def test_tracking_loss_releases_the_clutch_so_the_arm_cannot_jump(teleop, clock, ik):
    """Regression: the last clutched sample stayed live, so a hand reappearing 30 cm away lunged the arm."""
    feed(teleop, clock, vr(0.0), clutch=True)
    before = settle(teleop)
    for _ in range(HZ):  # hand drops out for a second while the control loop keeps running
        clock.now += 1.0 / HZ
        teleop.get_action()
    after = feed(teleop, clock, vr(-0.30), clutch=True, seconds=0.5, act=True)
    np.testing.assert_allclose(tip_of(after, ik), tip_of(before, ik), atol=1e-3)


def test_switching_controller_to_hand_releases_clutch_without_drift(teleop, clock, ik):
    """Regression: grip and wrist poses sit centimetres apart; the shared filter slid between them while clutched."""
    feed(teleop, clock, vr(0.0), clutch=True, source="controller")
    before = settle(teleop)
    after = feed(teleop, clock, vr(-0.05), clutch=True, seconds=0.5, source="hand", act=True)
    np.testing.assert_allclose(tip_of(after, ik), tip_of(before, ik), atol=1e-3)


def test_user_hand_must_be_left_or_right(tmp_path):
    with pytest.raises(ValueError):
        So101VuerTeleopConfig(id="x", calibration_dir=tmp_path, user_hand="Right")
