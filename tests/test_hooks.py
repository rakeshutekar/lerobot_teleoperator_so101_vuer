"""The gain, connect and seeding hooks, with no arm and no Vuer server.

Each test replaces the LeRobot method the hook wraps before installing the hook, so the
wrapper is exercised without opening a serial port, and pytest restores the real method.
"""
import pytest

from pipeline import hooks
from run_quest import ACCEL, ARM_P, BASE_P

MEASURED_POSE = {
    "shoulder_pan.pos": 2.5, "shoulder_lift.pos": -76.5, "elbow_flex.pos": 97.4,
    "wrist_flex.pos": 17.3, "wrist_roll.pos": 3.3, "gripper.pos": 4.8,
}


class FakeBus:
    """Records register writes the way apply_gains issues them."""

    def __init__(self):
        self.motors = {
            "shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
            "wrist_flex": 4, "wrist_roll": 5, "gripper": 6,
        }
        self.writes = []

    def write(self, register, motor, value, num_retry=0):
        self.writes.append((register, motor, value))


class FakeRobot:
    def __init__(self, observation=None):
        self.bus = FakeBus()
        self.is_calibrated = True
        self.calibration_fpath = "calibration/so101_follower.json"
        self._observation = observation or MEASURED_POSE

    def get_observation(self):
        return dict(self._observation)


@pytest.fixture(autouse=True)
def no_remembered_robot(monkeypatch):
    """Each test starts with no follower recorded, and leaves none behind."""
    monkeypatch.setattr(hooks, "_connected_robot", None)


@pytest.fixture
def follower_class():
    from lerobot.robots.so_follower.so_follower import SOFollower

    return SOFollower


def writes_of(robot, register):
    return {motor: value for name, motor, value in robot.bus.writes if name == register}


def test_gain_hook_runs_lerobots_configure_first_then_applies_the_gains(monkeypatch, follower_class):
    """LeRobot's configure() writes P=16 last; the hook must land after it, not before."""
    def lerobot_configure(self):
        for motor in self.bus.motors:
            self.bus.write("P_Coefficient", motor, 16)

    monkeypatch.setattr(follower_class, "configure", lerobot_configure)
    hooks.install_gains()
    robot = FakeRobot()

    follower_class.configure(robot)

    assert robot.bus.writes[0] == ("P_Coefficient", "shoulder_pan", 16), "the original ran first"
    p_coefficients = writes_of(robot, "P_Coefficient")
    assert p_coefficients["elbow_flex"] == ARM_P
    assert p_coefficients["shoulder_pan"] == BASE_P
    assert p_coefficients["gripper"] == 16, "the gripper keeps LeRobot's value"
    assert set(writes_of(robot, "Acceleration").values()) == {ACCEL}


def test_gain_hook_is_installed_once_however_often_it_is_called(monkeypatch, follower_class):
    monkeypatch.setattr(follower_class, "configure", lambda self: None)
    hooks.install_gains()
    hooks.install_gains()
    robot = FakeRobot()

    follower_class.configure(robot)

    assert len(writes_of(robot, "Acceleration")) == 5, "gains applied once per motor"


def test_connect_hook_never_asks_lerobot_to_calibrate(monkeypatch, follower_class):
    """A calibration prompt at the head of an autonomous run would block it forever."""
    seen = {}

    def lerobot_connect(self, calibrate: bool = True):
        seen["calibrate"] = calibrate

    monkeypatch.setattr(follower_class, "connect", lerobot_connect)
    hooks.install_noninteractive_connect()
    robot = FakeRobot()

    follower_class.connect(robot)

    assert seen["calibrate"] is False


def test_connect_hook_reports_a_calibration_mismatch_instead_of_prompting(
    monkeypatch, follower_class, caplog
):
    monkeypatch.setattr(follower_class, "connect", lambda self, calibrate=True: None)
    hooks.install_noninteractive_connect()
    robot = FakeRobot()
    robot.is_calibrated = False

    with caplog.at_level("ERROR", logger="pipeline.hooks"):
        follower_class.connect(robot)

    assert "re-calibrate" in caplog.text


def connected_teleop(scene_path, tmp_path):
    from lerobot_teleoperator_so101_vuer import So101VuerTeleop, So101VuerTeleopConfig

    teleop = So101VuerTeleop(
        So101VuerTeleopConfig(id="test", calibration_dir=tmp_path, mjcf_path=str(scene_path))
    )
    teleop.connect()
    return teleop


def test_seeding_hook_makes_the_first_action_the_measured_pose_not_the_nominal_one(
    monkeypatch, scene_path, tmp_path, follower_class
):
    """Without seeding the first action commands the nominal posture and the arm lurches to it."""
    from lerobot_teleoperator_so101_vuer import So101VuerTeleop

    # Stand in for connect(): build the kinematics, start no Vuer server or camera threads.
    monkeypatch.setattr(So101VuerTeleop, "connect", lambda self: self._init_kinematics())
    nominal_action = connected_teleop(scene_path, tmp_path).get_action()
    assert nominal_action != pytest.approx(MEASURED_POSE), "the arm is not already at the nominal pose"

    monkeypatch.setattr(follower_class, "connect", lambda self, calibrate=True: None)
    hooks.install_noninteractive_connect()
    hooks.install_teleop_seeding()
    follower_class.connect(FakeRobot())

    assert connected_teleop(scene_path, tmp_path).get_action() == pytest.approx(MEASURED_POSE)


def test_seeding_hook_refuses_to_start_when_no_arm_was_connected(monkeypatch, scene_path, tmp_path):
    from lerobot_teleoperator_so101_vuer import So101VuerTeleop

    monkeypatch.setattr(So101VuerTeleop, "connect", lambda self: self._init_kinematics())
    hooks.install_teleop_seeding()

    with pytest.raises(RuntimeError, match="cannot be seeded"):
        connected_teleop(scene_path, tmp_path)
