import numpy as np

from lerobot_teleoperator_so101_vuer.xr_input import parse_controllers, parse_hands


def col_major(T):
    return list(np.asarray(T).T.reshape(-1))


def pose_at(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def test_controller_pose_is_parsed_column_major():
    value = {"right": col_major(pose_at(0.1, 1.2, -0.3)), "rightState": {"squeezeValue": 0.9, "triggerValue": 0.0}}
    sample = parse_controllers(value, "right")
    np.testing.assert_allclose(sample.pose, pose_at(0.1, 1.2, -0.3))


def test_controller_grip_is_clutch_and_trigger_closes_gripper():
    held = parse_controllers({"right": col_major(np.eye(4)), "rightState": {"squeezeValue": 0.8, "triggerValue": 0.25}}, "right")
    loose = parse_controllers({"right": col_major(np.eye(4)), "rightState": {"squeezeValue": 0.2, "triggerValue": 1.0}}, "right")
    assert held.clutch is True and np.isclose(held.gripper_open, 0.75)
    assert loose.clutch is False and np.isclose(loose.gripper_open, 0.0)


def test_missing_or_short_controller_data_is_ignored():
    assert parse_controllers({}, "right") is None
    assert parse_controllers({"right": [0.0] * 5}, "right") is None


def test_non_finite_or_malformed_pose_is_rejected():
    nan_pose = col_major(np.eye(4))
    nan_pose[12] = float("nan")
    assert parse_controllers({"right": nan_pose, "rightState": {}}, "right") is None
    assert parse_controllers({"right": ["a"] * 16}, "right") is None
    assert parse_hands({"right": {"not": "a list"}}, "right") is None


def test_non_rigid_rotation_is_rejected():
    mirrored = np.diag([-1.0, 1.0, 1.0, 1.0])
    assert parse_controllers({"right": col_major(mirrored)}, "right") is None


def test_non_finite_or_malformed_buttons_fall_back_to_released():
    sample = parse_controllers(
        {"right": col_major(np.eye(4)), "rightState": {"squeezeValue": float("inf"), "triggerValue": "x"}}, "right"
    )
    assert sample.clutch is False and sample.gripper_open == 1.0
    assert parse_hands({"right": col_major(np.eye(4)), "rightState": "junk", "leftState": {"pinchValue": float("nan")}}, "right").clutch is False


def test_samples_report_their_source():
    assert parse_controllers({"right": col_major(np.eye(4))}, "right").source == "controller"
    assert parse_hands({"right": col_major(np.eye(4))}, "right").source == "hand"


def test_hand_pinch_distance_maps_proportionally_to_gripper():
    wide = parse_hands({"right": col_major(np.eye(4)) * 25, "rightState": {"pinchValue": 0.08}}, "right")
    shut = parse_hands({"right": col_major(np.eye(4)) * 25, "rightState": {"pinchValue": 0.015}}, "right")
    assert np.isclose(wide.gripper_open, 1.0) and np.isclose(shut.gripper_open, 0.0)


def test_other_hand_pinch_is_the_clutch():
    base = {"right": col_major(np.eye(4)) * 25, "rightState": {"pinchValue": 0.05}}
    engaged = parse_hands({**base, "leftState": {"pinchValue": 0.01}}, "right")
    no_left = parse_hands(base, "right")
    assert engaged.clutch is True and no_left.clutch is False
