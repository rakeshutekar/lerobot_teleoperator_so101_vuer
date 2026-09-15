import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot_teleoperator_so101_vuer.clutch import ClutchState, EETarget, clutch_step

SCALE = 0.5
START = EETarget(tip=np.array([0.24, 0.0, 0.05]), axis=np.array([1.0, 0.0, 0.0]), roll=0.0)


def vr_pose(pos=(0.0, 1.0, 0.0), rot=None):
    T = np.eye(4)
    T[:3, :3] = np.eye(3) if rot is None else rot
    T[:3, 3] = pos
    return T


def engage_then(pose_after, current=START, pose_before=None):
    state, target = clutch_step(ClutchState(), True, pose_before if pose_before is not None else vr_pose(), current, SCALE)
    return clutch_step(state, True, pose_after, target, SCALE)


def test_disengaged_gives_no_target():
    _, target = clutch_step(ClutchState(), False, vr_pose(), START, SCALE)
    assert target is None


def test_engaging_does_not_jump():
    _, target = clutch_step(ClutchState(), True, vr_pose(pos=(5.0, -3.0, 2.0)), START, SCALE)
    np.testing.assert_allclose(target.tip, START.tip)
    np.testing.assert_allclose(target.axis, START.axis)
    assert target.roll == START.roll


def test_hand_forward_moves_tip_forward_scaled():
    _, target = engage_then(vr_pose(pos=(0.0, 1.0, -0.10)))  # VR -Z is forward
    np.testing.assert_allclose(target.tip, START.tip + [0.05, 0.0, 0.0], atol=1e-9)


def test_hand_up_and_left_map_to_robot_z_and_y():
    _, up = engage_then(vr_pose(pos=(0.0, 1.10, 0.0)))
    _, left = engage_then(vr_pose(pos=(-0.10, 1.0, 0.0)))
    np.testing.assert_allclose(up.tip, START.tip + [0.0, 0.0, 0.05], atol=1e-9)
    np.testing.assert_allclose(left.tip, START.tip + [0.0, 0.05, 0.0], atol=1e-9)


def test_reclutching_after_moving_hand_does_not_jump():
    state, moved = engage_then(vr_pose(pos=(0.0, 1.0, -0.10)))
    state, released = clutch_step(state, False, vr_pose(pos=(0.3, 0.5, 0.4)), moved, SCALE)
    assert released is None
    _, regrabbed = clutch_step(state, True, vr_pose(pos=(0.3, 0.5, 0.4)), moved, SCALE)
    np.testing.assert_allclose(regrabbed.tip, moved.tip)


def test_turning_hand_left_turns_pointing_axis_left():
    turn = R.from_euler("y", 30, degrees=True).as_matrix()  # VR +Y is up = robot +Z
    _, target = engage_then(vr_pose(rot=turn))
    np.testing.assert_allclose(target.axis, [np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0], atol=1e-9)


def test_twisting_hand_about_its_forward_axis_changes_roll_only():
    twist = R.from_rotvec(np.radians(40) * np.array([0.0, 0.0, -1.0])).as_matrix()  # about VR forward
    _, target = engage_then(vr_pose(rot=twist))
    assert np.isclose(target.roll, np.radians(40), atol=1e-9)
    np.testing.assert_allclose(target.axis, START.axis, atol=1e-9)


def test_roll_keeps_accumulating_past_half_a_turn():
    """Regression: the twist wrapped at +-180 deg, flipping the wrist target by a full turn."""
    state, target = clutch_step(ClutchState(), True, vr_pose(), START, SCALE)
    for deg in (60, 120, 170, 190, 240):
        twist = R.from_rotvec(np.radians(deg) * np.array([0.0, 0.0, -1.0])).as_matrix()
        state, target = clutch_step(state, True, vr_pose(rot=twist), target, SCALE)
    assert np.isclose(target.roll, np.radians(240), atol=1e-9)


def test_step_returns_new_state_and_leaves_old_one_untouched():
    old = ClutchState()
    new, _ = clutch_step(old, True, vr_pose(), START, SCALE)
    assert new is not old and old.engaged is False and new.engaged is True
