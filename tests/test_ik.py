import numpy as np
import pytest

from lerobot_teleoperator_so101_vuer.clutch import EETarget
from lerobot_teleoperator_so101_vuer.ik import Q_NOMINAL, ArmIK


@pytest.fixture(scope="module")
def ik(scene_path):
    return ArmIK(scene_path)


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def converge(ik, seed, target, calls=10):
    q = seed
    for _ in range(calls):
        q = ik.solve(q, target)
    return q


def test_fk_matches_known_nominal_tip(ik):
    ee = ik.fk(Q_NOMINAL)
    np.testing.assert_allclose(ee.tip, [0.244, 0.0, 0.048], atol=2e-3)
    np.testing.assert_allclose(ee.axis, [0.660, 0.0, -0.751], atol=2e-3)


def test_roll_sign_for_this_model(ik):
    assert ik.roll_sign == -1.0


def test_recovers_a_reachable_pose(ik):
    goal = ik.fk(np.array([0.4, -0.4, 0.9, 0.3, 0.6]))
    q = converge(ik, Q_NOMINAL, goal)
    got = ik.fk(q)
    assert np.linalg.norm(got.tip - goal.tip) < 1e-3
    assert angle_between(got.axis, goal.axis) < 3.0


def test_warm_start_tracks_a_path_without_joint_jumps(ik):
    start = ik.fk(Q_NOMINAL)
    q = Q_NOMINAL
    for k in range(1, 81):  # 8 cm forward in 1 mm steps
        prev = q
        q = ik.solve(q, EETarget(tip=start.tip + [k * 1e-3, 0.0, 0.0], axis=start.axis, roll=start.roll))
        assert np.max(np.abs(q - prev)) < 0.05


def test_unreachable_target_stays_finite_and_in_limits(ik):
    q = converge(ik, Q_NOMINAL, EETarget(tip=np.array([1.0, 0.0, 0.5]), axis=np.array([1.0, 0.0, 0.0]), roll=0.0))
    assert np.all(np.isfinite(q))
    assert np.all(q >= ik.lower - 1e-9) and np.all(q <= ik.upper + 1e-9)


def test_roll_is_passed_straight_to_the_wrist(ik):
    q = converge(ik, Q_NOMINAL, EETarget(tip=ik.fk(Q_NOMINAL).tip, axis=ik.fk(Q_NOMINAL).axis, roll=0.7))
    assert np.isclose(q[4], ik.roll_sign * 0.7)


def test_position_wins_over_an_impossible_pointing_request(ik):
    tip = ik.fk(Q_NOMINAL).tip
    q = converge(ik, Q_NOMINAL, EETarget(tip=tip, axis=np.array([0.0, 1.0, 0.0]), roll=0.0), calls=20)
    assert np.linalg.norm(ik.fk(q).tip - tip) < 5e-3


@pytest.mark.parametrize("direction, distance", [((1, 0, 0), 0.40), ((0, 1, 0), 0.50), ((-1, 0, 0), 0.30), ((0, 0, -1), 0.30)])
def test_out_of_reach_drags_never_jump_joints(ik, direction, distance):
    """Regression: dragging the target out of reach flipped the elbow with 88-188 deg single-tick steps."""
    start = ik.fk(Q_NOMINAL)
    q, n = Q_NOMINAL, int(distance / 0.005)
    for k in list(range(1, n + 1)) + list(range(n, -1, -1)):
        prev = q
        q = ik.solve(q, EETarget(tip=start.tip + np.array(direction) * k * 0.005, axis=start.axis, roll=0.0))
        assert np.max(np.abs(q - prev)) <= np.radians(5.0) + 1e-9, f"step {np.degrees(np.max(np.abs(q - prev))):.1f} deg at {k * 5} mm"
    q = converge(ik, q, start)
    assert np.linalg.norm(ik.fk(q).tip - start.tip) < 1e-3


def test_tip_never_goes_below_the_floor(ik):
    start = ik.fk(Q_NOMINAL)
    q = Q_NOMINAL
    for k in range(1, 61):
        q = ik.solve(q, EETarget(tip=start.tip + np.array([0.0, 0.0, -k * 0.005]), axis=start.axis, roll=0.0))
    assert ik.fk(q).tip[2] >= 0.01 - 1e-3


def test_seed_outside_ik_margin_is_not_yanked_inward(ik):
    """Regression: the rest pose (elbow 97.4 deg, IK margin 93.8) moved the shoulder 15 deg on the first engaged tick."""
    rest = np.radians([2.5, -76.5, 97.4, 17.3, 3.3])
    q = ik.solve(rest, ik.fk(rest))
    assert np.max(np.abs(np.degrees(q - rest))) < 0.5


def test_infeasible_pointing_never_costs_tip_accuracy(ik):
    """Regression: a soft pointing weight pushed joints into limits and left the tip 27 mm off
    on points that are reachable to 0.1 mm when pointing is ignored. The circle is offset 2 cm
    forward and up so every point lies inside the workspace clamp (floor 1 cm, reach >= 9 cm).
    5 mm, not 0: where shoulder_lift sits on its limit the weak pointing term still costs up to
    2.9 mm. Both ways to remove that were worse (a nominal-posture pull drifts a still arm 5 deg per
    tick; masking pointing at limits lost tracking by 107 mm), so the trade is deliberate."""
    start = ik.fk(Q_NOMINAL)
    for k in range(20):
        a = 2 * np.pi * k / 20
        tip = start.tip + np.array([0.05 * (np.cos(a) - 1.0) + 0.02, 0.0, 0.05 * np.sin(a) + 0.02])
        q = converge(ik, Q_NOMINAL, EETarget(tip=tip, axis=start.axis, roll=0.0), calls=60)
        assert np.linalg.norm(ik.fk(q).tip - tip) < 5e-3, f"circle angle {np.degrees(a):.0f} deg"
