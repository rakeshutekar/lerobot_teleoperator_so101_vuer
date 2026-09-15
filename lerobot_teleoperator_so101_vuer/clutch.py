"""Clutch + relative mapping from XR hand motion to a gripper target in the robot base frame.

While the clutch is held, the target moves by the hand's motion since the clutch was engaged
(scaled for position); releasing freezes the arm so the hand can be repositioned freely.

Frames: robot base +X forward, +Y left, +Z up. WebXR +X right, +Y up, -Z forward (the way you
faced at recenter). Stand behind the arm, facing the direction it points.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation as R

M_VR_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
HAND_FORWARD_ROBOT = M_VR_TO_ROBOT @ np.array([0.0, 0.0, -1.0])  # WebXR local -Z, in robot axes


@dataclass(frozen=True)
class EETarget:
    tip: np.ndarray   # gripper tip position, robot base frame, metres
    axis: np.ndarray  # unit pointing direction of the gripper
    roll: float       # right-handed rotation about `axis`, radians


@dataclass(frozen=True)
class ClutchState:
    engaged: bool = False
    hand_pos0: np.ndarray | None = None  # hand position at engage, robot axes
    hand_rot0: np.ndarray | None = None  # hand orientation at engage, robot axes
    anchor: EETarget | None = None       # arm target at engage
    twist: float = 0.0                   # accumulated (unwrapped) hand twist since engage, radians


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _to_robot_axes(hand_T_vr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return M_VR_TO_ROBOT @ hand_T_vr[:3, 3], M_VR_TO_ROBOT @ hand_T_vr[:3, :3] @ M_VR_TO_ROBOT.T


def _twist_angle(rotation: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle in (-pi, pi] of the component of `rotation` about unit `axis` (swing-twist split)."""
    x, y, z, w = R.from_matrix(rotation).as_quat()
    return _wrap(2.0 * np.arctan2(float(np.dot([x, y, z], axis)), w))


def clutch_step(
    state: ClutchState, engaged: bool, hand_T_vr: np.ndarray, current: EETarget, motion_scale: float
) -> tuple[ClutchState, EETarget | None]:
    """Advance the clutch. Returns (new state, target), target None while released."""
    if not engaged:
        return (ClutchState() if state.engaged else state), None

    pos, rot = _to_robot_axes(hand_T_vr)
    if not state.engaged:  # rising edge: anchor here, so engaging never moves the arm
        return ClutchState(engaged=True, hand_pos0=pos, hand_rot0=rot, anchor=current), current

    delta_rot = rot @ state.hand_rot0.T
    axis = delta_rot @ state.anchor.axis
    twist = state.twist + _wrap(_twist_angle(delta_rot, state.hand_rot0 @ HAND_FORWARD_ROBOT) - state.twist)
    target = EETarget(
        tip=state.anchor.tip + motion_scale * (pos - state.hand_pos0),
        axis=axis / np.linalg.norm(axis),
        roll=state.anchor.roll + twist,
    )
    return replace(state, twist=twist), target
