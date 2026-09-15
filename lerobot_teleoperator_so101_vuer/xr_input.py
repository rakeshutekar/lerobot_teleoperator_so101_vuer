"""Parse Vuer CONTROLLER_MOVE / HAND_MOVE payloads into one sample type.

Controllers: grip button = clutch, trigger = close gripper (proportional).
Hands: pinch with the OTHER hand = clutch; tracked hand's thumb-index distance = gripper opening.
Payloads come from any browser on the LAN, so malformed or non-finite data is rejected here.
"""
from dataclasses import dataclass

import numpy as np

GRIP_CLUTCH_THRESHOLD = 0.5            # controller squeezeValue above this engages
PINCH_CLUTCH_M = 0.02                  # other hand's thumb-index distance below this engages
PINCH_CLOSED_M, PINCH_OPEN_M = 0.015, 0.08


@dataclass(frozen=True)
class XRSample:
    pose: np.ndarray       # 4x4 WebXR world pose of the controller grip or wrist
    clutch: bool
    gripper_open: float    # 0 = closed .. 1 = fully open
    source: str = "controller"  # "controller" or "hand": grip and wrist poses sit centimetres apart


def _pose(flat) -> np.ndarray | None:
    try:
        if flat is None or len(flat) < 16:
            return None
        pose = np.asarray(flat[:16], dtype=float).reshape(4, 4).T  # WebGL matrices are column-major
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(pose)) or abs(np.linalg.det(pose[:3, :3]) - 1.0) > 0.1:
        return None  # WebXR poses are rigid; anything else would crash the rotation filter
    return pose


def _state(value: dict, hand: str) -> dict:
    state = value.get(f"{hand}State")
    return state if isinstance(state, dict) else {}


def _number(state: dict, key: str, default: float) -> float:
    try:
        x = float(state.get(key, default))
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def parse_controllers(value: dict, hand: str) -> XRSample | None:
    pose = _pose(value.get(hand)) if isinstance(value, dict) else None
    if pose is None:
        return None
    state = _state(value, hand)
    return XRSample(
        pose=pose,
        clutch=_number(state, "squeezeValue", 0.0) > GRIP_CLUTCH_THRESHOLD,
        gripper_open=float(np.clip(1.0 - _number(state, "triggerValue", 0.0), 0.0, 1.0)),
        source="controller",
    )


def parse_hands(value: dict, hand: str) -> XRSample | None:
    pose = _pose(value.get(hand)) if isinstance(value, dict) else None  # first of 25 joints is the wrist
    if pose is None:
        return None
    other_pinch = _number(_state(value, "left" if hand == "right" else "right"), "pinchValue", np.inf)
    pinch = _number(_state(value, hand), "pinchValue", PINCH_OPEN_M)
    return XRSample(
        pose=pose,
        clutch=other_pinch < PINCH_CLUTCH_M,
        gripper_open=float(np.clip((pinch - PINCH_CLOSED_M) / (PINCH_OPEN_M - PINCH_CLOSED_M), 0.0, 1.0)),
        source="hand",
    )
