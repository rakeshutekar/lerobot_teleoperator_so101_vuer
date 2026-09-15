"""Damped least-squares IK for the SO-101 on the MuJoCo model (same method as robot-arm/draw_real.py).

Solves shoulder_pan..wrist_flex for gripper-tip position with a deliberately weak pointing-axis
term: a 5-axis arm cannot reach most orientations, and a stronger weight dragged joints into their
limits and left the tip up to 27 mm off (task-priority variants stalled or jumped branch at limits).
wrist_roll is taken straight from the target. Warm-started from, and faintly held near, the previous
solution; targets are clamped to the reachable workspace and joint change is capped per tick.
Joint angles are MuJoCo radians = LeRobot degrees 1:1.
"""
from pathlib import Path

import mujoco
import numpy as np

from .clutch import EETarget

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
Q_NOMINAL = np.array([0.0, -0.7, 1.1, 0.45, 0.0])  # draw_real.py's comfortable posture
TIP_SITE, GRIPPER_BODY = "gripperframe", "gripper"
POINTING_LOCAL = np.array([1.0, 0.0, 0.0])  # site +x points out of the jaws

POINTING_WEIGHT = 0.01  # metres of tip error worth one unit of |axis - desired| (0.35 at 20 deg)
POSTURE_WEIGHT = 0.003  # pull toward the previous solution (not a fixed posture: that drifted a still arm 5 deg/tick)
DAMPING = 1e-4
STEP = 0.7
ITERATIONS = 20
LIMIT_MARGIN = np.radians(3.0)

# Safety: out-of-reach targets made the solver flip elbow branch with 88-188 deg single-tick steps.
MAX_STEP = np.radians(5.0)          # max joint change per solve (one control tick)
FLOOR_Z = 0.01                      # m above the base plane
REACH_MIN, REACH_MAX = 0.09, 0.37   # m from the shoulder_lift axis (measured reach 0.082..0.411)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def clamp_to_workspace(tip: np.ndarray, shoulder: np.ndarray) -> np.ndarray:
    """Pull a tip target into a reachable shell around the shoulder and above the floor."""
    offset = np.asarray(tip, dtype=float) - shoulder
    r = float(np.linalg.norm(offset))
    if r > 1e-9:
        offset = offset * (np.clip(r, REACH_MIN, REACH_MAX) / r)
    clamped = shoulder + offset
    return np.array([clamped[0], clamped[1], max(float(clamped[2]), FLOOR_Z)])


class ArmIK:
    def __init__(self, mjcf_path: str | Path):
        path = Path(mjcf_path)
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo model not found: '{path}'. Pass --teleop.mjcf_path=<robot-arm>/sim/so101/scene.xml")
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
        self._body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self._qadr = np.array([self.model.jnt_qposadr[j] for j in jids])
        self._dofadr = np.array([self.model.jnt_dofadr[j] for j in jids])
        ranges = self.model.jnt_range[jids]
        self.lower, self.upper = ranges[:, 0] + LIMIT_MARGIN, ranges[:, 1] - LIMIT_MARGIN
        self._shoulder_jid = jids[1]

        self._set(Q_NOMINAL)
        roll_axis_world = self.data.xaxis[jids[4]]
        self.roll_sign = float(np.sign(np.dot(roll_axis_world, self._pointing())))

    def _set(self, q: np.ndarray) -> None:
        self.data.qpos[self._qadr] = q
        mujoco.mj_forward(self.model, self.data)

    def _pointing(self) -> np.ndarray:
        return self.data.site_xmat[self._site].reshape(3, 3) @ POINTING_LOCAL

    def fk(self, q: np.ndarray) -> EETarget:
        self._set(np.asarray(q, dtype=float))
        return EETarget(tip=self.data.site_xpos[self._site].copy(), axis=self._pointing(), roll=self.roll_sign * float(q[4]))

    def solve(self, seed: np.ndarray, target: EETarget) -> np.ndarray:
        """One warm-started IK solve. Returns new joint angles (radians); `seed` is not modified.

        The tip target is clamped to the workspace, a joint already past the IK margin (e.g. the
        folded rest elbow) is never pulled inward, and no joint moves more than MAX_STEP per call.
        """
        seed = np.array(seed, dtype=float)
        lower, upper = np.minimum(self.lower, seed), np.maximum(self.upper, seed)
        self._set(seed)
        tip_des = clamp_to_workspace(target.tip, self.data.xanchor[self._shoulder_jid].copy())
        q = seed.copy()
        q[4] = np.clip(self.roll_sign * target.roll, lower[4], upper[4])
        axis_des = target.axis / np.linalg.norm(target.axis)
        dofs = self._dofadr[:4]

        for _ in range(ITERATIONS):
            self._set(q)
            tip, axis = self.data.site_xpos[self._site].copy(), self._pointing()
            jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, tip, self._body)
            # d(axis)/dq = omega x axis = -skew(axis) @ jacr: the exact Jacobian of the axis error
            J = np.vstack([jacp[:, dofs], POINTING_WEIGHT * (-_skew(axis) @ jacr[:, dofs]), POSTURE_WEIGHT * np.eye(4)])
            e = np.concatenate([
                tip_des - tip,
                POINTING_WEIGHT * (axis_des - axis),
                POSTURE_WEIGHT * (seed[:4] - q[:4]),
            ])
            dq = np.linalg.solve(J.T @ J + DAMPING * np.eye(4), J.T @ e)
            q[:4] = np.clip(q[:4] + STEP * dq, lower[:4], upper[:4])
            if np.max(np.abs(dq)) < 1e-6:
                break
        return seed + np.clip(q - seed, -MAX_STEP, MAX_STEP)
