"""One-euro filter (Casiez et al., CHI 2012) for noisy XR tracking, as pure functions.

Low speed -> low cutoff (removes tremor); high speed -> cutoff rises with `beta` (little lag).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OneEuroState:
    x: np.ndarray | None = None   # last filtered value
    dx: np.ndarray | None = None  # last filtered derivative
    t: float | None = None        # time of last sample, seconds


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


def one_euro_step(
    state: OneEuroState, x: np.ndarray, t: float, min_cutoff: float, beta: float, d_cutoff: float = 1.0
) -> tuple[OneEuroState, np.ndarray]:
    """Filter one sample. Returns (new state, filtered value); never mutates its inputs."""
    x = np.array(x, dtype=float)
    if state.x is None or state.t is None:
        return OneEuroState(x=x, dx=np.zeros_like(x), t=t), x.copy()
    dt = t - state.t
    if dt <= 0.0:  # duplicate or out-of-order sample: keep the previous estimate
        return state, state.x.copy()

    dx_hat = _alpha(d_cutoff, dt) * ((x - state.x) / dt) + (1.0 - _alpha(d_cutoff, dt)) * state.dx
    a = _alpha(min_cutoff + beta * float(np.linalg.norm(dx_hat)), dt)
    x_hat = a * x + (1.0 - a) * state.x
    return OneEuroState(x=x_hat, dx=dx_hat, t=t), x_hat.copy()


def quat_continuous(prev: np.ndarray | None, q: np.ndarray) -> np.ndarray:
    """q and -q are the same rotation; pick the sign nearest `prev` so filtering never averages across."""
    q = np.array(q, dtype=float)
    if prev is None:
        return q
    return -q if float(np.dot(prev, q)) < 0.0 else q
