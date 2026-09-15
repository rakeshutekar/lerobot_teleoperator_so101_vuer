import numpy as np

from lerobot_teleoperator_so101_vuer.filters import OneEuroState, one_euro_step, quat_continuous

MIN_CUTOFF, BETA = 1.0, 5.0


def run(samples, dt):
    state, out = OneEuroState(), []
    for i, x in enumerate(samples):
        state, y = one_euro_step(state, np.asarray(x, dtype=float), i * dt, MIN_CUTOFF, BETA)
        out.append(y)
    return np.array(out)


def test_first_sample_passes_through_unchanged():
    _, y = one_euro_step(OneEuroState(), np.array([0.1, 0.2, 0.3]), 0.0, MIN_CUTOFF, BETA)
    np.testing.assert_allclose(y, [0.1, 0.2, 0.3])


def test_constant_signal_is_not_distorted():
    out = run([[0.3, -0.1, 1.2]] * 50, dt=1 / 90)
    np.testing.assert_allclose(out[-1], [0.3, -0.1, 1.2], atol=1e-12)


def test_reduces_jitter_on_a_static_hand():
    rng = np.random.default_rng(0)
    noisy = 0.3 + rng.normal(0.0, 0.005, size=(400, 3))
    out = run(noisy, dt=1 / 90)
    assert out[100:].std() < 0.4 * noisy[100:].std()


def test_follows_fast_motion_with_small_lag():
    t = np.arange(90) / 90
    ramp = np.stack([0.5 * t, np.zeros_like(t), np.zeros_like(t)], axis=1)  # 0.5 m/s
    out = run(ramp, dt=1 / 90)
    assert abs(out[-1, 0] - ramp[-1, 0]) < 0.035


def test_step_does_not_mutate_previous_state_or_input():
    state, _ = one_euro_step(OneEuroState(), np.zeros(3), 0.0, MIN_CUTOFF, BETA)
    x = np.array([1.0, 1.0, 1.0])
    new_state, _ = one_euro_step(state, x, 0.01, MIN_CUTOFF, BETA)
    np.testing.assert_array_equal(state.x, np.zeros(3))
    np.testing.assert_array_equal(x, [1.0, 1.0, 1.0])
    assert new_state is not state


def test_quat_continuous_flips_to_previous_hemisphere():
    prev = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat_continuous(prev, -prev), prev)
    np.testing.assert_allclose(quat_continuous(None, -prev), -prev)
