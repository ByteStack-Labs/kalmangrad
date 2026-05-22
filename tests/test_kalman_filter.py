"""Tests for the reference Kalman filter implementation.

Each test verifies a property the derivation in docs/derivation.md establishes.
The tests are traceable to the mathematics: the docstrings name the property and
the equation or section being checked. Read alongside the derivation, the suite
is a checklist of what the filter must honor.

The filter under test is the constant-coefficient recursion of Section 8. The
helpers below build a one-dimensional constant-velocity model (state is position
and velocity, a position-only sensor) and generate trajectories that either
match the filter's model exactly (for the innovation property tests) or follow a
deterministic ground truth (for the tracking tests).
"""

from __future__ import annotations

import numpy as np

from kalman_filter import KalmanFilter


def _constant_velocity_model(
    dt: float = 1.0, q: float = 0.01, r: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (F, H, Q, R) for a 1D constant-velocity model.

    The state is [position, velocity]. The sensor observes position only, so
    H = [1, 0]. Q is the continuous white-noise-acceleration discretization,
    which is positive definite. R is a scalar measurement variance.
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R = np.array([[r]])
    return F, H, Q, R


def _simulate_model_consistent(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0_mean: np.ndarray,
    P0: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a measurement sequence that matches the filter's model exactly.

    The true initial state is drawn from N(x0_mean, P0), the process noise from
    N(0, Q), and the measurement noise from N(0, R). A filter initialized at
    (x0_mean, P0) is then correctly specified from the first step, so the
    innovation properties of Section 7 hold without a burn-in transient. Returns
    the measurements with shape (n_steps, m).
    """
    n = F.shape[0]
    m = H.shape[0]
    chol_Q = np.linalg.cholesky(Q)
    chol_R = np.linalg.cholesky(R)
    chol_P0 = np.linalg.cholesky(P0)

    x = x0_mean + chol_P0 @ rng.standard_normal(n)
    measurements = []
    for _ in range(n_steps):
        x = F @ x + chol_Q @ rng.standard_normal(n)
        z = H @ x + chol_R @ rng.standard_normal(m)
        measurements.append(z)
    return np.array(measurements)


def _simulate_constant_velocity_truth(
    v: float,
    dt: float,
    n_steps: int,
    meas_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic constant-velocity ground truth and noisy measurements.

    The true position is exactly v * dt * k. Measurements are the true position
    plus zero-mean Gaussian noise of standard deviation meas_std. Returns
    (true_positions, measurements), each of length n_steps.
    """
    steps = np.arange(1, n_steps + 1)
    true_positions = v * dt * steps
    measurements = true_positions + rng.normal(0.0, meas_std, size=n_steps)
    return true_positions, measurements


# Structural and shape tests


def test_construction_stores_model_and_initial_state() -> None:
    """Construction sets x, P from x0, P0 and stores F, H, Q, R (Section 8.1, eq 8.1)."""
    F, H, Q, R = _constant_velocity_model()
    x0 = np.array([2.0, -1.0])
    P0 = np.array([[3.0, 0.5], [0.5, 4.0]])

    kf = KalmanFilter(F, H, Q, R, x0, P0)

    np.testing.assert_array_equal(kf.x, x0)
    np.testing.assert_array_equal(kf.P, P0)
    np.testing.assert_array_equal(kf.F, F)
    np.testing.assert_array_equal(kf.H, H)
    np.testing.assert_array_equal(kf.Q, Q)
    np.testing.assert_array_equal(kf.R, R)


def test_predict_and_update_preserve_shapes() -> None:
    """predict and update keep x shape (n,) and P shape (n, n) (Sections 4, 5)."""
    F, H, Q, R = _constant_velocity_model()
    n = F.shape[0]
    kf = KalmanFilter(F, H, Q, R, x0=np.zeros(n), P0=np.eye(n))

    kf.predict()
    assert kf.x.shape == (n,)
    assert kf.P.shape == (n, n)

    kf.update(np.array([1.0]))
    assert kf.x.shape == (n,)
    assert kf.P.shape == (n, n)


# Covariance property tests


def test_predict_does_not_shrink_covariance() -> None:
    """Prediction adds Q, so P_pred - P_prior is PSD and the trace grows (eq 4.9).

    With F = I the prediction reduces to P_pred = P_prior + Q. With positive
    definite Q the difference is exactly Q, which is positive definite, so the
    covariance grows in the positive-definite sense and the trace strictly
    increases. This isolates the additive role of the process noise.
    """
    n = 2
    F = np.eye(n)
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.2, 0.05], [0.05, 0.3]])
    R = np.array([[1.0]])
    P_prior = np.array([[1.0, 0.2], [0.2, 1.5]])

    kf = KalmanFilter(F, H, Q, R, x0=np.zeros(n), P0=P_prior)
    kf.predict()

    difference = kf.P - P_prior
    assert np.min(np.linalg.eigvalsh(difference)) > 0.0
    assert np.trace(kf.P) > np.trace(P_prior)


def test_update_does_not_grow_covariance() -> None:
    """An informative measurement reduces uncertainty: trace(P) decreases (eq 5.15).

    The optimal gain gives P_pred - P_updated = K H P_pred, which is positive
    semi-definite. The trace of the covariance therefore does not increase across
    an update, and strictly decreases for an informative measurement.
    """
    F, H, Q, R = _constant_velocity_model()
    kf = KalmanFilter(F, H, Q, R, x0=np.zeros(2), P0=np.eye(2) * 10.0)

    kf.predict()
    trace_before = np.trace(kf.P)
    P_before = kf.P.copy()

    kf.update(np.array([1.0]))
    trace_after = np.trace(kf.P)

    assert np.min(np.linalg.eigvalsh(P_before - kf.P)) > -1e-12
    assert trace_after < trace_before


def test_covariance_stays_symmetric() -> None:
    """The Joseph form keeps P symmetric across many cycles (Section 5.4, eq 5.17)."""
    F, H, Q, R = _constant_velocity_model()
    rng = np.random.default_rng(1)
    _, measurements = _simulate_constant_velocity_truth(
        v=1.0, dt=1.0, n_steps=200, meas_std=1.0, rng=rng
    )

    kf = KalmanFilter(F, H, Q, R, x0=np.zeros(2), P0=np.eye(2) * 10.0)
    max_asymmetry = 0.0
    for z in measurements:
        kf.predict()
        kf.update(np.array([z]))
        max_asymmetry = max(max_asymmetry, np.max(np.abs(kf.P - kf.P.T)))

    assert max_asymmetry < 1e-10


def test_covariance_stays_positive_definite() -> None:
    """The Joseph form keeps P positive definite across many cycles (Section 5.4).

    A covariance matrix must be positive definite. The Joseph form (eq 5.17) is a
    sum of two symmetric positive-semi-definite products, so the smallest
    eigenvalue of P stays positive even over a long run.
    """
    F, H, Q, R = _constant_velocity_model()
    rng = np.random.default_rng(2)
    _, measurements = _simulate_constant_velocity_truth(
        v=1.0, dt=1.0, n_steps=200, meas_std=1.0, rng=rng
    )

    kf = KalmanFilter(F, H, Q, R, x0=np.zeros(2), P0=np.eye(2) * 10.0)
    min_eigenvalue = np.inf
    for z in measurements:
        kf.predict()
        kf.update(np.array([z]))
        min_eigenvalue = min(min_eigenvalue, np.min(np.linalg.eigvalsh(kf.P)))

    assert min_eigenvalue > 0.0


# Estimation correctness tests


def test_filter_tracks_constant_velocity_target() -> None:
    """The filter smooths: its position RMS error is below the sensor noise (Section 6).

    The Kalman filter is the minimum mean squared error estimator under the model
    assumptions (Section 6.1). On a constant-velocity target it must therefore
    track the truth with smaller error than the raw measurements.
    """
    F, H, Q, R = _constant_velocity_model(dt=1.0, q=0.01, r=4.0)
    rng = np.random.default_rng(0)
    meas_std = 2.0
    true_positions, measurements = _simulate_constant_velocity_truth(
        v=1.0, dt=1.0, n_steps=50, meas_std=meas_std, rng=rng
    )

    kf = KalmanFilter(F, H, Q, R, x0=np.zeros(2), P0=np.eye(2) * 100.0)
    estimates = []
    for z in measurements:
        kf.predict()
        kf.update(np.array([z]))
        estimates.append(kf.x[0])
    estimates = np.array(estimates)

    filter_rms = np.sqrt(np.mean((estimates - true_positions) ** 2))
    measurement_rms = np.sqrt(np.mean((measurements - true_positions) ** 2))

    assert filter_rms < measurement_rms


def test_filter_converges_from_wrong_initial_guess() -> None:
    """A weak prior lets the estimate converge to the truth (Section 8.1).

    With a deliberately wrong initial state and a large initial covariance, the
    early estimate is dominated by the wrong guess. As measurements accumulate,
    the influence of the initialization fades and the estimate converges. The
    late-stage error is small relative to the early error.
    """
    F, H, Q, R = _constant_velocity_model(dt=1.0, q=0.01, r=1.0)
    rng = np.random.default_rng(3)
    v_true = 1.0
    n_steps = 100
    true_positions, measurements = _simulate_constant_velocity_truth(
        v=v_true, dt=1.0, n_steps=n_steps, meas_std=1.0, rng=rng
    )
    true_states = np.column_stack([true_positions, np.full(n_steps, v_true)])

    # Wrong initial guess (position off by 100, velocity of the wrong sign) and a
    # weak prior expressed as a large initial covariance.
    kf = KalmanFilter(F, H, Q, R, x0=np.array([100.0, -5.0]), P0=np.eye(2) * 1000.0)
    state_errors = []
    for z, true_state in zip(measurements, true_states, strict=True):
        kf.predict()
        kf.update(np.array([z]))
        state_errors.append(np.linalg.norm(kf.x - true_state))
    state_errors = np.array(state_errors)

    early_error = np.mean(state_errors[:10])
    late_error = np.mean(state_errors[-10:])

    assert late_error < 0.1 * early_error


# Innovation property tests (Section 7)


def test_innovations_are_recorded() -> None:
    """Each update records one innovation and one innovation covariance (Section 8.4)."""
    F, H, Q, R = _constant_velocity_model()
    rng = np.random.default_rng(4)
    n_steps = 25
    _, measurements = _simulate_constant_velocity_truth(
        v=1.0, dt=1.0, n_steps=n_steps, meas_std=1.0, rng=rng
    )

    kf = KalmanFilter(F, H, Q, R, x0=np.zeros(2), P0=np.eye(2))
    for z in measurements:
        kf.predict()
        kf.update(np.array([z]))

    assert len(kf.innovations) == n_steps
    assert len(kf.innovation_covariances) == n_steps


def test_innovation_sequence_is_zero_mean() -> None:
    """On a correctly specified filter the innovations are zero-mean (Section 7.2, eq 7.2).

    The innovation is the measurement minus its prediction. Both the prediction
    error and the measurement noise are zero-mean, so the innovation is too. The
    time-averaged innovation falls within a few standard errors of zero, where the
    standard error is set by the innovation covariance and the sample size.
    """
    F, H, Q, R = _constant_velocity_model(dt=1.0, q=0.02, r=1.0)
    rng = np.random.default_rng(5)
    n_steps = 4000
    x0_mean = np.array([0.0, 1.0])
    P0 = np.eye(2)
    measurements = _simulate_model_consistent(F, H, Q, R, x0_mean, P0, n_steps, rng)

    kf = KalmanFilter(F, H, Q, R, x0=x0_mean, P0=P0)
    for z in measurements:
        kf.predict()
        kf.update(z)

    innovations = np.array(kf.innovations).reshape(-1)
    mean_innovation = np.mean(innovations)

    # Standard error of the mean from the filter's own innovation covariances.
    mean_variance = np.mean([S[0, 0] for S in kf.innovation_covariances])
    standard_error = np.sqrt(mean_variance / n_steps)

    assert abs(mean_innovation) < 4.0 * standard_error


def test_normalized_innovation_squared_is_consistent() -> None:
    """The average NIS is close to m, the chi-squared mean (Section 7.3, eq 7.5).

    On a correctly specified filter the normalized innovation squared,
    nu^T S^-1 nu, follows a chi-squared distribution with m degrees of freedom.
    Its mean is m. The whiteness of the innovation sequence (Section 7.1) makes
    the per-step values independent, so the time-averaged NIS over a long run
    converges to m.
    """
    F, H, Q, R = _constant_velocity_model(dt=1.0, q=0.02, r=1.0)
    rng = np.random.default_rng(6)
    n_steps = 4000
    m = H.shape[0]
    x0_mean = np.array([0.0, 1.0])
    P0 = np.eye(2)
    measurements = _simulate_model_consistent(F, H, Q, R, x0_mean, P0, n_steps, rng)

    kf = KalmanFilter(F, H, Q, R, x0=x0_mean, P0=P0)
    for z in measurements:
        kf.predict()
        kf.update(z)

    nis_values = [
        nu @ np.linalg.solve(S, nu)
        for nu, S in zip(kf.innovations, kf.innovation_covariances, strict=True)
    ]
    average_nis = np.mean(nis_values)

    # Standard error of the mean of n_steps independent chi-squared(m) variables
    # is sqrt(2m / n_steps). Allow four standard errors.
    standard_error = np.sqrt(2.0 * m / n_steps)
    assert abs(average_nis - m) < 4.0 * standard_error


# Time-varying override test


def test_per_step_override_matches_stored_matrices() -> None:
    """Passing the stored matrices explicitly equals the default path (Section 2.1).

    The predict and update methods accept optional per-step F, Q, H, R for the
    time-varying model. Supplying the stored matrices through the override path
    must reproduce the default path exactly.
    """
    F, H, Q, R = _constant_velocity_model()
    x0 = np.array([0.5, -0.3])
    P0 = np.array([[2.0, 0.1], [0.1, 1.0]])
    z = np.array([1.7])

    default = KalmanFilter(F, H, Q, R, x0, P0)
    default.predict()
    default.update(z)

    override = KalmanFilter(F, H, Q, R, x0, P0)
    override.predict(F=F, Q=Q)
    override.update(z, H=H, R=R)

    np.testing.assert_array_equal(default.x, override.x)
    np.testing.assert_array_equal(default.P, override.P)
    np.testing.assert_array_equal(default.innovations[-1], override.innovations[-1])
