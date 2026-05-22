"""Tests for the innovation sequence diagnostics.

Each test verifies a property that Section 7 of docs/derivation.md
establishes for the innovation sequence of a correctly specified filter:
zero mean (Section 7.2), known covariance and the normalized innovation
squared (Section 7.3), and whiteness (Section 7.1). The tests follow the
pattern of test_kalman_filter.py: the data is generated to match the
filter's model exactly, the filter is therefore correctly specified from
the first step, and the tolerances are derived from the sampling
statistics rather than hand-tuned.

The diagnostics are computed by diagnostics.py. The tests check those
functions against an independent computation, against the chi-squared
sampling theory, and against deliberate model misspecification. They are
pure NumPy and deterministic under fixed seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from diagnostics import (
    CHI2_95_INTERVAL,
    average_nis,
    detect_divergence,
    innovation_autocorrelation,
    innovation_mean,
    nees,
    nis,
)
from kalman_filter import KalmanFilter


def _constant_velocity_model(
    dt: float = 1.0, q: float = 0.02, r: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (F, H, Q, R) for a 1D constant-velocity model (n=2, m=1).

    The state is [position, velocity] and the sensor observes position
    only. Q is the white-noise-acceleration discretization, which is
    positive definite. R is a scalar measurement variance.
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R = np.array([[r]])
    return F, H, Q, R


def _two_d_constant_velocity_model(
    dt: float = 1.0, q: float = 0.02, r: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (F, H, Q, R) for a 2D constant-velocity model (n=4, m=2).

    The state is [x, vx, y, vy], two independent constant-velocity
    chains. The sensor observes the two positions, so m=2. F and Q are
    block diagonal across the two chains, which keeps Q full rank and
    positive definite so the model-consistent simulation can factor it.
    """
    f_chain = np.array([[1.0, dt], [0.0, 1.0]])
    q_chain = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    zero = np.zeros((2, 2))
    F = np.block([[f_chain, zero], [zero, f_chain]])
    Q = np.block([[q_chain, zero], [zero, q_chain]])
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    R = r * np.eye(2)
    return F, H, Q, R


def _simulate_with_truth(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0_mean: np.ndarray,
    P0: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate model-consistent true states and measurements.

    The true initial state is drawn from N(x0_mean, P0), process noise
    from N(0, Q), and measurement noise from N(0, R). A filter
    initialized at (x0_mean, P0) is correctly specified from the first
    step. Returns (true_states, measurements) with shapes (n_steps, n)
    and (n_steps, m).
    """
    n = F.shape[0]
    m = H.shape[0]
    chol_Q = np.linalg.cholesky(Q)
    chol_R = np.linalg.cholesky(R)
    chol_P0 = np.linalg.cholesky(P0)

    x = x0_mean + chol_P0 @ rng.standard_normal(n)
    true_states = []
    measurements = []
    for _ in range(n_steps):
        x = F @ x + chol_Q @ rng.standard_normal(n)
        z = H @ x + chol_R @ rng.standard_normal(m)
        true_states.append(x.copy())
        measurements.append(z)
    return np.array(true_states), np.array(measurements)


def _run_filter(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    measurements: np.ndarray,
    true_states: np.ndarray | None = None,
) -> tuple[KalmanFilter, np.ndarray, np.ndarray]:
    """Run the filter, returning it with the per-step errors and covariances.

    When true_states is supplied, the estimation error (true minus
    estimate) is recorded after each update for the NEES check. The
    posterior covariance is always recorded.
    """
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    errors = []
    posteriors = []
    for k, z in enumerate(measurements):
        kf.predict()
        kf.update(z)
        posteriors.append(kf.P.copy())
        if true_states is not None:
            errors.append(true_states[k] - kf.x)
    return kf, np.array(errors), np.array(posteriors)


# Section 7.3: the normalized innovation squared


def test_nis_centers_on_m_scalar() -> None:
    """Average NIS is near m=1 under a correct scalar-measurement model (Section 7.3).

    Each NIS sample is chi-squared(m) with mean m. The whiteness of the
    innovation sequence (Section 7.1) makes the samples independent, so
    the average of n_steps of them has standard error sqrt(2m/n_steps).
    The tolerance is four of those standard errors.
    """
    F, H, Q, R = _constant_velocity_model()
    rng = np.random.default_rng(10)
    n_steps = 4000
    m = H.shape[0]
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2)
    true_states, measurements = _simulate_with_truth(F, H, Q, R, x0, P0, n_steps, rng)
    kf, _, _ = _run_filter(F, H, Q, R, x0, P0, measurements, true_states)

    average = float(np.mean(nis(kf.innovations, kf.innovation_covariances)))
    standard_error = np.sqrt(2.0 * m / n_steps)
    assert abs(average - m) < 4.0 * standard_error


def test_nis_centers_on_m_vector() -> None:
    """Average NIS is near m=2 under a correct vector-measurement model (Section 7.3).

    The same chi-squared argument holds for vector measurements, with m
    the measurement dimension. The standard error is sqrt(2m/n_steps).
    """
    F, H, Q, R = _two_d_constant_velocity_model()
    rng = np.random.default_rng(11)
    n_steps = 4000
    m = H.shape[0]
    x0 = np.array([0.0, 1.0, 0.0, -0.5])
    P0 = np.eye(4)
    true_states, measurements = _simulate_with_truth(F, H, Q, R, x0, P0, n_steps, rng)
    kf, _, _ = _run_filter(F, H, Q, R, x0, P0, measurements, true_states)

    average = float(np.mean(nis(kf.innovations, kf.innovation_covariances)))
    standard_error = np.sqrt(2.0 * m / n_steps)
    assert abs(average - m) < 4.0 * standard_error


def test_nis_shape_and_non_negativity() -> None:
    """nis returns shape (K,) and every value is non-negative (Section 7.3).

    Each NIS value is the quadratic form nu^T S^-1 nu with S positive
    definite, so it cannot be negative.
    """
    F, H, Q, R = _two_d_constant_velocity_model()
    rng = np.random.default_rng(12)
    n_steps = 500
    x0 = np.array([0.0, 1.0, 0.0, -0.5])
    P0 = np.eye(4)
    true_states, measurements = _simulate_with_truth(F, H, Q, R, x0, P0, n_steps, rng)
    kf, _, _ = _run_filter(F, H, Q, R, x0, P0, measurements, true_states)

    values = nis(kf.innovations, kf.innovation_covariances)
    assert values.shape == (n_steps,)
    assert np.all(values >= 0.0)


def test_nis_matches_independent_quadratic_form() -> None:
    """A hand-built case matches nu^T S^-1 nu computed independently (eq 7.5)."""
    nu = np.array([3.0, -1.0])
    S = np.array([[2.0, 0.5], [0.5, 1.0]])
    expected = float(nu @ np.linalg.solve(S, nu))

    value = nis([nu], [S])
    assert value.shape == (1,)
    np.testing.assert_allclose(value[0], expected)


# Section 7.3 counterpart: NEES against the state dimension


def test_nees_centers_on_n_and_is_distinct_from_nis() -> None:
    """Average NEES is near n while average NIS is near m, in one run (Section 7.3).

    NEES uses the state dimension n, NIS the measurement dimension m.
    With n=2 and m=1 the two targets differ, so checking both in the
    same simulation confirms the functions are not conflated. Each
    average has standard error sqrt(2d/n_steps) for its own dimension d.
    """
    F, H, Q, R = _constant_velocity_model()
    rng = np.random.default_rng(13)
    n_steps = 4000
    n = F.shape[0]
    m = H.shape[0]
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2)
    true_states, measurements = _simulate_with_truth(F, H, Q, R, x0, P0, n_steps, rng)
    kf, errors, posteriors = _run_filter(F, H, Q, R, x0, P0, measurements, true_states)

    average_nees = float(np.mean(nees(errors, posteriors)))
    average_nis_value = float(np.mean(nis(kf.innovations, kf.innovation_covariances)))

    nees_standard_error = np.sqrt(2.0 * n / n_steps)
    nis_standard_error = np.sqrt(2.0 * m / n_steps)
    assert abs(average_nees - n) < 4.0 * nees_standard_error
    assert abs(average_nis_value - m) < 4.0 * nis_standard_error


# Section 7.2: zero mean


def test_innovation_mean_is_zero_with_covariances() -> None:
    """Mean innovation is within its standard error of zero (Section 7.2).

    The innovation is zero-mean under a correct model. With the
    innovation covariances supplied, innovation_mean reports the
    standard error from the time-averaged S. The sample mean must fall
    within four of those standard errors of zero, per component.
    """
    F, H, Q, R = _two_d_constant_velocity_model()
    rng = np.random.default_rng(14)
    n_steps = 4000
    x0 = np.array([0.0, 1.0, 0.0, -0.5])
    P0 = np.eye(4)
    true_states, measurements = _simulate_with_truth(F, H, Q, R, x0, P0, n_steps, rng)
    kf, _, _ = _run_filter(F, H, Q, R, x0, P0, measurements, true_states)

    result = innovation_mean(kf.innovations, kf.innovation_covariances)
    assert result.mean.shape == (2,)
    assert np.all(result.standard_error > 0.0)
    assert np.all(np.abs(result.mean) < 4.0 * result.standard_error)


def test_innovation_mean_standard_error_without_covariances() -> None:
    """Without covariances the standard error comes from the sample variance.

    The optional innovation_covariances argument is omitted. The
    function then estimates the per-component standard error from the
    sample variance of the innovations. The mean stays within four such
    standard errors of zero.
    """
    F, H, Q, R = _constant_velocity_model()
    rng = np.random.default_rng(15)
    n_steps = 4000
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2)
    true_states, measurements = _simulate_with_truth(F, H, Q, R, x0, P0, n_steps, rng)
    kf, _, _ = _run_filter(F, H, Q, R, x0, P0, measurements, true_states)

    result = innovation_mean(kf.innovations)
    assert np.all(result.standard_error > 0.0)
    assert np.all(np.abs(result.mean) < 4.0 * result.standard_error)


# Section 7.1: whiteness


def test_innovation_autocorrelation_near_zero_scalar() -> None:
    """Lag-1 autocorrelation is near zero under a correct scalar model (Section 7.1).

    Under whiteness the lag-1 sample autocorrelation of the whitened
    innovations has standard deviation of order 1/sqrt(K). A three-sigma
    band, 3/sqrt(K), is the tolerance.
    """
    F, H, Q, R = _constant_velocity_model()
    rng = np.random.default_rng(16)
    n_steps = 4000
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2)
    true_states, measurements = _simulate_with_truth(F, H, Q, R, x0, P0, n_steps, rng)
    kf, _, _ = _run_filter(F, H, Q, R, x0, P0, measurements, true_states)

    rho = innovation_autocorrelation(kf.innovations, kf.innovation_covariances, lag=1)
    assert abs(rho) < 3.0 / np.sqrt(n_steps)


def test_innovation_autocorrelation_near_zero_vector() -> None:
    """Lag-1 autocorrelation is near zero under a correct vector model (Section 7.1).

    The whitening transform uses the Cholesky factor of each S, so the
    vector case reduces to a sequence with identity covariance. The same
    3/sqrt(K) band applies.
    """
    F, H, Q, R = _two_d_constant_velocity_model()
    rng = np.random.default_rng(17)
    n_steps = 4000
    x0 = np.array([0.0, 1.0, 0.0, -0.5])
    P0 = np.eye(4)
    true_states, measurements = _simulate_with_truth(F, H, Q, R, x0, P0, n_steps, rng)
    kf, _, _ = _run_filter(F, H, Q, R, x0, P0, measurements, true_states)

    rho = innovation_autocorrelation(kf.innovations, kf.innovation_covariances, lag=1)
    assert abs(rho) < 3.0 / np.sqrt(n_steps)


def test_innovation_autocorrelation_detects_correlation() -> None:
    """A correlated sequence produces a detectably nonzero autocorrelation (Section 7.1).

    An AR(1) sequence with coefficient phi has theoretical lag-1
    autocorrelation phi. With identity innovation covariances the
    whitening is the identity, so the statistic recovers that
    correlation, far outside the whiteness band of a correct model.
    """
    rng = np.random.default_rng(18)
    n_steps = 2000
    phi = 0.6
    series = np.zeros(n_steps)
    noise = rng.standard_normal(n_steps)
    for k in range(1, n_steps):
        series[k] = phi * series[k - 1] + noise[k]

    innovations = series.reshape(-1, 1)
    covariances = np.ones((n_steps, 1, 1))
    rho = innovation_autocorrelation(innovations, covariances, lag=1)

    # The whiteness band is 2/sqrt(K). The AR(1) correlation phi=0.6 sits
    # far above it; a loose floor of 0.4 confirms detection without
    # depending on the exact estimate.
    assert rho > 0.4
    assert rho > 2.0 / np.sqrt(n_steps)


# Section 7.3: the average_nis consistency report


def test_average_nis_consistency_statistics() -> None:
    """average_nis reports m, the standard error, the interval, and the inside rate.

    Under a correct model the expected value equals m, the standard
    error equals sqrt(2m/K), the per-sample interval equals the tabulated
    chi-squared 95 percent interval, and the inside fraction is near 0.95.
    The inside fraction is a binomial proportion with target 0.95 and
    standard error sqrt(0.95 * 0.05 / K); the tolerance is five of those.
    """
    F, H, Q, R = _constant_velocity_model()
    rng = np.random.default_rng(19)
    n_steps = 4000
    m = H.shape[0]
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2)
    true_states, measurements = _simulate_with_truth(F, H, Q, R, x0, P0, n_steps, rng)
    kf, _, _ = _run_filter(F, H, Q, R, x0, P0, measurements, true_states)

    result = average_nis(kf.innovations, kf.innovation_covariances)
    assert result.expected == m
    assert result.standard_error == pytest.approx(np.sqrt(2.0 * m / n_steps))
    assert result.per_sample_interval == CHI2_95_INTERVAL[m]

    binomial_se = np.sqrt(0.95 * 0.05 / n_steps)
    assert result.fraction_inside is not None
    assert abs(result.fraction_inside - 0.95) < 5.0 * binomial_se


def test_average_nis_untabulated_dimension_returns_none() -> None:
    """For m outside CHI2_95_INTERVAL the interval and inside fraction are None.

    The expected value and standard error are still reported, since they
    do not require a tabulated quantile.
    """
    rng = np.random.default_rng(20)
    n_steps = 300
    m = 7  # not tabulated in CHI2_95_INTERVAL
    assert m not in CHI2_95_INTERVAL

    innovations = rng.standard_normal((n_steps, m))
    covariances = np.broadcast_to(np.eye(m), (n_steps, m, m))
    result = average_nis(innovations, covariances)

    assert result.expected == m
    assert result.standard_error == pytest.approx(np.sqrt(2.0 * m / n_steps))
    assert result.per_sample_interval is None
    assert result.fraction_inside is None


# Section 7.4: the divergence verdict


def test_detect_divergence_false_under_correct_model() -> None:
    """detect_divergence returns False on a correctly specified filter (Section 7.4).

    The average NIS sits near m and few samples breach the bound, so
    neither divergence signal fires.
    """
    F, H, Q, R = _constant_velocity_model()
    rng = np.random.default_rng(21)
    n_steps = 2000
    m = H.shape[0]
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2)
    true_states, measurements = _simulate_with_truth(F, H, Q, R, x0, P0, n_steps, rng)
    kf, _, _ = _run_filter(F, H, Q, R, x0, P0, measurements, true_states)

    verdict = detect_divergence(kf.innovations, kf.innovation_covariances)
    assert verdict.diverged is False
    assert abs(verdict.average_nis - m) < 0.5


def test_detect_divergence_true_under_misspecification() -> None:
    """detect_divergence returns True for an overconfident filter (Section 7.4).

    The truth accelerates while the filter assumes constant velocity with
    a process noise covariance far too small. The filter becomes
    overconfident, the innovations grow while S stays small, and the NIS
    climbs far above m. Both divergence signals fire, and the supporting
    statistics are populated.
    """
    dt = 1.0
    n_steps = 30
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 1e-7 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R = np.array([[1.0]])
    x0 = np.array([0.0, 0.0])
    P0 = np.eye(2)
    m = H.shape[0]

    rng = np.random.default_rng(22)
    true_acceleration = 0.5
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    for k in range(1, n_steps + 1):
        true_position = 0.5 * true_acceleration * (dt * k) ** 2
        z = np.array([true_position + rng.normal(0.0, 1.0)])
        kf.predict()
        kf.update(z)

    verdict = detect_divergence(kf.innovations, kf.innovation_covariances)
    assert verdict.diverged is True
    assert verdict.expected == m
    assert verdict.average_nis > 10.0 * m
    assert verdict.breach_fraction > 0.5
    assert verdict.upper_bound == CHI2_95_INTERVAL[m][1]
    assert isinstance(verdict.criterion, str)
    assert verdict.criterion


# Input flexibility and error handling


def test_input_forms_agree() -> None:
    """List, 2D array, and scalar-sequence inputs produce the same NIS.

    The documented input forms are interchangeable. A list of (m,)
    arrays, a 2D array, and a sequence of scalars all describe the same
    scalar-measurement sequence and must yield identical NIS values.
    """
    nu_list = [np.array([0.5]), np.array([-0.3]), np.array([0.8])]
    S_list = [np.array([[2.0]]), np.array([[1.5]]), np.array([[1.0]])]
    nu_2d = np.array([[0.5], [-0.3], [0.8]])
    S_3d = np.array([[[2.0]], [[1.5]], [[1.0]]])

    from_list = nis(nu_list, S_list)
    from_arrays = nis(nu_2d, S_3d)
    from_scalars = nis([0.5, -0.3, 0.8], [2.0, 1.5, 1.0])

    np.testing.assert_allclose(from_list, from_arrays)
    np.testing.assert_allclose(from_scalars, from_arrays)


def test_shape_mismatch_raises() -> None:
    """A covariance count that disagrees with the innovations raises ValueError."""
    innovations = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
    covariances = [np.array([[1.0]]), np.array([[1.0]])]  # one short
    with pytest.raises(ValueError, match="shape"):
        nis(innovations, covariances)


def test_autocorrelation_invalid_lag_raises() -> None:
    """A non-positive lag and a lag at or beyond the sample count raise ValueError."""
    innovations = np.array([[0.1], [0.2], [0.3]])
    covariances = np.ones((3, 1, 1))
    with pytest.raises(ValueError, match="positive integer"):
        innovation_autocorrelation(innovations, covariances, lag=0)
    with pytest.raises(ValueError, match="smaller than the number of samples"):
        innovation_autocorrelation(innovations, covariances, lag=3)
