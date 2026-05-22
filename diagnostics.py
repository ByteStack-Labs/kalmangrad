"""Diagnostics for the Kalman filter innovation sequence.

This module implements the diagnostics established in Section 7 of
docs/derivation.md. Section 7 derives three statistical properties of
the innovation sequence that hold under a correctly specified filter:
zero mean (Section 7.2), known covariance (Section 7.3), and whiteness
(Section 7.1). The functions below compute the corresponding empirical
statistics from a recorded innovation sequence and decide whether the
sequence is consistent with those properties.

Two classes of diagnostic are distinguished. The first uses only the
filter's own outputs, the innovations and the innovation covariances,
and is therefore usable in production where the true state is unknown.
NIS (Section 7.3, equation 7.5), the innovation mean, the innovation
autocorrelation, the time averaged NIS, and the divergence verdict are
production diagnostics. The second class requires the ground truth
state and is usable only in simulation and validation. NEES is the only
such function in this module and is documented accordingly.

NumPy is the only dependency. Chi squared critical values for the NIS
test are tabulated for a small set of measurement dimensions so the
module remains free of SciPy. Values come from the standard chi squared
CDF, rounded to six decimal places.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

# Two sided 95 percent intervals of the chi squared distribution for
# small degrees of freedom. The first entry of each tuple is the 0.025
# quantile and the second is the 0.975 quantile. Under a correctly
# specified filter, a single NIS sample with m degrees of freedom falls
# inside this interval with probability 0.95 (Section 7.3, equation
# 7.5). The constants are tabulated rather than computed so the module
# remains free of SciPy.
CHI2_95_INTERVAL: dict[int, tuple[float, float]] = {
    1: (0.000982, 5.023886),
    2: (0.050636, 7.377759),
    3: (0.215795, 9.348404),
    4: (0.484419, 11.143287),
    5: (0.831212, 12.832502),
    6: (1.237344, 14.449375),
}


def _stack_vectors(vectors: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Coerce a sequence of vectors into a 2D array of shape (K, m).

    Accepts a list of 1D arrays of shape (m,), a 2D array of shape
    (K, m), or a sequence of scalars. Scalar inputs are promoted to a
    single column so the m equal to one case and the general vector
    case share one code path.
    """
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"vectors must be 1D or 2D, got ndim={arr.ndim}")
    return arr


def _stack_matrices(
    matrices: Sequence[np.ndarray] | np.ndarray,
    K: int,
    m: int,
    name: str,
) -> np.ndarray:
    """Coerce a sequence of (m, m) matrices into a 3D array (K, m, m)."""
    arr = np.asarray(matrices, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1, 1)
    if arr.shape != (K, m, m):
        raise ValueError(f"{name} must have shape ({K}, {m}, {m}), got {arr.shape}")
    return arr


def nis(
    innovations: Sequence[np.ndarray] | np.ndarray,
    innovation_covariances: Sequence[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Per step Normalized Innovation Squared (Section 7.3, eq 7.5).

    For each step k computes the scalar:

        epsilon_k = nu_k^T S_k^(-1) nu_k

    Under a correctly specified filter epsilon_k follows a chi squared
    distribution with m degrees of freedom, where m is the measurement
    dimension. The quadratic form is evaluated through np.linalg.solve,
    so the explicit inverse of S is avoided.

    This statistic uses only the filter's own outputs and is therefore
    usable in production.

    Args:
        innovations: Sequence of innovation vectors. A list of (m,)
            arrays, a 2D array of shape (K, m), or a sequence of scalars
            for the m equal to one case.
        innovation_covariances: Sequence of innovation covariance
            matrices. A list of (m, m) arrays, a 3D array of shape
            (K, m, m), or a sequence of scalars for the scalar case.

    Returns:
        Array of shape (K,) holding the per step NIS values.
    """
    nu = _stack_vectors(innovations)
    K, m = nu.shape
    S = _stack_matrices(innovation_covariances, K, m, "innovation_covariances")

    out = np.empty(K)
    for k in range(K):
        # Equation (7.5). np.linalg.solve evaluates S^(-1) nu without
        # forming the inverse, which is the numerically sound form.
        out[k] = float(nu[k] @ np.linalg.solve(S[k], nu[k]))
    return out


def nees(
    estimation_errors: Sequence[np.ndarray] | np.ndarray,
    covariances: Sequence[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Per step Normalized Estimation Error Squared.

    For each step k computes the scalar:

        eta_k = e_k^T P_k^(-1) e_k

    where e_k is the estimation error (true state minus estimate) and
    P_k is the posterior state covariance. Under a correctly specified
    filter eta_k follows a chi squared distribution with n degrees of
    freedom, where n is the state dimension. NEES is the ground truth
    aware counterpart of NIS.

    NEES requires the true state and is therefore a simulation and
    validation tool only. It is not usable in production, where the
    true state is by definition unknown. The production analogue is
    NIS, which depends only on the filter's own outputs.

    Args:
        estimation_errors: Sequence of error vectors e_k. A list of
            (n,) arrays or a 2D array of shape (K, n).
        covariances: Sequence of posterior state covariance matrices
            P_k. A list of (n, n) arrays or a 3D array of shape
            (K, n, n).

    Returns:
        Array of shape (K,) holding the per step NEES values.
    """
    e = _stack_vectors(estimation_errors)
    K, n = e.shape
    P = _stack_matrices(covariances, K, n, "covariances")

    out = np.empty(K)
    for k in range(K):
        out[k] = float(e[k] @ np.linalg.solve(P[k], e[k]))
    return out


class InnovationMean(NamedTuple):
    """Sample mean of the innovation sequence with its standard error.

    Attributes:
        mean: Sample mean vector, shape (m,).
        standard_error: Standard error of the mean per component, shape
            (m,). Computed from the time averaged innovation covariance
            when one is supplied, otherwise from the sample variance of
            the innovations.
    """

    mean: np.ndarray
    standard_error: np.ndarray


def innovation_mean(
    innovations: Sequence[np.ndarray] | np.ndarray,
    innovation_covariances: Sequence[np.ndarray] | np.ndarray | None = None,
) -> InnovationMean:
    """Time averaged innovation (Section 7.2, equation 7.2).

    Under a correctly specified filter the innovation sequence has
    zero mean. The sample mean should be close to zero within a
    sampling spread determined by the innovation covariance and the
    number of samples. A persistent nonzero mean indicates structural
    bias: a wrong dynamics matrix F, a wrong measurement matrix H, or
    unmodeled deterministic forcing. Section 7.2 states explicitly that
    tuning the noise covariances does not correct such a bias.

    This statistic uses only the filter's own outputs and is therefore
    usable in production.

    Args:
        innovations: Sequence of innovation vectors of shape (m,), a 2D
            array of shape (K, m), or a sequence of scalars.
        innovation_covariances: Optional sequence of innovation
            covariances S_k. When supplied, the standard error of the
            mean is computed from the time average of S, which is the
            filter's own estimate of the per sample variance. Otherwise
            the standard error is computed from the sample variance of
            the innovations.

    Returns:
        InnovationMean with the sample mean vector and the standard
        error of the mean per component.
    """
    nu = _stack_vectors(innovations)
    K, m = nu.shape
    sample_mean = nu.mean(axis=0)

    if innovation_covariances is not None:
        S = _stack_matrices(innovation_covariances, K, m, "innovation_covariances")
        # Section 7.3, equation 7.4: under a correct model the
        # innovation covariance is S_k. The marginal variance per
        # component is the diagonal of S, averaged over time.
        component_variance = np.diag(S.mean(axis=0))
    elif K > 1:
        component_variance = nu.var(axis=0, ddof=1)
    else:
        component_variance = np.zeros(m)

    standard_error = np.sqrt(component_variance / K)
    return InnovationMean(mean=sample_mean, standard_error=standard_error)


def innovation_autocorrelation(
    innovations: Sequence[np.ndarray] | np.ndarray,
    innovation_covariances: Sequence[np.ndarray] | np.ndarray,
    lag: int = 1,
) -> float:
    """Lag k autocorrelation of the whitened innovation sequence.

    The whiteness property of Section 7.1 (equation 7.1) states that
    innovations at different times are uncorrelated. The innovation
    sequence is heteroscedastic in general, since S_k varies with k.
    Each innovation is therefore whitened by the Cholesky factor of
    its covariance:

        w_k = L_k^(-1) nu_k,   L_k L_k^T = S_k

    Under a correctly specified filter the sequence w_k has identity
    covariance: E[w_k w_k^T] = L_k^(-1) S_k L_k^(-T) = I. The lag k
    sample autocorrelation of this sequence is:

        rho_lag = sum_{k > lag} w_k . w_{k-lag} / sum_k w_k . w_k

    which is approximately zero with sampling spread of order
    1 / sqrt(K). A value persistently outside this band at one or more
    lags indicates that whiteness is failing, which Section 7.4
    interprets as the filter not extracting all available information
    from the measurements.

    This statistic uses only the filter's own outputs and is therefore
    usable in production.

    Args:
        innovations: Sequence of innovation vectors of shape (m,), a 2D
            array of shape (K, m), or a sequence of scalars.
        innovation_covariances: Sequence of innovation covariance
            matrices of shape (m, m), a 3D array of shape (K, m, m), or
            a sequence of scalars.
        lag: Positive integer lag at which to evaluate the
            autocorrelation. Defaults to 1.

    Returns:
        The scalar lag k autocorrelation of the whitened innovation
        sequence.
    """
    if lag < 1:
        raise ValueError(f"lag must be a positive integer, got {lag}")

    nu = _stack_vectors(innovations)
    K, m = nu.shape
    if lag >= K:
        raise ValueError(f"lag {lag} must be smaller than the number of samples {K}")
    S = _stack_matrices(innovation_covariances, K, m, "innovation_covariances")

    # Whitening transform. Solving L_k w_k = nu_k gives
    # w_k = L_k^(-1) nu_k. The whitened sequence has identity
    # covariance under a correct model.
    whitened = np.empty((K, m))
    for k in range(K):
        L = np.linalg.cholesky(S[k])
        whitened[k] = np.linalg.solve(L, nu[k])

    numerator = float(np.sum(whitened[lag:] * whitened[:-lag]))
    denominator = float(np.sum(whitened * whitened))
    return numerator / denominator


class AverageNIS(NamedTuple):
    """Time averaged NIS with expectation and consistency interval.

    Attributes:
        average: Sample mean of the per step NIS over the recorded run.
        expected: m, the measurement dimension and the chi squared
            mean per sample.
        standard_error: sqrt(2 m / K), the standard error of the sample
            mean of a chi squared(m) random variable computed from K
            independent samples.
        per_sample_interval: The two sided 95 percent interval of
            chi squared(m) for a single NIS sample, when m is tabulated
            in CHI2_95_INTERVAL. None when m is outside the tabulated
            range.
        fraction_inside: Fraction of per step NIS samples that lie
            inside per_sample_interval. Expected to be near 0.95 under
            a correct model. None when per_sample_interval is None.
    """

    average: float
    expected: int
    standard_error: float
    per_sample_interval: tuple[float, float] | None
    fraction_inside: float | None


def average_nis(
    innovations: Sequence[np.ndarray] | np.ndarray,
    innovation_covariances: Sequence[np.ndarray] | np.ndarray,
) -> AverageNIS:
    """Time averaged NIS with consistency statistics (Section 7.3).

    Under a correctly specified filter each NIS sample is chi
    squared(m) with mean m and variance 2m. The sample mean of K
    independent samples then has mean m and standard error
    sqrt(2 m / K). A time averaged NIS far from m indicates a
    disagreement between the empirical innovation covariance and the
    filter computed S (Section 7.3): a value persistently above m
    means the filter is underestimating its own uncertainty, a value
    persistently below m means it is overestimating it.

    This statistic uses only the filter's own outputs and is therefore
    usable in production.

    Args:
        innovations: Sequence of innovation vectors of shape (m,), a 2D
            array of shape (K, m), or a sequence of scalars.
        innovation_covariances: Sequence of innovation covariance
            matrices of shape (m, m), a 3D array of shape (K, m, m), or
            a sequence of scalars.

    Returns:
        AverageNIS with the time averaged NIS, the expectation m, the
        standard error of the mean, the per sample chi squared 95
        percent interval when m is tabulated, and the fraction of per
        step NIS samples inside that interval.
    """
    nu = _stack_vectors(innovations)
    K, m = nu.shape
    values = nis(innovations, innovation_covariances)

    average = float(values.mean())
    standard_error = float(np.sqrt(2.0 * m / K))

    interval = CHI2_95_INTERVAL.get(m)
    if interval is not None:
        lower, upper = interval
        fraction_inside: float | None = float(
            np.mean((values >= lower) & (values <= upper))
        )
    else:
        fraction_inside = None

    return AverageNIS(
        average=average,
        expected=m,
        standard_error=standard_error,
        per_sample_interval=interval,
        fraction_inside=fraction_inside,
    )


class DivergenceVerdict(NamedTuple):
    """Result of detect_divergence with supporting statistics.

    Attributes:
        diverged: True when the criterion described in the docstring of
            detect_divergence is met.
        average_nis: Time averaged NIS over the recorded run.
        expected: m, the measurement dimension.
        breach_fraction: Fraction of per step NIS samples that exceed
            the upper 0.975 quantile of chi squared(m).
        upper_bound: The per sample upper chi squared bound used to
            compute breach_fraction.
        criterion: One line description of the decision rule applied.
    """

    diverged: bool
    average_nis: float
    expected: int
    breach_fraction: float
    upper_bound: float
    criterion: str


def detect_divergence(
    innovations: Sequence[np.ndarray] | np.ndarray,
    innovation_covariances: Sequence[np.ndarray] | np.ndarray,
    nis_factor: float = 3.0,
    breach_fraction_threshold: float = 0.2,
) -> DivergenceVerdict:
    """Divergence verdict from the innovation sequence (Section 7).

    Section 7 establishes that the innovation sequence is self
    diagnosing. Under a correctly specified filter the NIS samples are
    chi squared(m) and their time average is close to m. A diverged
    filter typically shows the opposite pattern, with NIS samples
    growing far beyond their expectation (Section 7.4, the
    overconfident case where the filter trusts a wrong model and
    stops trusting measurements).

    The verdict combines two signals into a single decision:

        1. The time averaged NIS exceeds nis_factor times m. Default
           nis_factor is 3.0, which is well outside the chi squared(m)
           95 percent per sample interval for the tabulated dimensions.
        2. The fraction of per step NIS samples that exceed the upper
           0.975 quantile of chi squared(m) exceeds
           breach_fraction_threshold. Default 0.2 is eight times the
           expected breach rate of 0.025 under a correct model.

    The verdict is True when both signals fire. Requiring both signals
    is robust to the tail of a correctly specified run, where
    occasional excursions of NIS are expected. The verdict is
    intentionally one sided. The divergence mode Section 7 highlights
    is the overconfident case, in which NIS grows. A persistently
    small NIS is the symmetric mode, in which the filter overestimates
    its uncertainty, and is read directly from average_nis.

    When m lies outside CHI2_95_INTERVAL, the upper chi squared bound
    is taken from the Gaussian approximation m + 1.96 sqrt(2 m). The
    approximation is conservative for small m (it underestimates the
    true upper quantile, so more samples are counted as breaches) and
    improves with m. The tabulated bound is used whenever available.

    This verdict uses only the filter's own outputs and is therefore
    usable in production.

    Args:
        innovations: Sequence of innovation vectors of shape (m,), a 2D
            array of shape (K, m), or a sequence of scalars.
        innovation_covariances: Sequence of innovation covariance
            matrices of shape (m, m), a 3D array of shape (K, m, m), or
            a sequence of scalars.
        nis_factor: Multiplier of m that the time averaged NIS must
            exceed for signal 1 to fire. Default 3.0.
        breach_fraction_threshold: Threshold for the fraction of per
            step NIS samples that exceed the per sample upper chi
            squared bound for signal 2 to fire. Default 0.2.

    Returns:
        DivergenceVerdict with the boolean diverged flag, the
        underlying statistics, and a one line description of the
        decision rule applied.
    """
    nu = _stack_vectors(innovations)
    m = nu.shape[1]
    values = nis(innovations, innovation_covariances)
    average = float(values.mean())

    interval = CHI2_95_INTERVAL.get(m)
    if interval is not None:
        upper = interval[1]
    else:
        # Gaussian approximation to the upper 0.975 quantile of
        # chi squared(m). Mean m, variance 2m, so the bound is
        # m + 1.96 sqrt(2 m).
        upper = float(m + 1.96 * np.sqrt(2.0 * m))

    breach_fraction = float(np.mean(values > upper))

    diverged = bool(
        average > nis_factor * m and breach_fraction > breach_fraction_threshold
    )
    criterion = (
        f"average NIS > {nis_factor} * m (m = {m}) and "
        f"fraction of NIS samples > {upper:.4f} exceeds "
        f"{breach_fraction_threshold}"
    )
    return DivergenceVerdict(
        diverged=diverged,
        average_nis=average,
        expected=m,
        breach_fraction=breach_fraction,
        upper_bound=upper,
        criterion=criterion,
    )
