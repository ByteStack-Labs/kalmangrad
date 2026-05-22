"""Example 4: diagnostic instrumentation on a correctly specified filter.

This example shows what a correctly operating filter looks like under the
innovation diagnostics of Section 7. It is the reference baseline against which
the divergence of Example 3 is read. It connects to Section 7 in full: the
whiteness of Section 7.1, the zero mean of Section 7.2, and the known covariance
and normalized innovation squared of Section 7.3.

The data is generated to match the filter's model exactly. The true initial
state is drawn from the prior N(x0, P0), the process noise from N(0, Q), and the
measurement noise from N(0, R). The filter is therefore correctly specified, and
the innovation sequence must satisfy its three properties.

The run collects the innovation sequence the filter records and reports three
diagnostics. The time-averaged NIS should be near m, the measurement dimension,
since each NIS sample is chi-squared(m) with mean m. The mean innovation should
be near zero. The lag-one autocorrelation of the normalized innovations should
be near zero, the signature of whiteness. Together these confirm the filter is
operating within its assumptions. This is the meaning of monitoring an estimator
in production: the filter reports on its own health.
"""

from __future__ import annotations

import numpy as np

from _logging import setup_logger
from kalman_filter import KalmanFilter

logger = setup_logger("04_diagnostic_instrumentation")

# Two-sided 95 percent interval of the chi-squared(1) distribution, the 0.025 and
# 0.975 quantiles. About 95 percent of single NIS samples fall inside it under a
# correct model. These constants avoid a SciPy dependency. Section 7.3.
CHI2_1_DOF_LOWER = 0.000982
CHI2_1_DOF_UPPER = 5.023886


def simulate_model_consistent(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0_mean: np.ndarray,
    P0: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate measurements that match the filter's model and prior exactly.

    The true initial state is drawn from N(x0_mean, P0). Process noise is drawn
    from N(0, Q) and measurement noise from N(0, R). A filter initialized at
    (x0_mean, P0) is then correctly specified from the first step.
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


def main() -> None:
    """Run the diagnostic instrumentation reference example."""
    rng = np.random.default_rng(3)

    dt = 1.0
    n_steps = 2000
    measurement_std = 1.0

    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 0.02 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R = np.array([[measurement_std**2]])

    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2)
    m = H.shape[0]

    measurements = simulate_model_consistent(F, H, Q, R, x0, P0, n_steps, rng)

    kf = KalmanFilter(F, H, Q, R, x0, P0)
    for z in measurements:
        kf.predict()
        kf.update(z)

    logger.info("diagnostic instrumentation on a correctly specified filter")
    logger.info("steps = %d, measurement dimension m = %d", n_steps, m)

    # Normalized innovation squared, one sample per step. Section 7.3, eq (7.5).
    nis_values = np.array(
        [
            float(nu @ np.linalg.solve(S, nu))
            for nu, S in zip(kf.innovations, kf.innovation_covariances)
        ]
    )
    average_nis = float(np.mean(nis_values))
    nis_standard_error = float(np.sqrt(2.0 * m / n_steps))
    inside = np.mean((nis_values >= CHI2_1_DOF_LOWER) & (nis_values <= CHI2_1_DOF_UPPER))

    logger.info("--- NIS consistency (Section 7.3) ---")
    logger.info("average NIS: %.4f (expected %d, standard error %.4f)", average_nis, m, nis_standard_error)
    logger.info(
        "fraction of NIS samples inside the chi-squared 95%% interval [%.4f, %.4f]: %.4f",
        CHI2_1_DOF_LOWER,
        CHI2_1_DOF_UPPER,
        float(inside),
    )

    # Zero-mean check. Section 7.2, equation (7.2).
    innovations = np.array(kf.innovations).reshape(-1)
    mean_innovation = float(np.mean(innovations))
    mean_variance = float(np.mean([S[0, 0] for S in kf.innovation_covariances]))
    innovation_standard_error = float(np.sqrt(mean_variance / n_steps))

    logger.info("--- zero mean (Section 7.2) ---")
    logger.info(
        "mean innovation: %.4f (standard error %.4f, expected near zero)",
        mean_innovation,
        innovation_standard_error,
    )

    # Whiteness check via the lag-one autocorrelation of the normalized
    # innovations. Section 7.1. Under whiteness this is near zero, with a
    # sampling spread of roughly 1 / sqrt(N).
    normalized = np.array(
        [
            float(nu[0] / np.sqrt(S[0, 0]))
            for nu, S in zip(kf.innovations, kf.innovation_covariances)
        ]
    )
    lag1_autocorrelation = float(
        np.sum(normalized[1:] * normalized[:-1]) / np.sum(normalized**2)
    )
    whiteness_bound = float(2.0 / np.sqrt(n_steps))

    logger.info("--- whiteness (Section 7.1) ---")
    logger.info(
        "lag-1 autocorrelation: %.4f (within +/- %.4f under whiteness)",
        lag1_autocorrelation,
        whiteness_bound,
    )

    consistent = (
        abs(average_nis - m) < 4.0 * nis_standard_error
        and abs(mean_innovation) < 4.0 * innovation_standard_error
        and abs(lag1_autocorrelation) < whiteness_bound
    )
    logger.info("--- verdict ---")
    logger.info("filter operating within its assumptions: %s", consistent)


if __name__ == "__main__":
    main()
