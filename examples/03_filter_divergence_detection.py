"""Example 3: detecting filter divergence from the innovation sequence.

This example demonstrates the central claim of Section 7: the innovation
sequence is self-diagnosing. It connects to the optimality conditions of
Section 6.3 and the innovation statistics of Section 7, in particular the
normalized innovation squared of Section 7.3.

The scenario is a deliberate model mismatch. The true target accelerates, but
the filter assumes constant velocity and is given a process noise covariance Q
that is far too small. The small Q makes the filter overconfident: its
covariance P collapses, the Kalman gain shrinks toward zero, and the filter
stops trusting measurements. The constant-velocity prediction cannot follow the
accelerating truth, so the estimate diverges.

The divergence is visible in the data the filter itself produces. The normalized
innovation squared, nu^T S^-1 nu, has expected value m = 1 under a correct
model and a chi-squared(1) distribution. As the filter diverges, the innovations
grow while S stays small, so the NIS climbs far beyond its expectation. The run
logs the NIS over time and shows it breaching the chi-squared bound and growing
without bound, which is the signature of a misspecified filter.
"""

from __future__ import annotations

import numpy as np

from _logging import setup_logger
from kalman_filter import KalmanFilter

logger = setup_logger("03_filter_divergence_detection")

# Critical value of the chi-squared(1) distribution at the 0.95 quantile. A
# single NIS sample exceeds this only 5 percent of the time under a correct
# model. This constant avoids a SciPy dependency. Section 7.3.
CHI2_1_DOF_95 = 3.841459


def main() -> None:
    """Run the filter divergence detection example."""
    rng = np.random.default_rng(2)

    dt = 1.0
    n_steps = 30
    measurement_std = 1.0
    true_acceleration = 0.5  # the truth maneuvers; the filter will not expect it

    # The filter's model: constant velocity with a process noise covariance that
    # is far too small for a maneuvering target. This is the misspecification.
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 1e-7 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R = np.array([[measurement_std**2]])

    x0 = np.array([0.0, 0.0])
    P0 = np.eye(2)
    kf = KalmanFilter(F, H, Q, R, x0, P0)

    m = H.shape[0]
    logger.info(
        "filter divergence detection: constant-velocity filter on an accelerating target"
    )
    logger.info(
        "true acceleration = %.2f, filter Q is deliberately tiny (overconfident)",
        true_acceleration,
    )
    logger.info("NIS expectation under a correct model is m = %d; 95%% bound is %.3f", m, CHI2_1_DOF_95)
    logger.info("%4s %12s %12s %12s %12s", "k", "true_pos", "estimate", "innovation", "NIS")

    nis_values = []
    breaches = 0
    for k in range(1, n_steps + 1):
        # Constant-acceleration ground truth, observed with noise.
        true_position = 0.5 * true_acceleration * (dt * k) ** 2
        measurement = true_position + rng.normal(0.0, measurement_std)

        kf.predict()
        kf.update(np.array([measurement]))

        nu = kf.innovations[-1]
        S = kf.innovation_covariances[-1]
        nis = float(nu @ np.linalg.solve(S, nu))  # Section 7.3, equation (7.5)
        nis_values.append(nis)
        if nis > CHI2_1_DOF_95:
            breaches += 1

        logger.info("%4d %12.4f %12.4f %12.4f %12.4f", k, true_position, kf.x[0], nu[0], nis)

    nis_values = np.array(nis_values)
    average_nis = float(np.mean(nis_values))
    breach_fraction = breaches / n_steps

    logger.info("average NIS over the run: %.4f (expected near %d under a correct model)", average_nis, m)
    logger.info(
        "fraction of steps with NIS above the 95%% bound: %.2f (expected near 0.05)",
        breach_fraction,
    )
    logger.info("final NIS: %.4f", nis_values[-1])
    diverged = average_nis > 10.0 * m and breach_fraction > 0.5
    logger.info(
        "divergence detected from the innovation sequence: %s",
        diverged,
    )


if __name__ == "__main__":
    main()
