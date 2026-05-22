"""Example 1: 1D position tracking, the hello world of the Kalman filter.

This example tracks an object moving along a line at roughly constant velocity,
observed through a noisy position-only sensor. It demonstrates the basic
predict and update cycle of the recursion. It connects to the prediction step
of Section 4, the update step of Section 5, and the complete algorithm of
Section 8.

The state is [position, velocity]. The dynamics are constant velocity, so the
position advances by velocity times the time step. The sensor reports position
only, which is the H = [1, 0] case named in Section 2.2. The filter recovers
both position and the unobserved velocity from the position measurements alone.

The run logs the true position, the measurement, and the filter estimate at each
step, then reports the root mean squared error of the measurements and of the
filter. The filter error is smaller: the filter smooths below the sensor noise,
which is the minimum mean squared error property of Section 6.
"""

from __future__ import annotations

import numpy as np

from _logging import setup_logger
from kalman_filter import KalmanFilter

logger = setup_logger("01_1d_position_tracking")


def main() -> None:
    """Run the 1D position tracking example."""
    rng = np.random.default_rng(0)

    dt = 1.0
    n_steps = 40
    true_velocity = 1.0
    measurement_std = 2.0

    # Constant-velocity model. State is [position, velocity]. Section 2.1, 2.2.
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 0.01 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R = np.array([[measurement_std**2]])

    # Wrong initial guess with a weak prior. The filter recovers regardless.
    x0 = np.array([0.0, 0.0])
    P0 = np.eye(2) * 100.0
    kf = KalmanFilter(F, H, Q, R, x0, P0)

    logger.info("1D position tracking: %d steps, measurement std = %.2f", n_steps, measurement_std)
    logger.info("%4s %12s %12s %12s", "k", "true_pos", "measurement", "estimate")

    true_positions = []
    measurements = []
    estimates = []
    for k in range(1, n_steps + 1):
        true_position = true_velocity * dt * k
        measurement = true_position + rng.normal(0.0, measurement_std)

        # The predict and update cycle of Section 8.
        kf.predict()
        kf.update(np.array([measurement]))

        estimate = kf.x[0]
        true_positions.append(true_position)
        measurements.append(measurement)
        estimates.append(estimate)
        logger.info("%4d %12.4f %12.4f %12.4f", k, true_position, measurement, estimate)

    true_positions = np.array(true_positions)
    measurements = np.array(measurements)
    estimates = np.array(estimates)

    measurement_rms = float(np.sqrt(np.mean((measurements - true_positions) ** 2)))
    filter_rms = float(np.sqrt(np.mean((estimates - true_positions) ** 2)))

    logger.info("final velocity estimate: %.4f (true %.4f)", kf.x[1], true_velocity)
    logger.info("measurement RMS error: %.4f", measurement_rms)
    logger.info("filter RMS error:      %.4f", filter_rms)
    logger.info(
        "the filter smooths below the sensor noise: %.4f < %.4f is %s",
        filter_rms,
        measurement_rms,
        filter_rms < measurement_rms,
    )


if __name__ == "__main__":
    main()
