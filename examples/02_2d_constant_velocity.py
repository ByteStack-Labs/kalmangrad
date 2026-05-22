"""Example 2: 2D constant-velocity tracking in a plane.

This example tracks an object moving across a plane at constant velocity,
observed through a noisy position-only sensor that reports both coordinates. It
demonstrates the filter on a multi-dimensional state. It connects to the general
matrix form of the state-space model in Section 2 and the prediction and update
steps of Sections 4 and 5, which are written for arbitrary dimension.

The state is [x, y, vx, vy], a four-dimensional vector. The dynamics are
constant velocity on each axis. The sensor observes the two positions, so the
measurement matrix H selects x and y from the state. The same recursion that
tracked a scalar position in Example 1 now tracks position and velocity in two
dimensions, with no change to the filter. Only the matrices grow.

The run logs the true position, the filter estimate, and the position error at a
sample of steps, then reports the final root mean squared position error of the
measurements and of the filter. The filter error is smaller in both coordinates.
"""

from __future__ import annotations

import numpy as np

from _logging import setup_logger
from kalman_filter import KalmanFilter

logger = setup_logger("02_2d_constant_velocity")


def main() -> None:
    """Run the 2D constant velocity example."""
    rng = np.random.default_rng(1)

    dt = 1.0
    n_steps = 40
    true_velocity = np.array([1.5, -0.8])  # vx, vy
    start_position = np.array([0.0, 0.0])
    measurement_std = 3.0

    # Constant-velocity dynamics on each axis. State is [x, y, vx, vy].
    F = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # The sensor observes the two positions only. Section 2.2.
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    # Process noise. Each axis gets the constant-velocity block coupling its
    # position and velocity. The state orders both positions before both
    # velocities, so the x block sits at indices (0, 2) and the y block at
    # indices (1, 3).
    cv_block = 0.01 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = cv_block
    Q[np.ix_([1, 3], [1, 3])] = cv_block

    R = (measurement_std**2) * np.eye(2)

    # Weak prior around a wrong initial guess.
    x0 = np.zeros(4)
    P0 = np.eye(4) * 100.0
    kf = KalmanFilter(F, H, Q, R, x0, P0)

    logger.info(
        "2D constant-velocity tracking: %d steps, measurement std = %.2f per axis",
        n_steps,
        measurement_std,
    )
    logger.info(
        "%4s %10s %10s %10s %10s %10s",
        "k",
        "true_x",
        "true_y",
        "est_x",
        "est_y",
        "pos_err",
    )

    true_positions = []
    measurements = []
    estimates = []
    for k in range(1, n_steps + 1):
        true_position = start_position + true_velocity * dt * k
        measurement = true_position + rng.normal(0.0, measurement_std, size=2)

        kf.predict()
        kf.update(measurement)

        estimate = kf.x[:2]
        true_positions.append(true_position)
        measurements.append(measurement)
        estimates.append(estimate)

        if k % 5 == 0 or k == 1:
            position_error = float(np.linalg.norm(estimate - true_position))
            logger.info(
                "%4d %10.3f %10.3f %10.3f %10.3f %10.3f",
                k,
                true_position[0],
                true_position[1],
                estimate[0],
                estimate[1],
                position_error,
            )

    true_positions = np.array(true_positions)
    measurements = np.array(measurements)
    estimates = np.array(estimates)

    measurement_rms = float(np.sqrt(np.mean(np.sum((measurements - true_positions) ** 2, axis=1))))
    filter_rms = float(np.sqrt(np.mean(np.sum((estimates - true_positions) ** 2, axis=1))))

    logger.info(
        "final velocity estimate: (%.4f, %.4f), true (%.4f, %.4f)",
        kf.x[2],
        kf.x[3],
        true_velocity[0],
        true_velocity[1],
    )
    logger.info("measurement RMS position error: %.4f", measurement_rms)
    logger.info("filter RMS position error:      %.4f", filter_rms)
    logger.info(
        "the filter smooths below the sensor noise: %.4f < %.4f is %s",
        filter_rms,
        measurement_rms,
        filter_rms < measurement_rms,
    )


if __name__ == "__main__":
    main()
