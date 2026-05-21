"""Reference implementation of the discrete-time linear Kalman filter.

This module implements the recursion derived in docs/derivation.md. The
structure of the code mirrors the structure of the mathematics. The
KalmanFilter class holds the state estimate and its covariance and evolves
them through a prediction step and an update step. Each computational line
names the equation it implements.

The notation matches the derivation. The state estimate is x, its covariance
is P, the state transition matrix is F, the process noise covariance is Q, the
measurement matrix is H, the measurement noise covariance is R, the measurement
is z, the innovation is nu, the innovation covariance is S, and the Kalman gain
is K. Matrices keep their conventional uppercase names so the code reads as the
equations.

NumPy is the only dependency. The linear algebra is written out in full so that
a reader with the derivation open can follow each step.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Discrete-time linear Kalman filter.

    The filter estimates a hidden state from noisy linear measurements under
    the linear-Gaussian assumptions of derivation Section 2. It holds the
    current state estimate x and its covariance P and advances them with the
    recursion of Section 8: a prediction step (Section 4) followed by an update
    step (Section 5).

    The model matrices F, H, Q, and R are supplied at construction and used by
    default on every step. The predict and update methods accept optional
    per-step overrides for the time-varying case of Section 2.1, where each
    matrix carries a time subscript. The constant case, with no overrides, is
    the common path.

    Attributes:
        F: State transition matrix, shape (n, n). Equation (2.1).
        H: Measurement matrix, shape (m, n). Equation (2.2).
        Q: Process noise covariance, shape (n, n). Equation (2.3).
        R: Measurement noise covariance, shape (m, m). Equation (2.4).
        x: Current state estimate, shape (n,).
        P: Current state covariance, shape (n, n).
        innovations: Recorded innovations nu, one per update. Section 7.
        innovation_covariances: Recorded innovation covariances S, one per
            update. Section 7.
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> None:
        """Initialize the filter from the model matrices and the initial state.

        Stores the constant model matrices and sets the current estimate and
        covariance from the initial values x0 and P0. This is the
        initialization of Section 8.1, equation (8.1).

        Args:
            F: State transition matrix, shape (n, n).
            H: Measurement matrix, shape (m, n).
            Q: Process noise covariance, shape (n, n).
            R: Measurement noise covariance, shape (m, m).
            x0: Initial state estimate, shape (n,).
            P0: Initial state covariance, shape (n, n).
        """
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        # Current estimate and covariance. Initialized from x0 and P0, then
        # evolved by predict and update. Equation (8.1).
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)

        # Innovation sequence, recorded for the diagnostics of Section 7.
        self.innovations: list[np.ndarray] = []
        self.innovation_covariances: list[np.ndarray] = []

    def predict(self, F: np.ndarray | None = None, Q: np.ndarray | None = None) -> None:
        """Propagate the state estimate and covariance through the dynamics.

        This is the prediction step of Section 4. It advances the estimate one
        time step using the dynamics alone, with no measurement. The covariance
        always grows, reflecting the uncertainty added by the process noise.

        The stored F and Q are used by default. Passing F or Q overrides the
        stored matrix for this step, supporting the time-varying model of
        Section 2.1. The state estimate x and covariance P are updated in place.

        Args:
            F: Optional state transition matrix for this step, shape (n, n).
                Defaults to the stored F.
            Q: Optional process noise covariance for this step, shape (n, n).
                Defaults to the stored Q.
        """
        F = self.F if F is None else np.asarray(F, dtype=float)
        Q = self.Q if Q is None else np.asarray(Q, dtype=float)

        # Equation (4.5): propagate the mean through the dynamics.
        self.x = F @ self.x

        # Equation (4.9): propagate the covariance and add the process noise.
        self.P = F @ self.P @ F.T + Q

    def update(
        self,
        z: np.ndarray,
        H: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ) -> None:
        """Incorporate a measurement into the state estimate and covariance.

        This is the update step of Section 5. It forms the innovation, the
        difference between the measurement and its prediction, weights it by
        the Kalman gain, and corrects the estimate. The covariance is reduced
        by the information the measurement contributes.

        The stored H and R are used by default. Passing H or R overrides the
        stored matrix for this step, supporting the time-varying model of
        Section 2.1. The state estimate x and covariance P are updated in place.

        The innovation nu and its covariance S are recorded for the innovation
        sequence diagnostics of Section 7.

        Args:
            z: Measurement, shape (m,).
            H: Optional measurement matrix for this step, shape (m, n).
                Defaults to the stored H.
            R: Optional measurement noise covariance for this step, shape
                (m, m). Defaults to the stored R.
        """
        z = np.asarray(z, dtype=float)
        H = self.H if H is None else np.asarray(H, dtype=float)
        R = self.R if R is None else np.asarray(R, dtype=float)

        # Equation (5.1): the innovation is the measurement minus its prediction.
        nu = z - H @ self.x

        # Equation (5.6): the innovation covariance.
        S = H @ self.P @ H.T + R

        # Equation (5.14): the Kalman gain, K = P H^T S^-1. The explicit inverse
        # is used so the code reads as the equation. S is a small (m by m)
        # matrix, so forming the inverse directly is both clear and inexpensive.
        K = self.P @ H.T @ np.linalg.inv(S)

        # Equation (5.7): correct the estimate with the weighted innovation.
        self.x = self.x + K @ nu

        # Equation (5.17): update the covariance in Joseph form. Section 5.4
        # explains why this form is used in place of the algebraically
        # equivalent standard form (5.15). Both terms below are symmetric
        # positive-semi-definite products, so their sum stays symmetric and
        # positive-definite under finite-precision arithmetic.
        I = np.eye(self.P.shape[0])
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Section 8.4: record(nu, S) for the diagnostics of Section 7.
        self.innovations.append(nu)
        self.innovation_covariances.append(S)
