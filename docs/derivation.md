# Derivation of the Kalman Filter

This document derives the discrete-time linear Kalman filter from first
principles. The derivation establishes what the filter is, why it works,
and what it assumes. The reference implementation in
[`kalman_filter.py`](../kalman_filter.py) is a direct expression of the
mathematics here.

## 1. Problem Statement

### 1.1 State estimation as a problem class

Many systems of engineering interest share a common structure. An
underlying state evolves over time according to known dynamics. That
state is observed through measurements that are noisy, indirect, or
both. The state itself is hidden. What is available are the measurements
and a model of how the state and measurements relate.

The state estimation problem is to recover the best estimate of the
hidden state from the sequence of measurements observed so far. "Best"
must be defined precisely. The answer depends on what assumptions hold
about the system and the noise.

Examples that fit this structure:

- Tracking a moving object whose position and velocity must be inferred
  from noisy position measurements
- Estimating the orientation of a spacecraft from gyroscope and star
  tracker readings
- Recovering a signal from a noisy sensor whose dynamics are known
- Estimating the parameters of a slowly time-varying system from
  measurements of its outputs

In each case, the state is what we want to know. The measurements are
what we have. The dynamics and measurement model are what we assume.

### 1.2 The discrete-time linear-Gaussian setting

The Kalman filter solves this problem under three specific assumptions.

**Time is discrete.** The state evolves in steps: $k = 0, 1, 2, \ldots$
Measurements arrive at the same discrete times. Continuous-time
formulations exist. This document does not cover them.

**The dynamics and measurement model are linear.** The state at time
$k$ is a linear function of the state at time $k-1$ plus noise. The
measurement at time $k$ is a linear function of the state at time $k$
plus noise. Nonlinear extensions exist, including the Extended Kalman
Filter and the Unscented Kalman Filter. This document does not cover
them.

**The noise is Gaussian.** Both the process noise and the measurement
noise are Gaussian with known covariance. The process noise represents
uncertainty in the dynamics. The measurement noise represents
uncertainty in observations.

Under these three assumptions, the optimal state estimator has a
closed-form recursive structure. That structure is the Kalman filter.

### 1.3 What we are solving and what we are assuming

The problem this document derives a solution to:

> Given a sequence of measurements $z_1, z_2, \ldots, z_k$ and a model
> of how the state evolves and how measurements relate to the state,
> compute the conditional mean and covariance of the state at time $k$
> given all measurements through time $k$.

The conditional mean is the best estimate in a sense made precise in
Section 6. The conditional covariance quantifies the uncertainty
remaining in that estimate.

The assumptions this derivation depends on:

- The state evolves according to a linear process model with additive
  Gaussian process noise (Section 2.1)
- Measurements are a linear function of the state with additive Gaussian
  measurement noise (Section 2.2)
- The process noise and measurement noise are mutually independent,
  white, and have known covariance matrices (Section 2.3)
- The initial state is Gaussian with known mean and covariance

These assumptions are restrictive. Many real systems violate them. The
Kalman filter remains useful in those cases as an approximation and as
the foundation for more general filters. Those extensions are out of
scope here. The derivation that follows holds exactly under the
assumptions above.
