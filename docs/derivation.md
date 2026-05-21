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

## 2. The State-Space Model

### 2.1 The process model

The state of the system at time $k$ is represented by a vector
$\mathbf{x}_k \in \mathbb{R}^n$. The state evolves from one time step
to the next according to:

$$
\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{w}_k
\tag{2.1}
$$

where $\mathbf{F}_k \in \mathbb{R}^{n \times n}$ is the state transition
matrix and $\mathbf{w}_k \in \mathbb{R}^n$ is the process noise.

The state transition matrix $\mathbf{F}_k$ encodes the dynamics of the
system. It specifies how the state at time $k-1$ propagates to time $k$
in the absence of noise. The subscript $k$ allows $\mathbf{F}$ to vary
with time. In many systems $\mathbf{F}$ is constant, in which case the
subscript is dropped.

The process noise $\mathbf{w}_k$ accounts for everything the model does
not capture exactly. Unmodeled forces, discretization error, and
stochastic variation in the dynamics are all absorbed into
$\mathbf{w}_k$. The size of $\mathbf{w}_k$ encodes how much the model
is trusted at each time step.

### 2.2 The measurement model

The state $\mathbf{x}_k$ is not observed directly. What is observed at
time $k$ is a measurement $\mathbf{z}_k \in \mathbb{R}^m$ related to
the state by:

$$
\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k
\tag{2.2}
$$

where $\mathbf{H}_k \in \mathbb{R}^{m \times n}$ is the measurement
matrix and $\mathbf{v}_k \in \mathbb{R}^m$ is the measurement noise.

The measurement matrix $\mathbf{H}_k$ specifies what aspects of the
state are observed and how. A position-only sensor observing a
state of position and velocity has $\mathbf{H} = [1, 0]$. A sensor
that observes a linear combination of state components has rows of
$\mathbf{H}$ corresponding to that combination. The measurement need
not be a one-to-one function of the state; $m < n$ is common and
expected.

The measurement noise $\mathbf{v}_k$ accounts for sensor error,
quantization, and any other source of corruption between the true
state and the recorded measurement. Like the process noise, its
size encodes how much the measurement is trusted.

### 2.3 The Gaussian noise assumptions

The process noise and measurement noise are assumed to be Gaussian with
zero mean and known covariance:

$$
\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)
\tag{2.3}
$$

$$
\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)
\tag{2.4}
$$

where $\mathbf{Q}_k \in \mathbb{R}^{n \times n}$ is the process noise
covariance and $\mathbf{R}_k \in \mathbb{R}^{m \times m}$ is the
measurement noise covariance. Both matrices are symmetric and positive
semi-definite. $\mathbf{R}_k$ is typically positive definite in
practice; a measurement with zero noise in some direction is rarely
physical.

**Zero mean.** The noise has no systematic bias. Any bias in the
dynamics or measurements is assumed to be modeled in $\mathbf{F}_k$ or
$\mathbf{H}_k$ rather than absorbed into the noise. A noise process
with non-zero mean is equivalent to a deterministic forcing term and
should be treated as such.

**Known covariance.** The covariance matrices $\mathbf{Q}_k$ and
$\mathbf{R}_k$ are specified by the modeler. They are not estimated by
the filter during operation. Misspecified covariances degrade filter
performance and are detectable through innovation sequence diagnostics.
Section 7 develops the statistical properties of the innovation that
make this detection possible.

**Whiteness.** The noise is uncorrelated across time:

$$
\mathbb{E}[\mathbf{w}_k \mathbf{w}_j^T] = \mathbf{Q}_k \delta_{kj}
\tag{2.5}
$$

$$
\mathbb{E}[\mathbf{v}_k \mathbf{v}_j^T] = \mathbf{R}_k \delta_{kj}
\tag{2.6}
$$

where $\delta_{kj}$ is the Kronecker delta. Each noise realization is
independent of every other.

**Mutual independence.** The process noise and measurement noise are
independent of each other at all times:

$$
\mathbb{E}[\mathbf{w}_k \mathbf{v}_j^T] = \mathbf{0} \quad \text{for all } k, j
\tag{2.7}
$$

**Initial state.** The initial state $\mathbf{x}_0$ is Gaussian with
known mean and covariance:

$$
\mathbf{x}_0 \sim \mathcal{N}(\hat{\mathbf{x}}_0, \mathbf{P}_0)
\tag{2.8}
$$

The initial state is independent of all process and measurement noise.

These assumptions are restrictive. Real systems violate them in
specific, characterizable ways. Section 7 develops the diagnostics that
detect when the assumptions hold and when they fail.

### 2.4 Notation table

The notation used throughout this derivation:

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| $\mathbf{x}_k$ | True state at time $k$ | $n$ |
| $\hat{\mathbf{x}}_k$ | Estimated state at time $k$ | $n$ |
| $\hat{\mathbf{x}}_k^-$ | Predicted state at time $k$ (before measurement) | $n$ |
| $\mathbf{P}_k$ | State estimate covariance at time $k$ | $n \times n$ |
| $\mathbf{P}_k^-$ | Predicted state covariance at time $k$ | $n \times n$ |
| $\mathbf{F}_k$ | State transition matrix | $n \times n$ |
| $\mathbf{Q}_k$ | Process noise covariance | $n \times n$ |
| $\mathbf{z}_k$ | Measurement at time $k$ | $m$ |
| $\mathbf{H}_k$ | Measurement matrix | $m \times n$ |
| $\mathbf{R}_k$ | Measurement noise covariance | $m \times m$ |
| $\mathbf{w}_k$ | Process noise | $n$ |
| $\mathbf{v}_k$ | Measurement noise | $m$ |
| $\boldsymbol{\nu}_k$ | Innovation (defined in Section 5.1) | $m$ |
| $\mathbf{S}_k$ | Innovation covariance (defined in Section 5.1) | $m \times m$ |
| $\mathbf{K}_k$ | Kalman gain (defined in Section 5.2) | $n \times m$ |
| $n$ | State dimension | scalar |
| $m$ | Measurement dimension | scalar |

State estimates are distinguished from true state by the hat: $\hat{\mathbf{x}}$
denotes an estimate of $\mathbf{x}$. Predicted quantities (before
incorporating a measurement) carry a superscript minus:
$\hat{\mathbf{x}}_k^-$ is the state estimate at time $k$ before the
measurement $\mathbf{z}_k$ has been processed. Updated quantities
(after incorporating the measurement) carry no superscript:
$\hat{\mathbf{x}}_k$ is the state estimate after $\mathbf{z}_k$ has
been processed.

The transition between predicted and updated quantities is the heart
of the filter. The next sections derive that transition.
