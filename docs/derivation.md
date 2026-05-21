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

## 3. The Bayesian Foundation

### 3.1 Recursive Bayesian estimation

The state estimation problem can be stated in Bayesian terms. At time
$k$, we have observed the measurements $\mathbf{z}_{1:k} = \{\mathbf{z}_1,
\mathbf{z}_2, \ldots, \mathbf{z}_k\}$. We want the posterior distribution
of the state given those measurements:

$$
p(\mathbf{x}_k \mid \mathbf{z}_{1:k})
\tag{3.1}
$$

This distribution captures everything that can be known about the state
at time $k$ from the available data. Its mean is the best estimate. Its
covariance is the uncertainty in that estimate.

The recursive structure comes from a simple observation. The posterior
at time $k$ can be computed from the posterior at time $k-1$ without
revisiting all past measurements. This is what makes the filter a
filter rather than a batch estimator.

The recursion proceeds in two steps:

$$
p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) \;\xrightarrow{\text{predict}}\;
p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) \;\xrightarrow{\text{update}}\;
p(\mathbf{x}_k \mid \mathbf{z}_{1:k})
\tag{3.2}
$$

The prediction step takes the posterior from the previous time step
forward through the dynamics, producing a prior at the current time step.
The update step incorporates the new measurement, producing the new
posterior. Each step has a precise mathematical form derived in the
sections that follow.

### 3.2 The prediction-update structure

The prediction step is an application of the law of total probability.
Given the posterior at time $k-1$ and the process model from Section
2.1, the predicted distribution at time $k$ is:

$$
p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) \, p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) \, d\mathbf{x}_{k-1}
\tag{3.3}
$$

The integral marginalizes over the previous state. The transition
density $p(\mathbf{x}_k \mid \mathbf{x}_{k-1})$ encodes the process
model. The result is the distribution of $\mathbf{x}_k$ conditioned on
all measurements up to but not including time $k$.

The update step is an application of Bayes' rule. Given the predicted
distribution and the measurement model from Section 2.2, the posterior
after observing $\mathbf{z}_k$ is:

$$
p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}
\tag{3.4}
$$

The likelihood $p(\mathbf{z}_k \mid \mathbf{x}_k)$ encodes the
measurement model. The denominator is a normalizing constant. The
numerator is the product of the prior at time $k$ and the likelihood
of the observed measurement.

Equations (3.3) and (3.4) are general. They hold for any process and
measurement model with well-defined transition and likelihood densities.
The Kalman filter is what these equations become under the linear-Gaussian
assumptions of Section 2.

### 3.3 Why Gaussian and linear produce closed form

The integrals and ratios in equations (3.3) and (3.4) are generally
intractable. For arbitrary process and measurement models, the
posterior at time $k$ may have no analytical form. Particle filters
and other Monte Carlo methods exist precisely because most state
estimation problems do not admit closed-form solutions.

The linear-Gaussian case is the exception. Under the assumptions of
Section 2:

- The transition density $p(\mathbf{x}_k \mid \mathbf{x}_{k-1})$ is
  Gaussian
- The likelihood $p(\mathbf{z}_k \mid \mathbf{x}_k)$ is Gaussian
- The initial distribution $p(\mathbf{x}_0)$ is Gaussian

Two facts about Gaussian distributions make the recursion close in
closed form.

**Linear transformations of Gaussians are Gaussian.** If
$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and
$\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{b}$, then $\mathbf{y}
\sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$.
This makes the prediction step in equation (3.3) analytically tractable.
The integral over a Gaussian transition density and a Gaussian prior
produces a Gaussian result.

**Conditional distributions of jointly Gaussian variables are Gaussian.**
If $\mathbf{x}$ and $\mathbf{z}$ are jointly Gaussian, then
$p(\mathbf{x} \mid \mathbf{z})$ is Gaussian with mean and covariance
given by the Gaussian conditioning formula (Appendix B). This makes
the update step in equation (3.4) analytically tractable. The ratio
of Gaussian likelihood and Gaussian prior produces a Gaussian
posterior.

The combination of these two facts means that if the posterior at
time $k-1$ is Gaussian, then the posterior at time $k$ is also
Gaussian. The recursion preserves the form. Because the initial
distribution is Gaussian by assumption, the posterior at every time
step is Gaussian.

A Gaussian distribution is fully characterized by its mean and
covariance. The recursion therefore reduces from "propagate a
distribution" to "propagate two quantities: a mean vector and a
covariance matrix." That reduction is what makes the Kalman filter
computationally tractable.

The sections that follow derive the explicit recursions for the mean
and covariance. Section 4 derives the prediction step. Section 5
derives the update step. Section 6 establishes the optimality
properties that hold under the linear-Gaussian assumptions.

## 4. The Prediction Step

### 4.1 Propagating the mean

The prediction step propagates the state estimate forward through the
dynamics. Starting from the posterior at time $k-1$:

$$
p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k-1}, \mathbf{P}_{k-1})
\tag{4.1}
$$

The predicted state at time $k$ is the conditional expectation of
$\mathbf{x}_k$ given the measurements through time $k-1$:

$$
\hat{\mathbf{x}}_k^- = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k-1}]
\tag{4.2}
$$

Substituting the process model from equation (2.1):

$$
\hat{\mathbf{x}}_k^- = \mathbb{E}[\mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{w}_k \mid \mathbf{z}_{1:k-1}]
\tag{4.3}
$$

Expectation is linear. The process noise $\mathbf{w}_k$ is independent
of all prior measurements and has zero mean. Equation (4.3) reduces to:

$$
\hat{\mathbf{x}}_k^- = \mathbf{F}_k \, \mathbb{E}[\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}] + \mathbb{E}[\mathbf{w}_k]
\tag{4.4}
$$

The first expectation is the posterior mean from time $k-1$. The
second expectation is zero by the noise assumption in equation (2.3).
The predicted mean is therefore:

$$
\boxed{\hat{\mathbf{x}}_k^- = \mathbf{F}_k \hat{\mathbf{x}}_{k-1}}
\tag{4.5}
$$

The predicted state is the previous state estimate propagated through
the state transition matrix. No measurement information has been used.
The dynamics alone determine the predicted state.

### 4.2 Propagating the covariance

The predicted covariance is the conditional covariance of $\mathbf{x}_k$
given the measurements through time $k-1$:

$$
\mathbf{P}_k^- = \mathbb{E}\left[(\mathbf{x}_k - \hat{\mathbf{x}}_k^-)(\mathbf{x}_k - \hat{\mathbf{x}}_k^-)^T \mid \mathbf{z}_{1:k-1}\right]
\tag{4.6}
$$

Substituting the process model and the predicted mean from equation
(4.5):

$$
\mathbf{x}_k - \hat{\mathbf{x}}_k^- = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{w}_k - \mathbf{F}_k \hat{\mathbf{x}}_{k-1} = \mathbf{F}_k(\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1}) + \mathbf{w}_k
\tag{4.7}
$$

The error in the predicted state has two contributions: the propagated
error from the previous estimate, and the new process noise. Both are
zero-mean. Substituting equation (4.7) into equation (4.6) and
expanding the outer product:

$$
\mathbf{P}_k^- = \mathbb{E}\left[\mathbf{F}_k(\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1})(\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1})^T \mathbf{F}_k^T \mid \mathbf{z}_{1:k-1}\right]
$$
$$
+ \mathbb{E}\left[\mathbf{F}_k(\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1}) \mathbf{w}_k^T \mid \mathbf{z}_{1:k-1}\right]
$$
$$
+ \mathbb{E}\left[\mathbf{w}_k (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1})^T \mathbf{F}_k^T \mid \mathbf{z}_{1:k-1}\right]
$$
$$
+ \mathbb{E}\left[\mathbf{w}_k \mathbf{w}_k^T \mid \mathbf{z}_{1:k-1}\right]
\tag{4.8}
$$

The two cross terms vanish. The process noise $\mathbf{w}_k$ is
independent of $\mathbf{x}_{k-1}$ and of all prior measurements; the
expectation of the product factors, and one factor is zero. The
remaining terms are the propagated covariance and the process noise
covariance:

$$
\boxed{\mathbf{P}_k^- = \mathbf{F}_k \mathbf{P}_{k-1} \mathbf{F}_k^T + \mathbf{Q}_k}
\tag{4.9}
$$

The predicted covariance has two terms. The first term, $\mathbf{F}_k
\mathbf{P}_{k-1} \mathbf{F}_k^T$, is the prior covariance transformed
by the dynamics. The second term, $\mathbf{Q}_k$, is the additional
uncertainty introduced by the process noise. The covariance always
grows in the prediction step. Time without measurement increases
uncertainty.

### 4.3 The predicted state distribution

Equations (4.5) and (4.9) give the mean and covariance of the
predicted state. Because the dynamics are linear and the noise is
Gaussian, the predicted state distribution is itself Gaussian:

$$
p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_k^-, \mathbf{P}_k^-)
\tag{4.10}
$$

This is the result of the marginalization integral in equation (3.3)
under linear-Gaussian assumptions. The integral does not need to be
evaluated explicitly. The Gaussian closure property from Section 3.3
guarantees the result.

The predicted distribution carries everything the dynamics know about
the state at time $k$ before the measurement $\mathbf{z}_k$ is
processed. The mean $\hat{\mathbf{x}}_k^-$ is the best estimate of the
state under the dynamics alone. The covariance $\mathbf{P}_k^-$
quantifies the uncertainty in that estimate.

The next section incorporates the measurement.

## 5. The Update Step

### 5.1 The innovation

When the measurement $\mathbf{z}_k$ arrives, it can be compared to what
the predicted state suggests it should be. The expected measurement
under the predicted state is $\mathbf{H}_k \hat{\mathbf{x}}_k^-$.
The difference between the actual measurement and the expected
measurement is the innovation:

$$
\boldsymbol{\nu}_k = \mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_k^-
\tag{5.1}
$$

The innovation captures what the measurement adds beyond what was
already predicted. If the predicted state were exact, the innovation
would be zero. The innovation is nonzero because of two sources: the
error in the predicted state, and the measurement noise.

Substituting the measurement model from equation (2.2) and the
predicted state from Section 4:

$$
\boldsymbol{\nu}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k - \mathbf{H}_k \hat{\mathbf{x}}_k^- = \mathbf{H}_k(\mathbf{x}_k - \hat{\mathbf{x}}_k^-) + \mathbf{v}_k
\tag{5.2}
$$

The first term is the prediction error projected through the
measurement matrix. The second term is the measurement noise. Both
are zero-mean, so the innovation has zero mean under the model
assumptions:

$$
\mathbb{E}[\boldsymbol{\nu}_k] = \mathbf{0}
\tag{5.3}
$$

The innovation covariance is:

$$
\mathbf{S}_k = \mathbb{E}[\boldsymbol{\nu}_k \boldsymbol{\nu}_k^T]
\tag{5.4}
$$

Substituting equation (5.2) and expanding:

$$
\mathbf{S}_k = \mathbb{E}\left[\mathbf{H}_k(\mathbf{x}_k - \hat{\mathbf{x}}_k^-)(\mathbf{x}_k - \hat{\mathbf{x}}_k^-)^T \mathbf{H}_k^T\right] + \mathbb{E}[\mathbf{v}_k \mathbf{v}_k^T]
\tag{5.5}
$$

The cross terms vanish because the measurement noise is independent
of the prediction error. The first remaining term is $\mathbf{H}_k
\mathbf{P}_k^- \mathbf{H}_k^T$ by the definition of $\mathbf{P}_k^-$
in equation (4.6). The second remaining term is $\mathbf{R}_k$ by
the noise assumption in equation (2.4). The innovation covariance is:

$$
\boxed{\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k}
\tag{5.6}
$$

The innovation covariance has two terms. The first term is the
prediction uncertainty projected into measurement space. The second
term is the measurement noise. Both are uncertainty contributions
to the difference between predicted and observed measurements.

The properties of the innovation sequence are central to filter
diagnostics. Section 7 develops them in detail.

### 5.2 The Kalman gain

The updated state estimate is a linear combination of the predicted
state and the innovation:

$$
\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}_k \boldsymbol{\nu}_k
\tag{5.7}
$$

The matrix $\mathbf{K}_k \in \mathbb{R}^{n \times m}$ is the Kalman
gain. It determines how much weight the new measurement receives
relative to the prediction. The derivation that follows establishes
the specific form of $\mathbf{K}_k$ that minimizes the updated
covariance.

The updated covariance is:

$$
\mathbf{P}_k = \mathbb{E}\left[(\mathbf{x}_k - \hat{\mathbf{x}}_k)(\mathbf{x}_k - \hat{\mathbf{x}}_k)^T\right]
\tag{5.8}
$$

Substituting the update equation (5.7):

$$
\mathbf{x}_k - \hat{\mathbf{x}}_k = \mathbf{x}_k - \hat{\mathbf{x}}_k^- - \mathbf{K}_k \boldsymbol{\nu}_k
\tag{5.9}
$$

Substituting the innovation from equation (5.2):

$$
\mathbf{x}_k - \hat{\mathbf{x}}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)(\mathbf{x}_k - \hat{\mathbf{x}}_k^-) - \mathbf{K}_k \mathbf{v}_k
\tag{5.10}
$$

The updated error is a linear combination of the prediction error and
the measurement noise. Substituting into the covariance definition
in equation (5.8) and expanding the outer product:

$$
\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^- (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T
\tag{5.11}
$$

The cross terms vanish by the independence of the measurement noise
and prediction error. Equation (5.11) gives the updated covariance as
a function of the choice of $\mathbf{K}_k$. This is the Joseph form
of the covariance update, developed further in Section 5.4.

The Kalman gain is the choice of $\mathbf{K}_k$ that minimizes the
trace of $\mathbf{P}_k$. The trace is the sum of the variances of
the state estimate components; minimizing it minimizes the total
mean squared error in the updated estimate.

Differentiating the trace of equation (5.11) with respect to
$\mathbf{K}_k$ and setting the derivative to zero:

$$
\frac{\partial \, \text{tr}(\mathbf{P}_k)}{\partial \mathbf{K}_k} = -2(\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^- \mathbf{H}_k^T + 2 \mathbf{K}_k \mathbf{R}_k = \mathbf{0}
\tag{5.12}
$$

Rearranging:

$$
\mathbf{K}_k (\mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k) = \mathbf{P}_k^- \mathbf{H}_k^T
\tag{5.13}
$$

The matrix on the left is the innovation covariance $\mathbf{S}_k$
from equation (5.6). Solving for the gain:

$$
\boxed{\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}_k^T \mathbf{S}_k^{-1}}
\tag{5.14}
$$

The Kalman gain weighs the prediction uncertainty against the
innovation uncertainty. The numerator $\mathbf{P}_k^- \mathbf{H}_k^T$
is the cross-covariance between the state and the measurement. The
denominator $\mathbf{S}_k^{-1}$ is the inverse of the innovation
covariance. Large prediction uncertainty pushes the gain higher,
giving more weight to the measurement. Large measurement noise pushes
the gain lower, giving more weight to the prediction. The filter
balances the two automatically.

### 5.3 The updated state distribution

Substituting the optimal gain from equation (5.14) into the covariance
expression in equation (5.11) and simplifying produces the standard
form of the updated covariance:

$$
\boxed{\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^-}
\tag{5.15}
$$

The updated covariance is the predicted covariance reduced by the
information contributed by the measurement. The factor $\mathbf{I} -
\mathbf{K}_k \mathbf{H}_k$ projects out the dimensions of the state
that the measurement informs about.

The updated state estimate from equation (5.7), the updated covariance
from equation (5.15), and the Gaussian closure property from Section
3.3 give the posterior distribution at time $k$:

$$
p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) = \mathcal{N}(\hat{\mathbf{x}}_k, \mathbf{P}_k)
\tag{5.16}
$$

This is the same result that would be obtained by applying the
Gaussian conditioning formula (Appendix B) directly to the joint
distribution of $\mathbf{x}_k$ and $\mathbf{z}_k$. The minimum-variance
derivation and the conditional-distribution derivation produce the
same answer under linear-Gaussian assumptions. The agreement is
expected. Section 6 develops the optimality properties that explain
why.

### 5.4 The Joseph form

Equation (5.11) gave the updated covariance as a function of any gain
$\mathbf{K}_k$, including non-optimal choices. With the optimal gain
from equation (5.14), equation (5.11) simplifies to the standard form
in equation (5.15). Algebraically the two are equivalent.

Numerically they are not.

The standard form $\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)
\mathbf{P}_k^-$ involves a subtraction. In finite-precision arithmetic,
subtraction of near-equal quantities introduces error. Over many filter
iterations, this error can compound, causing the updated covariance
$\mathbf{P}_k$ to lose symmetry or even positive-definiteness. A
covariance matrix that is not positive-definite is a structural
failure of the filter.

The Joseph form preserves positive-definiteness:

$$
\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^- (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T
\tag{5.17}
$$

Both terms in equation (5.17) are symmetric positive-semi-definite
products. Their sum is symmetric positive-semi-definite by construction.
The Joseph form is numerically robust even when the gain is slightly
suboptimal due to rounding error or covariance specification mismatch.

The reference implementation uses the Joseph form. The standard form
is mathematically cleaner; the Joseph form is what production
estimation deploys.