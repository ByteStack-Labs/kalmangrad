# Appendices to the Derivation of the Kalman Filter

These appendices support the derivation in [`derivation.md`](derivation.md).
Appendix A collects the matrix identities used. Appendix B states and
derives the Gaussian conditioning formula and shows how it produces the
Kalman update. Appendix C consolidates the notation.

## Appendix A: Matrix Identities

This appendix collects the matrix identities used in the derivation.
Each is stated without proof; references to standard treatments are
given where a proof is wanted.

### A.1 Trace derivatives

The Kalman gain derivation in Section 5.2 minimizes the trace of the
updated covariance with respect to the gain matrix. Two trace-derivative
identities are used.

For a matrix $\mathbf{K}$ and a matrix $\mathbf{A}$ that does not depend
on $\mathbf{K}$:

$$
\frac{\partial}{\partial \mathbf{K}} \, \text{tr}(\mathbf{K} \mathbf{A}) = \mathbf{A}^T
\tag{A.1}
$$

For a matrix $\mathbf{K}$ and a symmetric matrix $\mathbf{A}$ that does
not depend on $\mathbf{K}$:

$$
\frac{\partial}{\partial \mathbf{K}} \, \text{tr}(\mathbf{K} \mathbf{A} \mathbf{K}^T) = 2 \mathbf{K} \mathbf{A}
\tag{A.2}
$$

These follow from the definition of the matrix derivative as the matrix
of partial derivatives of the scalar trace with respect to each entry
of $\mathbf{K}$. A full treatment is given in the Matrix Cookbook
(Petersen and Pedersen).

The application in Section 5.2 expands the trace of the updated
covariance in equation (5.11):

$$
\text{tr}(\mathbf{P}_k) = \text{tr}\left[(\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^- (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T\right] + \text{tr}\left[\mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T\right]
\tag{A.3}
$$

Differentiating the first term with respect to $\mathbf{K}_k$ uses
identity (A.2) with $\mathbf{A} = \mathbf{P}_k^-$ after expanding the
product, producing $-2(\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)
\mathbf{P}_k^- \mathbf{H}_k^T$. Differentiating the second term uses
identity (A.2) with $\mathbf{A} = \mathbf{R}_k$, producing
$2 \mathbf{K}_k \mathbf{R}_k$. The sum is equation (5.12).

### A.2 Symmetry and positive-definiteness of products

The Joseph form in Section 5.4 relies on a structural fact about
products of the form $\mathbf{M} \mathbf{A} \mathbf{M}^T$.

If $\mathbf{A}$ is symmetric positive-semi-definite and $\mathbf{M}$ is
any conformable matrix, then $\mathbf{M} \mathbf{A} \mathbf{M}^T$ is
symmetric positive-semi-definite.

Symmetry is immediate: $(\mathbf{M} \mathbf{A} \mathbf{M}^T)^T =
\mathbf{M} \mathbf{A}^T \mathbf{M}^T = \mathbf{M} \mathbf{A} \mathbf{M}^T$,
using the symmetry of $\mathbf{A}$.

Positive-semi-definiteness follows from the definition. For any vector
$\mathbf{y}$:

$$
\mathbf{y}^T \mathbf{M} \mathbf{A} \mathbf{M}^T \mathbf{y} = (\mathbf{M}^T \mathbf{y})^T \mathbf{A} (\mathbf{M}^T \mathbf{y}) = \mathbf{u}^T \mathbf{A} \mathbf{u} \geq 0
\tag{A.4}
$$

where $\mathbf{u} = \mathbf{M}^T \mathbf{y}$ and the inequality holds
because $\mathbf{A}$ is positive-semi-definite.

The Joseph form is the sum of two such products:
$(\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^- (\mathbf{I} -
\mathbf{K}_k \mathbf{H}_k)^T$ with $\mathbf{A} = \mathbf{P}_k^-$, and
$\mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$ with $\mathbf{A} =
\mathbf{R}_k$. Both $\mathbf{P}_k^-$ and $\mathbf{R}_k$ are symmetric
positive-semi-definite. The sum of two symmetric positive-semi-definite
matrices is symmetric positive-semi-definite. The Joseph form therefore
preserves the structural properties a covariance matrix must have.

### A.3 The matrix inversion lemma

The matrix inversion lemma, also called the Sherman-Morrison-Woodbury
identity, relates the inverse of a sum to the inverses of its parts:

$$
(\mathbf{A} + \mathbf{U} \mathbf{C} \mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1} \mathbf{U} (\mathbf{C}^{-1} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U})^{-1} \mathbf{V} \mathbf{A}^{-1}
\tag{A.5}
$$

The lemma is not required for the derivation as presented. It is
included because it connects the covariance form of the Kalman filter
to the information form, where the inverse covariance is propagated
instead of the covariance. The information form is out of scope for
this document. The lemma is noted for readers who pursue that
connection.

## Appendix B: The Gaussian Conditioning Formula

The derivation refers to this result in Sections 3.3, 5.3, and 6.1.
It is the foundation for the claim that the Kalman filter computes the
exact conditional distribution under linear-Gaussian assumptions. This
appendix states the result and derives it.

### B.1 Statement

Let $\mathbf{x}$ and $\mathbf{y}$ be jointly Gaussian random vectors
with means, covariances, and cross-covariance:

$$
\begin{bmatrix} \mathbf{x} \\ \mathbf{y} \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_x \\ \boldsymbol{\mu}_y \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\ \boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy} \end{bmatrix} \right)
\tag{B.1}
$$

The conditional distribution of $\mathbf{x}$ given $\mathbf{y}$ is
Gaussian:

$$
p(\mathbf{x} \mid \mathbf{y}) = \mathcal{N}(\boldsymbol{\mu}_{x \mid y}, \boldsymbol{\Sigma}_{x \mid y})
\tag{B.2}
$$

with conditional mean:

$$
\boldsymbol{\mu}_{x \mid y} = \boldsymbol{\mu}_x + \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} (\mathbf{y} - \boldsymbol{\mu}_y)
\tag{B.3}
$$

and conditional covariance:

$$
\boldsymbol{\Sigma}_{x \mid y} = \boldsymbol{\Sigma}_{xx} - \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx}
\tag{B.4}
$$

The conditional mean is a linear function of the observation
$\mathbf{y}$. The conditional covariance does not depend on the
observed value of $\mathbf{y}$. These two facts are the structural
reason the Kalman filter has the form it does.

### B.2 Derivation

The conditional density is the joint density divided by the marginal:

$$
p(\mathbf{x} \mid \mathbf{y}) = \frac{p(\mathbf{x}, \mathbf{y})}{p(\mathbf{y})}
\tag{B.5}
$$

Both numerator and denominator are Gaussian. The ratio is the
exponential of a quadratic form in $\mathbf{x}$, which means the
conditional density is Gaussian. The derivation finds its mean and
covariance by completing the square in the exponent.

It is cleaner to work with the joint precision matrix, the inverse of
the joint covariance. Write the joint precision in block form:

$$
\begin{bmatrix} \boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\ \boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy} \end{bmatrix}^{-1} = \begin{bmatrix} \boldsymbol{\Lambda}_{xx} & \boldsymbol{\Lambda}_{xy} \\ \boldsymbol{\Lambda}_{yx} & \boldsymbol{\Lambda}_{yy} \end{bmatrix}
\tag{B.6}
$$

The exponent of the joint Gaussian is the quadratic form:

$$
-\frac{1}{2} \begin{bmatrix} \mathbf{x} - \boldsymbol{\mu}_x \\ \mathbf{y} - \boldsymbol{\mu}_y \end{bmatrix}^T \begin{bmatrix} \boldsymbol{\Lambda}_{xx} & \boldsymbol{\Lambda}_{xy} \\ \boldsymbol{\Lambda}_{yx} & \boldsymbol{\Lambda}_{yy} \end{bmatrix} \begin{bmatrix} \mathbf{x} - \boldsymbol{\mu}_x \\ \mathbf{y} - \boldsymbol{\mu}_y \end{bmatrix}
\tag{B.7}
$$

Expanding and collecting the terms in $\mathbf{x}$, the quadratic part
in $\mathbf{x}$ is governed by $\boldsymbol{\Lambda}_{xx}$. The
conditional precision is therefore $\boldsymbol{\Lambda}_{xx}$, and the
conditional covariance is its inverse:

$$
\boldsymbol{\Sigma}_{x \mid y} = \boldsymbol{\Lambda}_{xx}^{-1}
\tag{B.8}
$$

The block-inverse relationship between the covariance and precision
gives $\boldsymbol{\Lambda}_{xx}$ in terms of the covariance blocks. The
relevant block of the inverse is:

$$
\boldsymbol{\Lambda}_{xx} = (\boldsymbol{\Sigma}_{xx} - \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx})^{-1}
\tag{B.9}
$$

This is the inverse of the Schur complement of $\boldsymbol{\Sigma}_{yy}$
in the joint covariance. Substituting into equation (B.8) gives the
conditional covariance:

$$
\boldsymbol{\Sigma}_{x \mid y} = \boldsymbol{\Sigma}_{xx} - \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx}
\tag{B.10}
$$

This is equation (B.4). The conditional mean follows from the linear
term in the expansion of the exponent. Collecting the terms linear in
$\mathbf{x}$ and completing the square gives:

$$
\boldsymbol{\mu}_{x \mid y} = \boldsymbol{\mu}_x + \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} (\mathbf{y} - \boldsymbol{\mu}_y)
\tag{B.11}
$$

This is equation (B.3). The full algebra of the completing-the-square
step is standard and is given in Bishop (Pattern Recognition and
Machine Learning, Section 2.3) and Rasmussen and Williams (Gaussian
Processes for Machine Learning, Appendix A).

### B.3 Connection to the Kalman update

The Kalman update is the Gaussian conditioning formula applied to the
joint distribution of the state and the measurement.

At time $k$, before the measurement is processed, the predicted state
is Gaussian with mean $\hat{\mathbf{x}}_k^-$ and covariance
$\mathbf{P}_k^-$. The measurement is related to the state by
$\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$. The state
and measurement are jointly Gaussian with the following blocks.

The state mean and covariance:

$$
\boldsymbol{\mu}_x = \hat{\mathbf{x}}_k^-, \quad \boldsymbol{\Sigma}_{xx} = \mathbf{P}_k^-
\tag{B.12}
$$

The measurement mean and covariance, from the measurement model:

$$
\boldsymbol{\mu}_y = \mathbf{H}_k \hat{\mathbf{x}}_k^-, \quad \boldsymbol{\Sigma}_{yy} = \mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k = \mathbf{S}_k
\tag{B.13}
$$

The cross-covariance between state and measurement:

$$
\boldsymbol{\Sigma}_{xy} = \mathbf{P}_k^- \mathbf{H}_k^T
\tag{B.14}
$$

Substituting these into the conditioning formula for the mean,
equation (B.3):

$$
\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{P}_k^- \mathbf{H}_k^T \mathbf{S}_k^{-1} (\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_k^-)
\tag{B.15}
$$

The matrix $\mathbf{P}_k^- \mathbf{H}_k^T \mathbf{S}_k^{-1}$ is the
Kalman gain $\mathbf{K}_k$ from equation (5.14). The term
$\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_k^-$ is the innovation
$\boldsymbol{\nu}_k$ from equation (5.1). Equation (B.15) is therefore
the Kalman update from equation (5.7).

Substituting the same blocks into the conditioning formula for the
covariance, equation (B.4):

$$
\mathbf{P}_k = \mathbf{P}_k^- - \mathbf{P}_k^- \mathbf{H}_k^T \mathbf{S}_k^{-1} \mathbf{H}_k \mathbf{P}_k^-
\tag{B.16}
$$

Factoring $\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}_k^T \mathbf{S}_k^{-1}$
gives:

$$
\mathbf{P}_k = \mathbf{P}_k^- - \mathbf{K}_k \mathbf{H}_k \mathbf{P}_k^- = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^-
\tag{B.17}
$$

This is the standard form of the updated covariance from equation
(5.15). The Gaussian conditioning formula produces the Kalman update
exactly. The minimum-variance derivation in Section 5 and the
conditioning derivation here are two routes to the same result.

## Appendix C: Notation Reference

This appendix consolidates the notation used throughout the derivation.
The symbols are grouped by role. Where a symbol is defined in the body,
the defining section is noted.

### C.1 State and estimate quantities

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| $\mathbf{x}_k$ | True state at time $k$ | $n$ |
| $\hat{\mathbf{x}}_k$ | Updated state estimate at time $k$ (after measurement) | $n$ |
| $\hat{\mathbf{x}}_k^-$ | Predicted state estimate at time $k$ (before measurement) | $n$ |
| $\mathbf{P}_k$ | Updated state covariance at time $k$ | $n \times n$ |
| $\mathbf{P}_k^-$ | Predicted state covariance at time $k$ | $n \times n$ |

### C.2 Model quantities

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| $\mathbf{F}_k$ | State transition matrix | $n \times n$ |
| $\mathbf{Q}_k$ | Process noise covariance | $n \times n$ |
| $\mathbf{H}_k$ | Measurement matrix | $m \times n$ |
| $\mathbf{R}_k$ | Measurement noise covariance | $m \times m$ |
| $\mathbf{w}_k$ | Process noise | $n$ |
| $\mathbf{v}_k$ | Measurement noise | $m$ |

### C.3 Measurement and update quantities

| Symbol | Meaning | Dimension | Defined |
|--------|---------|-----------|---------|
| $\mathbf{z}_k$ | Measurement at time $k$ | $m$ | Section 2.2 |
| $\boldsymbol{\nu}_k$ | Innovation | $m$ | Section 5.1 |
| $\mathbf{S}_k$ | Innovation covariance | $m \times m$ | Section 5.1 |
| $\mathbf{K}_k$ | Kalman gain | $n \times m$ | Section 5.2 |
| $\epsilon_{\nu, k}$ | Normalized innovation squared (NIS) | scalar | Section 7.3 |

### C.4 Dimensions and indices

| Symbol | Meaning |
|--------|---------|
| $n$ | State dimension |
| $m$ | Measurement dimension |
| $k$ | Discrete time index |

### C.5 Conventions

The hat denotes an estimate. $\hat{\mathbf{x}}$ is an estimate of the
true state $\mathbf{x}$.

The superscript minus denotes a predicted quantity, computed before the
measurement at the current time step is processed. $\hat{\mathbf{x}}_k^-$
is the state estimate at time $k$ before $\mathbf{z}_k$ is incorporated.
The absence of a superscript denotes an updated quantity, computed after
the measurement is processed. $\hat{\mathbf{x}}_k$ is the state estimate
at time $k$ after $\mathbf{z}_k$ is incorporated.

Boldface lowercase denotes a vector. Boldface uppercase denotes a
matrix. The subscript $k$ denotes the time step. A subscript $k$ on a
model matrix allows the model to vary with time; when the model is
time-invariant the subscript is dropped.

The notation follows the conventions in Bar-Shalom, Li, and Kirubarajan
(Estimation with Applications to Tracking and Navigation).

### C.6 Operators and symbols

| Symbol | Meaning |
|--------|---------|
| $\mathbb{E}[\cdot]$ | Expectation |
| $\mathbb{E}[\cdot \mid \cdot]$ | Conditional expectation |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | Gaussian distribution with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$ |
| $p(\cdot)$ | Probability density |
| $p(\cdot \mid \cdot)$ | Conditional probability density |
| $\mathbf{z}_{1:k}$ | The measurement sequence $\{\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_k\}$ |
| $\delta_{kj}$ | Kronecker delta |
| $\text{tr}(\cdot)$ | Matrix trace |
| $(\cdot)^T$ | Matrix transpose |
| $(\cdot)^{-1}$ | Matrix inverse |
| $\mathbf{I}$ | Identity matrix |
| $\mathbf{0}$ | Zero vector or zero matrix |
