# Methodology

This document states the architectural decisions that govern this project and
the reasons for them. The decisions are consistent across the derivation, the
implementation, the diagnostics, and the tests. A reader who understands them
will find the same choices reflected throughout the work.

## The derivation is the source of truth

The mathematics in `docs/derivation.md` is established first. The code expresses
it. This ordering is deliberate: the derivation defines what is correct, and the
implementation realizes it, rather than the implementation defining behavior that
documentation later describes.

Every computational step in the code references the equation it implements. The
Kalman gain in `kalman_filter.py` carries the number of the equation that defines
it. The normalized innovation squared in `diagnostics.py` carries the number of
the equation it computes. A reader can hold the derivation open beside the code
and trace each line to its origin. When a question arises about why the code does
something, the answer is in the derivation, not in the code's history.

## The core depends on NumPy alone

The core of the project, the filter and its diagnostics, depends on NumPy and
nothing else. This keeps the linear algebra visible. The covariance update, the
gain computation, and the innovation statistics are written out in terms a reader
can follow, rather than delegated to a higher-level library that would hide the
operations the derivation describes.

Visualization is the one place a further dependency is useful, and it is isolated.
Matplotlib is an optional dependency, available through the `viz` extra. The
visualization module imports it lazily, so the package installs and the core runs
with NumPy alone. A reader who wants to see the diagnostics plotted installs the
extra. A reader who wants only the filter and its numerical diagnostics never
encounters matplotlib. Visualization is offered. It is never required.

## The diagnostics follow from the mathematics

The diagnostics are not heuristics added after the fact. They follow from the
statistical properties that Section 7 of the derivation establishes. Under a
correctly specified filter, the innovation sequence has zero mean, a known
covariance, and is white. These properties hold by mathematical necessity, and
they fail in characterizable ways when the model is wrong.

The diagnostic functions compute the empirical versions of these properties and
compare them to what the derivation predicts. The normalized innovation squared
tests the covariance. The innovation mean tests for bias. The autocorrelation
tests whiteness. Because the properties are derived rather than assumed, a reader
can verify why each diagnostic detects what it detects. The innovation sequence
is self-diagnosing because the derivation proves that it is.

## The tests verify the mathematics

The tests check that the implementation has the properties the derivation
establishes, not merely that the code runs without error. A test confirms that
the covariance remains symmetric and positive definite across a long run, that
the normalized innovation squared centers on the measurement dimension, that the
innovations are white, and that a misspecified filter is detected. These are the
properties the derivation proves; the tests confirm the code exhibits them.

Where a test asserts a statistical property, the tolerance is derived from the
sampling distribution rather than chosen by hand. A test of the average
normalized innovation squared uses the standard error of a chi-squared mean. A
test of whiteness uses the sampling spread of an autocorrelation estimate. The
tolerances are stated with their derivations, so a reader can see why a result
falls inside or outside the expected range.

## Scope

This project derives and implements the standard discrete-time linear Kalman
filter, with Gaussian noise and known model matrices. It is a teaching artifact
and a reference implementation. It is not a production estimation library.

Extensions to nonlinear systems, the extended Kalman filter, the unscented Kalman
filter, and the particle filter, are out of scope. The linear case is treated
completely, from the Bayesian foundation through the recursive algorithm and the
innovation diagnostics, on the principle that the linear filter understood fully
is the right foundation for the extensions that follow.
