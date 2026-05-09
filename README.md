# kalmangrad

> Kalman filtering from scratch with autodiff visualization, mathematical derivation, and diagnostic instrumentation for production estimation systems.

**Status:** v0.1 in development (private)
**License:** Apache 2.0
**Author:** Jesse Moses
**Publisher:** [ByteStack Labs](https://bytestacklabs.com)

---

## What kalmangrad is

kalmangrad is a from-scratch implementation of the standard linear Kalman filter built on a from-scratch autodiff engine, with:

- **Autodiff engine**: micrograd-style scalar autodiff with computational graph visualization, allowing gradients to flow through filter operations
- **Mathematical derivation**: every step traceable from probability theory to implementation, including gradient flow through prediction and update operations
- **Reference implementation**: code that reads like the math, no incidental complexity, built on the autodiff engine
- **Diagnostic instrumentation**: innovation sequence analysis, NIS, NEES, and covariance consistency checks that reveal whether the filter is operating correctly
- **Pedagogical structure**: comments and documentation point to the equations they implement; the computational graph is inspectable

## What kalmangrad is not

- A general-purpose filtering library (use FilterPy or similar for production deployment)
- An exhaustive Kalman tutorial (focused scope: standard KF, no EKF/UKF/particle in v0.1)
- A reference for nonlinear estimation (deferred to later versions)
- Built on PyTorch or JAX (autodiff is implemented from scratch, scalar-level, in the micrograd lineage)

## Quick start

```bash
# Installation (when published)
pip install kalmangrad

# Or from source
git clone https://github.com/ByteStack-Labs/kalmangrad.git
cd kalmangrad
pip install -e .
```

```python
# Worked example (coming with v0.1)
from kalmangrad import KalmanFilter, ConstantVelocityModel

# 1D constant velocity tracking with noisy position measurements
model = ConstantVelocityModel(dim=1, dt=0.1)
filter = KalmanFilter(model)

# Predict and update steps over a noisy measurement sequence
estimates, covariances = filter.run(measurements)

# Gradients flow through filter operations
loss = (estimates - ground_truth).pow(2).sum()
loss.backward()
```

Full worked examples live in [`examples/`](examples/) once v0.1 ships.

## Mathematical thesis

The Kalman filter is the optimal recursive Bayesian estimator for linear systems with Gaussian noise. Stated less formally: it is the closed-form answer to the question "given everything I have observed up to now, what is my best estimate of the system state, and how uncertain should I be about it?"

Three properties make it foundational:

1. **Recursive**: state and covariance update in constant memory; no need to retain history
2. **Optimal**: minimum mean square error estimator under linear-Gaussian assumptions
3. **Self-diagnostic**: innovation sequence statistics reveal when the filter is operating correctly

A fourth property emerges when the filter is built on autodiff: gradients of estimated states with respect to model parameters become directly computable, enabling parameter tuning, sensitivity analysis, and integration with broader differentiable systems.

The full derivation lives in [`docs/derivation.md`](docs/derivation.md), including gradient analysis through filter operations. The implementation in [`src/kalman.py`](src/kalman.py) is annotated with cross-references to the derivation, line by line.

## Repository structure

```
kalmangrad/
├── README.md                    # This file
├── LICENSE                      # Apache 2.0
├── CITATION.cff                 # Citation metadata
├── pyproject.toml               # Project configuration
├── Makefile                     # Common commands
├── .github/                     # CI configuration and templates
├── src/                         # Reference implementation
│   ├── autograd.py              # From-scratch autodiff engine
│   ├── kalman.py                # Core filter built on autograd
│   ├── diagnostics.py           # Innovation sequence analysis, NIS, NEES
│   └── visualize.py             # Diagnostic plots and computational graph visualization
├── docs/                        # Documentation
│   ├── derivation.md            # Full mathematical derivation including gradient analysis
│   ├── methodology.md           # How the work was done
│   └── examples_walkthrough.md  # Detailed walk-throughs
├── examples/                    # Worked examples
├── tests/                       # pytest suite
└── notebooks/                   # Optional interactive exploration
```

## Diagnostics

A correctly-functioning Kalman filter has statistical properties: innovations should be zero-mean and white, NIS should follow a chi-squared distribution, and reported covariance should match actual estimation error. kalmangrad includes:

- **Innovation sequence analysis**: tests that innovations are zero-mean, white, and covariance-consistent
- **Normalized Innovation Squared (NIS)**: chi-squared test for filter consistency
- **Normalized Estimation Error Squared (NEES)**: Monte Carlo consistency check (when ground truth available)
- **Covariance consistency monitoring**: detects when the filter's reported uncertainty diverges from actual error
- **Filter divergence detection**: early warning when assumptions break

Mathematical foundation in [`docs/derivation.md`](docs/derivation.md), implementation in [`src/diagnostics.py`](src/diagnostics.py).

## Development status

In active development toward v0.1.0. Repository is private during v0.1 development; visibility flips public on the v0.1.0 tag.

v0.1.0 ships:
- From-scratch autodiff engine with computational graph visualization
- Standard linear Kalman filter built on the autodiff engine
- Full mathematical derivation including gradient analysis
- Diagnostic instrumentation
- Worked examples demonstrating both estimation and gradient-based use cases

## Citation

If kalmangrad informs your work, citation is appreciated:

```bibtex
@software{kalmangrad,
  author = {Moses, Jesse},
  title = {kalmangrad: Kalman Filtering from Scratch with Diagnostic Instrumentation},
  year = {2026},
  publisher = {ByteStack Labs},
  url = {https://github.com/ByteStack-Labs/kalmangrad}
}
```

A full `CITATION.cff` ships with v0.1.0.

## License

Apache 2.0. See [LICENSE](LICENSE) for full terms.
