# kalmangrad: Kalman Filtering from Scratch, Built to Diagnose Itself

<p align="center">
  <a href="https://github.com/ByteStack-Labs/kalmangrad" target="_blank">
    <img src="https://raw.githubusercontent.com/ByteStack-Labs/kalmangrad/main/docs/assets/kalmangrad-hero.png" alt="kalmangrad: Kalman Filtering from Scratch, Built to Diagnose Itself" width="640"/>
  </a>
</p>
<p align="center">
  <a href="https://github.com/ByteStack-Labs/kalmangrad/releases/tag/v0.1.0">
    <img src="https://img.shields.io/badge/version-v0.1.0-brightgreen" alt="Version: v0.1.0">
  </a>
  <a href="https://github.com/ByteStack-Labs/kalmangrad/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0">
  </a>
  <a href="https://orcid.org/0009-0006-0322-7974">
    <img src="https://img.shields.io/badge/ORCID-0009--0006--0322--7974-A6CE39?logo=orcid&logoColor=white" alt="ORCID: 0009-0006-0322-7974">
  </a>
</p>

---

A from-scratch implementation of the standard linear Kalman filter: the full derivation, a reference implementation that reads like the math, and the diagnostic instrumentation that proves the filter is operating correctly.

Built to be read and to diagnose itself, not for production deployment. Every step is traceable from probability theory to code. Every consistency claim is checkable.

---

## What kalmangrad is

kalmangrad is a from-scratch implementation of the standard linear Kalman filter, with:

- **Mathematical derivation**: every step traceable from probability theory to implementation
- **Reference implementation**: code that reads like the math, no incidental complexity
- **Diagnostic instrumentation**: innovation sequence analysis, NIS, NEES, and covariance consistency checks that reveal whether the filter is operating correctly
- **Pedagogical structure**: comments and documentation point to the equations they implement

## What kalmangrad is not

- A general-purpose filtering library (use FilterPy or similar for production deployment)
- An exhaustive Kalman tutorial (focused scope: standard KF, no EKF/UKF/particle in v0.1)
- A reference for nonlinear estimation (deferred to later versions)

## Quick Start

Install from source:

```bash
git clone https://github.com/ByteStack-Labs/kalmangrad.git
cd kalmangrad
pip install -e .
```

```python
import numpy as np
from kalman_filter import KalmanFilter

# 1D constant-velocity model: the state is [position, velocity]
# and the sensor observes position only.
dt = 1.0
F = np.array([[1.0, dt], [0.0, 1.0]])    # state transition
H = np.array([[1.0, 0.0]])               # measurement model
Q = np.array([[0.25, 0.5], [0.5, 1.0]])  # process noise covariance
R = np.array([[4.0]])                    # measurement noise covariance
x0 = np.array([0.0, 0.0])                # initial state estimate
P0 = np.eye(2)                           # initial covariance

kf = KalmanFilter(F, H, Q, R, x0, P0)

# Run the predict and update cycle over a measurement sequence.
for z in measurements:
    kf.predict()
    kf.update(np.array([z]))
    estimate = kf.x    # current state estimate
    covariance = kf.P  # current covariance
```

For visualization, install the optional extra:

```bash
pip install -e ".[viz]"
```

Full worked examples live in [`examples/`](examples/).

## Mathematical Thesis

The Kalman filter is the optimal recursive Bayesian estimator for linear systems with Gaussian noise. Stated less formally: it is the closed-form answer to the question "given everything I have observed up to now, what is my best estimate of the system state, and how uncertain should I be about it?"

Three properties make it foundational:

1. **Recursive**: state and covariance update in constant memory; no need to retain history
2. **Optimal**: minimum mean square error estimator under linear-Gaussian assumptions
3. **Self-diagnostic**: innovation sequence statistics reveal when the filter is operating correctly

The full derivation lives in [`docs/derivation.md`](docs/derivation.md). The implementation in [`kalman_filter.py`](kalman_filter.py) is annotated with cross-references to the derivation, line by line.

## Repository Structure

```
kalmangrad/
├── README.md                    # This file
├── LICENSE                      # Apache 2.0
├── CITATION.cff                 # Citation metadata
├── pyproject.toml               # Project configuration
├── kalman_filter.py             # Core Kalman filter reference implementation
├── diagnostics.py               # Innovation sequence analysis, NIS, NEES
├── visualize.py                 # Matplotlib-based diagnostic plots
├── docs/                        # Documentation
│   ├── derivation.md            # Full mathematical derivation
│   ├── appendices.md            # Supporting derivations and notation
│   ├── methodology.md           # How the work was done
│   └── assets/                  # README hero and figures
├── examples/                    # Worked examples
└── tests/                       # pytest suite
```

## Diagnostics

A correctly functioning Kalman filter has statistical properties: innovations should be zero-mean and white, NIS should follow a chi-squared distribution, and reported covariance should match actual estimation error. kalmangrad includes:

- **Innovation sequence analysis**: tests that innovations are zero-mean, white, and covariance-consistent
- **Normalized Innovation Squared (NIS)**: chi-squared test for filter consistency
- **Normalized Estimation Error Squared (NEES)**: Monte Carlo consistency check (when ground truth is available)
- **Covariance consistency monitoring**: detects when the filter's reported uncertainty diverges from actual error
- **Filter divergence detection**: early warning when assumptions break

Mathematical foundation in [`docs/derivation.md`](docs/derivation.md), implementation in [`diagnostics.py`](diagnostics.py).

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

## License

Apache 2.0. See [LICENSE](LICENSE) for full terms.
```

Built by [Jesse Moses](https://github.com/Cre4T3Tiv3) at [ByteStack Labs](https://bytestacklabs.com).