"""Tests for the visualization module.

The tests split into two groups by matplotlib availability, the layout
the numpy-only-core principle calls for. Group 1 verifies the import
boundary and always runs: importing visualize must not require
matplotlib, and a plotting call without matplotlib must fail with an
actionable error. Group 2 verifies the plotting itself and is gated on
matplotlib being installed, skipping cleanly when it is not.

The plotting functions are exercised on a short, model-consistent filter
run. The tests assert that each returns a matplotlib Figure, that the
reference lines are drawn, and that a caller-supplied axes is honored.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator

import numpy as np
import pytest

import visualize
from kalman_filter import KalmanFilter


def _run_1d_filter(
    n_steps: int = 60,
) -> tuple[KalmanFilter, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a short, model-consistent 1D constant-velocity filter (m=1).

    Returns the filter (carrying the recorded innovations and innovation
    covariances) together with the estimates, measurements, true states,
    and posterior covariances, each as an array indexed by step.
    """
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 0.02 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R = np.array([[1.0]])
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2)

    rng = np.random.default_rng(30)
    chol_Q = np.linalg.cholesky(Q)
    chol_R = np.linalg.cholesky(R)
    chol_P0 = np.linalg.cholesky(P0)
    x = x0 + chol_P0 @ rng.standard_normal(2)
    truth = []
    measurements = []
    for _ in range(n_steps):
        x = F @ x + chol_Q @ rng.standard_normal(2)
        z = H @ x + chol_R @ rng.standard_normal(1)
        truth.append(x.copy())
        measurements.append(z)
    truth = np.array(truth)
    measurements = np.array(measurements)

    kf = KalmanFilter(F, H, Q, R, x0, P0)
    estimates = []
    posteriors = []
    for z in measurements:
        kf.predict()
        kf.update(z)
        estimates.append(kf.x.copy())
        posteriors.append(kf.P.copy())
    return kf, np.array(estimates), measurements, truth, np.array(posteriors)


def _run_2d_filter(
    n_steps: int = 50,
) -> tuple[KalmanFilter, np.ndarray, np.ndarray, np.ndarray]:
    """Run a short, model-consistent 2D constant-velocity filter (m=2).

    The state is [x, vx, y, vy] and the sensor observes the two
    positions. Returns the filter, the estimates, measurements, and true
    states, each as an array indexed by step.
    """
    dt = 1.0
    f_chain = np.array([[1.0, dt], [0.0, 1.0]])
    q_chain = 0.02 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    zero = np.zeros((2, 2))
    F = np.block([[f_chain, zero], [zero, f_chain]])
    Q = np.block([[q_chain, zero], [zero, q_chain]])
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    R = np.eye(2)
    x0 = np.array([0.0, 1.0, 0.0, -0.5])
    P0 = np.eye(4)

    rng = np.random.default_rng(31)
    chol_Q = np.linalg.cholesky(Q)
    chol_R = np.linalg.cholesky(R)
    chol_P0 = np.linalg.cholesky(P0)
    x = x0 + chol_P0 @ rng.standard_normal(4)
    truth = []
    measurements = []
    for _ in range(n_steps):
        x = F @ x + chol_Q @ rng.standard_normal(4)
        z = H @ x + chol_R @ rng.standard_normal(2)
        truth.append(x.copy())
        measurements.append(z)
    truth = np.array(truth)
    measurements = np.array(measurements)

    kf = KalmanFilter(F, H, Q, R, x0, P0)
    estimates = []
    for z in measurements:
        kf.predict()
        kf.update(z)
        estimates.append(kf.x.copy())
    return kf, np.array(estimates), measurements, truth


# Group 1: the lazy-import boundary. These run regardless of matplotlib.


def test_module_imports_core_only() -> None:
    """import visualize succeeds and exposes the plotting functions.

    The module imports with NumPy alone. The plotting entry points are
    present as attributes whether or not matplotlib is installed.
    """
    for name in (
        "plot_tracking",
        "plot_innovation_sequence",
        "plot_nis",
        "plot_covariance_evolution",
    ):
        assert hasattr(visualize, name)


def test_import_does_not_load_matplotlib() -> None:
    """Importing visualize must not pull matplotlib into sys.modules.

    This is the numpy-only-core guarantee. The test drops matplotlib and
    visualize from the module cache, imports visualize fresh, and asserts
    matplotlib was not loaded as a side effect. The original module cache
    is restored afterward so other tests are unaffected.
    """
    saved = dict(sys.modules)
    for name in list(sys.modules):
        if (
            name == "visualize"
            or name == "matplotlib"
            or name.startswith("matplotlib.")
        ):
            sys.modules.pop(name)
    try:
        importlib.import_module("visualize")
        assert "matplotlib" not in sys.modules
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


def test_plotting_without_matplotlib_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plotting call raises an actionable ImportError when matplotlib is absent.

    Setting matplotlib to None in the module cache makes the lazy import
    inside the plotting function fail, simulating an environment without
    the viz extra even when matplotlib happens to be installed. The error
    must name matplotlib and point at the viz extra.
    """
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    innovations = [np.array([0.1]), np.array([-0.2]), np.array([0.05])]
    covariances = [np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])]
    with pytest.raises(ImportError) as excinfo:
        visualize.plot_nis(innovations, covariances)

    message = str(excinfo.value)
    assert "matplotlib" in message
    assert "viz" in message


# Group 2: the plotting itself. Gated on matplotlib through importorskip.


class TestPlotting:
    """Plotting tests that require matplotlib.

    The autouse fixture skips the whole class when matplotlib is absent
    and otherwise selects the non-interactive Agg backend, so no display
    is needed and figures are closed after each test.
    """

    @pytest.fixture(autouse=True)
    def _matplotlib_agg(self) -> Iterator[None]:
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        yield
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_tracking_1d_returns_figure(self) -> None:
        """plot_tracking returns a Figure for the 1D time series layout."""
        from matplotlib.figure import Figure

        _, estimates, measurements, truth, _ = _run_1d_filter()
        fig = visualize.plot_tracking(
            estimates[:, 0],
            measurements=measurements[:, 0],
            truth=truth[:, 0],
        )
        assert isinstance(fig, Figure)

    def test_plot_tracking_2d_returns_figure(self) -> None:
        """plot_tracking returns a Figure for the 2D trajectory layout."""
        from matplotlib.figure import Figure

        _, estimates, measurements, truth = _run_2d_filter()
        fig = visualize.plot_tracking(
            estimates[:, [0, 2]],
            measurements=measurements,
            truth=truth[:, [0, 2]],
        )
        assert isinstance(fig, Figure)

    def test_plot_innovation_sequence_returns_figure(self) -> None:
        """plot_innovation_sequence returns a Figure on recorded innovations."""
        from matplotlib.figure import Figure

        kf, _, _, _, _ = _run_1d_filter()
        fig = visualize.plot_innovation_sequence(
            kf.innovations, kf.innovation_covariances
        )
        assert isinstance(fig, Figure)

    def test_plot_nis_scalar_draws_reference_lines(self) -> None:
        """plot_nis for m=1 draws the NIS, the expectation, and both bounds.

        The tabulated dimension m=1 has a chi-squared 95 percent interval,
        so the axes hold four lines: the NIS series, the expectation at m,
        and the lower and upper bounds.
        """
        from matplotlib.figure import Figure

        kf, _, _, _, _ = _run_1d_filter()
        fig = visualize.plot_nis(kf.innovations, kf.innovation_covariances)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        assert len(fig.axes[0].lines) == 4

    def test_plot_nis_vector_draws_reference_lines(self) -> None:
        """plot_nis for m=2 draws the NIS, the expectation, and both bounds.

        The dimension m=2 is also tabulated, so the same four lines appear.
        This confirms the vector measurement case is handled.
        """
        from matplotlib.figure import Figure

        kf, _, _, _ = _run_2d_filter()
        fig = visualize.plot_nis(kf.innovations, kf.innovation_covariances)
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].lines) == 4

    def test_plot_covariance_evolution_returns_figure(self) -> None:
        """plot_covariance_evolution returns a Figure with one line per state."""
        from matplotlib.figure import Figure

        _, _, _, _, posteriors = _run_1d_filter()
        fig = visualize.plot_covariance_evolution(posteriors)
        assert isinstance(fig, Figure)
        # One variance line per diagonal entry of P; the model has n=2.
        assert len(fig.axes[0].lines) == 2

    def test_functions_use_provided_axes(self) -> None:
        """A caller-supplied axes is drawn on and its figure is returned."""
        import matplotlib.pyplot as plt

        kf, _, _, _, _ = _run_1d_filter()
        fig, ax = plt.subplots()
        returned = visualize.plot_nis(kf.innovations, kf.innovation_covariances, ax=ax)
        assert returned is fig
        assert len(ax.lines) == 4
