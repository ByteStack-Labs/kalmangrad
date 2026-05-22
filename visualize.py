"""Teaching plots for the Kalman filter and its innovation diagnostics.

This module provides a small set of focused plots, each making one
pedagogical point about the filter or the diagnostics of Section 7 in
docs/derivation.md. The plots read the sequences the filter records
(the innovations and innovation covariances) and the statistics that
diagnostics.py computes from them.

Matplotlib is an optional dependency, available through the `viz` extra.
It is not a core dependency. This module therefore imports successfully
with NumPy alone: matplotlib is imported lazily inside each plotting
function, not at module level. A plotting call made without matplotlib
installed raises a clear error directing the reader to install the
extra. The core stays NumPy only. Visualization is offered to those who
want to see the diagnostics. It is never required to use or understand
them.

Each plotting function builds and returns a matplotlib Figure. The
functions do not call plt.show and do not write files. The caller
decides whether to display, save, or further customize the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from diagnostics import CHI2_95_INTERVAL, nis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _require_matplotlib() -> ModuleType:
    """Import matplotlib.pyplot lazily, with an actionable error if absent.

    Matplotlib is the only optional dependency and is not part of the
    NumPy only core. Importing it here, rather than at module level,
    keeps the bare import of this module free of matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib, which is not part of the "
            "NumPy only core. Install the optional viz extra with: "
            'uv pip install -e ".[viz]"  or  pip install kalmangrad[viz]'
        ) from exc
    return plt


def _figure_and_axes(plt: ModuleType, ax: Axes | None) -> tuple[Figure, Axes]:
    """Return the figure and axes to draw on, creating them if needed."""
    if ax is not None:
        return ax.figure, ax
    fig, new_ax = plt.subplots(figsize=(10, 5))
    return fig, new_ax


def _as_vectors(vectors: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Coerce a sequence of vectors into a 2D array of shape (K, m)."""
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _as_matrices(matrices: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Coerce a sequence of matrices into a 3D array of shape (K, d, d)."""
    arr = np.asarray(matrices, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1, 1)
    return arr


def plot_tracking(
    estimates: Sequence[np.ndarray] | np.ndarray,
    measurements: Sequence[np.ndarray] | np.ndarray | None = None,
    truth: Sequence[np.ndarray] | np.ndarray | None = None,
    times: np.ndarray | None = None,
    *,
    value_label: str = "position",
    ax: Axes | None = None,
) -> Figure:
    """Plot the filter estimate against the measurements and the truth.

    This is the picture of the filter doing its job (Sections 4 and 5):
    the estimate follows the truth while sitting below the scatter of
    the noisy measurements, because the update step weights each
    measurement by the Kalman gain rather than trusting it outright.

    Two layouts are produced depending on the shape of estimates. A
    one column (or 1D) input is drawn as a time series, one observed
    quantity against the step index. A two column input is drawn as a
    trajectory in the plane, the path through (x, y). The measurements
    and truth, when supplied, follow the same shape convention.

    Args:
        estimates: Filter estimates as a 1D array of shape (K,) for the
            time series layout, or a 2D array of shape (K, 2) for the
            trajectory layout.
        measurements: Optional measurements in the same shape as
            estimates. Drawn as a scatter.
        truth: Optional ground truth in the same shape as estimates.
            Drawn as a line.
        times: Optional step values for the time series layout. Defaults
            to 1 through K. Ignored in the trajectory layout.
        value_label: Axis label for the plotted quantity in the time
            series layout.
        ax: Optional axes to draw on. A new figure is created when None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    plt = _require_matplotlib()
    est = np.asarray(estimates, dtype=float)
    fig, ax = _figure_and_axes(plt, ax)

    if est.ndim == 2 and est.shape[1] == 2:
        if truth is not None:
            xy = np.asarray(truth, dtype=float)
            ax.plot(xy[:, 0], xy[:, 1], color="black", label="truth")
        if measurements is not None:
            zz = np.asarray(measurements, dtype=float)
            ax.scatter(
                zz[:, 0],
                zz[:, 1],
                s=12,
                color="tab:gray",
                alpha=0.5,
                label="measurements",
            )
        ax.plot(est[:, 0], est[:, 1], color="tab:blue", label="estimate")
        ax.set_xlabel("position x")
        ax.set_ylabel("position y")
        ax.set_title("Filter tracking: 2D trajectory (Sections 4, 5)")
        ax.set_aspect("equal", adjustable="datalim")
    else:
        series = est.reshape(-1)
        n_steps = series.shape[0]
        t = (
            np.arange(1, n_steps + 1)
            if times is None
            else np.asarray(times, dtype=float)
        )
        if truth is not None:
            ax.plot(
                t,
                np.asarray(truth, dtype=float).reshape(-1),
                color="black",
                label="truth",
            )
        if measurements is not None:
            ax.scatter(
                t,
                np.asarray(measurements, dtype=float).reshape(-1),
                s=12,
                color="tab:gray",
                alpha=0.5,
                label="measurements",
            )
        ax.plot(t, series, color="tab:blue", label="estimate")
        ax.set_xlabel("step k")
        ax.set_ylabel(value_label)
        ax.set_title("Filter tracking: estimate, measurements, truth (Sections 4, 5)")

    ax.legend()
    return fig


def plot_innovation_sequence(
    innovations: Sequence[np.ndarray] | np.ndarray,
    innovation_covariances: Sequence[np.ndarray] | np.ndarray,
    times: np.ndarray | None = None,
    *,
    component: int = 0,
    n_std: float = 2.0,
    ax: Axes | None = None,
) -> Figure:
    """Plot one innovation component within its predicted spread.

    The innovation has zero mean (Section 7.2) and covariance S
    (Section 7.3, equation 7.4). This plot draws one component of the
    innovation over time against a band of plus or minus n_std times
    the predicted standard deviation, the square root of the matching
    diagonal entry of S. Under a correctly specified filter the
    innovations scatter around zero and stay inside the band at the
    expected rate. A systematic drift away from zero signals the bias
    of Section 7.2. A scatter that does not match the band signals the
    covariance mismatch of Section 7.3.

    Args:
        innovations: Recorded innovations as a sequence of (m,) arrays,
            a 2D array of shape (K, m), or a sequence of scalars.
        innovation_covariances: Recorded innovation covariances as a
            sequence of (m, m) arrays, a 3D array of shape (K, m, m), or
            a sequence of scalars.
        times: Optional step values. Defaults to 1 through K.
        component: Index of the innovation component to plot. Defaults
            to 0.
        n_std: Half width of the predicted spread band in standard
            deviations. Defaults to 2.0.
        ax: Optional axes to draw on. A new figure is created when None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    plt = _require_matplotlib()
    nu = _as_vectors(innovations)
    S = _as_matrices(innovation_covariances)
    n_steps = nu.shape[0]
    t = np.arange(1, n_steps + 1) if times is None else np.asarray(times, dtype=float)

    series = nu[:, component]
    # Section 7.3, equation (7.4): the predicted innovation spread is read
    # from the diagonal of S. The band is plus or minus n_std of it.
    std = np.sqrt(S[:, component, component])

    fig, ax = _figure_and_axes(plt, ax)
    ax.axhline(0.0, color="black", linewidth=1.0, label="zero mean (Section 7.2)")
    ax.fill_between(
        t,
        -n_std * std,
        n_std * std,
        color="tab:blue",
        alpha=0.2,
        label=f"plus or minus {n_std:g} sqrt(S) (Section 7.3)",
    )
    ax.plot(
        t,
        series,
        color="tab:blue",
        marker=".",
        linestyle="none",
        label="innovation",
    )
    ax.set_xlabel("step k")
    ax.set_ylabel(f"innovation component {component}")
    ax.set_title("Innovation within its predicted spread (Sections 7.2, 7.3)")
    ax.legend()
    return fig


def plot_nis(
    innovations: Sequence[np.ndarray] | np.ndarray,
    innovation_covariances: Sequence[np.ndarray] | np.ndarray,
    times: np.ndarray | None = None,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot the NIS over time against its chi-squared reference lines.

    The normalized innovation squared (Section 7.3, equation 7.5) is
    chi-squared with m degrees of freedom under a correctly specified
    filter, so each sample has expectation m. This plot draws the NIS
    over time, a reference line at m, and the per sample 95 percent
    chi-squared bounds when m is tabulated in diagnostics. NIS hugging
    m with about 1 sample in 20 outside the bounds is the signature of
    a consistent filter. NIS climbing far above the upper bound and
    staying there is the divergence signature of Section 7.4.

    The NIS values are computed by diagnostics.nis. They are not
    recomputed here.

    Args:
        innovations: Recorded innovations as a sequence of (m,) arrays,
            a 2D array of shape (K, m), or a sequence of scalars.
        innovation_covariances: Recorded innovation covariances as a
            sequence of (m, m) arrays, a 3D array of shape (K, m, m), or
            a sequence of scalars.
        times: Optional step values. Defaults to 1 through K.
        ax: Optional axes to draw on. A new figure is created when None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    plt = _require_matplotlib()
    values = nis(innovations, innovation_covariances)
    nu = _as_vectors(innovations)
    n_steps, m = nu.shape
    t = np.arange(1, n_steps + 1) if times is None else np.asarray(times, dtype=float)

    fig, ax = _figure_and_axes(plt, ax)
    ax.plot(t, values, color="tab:blue", marker=".", linestyle="none", label="NIS")
    # Section 7.3: a single NIS sample is chi-squared(m) with mean m.
    ax.axhline(m, color="black", linewidth=1.2, label=f"expectation m = {m}")

    interval = CHI2_95_INTERVAL.get(m)
    if interval is not None:
        lower, upper = interval
        ax.axhline(
            upper,
            color="tab:red",
            linestyle="dashed",
            linewidth=1.0,
            label=f"95% bounds [{lower:.3f}, {upper:.3f}]",
        )
        ax.axhline(lower, color="tab:red", linestyle="dashed", linewidth=1.0)

    ax.set_xlabel("step k")
    ax.set_ylabel("NIS")
    ax.set_title(
        "Normalized innovation squared against chi-squared bounds (Sections 7.3, 7.4)"
    )
    ax.legend()
    return fig


def plot_covariance_evolution(
    covariances: Sequence[np.ndarray] | np.ndarray,
    times: np.ndarray | None = None,
    *,
    components: Sequence[int] | None = None,
    state_labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot state variances over time as the covariance settles.

    The posterior covariance P evolves through the predict and update
    recursion. Under a time invariant model it converges to a steady
    state, the fixed point of the recursion discussed in Section 6:
    the prediction step inflates the covariance, the update step
    contracts it, and the two reach balance. This plot draws selected
    diagonal entries of P, the per state variances, over time.

    Args:
        covariances: Recorded posterior covariances as a sequence of
            (n, n) arrays or a 3D array of shape (K, n, n).
        times: Optional step values. Defaults to 1 through K.
        components: State indices whose variances to plot. Defaults to
            all n diagonal entries.
        state_labels: Optional labels for the plotted states, indexed by
            state component. Defaults to var(x[i]).
        ax: Optional axes to draw on. A new figure is created when None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    plt = _require_matplotlib()
    P = _as_matrices(covariances)
    n_steps, n = P.shape[0], P.shape[1]
    t = np.arange(1, n_steps + 1) if times is None else np.asarray(times, dtype=float)
    selected = range(n) if components is None else components

    fig, ax = _figure_and_axes(plt, ax)
    for i in selected:
        variance = P[:, i, i]
        label = f"var(x[{i}])" if state_labels is None else state_labels[i]
        ax.plot(t, variance, marker=".", linestyle="solid", label=label)

    ax.set_xlabel("step k")
    ax.set_ylabel("state variance (diagonal of P)")
    ax.set_title("Covariance evolution toward steady state (Section 6)")
    ax.legend()
    return fig
