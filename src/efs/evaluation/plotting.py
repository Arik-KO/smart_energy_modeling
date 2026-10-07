"""Paper figure style, applied the same way to every figure.

Rules: 300 dpi, no title, 10 pt bold axis labels and tick labels, a legend,
and axis limits that always end on a tick at or beyond the data, so no curve
ever runs past the last labelled value.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only; works without a display (WSL, servers)

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def apply_paper_style(plot_cfg: dict) -> None:
    """Set global matplotlib defaults from the ``plotting`` config section."""
    size = int(plot_cfg["font_size"])
    weight = str(plot_cfg["font_weight"])
    plt.rcParams.update({
        "font.size": size,
        "axes.labelsize": size,
        "axes.labelweight": weight,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "axes.titlesize": size,
        "savefig.dpi": int(plot_cfg["dpi"]),
        "savefig.bbox": "tight",
    })


def nice_ticks(values: np.ndarray, n_bins: int = 6, include_zero: bool = False) -> np.ndarray:
    """Evenly spaced 'round' ticks whose first and last tick enclose all values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    low = float(values.min())
    high = float(values.max())
    if include_zero:
        low = min(low, 0.0)
    if high == low:
        high = low + 1.0
    ticks = MaxNLocator(nbins=n_bins).tick_values(low, high)
    # MaxNLocator brackets the data, but guard anyway: extend by one step if needed.
    step = ticks[1] - ticks[0]
    if ticks[-1] < high:
        ticks = np.append(ticks, ticks[-1] + step)
    if ticks[0] > low:
        ticks = np.insert(ticks, 0, ticks[0] - step)
    return ticks


def _bold(ax, plot_cfg: dict) -> None:
    weight = str(plot_cfg["font_weight"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(weight)


def _legend(ax, plot_cfg: dict, loc: str, ncol: int = 1) -> None:
    legend_box = ax.legend(loc=loc, ncol=ncol)
    for text in legend_box.get_texts():
        text.set_fontweight(str(plot_cfg["font_weight"]))


def _y_ticks(y_arrays: list[np.ndarray], include_zero: bool, headroom: float) -> np.ndarray:
    """Ticks enclosing the data, optionally leaving free space at the top
    (for a legend or an inset) by extending the range before choosing ticks."""
    all_y = np.concatenate([np.asarray(array, dtype=float).ravel() for array in y_arrays])
    all_y = all_y[np.isfinite(all_y)]
    if headroom > 0:
        low = min(all_y.min(), 0.0) if include_zero else all_y.min()
        all_y = np.append(all_y, all_y.max() + headroom * (all_y.max() - low))
    return nice_ticks(all_y, include_zero=include_zero)


def finalize_axes(ax, x_values: np.ndarray, y_arrays: list[np.ndarray], xlabel: str,
                  ylabel: str, plot_cfg: dict, include_zero: bool = True,
                  legend: bool = True, x_ticks: np.ndarray | None = None,
                  legend_loc: str = "best", headroom: float = 0.0, legend_ncol: int = 1) -> None:
    """Labels, bold ticks, data-enclosing limits and legend for one axes."""
    y_ticks = _y_ticks(y_arrays, include_zero, headroom)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_ticks[0], y_ticks[-1])

    if x_ticks is None:
        x_ticks = nice_ticks(np.asarray(x_values, dtype=float), n_bins=10, include_zero=True)
    ax.set_xticks(x_ticks)
    ax.set_xlim(x_ticks[0], x_ticks[-1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _bold(ax, plot_cfg)
    ax.grid(True, alpha=0.3)
    if legend:
        _legend(ax, plot_cfg, legend_loc, legend_ncol)


def finalize_categorical(ax, positions: np.ndarray, labels: list[str], y_arrays: list[np.ndarray],
                         xlabel: str, ylabel: str, plot_cfg: dict, legend: bool = True,
                         legend_loc: str = "best", headroom: float = 0.0,
                         legend_ncol: int = 1) -> None:
    """Like finalize_axes, for bar charts with named categories on the x-axis."""
    y_ticks = _y_ticks(y_arrays, True, headroom)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _bold(ax, plot_cfg)
    ax.grid(True, axis="y", alpha=0.3)
    if legend:
        _legend(ax, plot_cfg, legend_loc, legend_ncol)


def finalize_inset(inset, x_values: np.ndarray, y_arrays: list[np.ndarray], plot_cfg: dict,
                   x_ticks: np.ndarray | None = None) -> None:
    """Zoom panel: same enclosing-limits and bold-tick rules, no labels or legend."""
    finalize_axes(inset, x_values, y_arrays, "", "", plot_cfg, include_zero=False,
                  legend=False, x_ticks=x_ticks)


def new_figure(plot_cfg: dict, tall: bool = False, nrows: int = 1, sharex: bool = False):
    """Figure with the paper's size: 12 x 6 by default, 12 x 8 when tall."""
    size = plot_cfg["tall_figsize"] if tall else plot_cfg["figsize"]
    return plt.subplots(nrows=nrows, figsize=tuple(size), sharex=sharex)


def save_figure(fig, path_stem: Path, plot_cfg: dict) -> list[Path]:
    """Save as every configured format (e.g. PNG for review, PDF for LaTeX)."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for extension in plot_cfg["formats"]:
        path = path_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=int(plot_cfg["dpi"]))
        plt.close(fig) if False else None
        saved.append(path)
    plt.close(fig)
    return saved
