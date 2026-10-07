"""Make every figure for the revised paper from the saved results.

Nothing is retrained or re-simulated: the figures only read the CSV files
written by evaluate.py, schedule.py, value_of_accuracy.py and size_battery.py.

Usage (from the project root):
    python scripts/make_figures.py                     # all figures
    python scripts/make_figures.py --only lead_day dwt_ablation
    python scripts/make_figures.py --schedule-run runs/20260927-175514_schedule
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.evaluation.plotting import (
    apply_paper_style,
    finalize_axes,
    finalize_categorical,
    finalize_inset,
    new_figure,
    save_figure,
)
from efs.forecasting.recursive import seasonal_naive_forecast
from efs.scheduling.battery import daily_totals

ARCHS = ["lstm", "dwt_lstm", "tnn", "dwt_tnn"]
NAMES = {"lstm": "LSTM", "dwt_lstm": "DWT-LSTM", "tnn": "TNN", "dwt_tnn": "DWT-TNN",
         "seasonal_naive": "Naive baseline"}
LINESTYLES = {"lstm": ":", "dwt_lstm": "-.", "tnn": "--", "dwt_tnn": (0, (5, 1, 1, 1))}
HOUR_TICKS = np.arange(0, 721, 48)


# ----------------------------------------------------------------- helpers

def latest_run(runs_root: Path, suffix: str, required_file: str) -> Path:
    """Newest run folder named *_<suffix> that contains ``required_file``."""
    candidates = []
    for folder in sorted(runs_root.iterdir()):
        if folder.is_dir() and folder.name.endswith(f"_{suffix}") and (folder / required_file).exists():
            candidates.append(folder)
    if not candidates:
        raise FileNotFoundError(f"No run folder ending in '_{suffix}' with {required_file} in {runs_root}")
    return candidates[-1]


def seed_mean_forecast(final_root: Path, target: str, arch: str, strategy: str, column: str,
                       seeds: list[int]) -> pd.Series:
    """Hourly forecast averaged over the seeds (one line per model in the plots)."""
    series = []
    for seed in seeds:
        path = final_root / target / arch / strategy / "tuned" / f"seed_{seed}" / "test_forecast.csv"
        frame = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        series.append(frame[column])
    return pd.concat(series, axis=1).mean(axis=1)


def actual_series(final_root: Path, target: str) -> pd.Series:
    path = final_root / target / "lstm" / "dayblock_recursive" / "tuned" / "seed_0" / "test_forecast.csv"
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")["actual_kwh"]


# ----------------------------------------------------------------- figures

def fig_forecast(ctx, target: str, ylabel: str, archs: list[str], name: str) -> None:
    """Month-ahead December forecast vs actual (seed mean), with a one-week zoom."""
    config, plot_cfg = ctx.config, ctx.config["plotting"]
    final_root = resolve_path(config, config["final"]["output_dir"])
    seeds = [int(seed) for seed in config["final"]["seeds"]]
    actual = actual_series(final_root, target).to_numpy()
    hours = np.arange(len(actual))

    fig, ax = new_figure(plot_cfg)
    ax.plot(hours, actual, color="black", linewidth=1.0, label="Observation")
    curves = [actual]
    for arch in archs:
        forecast = seed_mean_forecast(final_root, target, arch, "dayblock_recursive",
                                      "month_ahead_kwh", seeds).to_numpy()
        ax.plot(hours, forecast, linestyle=LINESTYLES[arch], linewidth=1.3,
                label=f"{NAMES[arch]} (month-ahead, mean of {len(seeds)} seeds)")
        curves.append(forecast)
    finalize_axes(ax, hours, curves, "Hour of December 2019", ylabel, plot_cfg,
                  x_ticks=HOUR_TICKS, legend_loc="upper left", headroom=0.55)

    # Zoom on the first week, placed in the free space at the top right.
    inset = ax.inset_axes([0.58, 0.62, 0.40, 0.34])
    week = slice(0, 168)
    inset.plot(hours[week], actual[week], color="black", linewidth=1.0)
    for arch, curve in zip(archs, curves[1:]):
        inset.plot(hours[week], curve[week], linestyle=LINESTYLES[arch], linewidth=1.3)
    finalize_inset(inset, hours[week], [curve[week] for curve in curves], plot_cfg,
                   x_ticks=np.arange(0, 169, 24))
    save_figure(fig, ctx.out_dir / name, plot_cfg)


def fig_lead_day(ctx) -> None:
    """Daily-total error of the month-ahead demand forecast vs lead day."""
    config, plot_cfg = ctx.config, ctx.config["plotting"]
    final_root = resolve_path(config, config["final"]["output_dir"])
    seeds = [int(seed) for seed in config["final"]["seeds"]]

    fig, ax = new_figure(plot_cfg)
    curves = []
    for arch in ARCHS:
        per_seed = []
        for seed in seeds:
            path = final_root / "demand" / arch / "dayblock_recursive" / "tuned" / f"seed_{seed}" / "test_lead_days.csv"
            per_seed.append(pd.read_csv(path)["ape_percent"].to_numpy())
        mean = np.mean(per_seed, axis=0)
        lead = np.arange(1, len(mean) + 1)
        ax.plot(lead, mean, linestyle=LINESTYLES[arch], marker="o", markersize=3, label=NAMES[arch])
        curves.append(mean)

    # Seasonal naive from the same origin, for reference.
    dataset = ctx.dataset
    start = int(dataset.index.get_loc(pd.Timestamp(config["data"]["splits"]["test_start"])))
    values = dataset["demand_kwh"].to_numpy()
    naive = seasonal_naive_forecast(values, start, 720).reshape(30, 24).sum(axis=1)
    actual_daily = values[start:start + 720].reshape(30, 24).sum(axis=1)
    naive_ape = 100 * np.abs(naive - actual_daily) / actual_daily
    lead = np.arange(1, 31)
    ax.plot(lead, naive_ape, color="gray", linewidth=1.5, label="Seasonal naive")
    curves.append(naive_ape)

    finalize_axes(ax, lead, curves, "Lead time (days)", "Daily demand error (%)", plot_cfg,
                  x_ticks=np.arange(0, 31, 5), legend_loc="upper left", legend_ncol=2, headroom=0.15)
    save_figure(fig, ctx.out_dir / "fig_lead_day_error", plot_cfg)


def fig_horizons(ctx) -> None:
    """Demand hourly MAPE for 1-hour, week-ahead and month-ahead forecasts."""
    config, plot_cfg = ctx.config, ctx.config["plotting"]
    summary = pd.read_csv(resolve_path(config, config["final"]["output_dir"]) / "test_summary.csv")
    demand = summary[summary.target == "demand"]
    horizons = [("one_hour_ahead", "hourly_recursive", "1 hour ahead"),
                ("week_ahead", "dayblock_recursive", "Week ahead"),
                ("month_ahead", "dayblock_recursive", "Month ahead")]
    models = ARCHS + ["seasonal_naive"]
    width = 0.16
    positions = np.arange(len(horizons))

    fig, ax = new_figure(plot_cfg)
    tops = []
    for index, arch in enumerate(models):
        means, stds = [], []
        for horizon, strategy, _ in horizons:
            if arch == "seasonal_naive":
                row = demand[(demand.arch == arch) & (demand.horizon == horizon)]
            else:
                row = demand[(demand.arch == arch) & (demand.horizon == horizon) & (demand.strategy == strategy)]
            means.append(float(row["hourly_mape_mean"].iloc[0]))
            std = row["hourly_mape_std"].iloc[0]
            stds.append(0.0 if pd.isna(std) else float(std))
        offset = (index - (len(models) - 1) / 2) * width
        color = "gray" if arch == "seasonal_naive" else None
        ax.bar(positions + offset, means, width, yerr=stds, capsize=3, label=NAMES[arch], color=color)
        tops.append(np.array(means) + np.array(stds))

    finalize_categorical(ax, positions, [label for _, _, label in horizons], tops,
                         "Forecast horizon", "Hourly demand MAPE (%)", plot_cfg,
                         legend_loc="upper left", legend_ncol=3, headroom=0.25)
    save_figure(fig, ctx.out_dir / "fig_horizon_comparison", plot_cfg)


def fig_dwt_ablation(ctx) -> None:
    """Month-ahead December demand MAPE vs DWT level, per wavelet, both DWT models."""
    config, plot_cfg = ctx.config, ctx.config["plotting"]
    ablation = pd.read_csv(resolve_path(config, config["ablation"]["output_dir"]) / "test_summary.csv")
    final = pd.read_csv(resolve_path(config, config["final"]["output_dir"]) / "test_summary.csv")
    levels = [int(level) for level in config["ablation"]["levels"]]
    wavelets = list(config["ablation"]["wavelets"])
    markers = {"haar": "s", "db4": "o", "sym4": "^"}
    offsets = {"haar": -0.08, "db4": 0.0, "sym4": 0.08}

    fig, axes = new_figure(plot_cfg, tall=True, nrows=2, sharex=True)
    for ax, arch, plain in zip(axes, ["dwt_lstm", "dwt_tnn"], ["lstm", "tnn"]):
        curves = []
        for wavelet in wavelets:
            means, stds = [], []
            for level in levels:
                row = ablation[(ablation.target == "demand") & (ablation.arch == arch) &
                               (ablation.horizon == "month_ahead") &
                               (ablation.variant == f"{wavelet}_L{level}_approximation")]
                means.append(float(row["hourly_mape_mean"].iloc[0]))
                stds.append(float(row["hourly_mape_std"].iloc[0]))
            means, stds = np.array(means), np.array(stds)
            # Small horizontal offsets keep the error bars of the three wavelets apart.
            shifted = np.array(levels, dtype=float) + offsets[wavelet]
            ax.errorbar(shifted, means, yerr=stds, marker=markers[wavelet], capsize=4,
                        label=f"{NAMES[arch]}, {wavelet}")
            curves.extend([means + stds, means - stds])
        reference = final[(final.target == "demand") & (final.arch == plain) &
                          (final.strategy == "dayblock_recursive") & (final.horizon == "month_ahead")]
        no_dwt = float(reference["hourly_mape_mean"].iloc[0])
        ax.axhline(no_dwt, color="gray", linestyle="--", label=f"{NAMES[plain]} (no DWT)")
        curves.append(np.array([no_dwt]))
        finalize_axes(ax, np.array(levels), curves, "DWT decomposition level" if ax is axes[-1] else "",
                      "Hourly demand MAPE (%)", plot_cfg, include_zero=False,
                      x_ticks=np.array([0.5] + levels + [3.5]), legend_loc="upper left",
                      legend_ncol=2, headroom=0.6)
        ax.set_xticks(levels)
        ax.set_xlim(0.5, 3.5)
        for label in ax.get_xticklabels():
            label.set_fontweight(plot_cfg["font_weight"])
    save_figure(fig, ctx.out_dir / "fig_dwt_ablation", plot_cfg)


def fig_pv_drift(ctx) -> None:
    """November daily PV: a stable vs a diverged hour-by-hour recursion."""
    config, plot_cfg = ctx.config, ctx.config["plotting"]
    final_root = resolve_path(config, config["final"]["output_dir"])
    stable_seed, diverged_seed = config["figures"]["pv_drift_seeds"]
    folder = final_root / "pv" / "dwt_lstm" / "hourly_recursive" / "tuned"
    stable = pd.read_csv(folder / f"seed_{stable_seed}" / "val_forecast.csv",
                         parse_dates=["timestamp"], index_col="timestamp").resample("D").sum()
    diverged = pd.read_csv(folder / f"seed_{diverged_seed}" / "val_forecast.csv",
                           parse_dates=["timestamp"], index_col="timestamp").resample("D").sum()
    days = np.arange(1, len(stable) + 1)

    fig, ax = new_figure(plot_cfg)
    ax.plot(days, stable["actual_kwh"], color="black", marker="o", markersize=3, label="Observation")
    ax.plot(days, stable["forecast_kwh"], linestyle="--", marker="s", markersize=3,
            label=f"DWT-LSTM hour-by-hour, seed {stable_seed} (stable)")
    ax.plot(days, diverged["forecast_kwh"], linestyle="-.", marker="^", markersize=3,
            label=f"DWT-LSTM hour-by-hour, seed {diverged_seed} (diverged)")
    finalize_axes(ax, days, [stable["actual_kwh"], stable["forecast_kwh"], diverged["forecast_kwh"]],
                  "Day of November 2019", "Daily PV energy (kWh)", plot_cfg,
                  x_ticks=np.arange(0, 31, 5), legend_loc="upper right", headroom=0.3)
    save_figure(fig, ctx.out_dir / "fig_pv_recursion_drift", plot_cfg)


def fig_battery(ctx) -> None:
    """December battery charge and daily net import for three schedules."""
    config, plot_cfg = ctx.config, ctx.config["plotting"]
    fig_cfg = config["figures"]
    runs_root = resolve_path(config, config["project"]["runs_dir"])
    run = ctx.args.schedule_run or latest_run(runs_root, "schedule", "schedule_trajectories.csv")
    trajectories = pd.read_csv(Path(run) / "schedule_trajectories.csv")
    capacity = int(fig_cfg["battery_capacity_kwh"])
    trajectories = trajectories[trajectories.capacity_kwh == capacity]

    selections = [
        ("Perfect information (optimum)", (trajectories.scheme == "perfect_lp")),
        (f"GA, month-ahead {NAMES[fig_cfg['battery_arch']]} forecast (seed {fig_cfg['battery_seed']})",
         (trajectories.scheme == "ga") & (trajectories.horizon == "month_ahead") &
         (trajectories.arch == fig_cfg["battery_arch"]) & (trajectories.seed == int(fig_cfg["battery_seed"]))),
        ("Non-scheduled", (trajectories.scheme == "non_scheduled")),
    ]
    actual = daily_totals(ctx.dataset, config["data"]["splits"]["test_start"], config["data"]["splits"]["test_end"])
    no_battery_net = (actual["demand_kwh"] - actual["pv_kwh"]).to_numpy()
    days = np.arange(1, len(no_battery_net) + 1)
    threshold = float(config["scheduling"]["tariff"]["daily_threshold_kwh"])
    styles = ["-", "--", ":"]

    fig, (top, bottom) = new_figure(plot_cfg, tall=True, nrows=2, sharex=True)
    soc_curves = []
    net_curves = [no_battery_net, np.array([threshold])]
    bottom.plot(days, no_battery_net, color="gray", marker="o", markersize=3, label="No battery")
    for (label, mask), style in zip(selections, styles):
        rows = trajectories[mask].sort_values("date")
        top.plot(days, rows["soc_end_kwh"], linestyle=style, marker="o", markersize=3, label=label)
        bottom.plot(days, rows["net_import_kwh"], linestyle=style, label=label)
        soc_curves.append(rows["soc_end_kwh"].to_numpy())
        net_curves.append(rows["net_import_kwh"].to_numpy())
    soc_curves.append(np.array([0.0, capacity]))
    bottom.axhline(threshold, color="red", linewidth=1.0, label=f"Tier threshold ({threshold:.2f} kWh)")

    finalize_axes(top, days, soc_curves, "", "Battery charge (kWh)", plot_cfg,
                  x_ticks=np.arange(0, 31, 5), legend_loc="upper left", headroom=0.45)
    finalize_axes(bottom, days, net_curves, "Day of December 2019", "Daily net grid import (kWh)",
                  plot_cfg, x_ticks=np.arange(0, 31, 5), legend_loc="upper left",
                  legend_ncol=2, headroom=0.45)
    save_figure(fig, ctx.out_dir / "fig_battery_schedule", plot_cfg)


def fig_value_of_accuracy(ctx) -> None:
    """Capture ratio vs random error (top) and vs level bias (bottom); real models marked."""
    config, plot_cfg = ctx.config, ctx.config["plotting"]
    runs_root = resolve_path(config, config["project"]["runs_dir"])
    run = Path(ctx.args.accuracy_run or latest_run(runs_root, "value_of_accuracy", "accuracy_models.csv"))
    random_table = pd.read_csv(run / "accuracy_random.csv")
    bias_table = pd.read_csv(run / "accuracy_bias.csv")
    models = pd.read_csv(run / "accuracy_models.csv")

    fig, (top, bottom) = new_figure(plot_cfg, tall=True, nrows=2)

    # Top: random day-to-day error (GA; the rule-based curve is almost identical).
    curves = [np.array([0.0, 1.0])]
    for target, style in (("demand", "-"), ("pv", "--")):
        subset = random_table[(random_table.error_on == target) & (random_table.scheme == "ga")]
        stats = subset.groupby("sigma")["capture_ratio"].agg(["mean", "std"])
        sigma_percent = 100 * stats.index.to_numpy()
        top.plot(sigma_percent, stats["mean"], linestyle=style, marker="o", markersize=4,
                 label=f"{'Demand' if target == 'demand' else 'PV'} day-to-day error")
        top.fill_between(sigma_percent, stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=0.2)
        curves.extend([(stats["mean"] - stats["std"]).to_numpy(), (stats["mean"] + stats["std"]).to_numpy()])
    low, high = 100 * models["demand_sigma"].min(), 100 * models["demand_sigma"].max()
    top.axvspan(low, high, color="gray", alpha=0.25, label="Month-ahead models (demand)")
    top_max = max(100 * random_table["sigma"].max(), high)
    finalize_axes(top, np.array([0.0, top_max]), curves, "Day-to-day forecast error, sigma (%)",
                  "Capture ratio", plot_cfg, x_ticks=np.arange(0, 5 * np.ceil(top_max / 5) + 1, 5),
                  legend_loc="lower left", include_zero=False)

    # Bottom: level bias.
    curves = [np.array([0.0, 1.0])]
    for target, style in (("demand", "-"), ("pv", "--")):
        subset = bias_table[(bias_table.error_on == target) & (bias_table.scheme == "ga")].sort_values("factor")
        bottom.plot(subset["factor"], subset["capture_ratio"], linestyle=style, marker="o", markersize=4,
                    label=f"{'Demand' if target == 'demand' else 'PV'} level factor")
        curves.append(subset["capture_ratio"].to_numpy())
    low, high = models["pv_factor"].min(), models["pv_factor"].max()
    bottom.axvspan(low, high, color="gray", alpha=0.25, label="Month-ahead models (PV)")
    factors = np.concatenate([bias_table["factor"].to_numpy(), [high]])
    x_end = 0.5 * np.ceil(factors.max() / 0.5)
    finalize_axes(bottom, factors, curves, "Forecast / actual (level factor)", "Capture ratio",
                  plot_cfg, x_ticks=np.arange(0.0, x_end + 0.01, 0.5), legend_loc="upper right",
                  include_zero=False)
    save_figure(fig, ctx.out_dir / "fig_value_of_accuracy", plot_cfg)


def fig_sizing(ctx) -> None:
    """Best-case annual saving vs battery capacity for three export prices."""
    config, plot_cfg = ctx.config, ctx.config["plotting"]
    runs_root = resolve_path(config, config["project"]["runs_dir"])
    run = Path(ctx.args.sizing_run or latest_run(runs_root, "size_battery", "sizing.csv"))
    sizing = pd.read_csv(run / "sizing.csv")
    capacities = sorted(sizing["capacity_kwh"].unique())

    fig, ax = new_figure(plot_cfg)
    curves = []
    for (price, group), style in zip(sizing.groupby("export_price", sort=False), ["-", "--", ":"]):
        group = group.sort_values("capacity_kwh")
        ax.plot(group["capacity_kwh"], group["annual_saving_optimal"], linestyle=style, marker="o",
                label=f"Export price {price:.4g} $/kWh")
        curves.append(group["annual_saving_optimal"].to_numpy())
    finalize_axes(ax, np.array(capacities), curves, "Battery capacity (kWh)",
                  "Best-case annual saving ($/year)", plot_cfg,
                  x_ticks=np.arange(0, max(capacities) + 1, 10), legend_loc="upper left", headroom=0.2)
    save_figure(fig, ctx.out_dir / "fig_sizing_export_price", plot_cfg)


def fig_hourly_profile(ctx) -> None:
    """Average hourly demand and PV over 2019 and over the December test month."""
    config, plot_cfg = ctx.config, ctx.config["plotting"]
    dataset = ctx.dataset
    splits = config["data"]["splits"]
    december = dataset.loc[splits["test_start"]:splits["test_end"]]
    hours = np.arange(24)

    fig, ax = new_figure(plot_cfg)
    curves = []
    for frame, period, style in ((dataset, "2019", "-"), (december, "December 2019", "--")):
        demand = frame["demand_kwh"].groupby(frame.index.hour).mean().to_numpy()
        pv = frame["pv_kwh"].groupby(frame.index.hour).mean().to_numpy()
        ax.plot(hours, demand, linestyle=style, color="tab:blue", marker="o", markersize=3,
                label=f"Demand, {period}")
        ax.plot(hours, pv, linestyle=style, color="tab:orange", marker="s", markersize=3,
                label=f"PV supply, {period}")
        curves.extend([demand, pv])
    finalize_axes(ax, hours, curves, "Hour of day (local standard time)",
                  "Average energy (kWh per hour)", plot_cfg, x_ticks=np.arange(0, 25, 4),
                  legend_loc="upper left", headroom=0.35)
    save_figure(fig, ctx.out_dir / "fig_hourly_profile", plot_cfg)


# ----------------------------------------------------------------- main

FIGURES = {
    "demand_forecast": lambda ctx: fig_forecast(ctx, "demand", "Building energy demand (kWh)",
                                                ctx.config["figures"]["forecast_archs"], "fig_demand_forecast"),
    "pv_forecast": lambda ctx: fig_forecast(ctx, "pv", "PV energy supply (kWh)",
                                            ctx.config["figures"]["forecast_archs"], "fig_pv_forecast"),
    "lead_day": fig_lead_day,
    "horizons": fig_horizons,
    "dwt_ablation": fig_dwt_ablation,
    "pv_drift": fig_pv_drift,
    "battery": fig_battery,
    "value_of_accuracy": fig_value_of_accuracy,
    "sizing": fig_sizing,
    "hourly_profile": fig_hourly_profile,
}


def main() -> None:
    parser = build_parser("Make all paper figures from saved results")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None,
                        help="Make only these figures")
    parser.add_argument("--schedule-run", default=None, help="Run folder of schedule.py (default: latest)")
    parser.add_argument("--accuracy-run", default=None, help="Run folder of value_of_accuracy.py (default: latest)")
    parser.add_argument("--sizing-run", default=None, help="Run folder of size_battery.py (default: latest)")
    args = parser.parse_args()
    ctx = start_run(args, default_name="make_figures")
    config = ctx.config

    apply_paper_style(config["plotting"])
    ctx.args = args
    ctx.out_dir = resolve_path(config, config["figures"]["output_dir"])
    ctx.dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))

    names = args.only or list(FIGURES)
    failed = []
    for name in names:
        try:
            FIGURES[name](ctx)
            ctx.logger.info("Made %s", name)
        except FileNotFoundError as error:
            # A missing result file should not stop the other figures.
            ctx.logger.warning("Skipped %s: %s", name, error)
            failed.append(name)
    ctx.logger.info("Figures saved in %s", ctx.out_dir)
    if failed:
        ctx.logger.warning("Not made (missing inputs): %s", ", ".join(failed))


if __name__ == "__main__":
    main()
