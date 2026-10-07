"""Write the paper's LaTeX tables from the saved results (no numbers typed by hand).

Usage (from the project root):
    python scripts/make_tables.py
    python scripts/make_tables.py --set tables.style=booktabs
Then, in the manuscript:  \\input{tables/tab_forecast_demand.tex}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.evaluation.latex import fmt, fmt_mean_std, latex_table
from efs.forecasting.models import base_architecture, build_model, count_parameters, uses_dwt
from efs.preprocessing.wavelet import WaveletConfig

ARCHS = ["lstm", "dwt_lstm", "tnn", "dwt_tnn"]
NAMES = {"lstm": "LSTM", "dwt_lstm": "DWT-LSTM", "tnn": "TNN", "dwt_tnn": "DWT-TNN",
         "seasonal_naive": "Naive baseline", "-": "--"}


def latest_run(runs_root: Path, suffix: str, required_file: str) -> Path:
    candidates = []
    for folder in sorted(runs_root.iterdir()):
        if folder.is_dir() and folder.name.endswith(f"_{suffix}") and (folder / required_file).exists():
            candidates.append(folder)
    if not candidates:
        raise FileNotFoundError(f"No run folder ending in '_{suffix}' with {required_file}")
    return candidates[-1]


def pick(frame: pd.DataFrame, **conditions) -> pd.Series:
    """The single row matching all conditions (fails loudly if not exactly one)."""
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column] == value
    matches = frame[mask]
    if len(matches) != 1:
        raise KeyError(f"Expected one row for {conditions}, found {len(matches)}")
    return matches.iloc[0]


def ms(row: pd.Series, metric: str, digits: int) -> str:
    return fmt_mean_std(row.get(f"{metric}_mean"), row.get(f"{metric}_std"), digits)

def table_months(config: dict, per_month: pd.DataFrame, pooled: pd.DataFrame, style: str) -> str:
    months = [month["name"] for month in config["extra_months"]["months"]]
    body = []
    for arch in ARCHS + ["seasonal_naive"]:
        strategy = "-" if arch == "seasonal_naive" else "dayblock_recursive"
        row = [NAMES[arch]]
        for month in months:
            match = pick(per_month, month=month, arch=arch, strategy=strategy, horizon="month_ahead")
            row.append(ms(match, "hourly_mape", 1))
        overall = pick(pooled, arch=arch, strategy=strategy, horizon="month_ahead")
        row.extend([ms(overall, "hourly_mape", 1), ms(overall, "daily_mape", 1),
                    ms(overall, "skill_vs_naive", 2)])
        body.append(row)
    header = [["Model"] + [f"{month} MAPE (\\%)" for month in months] +
              ["All months MAPE (\\%)", "All months daily MAPE (\\%)", "All months skill"]]
    return latex_table(
        "Month-ahead demand forecasting on three test months with the frozen models "
        "(day-block recursion; mean $\\pm$ std over seeds, and over seeds and months for the pooled columns).",
        "tab:forecast_months", "l" + "c" * (len(months) + 3), header, body, style,
        notes="Models were trained on January--October 2019 and never updated. January and February 2020 "
              "were not used for any modelling decision. The evaluation stops before the COVID-19 "
              "lockdowns of March 2020.")
# ----------------------------------------------------------------- tables

def table_hyperparameters(config: dict, style: str) -> str:
    tuned_dir = resolve_path(config, config["tuning"]["output_dir"])
    body = []
    for target in ("demand", "pv"):
        for arch in ARCHS:
            with (tuned_dir / f"{target}_{arch}.yaml").open("r", encoding="utf-8") as handle:
                settings = yaml.safe_load(handle)
            params = settings["model_params"]
            if base_architecture(arch) == "lstm":
                size = f"hidden {params['hidden_size']}, layers {params['num_layers']}"
            else:
                size = (f"$d$ {params['d_model']}, heads {params['num_heads']}, "
                        f"layers {params['num_layers']}, FF {params['ff_dim']}")
            n_features = len(config["forecasting"]["calendar_features"][target])
            model = build_model(arch, n_features, 24, params)
            dwt = "--"
            if uses_dwt(arch):
                wavelet = WaveletConfig.from_config(config["dwt"], target)
                dwt = f"{wavelet.wavelet}, L{wavelet.level}"
            body.append([
                "Demand" if target == "demand" else "PV", NAMES[arch], str(settings["window_hours"]),
                size, fmt(params["dropout"], 2), str(params["mlp_units"]),
                f"{settings['training']['learning_rate']:.1e}", f"{settings['training']['weight_decay']:.1e}",
                dwt, f"{count_parameters(model):,}".replace(",", "{,}"),
            ])
        if target == "demand":
            body.append(["MIDRULE"])
    header = [["Target", "Model", "Window (h)", "Network size", "Dropout", "MLP units",
               "Learning rate", "Weight decay", "DWT", "Parameters"]]
    return latex_table(
        "Selected hyperparameters (30 TPE trials per model, objective: November month-ahead error).",
        "tab:hyperparameters", "llrlrrrrlr", header, body, style,
        notes="Transformer models use mean-and-last-token pooling; all models use day-block recursion "
              "(24 h per step). Parameter counts are for the 24-hour output head.")


def table_forecast_demand(config: dict, summary: pd.DataFrame, style: str) -> str:
    demand = summary[summary.target == "demand"]
    body = []
    for arch in ARCHS + ["seasonal_naive"]:
        if arch == "seasonal_naive":
            month = pick(demand, arch=arch, horizon="month_ahead")
            week = pick(demand, arch=arch, horizon="week_ahead")
            hour = pick(demand, arch=arch, horizon="one_hour_ahead")
        else:
            month = pick(demand, arch=arch, strategy="dayblock_recursive", horizon="month_ahead")
            week = pick(demand, arch=arch, strategy="dayblock_recursive", horizon="week_ahead")
            hour = pick(demand, arch=arch, strategy="hourly_recursive", horizon="one_hour_ahead")
        body.append([NAMES[arch], ms(hour, "hourly_mape", 1), ms(week, "hourly_mape", 1),
                     ms(month, "hourly_mape", 1), ms(month, "daily_mape", 1),
                     ms(month, "hourly_rmse", 3), ms(month, "hourly_r2", 2),
                     ms(month, "skill_vs_naive", 2)])
    header = [["Model", "1-h MAPE (\\%)", "Week MAPE (\\%)", "Month MAPE (\\%)",
               "Month daily MAPE (\\%)", "Month RMSE (kWh)", "Month $R^2$", "Month skill"]]
    return latex_table(
        "December 2019 demand forecasting (mean $\\pm$ std over 5 seeds).",
        "tab:forecast_demand", "lccccccc", header, body, style,
        notes="Week and month: day-block recursive forecasts from a fixed origin (weekly origins "
              "refreshed with observed data). 1-h: hour-by-hour models with observed inputs. "
              "Skill = 1 - RMSE/RMSE of the naive baseline with the same information.")


def table_forecast_pv(config: dict, summary: pd.DataFrame, style: str) -> str:
    pv = summary[summary.target == "pv"]
    body = []
    for arch in ARCHS + ["seasonal_naive"]:
        if arch == "seasonal_naive":
            month = pick(pv, arch=arch, horizon="month_ahead")
            week = pick(pv, arch=arch, horizon="week_ahead")
            hour = pick(pv, arch=arch, horizon="one_hour_ahead")
        else:
            month = pick(pv, arch=arch, strategy="dayblock_recursive", horizon="month_ahead")
            week = pick(pv, arch=arch, strategy="dayblock_recursive", horizon="week_ahead")
            hour = pick(pv, arch=arch, strategy="hourly_recursive", horizon="one_hour_ahead")
        body.append([NAMES[arch], ms(month, "daily_mae", 2), ms(month, "daylight_nmae", 1),
                     ms(month, "daylight_r2", 2), ms(month, "skill_vs_naive", 2),
                     ms(week, "skill_vs_naive", 2), ms(hour, "skill_vs_naive", 2)])
    header = [["Model", "Month daily MAE (kWh)", "Month daylight nMAE (\\%)", "Month daylight $R^2$",
               "Month skill", "Week skill", "1-h skill"]]
    return latex_table(
        "December 2019 PV forecasting (mean $\\pm$ std over 5 seeds).",
        "tab:forecast_pv", "lcccccc", header, body, style,
        notes="Daylight metrics exclude hours with observed output below "
              f"{config['evaluation']['daylight_threshold_kwh']} kWh; MAPE is not reported "
              "because it is undefined at night.")


def table_dwt_ablation(config: dict, test: pd.DataFrame, val: pd.DataFrame, final: pd.DataFrame,
                       target: str, style: str) -> str:
    metric = "hourly_mape" if target == "demand" else "daylight_nmae"
    metric_label = "MAPE" if target == "demand" else "daylight nMAE"
    body = []
    for wavelet in config["ablation"]["wavelets"]:
        for level in config["ablation"]["levels"]:
            variant = f"{wavelet}_L{level}_{config['ablation']['method']}"
            row = [wavelet, str(level)]
            for arch in config["ablation"]["architectures"]:
                val_row = pick(val, target=target, arch=arch, variant=variant)
                test_row = pick(test, target=target, arch=arch, variant=variant, horizon="month_ahead")
                row.append(fmt_mean_std(val_row["objective_mean"], val_row["objective_std"], 1))
                row.append(ms(test_row, metric, 1))
            body.append(row)
    body.append(["MIDRULE"])
    row = ["No DWT", "--"]
    for arch in config["ablation"]["architectures"]:
        plain = base_architecture(arch)
        test_row = pick(final, target=target, arch=plain, strategy="dayblock_recursive", horizon="month_ahead")
        row.extend(["--", ms(test_row, metric, 1)])
    body.append(row)

    arch_names = [NAMES[arch] for arch in config["ablation"]["architectures"]]
    header = [["Wavelet", "Level"] + [f"\\multicolumn{{2}}{{c}}{{{name}}}" for name in arch_names],
              ["", ""] + ["Nov (val.)", "Dec (test)"] * len(arch_names)]
    chosen = WaveletConfig.from_config(config["dwt"], target)
    title = "demand" if target == "demand" else "PV"
    return latex_table(
        f"DWT ablation for {title} (month-ahead {metric_label} in \\%, mean $\\pm$ std over "
        f"{len(config['ablation']['seeds'])} seeds).",
        f"tab:dwt_ablation_{target}", "ll" + "cc" * len(arch_names), header, body, style,
        notes=f"The configuration used in the final models ({chosen.wavelet}, level {chosen.level}) "
              "was selected on November only. No-DWT rows: final LSTM/TNN models (5 seeds).")


def table_scheduling(config: dict, schedule: pd.DataFrame, style: str) -> str:
    capacities = [int(capacity) for capacity in config["scheduling"]["battery"]["capacities_kwh"]]
    fixed = [("no_battery", "-", "-", "No battery", "--"),
             ("non_scheduled", "-", "-", "Non-scheduled", "--"),
             ("perfect_lp", "actual", "-", "Perfect information", "Actual")]
    body = []
    for scheme, horizon, arch, label, forecast in fixed:
        row = [label, forecast, "--"]
        for capacity in capacities:
            match = pick(schedule, capacity_kwh=capacity, scheme=scheme, horizon=horizon, arch=arch)
            row.extend([fmt(match["savings_mean"], 2), fmt(match["capture_ratio_mean"], 2)])
        body.append(row)
    body.append(["MIDRULE"])
    for scheme, label in (("ga", "GA"), ("rule_based", "Rule-based")):
        for horizon, horizon_label in (("month_ahead", "Month"), ("week_ahead", "Week")):
            for arch in config["scheduling"]["forecast_architectures"]:
                row = [label, NAMES[arch], horizon_label]
                for capacity in capacities:
                    match = pick(schedule, capacity_kwh=capacity, scheme=scheme, horizon=horizon, arch=arch)
                    row.extend([ms(match, "savings", 2), fmt(match["capture_ratio_mean"], 2)])
                body.append(row)
    header = [["Scheduler", "Forecast", "Horizon"] +
              [f"\\multicolumn{{2}}{{c}}{{{capacity} kWh}}" for capacity in capacities],
              ["", "", ""] + ["Saving (\\$)", "Capture"] * len(capacities)]
    return latex_table(
        "December 2019 scheduling, scored on observed demand and PV: saving relative to no battery "
        "and capture ratio (share of the perfect-information saving).",
        "tab:scheduling", "lll" + "cc" * len(capacities), header, body, style, font_size=r"\footnotesize",
        notes="Forecast-driven rows: mean $\\pm$ std over 5 seeds of day-block forecasts. "
              "The GA with perfect information reached the LP optimum exactly at every capacity.")


def table_sizing(config: dict, sizing: pd.DataFrame, style: str) -> str:
    costs = [int(cost) for cost in config["economics"]["battery_cost_per_kwh"]]
    body = []
    previous_price = None
    for _, row in sizing.iterrows():
        if previous_price is not None and row["export_price"] != previous_price:
            body.append(["MIDRULE"])
        previous_price = row["export_price"]
        line = [fmt(row["export_price"], 4), str(int(row["capacity_kwh"])),
                fmt(row["annual_saving_optimal"], 2), fmt(row["annual_saving_non_scheduled"], 2),
                fmt(row["break_even_cost_per_kwh"], 2)]
        for cost in costs:
            line.append(f"{row[f'npv_at_{cost}']:,.0f}".replace(",", "{,}"))
        body.append(line)
    econ = config["economics"]
    header = [["Export (\\$/kWh)", "Capacity (kWh)", "Best saving (\\$/yr)", "Non-sched. (\\$/yr)",
               "Break-even (\\$/kWh)"] + [f"NPV @ \\${cost}/kWh" for cost in costs]]
    return latex_table(
        "Battery economics from the perfect-information optimum on January--November 2019 "
        "(annualized upper bound on savings).",
        "tab:sizing", "rr" + "r" * (3 + len(costs)), header, body, style,
        notes=f"Lifetime {econ['lifetime_years']} years, discount rate {100 * econ['discount_rate']:.0f}\\%, "
              f"O\\&M \\${econ['om_per_year']:.0f}/year. Because savings are upper bounds, negative NPV "
              "means no scheduler could make the battery pay back under this tariff.")


# ----------------------------------------------------------------- main

def main() -> None:
    parser = build_parser("Write LaTeX tables from saved results")
    parser.add_argument("--schedule-run", default=None)
    parser.add_argument("--sizing-run", default=None)
    args = parser.parse_args()
    ctx = start_run(args, default_name="make_tables")
    config = ctx.config
    style = config["tables"]["style"]
    out_dir = resolve_path(config, config["tables"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = resolve_path(config, config["project"]["runs_dir"])

    final_summary = pd.read_csv(resolve_path(config, config["final"]["output_dir"]) / "test_summary.csv")
    ablation_root = resolve_path(config, config["ablation"]["output_dir"])
    schedule_run = Path(args.schedule_run or latest_run(runs_root, "schedule", "schedule_summary.csv"))
    sizing_run = Path(args.sizing_run or latest_run(runs_root, "size_battery", "sizing.csv"))

    builders = {
        "tab_hyperparameters": lambda: table_hyperparameters(config, style),
        "tab_forecast_demand": lambda: table_forecast_demand(config, final_summary, style),
        "tab_forecast_pv": lambda: table_forecast_pv(config, final_summary, style),
        "tab_dwt_ablation_demand": lambda: table_dwt_ablation(
            config, pd.read_csv(ablation_root / "test_summary.csv"),
            pd.read_csv(ablation_root / "val_summary.csv"), final_summary, "demand", style),
        "tab_dwt_ablation_pv": lambda: table_dwt_ablation(
            config, pd.read_csv(ablation_root / "test_summary.csv"),
            pd.read_csv(ablation_root / "val_summary.csv"), final_summary, "pv", style),
        "tab_scheduling": lambda: table_scheduling(config, pd.read_csv(schedule_run / "schedule_summary.csv"), style),
        "tab_sizing": lambda: table_sizing(config, pd.read_csv(sizing_run / "sizing.csv"), style),
    }
    months_dir = resolve_path(config, config["extra_months"]["output_dir"])
    if (months_dir / "months_summary.csv").exists():
        builders["tab_forecast_months"] = lambda: table_months(
            config, pd.read_csv(months_dir / "months_summary.csv"),
            pd.read_csv(months_dir / "months_pooled.csv"), style)
    else:
        ctx.logger.warning("No %s yet: skipping the multi-month table", months_dir / "months_summary.csv")
    for name, build in builders.items():
        text = build()
        (out_dir / f"{name}.tex").write_text(text, encoding="utf-8")
        ctx.logger.info("Wrote %s", out_dir / f"{name}.tex")
    ctx.logger.info("Sources: schedule %s | sizing %s", schedule_run.name, sizing_run.name)


if __name__ == "__main__":
    main()
