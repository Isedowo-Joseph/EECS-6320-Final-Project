#!/usr/bin/env python3
"""
Clean comparison plots for baseline vs improved approach.

Inputs:
- predictions_three_way.csv with baseline, previous MMD, and improved Fair MMD predictions

Outputs:
- overall TPR/precision plot with 95% bootstrap confidence intervals
- DP gap / EO gap plot with 95% bootstrap confidence intervals
- tidy CSV summaries used by the plots
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEX_MAP = {"1": "Male", "2": "Female"}
RACE_MAP = {"2": "Asian", "3": "Black", "5": "White"}
AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline vs improved confidence-interval graphs.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("MMD_Distribution_Outputs/fair_mmd_tuned_ci/predictions_three_way.csv"),
        help="Path to predictions_three_way.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("MMD_Distribution_Outputs/clean_ci_plots"),
        help="Directory for clean comparison outputs",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=500,
        help="Number of bootstrap resamples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling",
    )
    return parser.parse_args()


def prepare_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sex_label"] = df["applicant_sex"].astype(str).map(SEX_MAP)
    df["race_label"] = df["applicant_race_1"].astype(str).map(RACE_MAP)
    df["applicant_age"] = pd.Categorical(df["applicant_age"], categories=AGE_ORDER, ordered=True)
    return df


def safe_rate(numerator_mask: pd.Series, denominator_mask: pd.Series) -> float:
    denom = int(denominator_mask.sum())
    if denom == 0:
        return np.nan
    return float((numerator_mask & denominator_mask).sum() / denom)


def safe_precision(pred_mask: pd.Series, true_mask: pd.Series) -> float:
    denom = int(pred_mask.sum())
    if denom == 0:
        return np.nan
    return float((pred_mask & true_mask).sum() / denom)


def overall_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    pred_pos = df[pred_col] == 1
    true_pos = df["y_true"] == 1
    return {
        "tpr": safe_rate(pred_pos, true_pos),
        "precision": safe_precision(pred_pos, true_pos),
    }


def group_table(df: pd.DataFrame, pred_col: str, group_col: str) -> pd.DataFrame:
    rows = []
    for group, gdf in df.groupby(group_col, dropna=False, observed=False):
        if pd.isna(group) or len(gdf) == 0:
            continue
        pred_pos = gdf[pred_col] == 1
        true_pos = gdf["y_true"] == 1
        rows.append(
            {
                "group": str(group),
                "selection_rate": float(pred_pos.mean()),
                "tpr": safe_rate(pred_pos, true_pos),
                "precision": safe_precision(pred_pos, true_pos),
                "n": int(len(gdf)),
            }
        )
    out = pd.DataFrame(rows)
    if group_col == "applicant_age":
        out["group"] = pd.Categorical(out["group"], categories=AGE_ORDER, ordered=True)
        out = out.sort_values("group").reset_index(drop=True)
        out["group"] = out["group"].astype(str)
    else:
        out = out.sort_values("group").reset_index(drop=True)
    return out


def fairness_gap_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    metrics = {}
    for attr_name, group_col in [
        ("sex", "sex_label"),
        ("race", "race_label"),
        ("age", "applicant_age"),
    ]:
        tbl = group_table(df, pred_col, group_col)
        dp_values = tbl["selection_rate"].dropna()
        eo_values = tbl["tpr"].dropna()
        metrics[f"{attr_name}_demographic_parity_gap"] = float(dp_values.max() - dp_values.min()) if len(dp_values) else np.nan
        metrics[f"{attr_name}_equal_opportunity_gap"] = float(eo_values.max() - eo_values.min()) if len(eo_values) else np.nan
    return metrics


def bootstrap_summary(
    df: pd.DataFrame,
    pred_cols: dict[str, str],
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = len(df)
    overall_rows = []
    gap_rows = []

    for run_name, pred_col in pred_cols.items():
        overall_samples = {"tpr": [], "precision": []}
        gap_samples: dict[str, list[float]] = {}

        for _ in range(iterations):
            idx = rng.integers(0, n, size=n)
            sample = df.iloc[idx].reset_index(drop=True)

            overall = overall_metrics(sample, pred_col)
            for metric_name, value in overall.items():
                overall_samples[metric_name].append(value)

            gaps = fairness_gap_metrics(sample, pred_col)
            for metric_name, value in gaps.items():
                gap_samples.setdefault(metric_name, []).append(value)

        for metric_name, values in overall_samples.items():
            arr = np.array(values, dtype=float)
            overall_rows.append(
                {
                    "run": run_name,
                    "metric": metric_name,
                    "mean": float(np.nanmean(arr)),
                    "ci_lower_95": float(np.nanquantile(arr, 0.025)),
                    "ci_upper_95": float(np.nanquantile(arr, 0.975)),
                }
            )

        for metric_name, values in gap_samples.items():
            arr = np.array(values, dtype=float)
            gap_rows.append(
                {
                    "run": run_name,
                    "metric": metric_name,
                    "mean": float(np.nanmean(arr)),
                    "ci_lower_95": float(np.nanquantile(arr, 0.025)),
                    "ci_upper_95": float(np.nanquantile(arr, 0.975)),
                }
            )

    return pd.DataFrame(overall_rows), pd.DataFrame(gap_rows)


def yerr_from_ci(mean: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.vstack([np.maximum(mean - lower, 0.0), np.maximum(upper - mean, 0.0)])


def plot_overall_ci(overall_df: pd.DataFrame, outpath: Path) -> None:
    pivot = overall_df.pivot(index="metric", columns="run", values="mean").reset_index()
    base_ci = overall_df[overall_df["run"] == "baseline"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "baseline_lower", "ci_upper_95": "baseline_upper"}
    )
    mmd_ci = overall_df[overall_df["run"] == "previous_mmd"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "mmd_lower", "ci_upper_95": "mmd_upper"}
    )
    fair_ci = overall_df[overall_df["run"] == "improved_fair_mmd"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "improved_lower", "ci_upper_95": "improved_upper"}
    )
    pivot = pivot.merge(base_ci, on="metric").merge(mmd_ci, on="metric").merge(fair_ci, on="metric")
    metric_order = ["tpr", "precision"]
    pivot["metric"] = pd.Categorical(pivot["metric"], categories=metric_order, ordered=True)
    pivot = pivot.sort_values("metric").reset_index(drop=True)

    x = np.arange(len(pivot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width,
        pivot["baseline"],
        width=width,
        color="#1f77b4",
        label="Baseline",
        yerr=yerr_from_ci(
            pivot["baseline"].to_numpy(),
            pivot["baseline_lower"].to_numpy(),
            pivot["baseline_upper"].to_numpy(),
        ),
        capsize=4,
    )
    ax.bar(
        x,
        pivot["previous_mmd"],
        width=width,
        color="#ff7f0e",
        label="Previous MMD",
        yerr=yerr_from_ci(
            pivot["previous_mmd"].to_numpy(),
            pivot["mmd_lower"].to_numpy(),
            pivot["mmd_upper"].to_numpy(),
        ),
        capsize=4,
    )
    ax.bar(
        x + width,
        pivot["improved_fair_mmd"],
        width=width,
        color="#2ca02c",
        label="Improved Fair MMD",
        yerr=yerr_from_ci(
            pivot["improved_fair_mmd"].to_numpy(),
            pivot["improved_lower"].to_numpy(),
            pivot["improved_upper"].to_numpy(),
        ),
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["TPR", "Precision"])
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1)
    ax.set_title("TPR and Precision with 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_gap_ci(gap_df: pd.DataFrame, outpath: Path) -> None:
    metric_order = [
        "sex_demographic_parity_gap",
        "sex_equal_opportunity_gap",
        "race_demographic_parity_gap",
        "race_equal_opportunity_gap",
        "age_demographic_parity_gap",
        "age_equal_opportunity_gap",
    ]
    pivot = gap_df.pivot(index="metric", columns="run", values="mean").reset_index()
    base_ci = gap_df[gap_df["run"] == "baseline"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "baseline_lower", "ci_upper_95": "baseline_upper"}
    )
    mmd_ci = gap_df[gap_df["run"] == "previous_mmd"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "mmd_lower", "ci_upper_95": "mmd_upper"}
    )
    fair_ci = gap_df[gap_df["run"] == "improved_fair_mmd"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "improved_lower", "ci_upper_95": "improved_upper"}
    )
    pivot = pivot.merge(base_ci, on="metric").merge(mmd_ci, on="metric").merge(fair_ci, on="metric")
    pivot["metric"] = pd.Categorical(pivot["metric"], categories=metric_order, ordered=True)
    pivot = pivot.sort_values("metric").reset_index(drop=True)
    label_map = {
        "sex_demographic_parity_gap": "Sex\nDP gap",
        "sex_equal_opportunity_gap": "Sex\nEO gap",
        "race_demographic_parity_gap": "Race\nDP gap",
        "race_equal_opportunity_gap": "Race\nEO gap",
        "age_demographic_parity_gap": "Age\nDP gap",
        "age_equal_opportunity_gap": "Age\nEO gap",
    }
    labels = [label_map[m] for m in pivot["metric"]]

    x = np.arange(len(pivot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        x - width,
        pivot["baseline"],
        width=width,
        color="#1f77b4",
        label="Baseline",
        yerr=yerr_from_ci(
            pivot["baseline"].to_numpy(),
            pivot["baseline_lower"].to_numpy(),
            pivot["baseline_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.bar(
        x,
        pivot["previous_mmd"],
        width=width,
        color="#ff7f0e",
        label="Previous MMD",
        yerr=yerr_from_ci(
            pivot["previous_mmd"].to_numpy(),
            pivot["mmd_lower"].to_numpy(),
            pivot["mmd_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.bar(
        x + width,
        pivot["improved_fair_mmd"],
        width=width,
        color="#2ca02c",
        label="Improved Fair MMD",
        yerr=yerr_from_ci(
            pivot["improved_fair_mmd"].to_numpy(),
            pivot["improved_lower"].to_numpy(),
            pivot["improved_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Gap")
    ax.set_title("Demographic Parity Gap and Equal Opportunity Gap with 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_df(args.predictions)
    pred_cols = {
        "baseline": "baseline_pred",
        "previous_mmd": "mmd_pred",
        "improved_fair_mmd": "fair_mmd_pred",
    }

    overall_df, gap_df = bootstrap_summary(
        df=df,
        pred_cols=pred_cols,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    overall_df.to_csv(args.outdir / "overall_metrics_ci.csv", index=False)
    gap_df.to_csv(args.outdir / "dp_eo_gaps_ci.csv", index=False)

    plot_overall_ci(overall_df, args.outdir / "baseline_mmd_fair_tpr_precision_ci.png")
    plot_gap_ci(gap_df, args.outdir / "baseline_mmd_fair_dp_eo_gap_ci.png")

    print(f"Saved clean CI plots to: {args.outdir.resolve()}")
    print("Overall summary:", (args.outdir / "overall_metrics_ci.csv").resolve())
    print("Gap summary:", (args.outdir / "dp_eo_gaps_ci.csv").resolve())


if __name__ == "__main__":
    main()
