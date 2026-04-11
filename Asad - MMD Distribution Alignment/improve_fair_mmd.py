#!/usr/bin/env python3
"""
Train and compare three variants:
1. Baseline classifier
2. Original test-alignment MMD classifier
3. Fairness-aware group-conditional MMD classifier

The improved variant removes test-feature alignment from training and instead:
- aligns hidden features across sensitive groups within each label class
- penalizes group score disparities directly on the training batch
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from analyze_mmd_alignment import (
    Model,
    RunOutputs,
    SENSITIVE_COLS,
    SEED,
    evaluate_predictions,
    fairness_gap_summary,
    fairness_table,
    load_clean_data,
    make_eval_frame,
    multi_kernel_mmd,
    plot_calibration_curve,
    plot_confidence_margin,
    plot_training_losses,
    preprocess,
    save_run_artifacts,
    set_seed,
    split_data,
    train_baseline,
    train_mmd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improve fairness with group-aware MMD alignment.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("MMD_Distribution_Outputs/q2_cleaned_dataset.csv"),
        help="Path to the cleaned dataset CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("MMD_Distribution_Outputs/fair_mmd"),
        help="Directory for outputs.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer size.")
    parser.add_argument(
        "--lambda-mmd",
        type=float,
        default=0.2,
        help="Weight for group-conditional hidden-state MMD.",
    )
    parser.add_argument(
        "--lambda-fair",
        type=float,
        default=0.3,
        help="Weight for score-level fairness penalty.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=30,
        help="Bootstrap iterations for confidence intervals.",
    )
    return parser.parse_args()


def encode_sensitive(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    encoded: dict[str, np.ndarray] = {}
    for col in SENSITIVE_COLS:
        categories = sorted(train_df[col].astype(str).unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
        encoded[col] = train_df[col].astype(str).map(mapping).to_numpy(dtype=np.int64)
    return encoded


def make_fair_loader(
    features: np.ndarray,
    labels: np.ndarray,
    encoded_sensitive: dict[str, np.ndarray],
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(encoded_sensitive["applicant_sex"], dtype=torch.long),
        torch.tensor(encoded_sensitive["applicant_race_1"], dtype=torch.long),
        torch.tensor(encoded_sensitive["applicant_age"], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def pairwise_gap_loss(group_means: list[torch.Tensor]) -> torch.Tensor:
    if len(group_means) < 2:
        return torch.tensor(0.0, device=group_means[0].device if group_means else "cpu")
    losses = []
    for idx in range(len(group_means)):
        for jdx in range(idx + 1, len(group_means)):
            losses.append((group_means[idx] - group_means[jdx]).pow(2))
    return torch.stack(losses).mean()


def score_gap_loss(
    scores: torch.Tensor,
    group_codes: torch.Tensor,
    mask: torch.Tensor,
    min_count: int = 8,
) -> torch.Tensor:
    active_mask = mask.bool()
    device = scores.device
    if int(active_mask.sum().item()) == 0:
        return torch.tensor(0.0, device=device)

    means: list[torch.Tensor] = []
    for group_id in torch.unique(group_codes[active_mask]):
        group_mask = active_mask & (group_codes == group_id)
        if int(group_mask.sum().item()) < min_count:
            continue
        means.append(scores[group_mask].mean())
    return pairwise_gap_loss(means)


def fairness_penalty(
    scores: torch.Tensor,
    labels: torch.Tensor,
    sensitive_batches: Iterable[torch.Tensor],
) -> torch.Tensor:
    device = scores.device
    total_losses = []
    ones_mask = torch.ones_like(labels, dtype=torch.bool, device=device)
    positive_mask = labels >= 0.5
    negative_mask = ~positive_mask

    for group_codes in sensitive_batches:
        total_losses.append(score_gap_loss(scores, group_codes, ones_mask))
        total_losses.append(score_gap_loss(scores, group_codes, positive_mask))
        total_losses.append(score_gap_loss(scores, group_codes, negative_mask))

    valid_losses = [loss for loss in total_losses if torch.isfinite(loss)]
    if not valid_losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(valid_losses).mean()


def conditional_group_mmd(
    features: torch.Tensor,
    labels: torch.Tensor,
    sensitive_batches: Iterable[torch.Tensor],
    min_count: int = 8,
    max_samples: int = 32,
) -> torch.Tensor:
    device = features.device
    losses = []

    for group_codes in sensitive_batches:
        for label_value in (0.0, 1.0):
            label_mask = labels == label_value
            grouped_features = []
            for group_id in torch.unique(group_codes[label_mask]):
                group_mask = label_mask & (group_codes == group_id)
                count = int(group_mask.sum().item())
                if count < min_count:
                    continue
                group_features = features[group_mask]
                if count > max_samples:
                    perm = torch.randperm(count, device=device)[:max_samples]
                    group_features = group_features[perm]
                grouped_features.append(group_features)

            if len(grouped_features) < 2:
                continue

            for idx in range(len(grouped_features)):
                for jdx in range(idx + 1, len(grouped_features)):
                    losses.append(multi_kernel_mmd(grouped_features[idx], grouped_features[jdx]))

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def train_fair_mmd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    sensitive_train: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> RunOutputs:
    set_seed(SEED)
    model = Model(x_train.shape[1], hidden_dim=args.hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    train_loader = make_fair_loader(x_train, y_train, sensitive_train, args.batch_size, SEED)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        last_cls = 0.0
        last_mmd = 0.0
        last_fair = 0.0
        last_total = 0.0

        for x_batch, y_batch, sex_batch, race_batch, age_batch in train_loader:
            optimizer.zero_grad()
            scores = model(x_batch).squeeze()
            cls_loss = criterion(scores, y_batch)
            features = model.encode(x_batch)
            sensitive_batches = [sex_batch, race_batch, age_batch]
            feature_mmd = conditional_group_mmd(features, y_batch, sensitive_batches)
            fair_loss = fairness_penalty(scores, y_batch, sensitive_batches)
            total_loss = cls_loss + args.lambda_mmd * feature_mmd + args.lambda_fair * fair_loss
            total_loss.backward()
            optimizer.step()

            last_cls = float(cls_loss.item())
            last_mmd = float(feature_mmd.item())
            last_fair = float(fair_loss.item())
            last_total = float(total_loss.item())

        history.append(
            {
                "epoch": epoch,
                "loss": last_total,
                "classification_loss": last_cls,
                "group_mmd_loss": last_mmd,
                "fairness_loss": last_fair,
            }
        )

    model.eval()
    with torch.no_grad():
        test_scores = model(x_test_tensor).squeeze().cpu().numpy()
    test_preds = (test_scores >= 0.5).astype(int)
    return RunOutputs(name="fair_mmd", scores=test_scores, preds=test_preds, history=history)


def bootstrap_cis(y_true: np.ndarray, scores: np.ndarray, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_names = ["accuracy", "balanced_accuracy", "roc_auc", "brier_score", "mean_confidence"]
    n = len(y_true)
    rows = []
    for metric in metric_names:
        values = []
        for _ in range(iterations):
            idx = rng.integers(0, n, size=n)
            metrics = evaluate_predictions(y_true[idx], scores[idx])
            values.append(metrics[metric])
        values_arr = np.array(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "mean": float(values_arr.mean()),
                "ci_lower_95": float(np.quantile(values_arr, 0.025)),
                "ci_upper_95": float(np.quantile(values_arr, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_fairness_cis(eval_df: pd.DataFrame, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = len(eval_df)
    gap_rows = []
    selection_rows = []

    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        sample = eval_df.iloc[idx].reset_index(drop=True)
        for attribute, group_col in [
            ("sex", "sex_label"),
            ("race", "race_label"),
            ("age", "applicant_age"),
        ]:
            table = fairness_table(sample, group_col)
            summary = fairness_gap_summary(table)
            for metric_name, value in summary.items():
                gap_rows.append(
                    {
                        "metric": f"{attribute}_{metric_name}",
                        "value": float(value),
                    }
                )
            for _, row in table.iterrows():
                selection_rows.append(
                    {
                        "attribute": attribute,
                        "group": row["group"],
                        "metric": "selection_rate",
                        "value": float(row["selection_rate"]),
                    }
                )

    gap_df = pd.DataFrame(gap_rows)
    gap_ci = (
        gap_df.groupby("metric")["value"]
        .agg(
            mean="mean",
            ci_lower_95=lambda x: float(np.quantile(x, 0.025)),
            ci_upper_95=lambda x: float(np.quantile(x, 0.975)),
        )
        .reset_index()
    )

    selection_df = pd.DataFrame(selection_rows)
    selection_ci = (
        selection_df.groupby(["attribute", "group", "metric"])["value"]
        .agg(
            mean="mean",
            ci_lower_95=lambda x: float(np.quantile(x, 0.025)),
            ci_upper_95=lambda x: float(np.quantile(x, 0.975)),
        )
        .reset_index()
    )
    return gap_ci, selection_ci


def build_yerr(values: np.ndarray, lowers: np.ndarray, uppers: np.ndarray) -> np.ndarray:
    lower_err = np.maximum(values - lowers, 0.0)
    upper_err = np.maximum(uppers - values, 0.0)
    return np.vstack([lower_err, upper_err])


def plot_three_way_metrics(comparison_df: pd.DataFrame, ci_df: pd.DataFrame, outpath: Path) -> None:
    metric_order = ["accuracy", "balanced_accuracy", "roc_auc", "brier_score", "mean_confidence"]
    subset = comparison_df[comparison_df["metric"].isin(metric_order)].copy()
    baseline_ci = ci_df[ci_df["run"] == "baseline"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "baseline_ci_lower", "ci_upper_95": "baseline_ci_upper"}
    )
    mmd_ci = ci_df[ci_df["run"] == "mmd"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "mmd_ci_lower", "ci_upper_95": "mmd_ci_upper"}
    )
    fair_ci = ci_df[ci_df["run"] == "fair_mmd"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "fair_ci_lower", "ci_upper_95": "fair_ci_upper"}
    )
    subset = subset.merge(baseline_ci, on="metric").merge(mmd_ci, on="metric").merge(fair_ci, on="metric")
    x = np.arange(len(subset))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - width,
        subset["baseline"],
        width=width,
        label="Baseline",
        color="#1f77b4",
        yerr=build_yerr(
            subset["baseline"].to_numpy(),
            subset["baseline_ci_lower"].to_numpy(),
            subset["baseline_ci_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.bar(
        x,
        subset["mmd"],
        width=width,
        label="Current MMD",
        color="#ff7f0e",
        yerr=build_yerr(
            subset["mmd"].to_numpy(),
            subset["mmd_ci_lower"].to_numpy(),
            subset["mmd_ci_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.bar(
        x + width,
        subset["fair_mmd"],
        width=width,
        label="Improved Fair MMD",
        color="#2ca02c",
        yerr=build_yerr(
            subset["fair_mmd"].to_numpy(),
            subset["fair_ci_lower"].to_numpy(),
            subset["fair_ci_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subset["metric"], rotation=20)
    ax.set_ylabel("Value")
    ax.set_title("Performance comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_confidence_histogram_two_runs(
    baseline_scores: pd.Series,
    fair_scores: pd.Series,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].hist(baseline_scores, bins=30, alpha=0.8, color="#1f77b4")
    axes[0].set_title("Baseline approval probability")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")

    axes[1].hist(fair_scores, bins=30, alpha=0.8, color="#2ca02c")
    axes[1].set_title("Improved Fair MMD approval probability")
    axes[1].set_xlabel("Predicted probability")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_three_way_gaps(comparison_df: pd.DataFrame, gap_ci_df: pd.DataFrame, outpath: Path) -> None:
    subset = comparison_df[comparison_df["metric"].str.endswith("_gap")].copy()
    subset["label"] = subset["metric"].str.replace("_", " ")
    baseline_ci = gap_ci_df[gap_ci_df["run"] == "baseline"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "baseline_ci_lower", "ci_upper_95": "baseline_ci_upper"}
    )
    mmd_ci = gap_ci_df[gap_ci_df["run"] == "mmd"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "mmd_ci_lower", "ci_upper_95": "mmd_ci_upper"}
    )
    fair_ci = gap_ci_df[gap_ci_df["run"] == "fair_mmd"][["metric", "ci_lower_95", "ci_upper_95"]].rename(
        columns={"ci_lower_95": "fair_ci_lower", "ci_upper_95": "fair_ci_upper"}
    )
    subset = subset.merge(baseline_ci, on="metric").merge(mmd_ci, on="metric").merge(fair_ci, on="metric")
    x = np.arange(len(subset))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        x - width,
        subset["baseline"],
        width=width,
        label="Baseline",
        color="#1f77b4",
        yerr=build_yerr(
            subset["baseline"].to_numpy(),
            subset["baseline_ci_lower"].to_numpy(),
            subset["baseline_ci_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.bar(
        x,
        subset["mmd"],
        width=width,
        label="Current MMD",
        color="#ff7f0e",
        yerr=build_yerr(
            subset["mmd"].to_numpy(),
            subset["mmd_ci_lower"].to_numpy(),
            subset["mmd_ci_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.bar(
        x + width,
        subset["fair_mmd"],
        width=width,
        label="Improved Fair MMD",
        color="#2ca02c",
        yerr=build_yerr(
            subset["fair_mmd"].to_numpy(),
            subset["fair_ci_lower"].to_numpy(),
            subset["fair_ci_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subset["label"], rotation=35, ha="right")
    ax.set_ylabel("Gap size")
    ax.set_title("Fairness gap comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_group_selection_rates_with_ci(
    baseline_tbl: pd.DataFrame,
    fair_tbl: pd.DataFrame,
    baseline_ci: pd.DataFrame,
    fair_ci: pd.DataFrame,
    title: str,
    outpath: Path,
) -> None:
    merged = baseline_tbl[["group", "selection_rate"]].merge(
        fair_tbl[["group", "selection_rate"]],
        on="group",
        suffixes=("_baseline", "_fair"),
    )
    merged = merged.merge(
        baseline_ci[["group", "ci_lower_95", "ci_upper_95"]].rename(
            columns={"ci_lower_95": "baseline_ci_lower", "ci_upper_95": "baseline_ci_upper"}
        ),
        on="group",
    ).merge(
        fair_ci[["group", "ci_lower_95", "ci_upper_95"]].rename(
            columns={"ci_lower_95": "fair_ci_lower", "ci_upper_95": "fair_ci_upper"}
        ),
        on="group",
    )
    x = np.arange(len(merged))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        x - width / 2,
        merged["selection_rate_baseline"],
        width=width,
        label="Baseline",
        color="#1f77b4",
        yerr=build_yerr(
            merged["selection_rate_baseline"].to_numpy(),
            merged["baseline_ci_lower"].to_numpy(),
            merged["baseline_ci_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        merged["selection_rate_fair"],
        width=width,
        label="Improved Fair MMD",
        color="#2ca02c",
        yerr=build_yerr(
            merged["selection_rate_fair"].to_numpy(),
            merged["fair_ci_lower"].to_numpy(),
            merged["fair_ci_upper"].to_numpy(),
        ),
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(merged["group"], rotation=25)
    ax.set_title(title)
    ax.set_ylabel("Selection rate")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_clean_data(args.data)
    x_train, x_test, y_train, y_test = split_data(df)
    x_train_processed, x_test_processed = preprocess(x_train, x_test)
    sensitive_train = encode_sensitive(x_train)

    print("Training baseline...")
    baseline_outputs = train_baseline(x_train_processed, y_train.to_numpy(), x_test_processed, args)

    print("Training current MMD...")
    current_mmd_outputs = train_mmd(
        x_train_processed,
        y_train.to_numpy(),
        x_test_processed,
        y_test.to_numpy(),
        args,
    )

    print("Training improved Fair MMD...")
    fair_mmd_outputs = train_fair_mmd(
        x_train_processed,
        y_train.to_numpy(),
        x_test_processed,
        sensitive_train,
        args,
    )

    runs = {
        "baseline": baseline_outputs,
        "mmd": current_mmd_outputs,
        "fair_mmd": fair_mmd_outputs,
    }

    metrics_by_run: dict[str, dict[str, float]] = {}
    eval_by_run: dict[str, pd.DataFrame] = {}
    for run_name, outputs in runs.items():
        eval_df = make_eval_frame(x_test, y_test, outputs)
        eval_by_run[run_name] = eval_df
        metrics = evaluate_predictions(y_test.to_numpy(), outputs.scores)
        metrics_by_run[run_name] = save_run_artifacts(args.outdir, outputs, eval_df, metrics)

    comparison = (
        pd.DataFrame(
            {
                "baseline": pd.Series(metrics_by_run["baseline"]),
                "mmd": pd.Series(metrics_by_run["mmd"]),
                "fair_mmd": pd.Series(metrics_by_run["fair_mmd"]),
            }
        )
        .rename_axis("metric")
        .reset_index()
    )
    comparison["mmd_minus_baseline"] = comparison["mmd"] - comparison["baseline"]
    comparison["fair_mmd_minus_baseline"] = comparison["fair_mmd"] - comparison["baseline"]
    comparison["fair_mmd_minus_mmd"] = comparison["fair_mmd"] - comparison["mmd"]
    comparison.to_csv(args.outdir / "comparison_metrics.csv", index=False)

    predictions = x_test[SENSITIVE_COLS].copy()
    predictions["y_true"] = y_test.to_numpy()
    for run_name, outputs in runs.items():
        predictions[f"{run_name}_score"] = outputs.scores
        predictions[f"{run_name}_pred"] = outputs.preds
        predictions[f"{run_name}_confidence"] = np.maximum(outputs.scores, 1.0 - outputs.scores)
    predictions["fair_minus_baseline_score"] = predictions["fair_mmd_score"] - predictions["baseline_score"]
    predictions["fair_minus_baseline_confidence"] = (
        predictions["fair_mmd_confidence"] - predictions["baseline_confidence"]
    )
    predictions.to_csv(args.outdir / "predictions_three_way.csv", index=False)

    ci_frames = []
    gap_ci_frames = []
    selection_ci_frames = []
    for idx, (run_name, outputs) in enumerate(runs.items()):
        ci_df = bootstrap_cis(y_test.to_numpy(), outputs.scores, args.bootstrap_iterations, seed=SEED + idx)
        ci_df.insert(0, "run", run_name)
        ci_frames.append(ci_df)
        gap_ci_df, selection_ci_df = bootstrap_fairness_cis(eval_by_run[run_name], args.bootstrap_iterations, seed=SEED + 10 + idx)
        gap_ci_df.insert(0, "run", run_name)
        selection_ci_df.insert(0, "run", run_name)
        gap_ci_frames.append(gap_ci_df)
        selection_ci_frames.append(selection_ci_df)
    metric_ci_df = pd.concat(ci_frames, ignore_index=True)
    metric_ci_df.to_csv(args.outdir / "bootstrap_metric_cis.csv", index=False)
    gap_ci_df = pd.concat(gap_ci_frames, ignore_index=True)
    gap_ci_df.to_csv(args.outdir / "bootstrap_gap_cis.csv", index=False)
    selection_ci_df = pd.concat(selection_ci_frames, ignore_index=True)
    selection_ci_df.to_csv(args.outdir / "bootstrap_selection_rate_cis.csv", index=False)

    baseline_history = pd.DataFrame(baseline_outputs.history)
    current_mmd_history = pd.DataFrame(current_mmd_outputs.history)
    fair_mmd_history = pd.DataFrame(fair_mmd_outputs.history)

    plot_training_losses(baseline_history, current_mmd_history, args.outdir / "training_losses_baseline_vs_mmd.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fair_mmd_history["epoch"], fair_mmd_history["loss"], label="Fair MMD total", linewidth=2)
    ax.plot(
        fair_mmd_history["epoch"],
        fair_mmd_history["classification_loss"],
        label="Fair MMD BCE",
        linewidth=2,
    )
    ax.plot(fair_mmd_history["epoch"], fair_mmd_history["group_mmd_loss"], label="Group MMD", linewidth=2)
    ax.plot(fair_mmd_history["epoch"], fair_mmd_history["fairness_loss"], label="Fairness penalty", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Improved Fair MMD training losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "training_losses_fair_mmd.png", dpi=200)
    plt.close(fig)

    plot_confidence_histogram_two_runs(
        predictions["baseline_score"],
        predictions["fair_mmd_score"],
        args.outdir / "confidence_histograms_baseline_vs_fair.png",
    )
    fair_pred_df = pd.DataFrame(
        {
            "baseline_confidence": predictions["baseline_confidence"],
            "mmd_confidence": predictions["fair_mmd_confidence"],
        }
    )
    plot_confidence_margin(fair_pred_df, args.outdir / "confidence_comparison_baseline_vs_fair.png")

    calibration_df = pd.DataFrame(
        {
            "y_true": predictions["y_true"],
            "baseline_score": predictions["baseline_score"],
            "mmd_score": predictions["fair_mmd_score"],
        }
    )
    plot_calibration_curve(calibration_df, args.outdir / "calibration_curve_baseline_vs_fair.png")

    plot_three_way_metrics(comparison, metric_ci_df, args.outdir / "performance_three_way.png")
    plot_three_way_gaps(comparison, gap_ci_df, args.outdir / "fairness_gaps_three_way.png")

    plot_group_selection_rates_with_ci(
        pd.read_csv(args.outdir / "baseline/fairness_by_sex.csv"),
        pd.read_csv(args.outdir / "fair_mmd/fairness_by_sex.csv"),
        selection_ci_df[(selection_ci_df["run"] == "baseline") & (selection_ci_df["attribute"] == "sex")],
        selection_ci_df[(selection_ci_df["run"] == "fair_mmd") & (selection_ci_df["attribute"] == "sex")],
        "Selection rate by sex: baseline vs improved Fair MMD",
        args.outdir / "selection_rate_by_sex_baseline_vs_fair.png",
    )
    plot_group_selection_rates_with_ci(
        pd.read_csv(args.outdir / "baseline/fairness_by_race.csv"),
        pd.read_csv(args.outdir / "fair_mmd/fairness_by_race.csv"),
        selection_ci_df[(selection_ci_df["run"] == "baseline") & (selection_ci_df["attribute"] == "race")],
        selection_ci_df[(selection_ci_df["run"] == "fair_mmd") & (selection_ci_df["attribute"] == "race")],
        "Selection rate by race: baseline vs improved Fair MMD",
        args.outdir / "selection_rate_by_race_baseline_vs_fair.png",
    )
    plot_group_selection_rates_with_ci(
        pd.read_csv(args.outdir / "baseline/fairness_by_age.csv"),
        pd.read_csv(args.outdir / "fair_mmd/fairness_by_age.csv"),
        selection_ci_df[(selection_ci_df["run"] == "baseline") & (selection_ci_df["attribute"] == "age")],
        selection_ci_df[(selection_ci_df["run"] == "fair_mmd") & (selection_ci_df["attribute"] == "age")],
        "Selection rate by age: baseline vs improved Fair MMD",
        args.outdir / "selection_rate_by_age_baseline_vs_fair.png",
    )

    summary_lines = [
        "Improved fairness implementation",
        "-------------------------------",
        "This run compares the original baseline, the current test-alignment MMD, and an improved fairness-aware MMD.",
        "The improved method trains only on the training split and applies:",
        "- conditional hidden-state MMD across sensitive groups within label classes",
        "- score-level parity penalties across sex, race, and age",
        "",
        f"Outputs saved to: {args.outdir.resolve()}",
    ]
    (args.outdir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved improved fairness comparison to: {args.outdir.resolve()}")
    print("Comparison file:", (args.outdir / "comparison_metrics.csv").resolve())


if __name__ == "__main__":
    main()
