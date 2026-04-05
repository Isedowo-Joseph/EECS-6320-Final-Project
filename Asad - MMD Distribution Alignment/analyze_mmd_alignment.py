#!/usr/bin/env python3
"""
Reproduce the current MMD Distribution Alignment experiment from the cleaned
dataset and generate before/after comparison outputs.

Important:
- This script mirrors the current notebook's protocol, including the MMD model
  aligning training batches to unlabeled test features.
- That makes the "after" metrics useful for debugging the current
  implementation, but not a leak-free final evaluation protocol.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires PyTorch (`torch`). Install it in the Python environment "
        "used to run the notebook, then rerun the script."
    ) from exc


SEED = 42

BASELINE_NUMERIC_COLS = [
    "loan_amount",
    "income",
    "property_value",
    "loan_term",
]

BASELINE_CATEGORICAL_COLS = [
    "occupancy_type",
    "state_code",
    "county_code",
    "applicant_credit_scoring_model",
    "debt_to_income_ratio",
    "interest_only_payment",
]

SENSITIVE_COLS = [
    "applicant_sex",
    "applicant_race_1",
    "applicant_age",
]

FEATURE_COLS = BASELINE_NUMERIC_COLS + BASELINE_CATEGORICAL_COLS
TARGET_COL = "target"

SEX_MAP = {"1": "Male", "2": "Female"}
RACE_MAP = {"2": "Asian", "3": "Black", "5": "White"}
AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]


@dataclass
class RunOutputs:
    name: str
    scores: np.ndarray
    preds: np.ndarray
    history: list[dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze baseline vs MMD alignment outputs.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("MMD_Distribution_Outputs/q2_cleaned_dataset.csv"),
        help="Path to the cleaned dataset CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("MMD_Distribution_Outputs/analysis"),
        help="Directory where analysis CSVs and plots will be written.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer width.")
    parser.add_argument(
        "--lambda-mmd",
        type=float,
        default=1.0,
        help="Weight for the MMD alignment loss.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=300,
        help="Bootstrap iterations for metric confidence intervals.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in BASELINE_CATEGORICAL_COLS + SENSITIVE_COLS:
        df[col] = df[col].astype(str)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = df[FEATURE_COLS + SENSITIVE_COLS].copy()
    y = df[TARGET_COL].copy()
    return train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, BASELINE_NUMERIC_COLS),
            ("categorical", categorical_pipeline, BASELINE_CATEGORICAL_COLS),
        ]
    )


def preprocess(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    transformer = build_preprocessor()
    x_train_processed = transformer.fit_transform(x_train)
    x_test_processed = transformer.transform(x_test)
    if hasattr(x_train_processed, "toarray"):
        x_train_processed = x_train_processed.toarray()
    if hasattr(x_test_processed, "toarray"):
        x_test_processed = x_test_processed.toarray()
    return x_train_processed.astype(np.float32), x_test_processed.astype(np.float32)


class Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    dists = torch.cdist(x, y).pow(2)
    return torch.exp(-dists / (2 * sigma**2))


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    k_xx = kernel(x, x, sigma).mean()
    k_yy = kernel(y, y, sigma).mean()
    k_xy = kernel(x, y, sigma).mean()
    return k_xx + k_yy - 2 * k_xy


def multi_kernel_mmd(x: torch.Tensor, y: torch.Tensor, sigmas: Iterable[float] = (1.0, 5.0, 10.0)) -> torch.Tensor:
    losses = [compute_mmd(x, y, sigma=sigma) for sigma in sigmas]
    return sum(losses) / len(losses)


def evaluate_predictions(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "roc_auc": roc_auc_score(y_true, scores),
        "brier_score": brier_score_loss(y_true, scores),
        "positive_rate": float(preds.mean()),
        "mean_score": float(scores.mean()),
        "mean_confidence": float(np.maximum(scores, 1.0 - scores).mean()),
    }


def train_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
) -> RunOutputs:
    set_seed(SEED)
    model = Model(x_train.shape[1], hidden_dim=args.hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    train_loader = make_loader(x_train, y_train, args.batch_size, SEED, shuffle=True)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        last_loss = math.nan
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            scores = model(x_batch).squeeze()
            loss = criterion(scores, y_batch)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        history.append({"epoch": epoch, "loss": last_loss})

    model.eval()
    with torch.no_grad():
        test_scores = model(x_test_tensor).squeeze().cpu().numpy()
    test_preds = (test_scores >= 0.5).astype(int)
    return RunOutputs(name="baseline", scores=test_scores, preds=test_preds, history=history)


def train_mmd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> RunOutputs:
    set_seed(SEED)
    model = Model(x_train.shape[1], hidden_dim=args.hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    train_loader = make_loader(x_train, y_train, args.batch_size, SEED, shuffle=True)
    target_loader = make_loader(x_test, y_test, args.batch_size, SEED + 1, shuffle=True)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    history: list[dict[str, float]] = []
    target_iter = iter(target_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        last_cls = math.nan
        last_mmd = math.nan
        last_total = math.nan

        for x_batch, y_batch in train_loader:
            try:
                x_target, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                x_target, _ = next(target_iter)

            optimizer.zero_grad()
            batch_scores = model(x_batch).squeeze()
            cls_loss = criterion(batch_scores, y_batch)
            src_features = model.encode(x_batch)
            tgt_features = model.encode(x_target)
            align_loss = multi_kernel_mmd(src_features, tgt_features)
            loss = cls_loss + args.lambda_mmd * align_loss
            loss.backward()
            optimizer.step()

            last_cls = float(cls_loss.item())
            last_mmd = float(align_loss.item())
            last_total = float(loss.item())

        history.append(
            {
                "epoch": epoch,
                "loss": last_total,
                "classification_loss": last_cls,
                "mmd_loss": last_mmd,
            }
        )

    model.eval()
    with torch.no_grad():
        test_scores = model(x_test_tensor).squeeze().cpu().numpy()
    test_preds = (test_scores >= 0.5).astype(int)
    return RunOutputs(name="mmd", scores=test_scores, preds=test_preds, history=history)


def fairness_table(input_df: pd.DataFrame, group_col: str, positive_label: int = 1) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group_value, group_df in input_df.groupby(group_col):
        if pd.isna(group_value) or len(group_df) == 0:
            continue
        positive_true = int((group_df["y_true"] == positive_label).sum())
        negative_true = int((group_df["y_true"] != positive_label).sum())
        selection_rate = float((group_df["y_pred"] == positive_label).mean())
        base_positive_rate = float((group_df["y_true"] == positive_label).mean())
        tpr = float("nan")
        fpr = float("nan")
        if positive_true > 0:
            tpr = float(
                ((group_df["y_pred"] == positive_label) & (group_df["y_true"] == positive_label)).sum() / positive_true
            )
        if negative_true > 0:
            fpr = float(
                ((group_df["y_pred"] == positive_label) & (group_df["y_true"] != positive_label)).sum() / negative_true
            )
        rows.append(
            {
                "group": group_value,
                "n": int(len(group_df)),
                "base_positive_rate": base_positive_rate,
                "selection_rate": selection_rate,
                "true_positive_rate": tpr,
                "false_positive_rate": fpr,
                "mean_score": float(group_df["y_score"].mean()),
                "mean_confidence": float(group_df["confidence"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    if group_col == "applicant_age":
        out["group"] = pd.Categorical(out["group"], categories=AGE_ORDER, ordered=True)
        out = out.sort_values("group").reset_index(drop=True)
        out["group"] = out["group"].astype(str)
    overall_selection_rate = float((input_df["y_pred"] == positive_label).mean())
    out["selection_rate_minus_overall"] = out["selection_rate"] - overall_selection_rate
    return out


def fairness_gap_summary(tbl: pd.DataFrame) -> dict[str, float]:
    return {
        "demographic_parity_gap": float(tbl["selection_rate"].max() - tbl["selection_rate"].min()),
        "equal_opportunity_gap": float(tbl["true_positive_rate"].max() - tbl["true_positive_rate"].min()),
        "false_positive_rate_gap": float(tbl["false_positive_rate"].max() - tbl["false_positive_rate"].min()),
    }


def make_eval_frame(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    outputs: RunOutputs,
) -> pd.DataFrame:
    eval_df = x_test[SENSITIVE_COLS].copy()
    eval_df["y_true"] = y_test.to_numpy()
    eval_df["y_pred"] = outputs.preds
    eval_df["y_score"] = outputs.scores
    eval_df["confidence"] = np.maximum(outputs.scores, 1.0 - outputs.scores)
    eval_df["sex_label"] = eval_df["applicant_sex"].map(SEX_MAP)
    eval_df["race_label"] = eval_df["applicant_race_1"].map(RACE_MAP)
    return eval_df


def save_run_artifacts(
    outdir: Path,
    outputs: RunOutputs,
    eval_df: pd.DataFrame,
    metrics: dict[str, float],
) -> dict[str, float]:
    per_run_dir = outdir / outputs.name
    per_run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(outputs.history).to_csv(per_run_dir / "training_history.csv", index=False)
    eval_df.to_csv(per_run_dir / "predictions.csv", index=False)

    sex_tbl = fairness_table(eval_df, "sex_label")
    race_tbl = fairness_table(eval_df, "race_label")
    age_tbl = fairness_table(eval_df, "applicant_age")

    sex_tbl.to_csv(per_run_dir / "fairness_by_sex.csv", index=False)
    race_tbl.to_csv(per_run_dir / "fairness_by_race.csv", index=False)
    age_tbl.to_csv(per_run_dir / "fairness_by_age.csv", index=False)

    gap_rows = []
    for attribute, table in [("sex", sex_tbl), ("race", race_tbl), ("age", age_tbl)]:
        for metric_name, value in fairness_gap_summary(table).items():
            gap_rows.append({"attribute": attribute, "metric": metric_name, "value": value})
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(per_run_dir / "fairness_gap_summary.csv", index=False)

    metrics_with_gaps = metrics.copy()
    for _, row in gap_df.iterrows():
        metrics_with_gaps[f"{row['attribute']}_{row['metric']}"] = float(row["value"])
    pd.DataFrame(
        [{"metric": metric_name, "value": metric_value} for metric_name, metric_value in metrics_with_gaps.items()]
    ).to_csv(per_run_dir / "metrics.csv", index=False)

    return metrics_with_gaps


def bootstrap_cis(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_names: Iterable[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []
    metrics_per_sample = {name: [] for name in metric_names}
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        sample_true = y_true[idx]
        sample_scores = scores[idx]
        sample_metrics = evaluate_predictions(sample_true, sample_scores)
        for name in metric_names:
            metrics_per_sample[name].append(sample_metrics[name])

    for name in metric_names:
        values = np.array(metrics_per_sample[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "mean": float(values.mean()),
                "ci_lower_95": float(np.quantile(values, 0.025)),
                "ci_upper_95": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def plot_training_losses(baseline_history: pd.DataFrame, mmd_history: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(baseline_history["epoch"], baseline_history["loss"], label="Baseline loss", linewidth=2)
    ax.plot(mmd_history["epoch"], mmd_history["classification_loss"], label="MMD BCE loss", linewidth=2)
    ax.plot(mmd_history["epoch"], mmd_history["mmd_loss"], label="MMD alignment loss", linewidth=2)
    ax.plot(mmd_history["epoch"], mmd_history["loss"], label="MMD total loss", linewidth=2, alpha=0.8)
    ax.set_title("Training Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_confidence_histogram(pred_df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].hist(pred_df["baseline_score"], bins=30, alpha=0.8, color="#1f77b4")
    axes[0].set_title("Baseline approval probability")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")

    axes[1].hist(pred_df["mmd_score"], bins=30, alpha=0.8, color="#ff7f0e")
    axes[1].set_title("MMD approval probability")
    axes[1].set_xlabel("Predicted probability")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_confidence_margin(pred_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(pred_df["baseline_confidence"], bins=30, alpha=0.6, label="Baseline", color="#1f77b4")
    ax.hist(pred_df["mmd_confidence"], bins=30, alpha=0.6, label="MMD", color="#ff7f0e")
    ax.set_title("Prediction confidence comparison")
    ax.set_xlabel("Confidence = max(p, 1-p)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_metrics_comparison(comparison_df: pd.DataFrame, outpath: Path) -> None:
    metric_order = ["accuracy", "balanced_accuracy", "roc_auc", "brier_score", "mean_confidence"]
    subset = comparison_df[comparison_df["metric"].isin(metric_order)].copy()
    x = np.arange(len(subset))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, subset["baseline"], width=width, label="Baseline", color="#1f77b4")
    ax.bar(x + width / 2, subset["mmd"], width=width, label="MMD", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["metric"], rotation=20)
    ax.set_title("Before vs after performance metrics")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_gap_comparison(gap_df: pd.DataFrame, outpath: Path) -> None:
    gap_df = gap_df.copy()
    gap_df["label"] = gap_df["attribute"] + "\n" + gap_df["metric"].str.replace("_", " ")
    x = np.arange(len(gap_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, gap_df["baseline"], width=width, label="Baseline", color="#1f77b4")
    ax.bar(x + width / 2, gap_df["mmd"], width=width, label="MMD", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(gap_df["label"], rotation=35, ha="right")
    ax.set_title("Fairness gap comparison")
    ax.set_ylabel("Gap size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_group_selection_rates(
    baseline_tbl: pd.DataFrame,
    mmd_tbl: pd.DataFrame,
    title: str,
    outpath: Path,
) -> None:
    merged = baseline_tbl[["group", "selection_rate"]].merge(
        mmd_tbl[["group", "selection_rate"]],
        on="group",
        suffixes=("_baseline", "_mmd"),
    )
    x = np.arange(len(merged))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, merged["selection_rate_baseline"], width=width, label="Baseline", color="#1f77b4")
    ax.bar(x + width / 2, merged["selection_rate_mmd"], width=width, label="MMD", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["group"], rotation=25)
    ax.set_title(title)
    ax.set_ylabel("Selection rate")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_calibration_curve(pred_df: pd.DataFrame, outpath: Path, bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, score_col, color in [
        ("Baseline", "baseline_score", "#1f77b4"),
        ("MMD", "mmd_score", "#ff7f0e"),
    ]:
        scores = pred_df[score_col].to_numpy()
        y_true = pred_df["y_true"].to_numpy()
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        bin_ids = np.digitize(scores, bin_edges[1:-1], right=True)
        xs = []
        ys = []
        for bin_id in range(bins):
            mask = bin_ids == bin_id
            if mask.sum() == 0:
                continue
            xs.append(scores[mask].mean())
            ys.append(y_true[mask].mean())
        ax.plot(xs, ys, marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Calibration curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed approval rate")
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

    print("Training baseline model...")
    baseline_outputs = train_baseline(
        x_train_processed,
        y_train.to_numpy(),
        x_test_processed,
        args,
    )
    print("Training MMD model...")
    mmd_outputs = train_mmd(
        x_train_processed,
        y_train.to_numpy(),
        x_test_processed,
        y_test.to_numpy(),
        args,
    )

    print("Scoring predictions and saving tables...")
    baseline_eval = make_eval_frame(x_test, y_test, baseline_outputs)
    mmd_eval = make_eval_frame(x_test, y_test, mmd_outputs)

    baseline_metrics = evaluate_predictions(y_test.to_numpy(), baseline_outputs.scores)
    mmd_metrics = evaluate_predictions(y_test.to_numpy(), mmd_outputs.scores)

    baseline_metrics_all = save_run_artifacts(args.outdir, baseline_outputs, baseline_eval, baseline_metrics)
    mmd_metrics_all = save_run_artifacts(args.outdir, mmd_outputs, mmd_eval, mmd_metrics)

    comparison = (
        pd.DataFrame({"baseline": pd.Series(baseline_metrics_all), "mmd": pd.Series(mmd_metrics_all)})
        .rename_axis("metric")
        .reset_index()
    )
    comparison["delta"] = comparison["mmd"] - comparison["baseline"]
    comparison.to_csv(args.outdir / "comparison_metrics.csv", index=False)

    pred_df = x_test[SENSITIVE_COLS].copy()
    pred_df["y_true"] = y_test.to_numpy()
    pred_df["baseline_score"] = baseline_outputs.scores
    pred_df["baseline_pred"] = baseline_outputs.preds
    pred_df["baseline_confidence"] = np.maximum(baseline_outputs.scores, 1.0 - baseline_outputs.scores)
    pred_df["mmd_score"] = mmd_outputs.scores
    pred_df["mmd_pred"] = mmd_outputs.preds
    pred_df["mmd_confidence"] = np.maximum(mmd_outputs.scores, 1.0 - mmd_outputs.scores)
    pred_df["score_delta"] = pred_df["mmd_score"] - pred_df["baseline_score"]
    pred_df["confidence_delta"] = pred_df["mmd_confidence"] - pred_df["baseline_confidence"]
    pred_df.to_csv(args.outdir / "before_after_predictions.csv", index=False)

    print("Bootstrapping confidence intervals...")
    metric_names = ["accuracy", "balanced_accuracy", "roc_auc", "brier_score", "mean_confidence"]
    baseline_ci = bootstrap_cis(
        y_test.to_numpy(),
        baseline_outputs.scores,
        metric_names,
        args.bootstrap_iterations,
        seed=SEED,
    )
    baseline_ci.insert(0, "run", "baseline")
    mmd_ci = bootstrap_cis(
        y_test.to_numpy(),
        mmd_outputs.scores,
        metric_names,
        args.bootstrap_iterations,
        seed=SEED + 1,
    )
    mmd_ci.insert(0, "run", "mmd")
    pd.concat([baseline_ci, mmd_ci], ignore_index=True).to_csv(args.outdir / "bootstrap_metric_cis.csv", index=False)

    print("Generating plots...")
    baseline_history_df = pd.DataFrame(baseline_outputs.history)
    mmd_history_df = pd.DataFrame(mmd_outputs.history)
    plot_training_losses(baseline_history_df, mmd_history_df, args.outdir / "training_losses.png")
    plot_confidence_histogram(pred_df, args.outdir / "confidence_histograms.png")
    plot_confidence_margin(pred_df, args.outdir / "confidence_comparison.png")
    plot_calibration_curve(pred_df, args.outdir / "calibration_curve.png")
    plot_metrics_comparison(comparison, args.outdir / "metrics_comparison.png")

    gap_subset = comparison[comparison["metric"].str.endswith("_gap")].copy()
    gap_subset["attribute"] = gap_subset["metric"].str.split("_").str[0]
    gap_subset["metric"] = gap_subset["metric"].str.split("_").str[1:].str.join("_")
    plot_gap_comparison(gap_subset, args.outdir / "fairness_gap_comparison.png")

    plot_group_selection_rates(
        pd.read_csv(args.outdir / "baseline/fairness_by_sex.csv"),
        pd.read_csv(args.outdir / "mmd/fairness_by_sex.csv"),
        "Selection rate by sex",
        args.outdir / "selection_rate_by_sex.png",
    )
    plot_group_selection_rates(
        pd.read_csv(args.outdir / "baseline/fairness_by_race.csv"),
        pd.read_csv(args.outdir / "mmd/fairness_by_race.csv"),
        "Selection rate by race",
        args.outdir / "selection_rate_by_race.png",
    )
    plot_group_selection_rates(
        pd.read_csv(args.outdir / "baseline/fairness_by_age.csv"),
        pd.read_csv(args.outdir / "mmd/fairness_by_age.csv"),
        "Selection rate by age",
        args.outdir / "selection_rate_by_age.png",
    )

    audit_lines = [
        "Audit notes",
        "-----------",
        "1. The current notebook aligns training features to test features in the MMD objective.",
        "2. The notebook ROC AUC is computed on hard class labels; this script uses probabilities instead.",
        "3. Use these outputs to inspect the current implementation, not as a final leak-free benchmark.",
        "",
        "Key files",
        "---------",
        f"- {args.outdir / 'comparison_metrics.csv'}",
        f"- {args.outdir / 'bootstrap_metric_cis.csv'}",
        f"- {args.outdir / 'before_after_predictions.csv'}",
    ]
    (args.outdir / "audit_summary.txt").write_text("\n".join(audit_lines), encoding="utf-8")

    print(f"Saved analysis outputs to: {args.outdir.resolve()}")
    print("Main comparison file:", (args.outdir / "comparison_metrics.csv").resolve())
    print("Predictions file:", (args.outdir / "before_after_predictions.csv").resolve())


if __name__ == "__main__":
    main()
