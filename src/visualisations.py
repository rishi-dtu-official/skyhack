from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from .config import PATHS
from .scoring import TARGET_COLUMN

console = Console()


def _visualisation_dir() -> Path:
    path = PATHS.root / "visualisation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(fig: plt.Figure, filename: str) -> None:
    output_dir = _visualisation_dir()
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    console.log(f"Saved visualisation to {output_path}")


def _plot_calibration_curves(holdout: pd.DataFrame) -> None:
    required_cols = {
        "difficulty_probability_raw_model",
        "difficulty_probability_model",
        "difficulty_probability_blended",
    }
    if not required_cols.issubset(holdout.columns):
        console.log("Skipping calibration plot - required columns missing")
        return

    y_true = holdout[TARGET_COLUMN].astype(int).to_numpy()
    probability_sets = {
        "Raw model": holdout["difficulty_probability_raw_model"].to_numpy(),
        "Calibrated": holdout["difficulty_probability_model"].to_numpy(),
        "Blended": holdout["difficulty_probability_blended"].to_numpy(),
    }

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 6))

    for label, probs in probability_sets.items():
        if np.allclose(probs, probs[0]):  # constant predictions cannot form a curve
            continue
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=12, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    ax.set_title("Calibration Comparison", fontsize=14)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    _save_fig(fig, "calibration_comparison.png")


def _plot_confusion_matrix(holdout: pd.DataFrame, threshold: float) -> None:
    if TARGET_COLUMN not in holdout.columns:
        console.log("Skipping confusion matrix - target column missing")
        return

    if "difficulty_prediction" in holdout.columns:
        y_pred = holdout["difficulty_prediction"].astype(int).to_numpy()
    else:
        y_pred = (holdout["difficulty_probability_blended"].to_numpy() >= threshold).astype(int)

    y_true = holdout[TARGET_COLUMN].astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.set_theme(style="white")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Easy", "Predicted Difficult"],
        yticklabels=["Actual Easy", "Actual Difficult"],
        ax=ax,
    )
    ax.set_title("Blended Model Confusion Matrix", fontsize=14)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")

    for text in ax.texts:
        text.set_fontsize(12)

    _save_fig(fig, "confusion_matrix_blended.png")


def _plot_cost_comparison() -> None:
    cost_path = PATHS.artifacts_tables / "cost_benefit_analysis.json"
    if not cost_path.exists():
        console.log("Skipping cost comparison - missing cost analysis data")
        return

    with cost_path.open("r", encoding="utf-8") as fp:
        cost_data = json.load(fp)

    baseline = cost_data.get("expected_daily_cost_baseline")
    predictive = cost_data.get("expected_daily_cost_model")
    savings = cost_data.get("expected_daily_savings")

    if baseline is None or predictive is None or savings is None:
        console.log("Skipping cost comparison - incomplete cost data")
        return

    values = {
        "Reactive Ops": baseline,
        "Predictive": predictive,
        "Daily Savings": savings,
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    colors = ["#d9534f", "#5cb85c", "#00a9ce"]
    bars = ax.bar(values.keys(), np.array(list(values.values())) / 1_000, color=colors)
    ax.set_ylabel("Cost ($k)")
    ax.set_title("Cost Comparison Analysis", fontsize=14)

    for bar, value in zip(bars, values.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"${value/1000:,.0f}k",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

    _save_fig(fig, "expected_daily_cost_comparison.png")


def _plot_top_feature_importances(top_n: int = 12) -> None:
    feature_path = PATHS.artifacts_tables / "feature_importances.csv"
    if not feature_path.exists():
        console.log("Skipping feature importance plot - missing CSV")
        return

    feature_df = pd.read_csv(feature_path)
    if feature_df.empty or {"feature", "mean_abs_shap"}.issubset(feature_df.columns) is False:
        console.log("Skipping feature importance plot - malformed CSV")
        return

    subset = feature_df.head(top_n).copy()
    subset = subset.iloc[::-1]  # plot most important at top

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=subset,
        x="mean_abs_shap",
        y="feature",
        color="#4f6d7a",
        ax=ax,
    )
    ax.set_title("Top Feature Importances (mean |SHAP|)", fontsize=14)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")

    _save_fig(fig, "top_feature_importances.png")


def generate_visualisations(scored_df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    holdout = scored_df.loc[scored_df["dataset_split"] == "test"].copy()
    if holdout.empty:
        console.log("Skipping visualisation refresh - holdout dataset empty")
        return

    threshold = metrics.get("threshold", 0.5)

    _plot_calibration_curves(holdout)
    _plot_confusion_matrix(holdout, threshold)
    _plot_cost_comparison()
    _plot_top_feature_importances()