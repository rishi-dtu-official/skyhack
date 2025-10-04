from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

from .config import PATHS

console = Console()


def _save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def run_eda(flight_features: pd.DataFrame) -> Dict[str, float]:
    df = flight_features.copy()

    average_departure_delay = float(df["departure_delay_minutes"].mean())
    pct_late_departures = float((df["departure_delay_minutes"] > 0).mean())

    close_turn_count = int(df["turn_shortfall_flag"].sum())
    close_turn_pct = float((df["turn_shortfall_flag"].mean()))

    transfer_ratio_mean = float(df["transfer_ratio"].mean())

    load_stats = df["load_factor"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    load_delay_corr = float(df[["load_factor", "positive_departure_delay"]].corr().iloc[0, 1])

    ssr_delay_corr = float(df[["ssr_per_pnr", "positive_departure_delay"]].corr().iloc[0, 1])
    load_ssr_corr = float(df[["load_factor", "ssr_per_pnr"]].corr().iloc[0, 1])

    numerator = ssr_delay_corr - load_ssr_corr * load_delay_corr
    denominator = np.sqrt((1 - load_ssr_corr**2) * (1 - load_delay_corr**2))
    if denominator == 0:
        ssr_delay_partial = float("nan")
    else:
        ssr_delay_partial = float(numerator / denominator)

    metrics = {
        "average_departure_delay_minutes": average_departure_delay,
        "percent_late_departures": pct_late_departures,
        "flights_close_to_min_turn_count": close_turn_count,
        "flights_close_to_min_turn_pct": close_turn_pct,
        "mean_transfer_ratio": transfer_ratio_mean,
        "load_factor_stats": load_stats,
        "load_delay_correlation": load_delay_corr,
        "ssr_delay_correlation": ssr_delay_corr,
        "ssr_delay_partial_correlation": ssr_delay_partial,
    }

    _save_json(metrics, Path(PATHS.artifacts_tables) / "eda_metrics.json")

    _plot_delay_distribution(df)
    _plot_load_vs_delay(df)
    _plot_ssr_by_difficulty(df)

    return metrics


def _plot_delay_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["departure_delay_minutes"], bins=40, kde=True, ax=ax, color="#1f77b4")
    ax.axvline(df["departure_delay_minutes"].mean(), color="red", linestyle="--", label="Mean")
    ax.set_title("Departure Delay Distribution")
    ax.set_xlabel("Delay (minutes)")
    ax.legend()
    fig.tight_layout()
    output_path = PATHS.artifacts_figures / "departure_delay_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_load_vs_delay(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="load_factor",
        y="departure_delay_minutes",
        hue="difficulty_class",
        palette="viridis",
        ax=ax,
        alpha=0.6,
    )
    ax.set_title("Load Factor vs Departure Delay")
    ax.set_xlabel("Load Factor")
    ax.set_ylabel("Departure Delay (minutes)")
    fig.tight_layout()
    output_path = PATHS.artifacts_figures / "load_vs_delay.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_ssr_by_difficulty(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = df.groupby("difficulty_class")["ssr_per_pnr"].median().reset_index()
    sns.barplot(
        data=grouped,
        x="difficulty_class",
        y="ssr_per_pnr",
        ax=ax,
        color="#b44982",
    )
    ax.set_title("Median SSR per PNR by Difficulty Class")
    ax.set_xlabel("Difficulty Class")
    ax.set_ylabel("SSR per PNR")
    fig.tight_layout()
    output_path = PATHS.artifacts_figures / "ssr_per_pnr_by_difficulty.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
