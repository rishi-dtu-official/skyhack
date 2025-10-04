from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import PATHS


def destination_difficulty_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("scheduled_arrival_station_code")
        .agg(
            flights=("flight_identifier", "nunique"),
            avg_difficulty_probability=("difficulty_probability_model", "mean"),
            pct_difficult=("difficulty_class", lambda s: (s == "Difficult").mean()),
            avg_turn_buffer=("scheduled_vs_min_turn_buffer", "mean"),
            avg_transfer_ratio=("transfer_ratio", "mean"),
            avg_ssr_per_pnr=("ssr_per_pnr", "mean"),
            avg_load_factor=("load_factor", "mean"),
        )
        .reset_index()
    )

    summary = summary.sort_values("pct_difficult", ascending=False)
    summary.to_csv(Path(PATHS.artifacts_tables) / "destination_difficulty_summary.csv", index=False)
    return summary


def difficulty_driver_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = [
        "scheduled_vs_min_turn_buffer",
        "load_factor",
        "ssr_per_pnr",
        "transfer_ratio",
        "bags_per_pax",
        "pnr_pressure_index",
    ]

    difficult_means = df[df["difficulty_class"] == "Difficult"][agg_cols].mean()
    easy_means = df[df["difficulty_class"] == "Easy"][agg_cols].mean()

    driver_delta = (difficult_means - easy_means).rename("delta")
    driver_df = driver_delta.reset_index().rename(columns={"index": "feature"})
    driver_df = driver_df.sort_values("delta", ascending=False)
    driver_df.to_csv(Path(PATHS.artifacts_tables) / "difficulty_driver_deltas.csv", index=False)

    return driver_df


def recommended_actions(summary: pd.DataFrame, driver_df: pd.DataFrame) -> Dict[str, str]:
    actions = {}

    high_difficulty_destinations = summary.head(5)["scheduled_arrival_station_code"].tolist()
    if high_difficulty_destinations:
        actions[
            "focus_destinations"
        ] = ", ".join(high_difficulty_destinations)

    top_driver = driver_df.head(1)
    if not top_driver.empty:
        feature = top_driver.iloc[0]["feature"]
        if feature == "scheduled_vs_min_turn_buffer":
            actions["turn_time_management"] = (
                "Increase scheduled ground time or deploy surge staffing for rotations with minimal buffers."
            )
        elif feature == "transfer_ratio":
            actions["transfer_bag_priority"] = (
                "Prioritize transfer and hot-transfer bag handling windows; pre-alert baggage teams."
            )
        elif feature == "ssr_per_pnr":
            actions["customer_service_allocation"] = (
                "Stage additional mobility-assist agents and pre-board coordinators on high-SSR flights."
            )
        elif feature == "pnr_pressure_index":
            actions["booking_monitoring"] = (
                "Monitor late-booking surges daily and pre-stage flex staffing where short-lead bookings spike."
            )
        elif feature == "load_factor":
            actions["gate_staffing"] = (
                "Align gate staffing with passenger load peaks; plan overflow support for top-load departures."
            )
        else:
            actions["general"] = (
                "Review top driver metrics and implement targeted playbooks (staffing, sequencing, equipment)."
            )

    return actions
