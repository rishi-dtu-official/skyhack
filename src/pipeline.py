from __future__ import annotations

import json
from pathlib import Path

import duckdb
from rich.console import Console

from .config import PATHS
from .data_loaders import (
    load_airports,
    load_bag_level,
    load_flight_level,
    load_pnr_flight_level,
    load_pnr_remarks,
)
from .eda import run_eda
from .feature_engineering import FlightFeatureFrames, build_feature_frames
from .insights import destination_difficulty_summary, difficulty_driver_summary, recommended_actions
from .scoring import apply_scoring, train_difficulty_model
from .visualisations import generate_visualisations

console = Console()


def _ensure_directories() -> None:
    PATHS.artifacts_figures.mkdir(parents=True, exist_ok=True)
    PATHS.artifacts_tables.mkdir(parents=True, exist_ok=True)
    PATHS.artifacts_models.mkdir(parents=True, exist_ok=True)
    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)


def _run_sql_analytics(frames: FlightFeatureFrames, scored_df) -> None:
    console.log("Running SQL analytics")
    con = duckdb.connect(str(PATHS.duckdb_path))

    try:
        con.register("flight_features_df", scored_df)
        con.register("pnr_features_df", frames.pnr)
        con.register("bag_features_df", frames.bag)

        con.execute("CREATE OR REPLACE TABLE flight_features AS SELECT * FROM flight_features_df")
        con.execute("CREATE OR REPLACE TABLE pnr_features AS SELECT * FROM pnr_features_df")
        con.execute("CREATE OR REPLACE TABLE bag_features AS SELECT * FROM bag_features_df")

        sql_dir = PATHS.root / "sql"
        for sql_file in sorted(sql_dir.glob("*.sql")):
            query = sql_file.read_text()
            result = con.execute(query).df()
            output_path = PATHS.artifacts_tables / f"{sql_file.stem}.csv"
            result.to_csv(output_path, index=False)
            console.log(f"Saved SQL result to {output_path}")
    finally:
        con.close()


def _save_recommended_actions(actions: dict) -> None:
    path = PATHS.artifacts_tables / "recommended_actions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(actions, fp, indent=2)


def run_pipeline() -> None:
    console.rule("SkyHack Flight Difficulty Pipeline")
    _ensure_directories()

    console.log("Loading raw datasets")
    flight_level = load_flight_level()
    pnr_level = load_pnr_flight_level()
    pnr_remarks = load_pnr_remarks()
    bag_level = load_bag_level()
    airports = load_airports()

    frames = build_feature_frames(flight_level, pnr_level, pnr_remarks, bag_level, airports)

    console.log("Training difficulty model")
    model_outputs = train_difficulty_model(frames.flight)

    console.log("Applying scoring and exporting test_databaes.csv")
    scored = apply_scoring(frames.flight, model_outputs)

    console.log("Running exploratory data analysis")
    run_eda(scored)

    console.log("Refreshing presentation visualisations")
    generate_visualisations(scored, model_outputs.metrics)

    console.log("Deriving insights")
    dest_summary = destination_difficulty_summary(scored)
    driver_summary = difficulty_driver_summary(scored)
    actions = recommended_actions(dest_summary, driver_summary)
    _save_recommended_actions(actions)

    _run_sql_analytics(frames, scored)

    console.log("Pipeline complete")


if __name__ == "__main__":
    run_pipeline()
