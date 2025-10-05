from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from rich.console import Console

from .config import PATHS

console = Console()

FLIGHT_KEYS: List[str] = [
    "company_id",
    "flight_number",
    "scheduled_departure_date_local",
    "scheduled_departure_station_code",
    "scheduled_arrival_station_code",
]

WIDE_BODY_PREFIXES = (
    "B767",
    "B777",
    "B787",
    "B747",
    "A330",
    "A340",
    "A350",
    "A380",
)


def _map_daypart(hour: float) -> str:
    if pd.isna(hour):
        return "unknown"
    hour = int(hour)
    if hour >= 21 or hour < 5:
        return "overnight"
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "midday"
    return "evening"


DAYPART_CODE_MAP = {
    "overnight": 0,
    "morning": 1,
    "midday": 2,
    "evening": 3,
    "unknown": -1,
}


WAVE_BUCKET_CODE_MAP = {
    "early": 0,
    "mid": 1,
    "late": 2,
}


@dataclass
class FlightFeatureFrames:
    flight: pd.DataFrame
    pnr: pd.DataFrame
    ssr: pd.DataFrame
    bag: pd.DataFrame


def _standardize_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def prepare_flight_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _standardize_datetime(
        df,
        [
            "scheduled_departure_datetime_local",
            "scheduled_arrival_datetime_local",
            "actual_departure_datetime_local",
            "actual_arrival_datetime_local",
        ],
    )

    df["scheduled_departure_date_local"] = pd.to_datetime(
        df["scheduled_departure_date_local"], errors="coerce"
    )

    def _minutes_delta(actual, scheduled):
        delta = (actual - scheduled).dt.total_seconds() / 60.0
        return delta

    df["departure_delay_minutes"] = _minutes_delta(
        df["actual_departure_datetime_local"], df["scheduled_departure_datetime_local"]
    )
    df["arrival_delay_minutes"] = _minutes_delta(
        df["actual_arrival_datetime_local"], df["scheduled_arrival_datetime_local"]
    )

    df["positive_departure_delay"] = df["departure_delay_minutes"].clip(lower=0)
    df["positive_arrival_delay"] = df["arrival_delay_minutes"].clip(lower=0)

    df["scheduled_vs_min_turn_buffer"] = (
        df["scheduled_ground_time_minutes"] - df["minimum_turn_minutes"]
    )
    df["actual_vs_min_turn_buffer"] = (
        df["actual_ground_time_minutes"] - df["minimum_turn_minutes"]
    )
    df["turn_shortfall_flag"] = df["scheduled_vs_min_turn_buffer"] <= 5

    df["scheduled_ground_to_min_ratio"] = np.where(
        df["minimum_turn_minutes"] > 0,
        df["scheduled_ground_time_minutes"] / df["minimum_turn_minutes"],
        np.nan,
    )

    df["actual_ground_to_scheduled_ratio"] = np.where(
        df["scheduled_ground_time_minutes"] > 0,
        df["actual_ground_time_minutes"] / df["scheduled_ground_time_minutes"],
        np.nan,
    )

    df["departure_hour_local"] = df["scheduled_departure_datetime_local"].dt.hour
    df["day_of_week"] = df["scheduled_departure_datetime_local"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    df["departure_minute_local"] = df["scheduled_departure_datetime_local"].dt.minute
    df["departure_minutes_from_midnight"] = (
        df["departure_hour_local"] * 60 + df["departure_minute_local"]
    )

    daily_first_departure = df.groupby("scheduled_departure_date_local")[
        "departure_minutes_from_midnight"
    ].transform("min")
    df["minutes_since_first_departure"] = (
        df["departure_minutes_from_midnight"] - daily_first_departure
    ).clip(lower=0)
    df["is_first_departure_of_day"] = df["departure_minutes_from_midnight"].eq(
        daily_first_departure
    )

    df["departure_daypart"] = df["departure_hour_local"].apply(_map_daypart)
    df["departure_daypart_code"] = (
        df["departure_daypart"].map(DAYPART_CODE_MAP).fillna(-1).astype(int)
    )
    df["is_red_eye"] = df["departure_daypart"].eq("overnight")

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    df["departure_hour_sin"] = np.sin(
        2 * np.pi * df["departure_minutes_from_midnight"] / 1440.0
    )
    df["departure_hour_cos"] = np.cos(
        2 * np.pi * df["departure_minutes_from_midnight"] / 1440.0
    )

    original_index = df.index
    df = df.sort_values(
        [
            "scheduled_departure_date_local",
            "scheduled_departure_datetime_local",
            "scheduled_departure_station_code",
            "flight_number",
        ]
    )

    df["departure_wave_quantile"] = df.groupby("scheduled_departure_date_local")[
        "scheduled_departure_datetime_local"
    ].rank(method="first", pct=True)

    df["departure_wave_bucket"] = np.select(
        [
            df["departure_wave_quantile"] <= (1.0 / 3.0),
            (df["departure_wave_quantile"] > (1.0 / 3.0))
            & (df["departure_wave_quantile"] <= (2.0 / 3.0)),
            df["departure_wave_quantile"] > (2.0 / 3.0),
        ],
        ["early", "mid", "late"],
        default="mid",
    )
    df["departure_wave_bucket_code"] = (
        df["departure_wave_bucket"].map(WAVE_BUCKET_CODE_MAP).fillna(-1).astype(int)
    )

    wave_start_minutes = df.groupby(
        ["scheduled_departure_date_local", "departure_wave_bucket"]
    )["departure_minutes_from_midnight"].transform("min")
    df["minutes_since_bank_start"] = (
        df["departure_minutes_from_midnight"] - wave_start_minutes
    ).clip(lower=0)

    df["station_departure_rank"] = df.groupby(
        ["scheduled_departure_date_local", "scheduled_departure_station_code"]
    )['scheduled_departure_datetime_local'].rank(method="first")
    station_counts = df.groupby(
        ["scheduled_departure_date_local", "scheduled_departure_station_code"]
    )['scheduled_departure_datetime_local'].transform("count")
    df["station_departure_rank_pct"] = df["station_departure_rank"] / station_counts
    df["is_station_first_departure"] = df["station_departure_rank"].eq(1)
    df["is_station_last_departure"] = df["station_departure_rank"].eq(station_counts)

    df = df.sort_index().loc[original_index]

    df["flight_identifier"] = (
        df["company_id"].astype(str)
        + "-"
        + df["flight_number"].astype(str)
        + "-"
        + df["scheduled_departure_date_local"].dt.strftime("%Y-%m-%d")
    )

    return df


def prepare_pnr_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scheduled_departure_date_local"] = pd.to_datetime(
        df["scheduled_departure_date_local"], errors="coerce"
    )
    df["pnr_creation_date"] = pd.to_datetime(df["pnr_creation_date"], errors="coerce")

    df["booking_lead_time_days"] = (
        df["scheduled_departure_date_local"] - df["pnr_creation_date"]
    ).dt.days

    df["is_child_flag"] = df["is_child"].str.upper().eq("Y")
    df["is_stroller_flag"] = df["is_stroller_user"].str.upper().eq("Y")
    df["basic_economy_flag"] = df["basic_economy_ind"].astype(int).eq(1)

    grouping_keys = FLIGHT_KEYS + ["record_locator"]

    agg = (
        df.groupby(grouping_keys, dropna=False)
        .agg(
            total_pax=("total_pax", "max"),
            lap_child_count=("lap_child_count", "max"),
            has_child=("is_child_flag", "max"),
            has_stroller=("is_stroller_flag", "max"),
            has_basic_economy=("basic_economy_flag", "max"),
            booking_lead_time_days=("booking_lead_time_days", "max"),
        )
        .reset_index()
    )

    agg["late_booking_flag"] = agg["booking_lead_time_days"].lt(3)
    agg["ultra_late_booking_flag"] = agg["booking_lead_time_days"].lt(1)

    flight_features = (
        agg.groupby(FLIGHT_KEYS, dropna=False)
        .agg(
            total_pax=("total_pax", "sum"),
            total_pnr=("record_locator", "nunique"),
            pnrs_with_children=("has_child", "sum"),
            pnrs_with_strollers=("has_stroller", "sum"),
            pnrs_basic_economy=("has_basic_economy", "sum"),
            lap_child_count=("lap_child_count", "sum"),
            avg_booking_lead_time_days=("booking_lead_time_days", "mean"),
            pnr_short_lead_count=("late_booking_flag", "sum"),
            pnr_ultra_short_lead_count=("ultra_late_booking_flag", "sum"),
        )
        .reset_index()
    )

    flight_features["avg_pnr_size"] = np.where(
        flight_features["total_pnr"] > 0,
        flight_features["total_pax"] / flight_features["total_pnr"],
        np.nan,
    )
    flight_features["share_pnr_with_children"] = np.where(
        flight_features["total_pnr"] > 0,
        flight_features["pnrs_with_children"] / flight_features["total_pnr"],
        0,
    )
    flight_features["share_pnr_with_strollers"] = np.where(
        flight_features["total_pnr"] > 0,
        flight_features["pnrs_with_strollers"] / flight_features["total_pnr"],
        0,
    )
    flight_features["share_basic_economy_pnr"] = np.where(
        flight_features["total_pnr"] > 0,
        flight_features["pnrs_basic_economy"] / flight_features["total_pnr"],
        0,
    )
    flight_features["share_short_lead_pnr"] = np.where(
        flight_features["total_pnr"] > 0,
        flight_features["pnr_short_lead_count"] / flight_features["total_pnr"],
        0,
    )
    flight_features["share_ultra_short_lead_pnr"] = np.where(
        flight_features["total_pnr"] > 0,
        flight_features["pnr_ultra_short_lead_count"] / flight_features["total_pnr"],
        0,
    )

    return flight_features, agg


def prepare_ssr_features(pnr_flight_unique: pd.DataFrame, remarks: pd.DataFrame) -> pd.DataFrame:
    remarks = remarks.copy()
    remarks["special_service_request"] = remarks["special_service_request"].str.strip()

    ssr = remarks.merge(
        pnr_flight_unique[FLIGHT_KEYS + ["record_locator"]].drop_duplicates(),
        on=["record_locator", "flight_number"],
        how="left",
    )

    ssr = ssr.dropna(subset=["scheduled_departure_date_local"])

    ssr["ssr_type"] = ssr["special_service_request"].str.lower()

    ssr_counts = (
        ssr.groupby(FLIGHT_KEYS + ["ssr_type"], dropna=False)
        .agg(ssr_count=("record_locator", "nunique"))
        .reset_index()
    )

    ssr_pivot = (
        ssr_counts.pivot_table(
            index=FLIGHT_KEYS,
            columns="ssr_type",
            values="ssr_count",
            fill_value=0,
        )
        .rename(columns=lambda c: f"ssr_{c.replace(' ', '_')}" if isinstance(c, str) else c)
        .reset_index()
    )

    total_ssr = (
        ssr.groupby(FLIGHT_KEYS, dropna=False)
        .agg(total_ssr_requests=("record_locator", "count"), unique_ssr_types=("ssr_type", "nunique"))
        .reset_index()
    )

    ssr_features = total_ssr.merge(ssr_pivot, on=FLIGHT_KEYS, how="left")
    ssr_features = ssr_features.fillna(0)

    return ssr_features


def prepare_bag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scheduled_departure_date_local"] = pd.to_datetime(
        df["scheduled_departure_date_local"], errors="coerce"
    )

    grouped = (
        df.groupby(FLIGHT_KEYS, dropna=False)
        .agg(
            total_bags=("bag_tag_unique_number", "nunique"),
            origin_bags=("bag_type", lambda s: (s == "Origin").sum()),
            transfer_bags=("bag_type", lambda s: (s == "Transfer").sum()),
            hot_transfer_bags=("bag_type", lambda s: (s == "Hot Transfer").sum()),
        )
        .reset_index()
    )

    grouped["all_transfer_bags"] = (
        grouped["transfer_bags"] + grouped["hot_transfer_bags"]
    )
    grouped["transfer_ratio"] = np.where(
        grouped["total_bags"] > 0, grouped["all_transfer_bags"] / grouped["total_bags"], 0
    )
    grouped["hot_transfer_ratio"] = np.where(
        grouped["all_transfer_bags"] > 0,
        grouped["hot_transfer_bags"] / grouped["all_transfer_bags"],
        0,
    )

    return grouped


def combine_features(
    flight_df: pd.DataFrame,
    pnr_features: pd.DataFrame,
    ssr_features: pd.DataFrame,
    bag_features: pd.DataFrame,
    airports_df: pd.DataFrame,
) -> pd.DataFrame:
    df = flight_df.merge(pnr_features, on=FLIGHT_KEYS, how="left")
    df = df.merge(ssr_features, on=FLIGHT_KEYS, how="left")
    df = df.merge(bag_features, on=FLIGHT_KEYS, how="left")

    airports_df = airports_df.rename(
        columns={
            "airport_iata_code": "scheduled_arrival_station_code",
            "iso_country_code": "arrival_country_code",
        }
    )
    df = df.merge(airports_df, on="scheduled_arrival_station_code", how="left")

    df = df.fillna(
        {
            "total_pax": 0,
            "total_pnr": 0,
            "pnrs_with_children": 0,
            "pnrs_with_strollers": 0,
            "pnrs_basic_economy": 0,
            "lap_child_count": 0,
            "avg_booking_lead_time_days": np.nan,
            "pnr_short_lead_count": 0,
            "pnr_ultra_short_lead_count": 0,
            "total_ssr_requests": 0,
            "unique_ssr_types": 0,
            "ssr_airport_wheelchair": 0,
            "ssr_manual_wheelchair": 0,
            "ssr_unaccompanied_minor": 0,
            "ssr_electric_wheelchair": 0,
            "total_bags": 0,
            "origin_bags": 0,
            "transfer_bags": 0,
            "hot_transfer_bags": 0,
            "all_transfer_bags": 0,
            "transfer_ratio": 0,
            "hot_transfer_ratio": 0,
        }
    )

    df["load_factor"] = np.where(
        df["total_seats"] > 0, df["total_pax"] / df["total_seats"], np.nan
    )
    df["bags_per_pax"] = np.where(
        df["total_pax"] > 0, df["total_bags"] / df["total_pax"], np.nan
    )
    df["transfer_bags_per_pax"] = np.where(
        df["total_pax"] > 0, df["all_transfer_bags"] / df["total_pax"], np.nan
    )
    df["hot_transfer_bags_per_pax"] = np.where(
        df["total_pax"] > 0, df["hot_transfer_bags"] / df["total_pax"], np.nan
    )

    df["pnr_pressure_index"] = (
        df["share_short_lead_pnr"] * 0.6
        + df["share_ultra_short_lead_pnr"] * 1.2
        + df["share_basic_economy_pnr"] * 0.3
    )

    df["ssr_per_pnr"] = np.where(
        df["total_pnr"] > 0, df["total_ssr_requests"] / df["total_pnr"], 0
    )

    df["child_ratio"] = np.where(
        df["total_pax"] > 0, df["pnrs_with_children"] / df["total_pax"], 0
    )

    df["stroller_ratio"] = np.where(
        df["total_pax"] > 0, df["pnrs_with_strollers"] / df["total_pax"], 0
    )

    df["carrier_mainline_flag"] = df["carrier"].fillna("").str.lower().eq("mainline").astype(int)
    df["carrier_express_flag"] = df["carrier"].fillna("").str.lower().eq("express").astype(int)

    df["is_wide_body"] = df["fleet_type"].fillna("").str.startswith(WIDE_BODY_PREFIXES).astype(int)
    df["is_international"] = df["arrival_country_code"].fillna("US").ne("US").astype(int)
    df["international_service_intensity"] = df["is_international"] * df["ssr_per_pnr"]

    df["departure_wave_bucket_code"] = df["departure_wave_bucket_code"].fillna(-1).astype(int)
    df["departure_daypart_code"] = df["departure_daypart_code"].fillna(-1).astype(int)
    df["station_departure_rank_pct"] = df["station_departure_rank_pct"].fillna(0.5)
    df["is_station_first_departure"] = df["is_station_first_departure"].fillna(False)
    df["is_station_last_departure"] = df["is_station_last_departure"].fillna(False)
    df["is_first_departure_of_day"] = df["is_first_departure_of_day"].fillna(False)
    df["minutes_since_first_departure"] = df["minutes_since_first_departure"].fillna(0)
    df["minutes_since_bank_start"] = df["minutes_since_bank_start"].fillna(0)

    df["wide_body_load_factor"] = df["is_wide_body"] * df["load_factor"].fillna(0)
    df["red_eye_flag"] = df["is_red_eye"].astype(int)
    df["transfer_pressure_flag"] = (
        (df["transfer_ratio"] >= 0.7) | (df["transfer_bags_per_pax"].fillna(0) >= 0.5)
    ).astype(int)

    delay_component = np.clip(df["positive_departure_delay"] / 30.0, 0, 1)
    service_component = np.clip(df["ssr_per_pnr"], 0, 0.2) / 0.2
    baggage_component = np.clip(df["transfer_bags_per_pax"].fillna(0), 0, 0.6) / 0.6
    gate_component = np.where(
        df["turn_shortfall_flag"],
        1.0,
        np.clip(1 - (df["scheduled_vs_min_turn_buffer"].clip(lower=0) / 60.0), 0, 1),
    )
    turn_component = np.clip(1 - df["actual_ground_to_scheduled_ratio"].fillna(1.0), 0, 1)

    df["difficulty_index"] = (
        0.4 * delay_component
        + 0.2 * service_component
        + 0.2 * baggage_component
        + 0.1 * gate_component
        + 0.1 * turn_component
    )
    df["difficulty_index"] = df["difficulty_index"].clip(0, 1)

    df["difficulty_index_rank"] = df.groupby("scheduled_departure_date_local")[
        "difficulty_index"
    ].rank(method="average", pct=True)

    turn_buffer = df["scheduled_vs_min_turn_buffer"].fillna(np.inf)
    transfer_ratio = df["transfer_ratio"].fillna(0)
    ssr_per_pnr = df["ssr_per_pnr"].fillna(0)
    pnr_pressure = df["pnr_pressure_index"].fillna(0)
    international_flag = df["is_international"].fillna(0).astype(int)
    wide_body_flag = df["is_wide_body"].fillna(0).astype(int)

    acute_turn_flag = (turn_buffer < 10) & (transfer_ratio > 0.6)
    acute_service_flag = ssr_per_pnr > 0.15

    difficulty_points = (
        ((turn_buffer >= 10) & (turn_buffer < 20)).astype(int) * 2
        + ((transfer_ratio >= 0.45) & (transfer_ratio < 0.6)).astype(int) * 2
        + ((ssr_per_pnr >= 0.10) & (ssr_per_pnr < 0.15)).astype(int) * 2
        + (pnr_pressure > 0.5).astype(int)
        + ((international_flag == 1) | (wide_body_flag == 1)).astype(int)
    )
    compounded_risk_flag = difficulty_points >= 5

    predicted_delay_proxy = df.get("predicted_departure_delay_minutes")
    if predicted_delay_proxy is None:
        predicted_delay_proxy = df["positive_departure_delay"].fillna(0)
    else:
        predicted_delay_proxy = predicted_delay_proxy.fillna(0)
    delay_risk_flag = predicted_delay_proxy > 25

    difficulty_mask = acute_turn_flag | acute_service_flag | compounded_risk_flag

    df["difficulty_points"] = difficulty_points.astype(int)
    df["difficulty_actual_flag"] = difficulty_mask.astype(int)
    df["delay_risk_secondary_flag"] = delay_risk_flag.astype(int)

    trigger_labels = []
    for turn_flag, ssr_flag, comp_flag, points_value, delay_flag in zip(
        acute_turn_flag.to_numpy(),
        acute_service_flag.to_numpy(),
        compounded_risk_flag.to_numpy(),
        difficulty_points.to_numpy(),
        delay_risk_flag.to_numpy(),
    ):
        reasons = []
        if turn_flag:
            reasons.append("Acute turn stress (<10m buffer & transfer >0.6)")
        if ssr_flag:
            reasons.append("Acute service stress (SSR >0.15)")
        if comp_flag:
            reasons.append(f"Compounded risk (points={int(points_value)})")
        elif points_value >= 3:
            reasons.append(f"Difficulty points score {int(points_value)}")
        if delay_flag:
            reasons.append("Delay risk >25m (secondary)")
        trigger_labels.append(", ".join(reasons) if reasons else "None")

    df["difficulty_trigger_reason"] = pd.Series(trigger_labels, index=df.index, dtype="string")

    delay_bins = [0, 15, 30, 90, np.inf]
    delay_labels = ["on_time", "minor", "moderate", "major"]
    df["delay_severity"] = pd.cut(
        df["positive_departure_delay"].fillna(0),
        bins=delay_bins,
        labels=delay_labels,
        right=False,
        include_lowest=True,
    ).astype("string")

    df["flight_identifier"] = (
        df["company_id"].astype(str)
        + "-"
        + df["flight_number"].astype(str)
        + "-"
        + df["scheduled_departure_date_local"].dt.strftime("%Y-%m-%d")
    )

    return df


def build_feature_frames(
    flight_level: pd.DataFrame,
    pnr_level: pd.DataFrame,
    pnr_remarks: pd.DataFrame,
    bag_level: pd.DataFrame,
    airports: pd.DataFrame,
) -> FlightFeatureFrames:
    console.log("Preparing flight level features")
    flight_df = prepare_flight_level(flight_level)

    console.log("Preparing PNR features")
    pnr_features, pnr_unique = prepare_pnr_features(pnr_level)

    console.log("Preparing SSR features")
    ssr_features = prepare_ssr_features(pnr_unique, pnr_remarks)

    console.log("Preparing bag features")
    bag_features = prepare_bag_features(bag_level)

    console.log("Combining feature tables")
    combined = combine_features(flight_df, pnr_features, ssr_features, bag_features, airports)

    PATHS.data_processed.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PATHS.data_processed / "flight_master_features.parquet", index=False)
    pnr_features.to_parquet(PATHS.data_processed / "pnr_flight_features.parquet", index=False)

    return FlightFeatureFrames(
        flight=combined,
        pnr=pnr_features,
        ssr=ssr_features,
        bag=bag_features,
    )
