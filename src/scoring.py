from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import dump
from rich.console import Console
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from .config import PATHS

console = Console()

MODEL_FEATURES: List[str] = [
    "total_seats",
    "scheduled_ground_time_minutes",
    "minimum_turn_minutes",
    "scheduled_vs_min_turn_buffer",
    "actual_vs_min_turn_buffer",
    "scheduled_ground_to_min_ratio",
    "actual_ground_to_scheduled_ratio",
    "turn_shortfall_flag",
    "departure_hour_local",
    "departure_minutes_from_midnight",
    "departure_daypart_code",
    "departure_hour_sin",
    "departure_hour_cos",
    "departure_wave_bucket_code",
    "station_departure_rank_pct",
    "minutes_since_first_departure",
    "minutes_since_bank_start",
    "is_station_first_departure",
    "is_station_last_departure",
    "is_first_departure_of_day",
    "day_of_week",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "red_eye_flag",
    "carrier_mainline_flag",
    "carrier_express_flag",
    "is_wide_body",
    "is_international",
    "international_service_intensity",
    "wide_body_load_factor",
    "total_pax",
    "load_factor",
    "avg_pnr_size",
    "share_pnr_with_children",
    "share_pnr_with_strollers",
    "share_basic_economy_pnr",
    "share_short_lead_pnr",
    "share_ultra_short_lead_pnr",
    "pnr_pressure_index",
    "transfer_pressure_flag",
    "total_ssr_requests",
    "ssr_per_pnr",
    "ssr_airport_wheelchair",
    "ssr_manual_wheelchair",
    "ssr_unaccompanied_minor",
    "ssr_electric_wheelchair",
    "total_bags",
    "origin_bags",
    "all_transfer_bags",
    "transfer_ratio",
    "hot_transfer_ratio",
    "bags_per_pax",
    "transfer_bags_per_pax",
    "hot_transfer_bags_per_pax",
    "lap_child_count",
    "pnr_short_lead_count",
    "pnr_ultra_short_lead_count",
    "child_ratio",
    "stroller_ratio",
    "positive_arrival_delay",
]

BOOLEAN_FEATURES: List[str] = [
    "turn_shortfall_flag",
    "is_weekend",
    "is_station_first_departure",
    "is_station_last_departure",
    "is_first_departure_of_day",
    "red_eye_flag",
    "carrier_mainline_flag",
    "carrier_express_flag",
    "is_wide_body",
    "is_international",
    "transfer_pressure_flag",
]

TARGET_COLUMN = "difficulty_actual_flag"
RECALL_TARGET = 0.7
BETA = 2.0
RULE_BOOST = 0.2


@dataclass
class ModelOutputs:
    feature_frame: pd.DataFrame
    model: HistGradientBoostingClassifier
    calibrator: CalibratedClassifierCV
    threshold: float
    metrics: Dict[str, float]


def _prepare_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()
    for boolean_col in BOOLEAN_FEATURES:
        if boolean_col in model_df.columns:
            model_df[boolean_col] = model_df[boolean_col].astype(int)

    X = model_df[MODEL_FEATURES].copy()
    y = model_df[TARGET_COLUMN].astype(int)
    return model_df, X, y


def _rule_based_boost(df: pd.DataFrame) -> np.ndarray:
    def _smoothstep(array: np.ndarray) -> np.ndarray:
        clipped = np.clip(array, 0.0, 1.0)
        return clipped * clipped * (3 - 2 * clipped)

    turn_buffer = df["scheduled_vs_min_turn_buffer"].fillna(30.0).to_numpy(dtype=float)
    transfer_ratio = df["transfer_ratio"].fillna(0.0).to_numpy(dtype=float)
    ssr_per_pnr = df["ssr_per_pnr"].fillna(0.0).to_numpy(dtype=float)
    pnr_pressure = df["pnr_pressure_index"].fillna(0.0).to_numpy(dtype=float)

    turn_buffer_score = _smoothstep((20.0 - turn_buffer) / 10.0)
    transfer_score = _smoothstep((transfer_ratio - 0.45) / 0.15)
    turn_combo = turn_buffer_score * transfer_score

    ssr_score = _smoothstep((ssr_per_pnr - 0.10) / 0.05)
    pnr_score = _smoothstep((pnr_pressure - 0.50) / 0.20)

    difficulty_points = df.get("difficulty_points")
    if difficulty_points is not None:
        point_array = difficulty_points.to_numpy(dtype=float)
        point_score = _smoothstep((point_array - 3.0) / 2.0)
    else:
        point_score = np.zeros_like(turn_combo)

    combined_risk = np.maximum.reduce([turn_combo, ssr_score, pnr_score, point_score])
    return RULE_BOOST * combined_risk


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(scores, np.linspace(0.05, 0.95, 19))
    manual = np.array([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    candidates = np.clip(np.unique(np.concatenate([quantiles, manual])), 0, 1)
    return candidates


def _determine_threshold(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    candidates = _candidate_thresholds(scores)
    best_threshold = 0.5
    best_score = -np.inf
    best_precision = 0.0
    best_recall = 0.0

    overall_best = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f2": 0.0,
    }

    for threshold in candidates:
        preds = (scores >= threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f2 = fbeta_score(y_true, preds, beta=BETA, zero_division=0)

        if f2 > overall_best["f2"]:
            overall_best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f2": float(f2),
            }

        if recall >= RECALL_TARGET and f2 > best_score:
            best_threshold = float(threshold)
            best_score = f2
            best_precision = precision
            best_recall = recall

    if best_score < 0:
        best_threshold = overall_best["threshold"]
        best_precision = overall_best["precision"]
        best_recall = overall_best["recall"]
        best_score = overall_best["f2"]

    return {
        "threshold": float(best_threshold),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "f2": float(best_score),
    }


def _compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(int)
    metrics = {
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "f2": float(fbeta_score(y_true, preds, beta=BETA, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "positive_count": int(y_true.sum()),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_difficulty_model(df: pd.DataFrame) -> ModelOutputs:
    model_df, X, y = _prepare_model_matrix(df)
    dates = model_df["scheduled_departure_date_local"].copy()

    unique_dates = np.sort(dates.dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("Insufficient distinct dates to perform temporal validation.")

    cutoff_index = len(unique_dates) // 2
    cutoff_date = unique_dates[cutoff_index - 1]
    cutoff_ts = pd.Timestamp(cutoff_date)

    train_mask = dates <= cutoff_ts
    test_mask = ~train_mask

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    if X_test.empty:
        raise ValueError("Temporal split resulted in empty test set. Check date coverage.")

    n_groups = dates.loc[train_mask].nunique()
    n_splits = max(2, min(5, int(n_groups)))

    cv_pred = np.zeros(len(X_train))
    gkf = GroupKFold(n_splits=n_splits)

    console.log(
        f"Training difficulty model on {len(X_train)} flights (cutoff {cutoff_ts.date()})"
    )

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(X_train, y_train, groups=dates.loc[train_mask])
    , start=1):
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=5,
            max_iter=600,
            min_samples_leaf=50,
            l2_regularization=1.0,
            class_weight={0: 1.0, 1: 2.0},
            random_state=fold + 41,
        )
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        cv_pred[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]

    threshold_info = _determine_threshold(y_train.to_numpy(), cv_pred)
    threshold = threshold_info["threshold"]

    final_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_depth=5,
        max_iter=600,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight={0: 1.0, 1: 2.0},
        random_state=42,
    )
    final_model.fit(X_train, y_train)

    train_scores_raw = final_model.predict_proba(X_train)[:, 1]
    test_scores_raw = final_model.predict_proba(X_test)[:, 1]

    calibrator = CalibratedClassifierCV(final_model, method="isotonic", cv="prefit")
    calibrator.fit(X_test, y_test)

    train_scores = calibrator.predict_proba(X_train)[:, 1]
    test_scores = calibrator.predict_proba(X_test)[:, 1]

    cv_roc_auc = roc_auc_score(y_train, cv_pred)
    cv_pr_auc = average_precision_score(y_train, cv_pred)

    rule_train = _rule_based_boost(model_df.loc[train_mask])
    rule_test = _rule_based_boost(model_df.loc[test_mask])

    blended_train = np.clip(train_scores + RULE_BOOST * rule_train, 0, 1)
    blended_test = np.clip(test_scores + RULE_BOOST * rule_test, 0, 1)

    train_metrics = _compute_metrics(y_train.to_numpy(), blended_train, threshold)
    y_test_np = y_test.to_numpy()
    test_metrics_raw = _compute_metrics(y_test_np, test_scores_raw, threshold)
    test_metrics_calibrated = _compute_metrics(y_test_np, test_scores, threshold)
    test_metrics_blended = _compute_metrics(y_test_np, blended_test, threshold)

    test_preds_raw = (test_scores_raw >= threshold).astype(int)
    test_preds_blended = (blended_test >= threshold).astype(int)
    blended_true_positives = int(((test_preds_blended == 1) & (y_test_np == 1)).sum())
    blended_predicted_positive = int(test_preds_blended.sum())
    blended_false_positive = int(((test_preds_blended == 1) & (y_test_np == 0)).sum())
    blended_false_negative = int(((test_preds_blended == 0) & (y_test_np == 1)).sum())

    metrics: Dict[str, float] = {
        "positive_rate": float(y.mean()),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "validation_cutoff_date": str(cutoff_ts.date()),
        "cv_roc_auc": float(cv_roc_auc),
        "cv_pr_auc": float(cv_pr_auc),
        "threshold": threshold,
        "threshold_precision": threshold_info["precision"],
        "threshold_recall": threshold_info["recall"],
        "threshold_f2": threshold_info["f2"],
        "rule_boost": RULE_BOOST,
        "baseline_holdout_positive_rate": float(y_test.mean()),
        "holdout_actual_delay_rate": float(
            (model_df.loc[test_mask, "positive_departure_delay"] >= 15).mean()
        ),
    }

    metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
    metrics.update({f"test_raw_{k}": v for k, v in test_metrics_raw.items()})
    metrics.update({f"test_calibrated_{k}": v for k, v in test_metrics_calibrated.items()})
    metrics.update({f"test_blended_{k}": v for k, v in test_metrics_blended.items()})

    metrics.update(
        {
            "test_blended_true_positives": blended_true_positives,
            "test_blended_predicted_positive_count": blended_predicted_positive,
            "test_blended_false_positive_count": blended_false_positive,
            "test_blended_false_negative_count": blended_false_negative,
        }
    )

    model_df = model_df.copy()
    full_raw_scores = final_model.predict_proba(X)[:, 1]
    full_calibrated_scores = calibrator.predict_proba(X)[:, 1]
    model_df["difficulty_probability_raw_model"] = full_raw_scores
    model_df["difficulty_probability_model"] = full_calibrated_scores
    model_df["difficulty_probability_cv"] = np.nan
    model_df.loc[train_mask, "difficulty_probability_cv"] = cv_pred
    model_df["dataset_split"] = np.where(train_mask, "train", "test")

    return ModelOutputs(
        feature_frame=model_df,
        model=final_model,
        calibrator=calibrator,
        threshold=threshold,
        metrics=metrics,
    )


def _assign_daily_rankings(df: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    df = df.copy()

    df["difficulty_percentile"] = df.groupby("scheduled_departure_date_local")[
        probability_column
    ].rank(method="average", pct=True)

    df = df.sort_values(
        ["scheduled_departure_date_local", probability_column],
        ascending=[True, False],
    )
    df["daily_rank"] = df.groupby("scheduled_departure_date_local").cumcount() + 1

    df["difficulty_class"] = pd.Series("Easy", index=df.index, dtype="string")
    difficult_mask = df[TARGET_COLUMN] == 1

    difficulty_points = df.get("difficulty_points")
    if difficulty_points is None:
        points_series = pd.Series(0, index=df.index)
    else:
        points_series = difficulty_points

    medium_candidates = (
        (df[probability_column] >= 0.5)
        | ((points_series >= 3) & (points_series <= 4))
    )
    medium_mask = (~difficult_mask) & medium_candidates

    df.loc[medium_mask, "difficulty_class"] = "Medium"
    df.loc[difficult_mask, "difficulty_class"] = "Difficult"

    return df


def _format_top_drivers(shap_matrix: np.ndarray, feature_names: List[str], top_k: int = 3) -> List[str]:
    abs_values = np.abs(shap_matrix)
    drivers: List[str] = []
    for idx in range(shap_matrix.shape[0]):
        top_indices = np.argsort(abs_values[idx])[::-1][:top_k]
        parts = []
        for feature_idx in top_indices:
            impact = shap_matrix[idx, feature_idx]
            direction = "↑" if impact >= 0 else "↓"
            parts.append(f"{feature_names[feature_idx]} {direction}{abs(impact):.2f}")
        drivers.append("; ".join(parts))
    return drivers


def _save_cost_benefit_analysis(scored_df: pd.DataFrame, metrics: Dict[str, float]) -> None:
    holdout_df = scored_df.loc[scored_df["dataset_split"] == "test"].copy()
    total = len(holdout_df)
    if total == 0:
        return

    baseline_rate = metrics.get(
        "baseline_holdout_positive_rate",
        float(holdout_df[TARGET_COLUMN].mean()),
    )
    actual_delay_rate = metrics.get(
        "holdout_actual_delay_rate",
        float((holdout_df["positive_departure_delay"] >= 15).mean()),
    )

    false_negatives = int(
        ((holdout_df["difficulty_prediction"] == 0) & (holdout_df[TARGET_COLUMN] == 1)).sum()
    )
    false_positives = int(
        ((holdout_df["difficulty_prediction"] == 1) & (holdout_df[TARGET_COLUMN] == 0)).sum()
    )
    true_positives = int(
        ((holdout_df["difficulty_prediction"] == 1) & (holdout_df[TARGET_COLUMN] == 1)).sum()
    )

    false_negative_rate = false_negatives / total
    false_positive_rate = false_positives / total

    severity_costs = {
        "minor": 1000,
        "moderate": 4000,
        "major": 12000,
    }
    intervention_costs = {"light": 150, "medium": 400}
    flights_per_day_assumptions = {"weekday": 380, "peak": 420}

    holdout_df["delay_severity"] = pd.cut(
        holdout_df["positive_departure_delay"].fillna(0),
        bins=[0, 15, 30, 90, np.inf],
        labels=["on_time", "minor", "moderate", "major"],
        right=False,
        include_lowest=True,
    ).astype("string")

    severity_cost_map = {
        "on_time": 0.0,
        "minor": float(severity_costs["minor"]),
        "moderate": float(severity_costs["moderate"]),
        "major": float(severity_costs["major"]),
    }
    holdout_df["severity_cost"] = holdout_df["delay_severity"].map(severity_cost_map).fillna(0.0)

    light_intervention_mask = (
        (holdout_df["scheduled_vs_min_turn_buffer"].fillna(np.inf) >= 20)
        & (holdout_df["transfer_ratio"].fillna(0) <= 0.3)
        & (holdout_df["ssr_per_pnr"].fillna(0) <= 0.08)
    )
    holdout_df["intervention_level"] = pd.Series(
        np.where(light_intervention_mask, "light", "medium"), index=holdout_df.index
    ).astype("string")
    holdout_df["intervention_cost"] = holdout_df["intervention_level"].map(
        {"light": float(intervention_costs["light"]), "medium": float(intervention_costs["medium"])}
    )

    positive_mask = holdout_df[TARGET_COLUMN] == 1
    severity_positive = holdout_df.loc[positive_mask, "severity_cost"]
    avg_fn_cost = float(severity_positive.mean()) if not severity_positive.empty else 2500.0

    fp_mask = (holdout_df["difficulty_prediction"] == 1) & (holdout_df[TARGET_COLUMN] == 0)
    fn_mask = (holdout_df["difficulty_prediction"] == 0) & positive_mask

    intervention_positive = holdout_df.loc[fp_mask, "intervention_cost"]
    avg_fp_cost = float(intervention_positive.mean()) if not intervention_positive.empty else 350.0

    unique_dates = (
        holdout_df["scheduled_departure_date_local"].dt.normalize().dropna().unique()
    )
    total_days = len(unique_dates)
    if total_days > 0:
        weekday_days = int(sum(pd.Timestamp(day).weekday() < 5 for day in unique_dates))
        peak_days = int(total_days - weekday_days)
        weighted_flights = (
            weekday_days * flights_per_day_assumptions["weekday"]
            + peak_days * flights_per_day_assumptions["peak"]
        )
        expected_flights_per_day = weighted_flights / total_days
    else:
        weekday_days = peak_days = 0
        expected_flights_per_day = float(
            (flights_per_day_assumptions["weekday"] + flights_per_day_assumptions["peak"]) / 2
        )

    baseline_cost = baseline_rate * expected_flights_per_day * avg_fn_cost
    model_cost = (
        false_negative_rate * expected_flights_per_day * avg_fn_cost
        + false_positive_rate * expected_flights_per_day * avg_fp_cost
    )

    def _severity_counts(mask: pd.Series) -> Dict[str, int]:
        counts = holdout_df.loc[mask, "delay_severity"].value_counts()
        return {str(k): int(v) for k, v in counts.items()}

    all_mask = pd.Series(True, index=holdout_df.index)

    intervention_mix = holdout_df.loc[fp_mask, "intervention_level"].value_counts()
    intervention_counts = {str(k): int(v) for k, v in intervention_mix.items()}

    delay_source = (
        "predicted_departure_delay_minutes"
        if "predicted_departure_delay_minutes" in holdout_df.columns
        else "positive_departure_delay"
    )

    results = {
        "assumptions": {
            "delay_severity_costs": severity_costs,
            "intervention_costs": intervention_costs,
            "flights_per_day": {
                "weekday": flights_per_day_assumptions["weekday"],
                "peak": flights_per_day_assumptions["peak"],
                "weighted_average": float(expected_flights_per_day),
                "observed_days": {
                    "total": int(total_days),
                    "weekday": int(weekday_days),
                    "peak": int(peak_days),
                },
            },
            "delay_risk_minutes_source": delay_source,
        },
        "baseline_delay_rate": baseline_rate,
        "actual_delay_rate": actual_delay_rate,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positives / total,
        "average_cost_per_false_negative": avg_fn_cost,
        "average_cost_per_false_positive": avg_fp_cost,
        "expected_daily_cost_baseline": baseline_cost,
        "expected_daily_cost_model": model_cost,
        "expected_daily_savings": baseline_cost - model_cost,
        "expected_annual_savings": (baseline_cost - model_cost) * 365,
        "daily_false_negative_cost": false_negative_rate * expected_flights_per_day * avg_fn_cost,
        "daily_false_positive_cost": false_positive_rate * expected_flights_per_day * avg_fp_cost,
        "holdout_flight_count": total,
        "holdout_positive_count": int(positive_mask.sum()),
        "holdout_false_negative_count": false_negatives,
        "holdout_false_positive_count": false_positives,
        "holdout_true_positive_count": true_positives,
        "holdout_false_negative_cost_total": float(holdout_df.loc[fn_mask, "severity_cost"].sum()),
        "holdout_false_positive_cost_total": float(holdout_df.loc[fp_mask, "intervention_cost"].sum()),
        "severity_breakdown": {
            "all_flights": _severity_counts(all_mask),
            "positives": _severity_counts(positive_mask),
            "false_negatives": _severity_counts(fn_mask),
            "false_positives": _severity_counts(fp_mask),
        },
        "intervention_mix_false_positive": intervention_counts,
    }

    cost_path = PATHS.artifacts_tables / "cost_benefit_analysis.json"
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    with cost_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)


def apply_scoring(df: pd.DataFrame, model_outputs: ModelOutputs) -> pd.DataFrame:
    base_df = model_outputs.feature_frame.copy()
    raw_scores = model_outputs.model.predict_proba(base_df[MODEL_FEATURES])[:, 1]
    calibrated_scores = model_outputs.calibrator.predict_proba(base_df[MODEL_FEATURES])[:, 1]
    base_df["difficulty_probability_raw_model"] = raw_scores
    base_df["difficulty_probability_model"] = calibrated_scores

    rule_boost = _rule_based_boost(base_df)
    base_df["rule_based_boost"] = rule_boost
    base_df["rule_based_flag"] = (rule_boost > 0).astype(int)
    base_df["difficulty_probability_blended"] = np.clip(
        base_df["difficulty_probability_model"] + rule_boost, 0, 1
    )
    base_df["difficulty_prediction"] = (
        base_df["difficulty_probability_blended"] >= model_outputs.threshold
    ).astype(int)

    shap_explainer = shap.Explainer(
        model_outputs.model, base_df[MODEL_FEATURES], feature_names=MODEL_FEATURES
    )
    shap_values = shap_explainer(base_df[MODEL_FEATURES], check_additivity=False)
    shap_matrix = shap_values.values
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, 1]

    base_df["top_difficulty_drivers"] = _format_top_drivers(shap_matrix, MODEL_FEATURES)

    feature_importance = np.mean(np.abs(shap_matrix), axis=0)
    importance_df = (
        pd.DataFrame({"feature": MODEL_FEATURES, "mean_abs_shap": feature_importance})
        .sort_values("mean_abs_shap", ascending=False)
    )
    importance_path = PATHS.artifacts_tables / "feature_importances.csv"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_path, index=False)

    try:
        if shap_matrix.shape[0] > 2000:
            rng = np.random.default_rng(seed=42)
            sample_idx = rng.choice(shap_matrix.shape[0], size=2000, replace=False)
            shap_for_plot = shap_matrix[sample_idx]
            feature_sample = base_df.iloc[sample_idx][MODEL_FEATURES]
        else:
            shap_for_plot = shap_matrix
            feature_sample = base_df[MODEL_FEATURES]

        shap.dependence_plot(
            "scheduled_vs_min_turn_buffer",
            shap_for_plot,
            feature_sample,
            interaction_index="transfer_bags_per_pax",
            show=False,
        )
        plt.tight_layout()
        dep_path = PATHS.artifacts_figures / "shap_turn_buffer_vs_transfer.png"
        dep_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dep_path, dpi=200)
    except Exception as exc:
        console.log(f"Could not generate SHAP dependence plot: {exc}")
    finally:
        plt.close()

    base_df = _assign_daily_rankings(base_df, "difficulty_probability_blended")

    output_columns = [
        "company_id",
        "flight_number",
        "scheduled_departure_date_local",
        "scheduled_departure_station_code",
        "scheduled_arrival_station_code",
        "carrier",
        "fleet_type",
        "dataset_split",
        "difficulty_actual_flag",
        "difficulty_points",
        "difficulty_trigger_reason",
        "delay_risk_secondary_flag",
        "delay_severity",
        "difficulty_index",
        "difficulty_index_rank",
        "difficulty_probability_raw_model",
        "difficulty_probability_model",
        "difficulty_probability_blended",
        "rule_based_boost",
        "rule_based_flag",
        "difficulty_prediction",
        "difficulty_percentile",
        "daily_rank",
        "difficulty_class",
        "top_difficulty_drivers",
        "total_seats",
        "scheduled_ground_time_minutes",
        "minimum_turn_minutes",
        "scheduled_vs_min_turn_buffer",
        "turn_shortfall_flag",
        "total_pax",
        "load_factor",
        "transfer_ratio",
        "transfer_bags_per_pax",
        "pnr_pressure_index",
        "ssr_per_pnr",
        "total_bags",
        "all_transfer_bags",
        "bags_per_pax",
        "is_international",
        "is_wide_body",
        "red_eye_flag",
        "departure_daypart_code",
        "departure_wave_bucket_code",
        "station_departure_rank_pct",
        "minutes_since_first_departure",
        "minutes_since_bank_start",
        "is_first_departure_of_day",
        "departure_hour_local",
        "departure_minutes_from_midnight",
        "departure_hour_sin",
        "departure_hour_cos",
        "day_of_week",
        "day_of_week_sin",
        "day_of_week_cos",
        "carrier_mainline_flag",
        "carrier_express_flag",
        "international_service_intensity",
        "wide_body_load_factor",
    ]

    available_columns = [col for col in output_columns if col in base_df.columns]
    export_df = base_df[available_columns].copy()

    export_df = export_df.sort_values(
        ["scheduled_departure_date_local", "difficulty_probability_blended"],
        ascending=[True, False],
    )

    PATHS.output_score.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(PATHS.output_score, index=False)

    metrics_path = Path(PATHS.artifacts_tables) / "model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(model_outputs.metrics, fp, indent=2)

    model_path = Path(PATHS.artifacts_models) / "difficulty_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model_outputs.model, "calibrator": model_outputs.calibrator}, model_path)

    _save_cost_benefit_analysis(base_df, model_outputs.metrics)

    console.log(f"Difficulty score exported to {PATHS.output_score}")

    return base_df
