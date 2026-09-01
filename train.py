import logging
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
logger = logging.getLogger(__name__)
 
FEATURE_COLUMNS = [
    "hour", "day_of_week", "month",
    "temperature", "humidity", "aqi_change_rate", "aqi_change_rate_missing",
    "aqi", "pm25", "pm10",
]

CHANGE_RATE_FILL_VALUE = 0.0
 
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 is test
 
 
def _target_col(horizon_hours):
    return f"aqi_target_{horizon_hours}h"
 
 
def _impute_change_rate(df):
    df = df.copy()
    df["aqi_change_rate_missing"] = df["aqi_change_rate"].isna().astype(int)
    df["aqi_change_rate"] = df["aqi_change_rate"].fillna(CHANGE_RATE_FILL_VALUE)
    return df
 
 
def prepare_horizon_dataset(df, horizon_hours):
    target_col = _target_col(horizon_hours)
    if target_col not in df.columns:
        raise ValueError(
            f"train.py: expected column '{target_col}' — did you run "
            f"data_preparation.prepare_training_data() first?"
        )
 
    before = len(df)
    df = df[df[target_col].notna()].copy()
    df = _impute_change_rate(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    dropped = before - len(df)
    n_imputed = df["aqi_change_rate_missing"].sum()
    logger.info(
        "Horizon %dh: dropped %d/%d rows with no target match, %d remain "
        "(%d had aqi_change_rate imputed)",
        horizon_hours, dropped, before, len(df), n_imputed,
    )
    return df
 
 
def chronological_split(df):
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
 
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
 
    logger.info(
        "Split: train=%d, val=%d, test=%d (chronological)",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df
 
 
def _xy(df, horizon_hours):
    X = df[FEATURE_COLUMNS]
    y = df[_target_col(horizon_hours)]
    return X, y
 
 
def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }
 
 
def train_candidates(train_df, val_df, horizon_hours):
    X_train, y_train = _xy(train_df, horizon_hours)
    X_val, y_val = _xy(val_df, horizon_hours)
 
    candidates = {
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }
 
    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_metrics = evaluate(model, X_val, y_val)
        results[name] = {"model": model, "val_metrics": val_metrics}
        logger.info("[%s] val metrics: %s", name, val_metrics)
 
    winner_name = min(results, key=lambda k: results[k]["val_metrics"]["rmse"])
    logger.info(
        "Winner (lowest val RMSE): %s (RMSE=%.4f)",
        winner_name, results[winner_name]["val_metrics"]["rmse"],
    )
    return winner_name, results
 
 
def train_horizon(df, horizon_hours):
    horizon_df = prepare_horizon_dataset(df, horizon_hours)
    train_df, val_df, test_df = chronological_split(horizon_df)
 
    winner_name, candidate_results = train_candidates(train_df, val_df, horizon_hours)
    winner_model = candidate_results[winner_name]["model"]
 
    X_test, y_test = _xy(test_df, horizon_hours)
    test_metrics = evaluate(winner_model, X_test, y_test)
    logger.info(
        "[%s] FINAL test metrics (horizon=%dh): %s",
        winner_name, horizon_hours, test_metrics,
    )
 
    return {
        "horizon_hours": horizon_hours,
        "winner_name": winner_name,
        "winner_model": winner_model,
        "test_metrics": test_metrics,
        "val_metrics": candidate_results[winner_name]["val_metrics"],
        "all_candidates": {
            name: r["val_metrics"] for name, r in candidate_results.items()
        },
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }

HORIZONS_HOURS = [24, 48, 72]

def train_all_horizons(df):
    results = {}
    for h in HORIZONS_HOURS:
        logger.info("=== Training horizon: %dh ===", h)
        results[h] = train_horizon(df, horizon_hours=h)
 
    logger.info("=== Summary (test metrics, winning model per horizon) ===")
    for h, r in results.items():
        logger.info(
            "  %2dh: winner=%-13s rmse=%.3f mae=%.3f r2=%.4f (n_train=%d, n_val=%d, n_test=%d)",
            h, r["winner_name"], r["test_metrics"]["rmse"],
            r["test_metrics"]["mae"], r["test_metrics"]["r2"],
            r["n_train"], r["n_val"], r["n_test"],
        )
    return results
