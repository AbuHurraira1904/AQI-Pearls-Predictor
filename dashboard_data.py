import logging

import joblib
import pandas as pd

from feast_store import get_latest_row, get_all_rows
from hopsworks_client import get_model_registry
from register import get_current_champion_model, load_model
from train import FEATURE_COLUMNS, CHANGE_RATE_FILL_VALUE, HORIZONS_HOURS

logger = logging.getLogger(__name__)

TRAINING_RESULTS_PATH = "training_results.pkl"

# Standard US EPA AQI hazard bands.
HAZARD_BANDS = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


def classify_hazard(aqi_value):
    """Returns (label, color) for a given AQI value, per EPA bands above."""
    if aqi_value is None or pd.isna(aqi_value):
        return "Unknown", "#808080"
    for low, high, label, color in HAZARD_BANDS:
        if low <= aqi_value <= high:
            return label, color
    # Above 500 -- off the standard scale, still treat as Hazardous.
    if aqi_value > 500:
        return "Hazardous", "#7e0023"
    return "Unknown", "#808080"


def fetch_latest_features():
    """
    Latest single feature row from Feast, as a pandas Series. Returns None
    if the store is empty (e.g. first-ever run, or parquet not yet
    populated) so the dashboard can show a clear "no data yet" state
    instead of crashing.
    """
    latest = get_latest_row()
    if latest is None:
        logger.warning("No latest row available from Feast.")
        return None
    return latest


def fetch_historical_features(lookback_hours=None):
    """
    Full (or trimmed) history from Feast for the trend chart, sorted
    ascending by time (get_all_rows returns descending, matching store.py's
    convention -- this function flips it since a trend chart reads left to
    right chronologically).
    """
    df = get_all_rows()
    if df.empty:
        logger.warning("No historical rows available from Feast.")
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)
    if lookback_hours is not None:
        cutoff = df["timestamp"].max() - lookback_hours * 3600 * 1000
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df


def fetch_champion_models():
    """
    Loads the current champion model for each horizon from the Hopsworks
    model registry. Returns {horizon_hours: model_or_None} -- a horizon
    with no registered champion yet (e.g. never promoted) maps to None
    rather than raising, so the dashboard can show "not available yet"
    per-horizon instead of failing the whole page.
    """
    mr = get_model_registry()
    models = {}
    for h in HORIZONS_HOURS:
        champion_hw_model = get_current_champion_model(mr, h)
        if champion_hw_model is None:
            logger.warning("No champion model registered yet for horizon %dh", h)
            models[h] = None
            continue
        models[h] = load_model(champion_hw_model)
    return models


def load_training_results():
    """
    Loads the pickled training_results dict saved by main_train.py, used
    for on-demand SHAP regeneration in the dashboard. Returns None (rather
    than raising) if the file isn't present, so the dashboard can show a
    clear message instead of crashing -- e.g. right after a fresh clone
    where main_train.py hasn't run yet.
    """
    try:
        return joblib.load(TRAINING_RESULTS_PATH)
    except FileNotFoundError:
        logger.warning("%s not found -- SHAP regeneration unavailable until main_train.py runs.", TRAINING_RESULTS_PATH)
        return None


def build_feature_vector(latest_row):
    """
    Turns the latest Feast feature row into the exact single-row DataFrame
    train.py's models expect -- same FEATURE_COLUMNS, same
    aqi_change_rate_missing / fill-value handling as
    train._impute_change_rate(), reused directly from train.py rather than
    reimplemented here so the two can't silently drift apart.
    """
    row = latest_row.copy()

    change_rate_missing = int(pd.isna(row.get("aqi_change_rate")))
    change_rate = row.get("aqi_change_rate")
    if pd.isna(change_rate):
        change_rate = CHANGE_RATE_FILL_VALUE

    feature_values = {
        "hour": row["hour"],
        "day_of_week": row["day_of_week"],
        "month": row["month"],
        "temperature": row["temperature"],
        "humidity": row["humidity"],
        "aqi_change_rate": change_rate,
        "aqi_change_rate_missing": change_rate_missing,
        "aqi": row["aqi"],
        "pm25": row["pm25"],
        "pm10": row["pm10"],
    }

    X = pd.DataFrame([feature_values])[FEATURE_COLUMNS]
    return X


def predict_all_horizons(latest_row, models):
    """
    Returns {horizon_hours: predicted_aqi_or_None}. A horizon maps to None
    if there's no champion model registered for it yet.
    """
    X = build_feature_vector(latest_row)

    predictions = {}
    for h, model in models.items():
        if model is None:
            predictions[h] = None
            continue
        pred = model.predict(X)[0]
        predictions[h] = float(pred)
        logger.info("Horizon %dh prediction: %.2f", h, pred)

    return predictions