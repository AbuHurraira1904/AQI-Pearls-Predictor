import pandas as pd
import logging

logger = logging.getLogger(__name__)

CLEANED_DATA_JSON = "cleaned_backfill.json"
REQUIRED_COLUMNS = ["timestamp", "hour", "day_of_week", "month", "temperature", "humidity", "aqi_change_rate", "aqi", "pm25", "pm10"]

NON_NULLABLE_COLUMNS = ["timestamp", "hour", "day_of_week", "month", "temperature", "humidity", "aqi", "pm25", "pm10"]


HORIZON_HOURS = [24, 48, 72]

TARGET_MATCH_TOLERANCE = pd.Timedelta(minutes=45)

def get_cleaned_feature_data():
    df = pd.read_json(CLEANED_DATA_JSON, convert_dates=False)
    logger.info("Loaded %d raw rows from %s", len(df), CLEANED_DATA_JSON)

    return df

def validate_schema(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"data_preparation: missing required column(s): {missing}. "
            f"Got columns: {list(df.columns)}"
        )
 
    for col in NON_NULLABLE_COLUMNS:
        n_null = df[col].isna().sum()
        if n_null:
            raise ValueError(
                f"data_preparation: column '{col}' has {n_null} unexpected "
                f"null value(s). This column should never be null — check "
                f"upstream cleaning."
            )

    non_numeric = [
        c for c in REQUIRED_COLUMNS
        if c != "timestamp" and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(
            f"data_preparation: expected numeric dtype for {non_numeric}, "
            f"got: {[str(df[c].dtype) for c in non_numeric]}"
        )
 
    logger.info("Schema validation passed (%d rows)", len(df))

def add_horizon_target(df, horizon_hours):
    col_name = f"aqi_target_{horizon_hours}h"

    base = df[["dt"]].copy().sort_values("dt").reset_index(drop=True)
    base["lookup_time"] = (base["dt"] + pd.Timedelta(hours=horizon_hours)).astype(base["dt"].dtype)
 
    future = df[["dt", "aqi"]].rename(
        columns={"dt": "future_dt", "aqi": col_name}
    ).sort_values("future_dt").reset_index(drop=True)
    future["future_dt"] = future["future_dt"].astype(base["dt"].dtype)
 
    matched = pd.merge_asof(
        base.sort_values("lookup_time"),
        future,
        left_on="lookup_time",
        right_on="future_dt",
        direction="nearest",
        tolerance=TARGET_MATCH_TOLERANCE,
    )
 
    matched = matched.set_index("dt")[col_name]
    result = df.copy()
    result[col_name] = result["dt"].map(matched)
 
    n_matched = result[col_name].notna().sum()
    logger.info(
        "%s: matched %d/%d rows (tolerance=%s)",
        col_name, n_matched, len(result), TARGET_MATCH_TOLERANCE,
    )
    return result



def build_horizon_targets(df):
    df = df.copy()

    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    for h in HORIZON_HOURS:
        df = add_horizon_target(df, h)

    return df

def prepare_training_data(df):
    validate_schema(df)
    df = build_horizon_targets(df)
    return df