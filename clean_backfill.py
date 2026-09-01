from store import get_feature_store, get_latest_row, get_all_rows
import pandas as pd
import logging

FEATURE_GROUP_NAME = "hourly_city_aqi"
FEATURE_GROUP_VERSION = 1

TEMPORARY_JSON = "temporary_fetch_group_storage.json"
OUTPUT_JSON = "cleaned_backfill.json"

MAX_GAP_HOURS_FOR_RATE = 3.0

logger = logging.getLogger("BackfillCleaner")

def drop_null_change_rate_rows(df):
    length_before = len(df)

    df = df[df["aqi_change_rate"].notna()].copy()
    dropped_rows = length_before - len(df)

    if (dropped_rows):
        logger.info("Dropped %d row(s) with null aqi_change_rate "
            "(cold-start / timezone-corrupted row)",
            dropped_rows,
        )

    return df

def dedup_by_hour(df, source_priority=None):
    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["hour_bucket"] = df["dt"].dt.floor("h")

    df["offset_from_hour"] = (df["dt"] - df["hour_bucket"]).dt.total_seconds()

    sort_cols = ["offset_from_hour"]
    if "source" in df.columns:
        priority = source_priority or {"hopsworks": 0, "openaq": 1}
        df["_source_rank"] = df["source"].map(priority).fillna(99)
        sort_cols = ["_source_rank", "offset_from_hour"]


    before = len(df)
    df = (
        df.sort_values(sort_cols)
        .drop_duplicates(subset="hour_bucket", keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if "_source_rank" in df.columns:
        df = df.drop(columns=["_source_rank"])

    dropped = before - len(df)
    if dropped:
        logger.info(
            "Dropped %d duplicate same-hour row(s), keeping the reading "
            "closest to the top of each hour",
            dropped,
        )
    return df


def recompute_change_rate(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
 
    prev_aqi = df["aqi"].shift(1)
    prev_ts = df["dt"].shift(1)
    elapsed_hours = (df["dt"] - prev_ts).dt.total_seconds() / 3600.0
 
    rate = (df["aqi"] - prev_aqi) / elapsed_hours
    too_stale = elapsed_hours > MAX_GAP_HOURS_FOR_RATE
    rate = rate.mask(too_stale)
    df["aqi_change_rate"] = rate
    df["rate_gap_hours"] = elapsed_hours
 
    n_no_predecessor = prev_aqi.isna().sum()
    n_too_stale = too_stale.sum()
    logger.info(
        "Recomputed aqi_change_rate as AQI-points-per-hour (gap <= %.1fh). "
        "Null: %d row(s) with no predecessor, %d row(s) with gap > %.1fh",
        MAX_GAP_HOURS_FOR_RATE,
        n_no_predecessor,
        n_too_stale,
        MAX_GAP_HOURS_FOR_RATE,
    )
    return df



def clean(df, source_priority=None):
    df = drop_null_change_rate_rows(df)
    df = dedup_by_hour(df)
    df = recompute_change_rate(df)
    df = df.drop(columns=["dt", "hour_bucket", "offset_from_hour"])
    return df.sort_values("timestamp").reset_index(drop=True)

def fetch_raw_feature_data():
    # fs = get_feature_store()
    # df = get_all_rows(fs, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
    # df.to_json(TEMPORARY_JSON, orient="records", indent=4)
    df = pd.read_json(TEMPORARY_JSON, convert_dates=False)  # temp
    logger.info("Loaded %d raw rows from %s", len(df), TEMPORARY_JSON)  # temp
    return df


def get_feature_data():
    df = fetch_raw_feature_data()
    cleaned = clean(df)
    cleaned.to_json(OUTPUT_JSON, orient="records", indent=4)
    logger.info("Wrote %d cleaned rows to %s", len(cleaned), OUTPUT_JSON)
    return cleaned
