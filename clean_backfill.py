import pandas as pd
import logging

OUTPUT_JSON = "cleaned_backfill.json"

MAX_GAP_HOURS_FOR_RATE = 3.0

logger = logging.getLogger(__name__)

def normalize_timestamp_column(df):
    """
    Made with Claude to ensure df['timestamp'] is plain int64 epoch-milliseconds, regardless of
    how it arrived. This has now caused problems with the pipeline three separate times
    from three different causes (read_json auto-parsing dates, an explicit
    but wrong pd.to_datetime call, and Hopsworks' feature_group.read()
    returning a real datetime64 column since the feature store defines
    timestamp as a TIMESTAMP type). This is a fix that belongs here, once, at the
    start of clean(), rather than re-converted at each call site whenever a
    new data from a new source shows up with a different representation.
 
    Handles three shapes:
      - pure datetime64 (tz-aware or naive)
      - pure numeric (already epoch-ms)
      - mixed 'object' dtype — what you get when a datetime64 frame and an
        int64 frame are pd.concat'd BEFORE normalization: pandas doesn't
        keep it as datetime64, it degrades to a column of literal mixed
        Timestamp/int objects. pd.to_numeric can't parse a raw Timestamp,
        so this case needs element-wise conversion.
    """
    df = df.copy()
    col = df["timestamp"]
 
    if pd.api.types.is_datetime64_any_dtype(col):
        if col.dt.tz is None:
            col = col.dt.tz_localize("UTC")
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        df["timestamp"] = ((col - epoch) // pd.Timedelta(milliseconds=1)).astype("int64")
    elif col.dtype == object:
        def _to_ms(v):
            if isinstance(v, pd.Timestamp):
                if v.tzinfo is None:
                    v = v.tz_localize("UTC")
                return int(v.value // 1_000_000)
            return int(v)
        df["timestamp"] = col.map(_to_ms).astype("int64")
    else:
        df["timestamp"] = pd.to_numeric(col).astype("int64")
 
    return df



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
    df = normalize_timestamp_column(df)
    df = drop_null_change_rate_rows(df)
    df = dedup_by_hour(df, source_priority)
    df = recompute_change_rate(df)
    df = df.drop(columns=["dt", "hour_bucket", "offset_from_hour"])
    return df.sort_values("timestamp").reset_index(drop=True)


def get_cleaned_feature_data(df):

    cleaned = clean(df)
    cleaned.to_json(OUTPUT_JSON, orient="records", indent=4)
    logger.info("Wrote %d cleaned rows to %s", len(cleaned), OUTPUT_JSON)
    return cleaned
