import logging
import pandas as pd
from feast import FeatureStore

logger = logging.getLogger(__name__)

FEAST_REPO_PATH = "aqi_feature_repo/feature_repo"
PARQUET_PATH = "aqi_feature_repo/feature_repo/data/hourly_city_aqi.parquet"

FEATURE_REFS = [
    "hourly_city_aqi:aqi",
    "hourly_city_aqi:pm25",
    "hourly_city_aqi:pm10",
    "hourly_city_aqi:temperature",
    "hourly_city_aqi:humidity",
    "hourly_city_aqi:aqi_change_rate",
    "hourly_city_aqi:hour",
    "hourly_city_aqi:day_of_week",
    "hourly_city_aqi:month",
]

CITY_ID = "lahore"


def _normalize_timestamp(ts):
    """Feast's FileSource requires a real datetime64 column, not epoch-ms int."""
    if pd.api.types.is_integer_dtype(type(ts)) or isinstance(ts, (int,)):
        return pd.to_datetime(ts, unit="ms", utc=True)
    return pd.to_datetime(ts, utc=True)


def save_to_feast(fg_row):
    try:
        existing = pd.read_parquet(PARQUET_PATH)
    except FileNotFoundError:
        logger.warning("Feast parquet file not found at %s; starting fresh.", PARQUET_PATH)
        existing = pd.DataFrame()

    new_row = pd.DataFrame([fg_row])
    new_row["timestamp"] = new_row["timestamp"].apply(_normalize_timestamp)
    new_row["city_id"] = CITY_ID

    combined = pd.concat([existing, new_row], ignore_index=True)

    try:
        combined.to_parquet(PARQUET_PATH, index=False)
        logger.info("Row written to Feast parquet store at %s (%d total rows).", PARQUET_PATH, len(combined))
    except Exception as e:
        logger.error("Error writing to Feast parquet store: %s", e)
        raise


def get_all_rows():
    """
    Returns the full historical feature dataframe from Feast, with columns
    matching store.py's Hopsworks shape: 'timestamp' as epoch-ms int,
    'city_id' dropped, sorted by timestamp descending (same as
    store.get_all_rows).
    """
    try:
        existing = pd.read_parquet(PARQUET_PATH)
    except FileNotFoundError:
        logger.warning("Feast parquet file not found at %s.", PARQUET_PATH)
        return pd.DataFrame()

    if existing.empty:
        logger.warning("No data found in Feast parquet store at %s.", PARQUET_PATH)
        return pd.DataFrame()

    store = FeatureStore(repo_path=FEAST_REPO_PATH)

    entity_df = pd.DataFrame({
        "city_id": existing["city_id"],
        "event_timestamp": existing["timestamp"],
    })

    try:
        df = store.get_historical_features(
            entity_df=entity_df,
            features=FEATURE_REFS,
        ).to_df()
    except Exception as e:
        logger.error("Error retrieving historical features from Feast: %s", e)
        raise

    df = df.rename(columns={"event_timestamp": "timestamp"})
    df["timestamp"] = (df["timestamp"].astype("int64") // 1_000_000)
    df = df.drop(columns=["city_id"], errors="ignore")

    return df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)


def get_latest_row():
    df = get_all_rows()
    if df.empty:
        logger.warning("No data found in Feast parquet store; cannot get latest row.")
        return None
    latest_row = df.sort_values(by="timestamp", ascending=False).iloc[0]
    return latest_row