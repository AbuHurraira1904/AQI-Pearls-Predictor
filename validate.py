import json
import logging
import pandas as pd
from pathlib import Path

def get_aqi_dataFrame():
    with open("aqi_data.json", "r") as file:
        aqi_data = json.load(file)

    records = []

    for station_id, details in aqi_data.items():
        city = details.get("city", {})
        geo = city.get("geo", [None, None])
        iaqi = details.get("iaqi", {})
        
        record = {
            "station_id": station_id,
            "station_name": city.get("name"),
            "latitude": geo[0] if len(geo) > 0 else None,
            "longitude": geo[1] if len(geo) > 1 else None,
            "aqi": pd.to_numeric(details.get("aqi"), errors="coerce"),
            "dominant_pollutant": details.get("dominentpol"),
            "pm25": pd.to_numeric(iaqi.get("pm25", {}).get("v"), errors="coerce"),
            "pm10": pd.to_numeric(iaqi.get("pm10", {}).get("v"), errors="coerce"),
            "pm1": pd.to_numeric(iaqi.get("pm1", {}).get("v"), errors="coerce"),
            "temperature": pd.to_numeric(iaqi.get("t", {}).get("v"), errors="coerce"),
            "humidity": pd.to_numeric(iaqi.get("h", {}).get("v"), errors="coerce"),
            "timestamp": details.get("time", {}).get("s"),
            "attribution": details.get("attributions", [{}])[0].get("name") if details.get("attributions") else None
    }
        records.append(record)

    df= pd.DataFrame(records)

    return df

STALENESS_THRESHOLDS = {
    "The Urban Unit": pd.Timedelta(hours=1),
    "Pakistan Punjab Air Quality Network": pd.Timedelta(hours=3),
}
DEFAULT_STALENESS_THRESHOLD = pd.Timedelta(hours=3)

MIN_VALID_STATIONS = 5
OUTLIER_DEVIATION_AQI = 50
REQUIRED_FIELDS = ["aqi", "pm25"]

WEATHER_FIELDS = ["temperature", "humidity"]
WEATHER_SOURCE_NETWORK = "The Urban Unit"
LAST_KNOWN_WEATHER_PATH = Path("last_known_weather.json")

def flag_stale_data(df):
    df = df.copy()
    df["max_allowed_age"] = (   
        df["attribution"].map(STALENESS_THRESHOLDS).fillna(DEFAULT_STALENESS_THRESHOLD)
    )

    df["time_dt"] = pd.to_datetime(df["timestamp"])
    now = pd.Timestamp.now(tz=df["time_dt"].dt.tz)

    df["age"] = now - df["time_dt"]
    df["is_stale"] = df["age"] > df["max_allowed_age"]
    return df

def flag_invalid_data(df):
    df = df.copy()
    df["is_invalid_range"] = (df["aqi"] < 0) | (df["aqi"] > 500)
    return df

def flag_outliers(df):
    df = df.copy()
    citywide_median_aqi = df.loc[~df["is_stale"] & ~df["is_invalid_range"], "aqi"].median()
    df["is_outlier"] = (df["aqi"] - citywide_median_aqi).abs() > OUTLIER_DEVIATION_AQI
    return df

def report(df, valid_df, logger):
    logger.info("Logging validation report...")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("Missing values per column:\n%s", missing.to_string())
 
    duplicate_stations = df.duplicated(subset=["station_id"]).sum()
    if duplicate_stations:
        logger.warning("%d station(s) appeared more than once in this pull.", duplicate_stations)
 
    if len(valid_df) < MIN_VALID_STATIONS:
        logger.warning(
            "Only %d valid stations (need %d+). This hour's aggregate may not be reliable.",
            len(valid_df), MIN_VALID_STATIONS,
        )
    else:
        logger.info("Validation passed: %d/%d stations valid.", len(valid_df), len(df))
 
    stale_count = df["is_stale"].sum()
    range_count = df["is_invalid_range"].sum()
    outlier_count = df["is_outlier"].sum()
    logger.info(
        "Excluded breakdown -- stale: %d, out-of-range: %d, outliers: %d",
        stale_count, range_count, outlier_count,
    )
 
    summary_cols = [c for c in ["aqi", "pm25", "pm10", "temperature", "humidity"] if c in valid_df]
    logger.info(
        "Numerical summary (valid stations only):\n%s",
        valid_df[summary_cols].describe().to_string(),
    )

def fill_shared_weather_data(valid_df, logger):
    weather_source = valid_df[valid_df["attribution"] == WEATHER_SOURCE_NETWORK]
 
    if weather_source.empty:
        shared_weather = load_last_known_weather()
        if shared_weather is None:
            logger.warning(
                "No Urban Unit stations available this hour and no prior "
                "weather on record -- weather fields will be left blank."
            )
            return valid_df
        logger.warning(
            "No Urban Unit stations available this hour. Falling back to "
            "last known weather: %s", shared_weather,
        )
    else:
        shared_weather = weather_source[WEATHER_FIELDS].median().to_dict()
        save_last_known_weather(shared_weather)
 
    for field, value in shared_weather.items():
        valid_df[field] = valid_df[field].fillna(value)
 
    return valid_df
 
def load_last_known_weather():
    if not LAST_KNOWN_WEATHER_PATH.exists():
        return None
    return json.loads(LAST_KNOWN_WEATHER_PATH.read_text())
 
 
def save_last_known_weather(shared_weather: dict):
    LAST_KNOWN_WEATHER_PATH.write_text(json.dumps(shared_weather))

def validate_aqi_data():
    logger = logging.getLogger("AQI_Validator")

    df = get_aqi_dataFrame()

    df = flag_stale_data(df)
    df = flag_invalid_data(df)
    df = flag_outliers(df)

    valid_df = df[
        ~df["is_stale"]
        & ~df["is_invalid_range"]
        & ~df["is_outlier"]
        & df[REQUIRED_FIELDS].notna().all(axis=1)
    ].copy()

    valid_df = fill_shared_weather_data(valid_df, logger)

    report(df, valid_df, logger)

    json_string = valid_df.to_json(orient="records", date_format="iso")
    with open("clean_aqi_data.json", "w") as file:
            json.dump(json_string, file, indent=4)
    logger.info("AQI data cleaning completed and %d records saved to clean_aqi_data.json.", len(valid_df))
    
    return valid_df
