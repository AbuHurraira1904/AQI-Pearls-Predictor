import math
import configparser
import pandas as pd

IDW_POWER = 2
POLLUTANT_FIELDS = ["aqi", "pm25", "pm10"]
CENTROID_CONFIG_PATH = "city_centroid.ini"
WEATHER_SOURCE_NETWORK = "The Urban Unit"

def load_centroid_config():
    config = configparser.ConfigParser()
    config.read(CENTROID_CONFIG_PATH)
    centroid_lat = float(config["CENTROID"]["latitude"])
    centroid_lon = float(config["CENTROID"]["longitude"])
    return (centroid_lat, centroid_lon)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def inverse_distance_weighting(valid_df):
    centroid_lat, centroid_lon = load_centroid_config()

    distances = valid_df.apply(lambda row: haversine_distance(centroid_lat, centroid_lon, row["latitude"], row["longitude"]), axis=1)

    at_centroid = distances < 0.01
    if at_centroid.any():
        exact_row = valid_df[at_centroid].iloc[0]
        return {field: exact_row[field] for field in POLLUTANT_FIELDS}

    weights = 1 / (distances ** IDW_POWER)

    result = {}
    for field in POLLUTANT_FIELDS:
        weighted_sum = (valid_df[field] * weights).sum()
        total_weight = weights.sum()
        result[field] = weighted_sum / total_weight if total_weight != 0 else None

    return result

def shared_weather(valid_df):
    urban_unit_rows = valid_df[valid_df["attribution"] == WEATHER_SOURCE_NETWORK]
    if urban_unit_rows.empty:
        return None, None
    return urban_unit_rows["temperature"].median(), urban_unit_rows["humidity"].median()


def aqi_change_rate(current_aqi, last_feature_row):
    if last_feature_row is None:
        return None
    previous_aqi = last_feature_row["aqi"]
    if hasattr(previous_aqi, "iloc"):
        previous_aqi = previous_aqi.iloc[0]
    return current_aqi - previous_aqi



def assemble_aggregated_data(valid_df, last_feature_group):
    Pollutant_Values = inverse_distance_weighting(valid_df)

    temperature, humidity = shared_weather(valid_df)

    now = pd.Timestamp.now()

    Hour = now.hour
    Day = now.day_name()
    Month = now.month

    AQI_Change_Rate = aqi_change_rate(Pollutant_Values["aqi"], last_feature_group)

    return {
        "timestamp": now,
        "hour": now.hour,
        "day_of_week": now.dayofweek,
        "month": now.month,
        "temperature": temperature,
        "humidity": humidity,
        "aqi_change_rate": AQI_Change_Rate,
        **Pollutant_Values,
    }
