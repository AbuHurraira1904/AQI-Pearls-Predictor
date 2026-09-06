import pandas as pd

AQI_BANDS = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy (Sensitive)", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

FEATURE_COLUMNS = [
    "hour", "day_of_week", "month",
    "temperature", "humidity", "aqi_change_rate", "aqi_change_rate_missing",
    "aqi", "pm25", "pm10",
]

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def classify_hazard(aqi_value):
    """Returns (label, color) for a given AQI value, per EPA bands above."""
    if aqi_value is None or pd.isna(aqi_value):
        return "Unknown", "#808080"
    for low, high, label, color in AQI_BANDS:
        if low <= aqi_value <= high:
            return label, color
    # Above 500 still treated as Hazardous.
    if aqi_value > 500:
        return "Hazardous", "#7e0023"
    return "Unknown", "#808080"