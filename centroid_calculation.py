import configparser
import json
from fetch import fetch_all_uids
from validate import get_aqi_dataFrame

with open("aqi_data_snapshot.json", "r") as file:
        aqi_data = json.load(file)

df = get_aqi_dataFrame(aqi_data)

valid_geo = df.dropna(subset=["latitude", "longitude"])

centroid_lat = valid_geo["latitude"].mean()
centroid_lon = valid_geo["longitude"].mean()

config = configparser.ConfigParser()
config["CENTROID"] = {
    "latitude": f"{centroid_lat:.6f}",
    "longitude": f"{centroid_lon:.6f}",
    "station_count": str(len(valid_geo))
}

config_filename = "city_centroid.ini"
with open(config_filename, "w") as configfile:
    config.write(configfile)