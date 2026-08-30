import configparser
from validate import get_aqi_dataFrame

df = get_aqi_dataFrame()

valid_geo = df.dropna(subset=["latitude", "longitude"])

centroid_lat = valid_geo["latitude"].mean()
centroid_lon = valid_geo["longitude"].mean()

config = configparser.ConfigParser()
config["CENTROID"] = {
    "latitude": f"{centroid_lat:.6f}",
    "longitude": f"{centroid_lon:.6f}",
    "station_count": str(len(valid_geo))
}

config_filename = "centroid_config.ini"
with open(config_filename, "w") as configfile:
    config.write(configfile)