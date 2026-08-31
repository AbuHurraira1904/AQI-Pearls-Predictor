import logging
import os
import hopsworks
from dotenv import load_dotenv
import pandas as pd
    
logger = logging.getLogger("FeatureStoreLogger")

def get_feature_store():
    load_dotenv()
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY not found in environment variables. Please set it in your .env file.")

    project = hopsworks.login(
        project='Pearls_AQI_Predictor_1904',
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    return fs

def save_to_feature_store(fs, fg_row, feature_group_name, version):

    logger = logging.getLogger("FeatureStoreLogger")

    try:
        feature_group = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=version,
            description="Hourly city-wide AQI data for Lahore",
            primary_key=["timestamp"],
            event_time="timestamp",
            time_travel_format="HUDI"
        )
    except Exception as e:
        logger.error(f"Error creating or getting feature group: {e}")
        raise

    df = pd.DataFrame([fg_row])
    df["aqi_change_rate"] = df["aqi_change_rate"].astype(float)

    try:
        feature_group.insert(df, write_options={"wait_for_job": False})
        logger.info(f"Data inserted into feature group '{feature_group_name}' version {version}.")
    except Exception as e:
        logger.error(f"Error inserting data into feature group: {e}")
        raise

    return feature_group

def get_all_rows(fs, feature_group_name, version):
    logger = logging.getLogger("FeatureStoreLogger")

    try:
        feature_group = fs.get_feature_group(name=feature_group_name, version=version)
        if feature_group is None:
            logger.warning(f"Feature group '{feature_group_name}' version {version} does not exist.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error retrieving feature group: {e}")
        raise

    try:
        df = feature_group.read()
        return df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error reading data from feature group: {e}")
        raise

def get_latest_row(fs, feature_group_name, version):
    logger = logging.getLogger("FeatureStoreLogger")

    try:
        feature_group = fs.get_feature_group(name=feature_group_name, version=version)
        if feature_group is None:
            logger.warning(f"Feature group '{feature_group_name}' version {version} does not exist.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving feature group: {e}")
        raise

    try:
        df = feature_group.read()
        if df.empty:
            logger.warning(f"No data found in feature group '{feature_group_name}' version {version}.")
            return None
        latest_row = df.sort_values(by="timestamp", ascending=False).iloc[0]
        return latest_row
    except Exception as e:
        logger.error(f"Error reading data from feature group: {e}")
        raise