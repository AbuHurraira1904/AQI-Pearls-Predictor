import hopsworks
import os
from dotenv import load_dotenv


def get_hopsworks_project():
    load_dotenv()
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY not found in environment variables. Please set it in your .env file.")

    project = hopsworks.login(
        project='Pearls_AQI_Predictor_1904',
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY)

    return project

def get_feature_store():
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    return fs

def get_model_registry():
    project = get_hopsworks_project()
    mr = project.get_model_registry()
    return mr