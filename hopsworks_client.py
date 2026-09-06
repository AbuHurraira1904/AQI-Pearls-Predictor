import hopsworks
import os
from dotenv import load_dotenv

def _get_api_key():
    try:
        import streamlit as st
        if "HOPSWORKS_API_KEY" in st.secrets:
            return st.secrets["HOPSWORKS_API_KEY"]
    except Exception:
        pass
 
    load_dotenv()
    return os.getenv("HOPSWORKS_API_KEY")


def get_hopsworks_project():
    HOPSWORKS_API_KEY = _get_api_key()
    if not HOPSWORKS_API_KEY:
        raise ValueError(
            "HOPSWORKS_API_KEY not found. Set it in a local .env file, or "
            "in Streamlit Cloud's Settings -> Secrets when deployed."
        )
 
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