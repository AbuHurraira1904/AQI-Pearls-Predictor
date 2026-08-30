import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    project='Pearls_AQI_Predictor_1904',  # Replace with your project name
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key_value=os.getenv("HOPSWORKS_API_KEY") # Get from Hopsworks UI > Account Settings > API Keys
)