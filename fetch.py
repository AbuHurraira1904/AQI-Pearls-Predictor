import os
import json
import time
from dotenv import load_dotenv
import logging
import requests

load_dotenv()

AQICN_KEY = os.getenv("AQICN_API_KEY")
if not AQICN_KEY:
    raise ValueError("AQICN_API_KEY not found in environment variables. Please set it in your .env file.")

CITY = "Lahore"

def fetch_single_uid(session, uid, logger):
    try:
        response = session.get(f"https://api.waqi.info/feed/@{uid}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                return uid, data.get("data", {})
            else:
                logger.error(f"API Error for UID {uid}: {data.get('data')}")
        else:
            logger.error(f"HTTP Error {response.status_code} for UID {uid}")
    except requests.RequestException as e:
        logger.error(f"Request failed for UID {uid}: {e}")

    return uid, None

def fetch_all_uids():
    logger = logging.getLogger("AQI_Fetcher")

    try:
        with open("AQI_Stations.json", "r") as file:
            uids = [station["uid"] for station in json.load(file)]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading AQI_Stations.json: {e}")
        return {}

    logger.info(f"Fetching AQI data for {len(uids)} UIDs...")

    all_aqi_data = {}
    with requests.Session() as session:
        session.params = {"token": AQICN_KEY}

        for uid in uids:
            uid_res, data = fetch_single_uid(session, uid, logger)
            if data:
                all_aqi_data[uid_res] = data
            time.sleep(0.1)

    with open("aqi_data.json", "w") as file:
        json.dump(all_aqi_data, file, indent=4)

    logger.info("AQI data fetching completed and %d records saved to aqi_data.json.", len(all_aqi_data))

    return all_aqi_data