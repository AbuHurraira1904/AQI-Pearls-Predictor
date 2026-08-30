import os
import json
import time
from dotenv import load_dotenv
import requests

load_dotenv()

AQICN_KEY = os.getenv("AQICN_API_KEY")
if not AQICN_KEY:
    raise ValueError("AQICN_API_KEY not found in environment variables. Please set it in your .env file.")

city = "Lahore"

with open("AQI_Stations.json", "r") as file:
    uids = [station["uid"] for station in json.load(file)]

all_aqi_data = {}


with requests.Session() as session:
    session.params = {"token": AQICN_KEY}
    
    for uid in uids:
        try:
            response = session.get(f"https://api.waqi.info/feed/@{uid}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    all_aqi_data[uid] = data.get("data", {})
                else:
                    print(f"API Error for UID {uid}: {data.get('data')}")
            else:
                print(f"HTTP Error {response.status_code} for UID {uid}")
        except requests.RequestException as e:
            print(f"Request failed for UID {uid}: {e}")
        
        time.sleep(0.1)

with open("aqi_data.json", "w") as file:
    json.dump(all_aqi_data, file, indent=4)