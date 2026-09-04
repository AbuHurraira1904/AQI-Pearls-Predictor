import logging
from fetch import fetch_all_uids
from feast_store import save_to_feast, get_latest_row
from validate import validate_aqi_data
from aggregate import assemble_aggregated_data

#FEATURE_GROUP_NAME = "hourly_city_aqi"
#FEATURE_GROUP_VERSION = 3

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        handlers=[
            logging.FileHandler("FETCHER.log"),
            logging.StreamHandler()
        ],
        force=True,
    )

def main():
    setup_logging()

    aqi_data = fetch_all_uids()
    valid_aqi_data = validate_aqi_data(aqi_data)

    #Deprecated code for hopsworks feature store, now using feast feature store
    #fs = hopsworks_client.get_feature_store()
    #last_feature_row = get_latest_row(fs, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)

    last_feature_row = get_latest_row()
    fg_row = assemble_aggregated_data(valid_aqi_data, last_feature_row)

    #Deprecated code for hopsworks feature store, now using feast feature store
    #save_to_feature_store(fs, fg_row, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)

    save_to_feast(fg_row)
if __name__ == "__main__":
    main()