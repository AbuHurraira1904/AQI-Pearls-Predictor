import logging
import hopsworks_client
import pandas as pd
from clean_backfill import get_cleaned_feature_data
#from clean_extra_backfill import get_extra_data
from data_preparation import prepare_training_data
from train import train_all_horizons
from store import get_all_rows
from promote import promote_all
from main import FEATURE_GROUP_NAME
from main import FEATURE_GROUP_VERSION

TEMPORARY_JSON = "temporary_fetch_group_storage.json"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        handlers=[
            logging.FileHandler("TRAINER.log"),
            logging.StreamHandler()
        ],
        force=True,
    )

def main():
    setup_logging()

    logger = logging.getLogger(__name__)

    '''This code pulls the data from hopsworks only and stores it on a temporary json'''
    fs = hopsworks_client.get_feature_store()
    df = get_all_rows(fs, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
    df.to_json(TEMPORARY_JSON, orient="records", indent=4)

    '''This code temporarily reads the raw_data from temporary json which saves us the api calls'''
    # df = pd.read_json(TEMPORARY_JSON, convert_dates=False)  # temp
    # logger.info("Loaded %d raw rows from %s", len(df), TEMPORARY_JSON)  # temp

    '''This code only cleans data from hopsworks only and returns the cleaned data'''
    cleaned_df = get_cleaned_feature_data(df)

    '''Temporary function usage to join data from hopsworks and openaq and clean that and return the cleaned data'''
    #cleaned_df = get_extra_data(df) #temp

    prepared_df = prepare_training_data(cleaned_df)
    training_results = train_all_horizons(prepared_df)


    mr = hopsworks_client.get_model_registry()

    outcomes = promote_all(mr, training_results)

if __name__ == "__main__":
    main()