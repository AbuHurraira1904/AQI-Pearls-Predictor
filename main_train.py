import logging
import joblib
import hopsworks_client
from clean_backfill import get_cleaned_feature_data
from data_preparation import prepare_training_data
from train import train_all_horizons
from feast_store import get_all_rows
from promote import promote_all


REPO_PATH = "aqi_feature_repo/feature_repo"

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

    df = get_all_rows()

    cleaned_df = get_cleaned_feature_data(df)
    prepared_df = prepare_training_data(cleaned_df)
    training_results = train_all_horizons(prepared_df)

    joblib.dump(training_results, "training_results.pkl")
    logger.info("Saved training_results to training_results.pkl for downstream explainability (SHAP notebook)")

    mr = hopsworks_client.get_model_registry()
    outcomes = promote_all(mr, training_results)

if __name__ == "__main__":
    main()