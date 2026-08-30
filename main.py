import logging
from fetch import fetch_all_uids
from validate import validate_aqi_data

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        handlers=[
            logging.FileHandler("FETCHER.log"),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()

    aqi_data = fetch_all_uids()
    #validate_aqi_data()

if __name__ == "__main__":
    main()