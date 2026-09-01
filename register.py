
import logging
import shutil
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR_ROOT = Path("registered_models")

def _model_name(horizon_hours):
    return f"aqi_predictor_{horizon_hours}h"


def _serialize_model(model, horizon_hours):
    model_dir = MODEL_DIR_ROOT / _model_name(horizon_hours)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    logger.info("Serialized model to %s", model_path)
    return model_dir


def register_model(mr, horizon_hours, model, test_metrics,
                    winner_name, n_train, n_val, n_test):
    model_dir = _serialize_model(model, horizon_hours)

    description = (
        f"{winner_name} model predicting AQI {horizon_hours}h ahead. "
        f"Trained on {n_train} rows, validated on {n_val}, tested on {n_test}. "
        f"Selected via lowest validation RMSE among candidate models."
    )

    hopsworks_model = mr.python.create_model(
        name=_model_name(horizon_hours),
        metrics=test_metrics,
        description=description,
    )

    hopsworks_model.save(str(model_dir))
    logger.info(
        "Registered '%s' v%s in Hopsworks Model Registry (test metrics: %s)",
        hopsworks_model.name, hopsworks_model.version, test_metrics,
    )
    return hopsworks_model


def register_all(mr, results):
    registered = {}
    for horizon_hours, r in results.items():
        registered[horizon_hours] = register_model(
            mr,
            horizon_hours=horizon_hours,
            model=r["winner_model"],
            test_metrics=r["test_metrics"],
            winner_name=r["winner_name"],
            n_train=r["n_train"],
            n_val=r["n_val"],
            n_test=r["n_test"],
        )
    return registered