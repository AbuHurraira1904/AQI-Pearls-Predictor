import logging

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Matches train.py's FEATURE_COLUMNS ordering — kept in sync manually since
# shap_explain only ever receives X_test frames produced by train.py.
FEATURE_COLUMNS = [
    "hour", "day_of_week", "month",
    "temperature", "humidity", "aqi_change_rate", "aqi_change_rate_missing",
    "aqi", "pm25", "pm10",
]


def build_explainer(rf_model):
    """
    TreeExplainer only. If a horizon's winner isn't a RandomForestRegressor,
    this will raise rather than silently doing something wrong — SHAP support
    for non-tree models (Ridge) is intentionally out of scope right now.
    """
    if type(rf_model).__name__ != "RandomForestRegressor":
        raise TypeError(
            f"shap_explain.build_explainer: expected RandomForestRegressor, "
            f"got {type(rf_model).__name__}. TreeExplainer is RF-only in this "
            f"module — this horizon's winner needs a different explainer or "
            f"should be skipped."
        )
    return shap.TreeExplainer(rf_model)


def compute_shap_values(explainer, X_test):
    """
    Returns the raw shap.Explanation object. Computed once per horizon and
    passed into the plotting functions below, so we never recompute SHAP
    values per-plot.
    """
    logger.info("Computing SHAP values for %d test rows", len(X_test))
    return explainer(X_test)


def plot_summary(shap_values, figsize=(9, 6)):
    """Global beeswarm plot: feature importance + direction of effect."""
    fig = plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_dependence(shap_values, X_test, feature, interaction_feature="auto", figsize=(8, 5)):
    """
    Dependence plot for a single feature — how SHAP value (impact on
    prediction) varies with that feature's actual value, colored by the
    feature SHAP picks as the strongest interaction (or pass one explicitly).
    """
    fig, ax = plt.subplots(figsize=figsize)
    shap.dependence_plot(
        feature, shap_values.values, X_test,
        interaction_index=interaction_feature, ax=ax, show=False,
    )
    fig.tight_layout()
    return fig


def _best_worst_indices(y_test, y_pred):
    errors = (y_test.values - y_pred)
    best_idx = int(np.argmin(np.abs(errors)))
    worst_idx = int(np.argmax(np.abs(errors)))
    return best_idx, worst_idx


def plot_local_waterfall(shap_values, index, title_suffix="", figsize=(9, 6)):
    """Single-prediction waterfall: which features pushed this one prediction up/down."""
    fig = plt.figure(figsize=figsize)
    shap.plots.waterfall(shap_values[index], show=False)
    fig = plt.gcf()
    fig.suptitle(f"Local explanation{title_suffix}")
    fig.tight_layout()
    return fig


def plot_best_worst_waterfalls(shap_values, X_test, y_test, model, figsize=(9, 6)):
    """
    Convenience wrapper: finds the best- and worst-predicted rows in X_test
    (by absolute error) and returns their waterfall figures as a dict, e.g.
    {"best": fig, "worst": fig}. Uses the same model to generate predictions
    so the error is computed consistently with test_metrics.
    """
    y_pred = model.predict(X_test)
    best_idx, worst_idx = _best_worst_indices(y_test, y_pred)

    logger.info(
        "Best prediction at row %d (|error|=%.3f), worst at row %d (|error|=%.3f)",
        best_idx, abs(y_test.values[best_idx] - y_pred[best_idx]),
        worst_idx, abs(y_test.values[worst_idx] - y_pred[worst_idx]),
    )

    return {
        "best": plot_local_waterfall(shap_values, best_idx, title_suffix=" (best prediction)", figsize=figsize),
        "worst": plot_local_waterfall(shap_values, worst_idx, title_suffix=" (worst prediction)", figsize=figsize),
    }


def explain_horizon(training_result, figsize_summary=(9, 6)):
    """
    Top-level entry point for a notebook cell: given one horizon's entry from
    train_all_horizons()'s results dict, returns a dict of figures. Skips
    (returns None) if the winner for this horizon isn't Random Forest.
    """
    if training_result is None:
        return None

    if training_result["winner_name"] != "random_forest":
        logger.warning(
            "Horizon %dh: winner is '%s', not random_forest — SHAP skipped "
            "for this horizon.",
            training_result["horizon_hours"], training_result["winner_name"],
        )
        return None

    model = training_result["winner_model"]
    X_test = training_result["X_test"]
    y_test = training_result["y_test"]

    explainer = build_explainer(model)
    shap_values = compute_shap_values(explainer, X_test)

    figs = {
        "summary": plot_summary(shap_values, figsize=figsize_summary),
    }
    figs.update(plot_best_worst_waterfalls(shap_values, X_test, y_test, model))

    return figs