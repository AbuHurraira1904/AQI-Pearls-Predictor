import logging

import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DEFAULT_AGGREGATE_SAMPLE_SIZE = 20


def build_explainer(X_train):
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        mode="regression",
        discretize_continuous=True,
    )


def explain_instance(explainer, model, x_row, num_features=None):
    if num_features is None:
        num_features = len(x_row)
    return explainer.explain_instance(
        x_row.values, model.predict, num_features=num_features,
    )


def plot_local_explanation(explanation, title_suffix="", figsize=(9, 6)):
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(*figsize)
    fig.suptitle(f"LIME local explanation{title_suffix}")
    fig.tight_layout()
    return fig


def _best_worst_indices(y_test, y_pred):
    errors = y_test.values - y_pred
    best_idx = int(np.argmin(np.abs(errors)))
    worst_idx = int(np.argmax(np.abs(errors)))
    return best_idx, worst_idx


def plot_best_worst(explainer, model, X_test, y_test, figsize=(9, 6)):
    y_pred = model.predict(X_test)
    best_idx, worst_idx = _best_worst_indices(y_test, y_pred)

    logger.info(
        "LIME best prediction at row %d (|error|=%.3f), worst at row %d (|error|=%.3f)",
        best_idx, abs(y_test.values[best_idx] - y_pred[best_idx]),
        worst_idx, abs(y_test.values[worst_idx] - y_pred[worst_idx]),
    )

    best_exp = explain_instance(explainer, model, X_test.iloc[best_idx])
    worst_exp = explain_instance(explainer, model, X_test.iloc[worst_idx])

    return {
        "best": plot_local_explanation(best_exp, title_suffix=" (best prediction)", figsize=figsize),
        "worst": plot_local_explanation(worst_exp, title_suffix=" (worst prediction)", figsize=figsize),
    }


def compute_aggregate_importance(explainer, model, X_test, n_samples=DEFAULT_AGGREGATE_SAMPLE_SIZE):
    n_samples = min(n_samples, len(X_test))
    sampled = X_test.sample(n_samples, random_state=42)

    weight_totals = np.zeros(X_test.shape[1])
    for _, row in sampled.iterrows():
        exp = explain_instance(explainer, model, row)
        for feature_idx, weight in exp.as_map()[1]:
            weight_totals[feature_idx] += abs(weight)

    avg_importance = weight_totals / n_samples
    return dict(zip(X_test.columns, avg_importance))


def plot_aggregate_importance(avg_importance, n_samples=DEFAULT_AGGREGATE_SAMPLE_SIZE, figsize=(9, 6)):
    sorted_items = sorted(avg_importance.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, values, color="#4c72b0")
    ax.set_xlabel("Mean |LIME weight| across sampled test rows")
    ax.set_title(f"LIME Aggregate Feature Importance (n={n_samples} sampled rows)")
    fig.tight_layout()
    return fig


def explain_horizon(training_result, aggregate_sample_size=DEFAULT_AGGREGATE_SAMPLE_SIZE):
    if training_result is None:
        return None

    if "X_train" not in training_result:
        logger.error(
            "explain_horizon: training_result is missing 'X_train'. "
            "train.py's train_horizon() needs to return X_train in its "
            "result dict -- see the diff applied alongside this module."
        )
        return None

    model = training_result["winner_model"]
    X_train = training_result["X_train"]
    X_test = training_result["X_test"]
    y_test = training_result["y_test"]

    explainer = build_explainer(X_train)

    avg_importance = compute_aggregate_importance(explainer, model, X_test, n_samples=aggregate_sample_size)
    n_used = min(aggregate_sample_size, len(X_test))

    figs = {
        "summary": plot_aggregate_importance(avg_importance, n_samples=n_used),
    }
    figs.update(plot_best_worst(explainer, model, X_test, y_test))

    return figs