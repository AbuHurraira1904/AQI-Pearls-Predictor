import logging
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard_data import (
    classify_hazard,
    fetch_champion_models,
    fetch_historical_features,
    fetch_latest_features,
    load_training_results,
    predict_all_horizons,
    HAZARD_BANDS,
)
import eda_plots
from shap_explain import explain_horizon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Lahore AQI Predictor", page_icon="\U0001F32B", layout="wide")

REFRESH_INTERVAL_MINUTES = 12
TREND_LOOKBACK_HOURS = 7 * 24  # last week

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# Maps a stable display name -> (asset filename, function that builds the fig).
# The function is only called when the user hits "Regenerate" -- otherwise
# we just show whatever PNG is already sitting in assets/.
EDA_FIGURES = {
    "AQI Over Time": ("eda_timeseries.png", lambda df: eda_plots.plot_aqi_timeseries(df)),
    "Hourly Pattern": ("eda_hourly.png", lambda df: eda_plots.plot_hourly_pattern(df)),
    "Weekly Pattern": ("eda_weekly.png", lambda df: eda_plots.plot_weekly_pattern(df)),
    "AQI Distribution": ("eda_distribution.png", lambda df: eda_plots.plot_aqi_distribution(df)),
    "Pollutant Distributions": ("eda_pollutants.png", lambda df: eda_plots.plot_pollutant_distributions(df)),
    "Correlation Heatmap": ("eda_correlation.png", lambda df: eda_plots.plot_correlation_heatmap(df)),
    "AQI vs PM": ("eda_aqi_vs_pm.png", lambda df: eda_plots.plot_aqi_vs_pm(df)),
    "Change Rate Distribution": ("eda_change_rate.png", lambda df: eda_plots.plot_change_rate_distribution(df)),
    "Missingness Over Time": ("eda_missingness.png", lambda df: eda_plots.plot_missingness_over_time(df)),
}

# Reruns the whole script automatically every REFRESH_INTERVAL_MINUTES,
# without any user interaction -- plain Streamlit only reruns on user
# interaction, so this is what actually makes the "auto" in auto-refresh work.
st_autorefresh(interval=REFRESH_INTERVAL_MINUTES * 60 * 1000, key="data_refresh")


@st.cache_data(ttl=REFRESH_INTERVAL_MINUTES * 60)
def load_latest():
    return fetch_latest_features()


@st.cache_data(ttl=REFRESH_INTERVAL_MINUTES * 60)
def load_history():
    return fetch_historical_features(lookback_hours=TREND_LOOKBACK_HOURS)


@st.cache_resource(ttl=REFRESH_INTERVAL_MINUTES * 60)
def load_models():
    return fetch_champion_models()


def render_hazard_banner(aqi_value, context_label):
    label, color = classify_hazard(aqi_value)
    st.markdown(
        f"""
        <div style="background-color:{color}22; border-left: 6px solid {color};
                    padding: 12px 16px; border-radius: 6px; margin-bottom: 8px;">
            <strong>{context_label}: {aqi_value:.0f} AQI &mdash; {label}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trend_chart(history_df):
    fig = go.Figure()

    dt = pd.to_datetime(history_df["timestamp"], unit="ms", utc=True)
    fig.add_trace(go.Scatter(
        x=dt, y=history_df["aqi"], mode="lines", name="AQI",
        line=dict(color="black", width=1.5),
    ))

    for low, high, label, color in HAZARD_BANDS:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.08, line_width=0)

    fig.update_layout(
        title=f"Lahore AQI — Last {TREND_LOOKBACK_HOURS // 24} Days",
        xaxis_title="Time",
        yaxis_title="AQI",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def render_predictions_tab():
    latest_row = load_latest()
    if latest_row is None:
        st.error("No feature data available yet from the feature store. Check back after the next pipeline run.")
        return

    current_aqi = latest_row["aqi"]
    render_hazard_banner(current_aqi, "Current AQI")

    with st.spinner("Loading champion models and generating predictions..."):
        models = load_models()
        predictions = predict_all_horizons(latest_row, models)

    st.subheader("Forecasts")
    cols = st.columns(len(predictions))
    for col, (horizon, pred) in zip(cols, sorted(predictions.items())):
        with col:
            if pred is None:
                st.warning(f"No model registered yet for {horizon}h horizon.")
                continue
            label, color = classify_hazard(pred)
            st.metric(label=f"{horizon}h ahead", value=f"{pred:.0f} AQI", delta=f"{pred - current_aqi:+.0f} vs now")
            st.markdown(
                f'<span style="color:{color}; font-weight:600;">{label}</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader("Historical Trend")
    history_df = load_history()
    if history_df.empty:
        st.info("Not enough historical data yet to show a trend.")
    else:
        st.plotly_chart(render_trend_chart(history_df), use_container_width=True)

    with st.expander("Latest raw feature row"):
        st.dataframe(latest_row.to_frame().T, use_container_width=True)


def render_eda_tab():
    st.caption(
        "Static plots by default. Hit Regenerate to rebuild them from the "
        "full current feature history -- this re-fetches and re-joins all "
        "historical data from Feast, so it's slower than a normal page load."
    )

    if st.button("\U0001F504 Regenerate EDA plots", key="regen_eda"):
        with st.spinner("Fetching full historical data and regenerating EDA plots..."):
            full_history = fetch_historical_features(lookback_hours=None)
            if full_history.empty:
                st.error("No historical data available to plot.")
            else:
                for display_name, (filename, plot_fn) in EDA_FIGURES.items():
                    fig = plot_fn(full_history)
                    fig.savefig(os.path.join(ASSETS_DIR, filename), bbox_inches="tight", dpi=150)
                st.success("EDA plots regenerated.")

    for display_name, (filename, _) in EDA_FIGURES.items():
        path = os.path.join(ASSETS_DIR, filename)
        st.subheader(display_name)
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("Not generated yet -- hit Regenerate above.")


def render_shap_tab():
    st.caption(
        "Static plots by default, generated from the last main_train.py run "
        "(training_results.pkl). Hit Regenerate to re-run SHAP against "
        "that same file -- this does NOT retrain models, it only re-explains "
        "whichever models are currently saved in training_results.pkl."
    )

    if st.button("\U0001F504 Regenerate SHAP plots", key="regen_shap"):
        training_results = load_training_results()
        if training_results is None:
            st.error("training_results.pkl not found -- run main_train.py first.")
        else:
            with st.spinner("Computing SHAP values and regenerating plots..."):
                any_generated = False
                for h, result in training_results.items():
                    explanation = explain_horizon(result)
                    if explanation is None:
                        continue
                    any_generated = True
                    for name, fig in explanation.items():
                        fig.savefig(
                            os.path.join(ASSETS_DIR, f"shap_{h}h_{name}.png"),
                            bbox_inches="tight", dpi=150,
                        )
                if any_generated:
                    st.success("SHAP plots regenerated.")
                else:
                    st.warning("No horizon currently has a Random Forest winner to explain.")

    for h in (24, 48, 72):
        st.subheader(f"{h}h Horizon")
        found_any = False
        for name in ("summary", "best", "worst"):
            path = os.path.join(ASSETS_DIR, f"shap_{h}h_{name}.png")
            if os.path.exists(path):
                found_any = True
                st.image(path, caption=name, use_container_width=True)
        if not found_any:
            st.info("Not generated yet -- hit Regenerate above.")


def main():
    st.title("Lahore AQI Predictor")
    st.caption(
        f"Predictions auto-refresh every {REFRESH_INTERVAL_MINUTES} minutes from the current "
        f"champion models in the Hopsworks model registry."
    )

    tab_predictions, tab_eda, tab_shap = st.tabs(["Live Predictions", "EDA", "Explainability (SHAP)"])

    with tab_predictions:
        render_predictions_tab()

    with tab_eda:
        render_eda_tab()

    with tab_shap:
        render_shap_tab()


if __name__ == "__main__":
    main()