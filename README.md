# AQI-Pearls-Predictor

A 100% serverless machine learning pipeline that predicts Lahore's Air Quality Index (AQI) 24, 48, and 72 hours ahead — from hourly data collection to a live, explainable dashboard.

Built as part of the Pearls AQI Predictor internship project.

---

## What it does

- **Fetches** live AQI, pollutant, and weather readings every hour from 22 hand-curated stations across Lahore
- **Aggregates** them into a single city-wide feature row using inverse-distance weighting
- **Stores** the feature history in a Feast offline feature store
- **Trains** Ridge Regression and Random Forest models daily for each forecast horizon (24h / 48h / 72h), automatically promoting the better challenger model over the current champion
- **Registers** the winning models in the Hopsworks Model Registry
- **Explains** predictions using SHAP (tree models) and LIME (model-agnostic)
- **Serves** live predictions, historical trends, EDA, and explainability plots on a public Streamlit dashboard

---

## Architecture

```
┌─────────────────────┐      ┌──────────────────┐      ┌───────────────────────┐      ┌────────────────────┐
│  Hourly Feature      │      │  Feast Offline    │      │  Daily Training        │      │  Streamlit           │
│  Pipeline            │ ───► │  Feature Store     │ ───► │  Pipeline               │ ───► │  Dashboard            │
│  (GitHub Actions)    │      │  (Parquet in repo)  │      │  (GitHub Actions)      │      │  (Streamlit Cloud)   │
└─────────────────────┘      └──────────────────┘      └───────────────────────┘      └────────────────────┘
```

1. **Feature pipeline** — runs hourly (`main.py`): fetch → validate → aggregate → write to Feast
2. **Feature store** — Feast (file-based, offline), Parquet committed back to the repo as durable storage since GitHub Actions runners are ephemeral
3. **Training pipeline** — runs daily (`main_train.py`): load from Feast → clean/backfill → prepare targets → train candidates per horizon → champion/challenger promotion → register in Hopsworks
4. **Dashboard** — `app.py`, three tabs: Live Predictions, EDA, and Explainability (SHAP/LIME)

Scheduling is handled via GitHub Actions `workflow_dispatch`, triggered externally by [cron-job.org](https://cron-job.org).

---

## Tech stack

| Purpose | Tool |
|---|---|
| Data source | AQICN API (22 curated Lahore stations), OpenAQ (one-time historical backfill) |
| Feature store | [Feast](https://feast.dev) (offline, file-based) |
| Model registry | [Hopsworks](https://hopsworks.ai) |
| ML models | scikit-learn — Ridge Regression, Random Forest |
| Explainability | SHAP (`TreeExplainer`, Random Forest only), LIME (model-agnostic, all models) |
| Dashboard | Streamlit + Plotly + `streamlit-autorefresh` |
| Scheduling | GitHub Actions + cron-job.org |
| Deployment | Streamlit Community Cloud |

**Note on the feature store:** Hopsworks' free tier had reliability issues on the offline materialization side (Spark job failures affecting all feature groups project-wide). Feast was added as the feature store instead; Hopsworks is retained solely for the model registry.

---

## Repository structure

```
├── fetch.py              # Pulls raw AQI/pollutant/weather data from AQICN
├── validate.py            # Flags stale, out-of-range, and outlier station readings
├── aggregate.py            # IDW-weighted city-wide aggregation + change-rate calc
├── main.py                 # Hourly feature pipeline entry point
│
├── feast_store.py           # Feast read/write layer (aqi_feature_repo/)
├── aqi_feature_repo/          # Feast feature repo (feature_store.yaml, features.py, Parquet data)
├── store.py                  # (Deprecated) Hopsworks feature-store layer
│
├── clean_backfill.py          # Timestamp normalization, dedup, change-rate recompute
├── data_preparation.py         # Builds horizon targets (24h/48h/72h) via time-based merge
├── train.py                    # Candidate training, chronological split, evaluation
├── promote.py                   # Champion/challenger promotion logic
├── register.py                   # Hopsworks Model Registry read/write
├── hopsworks_client.py             # Hopsworks login/session helper
├── main_train.py                    # Daily training pipeline entry point
│
├── eda_plots.py                      # 10 reusable EDA plot functions
├── shap_explain.py                    # SHAP explainability (Random Forest only)
├── lime_explain.py                     # LIME explainability (model-agnostic)
│
├── constants.py                         # AQI hazard bands, day names, hazard classifier
├── dashboard_data.py                     # Data/Feast/Hopsworks logic for the dashboard (no Streamlit imports)
├── app.py                                 # Streamlit dashboard entry point
│
├── AQI_Stations.json                       # Station metadata (22 Lahore stations)
├── city_centroid.ini                        # Lahore centroid coordinates for IDW
│
└── notebooks/
    ├── AQI_EDA.ipynb                          # Exploratory data analysis
    ├── shap_explainability.ipynb               # SHAP walkthrough
    └── lime_explainability.ipynb                # LIME walkthrough
```

---

## Setup

### Prerequisites
- Python 3.11
- An [AQICN API token](https://aqicn.org/data-platform/token/)
- A [Hopsworks](https://hopsworks.ai) account and API key (free tier)

### Installation

```bash
git clone <repo-url>
cd AQI-Pearls-Predictor
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
AQICN_API_KEY=your_aqicn_token
HOPSWORKS_API_KEY=your_hopsworks_key
```

When deployed on Streamlit Community Cloud, set the same values under **Settings → Secrets** instead — `hopsworks_client.py` checks `st.secrets` first and falls back to `.env` locally.

### GitHub Actions setup

The Feast Parquet commit-back step requires write access:
**Settings → Actions → General → Workflow permissions → Read and write permissions**

---

## Running the pipelines

**Feature pipeline** (normally hourly, via GitHub Actions):
```bash
python main.py
```

**Training pipeline** (normally daily, via GitHub Actions):
```bash
python main_train.py
```

**Dashboard** (locally):
```bash
streamlit run app.py
```

---

## Modeling approach

- **Candidates:** Ridge Regression and Random Forest (`max_depth=15, min_samples_leaf=5` — constrained to keep serialized model size reasonable for deployment), trained separately per horizon
- **Split:** 70/15/15 **chronological** train/val/test split (not random) to respect the time-series nature of the data
- **Selection metric:** RMSE — chosen over R² as the headline metric, since the practical goal is threshold-based hazard alerting rather than variance explanation. Low R² on a single-season data window is expected and does not indicate model failure (see the EDA notebook's summary section for a full discussion)
- **Promotion:** A challenger only replaces the current champion if it beats it by more than 1% relative RMSE improvement, re-scored on the challenger's own fresh test split for a fair comparison

---

## Explainability

- **SHAP** (`shap_explain.py`) — `TreeExplainer`, applies only to horizons where Random Forest won (Ridge horizons are skipped with a logged warning, since TreeExplainer is tree-model-specific)
- **LIME** (`lime_explain.py`) — model-agnostic, works for any horizon regardless of winner; its "summary" view is an approximation built by averaging local feature weights across a sample of test rows, since LIME has no native global explanation

Both read directly from `training_results.pkl` (saved at the end of `main_train.py`) without retraining, so explanations always reflect the exact models evaluated in the most recent training run.

---

## Dashboard

Three tabs, refreshing automatically every 90 minutes:

- **Live Predictions** — current AQI, hazard-banded forecasts for all three horizons, historical trend chart
- **EDA** — static pre-generated plots (temporal patterns, distributions, correlations, data quality) with a manual "Regenerate" option
- **Explainability (SHAP / LIME)** — static pre-generated explanation plots per horizon, also with "Regenerate"

Plots are served as static PNGs by default rather than rendered live, for dashboard performance; regeneration is opt-in via button.

---

## Known limitations

- Single-season data window — monthly seasonality patterns and long-range R² are not yet meaningful
- Hopsworks free-tier feature store materialization is unreliable; Feast is used instead for all offline feature storage
- SHAP explainability is unavailable for horizons where Ridge (not Random Forest) wins

---

## Roadmap

- Split into `feature_pipeline/`, `training_pipeline/`, `dashboard/`, `common/`, `config/` subdirectories (deferred post-deadline to avoid import/path/CI risk)
- Extend the training window as more historical data accumulates

---

## Further reading

See the accompanying project report for design-decision rationale, debugging history, and detailed EDA findings.
