# Personalized Anomaly Detection for Athlete Load and Recovery Management

Master's thesis project. A baseline modeling/ machine learning pipeline that detects anomalies in athlete training load and flags injury risk using GPS tracking data and HR measurements.

## Overview

The pipeline takes per-player GPS season reports and subjective load data, builds personalized baselines, and trains predictive models to flag sessions where an athlete is at elevated injury risk.
An open source dataset was used for this analysis in this thesis and can be found on: https://zenodo.org/records/10033832. The dataset was stored in Google Cloud Storage when processing the data for the convenience of the project owners.

**Pipeline steps:**

1. **Data loading** — GPS session reports, injury history, and RPE data
2. **Supervised PCA** — Reduces the metric space to the dimensions most correlated with injury (three model-specific variants)
3. **Baseline modeling** — Three approaches per player:
   - *EWMA* (Exponentially Weighted Moving Average) 
   - *Bayesian* — probabilistic baseline using PyMC
   - *Population-based Moving Average* — population-level reference
4. **Internal vs External load analysis** — ACWR (Acute:Chronic Workload Ratio) to assess cumulative fatigue
5. **Deviation modeling** — Z-score flagging at low / moderate / high thresholds
6. **Feature engineering** — Per-player EWMA, ACWR, and MSWR workload features
7. **Predictive modeling** — Logistic Regression, Decision Tree, and Random Forest trained on deviation and workload features
8. **Evaluation** — Regression fit of baselines + classification metrics for injury prediction


## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.11+ (see [.python-version](.python-version)).

## Dependencies

- `numpy`, `pandas`, `scipy` — data processing
- `scikit-learn` — PCA, classification models, evaluation
- `pymc`, `arviz` — Bayesian baseline modeling
- `google-cloud-storage`, `pyarrow` — cloud data access

## Google Cloud

The `gcloud/` directory contains scripts for uploading and reading data from Google Cloud Storage, used for remote data management independently of the main pipeline. Note that these scripts are read-only and cannot be executed without authorized access to the project's Google Cloud Storage bucket. Access is restricted to the project owners to prevent unintended usage and associated costs.

## Data

The following data files are expected to be in this location:

| Path | Description |
|---|---|
| `season_reports/` | Per-player GPS session CSVs |
| `data/subjective/injury/injury.csv` | Injury history |
| `data/subjective/training-load/session.json` | RPE session data |

## Key Hyperparameters

These parameters can be configured at the top of [main.py](main.py):

| Parameter | Default | Description |
|---|---|---|
| `SPANS` | `[3..27]` | EWMA span candidates (sessions) |
| `CHRONIC` / `ACUTE` | `7` / `3` | Workload ratio windows |
| `ACWR_CHRONIC` / `ACWR_ACUTE` | `28` / `7` | ACWR windows for int/ext analysis |
| `LOW/MODERATE/HIGH_Z_SCORE_THRESHOLD` | `1` / `2` / `3` | Z-score flagging thresholds |

## Usage

```bash
python main.py
```
The script runs the full pipeline and prints evaluation results in the terminal.
The script takes roughly 3 minutes to run so for added convenience the most recent results have been saved as results.txt in the project's root.

## Cold start experiment
In order to perform the cold start experiment the path of the seasons reports in line 32 in main.py needs to changed from:
```bash
reports_folder = Path("season_reports")
```
to for example:
```bash
reports_folder = Path("season_reports_coldstart/season_reports_coldstart_10")
```
if wanting to test the experiment with only 10 sessions per player.


