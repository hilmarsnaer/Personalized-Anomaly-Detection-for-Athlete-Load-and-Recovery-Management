# main.py
# Personalized Anomaly Detection for Athlete Load and Recovery Management.
# This script orchestrates the entire analysis pipeline, from data loading and preprocessing to baseline modeling, deviation analysis, predictive modeling, and evaluation.

# Imports:
import pandas as pd
from pathlib import Path
import numpy as np
from pymc import model

# Data loading and preprocessing
from data_loading.data_load import load_data

# Baseline modeling
from supervised_PCA import supervised_pca
from baselines import run_ewma_baseline_evaluation, bayesian_baselines, population_based_moving_average_baseline
from int_vs_ext import analyze_internal_vs_external
from deviation_modeling import compute_z_scores, print_deviation_summary

# Predictive modeling
from feature_engineering import construct_features
from model_feature_sets import model_feature_set, train_test_split_data
from prediction_algorithms.logistic_regression import logistic_regression_model, cross_validate_logistic_regression
from prediction_algorithms.decision_tree import decision_tree_model, cross_validate_decision_tree
from prediction_algorithms.random_forest import random_forest_model, cross_validate_random_forest

# Evaluation
from evaluation import evaluate_baselines, evaluate_flags, print_evaluation
from prediction_algorithms.evaluation import print_model_comparison

# Folder containing all player reports
reports_folder = Path("season_reports")
test_reports = Path("gcloud/test_reports")
injury_history = Path("data/subjective/injury/injury.csv")
rpe_data = Path("data/subjective/training-load/session.json")

# ------------------------------------------------------------       
# Constants and Hyperparameters
# ------------------------------------------------------------ 

# Metrics to use for baseline modeling
METRICS = [
    "avg_speed",
    "max_speed",
    "avg_heart_rate",
    "max_heart_rate",
    "total_distance",
    "acceleration_impulse",
    "high_speed_distance",
]

# Spans to test for EWMA baseline modeling
SPANS = [3, 6, 9, 12, 15, 18, 21, 24, 27]  # Test for 3, 6, 9, 12, 15, 18, 21, and 24 --- This is N for the calculating alpha in the EWMA

# Workload feature engineering parameters
CHRONIC = 30 
ACUTE = 7 

# Z-score thresholds for flagging deviations
LOW_Z_SCORE_THRESHOLD = 1
MODERATE_Z_SCORE_THRESHOLD = 2
HIGH_Z_SCORE_THRESHOLD = 3

# PCA choices per model (tuneable)
ewma_m = 7
bayesian_m = 1
pop_m = 2


# ------------------------------------------------------------       
# Main functions
# ------------------------------------------------------------ 

def dataset_information(sessions, players):
    """Prints basic information about the dataset."""
    
    print("Total number of sessions:", len(sessions))
    print("Total number of players:",   len(players))

    # Check number of sessions for each player
    for player in players.values():
        print(f"Player {player.player_id} has {player.number_of_sessions()} sessions.")

def get_features(players, span, chronic, acute, variables):
    """Construct a list of data objects for each player."""
    Players_workload_metrics = construct_features(players, span, chronic, acute, variables)

    return Players_workload_metrics

def baseline_modeling(players, spans, ewma_metrics, population_metrics, bayesian_metrics):
    """Compute the EWMA and population baselines with model-specific PCA metrics."""
    # EWMA Baseline Modeling:
    print("Computing EWMA baselines...")
    best_ewma_span, average_span_errors_ewma, best_ewma_metric_errors, players_baselines = run_ewma_baseline_evaluation(
        players, ewma_metrics, spans
    )

    # Bayesian Baseline Modeling:
    print("\nComputing Bayesian baselines...")
    # include supervised PCA metrics in the Bayesian baseline fit
    players_baselines = bayesian_baselines(players, players_baselines, bayesian_metrics)

    # Population Based Moving Average Baseline
    print("Computing Population Based Moving Average baselines...")
    population_span, average_span_errors_population, best_population_metric_errors, population_baselines = population_based_moving_average_baseline(
        players, players_baselines, population_metrics, spans
    )

    return players_baselines, best_ewma_span, population_baselines, population_span


def analyse_baselines(players, players_baselines, population_baselines, metrics, low_threshold, moderate_threshold, high_threshold):
    """Analyze the baselines to identify deviations."""
    
    players_deviations = compute_z_scores(players, players_baselines, population_baselines, metrics, low_threshold=low_threshold, moderate_threshold=moderate_threshold, high_threshold=high_threshold)
    print_deviation_summary(players_deviations)
    
    return players_deviations


def apply_supervised_pca(players, metrics, n_components=1, m=2, attr_name="supervised_pca"):
    """Compute supervised PCA, attach it to sessions under `attr_name`, and rebuild player series.

    Returns (spca_vals, top_k)
    """
    all_sessions = []
    x_rows = []
    y_rows = []

    for player in players.values():
        for session in player.sessions.values():
            all_sessions.append(session)

            row = []
            for metric in metrics:
                value = getattr(session, metric, None)
                if value is None:
                    value = 0
                row.append(value)
            x_rows.append(row)

            # Target: whether the player has an injury on this session date.
            y_rows.append(1 if player.has_injury_on_date(session.session) else 0)

    if not x_rows:
        print("Warning: no sessions found, skipping supervised PCA")
        return None, None

    x_mat = np.nan_to_num(np.array(x_rows, dtype=float))
    y_vec = np.array(y_rows, dtype=int)
    m = min(m, x_mat.shape[1])

    components, pca_model, top_m = supervised_pca(x_mat, y_vec, n_components=n_components, m=m)
    spca_vals = components[:, 0] if components.ndim == 2 and components.shape[1] >= 1 else components

    for session, value in zip(all_sessions, spca_vals):
        setattr(session, attr_name, float(value))

    for player in players.values():
        player.build_unmodified_values()

    print(f"✔ Supervised PCA computed and attached to {len(spca_vals)} sessions as '{attr_name}' (top features: {list(top_m)})")
    return spca_vals, top_m


def main():
    print("\n========== LOADING DATA ==========\n")
    
    #--------------------------------------------------
    # Step 1: Data Loading
    #--------------------------------------------------
    print("[Step 1] Loading data from reports...")
    sessions, players = load_data(reports_folder, injury_history, rpe_data)
    
    print("✔ Data loading complete.")
    print(f"   - Total sessions loaded: {len(sessions)}")
    print(f"   - Total players loaded:  {len(players)}\n")
    
    print("Total sessions with marked injuries:", sum(1 for session in sessions if session.injury))

    # --------------------------------------------------
    # Step 2: Supervised PCA
    # --------------------------------------------------
    print("[Step 2] Computing supervised PCA components (model-specific)...")

    # compute two supervised PCA variants and attach as separate session attributes
    apply_supervised_pca(players, METRICS, n_components=1, m=ewma_m, attr_name="spca_ewma")
    apply_supervised_pca(players, METRICS, n_components=1, m=bayesian_m, attr_name="spca_bayesian")
    apply_supervised_pca(players, METRICS, n_components=1, m=pop_m, attr_name="spca_pop")

    #--------------------------------------------------
    # Step 3: Baseline Modeling (model-specific PCA metrics)
    #--------------------------------------------------
    print("[Step 3] Computing baselines")

    ewma_metrics = METRICS + ["spca_ewma"]
    bayesian_metrics = METRICS + ["spca_bayesian"]
    pop_metrics = METRICS + ["spca_pop"]


    baselines, best_ewma_span, population_baselines, population_span = baseline_modeling(
        players,
        SPANS,
        ewma_metrics,
        pop_metrics,
        bayesian_metrics
    )

    print("✔ Baseline modeling complete.\n")

    #--------------------------------------------------
    # Step 4: Internal vs External Load Analysis
    #--------------------------------------------------
    print("\n[Step 4] Analyzing internal vs external loads...")

    baselines = analyze_internal_vs_external(
        players,
        baselines,
        LOW_Z_SCORE_THRESHOLD,
        MODERATE_Z_SCORE_THRESHOLD,
        HIGH_Z_SCORE_THRESHOLD,
    )

    print("✔ Internal vs external analysis complete.\n")


    #--------------------------------------------------
    # Step 5: Deviation Modeling
    #--------------------------------------------------
    print("[Step 6] Analyzing deviations...")

    metrics_for_deviations = METRICS + ["spca_ewma", "spca_bayesian", "spca_pop"] 
    deviations = analyse_baselines(
        players,
        baselines,
        population_baselines,
        metrics_for_deviations,
        LOW_Z_SCORE_THRESHOLD,
        MODERATE_Z_SCORE_THRESHOLD,
        HIGH_Z_SCORE_THRESHOLD,
    )
    #--------------------------------------------------
    # Step 6: Workload Dynamics
    #--------------------------------------------------
    print("[Step 5] Analyzing workload dynamics...")
    
    players_workload_metrics = get_features(players, 9, CHRONIC, ACUTE, METRICS)
    
    print("✔ Workload dynamics analysis complete.\n")

    #--------------------------------------------------
    # Step 7: Predictive Modeling
    #--------------------------------------------------

    print("\n [Step 7] Training predictive models.... ")

    # Construct model feature set
    print("Model features constructed. Sample features for first session of first player:")

    # Random Split into train and test sets
    X, y = model_feature_set(
        players,
        players_workload_metrics,
        deviations,
        baselines
    )

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    
    # Cross Validate and evaluate models
    print("\nCross-validating logistic regression model...")
    cross_validate_logistic_regression(X_train, y_train)
    print("\nCross-validating decision tree model...")
    cross_validate_decision_tree(X_train, y_train)
    print("\nCross-validating random forest model...")
    cross_validate_random_forest(X_train, y_train)
    
    print("\nTraining and evaluating models on test set...")
    lr_model, lr_metrics = logistic_regression_model(X_train, y_train, X_test, y_test)
    dt_model, dt_metrics = decision_tree_model(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics = random_forest_model(X_train, y_train, X_test, y_test)
    
    #--------------------------------------------------
    # Step 8: Baseline Evaluation
    #--------------------------------------------------
    print("\n [Step 8] Evaluating baselines...")

    # Evaluate regression metrics (how well baselines match actual values)
    regression_results, regression_summary = evaluate_baselines(players, baselines, population_baselines, best_ewma_span, population_span)
    
    # Evaluate classification metrics (how well flags predict injuries)
    flag_results, flag_summary = evaluate_flags(players, deviations)
    
    # Print comprehensive evaluation summary
    print_evaluation(
    regression_results=regression_results,
    regression_summary=regression_summary,
    flag_results=flag_results,
    flag_summary=flag_summary,
    player_level=False  # Set to True for detailed per-player breakdown
    )
    
    # Print model comparison - Predictive models trained in Step 7 are compared side-by-side here
    print_model_comparison([lr_metrics, dt_metrics, rf_metrics])
    
    print("✔ Evaluation complete.")


    print("\n ========== ANALYSIS FINISHED ==========\n")




if __name__ == "__main__":
    main()
    
