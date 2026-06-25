# Baseline Models

import os
os.environ["PYTENSOR_FLAGS"] = "mode=FAST_COMPILE,cxx="

import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
from collections import defaultdict

from data_loading.data_classes import PlayerBaselines

#--------------------------------------------------
# EWMA Baseline
#--------------------------------------------------
def run_ewma_baseline_evaluation(players, metrics, spans):
    """Compute EWMA baselines for all spans and evaluate them to find the best span."""

    # Temporary storage for all spans
    all_span_baselines = compute_ewma_baselines_all_spans(players, spans, metrics)

    # Evaluate all spans
    average_span_errors, span_metric_errors = evaluate_spans(players, all_span_baselines, metrics, spans)
    
    best_span = min(
        average_span_errors,
        key=lambda span: average_span_errors[span]
        if average_span_errors[span] is not None
        else float("inf")
    )
    best_span_metric_errors = span_metric_errors[best_span]
    
    print_span_evaluation_results(
        model_name="EWMA",
        spans=spans,
        average_span_errors=average_span_errors,
        span_metric_errors=span_metric_errors,
        best_span=best_span,
        best_span_metric_errors=best_span_metric_errors
    )

    # Store ONLY best span in PlayerBaselines objects
    final_player_baselines = store_best_ewma_baselines(
        all_span_baselines,
        best_span
    )

    return (best_span, average_span_errors, best_span_metric_errors, final_player_baselines)

def compute_ewma_baselines_all_spans(players, spans, metrics):
    """
    Compute EWMA baselines for all players, metrics, and spans.
    """

    all_span_baselines = {}

    for player in players.values():

        all_span_baselines[player.player_id] = {}

        for span in spans:

            all_span_baselines[player.player_id][span] = {}

            for metric in metrics:

                values = player.unmodified_values.get(metric)

                if values is None:
                    continue

                all_span_baselines[player.player_id][span][metric] = lagged_ewma(
                    values,
                    span
                )

    return all_span_baselines

def store_best_ewma_baselines(all_span_baselines, best_span):
    """Store only the best EWMA span in the PlayerBaselines objects."""

    final_player_baselines = {}

    for player_id, span_baselines in all_span_baselines.items():

        player_baseline = PlayerBaselines(player_id=player_id)

        if best_span in span_baselines:
            player_baseline.ewma_baselines = span_baselines[best_span]

        final_player_baselines[player_id] = player_baseline

    return final_player_baselines


def lagged_ewma(values, span):
    """
    Compute the lagged EWMA for the given values and span. Lagged EWMA means the EWMA value 
    at time t is based on values up to time t-1, not including the current value at time t.
    """
    
    ewma_values = ewma(values, span)

    if not ewma_values:
        return []

    return [None] + ewma_values[:-1]

def ewma(values, span):
    """
    Calculate EWMA while handling missing values.
    If a value is missing (None), the previous EWMA value is carried forward instead of resetting the baseline.
    """

    if not values:
        return []

    alpha = 2 / (span + 1)

    ewma_values = []
    previous_ewma = None # Initialize the previous ewma value as None for the first observation

    for value in values:

        # Missing observation
        if value is None or value == 0 or pd.isna(value):

            # If no EWMA exists yet - The baseline has not been initialized
            if previous_ewma is None:
                ewma_values.append(None)

            # Carry previous EWMA forward
            else:
                ewma_values.append(previous_ewma)
            continue

        # First valid observation
        if previous_ewma is None:
            current_ewma = value
        # Standard EWMA update
        else:
            current_ewma = alpha * value + (1 - alpha) * previous_ewma

        ewma_values.append(current_ewma)
        previous_ewma = current_ewma

    return ewma_values

#--------------------------------------------------
# Bayesian Baseline
#--------------------------------------------------

def bayesian_baselines(players, players_baselines, metrics):
    """
    Fit Bayesian hierarchical baseline models for each metric.

    Stores Bayesian baselines as:
    metric_name → list of baseline values
    """

    prepared_data = prepare_bayesian_data(players, metrics) # Prepare data for Bayesian modeling

    # Check if prepared data is empty
    if prepared_data.empty:
        print("No data available for Bayesian baselines.")
        return players_baselines

    # Compute the Bayesian baselines for one metric at a time
    for metric in metrics:

        metric_data = prepared_data[prepared_data["metric"] == metric]

        if metric_data.empty:
            print(f"No data available for metric: {metric}")
            continue

        print(f"\nFitting Bayesian baseline for: {metric}")

        bayesian_result = fit_bayesian_model_for_metric(metric_data)

        updated_players_baselines = store_baselines(bayesian_result, players, players_baselines, metric)

    return updated_players_baselines

   
def prepare_bayesian_data(players, metrics):
    """
    Converts player objects into a dataframe for Bayesian modeling.

    Output:
    player_id | day_index | metric | value
    """

    rows = []

    for player in players.values():
        for metric in metrics:
            values = player.unmodified_values.get(metric)

            if values is None:
                continue

            for day_index, value in enumerate(values):
                if value is None or value == 0 or pd.isna(value):
                    continue

                rows.append({
                    "player_id": player.player_id,
                    "day_index": day_index,
                    "metric": metric,
                    "value": float(value)
                })

    return pd.DataFrame(rows)


def fit_bayesian_model_for_metric(df):
    """
    Fit a Bayesian model for the given metric data.
    """

    player_codes, player_ids = pd.factorize(df["player_id"]) # Factorize player IDs, creating a mapping
    values = df["value"].values.astype(float) # Convert values to float

    value_mean = np.mean(values) # Compute the mean of the values
    value_std = np.std(values) # Compute the standard deviation of the values

    if value_std == 0:
        value_std = 1.0

    # Define the probabilistic model.
    with pm.Model() as model:

        # mu_group : The average baseline across all players
        group_mean = pm.Normal(
            "group_mean",
            mu=value_mean,
            sigma=value_std * 2
        )
        # mu_group : The standard deviation of the group
        group_sd = pm.HalfNormal(
            "group_sd",
            sigma=value_std
        )
        # player_offset : The deviation of each player's baseline from the group mean
        player_offset = pm.Normal(
            "player_offset",
            mu=0,
            sigma=1,
            shape=len(player_ids)
        )
        # player_baseline : The estimated baseline for each player
        player_baseline = pm.Deterministic(
            "player_baseline",
            group_mean + player_offset * group_sd
        )
        # sigma : The standard deviation of the observations
        sigma = pm.HalfNormal(
            "sigma",
            sigma=value_std
        )
        # observed_values : The observed values for each player
        pm.Normal(
            "observed_values",
            mu=player_baseline[player_codes],
            sigma=sigma,
            observed=values
        )
        # Sample from the posterior
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            target_accept=0.95,
            random_seed=42
        )

    baseline_samples = trace.posterior["player_baseline"]
    baseline_means = baseline_samples.mean(dim=("chain", "draw")).values

    return {
        player_id: baseline
        for player_id, baseline in zip(player_ids, baseline_means)
    }


def store_baselines(results, players, players_baselines, metric):
    """
    Stores Bayesian baseline values in the PlayerBaselines object.

    Since the Bayesian model returns one stable baseline per player,
    the value is repeated across the player's time series.
    """

    for player_id, baseline_value in results.items():

        player_values = players[player_id].unmodified_values.get(metric)

        if player_values is None:
            continue

        baseline_series = []

        for value in player_values:

            if value is None or value == 0 or pd.isna(value):
                baseline_series.append(None)
            else:
                baseline_series.append(float(baseline_value))

        players_baselines[player_id].bayesian_baselines[metric] = baseline_series

    return players_baselines


#--------------------------------------------------
# Population baseline
#--------------------------------------------------
def population_based_moving_average_baseline(players, players_baselines, metrics, spans):
    """
    Compute and evaluate population-based moving average baselines.
    Only stores the best population span.
    """

    population_data = prepare_population_data(players, metrics)
    population_averages = compute_population_averages(population_data)

    # Temporary storage for all spans
    population_baselines = compute_population_baselines(population_averages, metrics, spans)
    
    metric_ranges = compute_metric_ranges(players, metrics)

    average_span_errors, span_metric_errors = evaluate_population_spans(
        players,
        population_baselines,
        metrics,
        spans,
        metric_ranges
    )

    best_span = min(
        average_span_errors,
        key=lambda span: average_span_errors[span]
        if average_span_errors[span] is not None
        else float("inf")
    )
    best_span_metric_errors = span_metric_errors[best_span]
    
    print_span_evaluation_results(
        model_name="Population",
        spans=spans,
        average_span_errors=average_span_errors,
        span_metric_errors=span_metric_errors,
        best_span=best_span,
        best_span_metric_errors=best_span_metric_errors
    )

    # Store only best span
    best_population_baselines = store_best_population_baseline(
        population_baselines,
        best_span
    )

    return best_span, average_span_errors, best_span_metric_errors, best_population_baselines


def store_best_population_baseline(population_baselines, best_span):
    """Store only the best population span in a simplified format for later use."""
    best_population_baselines = {}

    for metric, span_dict in population_baselines.items():

        if best_span not in span_dict:
            continue

        best_population_baselines[metric] = span_dict[best_span].rename(
            columns={f"ma_{best_span}": "population_baseline"}
        )

    return best_population_baselines


def prepare_population_data(players, metrics):
    """Prepares data for population-based baseline modeling."""
    
    daily_values = defaultdict(list) # Dictionary to store values for each day across all players
    rows = []

    for player_id, player in players.items():
        for session_date, session in player.sessions.items():

            for metric in metrics:
                value = getattr(session, metric, None)

                if value is not None and value != 0 and not pd.isna(value):
                    rows.append({
                        "player_id": player_id,
                        "date": pd.to_datetime(session_date),
                        "metric": metric,
                        "value": value
                    })

    population_data = pd.DataFrame(rows)

    return population_data


def compute_population_averages(population_data):
    """Computes population averages for each day and metric."""

    population_means = (
        population_data
        .groupby(["date", "metric"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "population_mean"})
        .sort_values(["metric", "date"])
    )

    return population_means


def compute_population_baselines(population_means, metrics, spans):
    """Computes population-based baselines for each day and metric."""

    population_baselines = {}

    for metric in metrics:
        metric_means = population_means[
            population_means["metric"] == metric
        ].copy()

        population_baselines[metric] = {}

        for window in spans:
            # Calculate the lagged moving average for the population mean
            metric_means[f"ma_{window}"] = (
                metric_means["population_mean"]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )

            population_baselines[metric][window] = metric_means[
                ["date", f"ma_{window}"]
            ].copy()
            
    return population_baselines


#--------------------------------------------------
# Span Evaluation
#--------------------------------------------------
def evaluate_spans(players, all_player_baselines, metrics, spans):
    """Evaluate the EWMA based baselines across different spans."""

    def get_actual_and_baseline(span):
        """Generator function to yield actual values and corresponding baseline values for a given span."""

        for player in players.values():

            for metric in metrics:

                actual_values = player.unmodified_values.get(metric)

                baseline_values = (
                    all_player_baselines
                    .get(player.player_id, {})
                    .get(span, {})
                    .get(metric)
                )

                if actual_values is not None and baseline_values is not None:
                    yield metric, actual_values, baseline_values # Pass the results as a generator for memory efficiency

    return evaluate_span_errors(players, metrics, spans, get_actual_and_baseline)

def evaluate_population_spans(players, population_baselines, metrics, spans, metric_ranges):
    """Evaluate population-based baselines across different spans."""

    def get_actual_and_baseline(span):
        """Generator function to yield actual values and corresponding population baseline values for a given span."""

        for player_id, player in players.items():

            actual_values_by_metric = {
                metric: []
                for metric in metrics
            }

            baseline_values_by_metric = {
                metric: []
                for metric in metrics
            }

            for session_date, session in player.sessions.items():

                session_date = pd.to_datetime(session_date) # Ensure session_date is a datetime object for accurate matching with population baseline dates

                for metric in metrics:

                    actual_value = getattr(session, metric, None)

                    if actual_value is None or actual_value == 0 or pd.isna(actual_value):
                        continue

                    baseline_df = population_baselines[metric][span]
                    baseline_column = f"ma_{span}"

                    matching_row = baseline_df[
                        baseline_df["date"] == session_date
                    ]

                    if matching_row.empty:
                        continue

                    baseline_value = matching_row[baseline_column].iloc[0]

                    if pd.isna(baseline_value):
                        continue

                    actual_values_by_metric[metric].append(actual_value)
                    baseline_values_by_metric[metric].append(baseline_value)

            for metric in metrics:

                if actual_values_by_metric[metric]:
                    yield (
                        metric,
                        actual_values_by_metric[metric],
                        baseline_values_by_metric[metric]
                    )

    return evaluate_span_errors(players, metrics, spans, get_actual_and_baseline, metric_ranges)

def mean_absolute_error(actual_values, baseline_values, metric_ranges=None, metric=None):
    """Compute MAE and range-normalized MAE between actual values and baseline values, handling missing data."""

    errors = []
    valid_actuals = []

    for actual, baseline in zip(actual_values, baseline_values):

        if actual is None or baseline is None:
            continue

        if actual == 0 or baseline == 0:
            continue

        if pd.isna(actual) or pd.isna(baseline):
            continue

        errors.append(abs(actual - baseline))
        valid_actuals.append(actual)

    if not errors:
        return None, None

    # --------------------------------------------------
    # MAE
    # --------------------------------------------------

    mae = sum(errors) / len(errors)

    # --------------------------------------------------
    # Range-normalized MAE
    # --------------------------------------------------

    if metric_ranges is not None and metric is not None:
        actual_range = metric_ranges.get(metric)
    else:
        actual_range = max(valid_actuals) - min(valid_actuals)

    if actual_range == 0 or pd.isna(actual_range):
        return mae, None

    nmae = mae / actual_range

    return mae, nmae



def evaluate_span_errors(players, metrics, spans, get_actual_and_baseline, metric_ranges=None):
    """
    Generic span evaluation function.
    Returns:
    - average error per span
    - average error per span per metric
    """
    span_errors = {span: [] for span in spans}

    span_metric_errors = {
        span: {metric: [] for metric in metrics}
        for span in spans
    }

    for span in spans:

        for metric, actual_values, baseline_values in get_actual_and_baseline(span):
            if metric_ranges:
                error, normalized_error = mean_absolute_error(actual_values, baseline_values, metric_ranges, metric)
            else:
                error, normalized_error = mean_absolute_error(actual_values, baseline_values)

            if error is not None:

                # nMAE per metric
                if normalized_error is not None:
                    span_metric_errors[span][metric].append(normalized_error)
                    span_errors[span].append(normalized_error)

    average_span_errors = {}

    for span, errors in span_errors.items():
        if errors:
            average_span_errors[span] = sum(errors) / len(errors)
        else:
            average_span_errors[span] = None

    average_span_metric_errors = {}

    for span, metric_errors in span_metric_errors.items():

        average_span_metric_errors[span] = {}

        for metric, errors in metric_errors.items():

            if errors:
                average_span_metric_errors[span][metric] = sum(errors) / len(errors)
            else:
                average_span_metric_errors[span][metric] = None

    return average_span_errors, average_span_metric_errors


def compute_metric_ranges(players, metrics):
    """
    Calculate the range of values for each metric across all players and sessions, to be used for range-normalized error calculations.
    """
    metric_ranges = {}

    for metric in metrics:

        all_values = []

        for player in players.values():

            values = player.unmodified_values.get(metric, [])

            all_values.extend([
                v for v in values
                if v is not None and v != 0 and not pd.isna(v)
            ])

        if all_values:
            metric_ranges[metric] = max(all_values) - min(all_values)
        else:
            metric_ranges[metric] = None

    return metric_ranges


def print_span_evaluation_results(
    model_name,
    spans,
    average_span_errors,
    span_metric_errors,
    best_span,
    best_span_metric_errors
):
    """
    Print standardized span evaluation results for any baseline model.
    """

    # ==================================================
    # Overall span evaluation
    # ==================================================

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} SPAN EVALUATION")
    print("=" * 60)

    for span in spans:

        error = average_span_errors.get(span)

        if error is not None:
            print(f"Span {span:<3} | Average nMAE: {error:.4f}")
        else:
            print(f"Span {span:<3} | Average nMAE: None")

    print("-" * 60)
    print(f"BEST {model_name.upper()} SPAN: {best_span}")
    print("=" * 60)

    # ==================================================
    # Per-metric table across all spans
    # ==================================================

    print(f"\n{model_name.upper()} PER-METRIC nMAE ACROSS ALL SPANS")
    print("=" * 120)

    header = f"{'Metric':<30}"

    for span in spans:
        header += f"{str(span):>12}"

    print(header)
    print("-" * 120)

    for metric in next(iter(span_metric_errors.values())).keys():

        row = f"{metric:<30}"

        for span in spans:

            error = span_metric_errors[span].get(metric)

            if error is not None:
                row += f"{error:>12.4f}"
            else:
                row += f"{'None':>12}"

        print(row)

    print("=" * 120)

    # ==================================================
    # Best span metric errors
    # ==================================================

    print(f"\nBEST {model_name.upper()} SPAN PER-METRIC ERRORS (Span {best_span})")
    print("-" * 60)

    for metric, error in best_span_metric_errors.items():

        if error is not None:
            print(f"{metric:<30} | nMAE: {error:.4f}")
        else:
            print(f"{metric:<30} | MAE: None")

    print("=" * 60 + "\n")

def compute_metric_ranges(players, metrics):
    """
    Calculate the range of values for each metric across all players and sessions, to be used for range-normalized error calculations.
    """

    metric_ranges = {}

    for metric in metrics:

        max_value = -float("inf")
        min_value = float("inf")

        max_player = None
        min_player = None

        for player in players.values():

            values = player.unmodified_values.get(metric, [])

            for value in values:

                if value is None or value == 0 or pd.isna(value):
                    continue

                if value > max_value:
                    max_value = value
                    max_player = player.player_id

                if value < min_value:
                    min_value = value
                    min_player = player.player_id

        metric_ranges[metric] = max_value - min_value

    return metric_ranges