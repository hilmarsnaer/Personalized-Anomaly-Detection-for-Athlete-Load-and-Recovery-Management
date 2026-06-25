import os
import numpy as np
import pandas as pd
from data_loading.data_classes import PlayerDeviationScores


# Set minimum standard deviation values for each metric to prevent extreme z-scores due to very low variability in residuals.
MIN_STD_BY_METRIC = {
    "avg_heart_rate": 5.0,
    "max_heart_rate": 8.0,       # slightly higher — max HR is noisier
    "avg_speed": 0.3,
    "max_speed": 0.5,
    "total_distance": 300.0,
    "acceleration_impulse": 30000.0,  # was 1000 — way too low
    "high_speed_distance": 100.0,     # was 50 — still too low
    "hr_residual": 5.0,               # was 1.0 — far too low
}

# Collects large deviation records across all z_score calls within one run
_large_deviation_records = []

# Output directory for results
RESULTS_DIR = "results"


#--------------------------------------------------
# Z Score
#--------------------------------------------------


def compute_z_scores(players, players_baselines, population_baselines, metrics, low_threshold, moderate_threshold, high_threshold):
    """
    Compute z-score deviations for each player and metric.
    """
    # Clear large deviation records at the start of each run
    global _large_deviation_records
    _large_deviation_records = []

    all_player_deviation_scores = {}

    for player in players.values():
        player_id = player.player_id
        player_deviations = PlayerDeviationScores(player_id=player_id)
        
        # Get player baselines:
        player_baselines= players_baselines[player_id]
        observed_values = player.unmodified_values
        ewma_baselines = player_baselines.ewma_baselines
        bayesian_baselines = player_baselines.bayesian_baselines
        external_vs_internal_baselines = player_baselines.int_vs_ext_baselines
        
        # Initialize nested dictionaries for this player
        player_deviations.ewma_z_scores = {}
        player_deviations.ewma_flags = {}
        
        player_deviations.bayesian_z_scores = {}
        player_deviations.bayesian_flags = {}
        
        player_deviations.population_based_moving_average_z_scores = {}
        player_deviations.population_based_moving_average_flags = {}
        
        player_deviations.external_vs_internal_z_scores = {}
        player_deviations.external_vs_internal_flags = {}
        
        for metric in metrics:
            actual_values = observed_values.get(metric)

            # -------------------------------
            # EWMA z-scores
            # -------------------------------
            ewma_values = ewma_baselines.get(metric)

            if actual_values is not None and ewma_values is not None:
                ewma_z = z_score(
                    actual_values,
                    ewma_values,
                    metric=metric,
                    player_id=player_id,
                    baseline_type="EWMA"
                )

                player_deviations.ewma_z_scores[metric] = ewma_z
                player_deviations.ewma_flags[metric] = classify_z_scores(ewma_z, low_threshold, moderate_threshold, high_threshold)
            
            # -------------------------------
            # Bayesian z-scores
            # -------------------------------
            bayesian_values = bayesian_baselines.get(metric)

            if actual_values is not None and bayesian_values is not None:
                bayesian_z = z_score(actual_values, bayesian_values, metric=metric, player_id=player_id, baseline_type="Bayesian")

                player_deviations.bayesian_z_scores[metric] = bayesian_z
                player_deviations.bayesian_flags[metric] = classify_z_scores(bayesian_z, low_threshold, moderate_threshold, high_threshold)
                
            # -----------------------------------------
            # Population Based Moving Average z-scores
            # -----------------------------------------
            
            population_values = get_population_values_for_player(player, population_baselines, metric)

            if actual_values is not None and population_values is not None:
                population_z = z_score(
                    actual_values,
                    population_values,
                    metric=metric,
                    player_id=player_id,
                    baseline_type="Population"
                )
                
                player_deviations.population_based_moving_average_z_scores[metric] = population_z
                player_deviations.population_based_moving_average_flags[metric] = classify_z_scores(population_z, low_threshold, moderate_threshold, high_threshold)            
        
        # -----------------------------------------
        # Internal vs External z-scores
        # -----------------------------------------
        
        if external_vs_internal_baselines:
            residuals = external_vs_internal_baselines.get("hr_residual")
            
            if residuals is not None:
                ext_int_z = z_score(
                    residuals,
                    [0] * len(residuals),
                    metric="hr_residual",
                    player_id=player_id,
                    baseline_type="External vs Internal"
                )
                
                player_deviations.external_vs_internal_z_scores["hr_residual"] = ext_int_z
                player_deviations.external_vs_internal_flags["hr_residual"] = classify_z_scores(ext_int_z, low_threshold, moderate_threshold, high_threshold)

        all_player_deviation_scores[player_id] = player_deviations

    # Write all large deviations collected during this run to CSV
    _write_large_deviations_to_file()

    return all_player_deviation_scores


def z_score(actual_values, baseline_values, metric=None, player_id=None, baseline_type=None):
    """
    Compute z-scores by normalizing residuals using the rolling std of the
    observed values up to time t, aligning with Equation 3.16 in the thesis.
    """
    min_std = MIN_STD_BY_METRIC.get(metric, 1.0)
    z_scores = []

    for i, (actual, baseline) in enumerate(zip(actual_values, baseline_values)):
        if actual is None or baseline is None:
            z_scores.append(None)
            continue

        # Only use observations up to and including time t
        past_values = [
            v for v in actual_values[:i + 1]
            if v is not None and np.isfinite(v)
        ]

        # Need at least 2 points to compute std
        if len(past_values) < 2:
            z_scores.append(None)
            continue

        # Compute the standard deviation of all observed values up to time t
        sigma_t = np.std(past_values, ddof=1)
        sigma_t = max(sigma_t, min_std)

        z_scores.append((actual - baseline) / sigma_t)

    # Collect large deviation records instead of printing
    for actual, baseline, z in zip(actual_values, baseline_values, z_scores):
        if z is not None and np.isfinite(z) and abs(z) > 12:
            _large_deviation_records.append({
                "baseline_type": baseline_type,
                "player_id": player_id,
                "metric": metric,
                "actual": actual,
                "baseline": baseline,
                "residual": actual - baseline,
                "z_score": z,
            })

    return z_scores


def _write_large_deviations_to_file(output_path=None):
    """
    Write all collected large deviation records to results/large_zscore_deviations.csv,
    sorted by absolute z-score descending.
    """
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "large_zscore_deviations.csv")

    # Create the results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not _large_deviation_records:
        print("No large deviations detected.")
        return

    df = pd.DataFrame(_large_deviation_records)
    df["abs_z_score"] = df["z_score"].abs()
    df = df.sort_values("abs_z_score", ascending=False).drop(columns="abs_z_score")
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"Large deviations written to: {output_path} ({len(df)} records)")


def get_population_values_for_player(player, population_baselines, metric):
    """
    Extract population baseline values for one player and one metric.
    """

    baseline_df = population_baselines.get(metric)

    if baseline_df is None:
        return None

    population_values = []

    for session_date in player.sessions.keys():

        session_date = pd.to_datetime(session_date)

        matching_row = baseline_df[
            baseline_df["date"] == session_date
        ]

        if matching_row.empty:
            population_values.append(None)
            continue

        baseline_value = matching_row["population_baseline"].iloc[0]

        if pd.isna(baseline_value):
            population_values.append(None)
        else:
            population_values.append(float(baseline_value))

    return population_values


def classify_z_scores(z_scores, low_threshold, moderate_threshold, high_threshold):
    """
    Classify z-score severity levels.
    """

    classifications = []

    for z in z_scores:

        if z is None:
            classifications.append(None)

        elif abs(z) >= high_threshold:
            classifications.append("high")

        elif abs(z) >= moderate_threshold:
            classifications.append("moderate")

        elif abs(z) >= low_threshold:
            classifications.append("low")

        else:
            classifications.append("normal")

    return classifications


def print_deviation_summary(all_player_deviation_scores):
    """Print a summary of deviation statistics across all players and metrics."""

    print("\n" + "=" * 90)
    print("DEVIATION MODELING SUMMARY")
    print("=" * 90)

    model_names = [
        ("EWMA", "ewma_z_scores", "ewma_flags"),
        ("Bayesian", "bayesian_z_scores", "bayesian_flags"),
        (
            "Population",
            "population_based_moving_average_z_scores",
            "population_based_moving_average_flags"
        ),
        (
            "External vs Internal",
            "external_vs_internal_z_scores",
            "external_vs_internal_flags"
        )
    ]

    for model_name, z_attr, flag_attr in model_names:

        print(f"\n{model_name}")
        print("-" * 90)

        metric_summary = {}

        for deviations in all_player_deviation_scores.values():

            z_dict = getattr(deviations, z_attr)
            flag_dict = getattr(deviations, flag_attr)

            for metric in z_dict:

                if metric not in metric_summary:
                    metric_summary[metric] = {
                        "max_abs_z": 0,
                        "low": 0,
                        "moderate": 0,
                        "high": 0,
                        "valid": 0
                    }

                z_scores = z_dict[metric]
                flags = flag_dict[metric]

                valid_z = [
                    z for z in z_scores
                    if z is not None and np.isfinite(z)
                ]

                if not valid_z:
                    continue

                metric_summary[metric]["max_abs_z"] = max(
                    metric_summary[metric]["max_abs_z"],
                    max(abs(z) for z in valid_z)
                )

                metric_summary[metric]["low"] += sum(
                    1 for f in flags if f == "low"
                )

                metric_summary[metric]["moderate"] += sum(
                    1 for f in flags if f == "moderate"
                )

                metric_summary[metric]["high"] += sum(
                    1 for f in flags if f == "high"
                )

                metric_summary[metric]["valid"] += len(valid_z)

        for metric, stats in metric_summary.items():

            if stats["valid"] > 0:
                low_pct = 100 * stats["low"] / stats["valid"]
                moderate_pct = 100 * stats["moderate"] / stats["valid"]
                high_pct = 100 * stats["high"] / stats["valid"]
            else:
                low_pct = moderate_pct = high_pct = 0

            print(
                f"{metric:<30} | "
                f"Valid: {stats['valid']:<5} | "
                f"Max |z|: {stats['max_abs_z']:>6.2f} | "
                f"Low: {stats['low']:<4} ({low_pct:>5.1f}%) | "
                f"Moderate: {stats['moderate']:<4} ({moderate_pct:>5.1f}%) | "
                f"High: {stats['high']:<4} ({high_pct:>5.1f}%)"
            )
