# Evaluation of baseline models
""" 
    *Evaluation Metrics

    ** Regression Metrics:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)
   - Median Absolute Error (MedAE)
   
   ** Classification Metrics:
    - Confusion Matrix: Accuracy, Specificity and Sensitivity (using TP, TN, FP, FN)
    - Area Under the ROC Curve (AUROC)
    - Precision, F1-Score

   Per-player modeling and compare their performace against 
   the injury history and create some statistics.
   
   Maybe do some per-metric per-model breakdown

   - Confidence intervals around error estimates
   - Statistical significance tests (paired t-tests)
   - Stability analysis (how consistent each baseline is)
   - Confustion matrix 

   """

from data_loading.data_classes import PlayerBaselines
import numpy as np
import sklearn.metrics as sklearnmetrics
import datetime
import collections as col
 
#--------------------------------------------------------------
# Helper functions:
#--------------------------------------------------------------

def _align_arrays(actual_values, baseline_values, as_binary=False):
    """
    Align two lists/arrays by removing positions where either value is None/NaN.
    Returns (actual_np, baseline_np) or (None, None) if there is insufficient data.
    """

    if not actual_values or not baseline_values:
        return None, None
    
    if  len(actual_values) != len(baseline_values):
        return None, None
    
    # Datatypes to float64
    actual_raw = np.array(actual_values, dtype=np.float64)
    baseline_raw = np.array(baseline_values, dtype=np.float64)

    # Create mask for valid (non-NaN) entries in both arrays
    mask = np.isfinite(actual_raw) & np.isfinite(baseline_raw)
    actual, baseline = actual_raw[mask], baseline_raw[mask]

    if as_binary:
        actual, baseline = actual.astype(int), baseline.astype(int)
        if len(np.unique(baseline)) < 2:  # b is injury_labels
            return None, None

    if len(actual) < 2:
        return None, None

    return actual, baseline

def _eval_model_regression(player_result, model_name, model_baselines, actual_dict):
    """Helper: fill player_result[model_name][metric] with regression metrics."""
    if not model_baselines or actual_dict is None:
        return
    player_result[model_name] = {}
    for metric, baseline_vals in model_baselines.items():
        actual_vals = actual_dict.get(metric)
        player_result[model_name][metric] = regression_metrics(actual_vals, baseline_vals)
 


def regression_metrics(actual_values, baseline_values):
    """ 
    Calculate regression metrics between actual and baseline values.
    Returns a dictionary of metrics or None if insufficient data.
    """
    actual, baseline = _align_arrays(actual_values, baseline_values)

    if actual is None or baseline is None:
        return None
    
    # Guard against division-by-zero in MAPE when actual contains zeros
    safe_for_mape = actual[actual != 0]
    safe_baseline = baseline[actual != 0]
    mape_val = (
        float(sklearnmetrics.mean_absolute_percentage_error(safe_for_mape, safe_baseline))
        if len(safe_for_mape) >= 2
        else np.nan
    )

    return {
        'mae': float(sklearnmetrics.mean_absolute_error(actual, baseline)),
        'mse': float(sklearnmetrics.mean_squared_error(actual, baseline)),
        'rmse': float(np.sqrt(sklearnmetrics.mean_squared_error(actual, baseline))),
        'mape': mape_val,
        'medae': float(sklearnmetrics.median_absolute_error(actual, baseline)),
        'n': int(len(actual))
    }

def classification_metrics(flags, injury_labels, z_scores=None):
    """ 
    Evaluate how well the baseline perform on binary flaging compared to actual injury labels.
    Returns a dictionary of metrics or None if insufficient data.
    """
    # Convert string flags ("normal", "moderate", "high") to binary (0, 1)
    # where 1 = flagged (moderate or high deviation), 0 = normal
    binary_flags = []
    for flag in flags:
        if flag is None:
            binary_flags.append(None)
        elif isinstance(flag, str):
            binary_flags.append(1 if flag in ("moderate", "high") else 0)
        else:
            binary_flags.append(int(flag))  # Already numeric
    
    # Align flags with labels
    flags_aligned, labels_aligned = _align_arrays(binary_flags, injury_labels, as_binary=True)

    if flags_aligned is None:
        return None
 
    tn, fp, fn, tp = sklearnmetrics.confusion_matrix(labels_aligned, flags_aligned, labels=[0, 1]).ravel().tolist()
 
    accuracy    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan   # recall / true-positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan   # true-negative rate
    precision   = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1_score= (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else np.nan
    )

    # AUROC from z-scores (if available), otherwise from binary flags
    auroc = np.nan
    if z_scores is not None:
        z_aligned, labels_for_auroc = _align_arrays(z_scores, injury_labels, as_binary=False)
        if z_aligned is not None:
            try:
                # Use absolute z-score as the score (higher |z| = more anomalous)
                auroc = float(sklearnmetrics.roc_auc_score(labels_for_auroc, np.abs(z_aligned)))
            except ValueError:
                auroc = np.nan

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'auroc': auroc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'n' : int(len(flags_aligned))
    }


def _build_injury_labels(player):
    """
    Build a list of binary injury labels aligned with the player's sorted sessions.
 
    A session is labelled 1 if its date falls within `lookahead_days` days before
    a recorded injury date (or on the injury date itself).
 
    Returns
    -------
    list of int (0 or 1), one per session in chronological order.
    """
    LOOKAHEAD_DAYS = 7   # flag sessions within 1 week before injury
 
    sessions = sorted(player.sessions.values(), key=lambda s: s.session)
    injury_dates = set()
    if hasattr(player, "injury_history") and player.injury_history:
        for date in player.injury_history:
            if isinstance(date, str):
                date = datetime.date.fromisoformat(date)
            injury_dates.add(date)
 
    labels = []
    for session in sessions:
        s_date = session.session
        if isinstance(s_date, str):
            s_date = datetime.date.fromisoformat(s_date)
        flagged = any(
            datetime.timedelta(0) <= (inj - s_date) <= datetime.timedelta(days=LOOKAHEAD_DAYS)
            for inj in injury_dates
        )
        labels.append(1 if flagged else 0)
 
    return labels

#--------------------------------------------------------------
# Main evaluation function:
#--------------------------------------------------------------

def evaluate_baselines(players, baselines, population_baselines, ewma_span, population_span):
    """
    Calculate the evaluation metrics for each player's baselines compared to their actual values and injury labels.

    Returns:
    results : dict  {player_id: {model_name: {metric: regression_metrics_dict}}}
    summary : dict  {model_name: {metric: averaged_regression_metrics_dict}}
    """

    results = {}    
    
    for player_id, player in players.items():
        player_baselines = baselines[player.player_id]
        if player_baselines is None:
            continue    
        
        results[player_id] = {}
 
        # --- EWMA baselines ---
        _eval_model_regression(
            results[player_id], "EWMA",
            player_baselines.ewma_baselines,
            player.unmodified_values,
        )

        # --- Bayesian baselines ---
        _eval_model_regression(
            results[player_id], "Bayesian",
            player_baselines.bayesian_baselines,
            player.unmodified_values,
        )

        # --- Internal vs External baselines ---
        # For interal vs. external, we need to compare the residials to zeros
        if player_baselines.int_vs_ext_baselines:
            residuals = player_baselines.int_vs_ext_baselines.get("hr_residual")
            if residuals is not None and isinstance(residuals, list):
                int_ext_actuals = {"hr_residual": residuals}
                int_ext_baselines_dict = {"hr_residual": [0.0] * len(residuals)}
                _eval_model_regression(
                    results[player_id], "IntVsExt",
                    int_ext_baselines_dict,
                    int_ext_actuals,
                )

        # --- Population baselines ---
        if population_baselines:
            # Match population baseline by date to player's sessions
            pop_baseline_values = {}
            for metric in player.unmodified_values.keys():
                pop_baseline_values[metric] = []
                baseline_df = population_baselines.get(metric)
                
                if baseline_df is None:
                    continue
                
                for session_date, session in player.sessions.items():
                    try:
                        import pandas as pd
                        session_date_pd = pd.to_datetime(session_date)
                        matching_row = baseline_df[baseline_df["date"] == session_date_pd]
                        
                        if not matching_row.empty:
                            baseline_value = matching_row["population_baseline"].iloc[0]
                            if not pd.isna(baseline_value):
                                pop_baseline_values[metric].append(baseline_value)
                    except:
                        continue
            
            _eval_model_regression(
                results[player_id], "Population",
                pop_baseline_values,
                player.unmodified_values,
            )
    summary = _summarise_regression(results)
    return results, summary


def _summarise_regression(results):
    """Average per-metric regression stats across all players for each model."""
    # Accumulate: {model: {metric: {stat: [values]}}}
    accumulator = col.defaultdict(lambda: col.defaultdict(lambda: col.defaultdict(list)))
 
    for player_id, models in results.items():
        for model_name, metrics in models.items():
            for metric, stats in metrics.items():
                if stats is None:
                    continue
                for stat, val in stats.items():
                    if stat == "N":
                        continue
                    if np.isfinite(val):
                        accumulator[model_name][metric][stat].append(val)
 
    summary = {}
    for model_name, metrics in accumulator.items():
        summary[model_name] = {}
        for metric, stats in metrics.items():
            summary[model_name][metric] = {
                stat: float(np.mean(vals)) for stat, vals in stats.items()
            }
 
    return summary



def evaluate_flags(players, deviation_scores):
    """
    Compute classification metrics for anomaly flags against injury ground truth.
 
    Parameters
    ----------
    players          : dict {player_id: Player}
    deviation_scores : dict {player_id: PlayerDeviationScores}
 
    Returns
    -------
    results : dict  {player_id: {model_name: {metric: classification_metrics_dict}}}
    summary : dict  {model_name: {metric: averaged_classification_metrics_dict}}
    """
    results = {}
 
    for player_id, player in players.items():
        dev_scores = deviation_scores.get(player_id)
        if dev_scores is None:
            continue
 
        injury_labels = _build_injury_labels(player)

        flag_sources = {
            "EWMA":       getattr(dev_scores, "ewma_flags",       None),
            "Bayesian":   getattr(dev_scores, "bayesian_flags",   None),
            "Population": getattr(dev_scores, "population_based_moving_average_flags", None),
            "IntVsExt":   getattr(dev_scores, "external_vs_internal_flags", None),
        }

        if not any(injury_labels):
            # No recorded injuries — skip classification for this player
            continue
 
        results[player_id] = {}
 
        
 
        for model_name, metric_flags in flag_sources.items():
            if not metric_flags:
                continue
            results[player_id][model_name] = {}

            # Map model names to z_score attribute names
            z_score_attr_map = {
                "EWMA": "ewma_z_scores",
                "Bayesian": "bayesian_z_scores",
                "Population": "population_based_moving_average_z_scores",
                "IntVsExt": "external_vs_internal_z_scores",
            }
            z_score_attr = z_score_attr_map.get(model_name)
            
            for metric, flags in metric_flags.items():
                z_scores = getattr(dev_scores, z_score_attr, {}).get(metric)
                        
                results[player_id][model_name][metric] = classification_metrics(
                    flags, injury_labels, z_scores=z_scores
                )
 
    summary = _summarise_classification(results)
    return results, summary


def _summarise_classification(results):
    """Average per-metric classification stats across all players for each model."""
    accumulator = col.defaultdict(lambda: col.defaultdict(lambda: col.defaultdict(list)))
 
    for player_id, models in results.items():
        for model_name, metrics in models.items():
            for metric, stats in metrics.items():
                if stats is None:
                    continue
                for stat, val in stats.items():
                    if stat in ("tp", "tn", "fp", "fn", "n"):
                        accumulator[model_name][metric][stat].append(val)
                    elif np.isfinite(val):
                        accumulator[model_name][metric][stat].append(val)
 
    summary = {}
    for model_name, metrics in accumulator.items():
        summary[model_name] = {}
        for metric, stats in metrics.items():
            agg = {}
            for stat, vals in stats.items():
                if stat in ("tp", "tn", "fp", "fn", "n"):
                    agg[stat] = int(np.sum(vals))
                else:
                    agg[stat] = float(np.mean(vals))
            summary[model_name][metric] = agg
 
    return summary


# ---------------------------------------------------------------------------
# Print statements
# ---------------------------------------------------------------------------
 
_REGRESSION_STATS  = ["mae", "mse", "rmse", "mape", "medae"]
_CLASS_STATS       = ["auroc", "accuracy", "sensitivity", "specificity", "precision", "f1_score"]
_CONFUSION_KEYS    = ["tp", "tn", "fp", "fn"]
 
 
def _fmt(val, pct=False):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "N/A"
    return f"{val * 100:6.2f}%" if pct else f"{val:8.4f}"
 
 
def print_evaluation(regression_results=None, regression_summary=None,
                     flag_results=None, flag_summary=None,
                     player_level=False):
    """
    Print a human-readable summary of evaluation results.
 
    Parameters
    ----------
    regression_results  : output of evaluate_baselines()[0]  (optional)
    regression_summary  : output of evaluate_baselines()[1]  (optional)
    flag_results        : output of evaluate_flags()[0]       (optional)
    flag_summary        : output of evaluate_flags()[1]       (optional)
    player_level        : if True, also print per-player breakdowns
    """
    SEP  = "=" * 72
    SEP2 = "-" * 72
 
    # ------------------------------------------------------------------
    # 1. Regression summary (averaged across players)
    # ------------------------------------------------------------------
    if regression_summary:
        print(f"\n{SEP}")
        print("  REGRESSION METRICS  (averaged across players)")
        print(SEP)
        for model_name, metrics in regression_summary.items():
            print(f"\n  Model: {model_name}")
            print(f"  {'metric':<28} {'mae':>8} {'rmse':>8} {'mape':>8} {'medae':>8}")
            print(f"  {SEP2}")
            for metric, stats in sorted(metrics.items()):
                print(
                    f"  {metric:<28}"
                    f" {_fmt(stats.get('mae'))}"
                    f" {_fmt(stats.get('rmse'))}"
                    f" {_fmt(stats.get('mape'))}"
                    f" {_fmt(stats.get('medae'))}"
                )
 
    # ------------------------------------------------------------------
    # 2. Classification summary (averaged across players)
    # ------------------------------------------------------------------
    if flag_summary:
        print(f"\n{SEP}")
        print("  CLASSIFICATION METRICS  (averaged across players with injuries)")
        print(SEP)
        for model_name, metrics in flag_summary.items():
            print(f"\n  Model: {model_name}")
            print(
                f"  {'Metric':<28}"
                f" {'auroc':>7} {'accuracy':>7} {'sensitivity':>7} {'specificity':>7}"
                f" {'precision':>7} {'f1_score':>7}"
                f" {'tp':>5} {'tn':>5} {'fp':>5} {'fn':>5}"
            )
            print(f"  {SEP2}")
            for metric, stats in sorted(metrics.items()):
                print(
                    f"  {metric:<28}"
                    f" {_fmt(stats.get('auroc'), pct=False):>7}"
                    f" {_fmt(stats.get('accuracy'), pct=True):>7}"
                    f" {_fmt(stats.get('sensitivity'), pct=True):>7}"
                    f" {_fmt(stats.get('specificity'), pct=True):>7}"
                    f" {_fmt(stats.get('precision'), pct=True):>7}"
                    f" {_fmt(stats.get('f1_score'), pct=False):>7}"
                    f" {stats.get('tp', 'N/A'):>5}"
                    f" {stats.get('tn', 'N/A'):>5}"
                    f" {stats.get('fp', 'N/A'):>5}"
                    f" {stats.get('fn', 'N/A'):>5}"
                )
