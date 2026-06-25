# Import necessary libraries and modules:
import numpy as np
import statistics as stats

from data_loading.data_classes import PlayerWorkloadMetrics

def construct_features(players, span, chronic, acute, metrics):
    """Construct unmodified, EWMA, ACWR, and MSWR features for a given player."""
    
    all_players_metrics = {}
    
    for player in players.values():
        
        player_metrics = PlayerWorkloadMetrics(player_id=player.player_id)

        for metric in metrics:
            values = player.unmodified_values.get(metric) # Get unmodified observations for the metric

            if not values:
                continue

            if all(value is None for value in values):
                ewma_result = values
                acwr_result = values
                mswr_result = values
            else:
                ewma_result = ewma(values, span)
                acwr_result = acwr(ewma_result, chronic, acute)
                mswr_result = mswr(ewma_result)

            player_metrics.ewma_values[metric] = ewma_result
            player_metrics.acwr_values[metric] = acwr_result
            player_metrics.mswr_values[metric] = mswr_result
            
        all_players_metrics[player.player_id] = player_metrics

    return all_players_metrics


# ------------------------------------------------------------       
# Exponential Weighted Moving Average (EWMA)
# ------------------------------------------------------------ 



def ewma(x, span):
    """
    Calculate EWMA while handling missing values.

    If x[t] is None, the returned EWMA at that position is also None.
    However, the EWMA is calculated only from the non-None values seen so far.
    """

    if not x:
        return []

    alpha = 2 / (span + 1)
    ewma_values = []

    previous_ewma = None

    for value in x:
        if value is None:
            ewma_values.append(None)
            continue

        if previous_ewma is None: #
            current_ewma = value
        else:
            current_ewma = alpha * value + (1 - alpha) * previous_ewma

        ewma_values.append(current_ewma)
        previous_ewma = current_ewma

    return ewma_values



def ewma_with_deviation_threshold(x, span, num_std=2.0):
    """
    Calculate EWMA baseline and deviation thresholds for anomaly detection.

    Parameters:
        x (list): Input data series
        span (int): Span for EWMA calculation
        num_std (float): Number of standard deviations for threshold

    Returns:
        dict: Contains 'baseline', 'upper_bound', 'lower_bound', 'residuals', 'threshold_std'
    """
    ewma_values = ewma(x, span)
    residuals = [x[i] - ewma_values[i] for i in range(len(x))]

    # Calculate residual statistics
    residual_std = stats.stdev(residuals) if len(residuals) > 1 else 0

    # Create bounds
    upper_bound = [baseline + (num_std * residual_std) for baseline in ewma_values]
    lower_bound = [baseline - (num_std * residual_std) for baseline in ewma_values]

    return {
        'baseline': ewma_values,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'residuals': residuals,
        'threshold_std': residual_std
    }



# ------------------------------------------------------------       
# Acute-Chronic Workload Ratio (ACWR)
# ------------------------------------------------------------ 

def acwr(x, chronic, acute):
    """
    Calculate ACWR while requiring consecutive valid values.

    If x[t] is None, the returned ACWR at that position is also None.
    ACWR is only calculated when there are at least `chronic`
    consecutive non-None values ending at the current position.
    """

    acwr_values = []

    for i in range(len(x)):
        current_value = x[i]

        if current_value is None:
            acwr_values.append(None)
            continue

        # Find the most recent consecutive block of non-None values
        consecutive_values = []

        j = i
        while j >= 0 and x[j] is not None:
            consecutive_values.insert(0, x[j])
            j -= 1

        # Need enough consecutive values for the chronic window
        if len(consecutive_values) < chronic:
            acwr_values.append(None)
            continue

        acute_values = consecutive_values[-acute:]
        chronic_values = consecutive_values[-chronic:]

        chronic_mean = sum(chronic_values) / len(chronic_values)

        if chronic_mean == 0:
            acwr_values.append(None)
            continue

        acute_mean = sum(acute_values) / len(acute_values)

        acwr_values.append(acute_mean / chronic_mean)

    return acwr_values

# ------------------------------------------------------------       
# Mean standard deviation workload ratio (MSWR)
# ------------------------------------------------------------  

def mswr(x):
    """
    Calculate MSWR while handling missing values.

    If x[t] is None, the returned MSWR at that position is also None.
    The calculation uses only the non-None values seen so far.
    """

    mswr_values = []
    valid_values = []

    for value in x:
        if value is None:
            mswr_values.append(None)
            continue

        valid_values.append(value)

        if len(valid_values) < 2:
            # Need at least two values to compute a standard deviation
            mswr_values.append(None)
            continue

        std = np.std(valid_values)

        if std == 0:
            mswr_values.append(0)
        else:
            mswr_values.append(stats.mean(valid_values) / std)

    return mswr_values





