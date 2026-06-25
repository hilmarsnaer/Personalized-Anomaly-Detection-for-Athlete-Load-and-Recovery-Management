#--------------------------------------------------------------
# This module models a comparison of internal (heart rate exertion) vs external (GPS-based) loads.
# It is session-level based comparison, so it generates a composite external load score and compares 
# it to an accumulated HR exertion score to give a session a int_vs_ext score, if that score is higher than X than flags.
# I think we should consider using the EWMA and ACWR to create the baseline etc...
# 
# Data reuirements: 
# hr_exertion, total_distance, high_speed_running, acceleration, these variances are accumulated measeures over a session
#--------------------------------------------------------------


import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from feature_engineering import acwr
from data_loading.data_classes import Player, PlayerWorkloadMetrics, PlayerBaselines
from deviation_modeling import z_score



ext_cols = ["total_distance", "high_speed_distance", "acceleration_impulse"] 

def prepare_int_vs_ext_data(players):
    """
    Convert player objects to DataFrame for analysis.
    Similar to prepare_bayesian_data in baselines.py
    """
    rows = []
    
    for player in players.values():
        for session_key, session in player.sessions.items():
            row = {
                "player_id": player.player_id,
                "session": session_key,
                "avg_heart_rate": session.avg_heart_rate,
                "total_distance": session.total_distance,
                "high_speed_distance": session.high_speed_distance,
                "acceleration_impulse": session.acceleration_impulse
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def analyze_internal_vs_external(players, baselines, low_z_threshold, acwr_chronic, acwr_acute, players_workload=None):
    """
    Analyze internal (heart rate) vs external (GPS-based) loads from player dataclasses.
    """
    if players_workload is None:
        players_workload = {pid: PlayerWorkloadMetrics(player_id=pid) for pid in players.keys()}
    
    # Extract data from player objects
    df = prepare_int_vs_ext_data(players)
    
    df = df.copy()

    # standardize external load variables across all sessions (handling missing values) - Compute Z-scores for each external load variable
    for col in ext_cols:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()

    # composite external load score (ignoring missing values per row)
    z_cols = [c + "_z" for c in ext_cols]
    df["external_composite"] = df[z_cols].mean(axis=1, skipna=True) # Takes the average of the three z-scores

    # create columns for results and set as nan or false
    for col in ["hr_expected", "hr_residual", "residual_zscore", "acwr", "model_r2", "model_slope", "model_intercept"]:
        df[col] = np.nan
    df["fatigue_flag"] = False

    # process each player
    for player, g in df.groupby("player_id"):
        idx = g.index
        player_data = df.loc[idx].copy()   
          
        # --- Linear Regression (Expected HR per player) ---
        # Remove rows with NaN in either external_composite or avg_heart_rate
        valid_mask = player_data[["external_composite", "avg_heart_rate"]].notna().all(axis=1)
        valid_data = player_data[valid_mask]
        valid_idx = valid_data.index
        
        x = valid_data[["external_composite"]].values
        y = valid_data["avg_heart_rate"].values
        
        if len(x) > 2:  # Need at least 3 points to fit reliably
            model = LinearRegression().fit(x, y)
            
            # store model parameters
            # produce expected hr from the model
            df.loc[valid_idx, "hr_expected"] = model.predict(x)
            # calculate residuals
            df.loc[valid_idx, "hr_residual"] = y - df.loc[valid_idx, "hr_expected"]
            # store model performance metrics
            # model coefficient of determination - how well the external load explains HR variation (0-1, 1 being perfect fit)
            df.loc[valid_idx, "model_r2"] = model.score(x, y)
            # model slope, how much HR changes per unit change in external load
            df.loc[valid_idx, "model_slope"] = model.coef_[0]
            # model intercept, predicted HR when external load = 0
            df.loc[valid_idx, "model_intercept"] = model.intercept_

            # EWMA-adjusted residual z-score (only for valid indices)
            residuals     = df.loc[valid_idx, "hr_residual"].tolist()
            expected  = df.loc[valid_idx, "hr_expected"].tolist()

            z_scores = z_score(
            actual_values  = [r + e for r, e in zip(residuals, expected)],  # reconstructed actual HR
            baseline_values= expected,
            metric         = "avg_heart_rate",
            player_id      = player,
            baseline_type  = "int_vs_ext",
            )

            for i, j in enumerate(valid_idx):
                df.loc[j, "residual_zscore"] = z_scores[i]

        # Calculate ACWR (only for valid indices)
        composite     = df.loc[valid_idx, "external_composite"].tolist()
        acwr_values    = acwr(composite, acwr_chronic, acwr_acute)
        # Convert None to np.nan for proper float64 assignment
        acwr_values = [np.nan if v is None else v for v in acwr_values]
        df.loc[valid_idx, "acwr"] = acwr_values

    high_residual = df["residual_zscore"] > low_z_threshold
    # both conditions need to be met to flag session
    df["fatigue_flag"] = high_residual

    for player_id, player_group in df.groupby("player_id"):
        baselines[player_id].int_vs_ext_baselines = {
            "hr_expected": player_group["hr_expected"].tolist(),
            "hr_residual": player_group["hr_residual"].tolist(),
            "residual_zscore": player_group["residual_zscore"].tolist(),
            "acwr": player_group["acwr"].tolist(),
            "external_composite": player_group["external_composite"].tolist(),
            "fatigue_flag": player_group["fatigue_flag"].tolist(),
            "model_r2": player_group["model_r2"].iloc[0],
            "model_slope": player_group["model_slope"].iloc[0],
            "model_intercept": player_group["model_intercept"].iloc[0]
        }

    return baselines
