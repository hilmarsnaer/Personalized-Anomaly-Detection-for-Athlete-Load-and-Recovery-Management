# model_feature_sets.py
# This module constructs feature sets for predictive modeling based on workload metrics, deviations, and baselines.

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from evaluation import _build_injury_labels


def model_feature_set(players, players_workload_metrics, deviations, players_baselines):
    """Construct a feature set for predictive modeling based on workload metrics, deviations, and baselines."""

    X = []
    y = []

    total_rows = 0
    kept_rows = 0
    skipped_rows = 0

    feature_names = [
        "acwr_hsd",
        "ewma_max_speed",
        "ewma_avg_hr",
        "z_total_distance",
        "z_acceleration",
        "z_max_hr",
        "mswr_acceleration",
        "external_composite"
    ]

    for player_id, player in players.items():

        sessions = sorted(player.sessions.values(), key=lambda s: s.session)

        workload = players_workload_metrics[player_id]
        deviation = deviations[player_id]
        baseline = players_baselines[player_id]

        nan_counts = {name: 0 for name in feature_names}

        labels = _build_injury_labels(player)  # CHANGED: build labels for all sessions of the player
        for idx, session in enumerate(sessions):

            total_rows += 1

            features = [
                workload.mswr_values["high_speed_distance"][idx],
                workload.ewma_values["max_speed"][idx],
                workload.ewma_values["avg_heart_rate"][idx],

                deviation.ewma_z_scores["total_distance"][idx],
                deviation.ewma_z_scores["acceleration_impulse"][idx],
                deviation.ewma_z_scores["max_heart_rate"][idx],

                workload.mswr_values["acceleration_impulse"][idx],

                baseline.int_vs_ext_baselines["external_composite"][idx],
            ]

            features = np.array(features, dtype=float)

            X.append(features)
            y.append(labels[idx])  

            kept_rows += 1

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    X = np.nan_to_num(X, nan=0.0)

    return X, y


def train_test_split_data(
    X,
    y,
    test_size=0.2,
    random_state=42
):
    """Split the data into training and testing sets while maintaining class distribution."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )