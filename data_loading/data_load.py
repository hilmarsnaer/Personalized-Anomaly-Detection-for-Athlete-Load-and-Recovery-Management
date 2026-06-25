# This file is used to process the raw data and prepare it for analysis

import csv
import json
from pathlib import Path
from datetime import datetime

from .data_classes import Session, Player


def load_data(directory_path, injury_history, rpe_data, sessions=None, players=None):
    """
    Load all CSV session files from a directory into Session and Player objects.
    Also adds injury history and RPE data.
    """

    if sessions is None:
        sessions = []

    if players is None:
        players = {}

    directory_path = Path(directory_path)
    injury_lookup = load_injury_history(injury_history)

    for file_path in sorted(directory_path.glob("*.csv")):
        if not file_path.name.endswith("_season_report.csv"):
            continue

        with open(file_path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                session = Session.from_csv_row(row)
                
                # Clean session
                session = clean_session_values(session)

                player_id = session.player_id

                # Convert session filename/date to clean date
                session_date = clean_session_date(session.session)
                session.session = session_date
                
                # Store injury label in session
                session.injury = (
                    player_id in injury_lookup
                    and any(entry["date"] == session_date for entry in injury_lookup[player_id])
                )
                
                # Create player if not already created
                if player_id not in players:
                    players[player_id] = Player(player_id=player_id)

                    # Add injury dates to the player
                    if player_id in injury_lookup:
                        for injury_entry in injury_lookup[player_id]:
                            players[player_id].add_injury_date(injury_entry["date"])

                sessions.append(session)
                players[player_id].add_session(session)
                

    for player in players.values():
        label_sessions_with_future_injury(player)
        player.build_unmodified_values()
                
    return sessions, players

def export_sessions_to_csv(sessions, output_directory):
    """
    Export one CSV file per session date.
    Each file contains all players from that session date.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "player_id",
        "session",
        "rows",
        "columns",
        "avg_speed",
        "max_speed",
        "avg_heart_rate",
        "max_heart_rate",
        "total_distance",
        "acceleration_impulse",
        "high_speed_distance",
        "duration",
        "injury",
        "future_injury",
    ]

    sessions_by_date = {}

    for session in sessions:
        session_date = session.session

        if session_date not in sessions_by_date:
            sessions_by_date[session_date] = []

        sessions_by_date[session_date].append(session)

    for session_date, session_list in sessions_by_date.items():

        file_name = f"{session_date}.csv"
        file_path = output_directory / file_name

        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for session in session_list:
                writer.writerow({
                    "player_id": session.player_id,
                    "session": session.session,
                    "rows": session.rows,
                    "columns": session.columns,
                    "avg_speed": session.avg_speed,
                    "max_speed": session.max_speed,
                    "avg_heart_rate": session.avg_heart_rate,
                    "max_heart_rate": session.max_heart_rate,
                    "total_distance": session.total_distance,
                    "acceleration_impulse": session.acceleration_impulse,
                    "high_speed_distance": session.high_speed_distance,
                    "duration": session.duration,
                    "injury": session.injury,
                    "future_injury": session.future_injury,
                })

    print(f"Exported {len(sessions_by_date)} session-date files to: {output_directory}")




def clean_session_date(session_name):
    """
    Extracts date from session filename.

    Example:
    2020-06-01-TeamA-playerid.parquet
    becomes:
    2020-06-01
    """
    return session_name[:10]


def load_injury_history(injury_history_path):
    """Loads injury history from a CSV file."""

    injury_lookup = {}

    if injury_history_path is None:
        return injury_lookup

    injury_history_path = Path(injury_history_path)

    with open(injury_history_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:

            # Original:
            # TeamA-b58af410-da77-479e-b93c-e03617b9f36d

            player_name = row["player_name"]

            # Remove Team name
            player_id = player_name.split("-", 1)[1]

            # Convert:
            # 20.03.2020 -> 2020-03-20
            injury_date = datetime.strptime(
                row["timestamp"],
                "%d.%m.%Y"
            ).strftime("%Y-%m-%d")

            injury_entry = {
                "date": injury_date,
            }

            if player_id not in injury_lookup:
                injury_lookup[player_id] = []

            injury_lookup[player_id].append(injury_entry)

    return injury_lookup



def clean_session_values(session):
    """
    Clean impossible or corrupted session-level metric values.
    """

    # Speed values, assuming m/s
    if session.avg_speed is not None and session.avg_speed > 10:
        session.avg_speed = None

    if session.max_speed is not None and session.max_speed > 10:
        session.max_speed = None

    # Heart rate values
    if session.avg_heart_rate is not None and session.avg_heart_rate > 210:
        session.avg_heart_rate = None

    if session.max_heart_rate is not None and session.max_heart_rate > 210:
        session.max_heart_rate = None

    # Workload/distance metrics
    if session.total_distance is not None and (session.total_distance > 20000 or session.total_distance < 0):
        session.total_distance = None

    if session.acceleration_impulse is not None and session.acceleration_impulse < 0:
        session.acceleration_impulse = None

    if session.high_speed_distance is not None and session.high_speed_distance > 5000:
        session.high_speed_distance = None

    return session



def label_sessions_with_future_injury(player):
    """If an injury date occurs after a session, and there is no session between"""

    sessions = sorted(
        player.sessions.values(),
        key=lambda s: s.session
    )

    session_dates = [s.session for s in sessions]

    for injury_date in player.injury_history:

        # Injury occurs on a session
        if injury_date in session_dates:

            idx = session_dates.index(injury_date)
            sessions[idx].future_injury = True

        # Injury occurs between sessions
        else:
            # Find the most recent session before the injury date
            previous_sessions = [
                i for i, date in enumerate(session_dates)
                if date < injury_date
            ]

            if previous_sessions:

                previous_idx = previous_sessions[-1]
                sessions[previous_idx].future_injury = True