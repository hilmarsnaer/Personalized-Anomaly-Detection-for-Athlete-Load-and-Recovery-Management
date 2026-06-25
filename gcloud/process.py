"""
Processing logic: compute metrics and aggregate session data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from stream import GCSParquetStreamer


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

@dataclass
class ProcessingStats:
    """Tracks raw vs. used data points across a player's sessions."""

    # GPS / distance
    gps_total_rows: int = 0
    gps_dropped_speed_gate: int = 0
    gps_dropped_jump_gate: int = 0
    gps_dropped_signal_quality: int = 0

    # Speed metrics
    speed_total_rows: int = 0
    speed_used_rows: int = 0
    speed_dropped_zero: int = 0
    speed_dropped_above_max: int = 0

    # Heart rate
    hr_total_rows: int = 0
    hr_used_rows: int = 0
    hr_dropped_below_min: int = 0
    hr_dropped_above_max: int = 0

    # Acceleration
    accel_total_rows: int = 0
    accel_used_rows: int = 0
    accel_dropped_speed_gate: int = 0
    accel_dropped_signal_quality: int = 0

    # High-speed distance
    hsd_total_rows: int = 0
    hsd_used_rows: int = 0
    hsd_skipped_no_speed_col: int = 0
    hsd_skipped_threshold_below_gate: int = 0
    hsd_skipped_too_few_rows: int = 0
    hsd_season_max_speed: float = 0.0
    hsd_threshold_used: float = 0.0
    hsd_dropped_below_threshold: int = 0

    # Session counts
    sessions_processed: int = 0

    def merge(self, other: "ProcessingStats") -> None:
        """Add another stats object into this one (in-place). Float fields use max, int fields sum."""
        float_max_fields = {"hsd_season_max_speed", "hsd_threshold_used"}
        for f in self.__dataclass_fields__:
            if f in float_max_fields:
                setattr(self, f, max(getattr(self, f), getattr(other, f)))
            else:
                setattr(self, f, getattr(self, f) + getattr(other, f))

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    def summary_lines(self) -> List[str]:
        def pct(used, total):
            return f"{100 * used / total:.1f}%" if total > 0 else "N/A"

        # Derive GPS used rows directly to avoid double-counting bugs
        gps_used = max(0, (self.gps_total_rows - 1)
                       - self.gps_dropped_speed_gate
                       - self.gps_dropped_jump_gate
                       - self.gps_dropped_signal_quality)

        lines = [
            f"  Sessions processed      : {self.sessions_processed}",
            f"  GPS rows  total/used    : {self.gps_total_rows} / {gps_used}"
            f"  ({pct(gps_used, self.gps_total_rows)} kept)",
            f"    dropped – speed gate  : {self.gps_dropped_speed_gate}",
            f"    dropped – jump gate   : {self.gps_dropped_jump_gate}",
            f"    dropped – signal qual : {self.gps_dropped_signal_quality}",
            f"  Speed rows total/used   : {self.speed_total_rows} / {self.speed_used_rows}"
            f"  ({pct(self.speed_used_rows, self.speed_total_rows)} kept)",
            f"    dropped – zero speed  : {self.speed_dropped_zero}",
            f"    dropped – above 10m/s : {self.speed_dropped_above_max}",
            f"  HR rows   total/used    : {self.hr_total_rows} / {self.hr_used_rows}"
            f"  ({pct(self.hr_used_rows, self.hr_total_rows)} kept)",
            f"    dropped – below 30bpm : {self.hr_dropped_below_min}",
            f"    dropped – above 210bpm: {self.hr_dropped_above_max}",
            f"  Accel rows total/used   : {self.accel_total_rows} / {self.accel_used_rows}"
            f"  ({pct(self.accel_used_rows, self.accel_total_rows)} kept)",
            f"    dropped – speed gate  : {self.accel_dropped_speed_gate}",
            f"    dropped – signal qual : {self.accel_dropped_signal_quality}",
            f"  HSD rows  total/used          : {self.hsd_total_rows} / {self.hsd_used_rows}"
            f"  ({pct(self.hsd_used_rows, self.hsd_total_rows)} kept)",
            f"    season_max_speed            : {self.hsd_season_max_speed:.2f} m/s",
            f"    threshold used (60% of max) : {self.hsd_season_max_speed * 0.60:.2f} m/s",
            f"    dropped – below threshold   : {self.hsd_dropped_below_threshold}",
            f"    dropped – threshold < gate  : {self.hsd_skipped_threshold_below_gate}",
            f"    skipped – too few rows      : {self.hsd_skipped_too_few_rows}",
        ]
        return lines


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def calculate_total_distance(
    chunk: pd.DataFrame,
    speed_gate: float = 0.3,
    jump_gate: float = 5.0,
    signal_quality_gate: float = 100,
    stats: Optional[ProcessingStats] = None,
) -> float:
    if 'lat' not in chunk.columns or 'lon' not in chunk.columns:
        return 0.0

    valid = chunk[['lat', 'lon', 'speed']].copy()
    n = len(valid)

    if stats is not None:
        stats.gps_total_rows += n

    METERS_PER_DEGREE = 111320
    lat_rad = np.radians(valid['lat'].to_numpy())
    lat = valid['lat'].to_numpy()
    lon = valid['lon'].to_numpy()

    distance_lat = np.diff(lat) * METERS_PER_DEGREE
    distance_lon = np.diff(lon) * METERS_PER_DEGREE * np.cos(lat_rad[:-1])
    step_distances = np.sqrt(distance_lat**2 + distance_lon**2)

    # Speed gate
    if 'speed' in valid.columns:
        speed = valid['speed'].to_numpy()
        low_speed_mask = (speed[:-1] < speed_gate) & (speed[1:] < speed_gate)
        if stats is not None:
            stats.gps_dropped_speed_gate += int(low_speed_mask.sum())
        step_distances[low_speed_mask] = 0.0

    # Signal quality gate
    if 'signal_quality' in valid.columns:
        signal_quality = valid['signal_quality'].to_numpy()
        low_quality_mask = signal_quality[:-1] < signal_quality_gate
        if stats is not None:
            stats.gps_dropped_signal_quality += int(low_quality_mask.sum())
        step_distances[low_quality_mask] = 0.0

    # Jump gate
    jump_mask = step_distances > jump_gate
    if stats is not None:
        stats.gps_dropped_jump_gate += int(jump_mask.sum())
    step_distances[jump_mask] = 0.0

    return round(float(step_distances.sum()), 6)


def calculate_acceleration_impulse(
    chunk: pd.DataFrame,
    speed_gate: float = 0.3,
    signal_quality_gate: float = 100,
    stats: Optional[ProcessingStats] = None,
) -> float:
    n = len(chunk)
    if stats is not None:
        stats.accel_total_rows += n

    valid_mask = pd.Series([True] * n, index=chunk.index)

    if 'speed' in chunk.columns:
        speed_low = chunk['speed'] < speed_gate
        if stats is not None:
            stats.accel_dropped_speed_gate += int(speed_low.sum())
        valid_mask &= ~speed_low

    if 'signal_quality' in chunk.columns:
        sig_low = chunk['signal_quality'] < signal_quality_gate
        if stats is not None:
            stats.accel_dropped_signal_quality += int(sig_low.sum())
        valid_mask &= ~sig_low

    filtered = chunk[valid_mask]

    if stats is not None:
        stats.accel_used_rows += len(filtered)

    if len(filtered) == 0:
        return 0.0

    magnitude = np.sqrt(filtered['accl_x']**2 + filtered['accl_y']**2 + filtered['accl_z']**2)
    return round(float(np.sum(magnitude)), 6)


def calculate_high_speed_distance(
    chunk: pd.DataFrame,
    speed_threshold: float,
    high_speed_gate: float = 5.5,
    stats: Optional[ProcessingStats] = None,
    season_max_speed: float = None,
) -> float:
    if 'speed' not in chunk.columns:
        if stats is not None:
            stats.hsd_skipped_no_speed_col += 1
        return 0.0

    if stats is not None:
        stats.hsd_total_rows += len(chunk)
        if season_max_speed is not None:
            stats.hsd_season_max_speed = max(stats.hsd_season_max_speed, season_max_speed)
        stats.hsd_threshold_used = speed_threshold

    if speed_threshold < high_speed_gate:
        if stats is not None:
            stats.hsd_skipped_threshold_below_gate += len(chunk)
        return 0.0

    below_threshold = chunk[chunk['speed'] < speed_threshold]
    high_speed_chunk = chunk[chunk['speed'] >= speed_threshold]

    if stats is not None:
        stats.hsd_dropped_below_threshold += len(below_threshold)
        stats.hsd_used_rows += len(high_speed_chunk)

    if len(high_speed_chunk) < 2:
        if stats is not None:
            stats.hsd_skipped_too_few_rows += 1
        return 0.0

    # No stats passed here to avoid double-counting GPS rows
    return calculate_total_distance(high_speed_chunk)


def compute_session_metrics(
    chunk: pd.DataFrame,
    player_id: str,
    session_name: str,
    season_max_speed: float = None,
    stats: Optional[ProcessingStats] = None,
) -> dict:
    metrics = {
        'player_id': player_id,
        'session': session_name,
        'rows': len(chunk),
        'columns': len(chunk.columns),
    }

    if 'speed' in chunk.columns:
        n = len(chunk)
        zero_mask = chunk['speed'] == 0
        above_mask = chunk['speed'] >= 10
        speed_mask = ~zero_mask & ~above_mask

        if stats is not None:
            stats.speed_total_rows += n
            stats.speed_dropped_zero += int(zero_mask.sum())
            stats.speed_dropped_above_max += int(above_mask.sum())
            stats.speed_used_rows += int(speed_mask.sum())

        metrics['avg_speed'] = chunk[speed_mask]['speed'].mean()
        metrics['max_speed'] = chunk[speed_mask]['speed'].max()

    if 'heart_rate' in chunk.columns:
        n = len(chunk)
        below_mask = chunk['heart_rate'] <= 29
        above_mask = chunk['heart_rate'] >= 210
        hr_mask = ~below_mask & ~above_mask

        if stats is not None:
            stats.hr_total_rows += n
            stats.hr_dropped_below_min += int(below_mask.sum())
            stats.hr_dropped_above_max += int(above_mask.sum())
            stats.hr_used_rows += int(hr_mask.sum())

        metrics['avg_heart_rate'] = chunk[hr_mask]['heart_rate'].mean()
        metrics['max_heart_rate'] = chunk[hr_mask]['heart_rate'].max()

    if 'lat' in chunk.columns and 'lon' in chunk.columns:
        metrics['total_distance'] = calculate_total_distance(chunk, stats=stats)

    if 'accl_x' in chunk.columns and 'accl_y' in chunk.columns and 'accl_z' in chunk.columns:
        metrics['acceleration_impulse'] = calculate_acceleration_impulse(chunk, stats=stats)

    if season_max_speed and season_max_speed > 0:
        high_speed_threshold = max(season_max_speed * 0.60, 5.5)
        metrics['high_speed_distance'] = calculate_high_speed_distance(
            chunk,
            speed_threshold=high_speed_threshold,
            high_speed_gate=5.5,
            stats=stats,
            season_max_speed=season_max_speed,
        )

    if stats is not None:
        stats.sessions_processed += 1

    return metrics


# ---------------------------------------------------------------------------
# Player processing
# ---------------------------------------------------------------------------

def process_player(
    streamer: GCSParquetStreamer,
    player_id: str,
    output_dir: str,
    max_sessions: int = None,
) -> tuple[bool, ProcessingStats]:
    print(f"\n  Processing player: {player_id}")

    season_metrics = []
    session_count = 0
    total_rows = 0
    session_chunks = []
    current_session = None
    season_max_speed = 0.0
    player_stats = ProcessingStats()

    # NOTE: season_max_speed is updated as chunks stream in, so sessions processed
    # earlier in the season may use a lower threshold than later sessions.
    # A two-pass approach would be needed for a perfectly consistent threshold.
    for session_name, chunk in streamer.stream_player(player_id, chunk_size=10000):
        if 'speed' in chunk.columns:
            speed_mask = (chunk['speed'] != 0) & (chunk['speed'] < 10)
            filtered_max = chunk[speed_mask]['speed'].max()
            if pd.notna(filtered_max) and filtered_max > season_max_speed:
                season_max_speed = filtered_max

        if session_name != current_session:
            if session_chunks:
                combined_chunk = pd.concat(session_chunks, ignore_index=True)
                metrics = compute_session_metrics(
                    combined_chunk, player_id, current_session,
                    season_max_speed=season_max_speed,
                    stats=player_stats,
                )
                season_metrics.append(metrics)
                total_rows += len(combined_chunk)
                print(f"    Session {session_count}: {current_session} ({len(combined_chunk)} rows)")

            current_session = session_name
            session_chunks = []
            session_count += 1

            if max_sessions and session_count > max_sessions:
                break

        session_chunks.append(chunk)

    # Last session
    if session_chunks:
        combined_chunk = pd.concat(session_chunks, ignore_index=True)
        metrics = compute_session_metrics(
            combined_chunk, player_id, current_session,
            season_max_speed=season_max_speed,
            stats=player_stats,
        )
        season_metrics.append(metrics)
        total_rows += len(combined_chunk)
        print(f"    Session {session_count}: {current_session} ({len(combined_chunk)} rows)")

    if season_metrics:
        output_path = Path(output_dir) / f"{player_id}_season_report.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(season_metrics).to_csv(output_path, index=False)

        stats_path = Path(output_dir) / f"{player_id}_processing_stats.csv"
        pd.DataFrame([player_stats.to_dict()]).to_csv(stats_path, index=False)

        print(f"    ✓ Saved: {output_path}")
        print(f"    Summary: {session_count} sessions, {total_rows} total rows")
        print(f"    Data retention summary:")
        for line in player_stats.summary_lines():
            print(line)

        return True, player_stats
    else:
        print(f"    ✗ No sessions found for player {player_id}")
        return False, player_stats