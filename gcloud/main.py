from stream import GCSParquetStreamer
from process import process_player, ProcessingStats
import argparse
import traceback
from datetime import datetime
import os
import pandas as pd


def main(bucket_name: str, output_dir: str = "./season_reports", player_id: str = None, max_sessions: int = None):
    print(f"Starting data pipeline from bucket: {bucket_name}")
    
    streamer = GCSParquetStreamer(bucket_name)
    
    error_log_file = os.path.join(output_dir, f"failed_players_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(output_dir, exist_ok=True)
    
    if player_id:
        player_ids = [player_id]
        if max_sessions:
            print(f"TEST MODE: Processing single player {player_id} (max {max_sessions} sessions)")
        else:
            print(f"TEST MODE: Processing single player: {player_id}")
    else:
        player_ids = streamer.list_player_ids()
        print(f"FULL MODE: Found {len(player_ids)} players")
    
    successful = 0
    failed = 0
    failed_players = []
    fleet_stats = ProcessingStats()
    
    for i, pid in enumerate(player_ids, 1):
        print(f"\n[{i}/{len(player_ids)}]", end=" ")
        try:
            success, player_stats = process_player(streamer, pid, output_dir, max_sessions=max_sessions)
            fleet_stats.merge(player_stats)
            if success:
                successful += 1
            else:
                failed += 1
                msg = f"FAILED: {pid} - process_player returned False"
                print(f"  ✗ {msg}")
                failed_players.append((pid, msg))
        except Exception as e:
            failed += 1
            error_traceback = traceback.format_exc()
            msg = f"EXCEPTION: {pid}\n{error_traceback}"
            print(f"  ✗ Error processing {pid}:")
            print(f"    {str(e)}")
            failed_players.append((pid, str(e)))
    
    # Write error log
    if failed_players:
        with open(error_log_file, 'w') as f:
            f.write(f"Failed players ({len(failed_players)}):\n")
            f.write("="*80 + "\n\n")
            for pid, error in failed_players:
                f.write(f"Player: {pid}\n")
                f.write(f"Error: {error}\n")
                f.write("-"*80 + "\n\n")
        print(f"\n⚠ Error details written to: {error_log_file}")

    # Save fleet-wide stats
    fleet_stats_path = os.path.join(output_dir, "fleet_processing_stats.csv")
    pd.DataFrame([fleet_stats.to_dict()]).to_csv(fleet_stats_path, index=False)

    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete")
    print(f"  Successful: {successful}/{len(player_ids)}")
    print(f"  Failed: {failed}/{len(player_ids)}")
    print(f"  Reports saved to: {output_dir}")
    print(f"  Fleet stats saved to: {fleet_stats_path}")
    if failed_players:
        print(f"  Failed players: {', '.join([p[0][:8] for p in failed_players])}")
    print(f"\n  Fleet-wide data retention:")
    for line in fleet_stats.summary_lines():
        print(line)
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream and process player season data from GCS")
    parser.add_argument("--bucket", default="monitoring-dataset-hrahil", help="GCS bucket name")
    parser.add_argument("--output", default="./season_reports", help="Output directory for season reports")
    parser.add_argument("--player", default=None, help="Optional: process only this player ID (for testing)")
    parser.add_argument("--limit-sessions", type=int, default=None, help="Optional: limit number of sessions to process (for testing)")
    
    args = parser.parse_args()
    
    main(args.bucket, args.output, args.player, args.limit_sessions)