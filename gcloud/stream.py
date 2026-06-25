"""
Stream parquet data from Google Cloud Storage in chunks.
Folder structure: bucket/year-month/year-month-day/year-month-day-Team-{PLAYER_UUID}.parquet
Player ID is extracted from filename (the UUID part).
"""

import pandas as pd
import io
from google.cloud import storage
from typing import Generator, Tuple, List, Dict
import re


class GCSParquetStreamer:
    """Stream parquet files from GCS bucket without downloading."""
    
    def __init__(self, bucket_name: str):
        """
        Initialize GCS client.
        
        Args:
            bucket_name: Name of GCS bucket
        """
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name
    
    @staticmethod
    def extract_player_id(filename: str) -> str:
        """
        Extract player ID (UUID) from parquet filename.
        Format: YYYY-MM-DD-Team-{PLAYER_UUID}.parquet
        
        Args:
            filename: Parquet filename
            
        Returns:
            Player UUID or None if not found
        """
        # Match UUID pattern at end of filename (before .parquet)
        match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', filename, re.IGNORECASE)
        return match.group(1) if match else None
    
    def list_all_parquet_files(self) -> Dict[str, List[str]]:
        """
        List all parquet files in bucket and group by player ID.
        
        Returns:
            Dictionary mapping player_id -> list of blob paths
        """
        player_files = {}
        blobs = self.bucket.list_blobs()
        
        for blob in blobs:
            if blob.name.endswith('.parquet'):
                filename = blob.name.split('/')[-1]
                player_id = self.extract_player_id(filename)
                
                if player_id:
                    if player_id not in player_files:
                        player_files[player_id] = []
                    player_files[player_id].append(blob.name)
        
        return player_files
    
    def list_player_ids(self) -> List[str]:
        """
        List all unique player IDs in the bucket.
        
        Returns:
            List of player IDs (UUIDs)
        """
        player_files = self.list_all_parquet_files()
        return sorted(list(player_files.keys()))
    
    def list_player_files(self, player_id: str) -> List[str]:
        """
        List all parquet files for a specific player.
        
        Args:
            player_id: Player UUID
            
        Returns:
            List of blob paths for this player
        """
        player_files = self.list_all_parquet_files()
        return player_files.get(player_id, [])
    
    def stream_player_session(self, blob_path: str, chunk_size: int = 10000) -> Generator[pd.DataFrame, None, None]:
        """
        Stream a single parquet file in chunks.
        
        Args:
            blob_path: Full path to parquet file in GCS
            chunk_size: Number of rows per chunk
            
        Yields:
            DataFrame chunks from the parquet file
        """
        blob = self.bucket.blob(blob_path)
        
        # Download parquet file to memory
        parquet_bytes = blob.download_as_bytes()
        parquet_file = io.BytesIO(parquet_bytes)
        
        # Read parquet in chunks
        df = pd.read_parquet(parquet_file)
        
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]
    
    def stream_player(self, player_id: str, chunk_size: int = 10000) -> Generator[Tuple[str, pd.DataFrame], None, None]:
        """
        Stream all sessions for a player across all dates.
        
        Args:
            player_id: Player UUID
            chunk_size: Number of rows per chunk
            
        Yields:
            Tuple of (session_file_name, dataframe_chunk)
        """
        file_paths = self.list_player_files(player_id)
        
        for file_path in file_paths:
            session_name = file_path.split('/')[-1]  # Get filename
            for chunk in self.stream_player_session(file_path, chunk_size):
                yield session_name, chunk