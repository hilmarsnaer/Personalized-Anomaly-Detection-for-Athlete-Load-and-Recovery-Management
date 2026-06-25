from google.cloud import storage
import pandas as pd
import pyarrow.parquet as pq
import traceback
import gc

client = storage.Client()
bucket = client.bucket("monitoring-dataset-hrahil")

PLAYER_ID = "afc13e53-2c77-48cb-8a78-743663bbb84f"  # change this

# Set to True only if you want the full data loaded into memory.
LOAD_DATA = False


def _print_debug(prefix, blob, extra=""):
    print(prefix, flush=True)
    print(f"  Blob name: {blob.name}", flush=True)
    print(f"  Blob size: {blob.size} bytes", flush=True)
    if extra:
        print(f"  {extra}", flush=True)


def load_player_parquets(bucket, player_id, prefix=None, load_data=False):
    frames = []
    matched_files = []
    total_rows = 0

    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".parquet"):
            continue
        if player_id not in blob.name:
            continue

        try:
            _print_debug(f"→ Inspecting: {blob.name}", blob)

            with blob.open("rb") as file_handle:
                first_bytes = file_handle.read(8)
                file_handle.seek(-8, 2)
                last_bytes = file_handle.read(8)
                file_handle.seek(0)

                print(f"  First 8 bytes:  {first_bytes.hex()}", flush=True)
                print(f"  Last 8 bytes:   {last_bytes.hex()}", flush=True)

                parquet_file = pq.ParquetFile(file_handle)
                metadata = parquet_file.metadata
                print(f"  Row groups:     {metadata.num_row_groups}", flush=True)
                print(f"  Rows in file:   {metadata.num_rows}", flush=True)
                print(f"  Columns:        {metadata.num_columns}", flush=True)

                if load_data:
                    df = parquet_file.read().to_pandas()
                    df["source_file"] = blob.name
                    frames.append(df)
                    total_rows += len(df)
                    print(f"✓ Loaded: {blob.name} ({len(df)} rows)", flush=True)
                else:
                    matched_files.append(blob.name)
                    total_rows += metadata.num_rows
                    print(f"✓ OK: {blob.name}", flush=True)

            gc.collect()
        except Exception as e:
            print(f"✗ Failed: {blob.name}")
            print(f"  Blob size: {blob.size} bytes")
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Exception message: {e}")
            print("  Full traceback:")
            print(traceback.format_exc())

    if not load_data:
        return pd.DataFrame(), matched_files, total_rows

    if not frames:
        return pd.DataFrame(), matched_files, total_rows

    combined = pd.concat(frames, ignore_index=True)
    return combined, matched_files, total_rows


df, files, total_rows = load_player_parquets(bucket, PLAYER_ID, load_data=LOAD_DATA)

print(f"\nLoaded {len(files)} files")
print(f"Total rows: {total_rows}")
print(f"Loaded rows in memory: {len(df)}")

print(df.head())
if not df.empty:
    print(df.iloc[0])

if not df.empty:
    df.iloc[:100].to_csv("./gcloud/bin/first_100_rows.csv", index=False)
    print("\n✓ Saved first 100 rows to first_100_rows.csv")