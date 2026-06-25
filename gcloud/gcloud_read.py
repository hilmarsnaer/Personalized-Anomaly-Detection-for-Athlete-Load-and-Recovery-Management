from google.cloud import storage
import pandas as pd
from io import BytesIO

client = storage.Client()
bucket = client.bucket("monitoring-dataset-hrahil")

# List all parquet files
""" print("Parquet files in bucket:")
for blob in bucket.list_blobs():
    if blob.name.endswith('.parquet'):
        print(f"  - {blob.name}") """

# Read a parquet file directly
blob = bucket.blob("2021-11/2021-11-13/2021-11-13-TeamB-afc13e53-2c77-48cb-8a78-743663bbb84f.parquet")  # Replace with your file path
data = blob.download_as_bytes()
df = pd.read_parquet(BytesIO(data))
print(df.head())
print(df.iloc[0])  # Prints first row with all columns
print(f"Total rows: {len(df)}") 

# Save first 100 rows to CSV
df.iloc[:100].to_csv("./gcloud/bin/first_100_rows.csv", index=False)
print("\n✓ Saved first 100 rows to first_100_rows.csv")

