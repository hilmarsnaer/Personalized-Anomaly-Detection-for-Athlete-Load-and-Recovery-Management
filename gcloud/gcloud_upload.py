from google.cloud import storage
import os
from pathlib import Path

# Create a storage client
try:
    print("Attempting to create Google Cloud Storage client...")
    client = storage.Client()
    print("✓ Client created successfully")
except Exception as e:
    print(f"ERROR creating client: {e}")
    exit(1)

# Listening all buckets
print("Buckets in project:")

# Set up paths
local_directory = "/Users/hilmarsnaer/2021"
bucket_name = "monitoring-dataset-hrahil"

# Get the bucket
bucket = client.bucket(bucket_name)

# Upload all files
print(f"Uploading files from {local_directory} to gs://{bucket_name}...")
uploaded_count = 0
for root, dirs, files in os.walk(local_directory):
    for file in files:
        file_path = os.path.join(root, file)
        # Create a relative path to preserve folder structure
        relative_path = os.path.relpath(file_path, local_directory)
        
        blob = bucket.blob(relative_path)
        blob.upload_from_filename(file_path)
        print(f"✓ Uploaded: {relative_path}")
        uploaded_count += 1

print(f"\nTotal files uploaded: {uploaded_count}")