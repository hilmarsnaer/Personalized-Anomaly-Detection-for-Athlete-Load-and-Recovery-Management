### Fékk þetta ekki til að virka

import duckdb as dd
import os
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# Try different credential sources
creds_path = Path.home() / ".config/gcloud/application_default_credentials.json"

# Load and refresh credentials to get a valid token
credentials = service_account.Credentials.from_service_account_file(str(creds_path))
credentials.refresh(Request())

# Get the access token
access_token = credentials.token

# Create DuckDB connection
conn = dd.connect()
conn.install_extension("httpfs")
conn.load_extension("httpfs")

# Create secret with OAuth2 bearer token
conn.execute(f"""
    CREATE SECRET (
        TYPE GCS,
        PROVIDER credential_chain
    )
""")

# Now query
query = """
SELECT * FROM 'gs://monitoring-dataset-hrahil/2020-10/2020-10-19/2020-10-19-TeamA-74afe68c-f348-414c-9754-6d6f9df12587.parquet'
LIMIT 10
"""

result = conn.query(query).to_df()
print(result)