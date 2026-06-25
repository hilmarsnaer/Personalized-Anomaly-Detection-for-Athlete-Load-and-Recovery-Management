# Here we take the EWMA of the past 28 morning/resting HRV values and take the acute/chronic ration on the EWMA values 

# Import 
from feature_engineering import new_ewma, acwr, mswr
import pandas as pd

# Import data
df = pd.read_csv("data.csv")   # file name
data = df["hrv"].tolist()     # column name


def hrv_anomaly_detection(data, alpha, span, acute, chronic):
    """Detect HRV anomalies using EWMA and ACWR."""
    if not data:
        return None

    # Calculate EWMA
    for i in data:
        ewma_values = new_ewma(data[:i + 1], span)

    # Calculate ACWR
    acwr_values = [None] * (chronic)  # Pad with None until we have enough data
    
    for j in range(chronic-1, len(ewma_values)):
        acwr_values.append(acwr(ewma_values[:j + 1], alpha, chronic, acute))

    # This metric is then compared with a threshold to determine if an anomaly is present. --> fatigue
    # This threshold is experimental for now but we need to find studies that could validate this approach.
    # Typical threshold for now could be 0.9 or lower.

    # Calculate MSWR based on ure data points
    for j in data:
        mswr_values = [mswr(data[:j + 1]) for j in range(len(data))]

    # Calculate MSWR based on ure data points
    for i in ewma_values:
        mswr_values_ewma = [mswr(ewma_values[:j + 1]) for j in range(len(ewma_values))]

    print("EWMA Values:", [f"{x:.2f}" for x in ewma_values])
    print("ACWR Values:", [f"{x:.2f}" if x is not None else None for x in acwr_values])
    print("MSWR Values:", [f"{x:.2f}" for x in mswr_values])
    print("MSWR Values (EWMA):", [f"{x:.2f}" for x in mswr_values_ewma])

# Maybe next steps could be visualizing the results
    
def main():
    alpha = 0.1
    span = 7
    acute = 7
    chronic = 30
    hrv_anomaly_detection(data, alpha, span, acute, chronic)
    
if __name__ == "__main__":
    main()
