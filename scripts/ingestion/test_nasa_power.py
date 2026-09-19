import requests
import pandas as pd

# Agra from our district_master
latitude = 27.074561
longitude = 77.891578

start_date = "20250101"
end_date = "20250131"

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR"
    f"&community=SB"
    f"&longitude={longitude}"
    f"&latitude={latitude}"
    f"&start={start_date}"
    f"&end={end_date}"
    f"&format=JSON"
)

print("Requesting NASA POWER data...")
print(url)

response = requests.get(url, timeout=60)

print("\nHTTP status:", response.status_code)

response.raise_for_status()

data = response.json()

# ============================================================
# METADATA
# ============================================================

print("\n" + "=" * 70)
print("NASA POWER METADATA")
print("=" * 70)

print("\nHeader:")
print(data.get("header"))

print("\nMessages:")
print(data.get("messages"))

print("\nParameter metadata:")
print(data.get("parameters"))

# ============================================================
# WEATHER DATA
# ============================================================

print("\n" + "=" * 70)
print("WEATHER DATA")
print("=" * 70)

parameters = data["properties"]["parameter"]

print("\nParameter keys:")
print(parameters.keys())

df = pd.DataFrame(parameters)

df.index.name = "date"

df = df.reset_index()

df["date"] = pd.to_datetime(
    df["date"].astype(str),
    format="%Y%m%d",
)

print("\nData shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isna().sum())

print("\nDate range:")
print(
    df["date"].min(),
    "to",
    df["date"].max(),
)

# ============================================================
# VARIABLE CHECKS
# ============================================================

print("\n" + "=" * 70)
print("VARIABLE CHECKS")
print("=" * 70)

for column in [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
]:
    print(f"\n{column}:")
    print(f"  Minimum: {df[column].min()}")
    print(f"  Maximum: {df[column].max()}")
    print(f"  Mean   : {df[column].mean()}")

print("\n" + "=" * 70)
print("NASA POWER TEST COMPLETE")
print("=" * 70)