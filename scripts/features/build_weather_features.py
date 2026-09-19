from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/weather/weather_daily_2011_2025.parquet"
OUTPUT = ROOT / "data/processed/weather/weather_district_year_2011_2025.csv"


def main():

    print("=" * 70)
    print("BUILD WEATHER DISTRICT-YEAR FEATURES")
    print("=" * 70)

    df = pd.read_parquet(
        INPUT,
        columns=[
            "census_district_code",
            "date",
            "T2M",
            "T2M_MAX",
            "T2M_MIN",
            "RH2M",
            "PRECTOTCORR",
        ],
    )

    print(f"Daily rows loaded : {len(df):,}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "RH2M",
        "PRECTOTCORR",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "census_district_code",
            "date",
        ]
    )

    df["year"] = df["date"].dt.year

    # Rainy day = precipitation >= 1 mm
    df["rainy_day"] = (
        df["PRECTOTCORR"] >= 1.0
    ).astype("int8")

    final = (
        df.groupby(
            [
                "census_district_code",
                "year",
            ],
            as_index=False,
        )
        .agg(
            avg_temperature_c=("T2M", "mean"),
            max_temperature_c=("T2M_MAX", "max"),
            min_temperature_c=("T2M_MIN", "min"),
            avg_humidity_pct=("RH2M", "mean"),
            total_rainfall_mm=("PRECTOTCORR", "sum"),
            rainy_days=("rainy_day", "sum"),
        )
    )

    # Round continuous features
    final["avg_temperature_c"] = final[
        "avg_temperature_c"
    ].round(2)

    final["max_temperature_c"] = final[
        "max_temperature_c"
    ].round(2)

    final["min_temperature_c"] = final[
        "min_temperature_c"
    ].round(2)

    final["avg_humidity_pct"] = final[
        "avg_humidity_pct"
    ].round(2)

    final["total_rainfall_mm"] = final[
        "total_rainfall_mm"
    ].round(2)

    # Validation
    if final.empty:
        raise ValueError("Weather feature dataset is empty.")

    if final.duplicated(
        subset=[
            "census_district_code",
            "year",
        ]
    ).any():
        raise ValueError(
            "Duplicate district-year rows detected."
        )

    if (
        final["total_rainfall_mm"] < 0
    ).any():
        raise ValueError(
            "Negative rainfall detected."
        )

    if (
        final["avg_humidity_pct"] < 0
    ).any() or (
        final["avg_humidity_pct"] > 100
    ).any():
        raise ValueError(
            "Invalid humidity values detected."
        )

    final = final.sort_values(
        [
            "census_district_code",
            "year",
        ]
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("OUTPUT")
    print("-" * 70)
    print(
        f"District-year rows : {len(final):,}"
    )
    print(
        f"Unique districts   : "
        f"{final['census_district_code'].nunique():,}"
    )
    print(
        f"Year range         : "
        f"{final['year'].min()} - "
        f"{final['year'].max()}"
    )

    print()
    print(f"OUTPUT : {OUTPUT}")
    print()
    print("STATUS: PASSED")


if __name__ == "__main__":
    main()