# NASA POWER Weather Data Provenance

## Dataset

Daily weather data for 640 Census 2011-compatible Indian districts.

## Source

NASA POWER Daily API.

## API Endpoint

NASA POWER temporal daily point API:

https://power.larc.nasa.gov/api/temporal/daily/point

## Source Dataset

NASA/POWER Source Native Resolution Daily Data.

The API metadata identifies MERRA2 and POWER as the underlying sources.

## Temporal Coverage

2011-01-01 to 2025-12-31.

## Geographic Coverage

640 districts based on the Census 2011-compatible district master.

Weather values were requested using the representative latitude and longitude associated with each district.

## Variables

- T2M — Temperature at 2 meters — °C
- T2M_MAX — Maximum temperature at 2 meters — °C
- T2M_MIN — Minimum temperature at 2 meters — °C
- RH2M — Relative humidity at 2 meters — %
- PRECTOTCORR — Corrected precipitation — mm/day

## Processing

1. District coordinates were obtained from the Census 2011 district geography.
2. NASA POWER daily point data was requested for each district.
3. Each district was saved as a separate raw CSV.
4. All 640 raw district files were independently validated.
5. Validated files were combined into a single daily weather dataset.
6. The combined dataset was saved as CSV and Parquet.

## Validation

- Expected districts: 640
- Downloaded districts: 640
- Failed districts: 0
- Rows per district: 5,479
- Total expected rows: 3,506,560
- Duplicate district-date records: 0
- Missing weather values: 0
- Date range: 2011-01-01 to 2025-12-31
- Relative humidity constrained to 0–100%.
- Precipitation checked for negative values.
- Maximum temperature checked against minimum temperature.

## Important Geographic Note

The latitude and longitude used for each district represent a district representative point, not a physical weather station.

Therefore, the NASA POWER values should be interpreted as gridded/model-derived meteorological conditions associated with the district representative point rather than direct station observations.

## Raw Data

Raw district-level files are stored under:

data/raw/weather/

## Processed Data

Combined datasets are stored under:

data/processed/weather/

- weather_daily_2011_2025.csv
- weather_daily_2011_2025.parquet

## Validation Report

data/reports/weather/nasa_power_weather_validation.csv

## Download Report

data/reports/weather/nasa_power_download_report.csv