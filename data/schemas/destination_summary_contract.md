# Destination Summary Data Contract (schema_version 1.1)

Version 1.1 renamed `experimental_heatwave_ann` to `experimental_ann_predictions`
(all hazards, produced only by `ml/hazard_inference.py`); nothing else changed.

Canonical data contract for `ml/destination_analysis.py`. Column-level
dictionaries for the underlying tables are in
`travelshield_feature_data_dictionary.csv` and `weather_data_dictionary.csv`.
Real example: `docs/samples/destination_summary_agra_2024-10.json`.

## 1. Identifiers

| Field | Meaning |
|---|---|
| `census_district_code` | Canonical key (integer 1-640), Census 2011 district, from `data/reference/district_master.csv`. Every district-level table joins on it. |
| `district_name`, `state_name` | Names as in the district master (Census 2011 spelling, e.g. `Arunanchal Pradesh`, `NCT of Delhi`). |
| Boundaries | Census 2011. Telangana districts (codes 532-541) are listed under Andhra Pradesh; Leh/Kargil (3, 4) under Jammu & Kashmir. |
| Coordinates | A representative point of the district polygon, **not** a weather station. |

State names differ between sources; they are matched after normalisation
(`&`/`and`, case, punctuation, and the aliases in `_STATE_ALIASES`).

## 2. Source tables and what they can support

| Hazard key | File | Spatial | Temporal | Years | Unit / meaning | Missing means |
|---|---|---|---|---|---|---|
| `weather` | `processed/weather/weather_district_year_2011_2025.csv` | district point (NASA POWER grid) | annual | 2011-2025 | degC, %, mm, days | not expected (complete at build) |
| `heatwave_threshold` | `processed/disaster/heatwave/heatwave_district_year_2011_2025.csv` | district point | annual | 2011-2025 | days with modelled Tmax >= 45 (severe >= 47) degC | not expected; **0 is a real observation** |
| `malaria` | `processed/disease/malaria/malaria_district_year_2000_2024.csv` | district | annual | 2000-2024 | reported cases / deaths (counts) | **unknown** (517 of 640 districts have any row; 2024 is through April) |
| `dengue` | `processed/disease/dengue/dengue_state_year_2021_2026.csv` | **state** | annual | 2021-2026 | reported cases / deaths (counts) | **unknown / `not_reported`** (blank or dash; 2026 provisional, through 2026-02-28) |
| `flood` | `processed/disaster/flood/flood_district_year_1967_2023.csv` | district | annual (event-derived) | 1967-2023 | recorded flood events | **unknown** - only district-years with an event have a row; 502 of 640 districts mapped |

No daily or monthly table is part of the repository (the daily NASA POWER
file is Git-ignored and currently absent). Therefore:

* A requested date range selects the **calendar years it overlaps**; values
  are annual and not restricted to the requested days or months.
* Month-specific values are never derived from annual data
  (`query_period.month_specific_data_available` is always `false`).

## 3. Output structure

```
schema_version, summary_type
destination        {census_district_code, district_name, state_name,
                    latitude, longitude, population_2011,
                    boundary_basis, coordinate_basis}
query_period       {start_date, end_date, requested, years_covered,
                    spans_partial_year, resolution_supported,
                    month_specific_data_available, note}
hazards            {weather, heatwave_threshold, malaria, dengue, flood}  (fixed order)
coverage_overview  {<hazard>: {status, last_observed_year, spatial_basis}}
precautions        [{hazard, text, supported_by, kind}]
experimental_ann_predictions {experimental, status, note, hazards: {heatwave, flood, malaria,
                    dengue, chikungunya}}  # each = the result of ml/hazard_inference.py
                    # (see ml/ML_MODEL_CONTRACT.md); optional, off by default; schema 1.1
limitations        [str]
disclaimer         str
```

Each hazard object has the same keys:

| Key | Type | Notes |
|---|---|---|
| `status` | string | `available`, `partial`, `no_records_in_period`, `no_records_for_destination`, `source_unavailable` |
| `status_reason` | string/null | Explains any non-`available` status |
| `resolution`, `spatial_basis` | object, string | `district_observation`, `district_point_model_derived`, or `state_level_context_not_district_specific` |
| `source`, `units`, `caveats`, (`definition`) | | Provenance and definitions |
| `records` | list | One object per year, sorted by year; missing values are `null` |
| `summary` | object/null | Descriptive aggregates over `records` only; `null` when no records |
| `coverage` | object | `source_year_range`, `destination_year_range`, `years_requested`, `years_with_records_in_period`, `years_without_records_in_period` (`null` for flood, where an absent year is not a gap indicator) |
| `last_observed_year` | int/null | Latest year the source holds for this destination (or its state, for dengue) |

Extra keys: `dengue.state_used` (`state_in_source`, `mapping`), `flood.absence_semantics = "unknown_not_zero"`.

### Dengue records and data quality

Dengue is state-level context; every dengue record is the source row,
unchanged, plus annotations:

| Field | Meaning |
|---|---|
| `dengue_cases`, `dengue_deaths` | Source values. `null` = blank/dash in the source = `not_reported` (never zero). |
| `source_state_label` | The state label exactly as in the source (e.g. `West Bengal*`; the asterisk is kept). |
| `reporting_status` | `reported`, or `not_reported` when both counts are blank/dash. |
| `data_quality.status` | `as_reported` (no known issue), `verification_pending`, or `not_reported`. |
| `data_quality.usable_as_verified_input` | `false` for `verification_pending` and `not_reported`. **Any future scoring, ranking or model input must use only records where this is `true`.** |
| `data_quality.flag_matches_source_value` | Only on flagged records; `false` means the source changed since the flag was written. |

Hazard-level: `dengue.data_quality.verification_pending_years` lists affected
years in the returned period; `dengue.summary.includes_verification_pending_records`
and `years_not_reported` repeat this. A hazard whose returned rows are all
blank/dash has `status: "not_reported"`.

Flags live in `data/reference/dengue_data_quality_flags.csv` (see
`dengue_data_dictionary.csv`). They annotate, never edit, the source. Current flag:
**`West Bengal*`, 2024, reported 441, `verification_pending`** - the meaning of
the asterisk and the completeness of the 2024 figure are unconfirmed. The value
is shown unchanged; no correction is made and it is not judged wrong from the
drop from 30,683 in 2023.

State mapping: Census-2011 Telangana districts (codes 532-541) use Telangana's
own series; Jammu & Kashmir figures stay at state level and are **never** assigned
to Leh or Kargil. Leh/Kargil use a separate `Ladakh` series only if the source
has one (none today, so `no_records_for_destination`); blank/dash Ladakh cells
are `not_reported`. All dengue output keeps
`spatial_basis: state_level_context_not_district_specific`.

### Status semantics

* `available` - every requested year (or, with no dates, at least one year) has a record.
* `partial` - some requested years have no record; they are listed, never zero-filled.
* `no_records_in_period` - the destination has records in other years only.
* `no_records_for_destination` - the source holds nothing for this destination. **Unknown, not zero.**
* `not_reported` - (dengue) the source rows for the period are blank/dash. Unknown, not zero.
* `source_unavailable` - the file is missing, unreadable, lacks required columns or has duplicate keys; only this hazard is affected.

## 4. Rules the contract guarantees

1. No overall numeric risk score, weighting or safety verdict is produced.
2. Missing is `null`/absent; zero appears only when the source records zero.
3. Disease counts are raw counts. No rate is computed (no aligned denominators).
4. Dengue is state-level context shared by all districts of the state.
   Census-2011 Telangana districts use the Telangana series; Leh/Kargil
   never receive Jammu & Kashmir figures. Records flagged
   `verification_pending` or `not_reported` are not verified input.
5. Heatwave figures are NASA POWER threshold indicators, not IMD declarations.
6. Dates are ISO `YYYY-MM-DD`; years are integers; output ordering is deterministic;
   the output contains no timestamps, secrets or local paths.
7. Precautions are generic, labelled `generic_guidance_not_official_advice`,
   and appear only when records for that hazard show non-zero values.

## 5. Errors

Raised by `build_destination_summary` / returned by `safe_destination_summary`
as `{"error": {"code", "message", ...}}`:

| `code` | When |
|---|---|
| `invalid_request` | no identifier; bad code type; bad/reversed/one-sided dates; code and name disagree |
| `destination_not_found` | unknown code/name (`suggestions` may be included) |
| `ambiguous_destination` | name matches several districts (`candidates` listed) |
| `data_unavailable` | the district reference file itself is missing |
