"""Repeatable data inventory and quality audit for Bharat TravelShield AI.

Run from the repository root:

    python -m ml.data_audit                       # print summary, write report
    python -m ml.data_audit --check               # no file written; exit 1 on errors
    python -m ml.data_audit --output ml/DATA_QUALITY_REPORT.md

Errors mean a structural failure (missing required file, missing columns,
duplicate keys, district codes absent from the district master, non-numeric
values). Warnings are known coverage limitations or plausibility concerns.
Nothing is imputed: missing values are only counted. The output is
deterministic for a given set of data files (no timestamps).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ml.destination_analysis import (
    DENGUE_FLAGS_FILE,
    NOT_REPORTED_TOKENS,
    _DENGUE_UNATTRIBUTABLE_CODES,
    _TELANGANA_CENSUS_CODES,
    _norm_state,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_REPORT = ROOT / "ml" / "DATA_QUALITY_REPORT.md"
HEATWAVE_ML_DATASET = ROOT / "ml" / "data" / "heatwave" / "heatwave_t_to_t1_dataset.parquet"


@dataclass(frozen=True)
class Spec:
    name: str
    path: str                       # relative to the repository root
    keys: tuple[str, ...]
    required: bool = True
    district_col: str | None = "census_district_code"
    state_col: str | None = None
    year_col: str | None = "year"
    temporal: str = "annual"
    spatial: str = "district"
    source: str = ""
    units: str = ""
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    nonneg: tuple[str, ...] = ()
    state_level_columns: tuple[str, ...] = ()
    known_gaps: tuple[str, ...] = ()
    missing_means: str = "Not documented; treated as unknown."
    na_tokens: tuple[str, ...] = ()   # cell text meaning "not reported"


SPECS: tuple[Spec, ...] = (
    Spec(
        "district_master", "data/reference/district_master.csv",
        keys=("census_district_code",), state_col="state_name", year_col=None,
        temporal="static", source="Census of India 2011 + district shapefile",
        units="degrees (lat/lon), persons",
        ranges={"latitude": (6, 38), "longitude": (68, 98)},
        nonneg=("population_2011", "households_2011"),
        known_gaps=("Census-2011 boundaries: Telangana districts are listed "
                    "under Andhra Pradesh and Ladakh under Jammu & Kashmir.",),
    ),
    Spec(
        "weather_year", "data/processed/weather/weather_district_year_2011_2025.csv",
        keys=("census_district_code", "year"),
        source="NASA POWER daily point API aggregated by year",
        units="degC, %, mm, days",
        ranges={"avg_temperature_c": (-30, 40), "max_temperature_c": (-10, 55),
                "min_temperature_c": (-60, 40), "avg_humidity_pct": (0, 100),
                "total_rainfall_mm": (0, 12000), "rainy_days": (0, 366)},
        missing_means="No missing values are expected (validated at build).",
        known_gaps=("Annual only; no monthly or travel-window weather table "
                    "is present in the repository.",),
    ),
    Spec(
        "heatwave_year", "data/processed/disaster/heatwave/heatwave_district_year_2011_2025.csv",
        keys=("census_district_code", "year"),
        source="Derived from NASA POWER daily T2M_MAX (>=45/47 degC)",
        units="days, degC",
        ranges={"heatwave_days": (0, 366), "severe_heatwave_days": (0, 366),
                "max_temperature_c": (-10, 55)},
        missing_means="Zero is a real observation (derived from complete daily data).",
        known_gaps=("Threshold indicator, not IMD-declared heatwave events.",),
    ),
    Spec(
        "malaria_district_year", "data/processed/disease/malaria/malaria_district_year_2000_2024.csv",
        keys=("census_district_code", "year"), state_col="state_name",
        source="NCVBDC district-wise malaria tables (PDF extraction)",
        units="reported cases / deaths", nonneg=("total_cases", "total_deaths"),
        missing_means="A district-year absent from the file is unknown, not zero.",
        known_gaps=("Only districts matched to the Census-2011 master are kept.",
                    "2024 rows are THROUGH_APRIL_2024 (partial year)."),
    ),
    Spec(
        "dengue_state_year", "data/processed/disease/dengue/dengue_state_year_2021_2026.csv",
        keys=("state_name", "year"), district_col=None, state_col="state_name",
        spatial="state", source="NCVBDC state-wise dengue tables",
        units="reported cases / deaths", nonneg=("dengue_cases", "dengue_deaths"),
        missing_means="A state-year absent from the file is unknown, not zero. "
                      "Blank or dash cells are 'not_reported', never zero.",
        na_tokens=NOT_REPORTED_TOKENS,
        known_gaps=("State-level only; not district observations.",
                    "No separate Ladakh series: Jammu and Kashmir figures are "
                    "not assigned to Leh or Kargil.",
                    "2026 rows are provisional (through 2026-02-28)."),
    ),
    Spec(
        "flood_district_year", "data/processed/disaster/flood/flood_district_year_1967_2023.csv",
        keys=("census_district_code", "year"), state_col="state_name",
        temporal="annual (event-derived)", source="India Flood Inventory v3 (deterministic LGD-to-Census mapping)",
        units="recorded events", nonneg=("flood_event_count",),
        missing_means="Only district-years with at least one event have a row; "
                      "absence is unknown, not zero.",
        known_gaps=("Unmapped/ambiguous LGD districts were excluded; coverage "
                    "is a subset of the 640 districts.",),
    ),
    Spec(
        "master_features", "data/processed/features/travelshield_district_year_features_2011_2024.csv",
        keys=("census_district_code", "year"), state_col="state_name",
        source="Joined output of the pipeline above", units="see data dictionary",
        state_level_columns=("dengue_cases", "dengue_deaths"),
        nonneg=("malaria_cases", "dengue_cases", "dengue_deaths", "flood_event_count"),
        missing_means="Missing disease/flood values are retained as missing.",
        known_gaps=("dengue_cases/dengue_deaths are state-level values copied "
                    "to each district row.",),
    ),
    Spec(
        "heatwave_ann_dataset", "ml/data/heatwave/heatwave_t_to_t1_dataset.parquet",
        keys=("census_district_code", "year"), required=False,
        source="ml/prepare_heatwave_dataset.py (legacy v1 heatwave ANN experiment)", units="see feature dictionary",
        known_gaps=("Legacy v1 experiment table, superseded by ml/data/hazards/heatwave; kept as evidence.",),
    ),
    *(Spec(
        f"model_dataset_{h}", f"ml/data/hazards/{h}/dataset.parquet",
        keys=("census_district_code", "feature_year"), required=False, state_col="state_name",
        year_col="feature_year", source="ml/hazard_datasets.py (v2 hazard ANN model dataset)",
        units="see ml/data/hazards/" + h + "/feature_dictionary.csv",
        missing_means="Rows with any missing required input are dropped, never imputed; no missing values remain.",
        known_gaps=(note,),
    ) for h, note in (
        ("heatwave", "Annual district rows; target is a NASA POWER threshold indicator, not an IMD declaration."),
        ("flood", "Presence-only flood labels; 502 of 640 districts; 'no record' is encoded as 0 for this target."),
        ("malaria", "Relative top-quartile reported-incidence target; districts without FULL_YEAR records are absent."),
    )),
    Spec(
        "weather_daily", "data/processed/weather/weather_daily_2011_2025.parquet",
        keys=("census_district_code", "date"), required=False, year_col=None,
        temporal="daily", source="NASA POWER daily point API", units="see weather dictionary",
        known_gaps=("Git-ignored large file; not part of the repository.",),
    ),
)


# --------------------------------------------------------------------------

def _read(path: Path, na_tokens: tuple[str, ...] = ()) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, na_values=list(na_tokens) or None)


def audit_dataset(
    spec: Spec,
    root: Path,
    master_codes: set[int],
    master_state_keys: set[str],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {
        "name": spec.name, "path": spec.path, "required": spec.required,
        "temporal_resolution": spec.temporal, "spatial_resolution": spec.spatial,
        "source": spec.source, "units": spec.units,
        "missing_value_semantics": spec.missing_means,
        "present": False,
    }
    path = root / spec.path
    if not path.is_file():
        msg = f"{spec.path} is missing."
        (errors if spec.required else warnings).append(
            msg if spec.required else msg + " Dependent work uses the available "
            "data and is limited accordingly."
        )
        if spec.name == "weather_daily":
            warnings.append(
                "Recovery: restore from the teammate's copy, or regenerate with "
                "scripts/ingestion/download_nasa_power.py then "
                "scripts/cleaning/combine_weather_data.py (640 districts, ~3.5M "
                "rows; a long API download, not run automatically)."
            )
        info.update(errors=errors, warnings=warnings)
        return info

    try:
        df = _read(path, spec.na_tokens)
    except Exception as exc:  # unreadable file is a structural failure
        errors.append(f"{spec.path} could not be read ({type(exc).__name__}).")
        info.update(errors=errors, warnings=warnings)
        return info

    info.update(present=True, size_bytes=path.stat().st_size,
                rows=int(len(df)), columns=list(df.columns))

    missing_cols = [c for c in spec.keys if c not in df.columns]
    if missing_cols:
        errors.append(f"Key column(s) missing: {missing_cols}.")
        info.update(errors=errors, warnings=warnings)
        return info

    # Numeric validity (non-numeric text in numeric columns is an error).
    numeric_cols = [c for c in (*spec.ranges, *spec.nonneg) if c in df.columns]
    numeric = {}
    for col in dict.fromkeys(numeric_cols):
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad = int((coerced.isna() & df[col].notna()).sum())
        if bad:
            errors.append(f"{col}: {bad} non-numeric value(s).")
        numeric[col] = coerced

    # Time range
    if spec.year_col and spec.year_col in df.columns:
        years = pd.to_numeric(df[spec.year_col], errors="coerce").dropna()
        info["year_range"] = [int(years.min()), int(years.max())] if len(years) else None
        info["years_present"] = int(years.nunique())
    elif spec.name == "weather_daily" and "date" in df.columns:
        info["date_range"] = [str(df["date"].min()), str(df["date"].max())]

    # Keys / duplicates
    dup_keys = int(df.duplicated(list(spec.keys)).sum())
    dup_rows = int(df.duplicated().sum())
    info["duplicate_key_rows"], info["duplicate_full_rows"] = dup_keys, dup_rows
    if dup_keys:
        errors.append(f"{dup_keys} duplicate key row(s) on {list(spec.keys)}.")
    elif dup_rows:
        warnings.append(f"{dup_rows} fully duplicated row(s).")

    # Missingness
    miss = df.isna().sum()
    info["missing_by_column"] = {
        c: {"count": int(n), "percent": round(100 * n / max(len(df), 1), 2)}
        for c, n in miss.items() if n
    }

    # Geography
    if spec.district_col and spec.district_col in df.columns:
        codes = pd.to_numeric(df[spec.district_col], errors="coerce")
        info["unique_districts"] = int(codes.nunique())
        unmatched = sorted({int(c) for c in codes.dropna().unique()} - master_codes)
        info["district_codes_not_in_master"] = unmatched
        info["district_master_coverage"] = f"{codes.nunique()}/{len(master_codes)}"
        if unmatched:
            errors.append(f"{len(unmatched)} district code(s) not in the district master: {unmatched[:10]}.")
        if codes.isna().any():
            errors.append(f"{int(codes.isna().sum())} row(s) with no district code.")
    if spec.state_col and spec.state_col in df.columns:
        states = sorted(df[spec.state_col].dropna().astype(str).unique())
        info["unique_states"] = len(states)
        if spec.name != "district_master":
            unmatched_states = sorted(s for s in states if _norm_state(s) not in master_state_keys)
            info["state_names_not_matching_master_after_aliasing"] = unmatched_states
            if unmatched_states and spec.spatial != "state":
                warnings.append(f"State names not matched to master: {unmatched_states}.")
            elif unmatched_states:
                warnings.append(
                    f"Dengue state names with no Census-2011 master equivalent: {unmatched_states} "
                    "(Telangana is handled via the documented district list)."
                )
        if spec.state_col == "state_name" and "*" in "".join(states):
            starred = [s for s in states if "*" in s]
            warnings.append(
                f"State label(s) carry a '*' footnote marker in the source {starred}; "
                "the label is preserved unchanged and its meaning is unconfirmed."
            )
        if spec.na_tokens:
            for col in ("dengue_cases", "dengue_deaths"):
                if col in df.columns and df[col].isna().any():
                    n = int(df[col].isna().sum())
                    info.setdefault("not_reported_cells", {})[col] = n
                    warnings.append(
                        f"{col}: {n} blank/dash cell(s) treated as not_reported (never zero)."
                    )

    # Plausibility
    range_findings = {}
    for col, (lo, hi) in spec.ranges.items():
        if col in numeric:
            n = int(((numeric[col] < lo) | (numeric[col] > hi)).sum())
            if n:
                range_findings[col] = n
                warnings.append(f"{col}: {n} value(s) outside plausible range [{lo}, {hi}].")
    for col in spec.nonneg:
        if col in numeric:
            n = int((numeric[col] < 0).sum())
            if n:
                range_findings[col] = n
                errors.append(f"{col}: {n} negative value(s).")
    info["out_of_range_counts"] = range_findings

    # Coverage flags
    for col in ("data_coverage",):
        if col in df.columns:
            info["data_coverage_values"] = {
                str(k): int(v) for k, v in sorted(df[col].value_counts().items())
            }
    if spec.state_level_columns:
        info["state_level_columns"] = list(spec.state_level_columns)
    warnings.extend(spec.known_gaps)
    info.update(errors=errors, warnings=warnings)
    return info


def check_master_dengue_join(root: Path, master: pd.DataFrame) -> list[str]:
    """Report join defects in the existing master feature table's dengue columns."""
    findings: list[str] = []
    feat_path = root / "data/processed/features/travelshield_district_year_features_2011_2024.csv"
    dengue_path = root / "data/processed/disease/dengue/dengue_state_year_2021_2026.csv"
    if not (feat_path.is_file() and dengue_path.is_file()):
        return findings
    feats = pd.read_csv(feat_path, usecols=["census_district_code", "state_name", "year", "dengue_cases"])
    dengue = pd.read_csv(dengue_path)
    dengue["key"] = dengue["state_name"].map(_norm_state)
    dengue_keys = set(dengue["key"])
    window = feats[feats["year"].between(int(dengue["year"].min()), 2024)]

    expected = window["state_name"].map(_norm_state).isin(dengue_keys)
    lost = window[expected & window["dengue_cases"].isna()]
    if len(lost):
        names = sorted(lost["state_name"].unique())
        findings.append(
            f"master_features: {len(lost)} district-year row(s) in "
            f"{int(dengue['year'].min())}-2024 have no dengue value although the "
            f"dengue source has the state (name aliases not applied in the builder): {names}."
        )
    tel = window[window["census_district_code"].isin(_TELANGANA_CENSUS_CODES)
                 & window["dengue_cases"].notna()]
    if len(tel):
        findings.append(
            f"master_features: {len(tel)} row(s) for the ten Census-2011 Telangana "
            "districts (codes 532-541) carry Andhra Pradesh dengue values, "
            "because Census 2011 lists them under Andhra Pradesh. "
            "The destination layer maps them to Telangana instead."
        )
    ladakh = window[window["census_district_code"].isin(_DENGUE_UNATTRIBUTABLE_CODES)
                    & window["dengue_cases"].notna()]
    if len(ladakh):
        findings.append(
            f"master_features: {len(ladakh)} row(s) for Leh/Kargil carry Jammu & Kashmir "
            "dengue values; the destination layer withholds dengue for them."
        )
    return findings


def check_dengue_flags(root: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Validate data-quality annotations against the dengue source.

    Returns (flag rows, errors, warnings). A flag whose year/label does not
    exist, or whose recorded value differs from the source, is an error, so
    an annotation can never silently outlive the record it describes.
    """
    path = root / "data" / DENGUE_FLAGS_FILE
    source_path = root / SOURCES_DENGUE
    if not path.is_file():
        return [], [], []
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    try:
        flags = pd.read_csv(path)
        need = {"state_name_in_source", "year", "reported_dengue_cases", "status", "reason"}
        if not need <= set(flags.columns):
            return [], [f"{DENGUE_FLAGS_FILE}: missing columns {sorted(need - set(flags.columns))}."], []
        source = pd.read_csv(source_path, na_values=list(NOT_REPORTED_TOKENS)) if source_path.is_file() else None
    except Exception as exc:
        return [], [f"{DENGUE_FLAGS_FILE} could not be read ({type(exc).__name__})."], []
    if source is None:
        return [], [f"{DENGUE_FLAGS_FILE} exists but the dengue source is missing."], []

    for r in flags.to_dict("records"):
        label, year = str(r["state_name_in_source"]), int(r["year"])
        hit = source[(source["state_name"] == label) & (source["year"] == year)]
        row = {"state_name_in_source": label, "year": year, "status": str(r["status"]),
               "reported_dengue_cases": int(r["reported_dengue_cases"])}
        if hit.empty:
            errors.append(f"dengue flag for '{label}' {year} matches no source record "
                          "(label must equal the source label exactly).")
        elif hit["dengue_cases"].iloc[0] != r["reported_dengue_cases"]:
            errors.append(f"dengue flag for '{label}' {year}: flagged value "
                          f"{int(r['reported_dengue_cases'])} differs from source value "
                          f"{hit['dengue_cases'].iloc[0]}.")
        if str(r["status"]) == "verification_pending":
            warnings.append(
                f"dengue '{label}' {year} (reported {int(r['reported_dengue_cases'])}) is "
                "verification_pending: kept unchanged, not verified input for scoring."
            )
        rows.append(row)
    return rows, errors, warnings


SOURCES_DENGUE = "data/processed/disease/dengue/dengue_state_year_2021_2026.csv"


def run_audit(root: Path | str = ROOT) -> dict[str, Any]:
    root = Path(root)
    master_path = root / "data/reference/district_master.csv"
    errors: list[str] = []
    master_codes: set[int] = set()
    master_state_keys: set[str] = set()
    master = None
    if master_path.is_file():
        master = pd.read_csv(master_path)
        master_codes = {int(c) for c in master["census_district_code"].dropna()}
        master_state_keys = {_norm_state(s) for s in master["state_name"].dropna()}
    else:
        errors.append("data/reference/district_master.csv is missing; geography checks skipped.")

    datasets = [audit_dataset(s, root, master_codes, master_state_keys) for s in SPECS]
    cross = check_master_dengue_join(root, master) if master is not None else []

    flag_rows, flag_errors, flag_warnings = check_dengue_flags(root)

    all_errors = errors + [f"{d['name']}: {e}" for d in datasets for e in d["errors"]] + flag_errors
    all_warnings = [f"{d['name']}: {w}" for d in datasets for w in d["warnings"]] + cross + flag_warnings
    return {
        "master_districts": len(master_codes),
        "datasets": datasets,
        "dengue_quality_flags": flag_rows,
        "cross_dataset_findings": cross,
        "errors": all_errors,
        "warnings": all_warnings,
        "status": "FAILED" if all_errors else "PASSED_WITH_WARNINGS" if all_warnings else "PASSED",
    }


# --------------------------------------------------------------------------

def _fmt_size(n: int) -> str:
    return f"{n / 1024:,.0f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:,.1f} MB"


def render_markdown(report: dict[str, Any]) -> str:
    out = [
        "# Data Quality Report",
        "",
        "Generated by `python -m ml.data_audit` (deterministic; no timestamps). "
        "Regenerate after any data change. Counts are recalculated from the "
        "current files, not copied from older build reports.",
        "",
        f"**Status: {report['status']}** | errors: {len(report['errors'])} | "
        f"warnings: {len(report['warnings'])} | district master: "
        f"{report['master_districts']} districts",
        "",
        "## Inventory",
        "",
        "| Dataset | File | Rows | Size | Years | Districts | Temporal | Spatial | Key dups |",
        "|---|---|---:|---:|---|---:|---|---|---:|",
    ]
    for d in report["datasets"]:
        if not d["present"]:
            out.append(f"| {d['name']} | `{d['path']}` | - | - | - | - | "
                       f"{d['temporal_resolution']} | {d['spatial_resolution']} | **MISSING** |")
            continue
        yr = d.get("year_range") or d.get("date_range")
        out.append(
            f"| {d['name']} | `{d['path']}` | {d['rows']:,} | {_fmt_size(d['size_bytes'])} | "
            f"{'-'.join(map(str, yr)) if yr else '-'} | {d.get('unique_districts', '-')} | "
            f"{d['temporal_resolution']} | {d['spatial_resolution']} | {d['duplicate_key_rows']} |"
        )

    out += ["", "## Missing values by column (only columns with missing values)", ""]
    any_missing = False
    for d in report["datasets"]:
        if d.get("missing_by_column"):
            any_missing = True
            cols = ", ".join(f"`{c}` {v['count']:,} ({v['percent']}%)"
                             for c, v in d["missing_by_column"].items())
            out.append(f"- **{d['name']}**: {cols}")
            out.append(f"  - Meaning: {d['missing_value_semantics']}")
    if not any_missing:
        out.append("None.")

    out += ["", "## Source, units and semantics", ""]
    for d in report["datasets"]:
        out.append(f"- **{d['name']}** - source: {d['source'] or 'n/a'}; units: {d['units'] or 'n/a'}; "
                   f"missing: {d['missing_value_semantics']}")
        if d.get("data_coverage_values"):
            out.append(f"  - data_coverage values: {d['data_coverage_values']}")
        if d.get("state_level_columns"):
            out.append(f"  - State-level values copied to district rows: {d['state_level_columns']}")
        if d.get("district_master_coverage"):
            out.append(f"  - District-master coverage: {d['district_master_coverage']}")

    out += ["", "## Data-quality annotations (source records are never edited)", ""]
    flags = report.get("dengue_quality_flags", [])
    if flags:
        out += ["| Source label | Year | Reported cases | Status |", "|---|---:|---:|---|"]
        out += [f"| {f['state_name_in_source']} | {f['year']} | {f['reported_dengue_cases']} | {f['status']} |"
                for f in flags]
        out += ["", "Source: `data/reference/dengue_data_quality_flags.csv`. The audit fails if a "
                "flagged value no longer matches the source."]
    else:
        out.append("None.")

    out += ["", "## Errors", ""]
    out += [f"- {e}" for e in report["errors"]] or ["None."]
    out += ["", "## Warnings (known limitations and findings)", ""]
    out += [f"- {w}" for w in report["warnings"]] or ["None."]
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(ROOT), help="Repository root.")
    parser.add_argument("--output", default=str(DEFAULT_REPORT), help="Markdown report path.")
    parser.add_argument("--json", dest="json_out", help="Also write the raw report as JSON.")
    parser.add_argument("--check", action="store_true", help="Do not write files.")
    args = parser.parse_args(argv)

    report = run_audit(args.root)
    markdown = render_markdown(report)
    if not args.check:
        Path(args.output).write_text(markdown, encoding="utf-8", newline="\n")
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(report['errors'])} error(s), "
          f"{len(report['warnings'])} warning(s)")
    for e in report["errors"]:
        print(f"ERROR: {e}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
