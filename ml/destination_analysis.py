"""Destination data-aggregation layer for Bharat TravelShield AI.

Combines the processed weather, heatwave-threshold, malaria, dengue and flood
tables into one JSON-serialisable summary for a single destination district.

Design rules (see ml/DATA_ANALYSIS_HANDOFF.md):
  * This is data aggregation, not forecasting. No overall risk score is built.
  * Every source here is annual (calendar-year) or coarser. A requested date
    range only selects the calendar years it overlaps; month-specific values
    are never derived from annual data.
  * A missing observation is reported as unknown (JSON null / absent), never
    as zero. Zero is reported only when the source records a zero.
  * Dengue is state-level context, not a district observation.
  * The heatwave ANN is optional and experimental; failures there never
    affect the rest of the summary.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"

SCHEMA_VERSION = "1.0"
ANN_MODEL_VERSION = "heatwave-ann-mlp-v1"

DISCLAIMER = (
    "Decision support only. This is not a diagnosis, an emergency alert, a "
    "guarantee of safety or a substitute for official advisories (IMD, "
    "NCVBDC/state health departments, state disaster-management authorities)."
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class DestinationAnalysisError(Exception):
    """Base class. `code` is stable and safe to expose in an API."""

    code = "error"

    def __init__(self, message: str, **extra: Any):
        super().__init__(message)
        self.message = message
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        body = {"code": self.code, "message": self.message}
        body.update(self.extra)
        return {"error": body}


class InvalidRequestError(DestinationAnalysisError):
    code = "invalid_request"


class DestinationNotFoundError(DestinationAnalysisError):
    code = "destination_not_found"


class AmbiguousDestinationError(DestinationAnalysisError):
    code = "ambiguous_destination"


class DataUnavailableError(DestinationAnalysisError):
    code = "data_unavailable"


# --------------------------------------------------------------------------
# Source registry
# --------------------------------------------------------------------------

# name -> (path relative to data dir, required columns, key columns)
SOURCES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "master": (
        "reference/district_master.csv",
        ("census_district_code", "district_name", "state_name",
         "latitude", "longitude", "population_2011"),
        ("census_district_code",),
    ),
    "weather": (
        "processed/weather/weather_district_year_2011_2025.csv",
        ("census_district_code", "year", "avg_temperature_c",
         "max_temperature_c", "min_temperature_c", "avg_humidity_pct",
         "total_rainfall_mm", "rainy_days"),
        ("census_district_code", "year"),
    ),
    "heatwave": (
        "processed/disaster/heatwave/heatwave_district_year_2011_2025.csv",
        ("census_district_code", "year", "heatwave_days",
         "severe_heatwave_days", "max_temperature_c"),
        ("census_district_code", "year"),
    ),
    "malaria": (
        "processed/disease/malaria/malaria_district_year_2000_2024.csv",
        ("census_district_code", "year", "total_cases", "total_deaths",
         "data_coverage"),
        ("census_district_code", "year"),
    ),
    "dengue": (
        "processed/disease/dengue/dengue_state_year_2021_2026.csv",
        ("state_name", "year", "dengue_cases", "dengue_deaths",
         "data_coverage"),
        ("state_name", "year"),
    ),
    "flood": (
        "processed/disaster/flood/flood_district_year_1967_2023.csv",
        ("census_district_code", "year", "flood_event_count"),
        ("census_district_code", "year"),
    ),
}

HAZARD_ORDER = ("weather", "heatwave_threshold", "malaria", "dengue", "flood")

# State names differ between sources (Census 2011 master vs. NCVBDC tables).
# Keys and values are already normalised (see _norm).
_STATE_ALIASES = {
    "andaman and nicobar island": "andaman and nicobar islands",
    "arunanchal pradesh": "arunachal pradesh",
    "dadara and nagar havelli": "dadra and nagar haveli",
    "nct of delhi": "delhi",
    "nct delhi": "delhi",
    "delhi nct": "delhi",
    "orissa": "odisha",
    "uttaranchal": "uttarakhand",
    "pondicherry": "puducherry",
}

# The Census-2011 master lists these ten districts under "Andhra Pradesh";
# the dengue source reports Telangana separately.
_TELANGANA_CENSUS_CODES = frozenset(range(532, 542))
_TELANGANA_NAMES = frozenset({
    "adilabad", "nizamabad", "karimnagar", "medak", "hyderabad",
    "rangareddy", "mahbubnagar", "nalgonda", "warangal", "khammam",
})
# Leh and Kargil (Census 2011, listed under Jammu & Kashmir) belong to
# Ladakh. Jammu and Kashmir figures are never assigned to them; only a
# separate Ladakh series can be used, and blank/dash Ladakh cells are
# "not reported", never zero.
_LADAKH_CENSUS_CODES = frozenset({3, 4})
_DENGUE_UNATTRIBUTABLE_CODES = _LADAKH_CENSUS_CODES  # kept for data_audit

# Cell contents in the dengue table that mean "not reported" (never zero).
NOT_REPORTED_TOKENS = ("-", "–", "—", "NR", "N/R", "n/a", "NA", "")

# Annotation file (does not alter source records).
DENGUE_FLAGS_FILE = "reference/dengue_data_quality_flags.csv"

VERIFICATION_PENDING = "verification_pending"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _norm(value: Any) -> str:
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm_state(value: Any) -> str:
    key = _norm(value)
    return _STATE_ALIASES.get(key, key)


def _py(value: Any) -> Any:
    """Convert pandas/numpy scalars to JSON types; NaN/inf/NA become None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
    return value


def _int_or_none(value: Any) -> int | None:
    value = _py(value)
    if value is None:
        return None
    return int(value)


def _records(df: pd.DataFrame, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for row in df.sort_values("year").to_dict("records"):
        item: dict[str, Any] = {"year": int(row["year"])}
        for col in columns:
            item[col] = _py(row[col])
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Data store
# --------------------------------------------------------------------------

class DataStore:
    """Loads each source once, validates its shape and caches it."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._frames: dict[str, pd.DataFrame | None] = {}
        self._problems: dict[str, str | None] = {}
        self._flags: dict[tuple[str, int], dict[str, Any]] | None = None

    def _load(self, name: str) -> None:
        rel, required, keys = SOURCES[name]
        path = self.data_dir / rel
        frame: pd.DataFrame | None = None
        problem: str | None = None

        if not path.is_file():
            problem = "source_file_missing"
        else:
            try:
                # Blank / dash cells in the dengue table mean "not reported".
                frame = pd.read_csv(
                    path,
                    na_values=list(NOT_REPORTED_TOKENS) if name == "dengue" else None,
                )
            except Exception:  # unreadable / empty / malformed CSV
                frame, problem = None, "source_file_unreadable"

        if frame is not None:
            missing = [c for c in required if c not in frame.columns]
            if missing:
                frame, problem = None, "source_schema_mismatch"

        if frame is not None:
            frame = frame.copy()
            if "census_district_code" in frame.columns:
                code = pd.to_numeric(frame["census_district_code"], errors="coerce")
                frame = frame[code.notna()].copy()
                frame["census_district_code"] = code[code.notna()].astype("int64")
            if "year" in frame.columns:
                year = pd.to_numeric(frame["year"], errors="coerce")
                frame = frame[year.notna()].copy()
                frame["year"] = year[year.notna()].astype("int64")
            if name == "dengue":
                frame["state_key"] = frame["state_name"].map(_norm_state)
                frame["source_state_label"] = frame["state_name"].astype(str)
                try:
                    for col in ("dengue_cases", "dengue_deaths"):
                        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
                except (TypeError, ValueError):  # non-integer counts
                    frame, problem = None, "source_schema_mismatch"
            if frame is not None and frame.duplicated(list(keys) if name != "dengue" else ["state_key", "year"]).any():
                frame, problem = None, "source_has_duplicate_keys"

        self._frames[name], self._problems[name] = frame, problem

    def table(self, name: str) -> pd.DataFrame | None:
        if name not in self._frames:
            self._load(name)
        return self._frames[name]

    def problem(self, name: str) -> str | None:
        self.table(name)
        return self._problems[name]

    def dengue_flags(self) -> dict[tuple[str, int], dict[str, Any]]:
        """Data-quality annotations keyed by (state_key, year).

        Read from data/reference/dengue_data_quality_flags.csv. The file only
        annotates source records; it never changes them. A missing or
        unreadable file means "no annotations".
        """
        if self._flags is None:
            self._flags = {}
            path = self.data_dir / DENGUE_FLAGS_FILE
            try:
                flags = pd.read_csv(path)
                for r in flags.to_dict("records"):
                    key = (_norm_state(r["state_name_in_source"]), int(r["year"]))
                    self._flags[key] = {
                        "status": str(r["status"]),
                        "source_state_label": str(r["state_name_in_source"]),
                        "reported_dengue_cases": _int_or_none(r["reported_dengue_cases"]),
                        "reason": str(r["reason"]),
                    }
            except Exception:  # absent/malformed annotation file
                self._flags = {}
        return self._flags

    def source_year_range(self, name: str) -> list[int] | None:
        frame = self.table(name)
        if frame is None or frame.empty:
            return None
        return [int(frame["year"].min()), int(frame["year"].max())]


_STORES: dict[str, DataStore] = {}


def get_store(data_dir: str | Path | None = None) -> DataStore:
    """Return a cached DataStore so large tables are not reloaded per call."""
    key = str(Path(data_dir).resolve() if data_dir else DEFAULT_DATA_DIR)
    if key not in _STORES:
        _STORES[key] = DataStore(data_dir)
    return _STORES[key]


def clear_store_cache() -> None:
    _STORES.clear()


# --------------------------------------------------------------------------
# Destination resolution
# --------------------------------------------------------------------------

def _clean_code(value: Any) -> int:
    if isinstance(value, bool):
        raise InvalidRequestError("district_code must be a whole number.")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int):
        raise InvalidRequestError("district_code must be a whole number.")
    return value


def resolve_destination(
    store: DataStore,
    district_code: Any = None,
    district_name: str | None = None,
    state_name: str | None = None,
) -> dict[str, Any]:
    """Resolve a destination to one canonical Census-2011 district."""
    master = store.table("master")
    if master is None:
        raise DataUnavailableError("The district reference data is unavailable.")

    if district_code is None and not district_name:
        raise InvalidRequestError(
            "Provide district_code, or district_name (with state_name if the "
            "name is not unique)."
        )
    for label, val in (("district_name", district_name), ("state_name", state_name)):
        if val is not None and (not isinstance(val, str) or not val.strip()):
            raise InvalidRequestError(f"{label} must be a non-empty string.")

    frame = master
    if district_code is not None:
        code = _clean_code(district_code)
        hits = frame[frame["census_district_code"] == code]
        if hits.empty:
            raise DestinationNotFoundError(f"No district with code {code}.")
        if district_name and _norm(district_name) not in {
            _norm(hits.iloc[0]["district_name"]),
            _norm(re.sub(r"\(.*?\)", "", hits.iloc[0]["district_name"])),
        }:
            raise InvalidRequestError(
                "district_name does not match the given district_code."
            )
    else:
        wanted = _norm(district_name)
        names = frame["district_name"].map(_norm)
        base = frame["district_name"].map(
            lambda n: _norm(re.sub(r"\(.*?\)", "", n))
        )
        hits = frame[(names == wanted) | (base == wanted)]
        if state_name:
            hits = hits[hits["state_name"].map(_norm_state) == _norm_state(state_name)]

        if hits.empty:
            pool = sorted(set(names))
            close = difflib.get_close_matches(wanted, pool, n=3, cutoff=0.75)
            suggestions = sorted(
                frame.loc[names.isin(close), "district_name"].unique().tolist()
            )
            raise DestinationNotFoundError(
                "No matching district found.", suggestions=suggestions
            )
        if len(hits) > 1:
            candidates = [
                {
                    "census_district_code": int(r.census_district_code),
                    "district_name": r.district_name,
                    "state_name": r.state_name,
                }
                for r in hits.sort_values("census_district_code").itertuples()
            ]
            raise AmbiguousDestinationError(
                "The district name matches more than one district; "
                "pass state_name or district_code.",
                candidates=candidates,
            )

    row = hits.iloc[0]
    return {
        "census_district_code": int(row["census_district_code"]),
        "district_name": str(row["district_name"]),
        "state_name": str(row["state_name"]),
        "latitude": _py(row["latitude"]),
        "longitude": _py(row["longitude"]),
        "population_2011": _int_or_none(row["population_2011"]),
        "boundary_basis": "Census 2011 district",
        "coordinate_basis": "district representative point (not a weather station)",
    }


# --------------------------------------------------------------------------
# Query period
# --------------------------------------------------------------------------

def _parse_date(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            pass
    raise InvalidRequestError(f"{label} must be a valid date in YYYY-MM-DD format.")


def parse_period(start_date: Any = None, end_date: Any = None) -> dict[str, Any]:
    """Validate the optional travel dates and describe how they are used."""
    if start_date is None and end_date is None:
        return {
            "start_date": None,
            "end_date": None,
            "requested": False,
            "years_covered": None,
            "spans_partial_year": None,
            "resolution_supported": "annual",
            "month_specific_data_available": False,
            "note": (
                "No dates given: all available calendar years are returned. "
                "All sources are annual or coarser."
            ),
        }
    if start_date is None or end_date is None:
        raise InvalidRequestError("Provide both start_date and end_date, or neither.")

    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if end < start:
        raise InvalidRequestError("end_date cannot be before start_date.")
    if start.year < 1900 or end.year > 2200:
        raise InvalidRequestError("Dates are outside the supported range (1900-2200).")

    whole_years = (
        start.month == 1 and start.day == 1 and end.month == 12 and end.day == 31
    )
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "requested": True,
        "years_covered": list(range(start.year, end.year + 1)),
        "spans_partial_year": not whole_years,
        "resolution_supported": "annual",
        "month_specific_data_available": False,
        "note": (
            "The requested dates select the calendar years they overlap. "
            "Every value is an annual (calendar-year) figure and is not "
            "restricted to the requested days or months; month-specific "
            "values are not available."
        ),
    }


# --------------------------------------------------------------------------
# Hazard builders
# --------------------------------------------------------------------------

def _empty_hazard(status: str, reason: str, **fixed: Any) -> dict[str, Any]:
    hazard = dict(fixed)
    hazard.update({
        "status": status,
        "status_reason": reason,
        "records": [],
        "summary": None,
        "coverage": {
            "source_year_range": None,
            "destination_year_range": None,
            "years_requested": None,
            "years_with_records_in_period": [],
            "years_without_records_in_period": None,
        },
        "last_observed_year": None,
    })
    return hazard


def _assemble(
    fixed: dict[str, Any],
    source_range: list[int] | None,
    dest_all: pd.DataFrame,
    in_period: pd.DataFrame,
    columns: tuple[str, ...],
    years: list[int] | None,
    track_missing_years: bool,
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    dest_years = sorted(int(y) for y in dest_all["year"].unique())
    period_years = sorted(int(y) for y in in_period["year"].unique())

    if dest_all.empty:
        status, reason = "no_records_for_destination", (
            "The source holds no retained record for this destination. "
            "This is unknown, not zero."
        )
    elif in_period.empty:
        status, reason = "no_records_in_period", (
            "The destination has records in other years but none in the "
            "requested period. This is unknown, not zero."
        )
    elif years is not None and track_missing_years and set(years) - set(period_years):
        status, reason = "partial", (
            "Some requested years have no record. They are unknown, not zero."
        )
    else:
        status, reason = "available", None

    hazard = dict(fixed)
    hazard.update({
        "status": status,
        "status_reason": reason,
        "records": _records(in_period, columns),
        "summary": summary if not in_period.empty else None,
        "coverage": {
            "source_year_range": source_range,
            "destination_year_range": (
                [dest_years[0], dest_years[-1]] if dest_years else None
            ),
            "years_requested": years,
            "years_with_records_in_period": period_years,
            "years_without_records_in_period": (
                sorted(set(years) - set(period_years))
                if years is not None and track_missing_years else None
            ),
        },
        "last_observed_year": dest_years[-1] if dest_years else None,
    })
    return hazard


def _select(frame: pd.DataFrame, mask: pd.Series, years: list[int] | None):
    dest_all = frame[mask]
    in_period = dest_all if years is None else dest_all[dest_all["year"].isin(years)]
    return dest_all, in_period


def _unavailable(store: DataStore, name: str, **fixed: Any) -> dict[str, Any]:
    return _empty_hazard(
        "source_unavailable",
        f"{store.problem(name)}: this hazard is excluded; others are unaffected.",
        **fixed,
    )


def _weather(store: DataStore, dest: dict, years) -> dict[str, Any]:
    cols = ("avg_temperature_c", "max_temperature_c", "min_temperature_c",
            "avg_humidity_pct", "total_rainfall_mm", "rainy_days")
    fixed = {
        "resolution": {"spatial": "district representative point", "temporal": "annual"},
        "spatial_basis": "district_point_model_derived",
        "source": "NASA POWER daily point data (MERRA-2/POWER), aggregated by year",
        "units": {
            "avg_temperature_c": "degC, annual mean of daily 2 m temperature",
            "max_temperature_c": "degC, highest daily maximum in the year",
            "min_temperature_c": "degC, lowest daily minimum in the year",
            "avg_humidity_pct": "%, annual mean 2 m relative humidity",
            "total_rainfall_mm": "mm, annual corrected precipitation",
            "rainy_days": "days with precipitation >= 1 mm",
        },
        "caveats": [
            "Gridded, model-derived values at the district representative "
            "point, not station observations.",
            "Annual figures only; monthly or travel-window weather is not "
            "provided by this table.",
        ],
    }
    frame = store.table("weather")
    if frame is None:
        return _unavailable(store, "weather", **fixed)
    dest_all, in_period = _select(
        frame, frame["census_district_code"] == dest["census_district_code"], years
    )
    summary = None
    if not in_period.empty:
        top = in_period.loc[in_period["max_temperature_c"].idxmax()]
        summary = {
            "years_with_records": int(len(in_period)),
            "mean_of_annual_avg_temperature_c": round(float(in_period["avg_temperature_c"].mean()), 2),
            "highest_daily_max_temperature_c": float(top["max_temperature_c"]),
            "highest_daily_max_year": int(top["year"]),
            "mean_annual_rainfall_mm": round(float(in_period["total_rainfall_mm"].mean()), 2),
        }
    return _assemble(fixed, store.source_year_range("weather"), dest_all,
                     in_period, cols, years, True, summary)


def _heatwave(store: DataStore, dest: dict, years) -> dict[str, Any]:
    cols = ("heatwave_days", "severe_heatwave_days", "max_temperature_c")
    fixed = {
        "resolution": {"spatial": "district representative point", "temporal": "annual"},
        "spatial_basis": "district_point_model_derived",
        "source": "Derived from NASA POWER daily T2M_MAX",
        "definition": (
            "heatwave_days: days with modelled daily maximum temperature "
            ">= 45 degC; severe_heatwave_days: >= 47 degC."
        ),
        "units": {
            "heatwave_days": "days", "severe_heatwave_days": "days",
            "max_temperature_c": "degC",
        },
        "caveats": [
            "A temperature-threshold indicator from gridded data. It is NOT "
            "an official IMD-declared heatwave record (IMD declarations also "
            "use departure-from-normal and spatial/temporal rules).",
        ],
    }
    frame = store.table("heatwave")
    if frame is None:
        return _unavailable(store, "heatwave", **fixed)
    dest_all, in_period = _select(
        frame, frame["census_district_code"] == dest["census_district_code"], years
    )
    summary = None
    if not in_period.empty:
        top = in_period.loc[in_period["heatwave_days"].idxmax()]
        summary = {
            "years_with_records": int(len(in_period)),
            "years_with_threshold_days": int((in_period["heatwave_days"] > 0).sum()),
            "total_threshold_days": int(in_period["heatwave_days"].sum()),
            "total_severe_threshold_days": int(in_period["severe_heatwave_days"].sum()),
            "most_threshold_days_in_a_year": int(top["heatwave_days"]),
            "most_threshold_days_year": int(top["year"]),
        }
    return _assemble(fixed, store.source_year_range("heatwave"), dest_all,
                     in_period, cols, years, True, summary)


def _malaria(store: DataStore, dest: dict, years) -> dict[str, Any]:
    cols = ("total_cases", "total_deaths", "data_coverage")
    fixed = {
        "resolution": {"spatial": "district", "temporal": "annual"},
        "spatial_basis": "district_observation",
        "source": "NCVBDC district-wise malaria tables (extracted from PDF)",
        "units": {"total_cases": "reported cases", "total_deaths": "reported deaths"},
        "caveats": [
            "Raw reported counts. Not rates and not comparable across "
            "districts of different population; no incidence is computed.",
            "Only districts successfully matched to the Census-2011 master "
            "are retained; a missing district-year is unknown, not zero.",
            "A row with data_coverage THROUGH_APRIL_2024 is a partial year.",
        ],
    }
    frame = store.table("malaria")
    if frame is None:
        return _unavailable(store, "malaria", **fixed)
    dest_all, in_period = _select(
        frame, frame["census_district_code"] == dest["census_district_code"], years
    )
    summary = None
    if not in_period.empty:
        summary = {
            "years_with_records": int(len(in_period)),
            "sum_of_reported_cases": int(in_period["total_cases"].sum()),
            "years_with_nonzero_cases": int((in_period["total_cases"] > 0).sum()),
            "includes_partial_year_records": bool(
                (in_period["data_coverage"] != "FULL_YEAR").any()
            ),
        }
    return _assemble(fixed, store.source_year_range("malaria"), dest_all,
                     in_period, cols, years, True, summary)


def _dengue_state_key(dest: dict, available: set[str]) -> tuple[str | None, str]:
    code = dest["census_district_code"]
    if code in _LADAKH_CENSUS_CODES:
        # Never fall back to Jammu and Kashmir.
        if "ladakh" in available:
            return "ladakh", "ladakh_series"
        return None, "no_separate_ladakh_series"
    if code in _TELANGANA_CENSUS_CODES and _norm(dest["district_name"]) in _TELANGANA_NAMES:
        return "telangana", "telangana_district_list"
    return _norm_state(dest["state_name"]), "state_name_match"


def _annotate_dengue_records(records, state_key, flags):
    """Add reporting status and data-quality status to each dengue record.

    Source values and labels are copied unchanged; flags only annotate.
    """
    for rec in records:
        flag = flags.get((state_key, rec["year"]))
        cases_missing = rec["dengue_cases"] is None
        deaths_missing = rec["dengue_deaths"] is None
        rec["reporting_status"] = (
            "reported" if not (cases_missing and deaths_missing)
            else "not_reported"
        )
        if flag:
            rec["data_quality"] = {
                "status": flag["status"],
                "usable_as_verified_input": False,
                "reason": flag["reason"],
                "flagged_source_label": flag["source_state_label"],
                "flagged_reported_cases": flag["reported_dengue_cases"],
                "flag_matches_source_value": (
                    flag["reported_dengue_cases"] == rec["dengue_cases"]
                ),
            }
        elif rec["reporting_status"] == "not_reported":
            rec["data_quality"] = {
                "status": "not_reported",
                "usable_as_verified_input": False,
                "reason": "Blank or dash in the source; unknown, never zero.",
            }
        else:
            rec["data_quality"] = {
                "status": "as_reported",
                "usable_as_verified_input": True,
            }
    return records


def _dengue(store: DataStore, dest: dict, years) -> dict[str, Any]:
    cols = ("dengue_cases", "dengue_deaths", "data_coverage", "source_state_label")
    fixed = {
        "resolution": {"spatial": "state", "temporal": "annual"},
        "spatial_basis": "state_level_context_not_district_specific",
        "source": "NCVBDC state-wise dengue tables",
        "units": {"dengue_cases": "reported cases", "dengue_deaths": "reported deaths"},
        "state_used": None,
        "data_quality": {"verification_pending_years": [], "note": None},
        "caveats": [
            "State-level totals shown as context for the destination's "
            "state. They are NOT observations for this district and are "
            "shared by every district of the state.",
            "Raw counts; not rates and not comparable across states of "
            "different population.",
            "Rows marked PROVISIONAL_* cover only part of the year.",
        ],
    }
    frame = store.table("dengue")
    if frame is None:
        return _unavailable(store, "dengue", **fixed)

    key, basis = _dengue_state_key(dest, set(frame["state_key"]))
    if key is None:
        return _empty_hazard(
            "no_records_for_destination",
            "The source has no separate Ladakh series. Jammu and Kashmir "
            "figures are not assigned to Leh or Kargil. Not reported: "
            "unknown, not zero.",
            **fixed,
        )
    dest_all, in_period = _select(frame, frame["state_key"] == key, years)
    labels = sorted(dest_all["source_state_label"].unique()) if not dest_all.empty else []
    fixed["state_used"] = {
        "state_in_source": labels[0] if len(labels) == 1 else None,
        "mapping": basis,
    }
    if basis == "telangana_district_list":
        fixed["caveats"].append(
            "Census 2011 lists this district under Andhra Pradesh; it is "
            "mapped to Telangana's own separately reported state-level series."
        )
    elif basis == "ladakh_series":
        fixed["caveats"].append(
            "Ladakh's own state-level series is used; blank or dash cells "
            "are 'not_reported', never zero."
        )
    elif dest["census_district_code"] not in _LADAKH_CENSUS_CODES and key == "jammu and kashmir":
        fixed["caveats"].append(
            "Jammu and Kashmir state-level figures; not assigned to Leh or Kargil."
        )

    summary = None
    if not in_period.empty:
        reported = in_period["dengue_cases"].dropna()
        summary = {
            "years_with_records": int(len(in_period)),
            "years_not_reported": int(in_period["dengue_cases"].isna().sum()),
            "sum_of_reported_state_cases": int(reported.sum()) if len(reported) else None,
            "includes_provisional_records": bool(
                (in_period["data_coverage"] != "FULL_YEAR").any()
            ),
        }
    hazard = _assemble(fixed, store.source_year_range("dengue"), dest_all,
                       in_period, cols, years, True, summary)

    flags = store.dengue_flags()
    _annotate_dengue_records(hazard["records"], key, flags)
    pending = [r["year"] for r in hazard["records"]
               if r["data_quality"]["status"] == VERIFICATION_PENDING]
    hazard["data_quality"] = {
        "verification_pending_years": pending,
        "note": (
            "Records with verification_pending keep their source value and "
            "label unchanged but are not verified input; do not use them in "
            "any scoring or ranking until the data owner confirms them."
            if pending else None
        ),
    }
    if hazard["summary"] is not None:
        hazard["summary"]["includes_verification_pending_records"] = bool(pending)
    if hazard["records"] and all(r["reporting_status"] == "not_reported" for r in hazard["records"]):
        hazard["status"] = "not_reported"
        hazard["status_reason"] = (
            "The source row(s) are blank or dash. Not reported: unknown, not zero."
        )
    return hazard


def _flood(store: DataStore, dest: dict, years) -> dict[str, Any]:
    cols = ("flood_event_count",)
    fixed = {
        "resolution": {"spatial": "district", "temporal": "annual"},
        "spatial_basis": "district_observation",
        "source": "India Flood Inventory v3, deterministic LGD-to-Census mapping",
        "units": {"flood_event_count": "recorded flood events in the year"},
        "absence_semantics": "unknown_not_zero",
        "caveats": [
            "The table lists only district-years with at least one recorded "
            "event. A year with no row is NOT confirmed to have had no flood.",
            "Only districts with a deterministic LGD-to-Census mapping are "
            "present (502 of 640 in the current build); other districts are "
            "unknown, not flood-free.",
        ],
    }
    frame = store.table("flood")
    if frame is None:
        return _unavailable(store, "flood", **fixed)
    dest_all, in_period = _select(
        frame, frame["census_district_code"] == dest["census_district_code"], years
    )
    summary = None
    if not in_period.empty:
        summary = {
            "years_with_recorded_events": int(len(in_period)),
            "total_recorded_events": int(in_period["flood_event_count"].sum()),
        }
    return _assemble(fixed, store.source_year_range("flood"), dest_all,
                     in_period, cols, years, False, summary)


# --------------------------------------------------------------------------
# Precautions
# --------------------------------------------------------------------------

_GUIDANCE_KIND = "generic_guidance_not_official_advice"


def _precautions(hazards: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    """Cautious generic precautions, listed only where records support them."""
    out = []

    def has(name: str, test) -> bool:
        return any(test(r) for r in hazards[name]["records"])

    if has("heatwave_threshold", lambda r: (r["heatwave_days"] or 0) > 0):
        out.append({
            "hazard": "heatwave_threshold",
            "text": (
                "Plan strenuous outdoor activity outside the hottest hours, "
                "drink water regularly and know the signs of heat illness. "
                "Check current IMD heat advisories before travel."
            ),
            "supported_by": "Modelled days at or above 45 degC appear in the returned years.",
        })
    if has("malaria", lambda r: (r["total_cases"] or 0) > 0):
        out.append({
            "hazard": "malaria",
            "text": (
                "Reduce mosquito bites (repellent, covered clothing, nets where "
                "advised) and seek medical care promptly for fever. Ask a "
                "health professional about travel-health advice."
            ),
            "supported_by": "Reported malaria cases appear for this district in the returned years.",
        })
    if has("dengue", lambda r: (r["dengue_cases"] or 0) > 0
           and r["data_quality"]["usable_as_verified_input"]):
        out.append({
            "hazard": "dengue",
            "text": (
                "Dengue mosquitoes bite mainly by day: use repellent and "
                "covered clothing, avoid standing water, and seek medical care "
                "for high fever or severe symptoms."
            ),
            "supported_by": "Reported dengue cases appear for the destination's state in the returned years.",
        })
    if has("flood", lambda r: (r["flood_event_count"] or 0) > 0):
        out.append({
            "hazard": "flood",
            "text": (
                "Check state and district disaster-management advisories "
                "and forecasts before travel, especially in the monsoon, and "
                "avoid flooded roads."
            ),
            "supported_by": "Recorded flood events appear for this district in the returned years.",
        })
    for item in out:
        item["kind"] = _GUIDANCE_KIND
    return out


# --------------------------------------------------------------------------
# Optional experimental heatwave ANN
# --------------------------------------------------------------------------

_ANN_CAVEATS = [
    "Experimental. Target: at least one NASA POWER-derived day >= 45 degC "
    "during the target calendar year (not an IMD-declared heatwave).",
    "District-level annual classification; not a travel-date or month forecast.",
    "The raw score is uncalibrated and must not be read as a probability or "
    "converted to a risk level.",
]


def _ann_block(store: DataStore, dest: dict, feature_year: int | None) -> dict[str, Any]:
    base = {
        "experimental": True,
        "model_version": ANN_MODEL_VERSION,
        "caveats": list(_ANN_CAVEATS),
        "feature_year": None,
        "target_year": None,
        "prediction": None,
    }
    weather, heat = store.table("weather"), store.table("heatwave")
    if weather is None or heat is None:
        return {**base, "status": "unavailable",
                "reason": "Weather or heatwave source data is unavailable."}

    code = dest["census_district_code"]
    w = weather[weather["census_district_code"] == code]
    h = heat[heat["census_district_code"] == code]
    common = sorted(set(w["year"]) & set(h["year"]))
    if feature_year is None:
        if not common:
            return {**base, "status": "unavailable",
                    "reason": "No year has both weather and heatwave records."}
        feature_year = int(common[-1])
    elif isinstance(feature_year, bool) or not isinstance(feature_year, int):
        return {**base, "status": "unavailable",
                "reason": "ann_feature_year must be a whole number."}
    elif feature_year not in common:
        return {**base, "status": "unavailable", "feature_year": feature_year,
                "target_year": feature_year + 1,
                "reason": "No complete feature record exists for that year."}

    base.update(feature_year=feature_year, target_year=feature_year + 1)
    wr = w[w["year"] == feature_year].iloc[0]
    hr = h[h["year"] == feature_year].iloc[0]
    try:
        total_rain = float(wr["total_rainfall_mm"])
        rainy = float(wr["rainy_days"])
        features = {
            "latitude": float(dest["latitude"]),
            "longitude": float(dest["longitude"]),
            "population_2011": float(dest["population_2011"]),
            "avg_temperature_c": float(wr["avg_temperature_c"]),
            "max_temperature_c": float(wr["max_temperature_c"]),
            "min_temperature_c": float(wr["min_temperature_c"]),
            "avg_humidity_pct": float(wr["avg_humidity_pct"]),
            "total_rainfall_mm": total_rain,
            "rainy_days": rainy,
            "heatwave_days_T": float(hr["heatwave_days"]),
            "severe_heatwave_days_T": float(hr["severe_heatwave_days"]),
            "has_heatwave_T": float(hr["heatwave_days"] > 0),
            "temp_range_c": round(float(wr["max_temperature_c"]) - float(wr["min_temperature_c"]), 2),
            "rain_intensity": round(total_rain / (rainy + 1), 4),
        }
        from ml.heatwave_inference import predict_heatwave  # optional dependency

        prediction = predict_heatwave(features)
    except FileNotFoundError:
        return {**base, "status": "unavailable",
                "reason": "ANN model artifacts are not installed."}
    except ImportError:
        return {**base, "status": "unavailable",
                "reason": "Optional ML dependencies are not installed."}
    except (ValueError, TypeError) as exc:
        return {**base, "status": "unavailable",
                "reason": f"Feature record failed validation: {exc}"}
    except Exception:
        return {**base, "status": "unavailable",
                "reason": "The experimental model could not be run."}

    return {**base, "status": "available", "reason": None,
            "prediction": prediction}


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

_LIMITATIONS = [
    "Aggregation of existing data, not a forecast; no overall risk score or "
    "safety verdict is produced.",
    "All sources are annual or coarser. Month- or day-specific values are "
    "not available and are never derived from annual data.",
    "A missing record means unknown. It is never treated as zero.",
    "Disease counts are raw reported counts, not rates; do not compare them "
    "across places of different population.",
    "Dengue is state-level context shared by all districts of the state.",
    "Correlation shown by co-occurring data is not causation.",
]


def build_destination_summary(
    district_code: Any = None,
    district_name: str | None = None,
    state_name: str | None = None,
    start_date: Any = None,
    end_date: Any = None,
    include_experimental_ann: bool = False,
    ann_feature_year: int | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return one JSON-serialisable summary for a destination district.

    Raises DestinationAnalysisError subclasses for invalid input or when the
    district reference data is unavailable. A missing hazard source never
    raises; that hazard is marked `source_unavailable`.
    """
    store = get_store(data_dir)
    period = parse_period(start_date, end_date)
    dest = resolve_destination(store, district_code, district_name, state_name)
    years = period["years_covered"]

    hazards = {
        "weather": _weather(store, dest, years),
        "heatwave_threshold": _heatwave(store, dest, years),
        "malaria": _malaria(store, dest, years),
        "dengue": _dengue(store, dest, years),
        "flood": _flood(store, dest, years),
    }
    hazards = {name: hazards[name] for name in HAZARD_ORDER}

    if include_experimental_ann:
        ann = _ann_block(store, dest, ann_feature_year)
    else:
        ann = {"experimental": True, "status": "not_requested",
               "model_version": ANN_MODEL_VERSION}

    return {
        "schema_version": SCHEMA_VERSION,
        "summary_type": "destination_data_summary",
        "destination": dest,
        "query_period": period,
        "hazards": hazards,
        "coverage_overview": {
            name: {
                "status": h["status"],
                "last_observed_year": h["last_observed_year"],
                "spatial_basis": h.get("spatial_basis"),
            }
            for name, h in hazards.items()
        },
        "precautions": _precautions(hazards),
        "experimental_heatwave_ann": ann,
        "limitations": list(_LIMITATIONS),
        "disclaimer": DISCLAIMER,
    }


def safe_destination_summary(**kwargs: Any) -> dict[str, Any]:
    """Like build_destination_summary but returns {"error": {...}} on failure.

    Convenient for an API layer that wants one dict either way.
    """
    try:
        return build_destination_summary(**kwargs)
    except DestinationAnalysisError as exc:
        return exc.to_dict()


def main(argv: list[str] | None = None) -> int:
    """CLI: print a destination summary (or a JSON error) to stdout."""
    import argparse
    import json

    p = argparse.ArgumentParser(description="Print a destination data summary as JSON.")
    p.add_argument("--district-code", type=int)
    p.add_argument("--district-name")
    p.add_argument("--state-name")
    p.add_argument("--start-date", help="YYYY-MM-DD")
    p.add_argument("--end-date", help="YYYY-MM-DD")
    p.add_argument("--with-ann", action="store_true",
                   help="Also run the experimental heatwave ANN (optional).")
    p.add_argument("--ann-feature-year", type=int)
    p.add_argument("--data-dir")
    p.add_argument("--output", help="Write JSON here instead of stdout.")
    a = p.parse_args(argv)

    result = safe_destination_summary(
        district_code=a.district_code, district_name=a.district_name,
        state_name=a.state_name, start_date=a.start_date, end_date=a.end_date,
        include_experimental_ann=a.with_ann, ann_feature_year=a.ann_feature_year,
        data_dir=a.data_dir,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if a.output:
        Path(a.output).write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text, end="")
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
