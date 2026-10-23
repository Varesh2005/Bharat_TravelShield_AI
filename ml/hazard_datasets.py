"""Model datasets for the hazard ANNs (heatwave, flood, malaria) + readiness evidence.

One row = one district and one *feature year* T. Features use only information
available at the end of year T. The target describes year T+1.

    python -m ml.hazard_datasets --build      # write ml/data/hazards/<hazard>/...
    python -m ml.hazard_datasets --check      # rebuild in memory, compare with files
    python -m ml.hazard_datasets --readiness  # print which hazards can be modelled

Rules (see ml/ANN_DESIGN_DECISION.md):
  * Rows with any missing required input are dropped and counted, never imputed.
  * Splits are chronological by TARGET year; no random row split.
  * Everything is annual, so predictions are "next calendar year", never a
    travel-month forecast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "ml" / "data" / "hazards"

# Chronological split by target year, identical for every hazard.
TRAIN_TARGET_MAX = 2019
VAL_TARGET_YEARS = (2020, 2021)
TEST_TARGET_MIN = 2022

# A calibrated probability is displayed only if the forward calibration check on the VALIDATION
# years (Platt fitted on earlier validation years, ECE measured on the last validation year)
# is <= this value. Test years never take part in this decision. The 0.10 cut-off is a conventional
# engineering value adopted after the first exploratory run (see ml/ANN_DESIGN_DECISION.md section 8).
ECE_ACCEPTABLE = 0.10

# Readiness criteria (stated before looking at any model result).
MIN_ROWS = {"train": 300, "val": 100, "test": 100}
MIN_POSITIVES_PER_SPLIT = 20

HAZARDS = ("heatwave", "flood", "malaria")


def split_of(target_year: int) -> str:
    if target_year <= TRAIN_TARGET_MAX:
        return "train"
    if VAL_TARGET_YEARS[0] <= target_year <= VAL_TARGET_YEARS[1]:
        return "val"
    return "test"


GEOGRAPHY = ["latitude", "longitude", "log_population"]
WEATHER = ["avg_temperature_c", "max_temperature_c", "min_temperature_c",
           "temp_range_c", "avg_humidity_pct", "total_rainfall_mm", "rainy_days"]


@dataclass(frozen=True)
class HazardSpec:
    name: str
    title: str
    target: str
    target_definition: str
    geography: str
    horizon: str
    label_source: str
    label_reliability: str
    groups: dict[str, list[str]]
    persistence_feature: str          # 0/1 feature used by the persistence baseline
    feature_years: tuple[int, int]
    limitations: tuple[str, ...]

    @property
    def features(self) -> list[str]:
        return [f for g in self.groups.values() for f in g]


SPECS: dict[str, HazardSpec] = {
    "heatwave": HazardSpec(
        name="heatwave",
        title="Heatwave-threshold occurrence",
        target="target_heat_next_year",
        target_definition=("1 if the district has at least one day with modelled "
                           "maximum temperature >= 45 degC in year T+1, else 0."),
        geography="district (Census 2011), representative point",
        horizon="next calendar year (annual)",
        label_source="NASA POWER daily T2M_MAX thresholded at 45 degC (district year table)",
        label_reliability=("Complete and consistent for all 640 districts, but gridded "
                           "model data thresholded by us: NOT an IMD-declared heatwave."),
        groups={"geography": GEOGRAPHY,
                "weather_T": WEATHER,
                "hazard_history": ["heatwave_days_T", "severe_heatwave_days_T", "heatwave_days_Tm1"]},
        persistence_feature="has_heatwave_T",
        feature_years=(2012, 2024),
        limitations=("Annual district target; not a travel-date or month forecast.",
                     "Label is a temperature-threshold indicator, not an official event.",
                     "2021 is an anomalous year (4.7% positive nationally)."),
    ),
    "flood": HazardSpec(
        name="flood",
        title="Recorded flood occurrence",
        target="target_flood_next_year",
        target_definition=("1 if the India Flood Inventory records at least one flood "
                           "event for the district in year T+1, else 0 (= none recorded)."),
        geography="district (Census 2011), 502 districts with a deterministic LGD mapping",
        horizon="next calendar year (annual)",
        label_source="India Flood Inventory v3 via deterministic LGD-to-Census mapping",
        label_reliability=("PRESENCE-ONLY: the inventory lists events; a year with no "
                           "record is treated as 'none recorded' for this target, which is "
                           "not proof that no flood happened. Districts with no recorded "
                           "event at all are indistinguishable from unmapped ones and are "
                           "excluded, which may overstate flood-proneness."),
        groups={"geography": GEOGRAPHY,
                "weather_T": WEATHER[4:] + ["avg_temperature_c", "max_temperature_c"],
                "hazard_history": ["recorded_flood_events_T", "recorded_flood_events_Tm1",
                                   "flood_years_last5", "flood_years_last10"]},
        persistence_feature="flood_recorded_T",
        feature_years=(2011, 2022),
        limitations=("Negatives mean 'no record', not 'confirmed no flood'.",
                     "Only 502 of 640 districts; unmapped districts have no flood prediction.",
                     "Annual district target; no flash-flood or event-timing information."),
    ),
    "malaria": HazardSpec(
        name="malaria",
        title="High reported malaria incidence",
        target="target_malaria_high_next_year",
        target_definition=("1 if the district's reported malaria cases per 100,000 "
                           "(2011 population) in year T+1 are in the top quartile of all "
                           "reporting districts that year, else 0."),
        geography="district (Census 2011), 517 districts that appear in the NCVBDC tables",
        horizon="next calendar year (annual)",
        label_source="NCVBDC district-wise malaria tables (PDF extraction), FULL_YEAR rows only",
        label_reliability=("Reported counts from PDF extraction; relative ranking within a "
                           "year because national totals fell from 1.07M (2011) to 0.12M "
                           "(2021). Population is the 2011 census for every year."),
        groups={"geography": GEOGRAPHY,
                "weather_T": WEATHER,
                "hazard_history": ["log1p_incidence_T", "log1p_incidence_Tm1",
                                   "incidence_rank_pct_T", "high_burden_T"]},
        persistence_feature="high_burden_T",
        feature_years=(2012, 2022),
        limitations=("Relative burden among reporting districts, not an absolute risk.",
                     "Not corrected for under-reporting or population change since 2011.",
                     "2024 is a partial year and is never used."),
    ),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _csv(data_dir: Path, rel: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / rel)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SOURCE_FILES = {
    "master": "reference/district_master.csv",
    "weather": "processed/weather/weather_district_year_2011_2025.csv",
    "heatwave": "processed/disaster/heatwave/heatwave_district_year_2011_2025.csv",
    "flood": "processed/disaster/flood/flood_district_year_1967_2023.csv",
    "malaria": "processed/disease/malaria/malaria_district_year_2000_2024.csv",
    "dengue": "processed/disease/dengue/dengue_state_year_2021_2026.csv",
}


def _static(data_dir: Path) -> pd.DataFrame:
    m = _csv(data_dir, SOURCE_FILES["master"])
    m = m[["census_district_code", "state_name", "latitude", "longitude", "population_2011"]].copy()
    m["log_population"] = np.log1p(m["population_2011"].astype(float))
    return m.drop(columns="population_2011")


def _weather(data_dir: Path) -> pd.DataFrame:
    w = _csv(data_dir, SOURCE_FILES["weather"]).copy()
    w["temp_range_c"] = (w["max_temperature_c"] - w["min_temperature_c"]).round(2)
    return w


def _finish(frame: pd.DataFrame, spec: HazardSpec, exclusions: dict[str, int],
            require_target: bool = True) -> tuple[pd.DataFrame, dict]:
    """Drop incomplete rows (counted), order deterministically, add split.

    With require_target=False (inference) the newest feature years are kept
    and the target stays NaN because year T+1 has not been observed.
    """
    cols = ["census_district_code", "state_name", "feature_year", "target_year",
            *spec.features, spec.persistence_feature, spec.target]
    cols = list(dict.fromkeys(cols))
    frame = frame[cols]
    need = [*spec.features, spec.persistence_feature] + ([spec.target] if require_target else [])
    incomplete = frame[need].isna().any(axis=1)
    exclusions = dict(exclusions)
    exclusions["rows_missing_required_input_dropped"] = int(incomplete.sum())
    frame = frame[~incomplete].copy()
    frame["split"] = frame["target_year"].map(split_of)
    frame = frame.sort_values(["feature_year", "census_district_code"]).reset_index(drop=True)
    ints = ["census_district_code", "feature_year", "target_year", spec.persistence_feature]
    ints += [spec.target] if require_target else []
    for c in ints:
        frame[c] = frame[c].astype("int64")
    return frame, exclusions


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def build_heatwave(data_dir: Path, inference: bool = False) -> tuple[pd.DataFrame, dict]:
    spec = SPECS["heatwave"]
    w, h, st = _weather(data_dir), _csv(data_dir, SOURCE_FILES["heatwave"]), _static(data_dir)
    h = h[["census_district_code", "year", "heatwave_days", "severe_heatwave_days"]]
    cur = w.merge(h, on=["census_district_code", "year"], how="inner")
    cur = cur.rename(columns={"year": "feature_year", "heatwave_days": "heatwave_days_T",
                              "severe_heatwave_days": "severe_heatwave_days_T"})
    prev = h.assign(feature_year=h["year"] + 1)[["census_district_code", "feature_year", "heatwave_days"]]
    prev = prev.rename(columns={"heatwave_days": "heatwave_days_Tm1"})
    nxt = h.assign(feature_year=h["year"] - 1)[["census_district_code", "feature_year", "heatwave_days"]]
    nxt = nxt.rename(columns={"heatwave_days": "target_days"})
    f = cur.merge(prev, on=["census_district_code", "feature_year"], how="left")
    f = f.merge(nxt, on=["census_district_code", "feature_year"], how="left")
    f = f.merge(st, on="census_district_code", how="left")
    lo, hi = spec.feature_years
    hi = int(w["year"].max()) if inference else hi
    f = f[f["feature_year"].between(lo, hi)].copy()
    f["target_year"] = f["feature_year"] + 1
    f["has_heatwave_T"] = (f["heatwave_days_T"] > 0).astype(int)
    f[spec.target] = np.where(f["target_days"].isna(), np.nan, (f["target_days"] > 0).astype(float))
    return _finish(f, spec, {}, require_target=not inference)


def build_flood(data_dir: Path, inference: bool = False) -> tuple[pd.DataFrame, dict]:
    spec = SPECS["flood"]
    w, st = _weather(data_dir), _static(data_dir)
    fl = _csv(data_dir, SOURCE_FILES["flood"])[["census_district_code", "year", "flood_event_count"]]
    # Cohort rule (no future information): a district is in scope only if the inventory records an
    # event for it on or before the training cutoff. Real data: all 502 qualify, so no district
    # enters because of validation/test-year labels (checked by a unit test on synthetic data).
    first_event = fl[fl["flood_event_count"] > 0].groupby("census_district_code")["year"].min()
    mapped = sorted(first_event[first_event <= TRAIN_TARGET_MAX].index)
    late_entrants = int((first_event > TRAIN_TARGET_MAX).sum())
    fl = fl[fl["census_district_code"].isin(mapped)]
    events = {(int(r.census_district_code), int(r.year)): int(r.flood_event_count)
              for r in fl.itertuples()}
    years_with = {}
    for (d, y), n in events.items():
        if n > 0:
            years_with.setdefault(d, set()).add(y)

    rows = []
    lo, hi = spec.feature_years
    last_label_year = int(fl["year"].max())
    if inference:
        hi = min(int(w["year"].max()), last_label_year)
    wk = w.set_index(["census_district_code", "year"])
    for d in mapped:
        ys = years_with.get(d, set())
        for t in range(lo, hi + 1):
            if (d, t) not in wk.index:
                continue
            rows.append({
                "census_district_code": d, "feature_year": t, "target_year": t + 1,
                "recorded_flood_events_T": events.get((d, t), 0),
                "recorded_flood_events_Tm1": events.get((d, t - 1), 0),
                "flood_years_last5": sum((t - k) in ys for k in range(5)),
                "flood_years_last10": sum((t - k) in ys for k in range(10)),
                "flood_recorded_T": int(events.get((d, t), 0) > 0),
                spec.target: (int(events.get((d, t + 1), 0) > 0) if t + 1 <= last_label_year else np.nan),
            })
    f = pd.DataFrame(rows)
    f = f.merge(w.rename(columns={"year": "feature_year"}), on=["census_district_code", "feature_year"], how="left")
    f = f.merge(st, on="census_district_code", how="left")
    return _finish(f, spec, {"unmapped_or_never_flooded_districts_excluded": int(len(st) - len(mapped)),
                             "districts_excluded_because_first_event_is_after_training_cutoff": late_entrants},
                   require_target=not inference)


def build_malaria(data_dir: Path, inference: bool = False) -> tuple[pd.DataFrame, dict]:
    spec = SPECS["malaria"]
    w, st = _weather(data_dir), _static(data_dir)
    pop = _csv(data_dir, SOURCE_FILES["master"])[["census_district_code", "population_2011"]]
    ma = _csv(data_dir, SOURCE_FILES["malaria"])
    ma = ma[ma["data_coverage"] == "FULL_YEAR"].copy()
    ma["census_district_code"] = ma["census_district_code"].astype(int)
    ma = ma.merge(pop, on="census_district_code", how="left", suffixes=("_src", ""))
    ma["incidence"] = ma["total_cases"] / ma["population_2011"] * 1e5
    ma = ma[["census_district_code", "year", "incidence"]]
    q75 = ma.groupby("year")["incidence"].transform(lambda s: s.quantile(0.75))
    ma["high"] = (ma["incidence"] >= q75).astype(int)
    ma["rank_pct"] = ma.groupby("year")["incidence"].rank(pct=True)
    ma["log1p_inc"] = np.log1p(ma["incidence"])

    def shifted(df, col, name, delta):
        s = df[["census_district_code", "year", col]].copy()
        s["feature_year"] = s["year"] + delta
        return s.drop(columns="year").rename(columns={col: name})

    cur = ma.rename(columns={"year": "feature_year", "log1p_inc": "log1p_incidence_T",
                             "rank_pct": "incidence_rank_pct_T", "high": "high_burden_T"})
    cur = cur[["census_district_code", "feature_year", "log1p_incidence_T",
               "incidence_rank_pct_T", "high_burden_T"]]
    prev = shifted(ma, "log1p_inc", "log1p_incidence_Tm1", +1)
    nxt = shifted(ma, "high", spec.target, -1)
    f = cur.merge(prev, on=["census_district_code", "feature_year"], how="left")
    f = f.merge(nxt, on=["census_district_code", "feature_year"], how="left")
    f = f.merge(w.rename(columns={"year": "feature_year"}), on=["census_district_code", "feature_year"], how="left")
    f = f.merge(st, on="census_district_code", how="left")
    lo, hi = spec.feature_years
    hi = int(ma["year"].max()) if inference else hi
    f = f[f["feature_year"].between(lo, hi)].copy()
    f["target_year"] = f["feature_year"] + 1
    return _finish(f, spec, {"note": "district-years without a FULL_YEAR record are unknown and absent"},
                   require_target=not inference)


BUILDERS = {"heatwave": build_heatwave, "flood": build_flood, "malaria": build_malaria}


# --------------------------------------------------------------------------
# Metadata, writing, checking
# --------------------------------------------------------------------------

def describe(frame: pd.DataFrame, spec: HazardSpec, exclusions: dict, data_dir: Path) -> dict[str, Any]:
    splits = {}
    for name in ("train", "val", "test"):
        part = frame[frame["split"] == name]
        splits[name] = {
            "rows": int(len(part)),
            "positives": int(part[spec.target].sum()),
            "prevalence": round(float(part[spec.target].mean()), 4) if len(part) else None,
            "target_years": [int(part["target_year"].min()), int(part["target_year"].max())] if len(part) else None,
            "districts": int(part["census_district_code"].nunique()),
        }
    return {
        "hazard": spec.name,
        "title": spec.title,
        "target": spec.target,
        "target_definition": spec.target_definition,
        "geography": spec.geography,
        "horizon": spec.horizon,
        "label_source": spec.label_source,
        "label_reliability": spec.label_reliability,
        "limitations": list(spec.limitations),
        "features": spec.features,
        "feature_groups": spec.groups,
        "persistence_feature": spec.persistence_feature,
        "feature_year_range": [int(frame["feature_year"].min()), int(frame["feature_year"].max())],
        "split_rule": {"train": f"target_year <= {TRAIN_TARGET_MAX}",
                       "val": f"target_year in {list(VAL_TARGET_YEARS)}",
                       "test": f"target_year >= {TEST_TARGET_MIN}"},
        "splits": splits,
        "rows": int(len(frame)),
        "exclusions": exclusions,
        "missing_value_policy": "rows with any missing required input are dropped; nothing is imputed",
        "source_files": {k: {"path": "data/" + SOURCE_FILES[k],
                             "sha256": _sha256(data_dir / SOURCE_FILES[k])}
                         for k in sorted({"master", "weather", spec.name})},
    }


def feature_dictionary(spec: HazardSpec) -> pd.DataFrame:
    notes = {
        "latitude": "district representative point latitude (deg)",
        "longitude": "district representative point longitude (deg)",
        "log_population": "log(1 + Census 2011 population)",
        "temp_range_c": "max_temperature_c - min_temperature_c (T)",
        "heatwave_days_T": "days with modelled Tmax >= 45 degC in year T",
        "severe_heatwave_days_T": "days with modelled Tmax >= 47 degC in year T",
        "heatwave_days_Tm1": "heatwave_days in year T-1",
        "recorded_flood_events_T": "inventory flood events in T (0 = none recorded)",
        "recorded_flood_events_Tm1": "inventory flood events in T-1 (0 = none recorded)",
        "flood_years_last5": "years in T-4..T with a recorded flood",
        "flood_years_last10": "years in T-9..T with a recorded flood",
        "log1p_incidence_T": "log(1 + reported cases per 100k, 2011 pop) in T",
        "log1p_incidence_Tm1": "same for T-1",
        "incidence_rank_pct_T": "percentile rank of incidence among reporting districts in T",
        "high_burden_T": "1 if district was top-quartile incidence in T",
    }
    rows = []
    for group, feats in spec.groups.items():
        for f in feats:
            rows.append({"hazard": spec.name, "feature": f, "group": group,
                         "description": notes.get(f, "annual NASA POWER weather summary for year T"),
                         "available_at_prediction_time": "yes (end of year T)"})
    return pd.DataFrame(rows)


def build_all(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, tuple[pd.DataFrame, dict]]:
    out = {}
    for name in HAZARDS:
        frame, excl = BUILDERS[name](Path(data_dir))
        out[name] = (frame, describe(frame, SPECS[name], excl, Path(data_dir)))
    return out


def write_all(data_dir: Path = DEFAULT_DATA_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    out_dir = Path(out_dir)
    built = build_all(data_dir)
    for name, (frame, meta) in built.items():
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(d / "dataset.parquet", index=False)
        feature_dictionary(SPECS[name]).to_csv(d / "feature_dictionary.csv", index=False, lineterminator="\n")
        (d / "dataset_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                                                 encoding="utf-8", newline="\n")
    readiness = hazard_readiness(data_dir, built)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "readiness.json").write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8", newline="\n")
    return readiness


def check_files(data_dir: Path = DEFAULT_DATA_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> list[str]:
    """Rebuild in memory and compare with the files on disk. Returns problems."""
    problems = []
    for name, (frame, meta) in build_all(data_dir).items():
        path = Path(out_dir) / name / "dataset.parquet"
        if not path.is_file():
            problems.append(f"{name}: dataset.parquet missing")
            continue
        disk = pd.read_parquet(path)
        if not disk.equals(frame):
            problems.append(f"{name}: dataset.parquet differs from a fresh build")
        meta_path = Path(out_dir) / name / "dataset_metadata.json"
        if not meta_path.is_file() or json.loads(meta_path.read_text(encoding="utf-8")) != json.loads(json.dumps(meta)):
            problems.append(f"{name}: dataset_metadata.json differs from a fresh build")
    return problems


# --------------------------------------------------------------------------
# Readiness evidence
# --------------------------------------------------------------------------

def _criteria(splits: dict[str, dict]) -> list[str]:
    failures = []
    for name, minimum in MIN_ROWS.items():
        s = splits[name]
        if s["rows"] < minimum:
            failures.append(f"{name}: {s['rows']} rows < required {minimum}")
        if s["positives"] < MIN_POSITIVES_PER_SPLIT or s["rows"] - s["positives"] < MIN_POSITIVES_PER_SPLIT:
            failures.append(f"{name}: fewer than {MIN_POSITIVES_PER_SPLIT} examples of one class "
                            f"({s['positives']} positive of {s['rows']})")
    return failures


def hazard_readiness(data_dir: Path = DEFAULT_DATA_DIR,
                     built: dict | None = None) -> dict[str, Any]:
    """Evidence-based status for every hazard named in the project plan."""
    data_dir = Path(data_dir)
    built = built or build_all(data_dir)
    report: dict[str, Any] = {"criteria": {"min_rows": MIN_ROWS,
                                            "min_examples_per_class_per_split": MIN_POSITIVES_PER_SPLIT,
                                            "split": "chronological by target year (train <= 2019, val 2020-2021, test >= 2022)"},
                              "hazards": {}}
    for name, (frame, meta) in built.items():
        failures = _criteria(meta["splits"])
        entry = {"status": "supported" if not failures else "unsupported",
                 "unmet_criteria": failures, "rows": meta["rows"], "splits": meta["splits"],
                 "label_reliability": meta["label_reliability"]}
        if name == "flood":
            entry["status"] = "supported_with_caveats" if not failures else "unsupported"
            entry["caveat"] = "presence-only labels (see label_reliability)"
        report["hazards"][name] = entry

    # Dengue: state-year, 2021-2026 only.
    dpath = data_dir / SOURCE_FILES["dengue"]
    if dpath.is_file():
        d = pd.read_csv(dpath, na_values=["-", "–", "—", "NR"])
        full = d[d["data_coverage"] == "FULL_YEAR"].dropna(subset=["dengue_cases"])
        years = sorted(int(y) for y in full["year"].unique())
        target_years = [y for y in years if (y - 1) in years]
        per_split = {"train": 0, "val": 0, "test": 0}
        for y in target_years:
            n = int((full["year"] == y).sum())
            per_split[split_of(y)] += n
        report["hazards"]["dengue"] = {
            "status": "unsupported",
            "unmet_criteria": [f"{k}: {v} labelled state-year transitions < required {MIN_ROWS[k]}"
                               for k, v in per_split.items() if v < MIN_ROWS[k]],
            "evidence": {"state_series": int(full["state_name"].nunique()),
                         "full_year_labels": years,
                         "transitions_per_split": per_split,
                         "geography": "state only; district labels would be copies of state values"},
            "recovery_path": ("Obtain a longer state-year (ideally monthly) dengue series, e.g. earlier NCVBDC "
                              "years, so that training years before 2020 exist; keep the "
                              "West Bengal 2024 'verification_pending' record out of scoring until confirmed."),
        }
    else:
        report["hazards"]["dengue"] = {"status": "unsupported", "unmet_criteria": ["source file missing"]}

    report["hazards"]["chikungunya"] = {
        "status": "unsupported",
        "unmet_criteria": ["no chikungunya data exists anywhere in the repository"],
        "recovery_path": "Obtain NCVBDC chikungunya state/district tables, then repeat this readiness check.",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--readiness", action="store_true")
    a = p.parse_args(argv)
    if a.build:
        r = write_all(Path(a.data_dir), Path(a.out_dir))
    elif a.check:
        problems = check_files(Path(a.data_dir), Path(a.out_dir))
        print("OK: datasets match a fresh build" if not problems else "\n".join(problems))
        return 1 if problems else 0
    else:
        r = hazard_readiness(Path(a.data_dir))
    for name, e in r["hazards"].items():
        print(f"{name:12} {e['status']:24} {('; '.join(e['unmet_criteria'])) or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
