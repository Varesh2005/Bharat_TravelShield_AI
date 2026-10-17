"""Tests for ml/destination_analysis.py.

All unit tests use a tiny SYNTHETIC fixture written to a temporary folder.
No network and no large dataset is needed. Two tests use the real processed
data and are skipped when it is absent.

Run from the repository root:
    python -m unittest ml.test_destination_analysis -v
"""

import json
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

from ml import destination_analysis as da
from ml.destination_analysis import (
    AmbiguousDestinationError,
    DataUnavailableError,
    DestinationNotFoundError,
    InvalidRequestError,
    build_destination_summary,
    safe_destination_summary,
)

ROOT = Path(__file__).resolve().parents[1]

MASTER_COLS = ["census_state_code", "census_district_code", "district_name", "state_name",
               "population_2011", "latitude", "longitude"]
# SYNTHETIC rows: real district names (to exercise aliasing/ambiguity), invented values.
MASTER = [
    (2, 30, "Bilaspur", "Himachal Pradesh", 100, 31.3, 76.7),
    (22, 406, "Bilaspur", "Chhattisgarh", 200, 22.1, 82.1),
    (28, 536, "Hyderabad", "Andhra Pradesh", 300, 17.4, 78.5),
    (1, 3, "Leh(Ladakh)", "Jammu & Kashmir", 400, 34.1, 77.5),
    (12, 10, "Papum Pare", "Arunanchal Pradesh", 500, 27.1, 93.6),
    (19, 20, "Kolkata", "West Bengal", 600, 22.6, 88.4),
    (1, 4, "Kargil", "Jammu & Kashmir", 700, 34.5, 76.1),
    (1, 5, "Srinagar", "Jammu & Kashmir", 800, 34.0, 74.8),
]


def _write(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)


def make_synthetic_data(data_dir: Path) -> None:
    rel = {name: spec[0] for name, spec in da.SOURCES.items()}
    _write(data_dir / rel["master"], MASTER_COLS, MASTER)

    wcols = ["census_district_code", "year", "avg_temperature_c", "max_temperature_c",
             "min_temperature_c", "avg_humidity_pct", "total_rainfall_mm", "rainy_days"]
    weather = [(c, y, 25.0 + y % 3, 40.0 + y % 5, 5.0, 60.0, 800.0 + y % 7, 80)
               for c in (30, 406, 536, 3, 10, 20) for y in (2020, 2021, 2022)]
    _write(data_dir / rel["weather"], wcols, weather)

    hcols = ["census_district_code", "year", "heatwave_days", "severe_heatwave_days",
             "max_temperature_c"]
    heat = [(c, y, 0, 0, 40.0) for c in (30, 406, 3, 10, 20) for y in (2020, 2021, 2022)]
    heat += [(536, 2020, 0, 0, 41.0), (536, 2021, 5, 1, 47.5), (536, 2022, 2, 0, 45.5)]
    _write(data_dir / rel["heatwave"], hcols, heat)

    mcols = ["census_district_code", "year", "total_cases", "total_deaths", "data_coverage"]
    malaria = [(536, 2020, 0, 0, "FULL_YEAR"), (536, 2021, 12, 0, "FULL_YEAR"),
               (3, 2021, 4, 0, "FULL_YEAR")]
    _write(data_dir / rel["malaria"], mcols, malaria)

    dcols = ["state_name", "year", "dengue_cases", "dengue_deaths", "data_coverage"]
    dengue = [
        ("Telangana", 2021, 100, 1, "FULL_YEAR"),
        ("Telangana", 2022, 200, 2, "PROVISIONAL_THROUGH_2026_02_28"),
        ("Andhra Pradesh", 2021, 999, 9, "FULL_YEAR"),
        ("Arunachal Pradesh", 2021, 7, 0, "FULL_YEAR"),
        ("West Bengal*", 2021, 55, 0, "FULL_YEAR"),
        ("West Bengal*", 2024, 441, 0, "FULL_YEAR"),
        ("Jammu and Kashmir", 2021, 5, 0, "FULL_YEAR"),
        ("Himachal Pradesh", 2021, 0, 0, "FULL_YEAR"),
    ]
    _write(data_dir / rel["dengue"], dcols, dengue)
    _write(data_dir / da.DENGUE_FLAGS_FILE,
           ["state_name_in_source", "year", "reported_dengue_cases", "status", "reason"],
           [("West Bengal*", 2024, 441, "verification_pending", "synthetic reason")])

    _write(data_dir / rel["flood"], ["census_district_code", "year", "flood_event_count"],
           [(536, 2021, 2)])


def _walk(obj, path=""):
    """Yield (path, key, value) for every dict entry in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


class SyntheticBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        make_synthetic_data(self.tmp)
        da.clear_store_cache()
        self.addCleanup(da.clear_store_cache)

    def summary(self, **kw):
        kw.setdefault("data_dir", self.tmp)
        return build_destination_summary(**kw)

    def source_path(self, name):
        return self.tmp / da.SOURCES[name][0]


class ResolutionTests(SyntheticBase):
    def test_resolve_by_code_and_by_name(self):
        by_code = self.summary(district_code=536)["destination"]
        by_name = self.summary(district_name="  hyderabad ")["destination"]
        self.assertEqual(by_code, by_name)
        self.assertEqual(by_code["census_district_code"], 536)
        self.assertEqual(by_code["state_name"], "Andhra Pradesh")

    def test_code_accepts_numeric_string_and_whole_float(self):
        self.assertEqual(self.summary(district_code="536")["destination"]["district_name"], "Hyderabad")
        self.assertEqual(self.summary(district_code=536.0)["destination"]["district_name"], "Hyderabad")

    def test_ambiguous_name_lists_candidates_and_state_resolves_it(self):
        with self.assertRaises(AmbiguousDestinationError) as ctx:
            self.summary(district_name="Bilaspur")
        codes = [c["census_district_code"] for c in ctx.exception.extra["candidates"]]
        self.assertEqual(codes, [30, 406])
        resolved = self.summary(district_name="Bilaspur", state_name="Chhattisgarh")
        self.assertEqual(resolved["destination"]["census_district_code"], 406)

    def test_unknown_destination_and_suggestions(self):
        with self.assertRaises(DestinationNotFoundError) as ctx:
            self.summary(district_name="Hyderabadd")
        self.assertIn("Hyderabad", ctx.exception.extra["suggestions"])
        with self.assertRaises(DestinationNotFoundError):
            self.summary(district_code=99999)

    def test_state_alias_and_parenthesised_name(self):
        self.assertEqual(
            self.summary(district_name="Papum Pare", state_name="Arunachal Pradesh")
            ["destination"]["census_district_code"], 10)
        self.assertEqual(self.summary(district_name="Leh")["destination"]["census_district_code"], 3)

    def test_invalid_requests(self):
        for kwargs in (
            {}, {"district_code": True}, {"district_code": "abc"}, {"district_code": 1.5},
            {"district_name": "   "}, {"district_name": 5},
            {"district_code": 536, "district_name": "Kolkata"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(InvalidRequestError):
                self.summary(**kwargs)

    def test_safe_wrapper_returns_error_dict_without_paths(self):
        out = safe_destination_summary(district_name="Bilaspur", data_dir=self.tmp)
        self.assertEqual(out["error"]["code"], "ambiguous_destination")
        self.assertNotIn(str(self.tmp), json.dumps(out))

    def test_missing_master_is_a_controlled_error(self):
        self.source_path("master").unlink()
        da.clear_store_cache()
        with self.assertRaises(DataUnavailableError):
            self.summary(district_code=536)


class PeriodTests(SyntheticBase):
    def test_valid_range_selects_calendar_years(self):
        period = self.summary(district_code=536, start_date="2020-11-15", end_date="2021-02-01")["query_period"]
        self.assertEqual(period["years_covered"], [2020, 2021])
        self.assertTrue(period["spans_partial_year"])
        self.assertEqual(period["resolution_supported"], "annual")

    def test_date_objects_and_whole_year(self):
        period = self.summary(district_code=536, start_date=date(2021, 1, 1), end_date=date(2021, 12, 31))["query_period"]
        self.assertFalse(period["spans_partial_year"])

    def test_invalid_and_reversed_dates(self):
        cases = [
            ("2021-13-01", "2021-12-01"), ("2021-1-1", "2021-12-01"), ("tomorrow", "2021-12-01"),
            ("2021-02-30", "2021-03-01"), (20210101, "2021-12-01"),
            ("2021-06-02", "2021-06-01"),     # reversed
            ("2021-06-01", None), (None, "2021-06-01"),
            ("1800-01-01", "1800-02-01"),
        ]
        for start, end in cases:
            with self.subTest(start=start, end=end), self.assertRaises(InvalidRequestError):
                self.summary(district_code=536, start_date=start, end_date=end)

    def test_month_requested_but_only_annual_data(self):
        s = self.summary(district_code=536, start_date="2021-07-01", end_date="2021-07-31")
        self.assertFalse(s["query_period"]["month_specific_data_available"])
        self.assertIn("not restricted", s["query_period"]["note"])
        for rec in s["hazards"]["weather"]["records"]:
            self.assertEqual(set(rec) - {"year"}, set(da.SOURCES["weather"][1]) - {"census_district_code", "year"})
        self.assertEqual([r["year"] for r in s["hazards"]["weather"]["records"]], [2021])
        keys = {k for _, k, _ in _walk(s["hazards"])}
        self.assertFalse(keys & {"month", "months", "monthly", "date"})

    def test_no_dates_returns_all_years(self):
        s = self.summary(district_code=536)
        self.assertIsNone(s["query_period"]["years_covered"])
        self.assertEqual(len(s["hazards"]["weather"]["records"]), 3)


class HazardSemanticsTests(SyntheticBase):
    def test_observed_zero_is_kept_as_zero(self):
        s = self.summary(district_code=536, start_date="2020-01-01", end_date="2020-12-31")
        record = s["hazards"]["malaria"]["records"][0]
        self.assertEqual((record["year"], record["total_cases"], record["total_deaths"]), (2020, 0, 0))
        self.assertEqual(s["hazards"]["heatwave_threshold"]["records"][0]["heatwave_days"], 0)
        # a zero observation produces no precaution
        self.assertEqual(s["precautions"], [])

    def test_missing_district_is_unknown_not_zero(self):
        s = self.summary(district_code=30)    # Bilaspur HP: no malaria rows, no flood rows
        for name in ("malaria", "flood"):
            h = s["hazards"][name]
            self.assertEqual(h["status"], "no_records_for_destination")
            self.assertEqual(h["records"], [])
            self.assertIsNone(h["summary"])
            self.assertIsNone(h["last_observed_year"])
            self.assertIn("not zero", h["status_reason"])

    def test_missing_years_inside_period_are_listed_not_zero_filled(self):
        s = self.summary(district_code=536, start_date="2020-01-01", end_date="2022-12-31")
        malaria = s["hazards"]["malaria"]
        self.assertEqual(malaria["status"], "partial")
        self.assertEqual(malaria["coverage"]["years_without_records_in_period"], [2022])
        self.assertEqual([r["year"] for r in malaria["records"]], [2020, 2021])

    def test_destination_has_records_but_none_in_period(self):
        s = self.summary(district_code=536, start_date="2023-01-01", end_date="2023-12-31")
        malaria = s["hazards"]["malaria"]
        self.assertEqual(malaria["status"], "no_records_in_period")
        self.assertEqual(malaria["last_observed_year"], 2021)
        self.assertEqual(malaria["records"], [])

    def test_flood_absent_year_is_not_zero(self):
        s = self.summary(district_code=536, start_date="2020-01-01", end_date="2022-12-31")
        flood = s["hazards"]["flood"]
        self.assertEqual(flood["absence_semantics"], "unknown_not_zero")
        self.assertEqual([r["year"] for r in flood["records"]], [2021])
        self.assertIsNone(flood["coverage"]["years_without_records_in_period"])
        self.assertEqual(flood["status"], "available")

    def test_period_entirely_outside_every_source(self):
        s = self.summary(district_code=536, start_date="2035-05-01", end_date="2035-05-31")
        for name, h in s["hazards"].items():
            self.assertEqual(h["records"], [], name)
            self.assertIn(h["status"], {"no_records_in_period"}, name)
        self.assertEqual(s["precautions"], [])

    def test_hazards_independent_and_ordered(self):
        s = self.summary(district_code=536)
        self.assertEqual(list(s["hazards"]), ["weather", "heatwave_threshold", "malaria", "dengue", "flood"])
        self.assertEqual(set(s["coverage_overview"]), set(s["hazards"]))

    def test_heatwave_is_labelled_a_threshold_indicator(self):
        h = self.summary(district_code=536)["hazards"]["heatwave_threshold"]
        self.assertIn("NOT an official IMD-declared", " ".join(h["caveats"]))
        self.assertEqual(h["summary"]["total_threshold_days"], 7)


class StateLevelDengueTests(SyntheticBase):
    def test_dengue_is_labelled_state_level(self):
        h = self.summary(district_code=20)["hazards"]["dengue"]     # West Bengal*
        self.assertEqual(h["spatial_basis"], "state_level_context_not_district_specific")
        self.assertEqual(h["resolution"]["spatial"], "state")
        self.assertEqual(h["state_used"]["state_in_source"], "West Bengal*")   # label preserved
        self.assertEqual(h["records"][0]["source_state_label"], "West Bengal*")
        self.assertIn("NOT observations for this district", " ".join(h["caveats"]))
        self.assertEqual(h["records"][0]["dengue_cases"], 55)

    def test_state_alias_between_master_and_source(self):
        h = self.summary(district_code=10)["hazards"]["dengue"]     # Arunanchal -> Arunachal
        self.assertEqual(h["records"][0]["dengue_cases"], 7)

    def test_census_2011_telangana_district_uses_telangana_series(self):
        h = self.summary(district_code=536)["hazards"]["dengue"]
        self.assertEqual(h["state_used"], {"state_in_source": "Telangana", "mapping": "telangana_district_list"})
        self.assertEqual([r["dengue_cases"] for r in h["records"]], [100, 200])   # not 999
        self.assertTrue(h["summary"]["includes_provisional_records"])

    def test_unattributable_ladakh_district_gets_no_dengue(self):
        h = self.summary(district_code=3)["hazards"]["dengue"]
        self.assertEqual(h["status"], "no_records_for_destination")
        self.assertEqual(h["records"], [])
        self.assertIn("not assigned to Leh or Kargil", h["status_reason"])

    def test_observed_state_zero_stays_zero_and_gives_no_precaution(self):
        s = self.summary(district_code=30, start_date="2021-01-01", end_date="2021-12-31")
        self.assertEqual(s["hazards"]["dengue"]["records"][0]["dengue_cases"], 0)
        self.assertNotIn("dengue", [p["hazard"] for p in s["precautions"]])


class DengueQualityTests(SyntheticBase):
    """Confirmed dengue data-quality decisions (regression tests)."""

    def dengue(self, code, start, end):
        return self.summary(district_code=code, start_date=start, end_date=end)["hazards"]["dengue"]

    def test_west_bengal_2024_is_verification_pending_with_value_and_label_kept(self):
        h = self.dengue(20, "2024-01-01", "2024-12-31")
        rec = h["records"][0]
        self.assertEqual((rec["year"], rec["dengue_cases"], rec["source_state_label"]), (2024, 441, "West Bengal*"))
        self.assertEqual(rec["reporting_status"], "reported")
        self.assertEqual(rec["data_quality"]["status"], "verification_pending")
        self.assertFalse(rec["data_quality"]["usable_as_verified_input"])
        self.assertTrue(rec["data_quality"]["flag_matches_source_value"])
        self.assertEqual(h["data_quality"]["verification_pending_years"], [2024])
        self.assertTrue(h["summary"]["includes_verification_pending_records"])
        self.assertEqual(h["summary"]["sum_of_reported_state_cases"], 441)    # not altered or dropped

    def test_flag_makes_no_correction_and_no_inference_from_the_year_on_year_drop(self):
        h = self.dengue(20, "2021-01-01", "2024-12-31")
        by_year = {r["year"]: r for r in h["records"]}
        self.assertEqual(by_year[2024]["dengue_cases"], 441)
        banned = {"corrected_value", "corrected_cases", "estimated_cases", "imputed", "suspected_error"}
        self.assertFalse(banned & {k for _, k, _ in _walk(h)})
        # other years of the same starred series are not flagged
        self.assertEqual(by_year[2021]["data_quality"], {"status": "as_reported", "usable_as_verified_input": True})

    def test_source_file_is_not_modified_by_flagging(self):
        path = self.source_path("dengue")
        before = path.read_bytes()
        self.dengue(20, "2024-01-01", "2024-12-31")
        self.assertEqual(path.read_bytes(), before)
        self.assertIn(b"West Bengal*", before)

    def test_pending_record_is_not_used_to_support_a_precaution(self):
        only_2024 = self.summary(district_code=20, start_date="2024-01-01", end_date="2024-12-31")
        self.assertNotIn("dengue", [p["hazard"] for p in only_2024["precautions"]])
        with_2021 = self.summary(district_code=20, start_date="2021-01-01", end_date="2024-12-31")
        self.assertIn("dengue", [p["hazard"] for p in with_2021["precautions"]])

    def test_changed_source_value_is_reported_by_the_flag(self):
        path = self.source_path("dengue")
        df = pd.read_csv(path)
        df.loc[(df.state_name == "West Bengal*") & (df.year == 2024), "dengue_cases"] = 500
        df.to_csv(path, index=False)
        da.clear_store_cache()
        rec = self.dengue(20, "2024-01-01", "2024-12-31")["records"][0]
        self.assertEqual(rec["dengue_cases"], 500)                       # source value shown as is
        self.assertFalse(rec["data_quality"]["flag_matches_source_value"])
        self.assertEqual(rec["data_quality"]["status"], "verification_pending")

    def test_missing_flag_file_means_no_annotations_not_a_crash(self):
        (self.tmp / da.DENGUE_FLAGS_FILE).unlink()
        da.clear_store_cache()
        rec = self.dengue(20, "2024-01-01", "2024-12-31")["records"][0]
        self.assertEqual(rec["data_quality"]["status"], "as_reported")

    def test_telangana_district_uses_telangana_series_labelled_state_level(self):
        s = self.summary(district_code=536, start_date="2021-01-01", end_date="2021-12-31")
        h = s["hazards"]["dengue"]
        self.assertEqual(h["spatial_basis"], "state_level_context_not_district_specific")
        self.assertEqual(s["coverage_overview"]["dengue"]["spatial_basis"], "state_level_context_not_district_specific")
        self.assertEqual(h["resolution"]["spatial"], "state")
        self.assertEqual(h["state_used"], {"state_in_source": "Telangana", "mapping": "telangana_district_list"})
        self.assertEqual(h["records"][0]["dengue_cases"], 100)             # not Andhra Pradesh's 999
        self.assertEqual(h["records"][0]["source_state_label"], "Telangana")
        self.assertIn("NOT observations for this district", " ".join(h["caveats"]))

    def test_leh_and_kargil_never_receive_jammu_and_kashmir_figures(self):
        for code in (3, 4):
            with self.subTest(code=code):
                h = self.dengue(code, "2021-01-01", "2021-12-31")
                self.assertEqual(h["status"], "no_records_for_destination")
                self.assertEqual(h["records"], [])
                self.assertIsNone(h["summary"])
                self.assertIsNone(h["state_used"])
        # ... while a non-Ladakh J&K district keeps the state-level J&K series
        srinagar = self.dengue(5, "2021-01-01", "2021-12-31")
        self.assertEqual(srinagar["records"][0]["dengue_cases"], 5)
        self.assertEqual(srinagar["spatial_basis"], "state_level_context_not_district_specific")
        self.assertIn("not assigned to Leh or Kargil", " ".join(srinagar["caveats"]))

    def _add_ladakh_rows(self):
        with open(self.source_path("dengue"), "a", encoding="utf-8", newline="") as f:
            f.write("Ladakh,2021,-,–,FULL_YEAR\nLadakh,2022,,,FULL_YEAR\nLadakh,2023,12,0,FULL_YEAR\n")
        da.clear_store_cache()

    def test_blank_or_dash_ladakh_values_are_not_reported_never_zero(self):
        self._add_ladakh_rows()
        for start, end in (("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")):
            with self.subTest(start=start):
                h = self.dengue(3, start, end)
                rec = h["records"][0]
                self.assertIsNone(rec["dengue_cases"])
                self.assertIsNone(rec["dengue_deaths"])
                self.assertEqual(rec["reporting_status"], "not_reported")
                self.assertEqual(rec["data_quality"]["status"], "not_reported")
                self.assertFalse(rec["data_quality"]["usable_as_verified_input"])
                self.assertEqual(h["status"], "not_reported")
                self.assertIsNone(h["summary"]["sum_of_reported_state_cases"])
        mixed = self.dengue(4, "2021-01-01", "2023-12-31")
        self.assertEqual([r["dengue_cases"] for r in mixed["records"]], [None, None, 12])
        self.assertEqual(mixed["summary"]["sum_of_reported_state_cases"], 12)
        self.assertEqual(mixed["summary"]["years_not_reported"], 2)
        self.assertEqual(mixed["state_used"]["mapping"], "ladakh_series")
        self.assertNotIn(5, [r["dengue_cases"] for r in mixed["records"]])  # J&K's value never leaks

    def test_not_reported_ladakh_gives_no_precaution_but_observed_state_zero_stays_zero(self):
        self._add_ladakh_rows()
        s = self.summary(district_code=3, start_date="2021-01-01", end_date="2022-12-31")
        self.assertNotIn("dengue", [p["hazard"] for p in s["precautions"]])
        zero = self.dengue(30, "2021-01-01", "2021-12-31")["records"][0]   # Himachal: real zero
        self.assertEqual((zero["dengue_cases"], zero["reporting_status"]), (0, "reported"))
        self.assertEqual(zero["data_quality"]["status"], "as_reported")

    def test_no_score_or_verified_claim_added(self):
        h = self.dengue(20, "2021-01-01", "2024-12-31")
        self.assertFalse(any("score" in k.lower() for _, k, _ in _walk(h)))
        json.dumps(h, allow_nan=False)


class SourceFailureTests(SyntheticBase):
    def test_missing_hazard_file_does_not_break_other_hazards(self):
        self.source_path("malaria").unlink()
        da.clear_store_cache()
        s = self.summary(district_code=536)
        self.assertEqual(s["hazards"]["malaria"]["status"], "source_unavailable")
        self.assertIn("source_file_missing", s["hazards"]["malaria"]["status_reason"])
        self.assertEqual(s["hazards"]["weather"]["status"], "available")
        self.assertEqual(s["hazards"]["dengue"]["status"], "available")
        self.assertNotIn(str(self.tmp), json.dumps(s))

    def test_schema_mismatch_and_unreadable_files(self):
        pd.DataFrame({"census_district_code": [536], "year": [2021]}).to_csv(self.source_path("flood"), index=False)
        self.source_path("heatwave").write_bytes(b"")
        da.clear_store_cache()
        s = self.summary(district_code=536)
        self.assertIn("source_schema_mismatch", s["hazards"]["flood"]["status_reason"])
        self.assertIn("source_file_unreadable", s["hazards"]["heatwave_threshold"]["status_reason"])

    def test_duplicate_keys_make_the_source_unusable(self):
        path = self.source_path("weather")
        df = pd.read_csv(path)
        pd.concat([df, df.iloc[[0]]]).to_csv(path, index=False)
        da.clear_store_cache()
        s = self.summary(district_code=536)
        self.assertEqual(s["hazards"]["weather"]["status"], "source_unavailable")
        self.assertIn("duplicate_keys", s["hazards"]["weather"]["status_reason"])
        self.assertEqual(s["hazards"]["weather"]["records"], [])


class OutputContractTests(SyntheticBase):
    def test_json_serialisable_and_deterministic(self):
        kwargs = dict(district_code=536, start_date="2020-01-01", end_date="2022-12-31")
        first, second = self.summary(**kwargs), self.summary(**kwargs)
        text = json.dumps(first, allow_nan=False, sort_keys=True)     # no NaN/inf allowed
        self.assertEqual(text, json.dumps(second, allow_nan=False, sort_keys=True))
        self.assertEqual(json.loads(text), first)

    def test_no_invented_overall_score_or_verdict(self):
        s = self.summary(district_code=536)
        banned = ("score", "overall", "risk_level", "risk_score", "safe", "verdict", "rating")
        for path, key, _ in _walk({k: v for k, v in s.items() if k != "experimental_ann_predictions"}):
            self.assertFalse(any(b in key.lower() for b in banned), f"{path}/{key}")

    def test_top_level_shape(self):
        s = self.summary(district_code=536)
        self.assertEqual(
            list(s),
            ["schema_version", "summary_type", "destination", "query_period", "hazards",
             "coverage_overview", "precautions", "experimental_ann_predictions", "limitations", "disclaimer"])
        self.assertEqual(s["schema_version"], "1.1")
        for h in s["hazards"].values():
            self.assertTrue({"status", "records", "summary", "coverage", "last_observed_year",
                             "caveats", "source", "resolution", "units"} <= set(h))
        self.assertTrue(all(p["kind"] == "generic_guidance_not_official_advice" for p in s["precautions"]))

    def test_precautions_only_where_records_support_them(self):
        s = self.summary(district_code=536, start_date="2021-01-01", end_date="2021-12-31")
        self.assertEqual([p["hazard"] for p in s["precautions"]],
                         ["heatwave_threshold", "malaria", "dengue", "flood"])

    def test_no_local_paths_in_output(self):
        text = json.dumps(self.summary(district_code=536))
        self.assertNotIn(str(self.tmp), text)
        self.assertNotIn("C:\\", text)


class OptionalAnnTests(SyntheticBase):
    """The hazard ANN block is optional: it must never break the data summary."""

    HAZARDS = ("heatwave", "flood", "malaria", "dengue", "chikungunya")

    def test_ann_not_requested_by_default(self):
        ann = self.summary(district_code=536)["experimental_ann_predictions"]
        self.assertEqual(ann, {"experimental": True, "status": "not_requested"})

    def test_missing_artifacts_do_not_crash_the_summary(self):
        with mock.patch("ml.hazard_inference.predict_for_district", side_effect=FileNotFoundError("/secret/path/model.joblib")):
            s = self.summary(district_code=536, include_experimental_ann=True)
        for h in self.HAZARDS:
            self.assertEqual(s["experimental_ann_predictions"]["hazards"][h]["status"], "unavailable")
            self.assertIsNone(s["experimental_ann_predictions"]["hazards"][h]["prediction"])
        self.assertNotIn("secret", json.dumps(s))
        self.assertEqual(s["hazards"]["weather"]["status"], "available")

    def test_missing_ml_dependencies_do_not_crash_the_summary(self):
        with mock.patch("ml.hazard_inference.predict_for_district", side_effect=ImportError("sklearn")):
            ann = self.summary(district_code=536, include_experimental_ann=True)["experimental_ann_predictions"]
        self.assertIn("dependencies", ann["hazards"]["flood"]["reason"])

    def test_unexpected_model_error_is_contained(self):
        with mock.patch("ml.hazard_inference.predict_for_district", side_effect=RuntimeError("boom /x/y")):
            ann = self.summary(district_code=536, include_experimental_ann=True)["experimental_ann_predictions"]
        self.assertEqual(ann["hazards"]["malaria"]["status"], "unavailable")
        self.assertNotIn("/x/y", json.dumps(ann))

    def test_arguments_are_passed_through_to_the_single_ml_interface(self):
        calls = []

        def fake(hazard, code, year, explain=True, data_dir=None, **kw):
            calls.append((hazard, code, year, explain, Path(data_dir) == self.tmp))
            return {"hazard": hazard, "status": "ok", "prediction": {"predicted_class": 0}}

        with mock.patch("ml.hazard_inference.predict_for_district", side_effect=fake):
            ann = self.summary(district_code=536, include_experimental_ann=True, ann_feature_year=2021)["experimental_ann_predictions"]
        self.assertEqual([c[0] for c in calls], list(self.HAZARDS))
        self.assertTrue(all(c[1:] == (536, 2021, False, True) for c in calls))
        self.assertTrue(ann["experimental"])
        self.assertIn("none is a travel-date or month forecast", ann["note"])
        self.assertNotIn("score", json.dumps({k: v for k, v in ann.items() if k != "hazards"}).lower().replace("no combined score", ""))

    def test_real_interface_on_synthetic_data_is_contained_and_json_safe(self):
        # Uses the committed models if present; otherwise every hazard is 'unavailable'.
        s = self.summary(district_code=536, include_experimental_ann=True)
        ann = s["experimental_ann_predictions"]["hazards"]
        json.dumps(s, allow_nan=False)
        self.assertEqual(set(ann), set(self.HAZARDS))
        for h in ("dengue", "chikungunya"):
            self.assertEqual(ann[h]["status"], "unsupported")
            self.assertIsNone(ann[h]["prediction"])
        for h in ("heatwave", "flood", "malaria"):
            self.assertIn(ann[h]["status"], {"ok", "unavailable", "invalid_input"})
        self.assertEqual(s["hazards"]["weather"]["status"], "available")

    def test_invalid_feature_year_is_reported_not_raised(self):
        for year in (1999, "2021", True):
            with self.subTest(year=year):
                ann = self.summary(district_code=536, include_experimental_ann=True, ann_feature_year=year)["experimental_ann_predictions"]["hazards"]
                for h in ("heatwave", "flood", "malaria"):
                    self.assertNotEqual(ann[h]["status"], "ok")
                    self.assertIsNone(ann[h]["prediction"])


@unittest.skipUnless(
    all((ROOT / "data" / rel).is_file() for rel, _, _ in da.SOURCES.values()),
    "real processed data not present",
)
class RealDataTests(unittest.TestCase):
    def setUp(self):
        da.clear_store_cache()

    def test_real_summary_is_serialisable_with_expected_provenance(self):
        s = build_destination_summary(district_name="Agra", start_date="2022-04-01", end_date="2022-06-30")
        json.dumps(s, allow_nan=False)
        self.assertEqual(s["destination"]["state_name"], "Uttar Pradesh")
        self.assertEqual(s["hazards"]["dengue"]["spatial_basis"], "state_level_context_not_district_specific")
        self.assertEqual(s["hazards"]["weather"]["records"][0]["year"], 2022)

    def test_real_ann_block_uses_the_training_feature_record(self):
        dataset = ROOT / "ml" / "data" / "hazards" / "heatwave" / "dataset.parquet"
        models = ROOT / "ml" / "models" / "hazards" / "heatwave" / "ann_bundle.joblib"
        if not (dataset.is_file() and models.is_file()):
            self.skipTest("hazard dataset / model not present")
        from ml.hazard_datasets import SPECS

        train = pd.read_parquet(dataset)
        row = train[(train.census_district_code == 1) & (train.feature_year == 2012)].iloc[0]
        ann = build_destination_summary(district_code=1, include_experimental_ann=True,
                                        ann_feature_year=2012)["experimental_ann_predictions"]["hazards"]["heatwave"]
        self.assertEqual(ann["status"], "ok")
        self.assertEqual((ann["feature_year"], ann["target_year"]), (2012, 2013))
        for name in SPECS["heatwave"].features:
            self.assertAlmostEqual(ann["input_features"][name], float(row[name]), places=6, msg=name)


if __name__ == "__main__":
    unittest.main()
