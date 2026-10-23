"""Tests for ml/data_audit.py using a small SYNTHETIC repository tree.

Run from the repository root:
    python -m unittest ml.test_data_audit -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml import data_audit as audit

ROOT = Path(__file__).resolve().parents[1]


def _csv(root: Path, rel: str, columns: list[str], rows: list[tuple]) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def build_tree(root: Path) -> None:
    """Write a minimal, internally consistent tree (all values synthetic)."""
    _csv(root, "data/reference/district_master.csv",
         ["census_district_code", "district_name", "state_name", "latitude", "longitude",
          "population_2011", "households_2011"],
         [(1, "A", "Assam", 26.0, 92.0, 100, 20), (2, "B", "Assam", 26.5, 92.5, 200, 40),
          (3, "C", "Goa", 15.3, 74.0, 300, 60)])
    _csv(root, "data/processed/weather/weather_district_year_2011_2025.csv",
         ["census_district_code", "year", "avg_temperature_c", "max_temperature_c",
          "min_temperature_c", "avg_humidity_pct", "total_rainfall_mm", "rainy_days"],
         [(c, y, 25.0, 40.0, 10.0, 60.0, 900.0, 90) for c in (1, 2, 3) for y in (2020, 2021)])
    _csv(root, "data/processed/disaster/heatwave/heatwave_district_year_2011_2025.csv",
         ["census_district_code", "year", "heatwave_days", "severe_heatwave_days", "max_temperature_c"],
         [(c, y, 0, 0, 40.0) for c in (1, 2, 3) for y in (2020, 2021)])
    _csv(root, "data/processed/disease/malaria/malaria_district_year_2000_2024.csv",
         ["census_district_code", "year", "state_name", "total_cases", "total_deaths", "data_coverage"],
         [(1, 2020, "Assam", 0, 0, "FULL_YEAR"), (2, 2021, "Assam", 5, 0, "THROUGH_APRIL_2024")])
    _csv(root, "data/processed/disease/dengue/dengue_state_year_2021_2026.csv",
         ["state_name", "year", "dengue_cases", "dengue_deaths", "data_coverage"],
         [("Assam", 2021, 10, 0, "FULL_YEAR"), ("Goa", 2021, 3, 0, "FULL_YEAR")])
    _csv(root, "data/processed/disaster/flood/flood_district_year_1967_2023.csv",
         ["census_district_code", "year", "state_name", "flood_event_count"],
         [(1, 2020, "Assam", 2)])
    _csv(root, "data/processed/features/travelshield_district_year_features_2011_2024.csv",
         ["census_district_code", "year", "state_name", "malaria_cases", "dengue_cases",
          "dengue_deaths", "flood_event_count"],
         [(1, 2021, "Assam", None, 10.0, 0.0, None), (3, 2021, "Goa", None, 3.0, 0.0, None)])


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        build_tree(self.root)

    def dataset(self, report, name):
        return next(d for d in report["datasets"] if d["name"] == name)

    def test_clean_tree_has_no_errors_and_reports_basics(self):
        report = audit.run_audit(self.root)
        self.assertEqual(report["errors"], [])
        weather = self.dataset(report, "weather_year")
        self.assertEqual((weather["rows"], weather["year_range"], weather["unique_districts"]),
                         (6, [2020, 2021], 3))
        self.assertEqual(weather["district_master_coverage"], "3/3")
        self.assertEqual(weather["duplicate_key_rows"], 0)

    def test_missing_values_are_counted_never_imputed(self):
        feats = self.root / "data/processed/features/travelshield_district_year_features_2011_2024.csv"
        before = feats.read_bytes()
        report = audit.run_audit(self.root)
        self.assertEqual(feats.read_bytes(), before)
        missing = self.dataset(report, "master_features")["missing_by_column"]
        self.assertEqual(missing["malaria_cases"], {"count": 2, "percent": 100.0})
        self.assertNotIn("dengue_cases", missing)

    def test_optional_missing_file_is_warning_with_recovery_not_error(self):
        report = audit.run_audit(self.root)
        daily = self.dataset(report, "weather_daily")
        self.assertFalse(daily["present"])
        self.assertEqual(daily["errors"], [])
        self.assertTrue(any("Recovery" in w for w in daily["warnings"]))
        self.assertTrue(any("weather_daily" in w for w in report["warnings"]))

    def test_required_missing_file_is_error_and_exit_code_nonzero(self):
        (self.root / "data/processed/disease/malaria/malaria_district_year_2000_2024.csv").unlink()
        report = audit.run_audit(self.root)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(any("malaria" in e and "missing" in e for e in report["errors"]))
        self.assertEqual(audit.main(["--root", str(self.root), "--check"]), 1)

    def test_duplicate_keys_are_errors(self):
        path = self.root / "data/processed/weather/weather_district_year_2011_2025.csv"
        df = pd.read_csv(path)
        pd.concat([df, df.iloc[[0]]]).to_csv(path, index=False)
        weather = self.dataset(audit.run_audit(self.root), "weather_year")
        self.assertEqual(weather["duplicate_key_rows"], 1)
        self.assertTrue(any("duplicate key" in e for e in weather["errors"]))

    def test_unmatched_district_codes_are_errors(self):
        path = self.root / "data/processed/disaster/flood/flood_district_year_1967_2023.csv"
        df = pd.read_csv(path)
        df.loc[0, "census_district_code"] = 999
        df.to_csv(path, index=False)
        flood = self.dataset(audit.run_audit(self.root), "flood_district_year")
        self.assertEqual(flood["district_codes_not_in_master"], [999])
        self.assertTrue(any("not in the district master" in e for e in flood["errors"]))

    def test_non_numeric_and_negative_counts_are_errors_and_range_is_warning(self):
        weather_path = self.root / "data/processed/weather/weather_district_year_2011_2025.csv"
        df = pd.read_csv(weather_path)
        df["total_rainfall_mm"] = df["total_rainfall_mm"].astype(object)
        df.loc[0, "total_rainfall_mm"] = "abc"   # ("n/a" would be read by pandas as missing)
        df.loc[1, "avg_humidity_pct"] = 150.0
        df.to_csv(weather_path, index=False)
        flood_path = self.root / "data/processed/disaster/flood/flood_district_year_1967_2023.csv"
        flood = pd.read_csv(flood_path)
        flood.loc[0, "flood_event_count"] = -1
        flood.to_csv(flood_path, index=False)
        report = audit.run_audit(self.root)
        weather = self.dataset(report, "weather_year")
        self.assertTrue(any("non-numeric" in e for e in weather["errors"]))
        self.assertEqual(weather["out_of_range_counts"], {"avg_humidity_pct": 1})
        self.assertTrue(any("outside plausible range" in w for w in weather["warnings"]))
        self.assertTrue(any("negative" in e for e in self.dataset(report, "flood_district_year")["errors"]))

    def test_state_level_columns_and_partial_coverage_are_flagged(self):
        report = audit.run_audit(self.root)
        self.assertEqual(self.dataset(report, "master_features")["state_level_columns"],
                         ["dengue_cases", "dengue_deaths"])
        self.assertEqual(self.dataset(report, "dengue_state_year")["spatial_resolution"], "state")
        self.assertIn("THROUGH_APRIL_2024",
                      self.dataset(report, "malaria_district_year")["data_coverage_values"])

    def test_master_dengue_join_defects_are_detected(self):
        # Fixture: Goa has a dengue row but the feature row is missing it.
        path = self.root / "data/processed/features/travelshield_district_year_features_2011_2024.csv"
        df = pd.read_csv(path)
        df.loc[df.census_district_code == 3, "dengue_cases"] = None
        df.to_csv(path, index=False)
        report = audit.run_audit(self.root)
        self.assertTrue(any("no dengue value" in f and "Goa" in f for f in report["cross_dataset_findings"]))

    def _write_flag(self, label="Assam", year=2021, value=10):
        _csv(self.root, "data/reference/dengue_data_quality_flags.csv",
             ["state_name_in_source", "year", "reported_dengue_cases", "status", "reason"],
             [(label, year, value, "verification_pending", "synthetic")])

    def test_valid_verification_pending_flag_is_a_warning_and_listed(self):
        self._write_flag()
        report = audit.run_audit(self.root)
        self.assertEqual(report["errors"], [])
        self.assertTrue(any("verification_pending" in w for w in report["warnings"]))
        self.assertEqual(report["dengue_quality_flags"][0]["reported_dengue_cases"], 10)
        self.assertIn("Data-quality annotations", audit.render_markdown(report))

    def test_flag_whose_value_or_label_no_longer_matches_source_is_an_error(self):
        self._write_flag(value=11)
        self.assertTrue(any("differs from source" in e for e in audit.run_audit(self.root)["errors"]))
        self._write_flag(label="Assam*")
        self.assertTrue(any("matches no source record" in e for e in audit.run_audit(self.root)["errors"]))

    def test_dash_and_blank_dengue_cells_are_not_reported_not_errors_or_zero(self):
        path = self.root / "data/processed/disease/dengue/dengue_state_year_2021_2026.csv"
        with open(path, "a", encoding="utf-8", newline="") as f:
            f.write("Ladakh,2021,-,–,FULL_YEAR\nLadakh,2022,,,FULL_YEAR\n")
        report = audit.run_audit(self.root)
        dengue = self.dataset(report, "dengue_state_year")
        self.assertEqual(dengue["errors"], [])
        self.assertEqual(dengue["not_reported_cells"], {"dengue_cases": 2, "dengue_deaths": 2})
        self.assertEqual(dengue["missing_by_column"]["dengue_cases"]["count"], 2)   # counted missing, not zero
        self.assertTrue(any("never zero" in w for w in dengue["warnings"]))

    def test_report_is_deterministic_and_has_no_local_paths(self):
        first = audit.render_markdown(audit.run_audit(self.root))
        second = audit.render_markdown(audit.run_audit(self.root))
        self.assertEqual(first, second)
        self.assertNotIn(str(self.root), first)
        self.assertIn("Status:", first)

    def test_cli_writes_markdown_and_json(self):
        out, js = self.root / "report.md", self.root / "report.json"
        code = audit.main(["--root", str(self.root), "--output", str(out), "--json", str(js)])
        self.assertEqual(code, 0)
        self.assertIn("# Data Quality Report", out.read_text(encoding="utf-8"))
        json.loads(js.read_text(encoding="utf-8"))


@unittest.skipUnless((ROOT / "data/reference/district_master.csv").is_file(), "real data not present")
class RealDataAuditTests(unittest.TestCase):
    def test_real_repository_has_no_structural_errors(self):
        report = audit.run_audit(ROOT)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["master_districts"], 640)


if __name__ == "__main__":
    unittest.main()
