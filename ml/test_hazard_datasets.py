"""Tests for ml/hazard_datasets.py: preparation, leakage, missing-data policy, readiness.

All unit tests use a SYNTHETIC data folder (no network, no real data). Two tests
check the real processed data and are skipped when it is absent.

    python -m unittest ml.test_hazard_datasets -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml import hazard_datasets as hd

ROOT = Path(__file__).resolve().parents[1]
N_DISTRICTS = 48
YEARS = range(2011, 2026)


def make_synthetic_data(data_dir: Path, seed: int = 7, learnable: bool = True) -> None:
    """Write a small, fully SYNTHETIC copy of the processed tables."""
    rng = np.random.default_rng(seed)
    codes = np.arange(1, N_DISTRICTS + 1)
    states = [f"State{(c - 1) % 8}" for c in codes]
    prone = rng.uniform(0, 1, N_DISTRICTS)               # persistent district tendency
    (data_dir / "reference").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"census_district_code": codes, "district_name": [f"D{c}" for c in codes],
                  "state_name": states, "latitude": rng.uniform(8, 35, N_DISTRICTS),
                  "longitude": rng.uniform(70, 95, N_DISTRICTS),
                  "population_2011": rng.integers(2e5, 5e6, N_DISTRICTS)}).to_csv(
        data_dir / hd.SOURCE_FILES["master"], index=False)

    w, h, fl, ma = [], [], [], []
    for i, c in enumerate(codes):
        for y in YEARS:
            tmax = 33 + 14 * prone[i] + rng.normal(0, 1.5)
            w.append((c, y, 20 + 8 * prone[i], tmax, 8.0, 55 + 20 * (1 - prone[i]),
                      600 + 1500 * prone[i] + rng.normal(0, 80), int(60 + 40 * prone[i])))
            days = int(max(0, rng.poisson(max(0.0, (tmax - 43)) * 2)))
            h.append((c, y, days, int(days // 4), round(tmax, 2)))
            if y <= 2023 and rng.uniform() < (0.15 + 0.6 * prone[i]) and i % 6 != 0:
                fl.append((c, y, int(rng.integers(1, 4))))
            if i % 9 != 0:                                    # some districts never reported
                cov = "THROUGH_APRIL_2024" if y == 2024 else "FULL_YEAR"
                if y <= 2024:
                    ma.append((c, y, int(rng.lognormal(3 + 3 * prone[i], 0.6)), 0, cov))
    (data_dir / hd.SOURCE_FILES["weather"]).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(w, columns=["census_district_code", "year", "avg_temperature_c", "max_temperature_c",
                             "min_temperature_c", "avg_humidity_pct", "total_rainfall_mm", "rainy_days"]
                 ).to_csv(data_dir / hd.SOURCE_FILES["weather"], index=False)
    for key, rows, cols in (
        ("heatwave", h, ["census_district_code", "year", "heatwave_days", "severe_heatwave_days", "max_temperature_c"]),
        ("flood", fl, ["census_district_code", "year", "flood_event_count"]),
        ("malaria", ma, ["census_district_code", "year", "total_cases", "total_deaths", "data_coverage"]),
    ):
        path = data_dir / hd.SOURCE_FILES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    path = data_dir / hd.SOURCE_FILES["dengue"]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([(s, y, 100 + y, 1, "FULL_YEAR") for s in sorted(set(states)) for y in range(2021, 2026)],
                 columns=["state_name", "year", "dengue_cases", "dengue_deaths", "data_coverage"]).to_csv(path, index=False)


class SyntheticBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        make_synthetic_data(self.tmp)


class SplitTests(unittest.TestCase):
    def test_split_boundaries_are_chronological(self):
        self.assertEqual([hd.split_of(y) for y in (2013, 2019, 2020, 2021, 2022, 2025)],
                         ["train", "train", "val", "val", "test", "test"])

    def test_every_hazard_has_disjoint_ordered_splits(self):
        for name, builder in hd.BUILDERS.items():
            with self.subTest(hazard=name):
                with tempfile.TemporaryDirectory() as d:
                    make_synthetic_data(Path(d))
                    frame, _ = builder(Path(d))
                by = {s: frame[frame["split"] == s]["target_year"] for s in ("train", "val", "test")}
                self.assertTrue(all(len(v) for v in by.values()))
                self.assertLess(by["train"].max(), by["val"].min())
                self.assertLess(by["val"].max(), by["test"].min())
                self.assertTrue((frame["target_year"] == frame["feature_year"] + 1).all())


class LeakageTests(SyntheticBase):
    """Changing anything after year T must not change the features of year T."""

    def _perturb_future(self, cutoff: int):
        for key in ("weather", "heatwave", "flood", "malaria"):
            path = self.tmp / hd.SOURCE_FILES[key]
            df = pd.read_csv(path)
            late = df["year"] > cutoff
            for col in df.columns:
                if col in ("census_district_code", "year", "data_coverage"):
                    continue
                df.loc[late, col] = (df.loc[late, col] * 3 + 11).round(2)
            df.to_csv(path, index=False)

    def test_features_up_to_T_do_not_depend_on_later_years(self):
        before = {h: b(self.tmp, inference=True)[0] for h, b in hd.BUILDERS.items()}
        cutoff = 2016
        self._perturb_future(cutoff)
        for h, builder in hd.BUILDERS.items():
            with self.subTest(hazard=h):
                after = builder(self.tmp, inference=True)[0]
                feats = hd.SPECS[h].features
                a = before[h][before[h]["feature_year"] <= cutoff].set_index(["census_district_code", "feature_year"])
                b = after[after["feature_year"] <= cutoff].set_index(["census_district_code", "feature_year"])
                self.assertGreater(len(a), 100)
                idx = a.index.intersection(b.index)
                self.assertEqual(len(idx), len(a))
                pd.testing.assert_frame_equal(a.loc[idx, feats], b.loc[idx, feats])

    def test_malaria_label_threshold_uses_only_its_own_year(self):
        frame, _ = hd.build_malaria(self.tmp)
        by_year = frame.groupby("target_year")["target_malaria_high_next_year"].mean()
        self.assertTrue(by_year.between(0.1, 0.45).all(), by_year.to_dict())

    def test_inference_frame_is_a_superset_with_unknown_targets(self):
        for h, builder in hd.BUILDERS.items():
            with self.subTest(hazard=h):
                train, _ = builder(self.tmp)
                inf, _ = builder(self.tmp, inference=True)
                self.assertGreater(len(inf), len(train))
                newest = inf[inf["feature_year"] == inf["feature_year"].max()]
                self.assertTrue(newest[hd.SPECS[h].target].isna().all())


class MissingDataPolicyTests(SyntheticBase):
    def test_missing_malaria_year_is_unknown_not_zero(self):
        path = self.tmp / hd.SOURCE_FILES["malaria"]
        df = pd.read_csv(path)
        code = int(df["census_district_code"].iloc[0])
        df = df[~((df["census_district_code"] == code) & (df["year"] == 2016))]
        df.to_csv(path, index=False)
        frame, _ = hd.build_malaria(self.tmp)
        rows = frame[frame["census_district_code"] == code]
        # T=2015 needs 2016 as target, T=2016 needs 2016 as feature, T=2017 needs 2016 as lag
        self.assertTrue({2015, 2016, 2017}.isdisjoint(set(rows["feature_year"])))
        self.assertIn(2018, set(rows["feature_year"]))      # other years of the district remain

    def test_partial_year_malaria_rows_are_never_used(self):
        frame, _ = hd.build_malaria(self.tmp, inference=True)
        self.assertLessEqual(frame["target_year"].max(), 2024)
        self.assertNotIn(2024, set(frame["feature_year"]))

    def test_weather_gap_drops_rows_and_is_counted(self):
        path = self.tmp / hd.SOURCE_FILES["weather"]
        df = pd.read_csv(path)
        df = df[~((df["census_district_code"] == 3) & (df["year"] == 2015))]
        df.to_csv(path, index=False)
        frame, meta = hd.build_heatwave(self.tmp)
        rows = frame[frame["census_district_code"] == 3]
        # T=2015 has no weather; T=2016 lacks the weather-free lag? (lag uses heatwave table, so it stays)
        self.assertNotIn(2015, set(rows["feature_year"]))
        self.assertIn(2016, set(rows["feature_year"]))
        self.assertTrue((frame[hd.SPECS["heatwave"].features].notna()).all().all())

    def test_flood_excludes_unmapped_districts_and_documents_it(self):
        frame, excl = hd.build_flood(self.tmp)
        fl = pd.read_csv(self.tmp / hd.SOURCE_FILES["flood"])
        first = fl.groupby("census_district_code")["year"].min()
        expected = set(first[first <= hd.TRAIN_TARGET_MAX].index)
        self.assertEqual(set(frame["census_district_code"]), expected)
        self.assertEqual(excl["unmapped_or_never_flooded_districts_excluded"], N_DISTRICTS - len(expected))
        self.assertIn("none recorded", hd.SPECS["flood"].target_definition)

    def test_flood_cohort_never_depends_on_validation_or_test_labels(self):
        path = self.tmp / hd.SOURCE_FILES["flood"]
        df = pd.read_csv(path)
        _, before = hd.build_flood(self.tmp)
        first = df.groupby("census_district_code")["year"].min()
        late = int(first[first <= hd.TRAIN_TARGET_MAX].index[0])          # currently in scope
        df = df[df["census_district_code"] != late]
        extra = pd.DataFrame({"census_district_code": [late], "year": [2022], "flood_event_count": [3]})
        pd.concat([df, extra]).to_csv(path, index=False)      # first-ever event only in a test year
        frame, excl = hd.build_flood(self.tmp)
        self.assertNotIn(late, set(frame["census_district_code"]))
        self.assertEqual(excl["districts_excluded_because_first_event_is_after_training_cutoff"],
                         before["districts_excluded_because_first_event_is_after_training_cutoff"] + 1)
        inf, _ = hd.build_flood(self.tmp, inference=True)
        self.assertNotIn(late, set(inf["census_district_code"]))

    def test_no_imputation_anywhere(self):
        for h, builder in hd.BUILDERS.items():
            frame, _ = builder(self.tmp)
            self.assertFalse(frame.isna().any().any(), h)


class ReadinessTests(SyntheticBase):
    def test_dengue_and_chikungunya_are_unsupported_with_evidence(self):
        r = hd.hazard_readiness(self.tmp)["hazards"]
        self.assertEqual(r["dengue"]["status"], "unsupported")
        self.assertEqual(r["dengue"]["evidence"]["transitions_per_split"]["train"], 0)
        self.assertIn("recovery_path", r["dengue"])
        self.assertEqual(r["chikungunya"]["status"], "unsupported")

    def test_too_small_hazard_is_unsupported(self):
        # 48 synthetic districts cannot satisfy the row minimum for every split
        r = hd.hazard_readiness(self.tmp)["hazards"]
        self.assertTrue(any(v["status"] == "unsupported" for k, v in r.items() if k in hd.HAZARDS))

    def test_metadata_and_dictionary_are_json_safe_and_path_free(self):
        out = self.tmp / "out"
        hd.write_all(self.tmp, out)
        for h in hd.HAZARDS:
            text = (out / h / "dataset_metadata.json").read_text(encoding="utf-8")
            json.loads(text)
            self.assertNotIn(str(self.tmp), text)
            fd = pd.read_csv(out / h / "feature_dictionary.csv")
            self.assertEqual(list(fd["feature"]), hd.SPECS[h].features)
        self.assertEqual(hd.check_files(self.tmp, out), [])
        (out / "flood" / "dataset.parquet").unlink()
        self.assertTrue(hd.check_files(self.tmp, out))


@unittest.skipUnless(all((ROOT / "data" / rel).is_file() for rel in hd.SOURCE_FILES.values()), "real data not present")
class RealDataTests(unittest.TestCase):
    def test_committed_datasets_match_a_fresh_build(self):
        self.assertEqual(hd.check_files(), [])

    def test_real_readiness_matches_the_documented_decision(self):
        r = hd.hazard_readiness()["hazards"]
        self.assertEqual(r["heatwave"]["status"], "supported")
        self.assertEqual(r["flood"]["status"], "supported_with_caveats")
        self.assertEqual(r["malaria"]["status"], "supported")
        self.assertEqual(r["dengue"]["status"], "unsupported")
        self.assertEqual(r["chikungunya"]["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
