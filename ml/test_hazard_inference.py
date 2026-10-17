"""Tests for ml/hazard_inference.py.

Most tests train tiny models on SYNTHETIC data in a temp folder. The
"real artifact" tests use the committed models and are skipped if absent.

    python -m unittest ml.test_hazard_inference -v
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml import hazard_datasets as hd
from ml import hazard_inference as hi
from ml import hazard_training as ht
from ml.test_hazard_datasets import N_DISTRICTS, make_synthetic_data

ROOT = Path(__file__).resolve().parents[1]
REAL_MODELS = all((ROOT / "ml" / "models" / "hazards" / h / f).is_file()
                  for h in hd.HAZARDS for f in ("ann_bundle.joblib", "model_card.json"))
REAL_DATA = all((ROOT / "data" / rel).is_file() for rel in hd.SOURCE_FILES.values())


class SyntheticModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.data = cls.tmp / "data"
        make_synthetic_data(cls.data)
        cls.models = cls.tmp / "models"
        for h in hd.HAZARDS:
            frame, _ = hd.BUILDERS[h](cls.data)
            ht.train_hazard(h, frame=frame, meta={}, model_dir=cls.models, result_dir=cls.tmp / "res",
                            quick=True, plots=False)
        hi.clear_caches()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        hi.clear_caches()

    def district(self, hazard, code=None, year=None, **kw):
        frame = hd.BUILDERS[hazard](self.data, inference=True)[0]
        code = code if code is not None else int(frame["census_district_code"].iloc[0])
        return hi.predict_for_district(hazard, code, year, data_dir=self.data, model_dir=self.models, **kw)

    def valid_features(self, hazard):
        row = hd.BUILDERS[hazard](self.data, inference=True)[0].iloc[0]
        return {f: float(row[f]) for f in hd.SPECS[hazard].features}

    # ---- success path --------------------------------------------------------
    def test_every_supported_hazard_returns_a_complete_json_result(self):
        for h in hd.HAZARDS:
            with self.subTest(hazard=h):
                out = self.district(h)
                self.assertEqual(out["status"], "ok")
                json.dumps(out, allow_nan=False)
                for k in ("model_version", "target", "geography", "horizon", "prediction", "confidence",
                          "model_reliability", "training_window", "limitations", "explanation",
                          "feature_year", "target_year", "district"):
                    self.assertIn(k, out)
                self.assertEqual(out["target_year"], out["feature_year"] + 1)
                p = out["prediction"]
                if p["calibrated_probability"] is None:          # no class without accepted calibration
                    self.assertIsNone(p["predicted_class"])
                    self.assertIsNone(p["predicted_label"])
                else:
                    self.assertIn(p["predicted_class"], (0, 1))
                    self.assertIn(p["predicted_label"], hi.CLASS_LABELS[h].values())
                self.assertTrue(0.0 <= p["raw_model_score"] <= 1.0)
                self.assertIsNone(p["risk_band"])
                self.assertTrue(p["risk_band_status"].startswith("unavailable"))
                self.assertEqual(out["model_version"], f"{h}-ann-mlp-v2")

    def test_deterministic_and_no_local_paths(self):
        a, b = self.district("flood"), self.district("flood")
        self.assertEqual(a, b)
        text = json.dumps(a)
        self.assertNotIn(str(self.tmp), text)
        self.assertNotIn("C:\\", text)

    def test_explanation_only_names_real_model_inputs(self):
        for h in hd.HAZARDS:
            out = self.district(h)
            names = {f["feature"] for f in out["explanation"]["top_factors"]}
            self.assertTrue(names <= set(hd.SPECS[h].features))
            self.assertTrue(out["explanation"]["approximate"])
        self.assertNotIn("explanation", self.district("flood", explain=False))

    # ---- invalid input -----------------------------------------------------------
    def test_invalid_feature_records_are_rejected_with_reasons(self):
        good = self.valid_features("heatwave")
        cases = {
            "missing": {k: v for k, v in good.items() if k != "avg_temperature_c"},
            "extra": {**good, "bogus": 1.0},
            "nan": {**good, "avg_temperature_c": float("nan")},
            "inf": {**good, "total_rainfall_mm": float("inf")},
            "string": {**good, "rainy_days": "many"},
            "bool": {**good, "rainy_days": True},
            "humidity range": {**good, "avg_humidity_pct": 140.0},
            "latitude range": {**good, "latitude": 91.0},
            "max below min": {**good, "max_temperature_c": good["min_temperature_c"] - 5, "temp_range_c": 0.0},
            "severe > days": {**good, "heatwave_days_T": 1.0, "severe_heatwave_days_T": 5.0},
            "fractional days": {**good, "rainy_days": 10.5},
        }
        for name, rec in cases.items():
            with self.subTest(case=name):
                out = hi.predict_hazard("heatwave", rec, model_dir=self.models)
                self.assertEqual(out["status"], "invalid_input", out)
                self.assertTrue(out["errors"])
                self.assertIsNone(out["prediction"])
        self.assertEqual(hi.predict_hazard("heatwave", [1, 2, 3], model_dir=self.models)["status"], "invalid_input")

    def test_non_string_hazard_never_raises(self):
        for bad in (None, 123, ["flood"], {"a": 1}, ("flood",)):
            with self.subTest(hazard=bad):
                out = hi.predict_hazard(bad, {}, model_dir=self.models)
                self.assertEqual(out["status"], "invalid_input")
                self.assertIsNone(out["prediction"])
                self.assertEqual(hi.predict_for_district(bad, 5, data_dir=self.data, model_dir=self.models)["status"],
                                 "invalid_input")

    def test_numpy_integers_are_accepted_for_district_and_year(self):
        import numpy as np

        code = int(hd.BUILDERS["flood"](self.data, inference=True)[0]["census_district_code"].iloc[0])
        out = hi.predict_for_district("flood", np.int64(code), np.int64(2021), data_dir=self.data, model_dir=self.models)
        self.assertEqual(out["status"], "ok")
        json.dumps(out, allow_nan=False)

    def test_class_labels_do_not_overstate_the_target(self):
        self.assertEqual(set(hi.CLASS_LABELS), set(hd.HAZARDS))
        self.assertIn("recorded", hi.CLASS_LABELS["flood"][1])            # presence-only inventory target
        self.assertIn("threshold", hi.CLASS_LABELS["heatwave"][1])        # not an IMD declaration
        self.assertIn("reported", hi.CLASS_LABELS["malaria"][1])          # reported, relative burden

    def test_class_is_withheld_together_with_the_probability(self):
        card_path = self.models / "malaria" / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        original = card["calibration"]["display_probability"]
        try:
            card["calibration"]["display_probability"] = False
            card_path.write_text(json.dumps(card), encoding="utf-8")
            hi.clear_caches()
            p = self.district("malaria")["prediction"]
            self.assertIsNone(p["predicted_class"])
            self.assertIsNone(p["predicted_label"])
            self.assertIsNone(p["calibrated_probability"])
            self.assertIsNone(p["decision_threshold"])
            self.assertIn("withheld", p["predicted_class_note"])
            self.assertTrue(0.0 <= p["raw_model_score"] <= 1.0)
        finally:
            card["calibration"]["display_probability"] = original
            card_path.write_text(json.dumps(card), encoding="utf-8")
            hi.clear_caches()

    def test_unsupported_hazards_are_blocked_on_every_entry_point(self):
        for h in ("dengue", "chikungunya"):
            for out in (hi.predict_hazard(h, self.valid_features("flood"), model_dir=self.models),
                        hi.predict_for_district(h, 5, 2021, data_dir=self.data, model_dir=self.models)):
                self.assertEqual(out["status"], "unsupported")
                self.assertIsNone(out["prediction"])
                self.assertNotIn("raw_model_score", json.dumps(out))
                self.assertNotIn("input_features", out)
            run = subprocess.run([sys.executable, "-m", "ml.hazard_inference", "--hazard", h, "--district-code", "5"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 1)                            # CLI signals non-success
            self.assertEqual(json.loads(run.stdout)["status"], "unsupported")

    def test_valid_explicit_record_predicts(self):
        out = hi.predict_hazard("malaria", self.valid_features("malaria"), model_dir=self.models)
        self.assertEqual(out["status"], "ok")

    def test_unknown_hazard_and_bad_district_arguments(self):
        self.assertEqual(hi.predict_hazard("tsunami", {}, model_dir=self.models)["status"], "invalid_input")
        for bad in ("536", 1.5, True, None):
            with self.subTest(code=bad):
                out = hi.predict_for_district("flood", bad, data_dir=self.data, model_dir=self.models)
                self.assertEqual(out["status"], "invalid_input")
        out = hi.predict_for_district("flood", 3, "2020", data_dir=self.data, model_dir=self.models)
        self.assertEqual(out["status"], "invalid_input")

    # ---- unsupported / unavailable -----------------------------------------------
    def test_unsupported_hazards_never_return_a_prediction(self):
        for h in ("dengue", "chikungunya"):
            out = hi.predict_hazard(h, self.valid_features("flood"), model_dir=self.models)
            self.assertEqual(out["status"], "unsupported")
            self.assertIsNone(out["prediction"])
            self.assertIn("recovery_path", out)
            self.assertEqual(hi.predict_for_district(h, 5, model_dir=self.models)["status"], "unsupported")
        listing = hi.list_hazards(self.models)
        self.assertEqual(listing["dengue"]["status"], "unsupported")
        self.assertEqual(listing["flood"]["status"], "supported_with_caveats")

    def test_missing_artifacts_report_unavailable(self):
        with tempfile.TemporaryDirectory() as empty:
            out = hi.predict_hazard("flood", self.valid_features("flood"), model_dir=empty)
            self.assertEqual(out["status"], "unavailable")
            self.assertIsNone(out["prediction"])
            self.assertEqual(hi.list_hazards(empty)["flood"]["status"], "unavailable")

    def test_uncovered_district_or_year_is_unavailable_not_zero(self):
        flood_frame = hd.BUILDERS["flood"](self.data, inference=True)[0]
        outside = (set(range(1, N_DISTRICTS + 1)) - set(flood_frame["census_district_code"])).pop()
        out = hi.predict_for_district("flood", int(outside), data_dir=self.data, model_dir=self.models)
        self.assertEqual(out["status"], "unavailable")
        self.assertEqual(out["confidence"]["data_coverage"], "none")
        self.assertIsNone(out.get("prediction"))
        code = int(flood_frame["census_district_code"].iloc[0])
        out = hi.predict_for_district("flood", code, 1999, data_dir=self.data, model_dir=self.models)
        self.assertEqual(out["status"], "unavailable")
        self.assertIn("1999", out["reason"])

    def test_missing_input_data_is_never_imputed(self):
        path = self.data / hd.SOURCE_FILES["weather"]
        original = path.read_text(encoding="utf-8")
        try:
            df = pd.read_csv(path)
            df = df[~((df["census_district_code"] == 5) & (df["year"] == 2022))]
            df.to_csv(path, index=False)
            hi.clear_caches()
            out = hi.predict_for_district("heatwave", 5, 2022, data_dir=self.data, model_dir=self.models)
            self.assertEqual(out["status"], "unavailable")
            self.assertIsNone(out.get("prediction"))
        finally:
            path.write_text(original, encoding="utf-8")
            hi.clear_caches()

    # ---- calibration gating --------------------------------------------------------
    def test_probability_is_withheld_when_the_validation_calibration_check_fails(self):
        card_path = self.models / "flood" / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        original = card["calibration"]["display_probability"]
        try:
            card["calibration"]["display_probability"] = False
            card_path.write_text(json.dumps(card), encoding="utf-8")
            hi.clear_caches()
            p = self.district("flood")["prediction"]
            self.assertIsNone(p["calibrated_probability"])
            self.assertIsNone(p["decision_threshold"])
            self.assertIn("Withheld", p["calibration_note"])
            self.assertIn("validation years", p["calibration_note"])
            self.assertIn("raw_model_score", p)
        finally:
            card["calibration"]["display_probability"] = original
            card_path.write_text(json.dumps(card), encoding="utf-8")
            hi.clear_caches()

    # ---- fresh process ---------------------------------------------------------------
    def test_fresh_process_gives_identical_output(self):
        code = int(hd.BUILDERS["flood"](self.data, inference=True)[0]["census_district_code"].iloc[0])
        script = ("import json,sys;from ml.hazard_inference import predict_for_district as p;"
                  f"print(json.dumps(p('flood',{code},None,True,sys.argv[1],sys.argv[2])))")
        run = subprocess.run([sys.executable, "-c", script, str(self.data), str(self.models)],
                             cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), json.loads(json.dumps(self.district("flood"))))


@unittest.skipUnless(REAL_MODELS and REAL_DATA, "committed model artifacts / data not present")
class RealArtifactTests(unittest.TestCase):
    def setUp(self):
        hi.clear_caches()

    def test_listing_matches_the_documented_decision(self):
        listing = hi.list_hazards()
        self.assertEqual({k: v["status"] for k, v in listing.items()},
                         {"heatwave": "supported", "flood": "supported_with_caveats", "malaria": "supported",
                          "dengue": "unsupported", "chikungunya": "unsupported"})
        self.assertEqual({h: listing[h]["calibrated_probability_available"] for h in ("heatwave", "flood", "malaria")},
                         {"heatwave": False, "flood": False, "malaria": True})

    def test_real_prediction_has_documented_fields_and_heatwave_probability_is_withheld(self):
        for h, code in (("heatwave", 536), ("flood", 536), ("malaria", 146)):
            out = hi.predict_for_district(h, code)
            self.assertEqual(out["status"], "ok", out)
            self.assertEqual(out["model_version"], f"{h}-ann-mlp-v2")
            json.dumps(out, allow_nan=False)
        for hazard in ("heatwave", "flood"):             # both fail the validation-year calibration check
            p = hi.predict_for_district(hazard, 536)["prediction"]
            self.assertIsNone(p["calibrated_probability"], hazard)
            self.assertIsNone(p["predicted_class"], hazard)   # no class either: threshold is on the probability scale
            self.assertIsNone(p["predicted_label"], hazard)
            self.assertIsNone(p["decision_threshold"], hazard)
            self.assertIn("validation years", p["calibration_note"])
        ma = hi.predict_for_district("malaria", 146)["prediction"]
        self.assertIsNotNone(ma["calibrated_probability"])
        self.assertIn(ma["predicted_class"], (0, 1))

    def test_district_without_malaria_records_is_unavailable(self):
        out = hi.predict_for_district("malaria", 327)       # Darjiling: no malaria rows in the source
        self.assertEqual(out["status"], "unavailable")
        self.assertIsNone(out.get("prediction"))

    def test_real_fresh_process(self):
        run = subprocess.run([sys.executable, "-m", "ml.hazard_inference", "--hazard", "flood", "--district-code", "536"],
                             cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), hi.predict_for_district("flood", 536))


if __name__ == "__main__":
    unittest.main()
