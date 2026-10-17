"""Smoke tests for ml/demo_local.py (the read-only beginner demo). Uses the real processed data and
saved models, so they are skipped when those are absent.

    python -m unittest ml.test_demo_local -v
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ml import demo_local
from ml import hazard_datasets as hd

ROOT = Path(__file__).resolve().parents[1]
READY = (all((ROOT / "data" / rel).is_file() for rel in hd.SOURCE_FILES.values())
         and all((ROOT / "ml" / "models" / "hazards" / h / "ann_bundle.joblib").is_file() for h in hd.HAZARDS))


def run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = demo_local.main(list(argv))
    return code, buf.getvalue()


@unittest.skipUnless(READY, "real data / models not present")
class DemoTests(unittest.TestCase):
    def test_default_destination_view_is_agra_and_readable(self):
        code, out = run("destination")
        self.assertEqual(code, 0)
        for text in ("Agra (Uttar Pradesh)", "PART 1", "PART 2", "[dengue] status=available",
                     "state_level_context_not_district_specific", "WITHHELD", "[chikungunya]", "unsupported"):
            self.assertIn(text, out)
        self.assertNotIn("risk score:", out.lower())

    def test_json_mode_is_valid_json(self):
        code, out = run("destination", "--json", "--no-ann")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["destination"]["census_district_code"], 146)

    def test_ambiguous_name_is_reported_not_silently_filtered(self):
        code, out = run("destination", "--district-name", "Bilaspur", "--no-ann")
        self.assertEqual(code, 1)
        self.assertIn("ambiguous_destination", out)

    def test_features_file_round_trip_for_every_supported_hazard(self):
        with tempfile.TemporaryDirectory() as tmp:
            for h in hd.HAZARDS:
                path = Path(tmp) / f"{h}.json"
                self.assertEqual(run("features", "--hazard", h, "--district-code", "146", "--out", str(path))[0], 0)
                self.assertEqual(list(json.loads(path.read_text(encoding="utf-8"))), hd.SPECS[h].features)
                code, out = run("predict", "--hazard", h, "--features-file", str(path))
                self.assertEqual(code, 0, out)
                self.assertIn("status      : ok", out)

    def test_bad_missing_and_unsupported_inputs_give_no_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "flood.json"
            run("features", "--hazard", "flood", "--district-code", "146", "--out", str(good))
            rec = json.loads(good.read_text(encoding="utf-8"))
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({**rec, "avg_humidity_pct": 140}), encoding="utf-8")
            code, out = run("predict", "--hazard", "flood", "--features-file", str(bad), "--no-explain")
            self.assertEqual(code, 1)
            self.assertIn("invalid_input", out)
            self.assertIn("none (nothing was invented)", out)
            rec.pop("total_rainfall_mm")
            miss = Path(tmp) / "miss.json"
            miss.write_text(json.dumps(rec), encoding="utf-8")
            self.assertIn("missing features", run("predict", "--hazard", "flood", "--features-file", str(miss))[1])
            code, out = run("predict", "--hazard", "dengue", "--features-file", str(good))
            self.assertEqual((code, "unsupported" in out), (1, True))
            self.assertEqual(run("predict", "--hazard", "flood", "--features-file", str(Path(tmp) / "nope.json"))[0], 1)
            code, out = run("features", "--hazard", "malaria", "--district-code", "327", "--out", str(Path(tmp) / "x.json"))
            self.assertEqual(code, 1)
            self.assertIn("Nothing is imputed", out)


if __name__ == "__main__":
    unittest.main()
