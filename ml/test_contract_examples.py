"""Executes the copy-paste example in ml/ML_MODEL_CONTRACT.md against the saved models, so the handoff
document cannot silently go stale. Skipped when the real data or model artifacts are absent.

    python -m unittest ml.test_contract_examples -v
"""

import json
import re
import unittest
from pathlib import Path

from ml import hazard_datasets as hd

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "ml" / "ML_MODEL_CONTRACT.md"
READY = (all((ROOT / "data" / rel).is_file() for rel in hd.SOURCE_FILES.values())
         and all((ROOT / "ml" / "models" / "hazards" / h / "ann_bundle.joblib").is_file() for h in hd.HAZARDS))


def load_example() -> dict:
    text = CONTRACT.read_text(encoding="utf-8")
    m = re.search(r"<!-- handoff-example -->\s*```python\n(.*?)```", text, re.S)
    assert m, "handoff example block not found in the contract"
    ns: dict = {"__name__": "handoff_example"}
    exec(compile(m.group(1), "ML_MODEL_CONTRACT.md:handoff-example", "exec"), ns)   # noqa: S102 (our own doc)
    return ns


@unittest.skipUnless(READY, "real data / models not present")
class ContractExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_example()

    def test_documented_scenarios_give_the_documented_status_codes(self):
        card = self.ns["hazard_card"]
        expected = {("malaria", 146): 200, ("flood", 536): 200, ("heatwave", 536): 200,
                    ("dengue", 146): 501, ("chikungunya", 146): 501, ("malaria", 327): 404, ("tsunami", 146): 422}
        for (hazard, code), want in expected.items():
            got, body = card(hazard, code)
            self.assertEqual(got, want, (hazard, code, body))
            json.dumps(body, allow_nan=False)

    def test_probability_is_shown_only_where_the_contract_says(self):
        card = self.ns["hazard_card"]
        self.assertIsNotNone(card("malaria", 146)[1]["probability"])
        for hazard in ("flood", "heatwave"):
            body = card(hazard, 536)[1]
            self.assertIsNone(body["probability"], hazard)
            self.assertIsNone(body["label"], hazard)
            self.assertIsNotNone(body["raw_score_for_ranking_only"])
            self.assertIn("Withheld", body["note"])
        self.assertIsNone(card("malaria", 146)[1]["risk_band"])

    def test_non_ok_bodies_carry_a_reason_and_no_prediction(self):
        card = self.ns["hazard_card"]
        for hazard, code in (("dengue", 146), ("malaria", 327), ("tsunami", 146)):
            status, body = card(hazard, code)
            self.assertNotEqual(status, 200)
            self.assertIsNone(body["prediction"])
            self.assertTrue(body["message"])

    def test_listing_in_the_example_matches_the_contract_text(self):
        self.assertIs(self.ns["list_hazards"]()["heatwave"]["calibrated_probability_available"], False)
        text = CONTRACT.read_text(encoding="utf-8")
        for hazard, model in (("heatwave", "heatwave-ann-mlp-v2"), ("flood", "flood-ann-mlp-v2"),
                              ("malaria", "malaria-ann-mlp-v2")):
            self.assertIn(model, text)
            self.assertEqual(self.ns["hazard_card"](hazard, 146 if hazard == "malaria" else 536)[1]["model_version"], model)


if __name__ == "__main__":
    unittest.main()
