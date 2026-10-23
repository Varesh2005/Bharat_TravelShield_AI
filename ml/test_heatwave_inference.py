import unittest
from pathlib import Path

import joblib
import pandas as pd

from ml.heatwave_inference import (
    APPROVED_PREDICTORS,
    MODEL_PATH,
    SCALER_PATH,
    predict_heatwave,
    validate_features,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "ml" / "data" / "heatwave" / "heatwave_t_to_t1_dataset.parquet"


class HeatwaveInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = pd.read_parquet(DATASET_PATH)
        cls.row = cls.dataset.iloc[0]
        cls.features = {
            name: float(cls.row[name])
            for name in APPROVED_PREDICTORS
        }
        cls.model = joblib.load(MODEL_PATH)
        cls.scaler = joblib.load(SCALER_PATH)

    def test_feature_schema_contains_exactly_14_predictors(self):
        self.assertEqual(len(APPROVED_PREDICTORS), 14)
        self.assertEqual(len(set(APPROVED_PREDICTORS)), 14)

    def test_valid_features_are_accepted(self):
        validated = validate_features(self.features)
        self.assertEqual(list(validated), list(APPROVED_PREDICTORS))

    def test_prediction_matches_saved_model(self):
        result = predict_heatwave(self.features)

        frame = pd.DataFrame(
            [[self.features[name] for name in APPROVED_PREDICTORS]],
            columns=APPROVED_PREDICTORS,
        )
        scaled = self.scaler.transform(frame)

        expected_class = int(self.model.predict(scaled)[0])
        positive_index = list(self.model.classes_).index(1)
        expected_score = float(
            self.model.predict_proba(scaled)[0, positive_index]
        )

        self.assertEqual(result["predicted_class"], expected_class)
        self.assertEqual(
            result["raw_positive_score"],
            round(expected_score, 6),
        )
        self.assertEqual(result["classification_threshold"], 0.5)
        self.assertIn(result["predicted_class"], (0, 1))
        self.assertEqual(
            result["predicted_label"],
            "heatwave_occurrence"
            if expected_class == 1
            else "no_heatwave_occurrence",
        )

    def test_missing_feature_is_rejected(self):
        features = dict(self.features)
        features.pop(APPROVED_PREDICTORS[0])

        with self.assertRaisesRegex(ValueError, "missing features"):
            validate_features(features)

    def test_unexpected_feature_is_rejected(self):
        features = dict(self.features)
        features["target_has_heatwave_T1"] = 1

        with self.assertRaisesRegex(ValueError, "unexpected features"):
            validate_features(features)

    def test_non_finite_value_is_rejected(self):
        features = dict(self.features)
        features["latitude"] = float("nan")

        with self.assertRaisesRegex(ValueError, "must be finite"):
            validate_features(features)

    def test_non_numeric_value_is_rejected(self):
        features = dict(self.features)
        features["latitude"] = "Mumbai"

        with self.assertRaisesRegex(ValueError, "numeric value"):
            validate_features(features)

    def test_inconsistent_heatwave_indicator_is_rejected(self):
        features = dict(self.features)
        features["has_heatwave_T"] = 1.0 - features["has_heatwave_T"]

        with self.assertRaisesRegex(ValueError, "must agree"):
            validate_features(features)

    def test_inconsistent_temperature_range_is_rejected(self):
        features = dict(self.features)
        features["temp_range_c"] += 1.0

        with self.assertRaisesRegex(ValueError, "temp_range_c is inconsistent"):
            validate_features(features)

    def test_uncalibrated_score_warning_is_present(self):
        result = predict_heatwave(self.features)

        self.assertIn("Uncalibrated", result["score_interpretation"])
        self.assertTrue(result["limitations"])


if __name__ == "__main__":
    unittest.main(verbosity=2)