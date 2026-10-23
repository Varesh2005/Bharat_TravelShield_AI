"""Tests for ml/hazard_training.py (protocol correctness, not model quality).

Uses SYNTHETIC data and the quick grid. The key guarantees tested:
  * the test years never influence the ANN, calibration or threshold;
  * baselines use exactly the same rows as the ANN;
  * training is reproducible; artifacts are written and loadable.

    python -m unittest ml.test_hazard_training -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml import hazard_datasets as hd
from ml import hazard_training as ht
from ml.test_hazard_datasets import make_synthetic_data


class MathTests(unittest.TestCase):
    def test_platt_is_monotonic_and_identity_when_one_class(self):
        a, b, fitted = ht.fit_platt(np.array([.1, .2, .8, .9]), np.array([0, 0, 1, 1]))
        self.assertTrue(fitted)
        p = ht.apply_platt(np.array([.1, .5, .9]), a, b)
        self.assertTrue(np.all(np.diff(p) > 0))
        self.assertEqual(ht.fit_platt(np.array([.1, .2]), np.array([0, 0])), (1.0, 0.0, False))

    def test_forward_calibration_check_detects_base_rate_shift_using_validation_only(self):
        rng = np.random.default_rng(0)
        n = 600
        years = np.array([2020] * n + [2021] * n)
        score = np.concatenate([rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
        # stable world: outcome follows the score in both years -> check passes
        y_stable = (rng.uniform(0, 1, 2 * n) < score).astype(int)
        ok = ht.forward_calibration_check(years, y_stable, score)
        self.assertTrue(ok["display_probability"], ok)
        self.assertEqual((ok["fit_target_years"], ok["eval_target_year"]), ([2020], 2021))
        self.assertFalse(ok["uses_test_years"])
        # base-rate shift: 2021 has far fewer events at the same scores -> check fails
        y_shift = y_stable.copy()
        y_shift[n:] = (rng.uniform(0, 1, n) < 0.15 * score[n:]).astype(int)
        bad = ht.forward_calibration_check(years, y_shift, score)
        self.assertFalse(bad["display_probability"], bad)
        self.assertGreater(bad["ece"], ht.ECE_ACCEPTABLE)

    def test_forward_calibration_check_needs_two_years_and_two_classes(self):
        one = ht.forward_calibration_check(np.array([2020, 2020]), np.array([0, 1]), np.array([.2, .8]))
        self.assertFalse(one["display_probability"])
        self.assertIsNone(one["ece"])
        single = ht.forward_calibration_check(np.array([2020, 2020, 2021]), np.array([0, 0, 1]), np.array([.2, .3, .8]))
        self.assertFalse(single["display_probability"])

    def test_ece_bounds_and_perfect_case(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(ht.ece(y, y.astype(float))[0], 0.0)
        self.assertAlmostEqual(ht.ece(y, 1 - y.astype(float))[0], 1.0)

    def test_threshold_uses_only_the_given_data(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([.1, .2, .33, .62, .7, .9])
        self.assertTrue(.33 < ht.best_threshold(y, p) <= .62)

    def test_oversampling_balances_only_what_it_is_given(self):
        X, y = np.arange(20).reshape(10, 2).astype(float), np.array([0] * 8 + [1] * 2)
        X2, y2 = ht._oversample(X, y, 1)
        self.assertEqual((y2 == 1).sum(), (y2 == 0).sum())
        self.assertEqual(len(y), 10)                      # input untouched

    def test_metrics_report_more_than_accuracy(self):
        y = np.array([0, 0, 1, 1, 1, 0])
        s = np.array([.1, .4, .35, .8, .7, .2])
        m = ht.metrics(y, s, s >= .5, s)
        for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "macro_f1", "brier", "log_loss", "ece", "confusion", "per_class"):
            self.assertIn(k, m)
        self.assertNotIn("accuracy", m)
        self.assertEqual(sum(m["confusion"].values()), 6)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        make_synthetic_data(cls.tmp)
        cls.frame, cls.meta = hd.build_heatwave(cls.tmp)[0], {}
        cls.spec = hd.SPECS["heatwave"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, frame, tag):
        out = self.tmp / tag
        r = ht.train_hazard("heatwave", spec=self.spec, frame=frame, meta=self.meta,
                            model_dir=out / "models", result_dir=out / "res", quick=True, plots=False)
        return r, out

    def test_ann_is_primary_and_baselines_share_rows(self):
        r, out = self._run(self.frame, "a")
        main = r["primary_split"]
        self.assertEqual(set(main["baselines"]), {"Persistence (last year repeats)", "Climatology (district training rate)",
                                                  "Logistic Regression", "Random Forest"})
        self.assertEqual(main["ann"]["test"]["n"], main["rows"]["test"])
        for b in main["baselines"].values():
            self.assertEqual(b["test"]["n"], main["ann"]["test"]["n"])      # same test rows for every model
            self.assertEqual(b["test"]["prevalence"], main["ann"]["test"]["prevalence"])
        self.assertIn("without_weather_T", r["ablation"])
        self.assertTrue(r["held_out_states"]["held_out_states"])
        self.assertTrue(set(r["held_out_states"]["held_out_states"]).isdisjoint(
            set(self.frame[self.frame["split"] == "train"]["state_name"]) - set(r["held_out_states"]["held_out_states"])))

    def test_test_labels_never_influence_the_model(self):
        shuffled = self.frame.copy()
        rng = np.random.default_rng(0)
        is_test = shuffled["split"] == "test"
        shuffled.loc[is_test, self.spec.target] = rng.permutation(shuffled.loc[is_test, self.spec.target].to_numpy())
        r1, o1 = self._run(self.frame, "orig")
        r2, o2 = self._run(shuffled, "shuf")
        m1, m2 = r1["primary_split"], r2["primary_split"]
        self.assertEqual(m1["ann_selected_config"]["label"], m2["ann_selected_config"]["label"])
        self.assertEqual(m1["ann"]["threshold_from_validation"], m2["ann"]["threshold_from_validation"])
        self.assertEqual(m1["ann"]["platt"], m2["ann"]["platt"])
        self.assertEqual(m1["ann"]["validation"], m2["ann"]["validation"])
        b1 = joblib.load(o1 / "models" / "heatwave" / "ann_bundle.joblib")
        b2 = joblib.load(o2 / "models" / "heatwave" / "ann_bundle.joblib")
        for w1, w2 in zip(b1["net"].coefs_, b2["net"].coefs_):
            np.testing.assert_array_equal(w1, w2)
        self.assertNotEqual(m1["ann"]["test"]["confusion"], m2["ann"]["test"]["confusion"])   # test score does change
        # the probability-display decision must not move when only test labels change
        c1 = json.loads((o1 / "models" / "heatwave" / "model_card.json").read_text(encoding="utf-8"))["calibration"]
        c2 = json.loads((o2 / "models" / "heatwave" / "model_card.json").read_text(encoding="utf-8"))["calibration"]
        self.assertEqual(c1["validation_forward_check"], c2["validation_forward_check"])
        self.assertEqual(c1["display_probability"], c2["display_probability"])
        self.assertNotEqual(c1["test_ece_calibrated"], c2["test_ece_calibrated"])             # information only
        self.assertFalse(c1["validation_forward_check"]["uses_test_years"])

    def test_training_is_reproducible(self):
        r1, _ = self._run(self.frame, "r1")
        r2, _ = self._run(self.frame, "r2")
        self.assertEqual(r1["primary_split"]["ann"], r2["primary_split"]["ann"])
        self.assertEqual(r1["seed_stability"], r2["seed_stability"])

    def test_artifacts_and_model_card(self):
        r, out = self._run(self.frame, "art")
        card = json.loads((out / "models" / "heatwave" / "model_card.json").read_text(encoding="utf-8"))
        arch = card["architecture"]
        self.assertEqual(arch["input"], len(self.spec.features))
        self.assertEqual(arch["output"], "1 unit, logistic sigmoid")
        self.assertEqual(arch["seed"], ht.SEED)
        self.assertIsNone(card["risk_bands"])
        self.assertEqual(card["features"], self.spec.features)
        self.assertNotIn(str(self.tmp), json.dumps(card))
        bundle = joblib.load(out / "models" / "heatwave" / "ann_bundle.joblib")
        self.assertEqual(bundle["features"], self.spec.features)
        self.assertEqual(bundle["net"].hidden_layer_sizes, tuple(arch["hidden"]))
        res = json.loads((out / "res" / "heatwave" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["model_version"], "heatwave-ann-mlp-v2")

    def test_scaler_is_fitted_on_train_only(self):
        r, out = self._run(self.frame, "sc")
        bundle = joblib.load(out / "models" / "heatwave" / "ann_bundle.joblib")
        train = self.frame[self.frame["split"] == "train"]
        np.testing.assert_allclose(bundle["scaler"].mean_, train[self.spec.features].mean().to_numpy(), rtol=1e-9)

    def test_climatology_uses_training_rates_only(self):
        train = self.frame[self.frame["split"] == "train"]
        test = self.frame[self.frame["split"] == "test"]
        s = ht.climatology_scores(train, test, self.spec.target)
        expected = train.groupby("census_district_code")[self.spec.target].mean()
        d = test["census_district_code"].iloc[0]
        self.assertAlmostEqual(s[0], expected[d])
        unseen = test.assign(census_district_code=99999)
        self.assertAlmostEqual(ht.climatology_scores(train, unseen, self.spec.target)[0], train[self.spec.target].mean())


if __name__ == "__main__":
    unittest.main()
