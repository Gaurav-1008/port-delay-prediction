import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from port_delay import (
    FEATURES, PORTS, RAW_INPUTS, feature_frame, predict_one, simulate, temporal_split,
)

ROOT = Path(__file__).resolve().parents[1]


class PortDelayTests(unittest.TestCase):
    def test_reproducible_simulation_and_ground_truth(self):
        first = simulate(80, warmup=0, missing_rate=0)
        pd.testing.assert_frame_equal(first, simulate(80, warmup=0, missing_rate=0))
        self.assertTrue(first.arrival_time.is_monotonic_increasing)
        self.assertTrue(first.visit_id.is_unique)
        self.assertTrue((first.service_end_time > first.berth_start_time).all())
        expected_wait = (first.berth_start_time - first.arrival_time).dt.total_seconds() / 3600
        np.testing.assert_allclose(first.waiting_hours, expected_wait, atol=1e-8)
        np.testing.assert_array_equal(first.is_delayed, expected_wait > 6)

    def test_snapshots_only_use_prior_visits(self):
        data = simulate(100, warmup=0, missing_rate=0)
        for _, row in data.iterrows():
            previous = data[(data.port_id == row.port_id) & (data.arrival_time < row.arrival_time)]
            waiting = int((previous.berth_start_time > row.arrival_time).sum())
            occupied = int(((previous.berth_start_time <= row.arrival_time) &
                            (previous.service_end_time > row.arrival_time)).sum())
            recent = int((previous.arrival_time >= row.arrival_time - pd.Timedelta(hours=6)).sum())
            self.assertEqual(row.queue_length, waiting)
            self.assertEqual(row.busy_berths, occupied)
            self.assertEqual(row.arrivals_past_6h, recent)
            completed = previous[previous.service_end_time <= row.arrival_time]
            expected = ((completed.service_end_time - completed.berth_start_time).dt.total_seconds() / 3600).mean()
            if completed.empty:
                expected = PORTS[row.port_id][1]
            self.assertAlmostEqual(row.past_avg_service_hours, expected, places=6)
            self.assertLessEqual(occupied, row.operational_berths)

    def test_labels_resolved_before_training_cutoffs(self):
        data = simulate()
        train, validation, test, details = temporal_split(data)
        self.assertTrue((train.label_available_time < pd.Timestamp(details["validation_start"])).all())
        self.assertTrue((validation.label_available_time < pd.Timestamp(details["test_start"])).all())
        self.assertLess(train.arrival_time.max(), validation.arrival_time.min())
        self.assertLess(validation.arrival_time.max(), test.arrival_time.min())
        self.assertEqual(len(test), 100)
        delayed = data[data.is_delayed == 1]
        self.assertTrue((delayed.label_available_time > delayed.arrival_time + pd.Timedelta(hours=6)).all())

    def test_input_validation_and_outcome_independence(self):
        row = simulate(1, warmup=0, missing_rate=0).iloc[0]
        valid = {key: float(row[key]) for key in RAW_INPUTS}
        features = feature_frame(pd.DataFrame([valid]))
        with_outcome = {**valid, "waiting_hours": 9999, "is_delayed": 1, "berth_start_time": "2099-01-01"}
        pd.testing.assert_frame_equal(features, feature_frame(pd.DataFrame([with_outcome])))
        for changed in [
            {"operational_berths": 0}, {"queue_length": -1}, {"queue_length": 1.5},
            {"busy_berths": valid["operational_berths"] + 1}, {"weather_severity": 3},
            {"past_avg_service_hours": -2}, {"queue_length": True},
            {"arrivals_past_6h": float("inf")}, {"queue_length": None},
        ]:
            with self.subTest(changed=changed), self.assertRaises((ValueError, TypeError)):
                feature_frame(pd.DataFrame([{**valid, **changed}]))
        with self.assertRaises(ValueError):
            feature_frame(pd.DataFrame([{key: value for key, value in valid.items() if key != "queue_length"}]))

    def test_saved_model_explanations_and_reload(self):
        artifact = joblib.load(ROOT / "results/model.joblib")
        probabilities = []
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model.joblib"
            joblib.dump(artifact, target)
            restored = joblib.load(target)
            for name in ["low_risk", "high_risk"]:
                observation = json.loads((ROOT / f"examples/{name}.json").read_text())
                result = predict_one(restored, observation)
                expected = float(artifact["pipeline"].predict_proba(feature_frame(pd.DataFrame([observation])))[0, 1])
                self.assertAlmostEqual(result["delay_probability"], expected)
                log_odds = result["reference_log_odds"] + sum(
                    item["contribution_log_odds"] for item in result["factor_contributions"]
                )
                self.assertAlmostEqual(1 / (1 + np.exp(-log_odds)), expected)
                delayed = result["prediction"] == "delay_over_6_hours"
                self.assertAlmostEqual(result["confidence_score"], expected if delayed else 1 - expected)
                self.assertEqual({item["feature"] for item in result["factor_contributions"]}, set(FEATURES))
                probabilities.append(expected)
        self.assertLess(probabilities[0], probabilities[1])

    def test_missing_observations_are_imputed_and_flagged(self):
        artifact = joblib.load(ROOT / "results/model.joblib")
        observation = json.loads((ROOT / "examples/low_risk.json").read_text())
        observation.update(weather_severity=None, past_avg_service_hours=None)
        result = predict_one(artifact, observation)
        self.assertTrue(0 <= result["delay_probability"] <= 1)
        self.assertEqual(set(result["missing_fields_imputed"]), {"weather_severity", "past_avg_service_hours"})
        imputed_features = {item["feature"] for item in result["factor_contributions"] if item["imputed"]}
        self.assertEqual(imputed_features, {"weather_severity", "past_avg_service_hours"})


if __name__ == "__main__":
    unittest.main()
