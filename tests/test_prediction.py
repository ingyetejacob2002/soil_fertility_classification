import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.prediction import ArtifactError, load_artifacts, predict_dataframe, predict_one
from src.train_model import build_schema, compare_models, get_candidates, make_pipeline, select_best
from src.validation import validate_input

HAVE_ARTIFACTS = config.MODEL_PATH.exists() and config.SCHEMA_PATH.exists() and config.TEST_PATH.exists()
SKIP = "Run preprocessing and training first (see README)."


class TestTrainingAndPersistence(unittest.TestCase):
    def test_train_save_load_roundtrip(self):
        train, test = pd.read_csv(config.TRAIN_PATH), pd.read_csv(config.TEST_PATH)
        cands = {k: v for k, v in get_candidates().items() if k in ("Decision Tree", "Naive Bayes (Gaussian)")}
        results, fitted = compare_models(train, test, cands, cv_folds=3)
        best = select_best(results)
        self.assertIn(best, cands)
        with tempfile.TemporaryDirectory() as d:
            m, s = Path(d) / "m.joblib", Path(d) / "s.json"
            joblib.dump(fitted[best], m)
            s.write_text(json.dumps(build_schema(train)))
            art = load_artifacts(m, s, Path(d) / "none.json")
            self.assertIsNone(art["metadata"])
            a = fitted[best].predict(test[config.FEATURES]); b = art["model"].predict(test[config.FEATURES])
            self.assertTrue((a == b).all())


@unittest.skipUnless(HAVE_ARTIFACTS, SKIP)
class TestSavedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.art = load_artifacts()
        cls.test = pd.read_csv(config.TEST_PATH)

    def test_feature_schema_consistency(self):
        self.assertEqual(list(self.art["model"].feature_names_in_), config.FEATURES)
        self.assertEqual(self.art["schema"]["features"], config.FEATURES)

    def test_predictions_are_valid_classes_on_heldout_rows(self):
        preds = predict_dataframe(self.art["model"], self.art["schema"], self.test)
        self.assertEqual(len(preds), len(self.test))
        self.assertTrue(set(preds) <= set(config.CLASS_NAMES))

    def test_beats_majority_baseline(self):
        preds = pd.Series(predict_dataframe(self.art["model"], self.art["schema"], self.test))
        y = self.test[config.TARGET]
        self.assertGreater((preds == y).mean(), y.value_counts(normalize=True).max())

    def test_single_prediction_matches_batch_and_workflow(self):
        schema = self.art["schema"]
        batch = predict_dataframe(self.art["model"], schema, self.test.head(15))
        for i in range(15):
            row = {f: float(self.test.loc[i, f]) for f in schema["features"]}
            check = validate_input(row, schema)
            self.assertTrue(check.ok)
            out = predict_one(self.art["model"], schema, check.values)
            self.assertEqual(out["code"], batch[i])
            self.assertIn(out["label"], config.CLASS_NAMES.values())
            if out["probabilities"]:
                self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=6)

    def test_predict_one_rejects_missing_feature(self):
        with self.assertRaises(ValueError):
            predict_one(self.art["model"], self.art["schema"], {"N": 1.0})


@unittest.skipUnless(HAVE_ARTIFACTS, SKIP)
class TestArtifactErrors(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_missing_model(self):
        with self.assertRaises(ArtifactError):
            load_artifacts(self.tmp / "no.joblib", config.SCHEMA_PATH)

    def test_missing_schema(self):
        with self.assertRaises(ArtifactError):
            load_artifacts(config.MODEL_PATH, self.tmp / "no.json")

    def test_corrupted_model(self):
        bad = self.tmp / "bad.joblib"; bad.write_bytes(b"not a model")
        with self.assertRaises(ArtifactError):
            load_artifacts(bad, config.SCHEMA_PATH)

    def test_wrong_object_in_model_file(self):
        p = self.tmp / "obj.joblib"; joblib.dump({"a": 1}, p)
        with self.assertRaises(ArtifactError):
            load_artifacts(p, config.SCHEMA_PATH)

    def test_corrupted_schema(self):
        bad = self.tmp / "bad.json"; bad.write_text("{ nope")
        with self.assertRaises(ArtifactError):
            load_artifacts(config.MODEL_PATH, bad)

    def test_schema_model_mismatch(self):
        schema = json.loads(config.SCHEMA_PATH.read_text())
        schema["features"] = list(reversed(schema["features"]))
        p = self.tmp / "s.json"; p.write_text(json.dumps(schema))
        with self.assertRaises(ArtifactError):
            load_artifacts(config.MODEL_PATH, p)


if __name__ == "__main__":
    unittest.main()
