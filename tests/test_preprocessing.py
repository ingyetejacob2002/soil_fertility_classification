import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_preprocessing import (DatasetError, build_preprocessor, clean_dataset, load_raw,
                                    run_preprocessing, split_data)


def synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(1, 10, size=(n, len(config.FEATURES))), columns=config.FEATURES)
    df["pH"] = rng.uniform(5, 9, n)
    df[config.TARGET] = rng.choice([0, 1, 2], size=n, p=[.45, .45, .10])
    return df


class TestLoading(unittest.TestCase):
    def test_loads_real_dataset(self):
        df = load_raw(config.RAW_PATH)
        self.assertEqual(list(df.columns), config.FEATURES + [config.TARGET])
        self.assertEqual(len(df), 880)

    def test_missing_file(self):
        with self.assertRaises(DatasetError):
            load_raw("does_not_exist.csv")

    def test_wrong_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.csv"
            p.write_text("a,b\n1,2\n")
            with self.assertRaises(DatasetError):
                load_raw(p)

    def test_unknown_target_code(self):
        with tempfile.TemporaryDirectory() as d:
            df = synthetic(10)
            df[config.TARGET] = 7
            p = Path(d) / "x.csv"
            df.to_csv(p, index=False)
            with self.assertRaises(DatasetError):
                load_raw(p)


class TestCleaning(unittest.TestCase):
    def test_removes_implausible_and_duplicates(self):
        df = synthetic(50)
        bad = df.iloc[[0]].copy(); bad["pH"] = 0.9
        dup = df.iloc[[5]].copy()
        out, log = clean_dataset(pd.concat([df, bad, dup], ignore_index=True))
        self.assertEqual(log["dropped_implausible"], 1)
        self.assertEqual(log["dropped_duplicates"], 1)
        self.assertTrue((out["pH"] >= 3).all())

    def test_real_data_removes_four_rows(self):
        clean, log = clean_dataset(load_raw())
        self.assertEqual(log["dropped_implausible"], 4)
        self.assertEqual(len(clean), 876)


class TestSplitAndPipeline(unittest.TestCase):
    def test_split_stratified_and_complete(self):
        df = synthetic(400)
        tr, te = split_data(df)
        self.assertEqual(len(tr) + len(te), len(df))
        for c in (0, 1, 2):
            self.assertAlmostEqual((tr[config.TARGET] == c).mean(), (te[config.TARGET] == c).mean(), delta=0.03)

    def test_scaler_fitted_on_training_only(self):
        df = synthetic(300)
        tr, te = split_data(df)
        pre = build_preprocessor(True).fit(tr[config.FEATURES])
        scaler = pre.named_transformers_["numeric"].named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, tr[config.FEATURES].mean().values)
        self.assertFalse(np.allclose(scaler.mean_, df[config.FEATURES].mean().values))

    def test_missing_is_imputed_with_median_not_zero(self):
        df = synthetic(100)
        pre = build_preprocessor(False).fit(df[config.FEATURES])
        row = df[config.FEATURES].iloc[[0]].copy(); row.loc[row.index[0], "K"] = np.nan
        out = pre.transform(row)
        self.assertAlmostEqual(out[0, config.FEATURES.index("K")], df["K"].median())

    def test_run_preprocessing_writes_files(self):
        with tempfile.TemporaryDirectory() as d:
            log = run_preprocessing(config.RAW_PATH, Path(d) / "proc", Path(d) / "audit.txt")
            for name in ("train.csv", "test.csv", "soil_fertility_clean.csv", "preprocessing_log.json"):
                self.assertTrue((Path(d) / "proc" / name).exists(), name)
            self.assertEqual(log["train_rows"] + log["test_rows"], log["rows_out"])


if __name__ == "__main__":
    unittest.main()
