"""Streamlit startup test. Skipped automatically when streamlit is not installed."""
import unittest
from pathlib import Path

from src import config

try:
    from streamlit.testing.v1 import AppTest
    HAVE_ST = True
except Exception:  # noqa: BLE001
    HAVE_ST = False

APP = str(Path(config.ROOT) / "app.py")
READY = config.MODEL_PATH.exists() and config.SCHEMA_PATH.exists()


@unittest.skipUnless(HAVE_ST, "streamlit is not installed")
@unittest.skipUnless(READY, "model artifacts not found; train the model first")
class TestApp(unittest.TestCase):
    def test_starts_without_exception(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        self.assertEqual(len(at.exception), 0)
        self.assertIn("Soil Fertility", at.sidebar.title[0].value)

    def test_input_page_blocks_empty_submission(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        at.sidebar.radio[0].set_value("Soil Sample Input").run()
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertGreater(len(at.error), 0)

    def test_valid_submission_shows_result(self):
        import pandas as pd
        row = pd.read_csv(config.TEST_PATH).iloc[0]
        at = AppTest.from_file(APP, default_timeout=60).run()
        at.sidebar.radio[0].set_value("Soil Sample Input").run()
        for ni in at.number_input:
            ni.set_value(float(row[ni.key[3:]]))
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.sidebar.radio[0].value, "Classification Results")


if __name__ == "__main__":
    unittest.main()
