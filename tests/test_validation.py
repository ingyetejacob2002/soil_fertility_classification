import unittest

from src.data_preprocessing import clean_dataset, load_raw, split_data
from src.train_model import build_schema
from src.validation import validate_input

TRAIN, _ = split_data(clean_dataset(load_raw())[0])
SCHEMA = build_schema(TRAIN)
GOOD = {f: float(TRAIN[f].median()) for f in SCHEMA["features"]}


class TestValidation(unittest.TestCase):
    def test_valid(self):
        r = validate_input(GOOD, SCHEMA)
        self.assertTrue(r.ok); self.assertEqual(r.errors, []); self.assertEqual(len(r.values), 12)

    def test_numeric_strings_accepted(self):
        r = validate_input({k: str(v) for k, v in GOOD.items()}, SCHEMA)
        self.assertTrue(r.ok)

    def test_missing_none_and_blank_and_absent(self):
        for bad in (None, "", "  "):
            r = validate_input({**GOOD, "N": bad}, SCHEMA)
            self.assertFalse(r.ok); self.assertIn("missing", r.errors[0])
        absent = {k: v for k, v in GOOD.items() if k != "pH"}
        self.assertFalse(validate_input(absent, SCHEMA).ok)

    def test_invalid_types(self):
        for bad in ("abc", True, [1], float("nan"), float("inf")):
            self.assertFalse(validate_input({**GOOD, "K": bad}, SCHEMA).ok, repr(bad))

    def test_negative(self):
        r = validate_input({**GOOD, "P": -1}, SCHEMA)
        self.assertFalse(r.ok); self.assertIn("negative", r.errors[0])

    def test_out_of_range(self):
        self.assertFalse(validate_input({**GOOD, "pH": 15}, SCHEMA).ok)
        self.assertFalse(validate_input({**GOOD, "N": 1e6}, SCHEMA).ok)

    def test_outside_training_range_warns_but_ok(self):
        hi = SCHEMA["feature_info"]["N"]["observed_max"] * 1.5
        r = validate_input({**GOOD, "N": hi}, SCHEMA)
        self.assertTrue(r.ok); self.assertEqual(len(r.warnings), 1)

    def test_unexpected_key_and_bad_container(self):
        self.assertFalse(validate_input({**GOOD, "yield": 3}, SCHEMA).ok)
        self.assertFalse(validate_input("not a dict", SCHEMA).ok)

    def test_no_values_returned_when_invalid(self):
        self.assertEqual(validate_input({**GOOD, "N": None}, SCHEMA).values, {})


if __name__ == "__main__":
    unittest.main()
