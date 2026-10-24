"""Central configuration: paths, feature schema, class labels and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "dataset1.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "soil_fertility_clean.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
PREPROCESS_LOG_PATH = PROCESSED_DIR / "preprocessing_log.json"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "soil_fertility_model.joblib"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR = ROOT / "reports"
AUDIT_REPORT_PATH = REPORTS_DIR / "data_audit.txt"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
EVAL_REPORT_PATH = REPORTS_DIR / "model_evaluation.txt"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_PNG_PATH = REPORTS_DIR / "confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Feature order used for training AND prediction.
FEATURES = ["N", "P", "K", "pH", "EC", "OC", "S", "Zn", "Fe", "Cu", "Mn", "B"]
TARGET = "Output"

# Integer codes come from the dataset. The names Low/Medium/High follow the
# published studies that use this dataset; the dataset file itself does not
# document the mapping (see README, "Dataset limitations").
CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}

FEATURE_LABELS = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "pH": "Soil pH", "EC": "Electrical conductivity (EC)",
    "OC": "Organic carbon (OC)", "S": "Sulphur (S)", "Zn": "Zinc (Zn)",
    "Fe": "Iron (Fe)", "Cu": "Copper (Cu)", "Mn": "Manganese (Mn)",
    "B": "Boron (B)",
}
# Units as reported in secondary literature describing this dataset (N, P, K,
# EC, OC). Micronutrient units are assumed to be mg/kg; the file does not state units.
UNITS = {
    "N": "mg/kg", "P": "mg/kg", "K": "mg/kg", "pH": "pH units",
    "EC": "dS/m", "OC": "%", "S": "mg/kg", "Zn": "mg/kg", "Fe": "mg/kg",
    "Cu": "mg/kg", "Mn": "mg/kg", "B": "mg/kg",
}
STEPS = {"N": 1.0, "P": 0.1, "K": 1.0, "pH": 0.01, "EC": 0.01, "OC": 0.01,
         "S": 0.1, "Zn": 0.01, "Fe": 0.01, "Cu": 0.01, "Mn": 0.01, "B": 0.01}
FEATURE_GROUPS = {
    "Soil reaction and salinity": ["pH", "EC"],
    "Organic matter and macronutrients": ["OC", "N", "P", "K"],
    "Secondary nutrient and micronutrients": ["S", "Zn", "Fe", "Cu", "Mn", "B"],
}

# Data-quality filter used ONLY when cleaning the dataset: rows outside these
# bounds are physically implausible or isolated extreme values (see audit).
PLAUSIBILITY = {"pH": (3.0, 12.0), "OC": (0.0, 10.0), "Zn": (0.0, 10.0), "Fe": (0.0, 40.0)}

# Input guard: hard upper limit = this multiple of the largest value seen in training.
HARD_LIMIT_MULTIPLIER = 3.0
PH_HARD_LIMITS = (0.0, 14.0)
