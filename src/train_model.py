"""Train, compare and persist the soil fertility classifier.

Model selection uses ONLY the training data (5-fold stratified cross-validation,
macro-F1). The held-out test set is used for reporting, never for selection.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data_preprocessing import build_preprocessor

SEED = config.RANDOM_STATE


def get_candidates() -> dict:
    """name -> dict(factory, scale, diagnostic). Diagnostic models are reported but never selected."""
    return {
        "Baseline (most frequent class)": dict(factory=lambda: DummyClassifier(strategy="most_frequent"), scale=False, diagnostic=True),
        "Decision stump (depth 1)": dict(factory=lambda: DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=SEED), scale=False, diagnostic=True),
        "Decision Tree": dict(factory=lambda: DecisionTreeClassifier(class_weight="balanced", random_state=SEED), scale=False, diagnostic=False),
        "Random Forest": dict(factory=lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=1), scale=False, diagnostic=False),
        "Support Vector Machine (RBF)": dict(factory=lambda: SVC(class_weight="balanced", probability=True, random_state=SEED), scale=True, diagnostic=False),
        "K-Nearest Neighbors": dict(factory=lambda: KNeighborsClassifier(), scale=True, diagnostic=False),
        "Naive Bayes (Gaussian)": dict(factory=lambda: GaussianNB(), scale=False, diagnostic=False),
        "Gradient Boosting": dict(factory=lambda: GradientBoostingClassifier(random_state=SEED), scale=False, diagnostic=False),
        "Neural Network (MLP)": dict(factory=lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=SEED), scale=True, diagnostic=False),
    }


def make_pipeline(estimator, scale: bool) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(scale)), ("classifier", estimator)])


def split_xy(df: pd.DataFrame):
    return df[config.FEATURES], df[config.TARGET]


def compare_models(train: pd.DataFrame, test: pd.DataFrame, candidates: dict | None = None,
                   cv_folds: int = config.CV_FOLDS) -> tuple[pd.DataFrame, dict]:
    candidates = candidates or get_candidates()
    X_tr, y_tr = split_xy(train)
    X_te, y_te = split_xy(test)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, spec in candidates.items():
        pipe = make_pipeline(spec["factory"](), spec["scale"])
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=1,
                                scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"})
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "model": name, "diagnostic_only": spec["diagnostic"],
            "cv_accuracy_mean": scores["test_accuracy"].mean(), "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_f1_macro_mean": scores["test_f1_macro"].mean(), "cv_f1_macro_std": scores["test_f1_macro"].std(),
            "train_accuracy": accuracy_score(y_tr, pipe.predict(X_tr)),
            "test_accuracy": accuracy_score(y_te, pred),
            "test_precision_macro": precision_score(y_te, pred, average="macro", zero_division=0),
            "test_recall_macro": recall_score(y_te, pred, average="macro", zero_division=0),
            "test_f1_macro": f1_score(y_te, pred, average="macro", zero_division=0),
        })
        fitted[name] = pipe
    return pd.DataFrame(rows), fitted


def select_best(results: pd.DataFrame) -> str:
    """Highest cross-validated macro-F1 (training data only); ties broken by CV accuracy."""
    eligible = results[~results["diagnostic_only"]]
    ranked = eligible.sort_values(["cv_f1_macro_mean", "cv_accuracy_mean"], ascending=False)
    return str(ranked.iloc[0]["model"])


def build_schema(train: pd.DataFrame) -> dict:
    """Feature schema shipped with the model: order, units, observed ranges, input limits."""
    feats = {}
    for f in config.FEATURES:
        lo, hi = float(train[f].min()), float(train[f].max())
        hard = config.PH_HARD_LIMITS if f == "pH" else (0.0, round(hi * config.HARD_LIMIT_MULTIPLIER, 4))
        feats[f] = {"label": config.FEATURE_LABELS[f], "unit": config.UNITS[f],
                    "observed_min": lo, "observed_max": hi,
                    "hard_min": hard[0], "hard_max": hard[1], "step": config.STEPS[f]}
    medians = {config.CLASS_NAMES[c]: {f: float(v) for f, v in g[config.FEATURES].median().items()}
               for c, g in train.groupby(config.TARGET)}
    return {"features": config.FEATURES, "feature_info": feats,
            "class_names": {str(k): v for k, v in config.CLASS_NAMES.items()},
            "class_medians_train": medians}


def main() -> None:
    if not (config.TRAIN_PATH.exists() and config.TEST_PATH.exists()):
        raise SystemExit("ERROR: processed data not found. Run: python -m src.data_preprocessing")
    train, test = pd.read_csv(config.TRAIN_PATH), pd.read_csv(config.TEST_PATH)
    print(f"Training on {len(train)} rows, comparing candidates ({config.CV_FOLDS}-fold CV)...")
    results, fitted = compare_models(train, test)
    best = select_best(results)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    results.to_csv(config.COMPARISON_PATH, index=False)
    joblib.dump(fitted[best], config.MODEL_PATH)
    config.SCHEMA_PATH.write_text(json.dumps(build_schema(train), indent=2), encoding="utf-8")
    log = json.loads(config.PREPROCESS_LOG_PATH.read_text(encoding="utf-8")) if config.PREPROCESS_LOG_PATH.exists() else {}
    dist = lambda d: {config.CLASS_NAMES[int(k)]: int(v) for k, v in d[config.TARGET].value_counts().sort_index().items()}  # noqa: E731
    metadata = {
        "selected_algorithm": best,
        "selection_rule": "highest mean 5-fold cross-validated macro-F1 on the training set (diagnostic models excluded)",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scikit_learn_version": sklearn.__version__, "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "dataset_file": "dataset1.csv (Kaggle 'Soil Fertility Dataset')",
        "rows_raw": log.get("rows_in"), "rows_clean": log.get("rows_out", len(train) + len(test)),
        "rows_removed_implausible": log.get("dropped_implausible"),
        "n_train": len(train), "n_test": len(test),
        "class_distribution": {"train": dist(train), "test": dist(test)},
        "features": config.FEATURES,
        "cv_f1_macro": float(results.loc[results.model == best, "cv_f1_macro_mean"].iloc[0]),
        "cv_accuracy": float(results.loc[results.model == best, "cv_accuracy_mean"].iloc[0]),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    show = results.drop(columns=["diagnostic_only"]).round(3)
    print(show.to_string(index=False))
    print(f"\nSelected model: {best}\nSaved: models/{Path(config.MODEL_PATH).name}")


if __name__ == "__main__":
    main()
