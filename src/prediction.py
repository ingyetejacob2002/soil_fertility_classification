"""Loading saved artifacts and producing predictions with the SAME pipeline used in training."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src import config


class ArtifactError(Exception):
    """Raised with a user-friendly message when model artifacts are missing or unusable."""


def load_artifacts(model_path: Path | str = config.MODEL_PATH,
                   schema_path: Path | str = config.SCHEMA_PATH,
                   metadata_path: Path | str = config.METADATA_PATH) -> dict:
    model_path, schema_path, metadata_path = Path(model_path), Path(schema_path), Path(metadata_path)
    hint = " Run: python -m src.data_preprocessing  then  python -m src.train_model"
    if not model_path.exists():
        raise ArtifactError("The trained model file was not found." + hint)
    if not schema_path.exists():
        raise ArtifactError("The feature schema file was not found." + hint)
    try:
        model = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError("The trained model file is corrupted or incompatible with the installed "
                            "scikit-learn version. Re-train the model." + hint) from exc
    if not hasattr(model, "predict") or not hasattr(model, "feature_names_in_"):
        raise ArtifactError("The model file does not contain a valid fitted pipeline." + hint)
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["features"], schema["feature_info"], schema["class_names"]
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError("The feature schema file is corrupted." + hint) from exc
    if list(model.feature_names_in_) != list(schema["features"]):
        raise ArtifactError("The model and the feature schema do not match (different features or order)." + hint)
    metadata = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            metadata = None
    return {"model": model, "schema": schema, "metadata": metadata}


def class_name(schema: dict, code) -> str:
    return schema["class_names"].get(str(int(code)), str(code))


def predict_one(model, schema: dict, values: dict) -> dict:
    """Predict a single validated sample. `values` must contain every feature."""
    missing = [f for f in schema["features"] if f not in values]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    frame = pd.DataFrame([[values[f] for f in schema["features"]]], columns=schema["features"])
    code = int(model.predict(frame)[0])
    result = {"code": code, "label": class_name(schema, code), "probabilities": None}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)[0]
        result["probabilities"] = {class_name(schema, c): float(p) for c, p in zip(model.classes_, proba)}
    return result


def predict_dataframe(model, schema: dict, frame: pd.DataFrame) -> list[int]:
    """Predict many rows (used for testing on held-out data)."""
    return [int(c) for c in model.predict(frame[schema["features"]])]
