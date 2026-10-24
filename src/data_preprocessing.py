"""Dataset loading, auditing, cleaning, splitting and the preprocessing pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


class DatasetError(Exception):
    """Raised when the dataset is missing, unreadable or has the wrong schema."""


def load_raw(path: Path | str = config.RAW_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset file '{path.name}' not found. Place it in data/raw/.")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Could not read '{path.name}': {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in config.FEATURES + [config.TARGET] if c not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {missing}")
    df = df[config.FEATURES + [config.TARGET]].copy()
    for col in config.FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[config.TARGET] = pd.to_numeric(df[config.TARGET], errors="coerce")
    unknown = set(df[config.TARGET].dropna().unique()) - set(config.CLASS_NAMES)
    if unknown:
        raise DatasetError(
            f"Unexpected target codes {sorted(unknown)}; expected {sorted(config.CLASS_NAMES)}.")
    return df


def flag_implausible(df: pd.DataFrame) -> pd.Series:
    """Return {row index: reason} for rows outside config.PLAUSIBILITY."""
    reasons: dict = {}
    for col, (lo, hi) in config.PLAUSIBILITY.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        for idx, val in bad[col].items():
            reasons.setdefault(idx, []).append(f"{col}={val} outside [{lo}, {hi}]")
    return pd.Series({i: "; ".join(r) for i, r in reasons.items()}, dtype=object)


def audit_dataset(df: pd.DataFrame) -> str:
    lines = ["DATA AUDIT (raw dataset)", "=" * 60,
             f"Rows: {len(df)}   Columns: {df.shape[1]}",
             f"Missing values (total): {int(df.isna().sum().sum())}",
             f"Exact duplicate rows: {int(df.duplicated().sum())}", "",
             "Class distribution:"]
    counts = df[config.TARGET].value_counts(dropna=False).sort_index()
    for code, n in counts.items():
        name = config.CLASS_NAMES.get(code, "missing")
        lines.append(f"  {code} ({name}): {n}  ({n / len(df):.1%})")
    lines += ["", "Feature summary:", df[config.FEATURES].describe().T.round(3).to_string(), "",
              "Rows outside the plausibility bounds (candidate data errors):"]
    flagged = flag_implausible(df)
    for idx, reason in flagged.items():
        lines.append(f"  raw row {idx}: {reason}")
    if flagged.empty:
        lines.append("  none")
    return "\n".join(lines)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove rows with missing target, exact duplicates and implausible values.

    No statistic is learned from the data here, so cleaning cannot leak
    information from the test set. Missing feature values (none in this
    dataset) are handled by the imputer inside the model pipeline.
    """
    log: dict = {"rows_in": int(len(df))}
    out = df.dropna(subset=[config.TARGET])
    log["dropped_missing_target"] = int(len(df) - len(out))
    n = len(out)
    out = out.drop_duplicates()
    log["dropped_duplicates"] = int(n - len(out))
    flagged = flag_implausible(out)
    log["dropped_implausible"] = int(len(flagged))
    log["implausible_rows"] = {str(k): v for k, v in flagged.items()}
    out = out.drop(index=flagged.index).reset_index(drop=True)
    out[config.TARGET] = out[config.TARGET].astype(int)
    log["rows_out"] = int(len(out))
    return out, log


def split_data(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
               seed: int = config.RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=test_size, random_state=seed,
                                   stratify=df[config.TARGET])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Median imputation (+ optional standard scaling).

    Used inside a scikit-learn Pipeline, so it is fitted on training data only
    and applied identically at prediction time. Columns are selected by name.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([("numeric", Pipeline(steps), config.FEATURES)], remainder="drop")


def run_preprocessing(raw_path: Path | str = config.RAW_PATH,
                      out_dir: Path | str = config.PROCESSED_DIR,
                      report_path: Path | str = config.AUDIT_REPORT_PATH) -> dict:
    out_dir, report_path = Path(out_dir), Path(report_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    raw = load_raw(raw_path)
    audit = audit_dataset(raw)
    clean, log = clean_dataset(raw)
    train, test = split_data(clean)
    clean.to_csv(out_dir / "soil_fertility_clean.csv", index=False)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    log.update({
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE,
        "stratified": True, "features": config.FEATURES,
        "procedure": [
            "load and validate schema", "coerce numeric columns",
            "drop rows with missing target", "drop exact duplicates",
            "drop rows outside plausibility bounds", "stratified 80/20 split",
            "imputation/scaling are fitted inside the model pipeline on training data only"],
    })
    (out_dir / "preprocessing_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    report_path.write_text(audit + "\n\nCLEANING LOG\n" + json.dumps(log, indent=2), encoding="utf-8")
    return log


if __name__ == "__main__":
    try:
        result = run_preprocessing()
    except DatasetError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(f"Rows in: {result['rows_in']}  rows out: {result['rows_out']}  "
          f"(removed implausible: {result['dropped_implausible']})")
    print(f"Train: {result['train_rows']}  Test: {result['test_rows']}")
    print("Audit written to reports/data_audit.txt")
