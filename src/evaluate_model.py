"""Evaluate the saved model on the held-out test set and write reports."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             precision_score, recall_score)

from src import config  # noqa: E402


def evaluate(model, train: pd.DataFrame, test: pd.DataFrame, metadata: dict | None = None) -> dict:
    labels = sorted(config.CLASS_NAMES)
    names = [config.CLASS_NAMES[c] for c in labels]
    X_te, y_te = test[config.FEATURES], test[config.TARGET]
    X_tr, y_tr = train[config.FEATURES], train[config.TARGET]
    pred = model.predict(X_te)
    p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=labels, zero_division=0)
    metrics = {
        "selected_algorithm": (metadata or {}).get("selected_algorithm", type(model[-1]).__name__),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "train_accuracy": float(accuracy_score(y_tr, model.predict(X_tr))),
        "test_accuracy": float(accuracy_score(y_te, pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "test_precision_macro": float(precision_score(y_te, pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_te, pred, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "test_precision_weighted": float(precision_score(y_te, pred, average="weighted", zero_division=0)),
        "test_recall_weighted": float(recall_score(y_te, pred, average="weighted", zero_division=0)),
        "test_f1_weighted": float(f1_score(y_te, pred, average="weighted", zero_division=0)),
        "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, n in enumerate(names)},
        "class_labels": names,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=labels).tolist(),
        "classification_report": classification_report(y_te, pred, labels=labels, target_names=names, zero_division=0),
        "class_distribution": {
            "train": {n: int((y_tr == c).sum()) for c, n in zip(labels, names)},
            "test": {n: int((y_te == c).sum()) for c, n in zip(labels, names)}},
    }
    return metrics


def write_report(m: dict, path=config.EVAL_REPORT_PATH) -> None:
    cm = pd.DataFrame(m["confusion_matrix"], index=[f"true {n}" for n in m["class_labels"]],
                      columns=[f"pred {n}" for n in m["class_labels"]])
    text = f"""MODEL EVALUATION REPORT
{'=' * 60}
Selected algorithm : {m['selected_algorithm']}
Training rows      : {m['n_train']}   Test rows: {m['n_test']}
Class distribution : train {m['class_distribution']['train']}  test {m['class_distribution']['test']}

TRAINING performance (optimistic; the model has seen these rows)
  accuracy: {m['train_accuracy']:.4f}

HELD-OUT TEST performance (rows never used for training or model selection)
  accuracy          : {m['test_accuracy']:.4f}
  balanced accuracy : {m['test_balanced_accuracy']:.4f}
  precision (macro) : {m['test_precision_macro']:.4f}   (weighted) {m['test_precision_weighted']:.4f}
  recall    (macro) : {m['test_recall_macro']:.4f}   (weighted) {m['test_recall_weighted']:.4f}
  F1        (macro) : {m['test_f1_macro']:.4f}   (weighted) {m['test_f1_weighted']:.4f}

Confusion matrix (rows = true class, columns = predicted class)
{cm.to_string()}

Classification report
{m['classification_report']}
"""
    Path_ = Path(path)
    Path_.parent.mkdir(parents=True, exist_ok=True)
    Path_.write_text(text, encoding="utf-8")


def save_confusion_png(m: dict, path=config.CONFUSION_PNG_PATH) -> None:
    cm, names = np.array(m["confusion_matrix"]), m["class_labels"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (held-out test set)")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    for p in (config.MODEL_PATH, config.TRAIN_PATH, config.TEST_PATH):
        if not p.exists():
            raise SystemExit(f"ERROR: {p.name} not found. Run preprocessing and training first.")
    model = joblib.load(config.MODEL_PATH)
    meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8")) if config.METADATA_PATH.exists() else {}
    m = evaluate(model, pd.read_csv(config.TRAIN_PATH), pd.read_csv(config.TEST_PATH), meta)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")
    write_report(m)
    save_confusion_png(m)
    print(config.EVAL_REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
