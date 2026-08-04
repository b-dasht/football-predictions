"""Reusable evaluation metrics, shared by every model-training stage."""

import json

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
    r2_score,
)

from src.config import MODELS_PATH, RESULTS_LOG_PATH

RESULT_LABELS = ["Away", "Draw", "Home"]  # index order matches TargetResult's 0/1/2 encoding


def classification_metrics(y_true, y_pred, y_proba=None, labels: list[int] | None = None,
                            label_names: list[str] | None = None) -> dict:
    """Accuracy, per-class precision/recall/F1, confusion matrix, and log loss (per §15).

    Per-class metrics matter here specifically because Draw is a minority
    class (see docs/EDA_FINDINGS.md) - an aggregate accuracy alone can
    look good while draw recall is near zero.

    labels/label_names default to all three classes, but can be restricted
    (e.g. labels=[0, 2], label_names=["Away", "Home"]) to score a subset -
    used for a Home/Away-only comparison that excludes Draw entirely.
    """
    if labels is None:
        labels = [0, 1, 2]
    if label_names is None:
        label_names = RESULT_LABELS

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_per_class": dict(zip(label_names, precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0))),
        "recall_per_class": dict(zip(label_names, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0))),
        "f1_per_class": dict(zip(label_names, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0))),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels),
            index=[f"True_{label}" for label in label_names],
            columns=[f"Pred_{label}" for label in label_names],
        ),
    }
    if y_proba is not None:
        metrics["log_loss"] = log_loss(y_true, y_proba, labels=labels)
    return metrics


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R² (per §15)."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def log_classification_metrics(name: str, metrics: dict, label_names: list[str] | None = None) -> None:
    if label_names is None:
        label_names = RESULT_LABELS
    logger.info(f"[{name}] accuracy={metrics['accuracy']:.3f}")
    for label in label_names:
        logger.info(
            f"[{name}] {label}: precision={metrics['precision_per_class'][label]:.3f} "
            f"recall={metrics['recall_per_class'][label]:.3f} f1={metrics['f1_per_class'][label]:.3f}"
        )
    if "log_loss" in metrics:
        logger.info(f"[{name}] log_loss={metrics['log_loss']:.3f}")
    logger.info(f"[{name}] confusion matrix:\n{metrics['confusion_matrix']}")


def log_regression_metrics(name: str, metrics: dict) -> None:
    logger.info(f"[{name}] MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.3f}")


def log_result(model_name: str, task: str, framing: str, metrics: dict) -> None:
    """Append one row to reports/results_log.csv - the persistent history of
    every model trained. Read-modify-write rather than a plain append,
    because classification and regression rows have different columns
    (accuracy/log_loss/recall_* vs mae/rmse/r2); appending a differently-
    shaped row to a CSV via pandas' append mode would silently misalign
    columns, so the whole file is rewritten each time instead - cheap at
    this scale (at most a few dozen rows).
    """
    row = {
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "task": task,
        "framing": framing,
    }
    if task == "classification":
        row["accuracy"] = metrics["accuracy"]
        row["log_loss"] = metrics.get("log_loss")
        for label, value in metrics["recall_per_class"].items():
            row[f"recall_{label}"] = value
    else:
        row["mae"] = metrics["mae"]
        row["rmse"] = metrics["rmse"]
        row["r2"] = metrics["r2"]

    RESULTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if RESULTS_LOG_PATH.exists():
        combined = pd.concat([pd.read_csv(RESULTS_LOG_PATH), new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(RESULTS_LOG_PATH, index=False)
    logger.info(f"Logged result: {model_name} ({task}/{framing}) -> {RESULTS_LOG_PATH}")


def save_model_with_metadata(model, name: str, task: str, framing: str, metrics: dict) -> None:
    """Save a fitted pipeline via joblib, plus a companion JSON with its
    hyperparameters and full evaluation metrics, and log a summary row to
    the results history (per copilot-instructions.md #17: store model,
    pipeline, parameters, and evaluation results together, not just the
    model itself).

    model may be None for a baseline that isn't an actual trained model
    (e.g. the Bet365 odds baseline) - only the JSON (metrics, no
    hyperparameters) and the results log row are written in that case, so
    it still shows up in every comparison built from models/*.json, just
    without a .pkl or any parameters.
    """
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    parameters = {}
    if model is not None:
        model_path = MODELS_PATH / f"{name}.pkl"
        joblib.dump(model, model_path)
        # Only the final estimator's hyperparameters are worth recording -
        # the imputer/scaler steps are fixed preprocessing, not tuned.
        parameters = model.named_steps["model"].get_params()

    serializable_metrics = {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in metrics.items()}
    metadata = {"task": task, "framing": framing, "parameters": parameters, "metrics": serializable_metrics}

    metadata_path = MODELS_PATH / f"{name}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)  # default=str covers any numpy scalar types

    log_result(name, task, framing, metrics)
    saved = f"{metadata_path}" if model is None else f"{model_path} and {metadata_path}"
    logger.info(f"Saved {saved}")
