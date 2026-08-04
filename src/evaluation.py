"""Reusable evaluation metrics, shared by every model-training stage."""

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
