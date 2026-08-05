import numpy as np
import pandas as pd

from src.evaluation import classification_metrics, log_result, regression_metrics


def test_classification_metrics_default_three_class():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 1, 2, 0]
    metrics = classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert set(metrics["precision_per_class"].keys()) == {"Away", "Draw", "Home"}
    assert metrics["confusion_matrix"].shape == (3, 3)


def test_classification_metrics_restricted_labels_excludes_draw():
    """The Home/Away-only framing: only 2 classes scored, Draw absent entirely."""
    y_true = [0, 2, 0, 2]
    y_pred = [0, 2, 2, 2]
    metrics = classification_metrics(y_true, y_pred, labels=[0, 2], label_names=["Away", "Home"])

    assert metrics["accuracy"] == 0.75
    assert set(metrics["precision_per_class"].keys()) == {"Away", "Home"}
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert list(metrics["confusion_matrix"].columns) == ["Pred_Away", "Pred_Home"]


def test_regression_metrics_perfect_prediction():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    metrics = regression_metrics(y_true, y_pred)

    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1.0
    assert metrics["outcome_accuracy"] == 1.0


def test_regression_metrics_outcome_accuracy_rounds_before_taking_sign():
    """0.4 rounds to 0 (Draw), matching a true goal difference of 0 - the
    conversion goes through round() first so a near-zero prediction isn't
    unfairly scored against a draw just for missing exact 0.0."""
    y_true = [0, 3, -2]
    y_pred = [0.4, 2.6, -0.4]  # rounds to 0, 3, 0 -> outcomes Draw, Home, Draw
    metrics = regression_metrics(y_true, y_pred)

    # True outcomes: Draw, Home, Away. Predicted (rounded): Draw, Home, Draw.
    # 2 of 3 match.
    assert metrics["outcome_accuracy"] == 2 / 3


def test_classification_metrics_two_class_includes_auroc():
    y_true = [0, 2, 0, 2, 0, 2]
    y_proba = np.array([
        [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9],
    ])
    y_pred = [0, 2, 0, 2, 0, 2]
    metrics = classification_metrics(y_true, y_pred, y_proba, labels=[0, 2], label_names=["Away", "Home"])

    assert "auroc" in metrics
    assert metrics["auroc"] == 1.0  # perfectly separated by proba here


def test_classification_metrics_three_class_has_no_auroc():
    """AUROC is deliberately binary-only - a 3-class one-vs-rest macro
    average would be a less actionable number than accuracy/log loss
    already give, so it's not computed for the 3-class framing."""
    y_true = [0, 1, 2, 2]
    y_pred = [0, 1, 2, 0]
    y_proba = np.array([
        [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.6, 0.2, 0.2],
    ])
    metrics = classification_metrics(y_true, y_pred, y_proba)

    assert "auroc" not in metrics


def _regression_metrics_dict(mae: float) -> dict:
    return {"mae": mae, "rmse": mae * 1.2, "r2": 0.2, "outcome_accuracy": 0.5}


def test_log_result_skips_duplicate_of_last_logged_row(tmp_path, monkeypatch):
    """Retraining a model that didn't actually change (e.g. an unrelated
    retrain of every model, or a tuning round scoped to other model types)
    must not add a redundant duplicate row - otherwise results_log.csv
    fills up with noise and plot_tuning_progress's run-count becomes
    meaningless."""
    monkeypatch.setattr("src.evaluation.RESULTS_LOG_PATH", tmp_path / "results_log.csv")

    log_result("random_forest_regressor", "regression", "regression", _regression_metrics_dict(1.25))
    log_result("random_forest_regressor", "regression", "regression", _regression_metrics_dict(1.25))

    logged = pd.read_csv(tmp_path / "results_log.csv")
    assert len(logged) == 1


def test_log_result_logs_a_genuine_change(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluation.RESULTS_LOG_PATH", tmp_path / "results_log.csv")

    log_result("random_forest_regressor", "regression", "regression", _regression_metrics_dict(1.25))
    log_result("random_forest_regressor", "regression", "regression", _regression_metrics_dict(1.10))

    logged = pd.read_csv(tmp_path / "results_log.csv")
    assert len(logged) == 2
    assert list(logged["mae"]) == [1.25, 1.10]


def test_log_result_dedup_is_scoped_to_the_same_model_name(tmp_path, monkeypatch):
    """Two different models with identical metrics must both still get
    logged - the duplicate check only compares a model against its own history."""
    monkeypatch.setattr("src.evaluation.RESULTS_LOG_PATH", tmp_path / "results_log.csv")

    log_result("random_forest_regressor", "regression", "regression", _regression_metrics_dict(1.25))
    log_result("xgboost_regressor", "regression", "regression", _regression_metrics_dict(1.25))

    logged = pd.read_csv(tmp_path / "results_log.csv")
    assert len(logged) == 2
