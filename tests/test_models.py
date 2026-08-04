import numpy as np
import pandas as pd

from src.models import (
    build_baseline_classifier,
    build_baseline_regressor,
    build_random_forest_classifier,
    build_random_forest_regressor,
    build_xgboost_classifier,
    build_xgboost_regressor,
    odds_baseline_binary_predictions,
    odds_baseline_predictions,
    train_and_save_classifier,
    train_and_save_regressor,
)


def test_odds_baseline_predicts_highest_implied_probability():
    df = pd.DataFrame({
        "ImpliedProbHome": [0.6, 0.2, 0.40],
        "ImpliedProbDraw": [0.25, 0.3, 0.35],
        "ImpliedProbAway": [0.15, 0.5, 0.25],
    })
    predictions, proba = odds_baseline_predictions(df)

    assert list(predictions) == [2, 0, 2]  # Home, Away, Home
    assert proba.shape == (3, 3)


def test_build_baseline_classifier_has_imputer_scaler_and_model_steps():
    pipeline = build_baseline_classifier()
    assert list(pipeline.named_steps.keys()) == ["imputer", "scaler", "model"]


def test_build_baseline_classifier_handles_missing_values():
    """The whole point of including SimpleImputer: fitting must not error
    on NaN inputs, which is exactly what the real feature table contains."""
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [2.0, 3.0, np.nan, 5.0]})
    y = [0, 1, 0, 1]
    pipeline = build_baseline_classifier()
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == 4


def test_build_baseline_regressor_has_imputer_scaler_and_model_steps():
    pipeline = build_baseline_regressor()
    assert list(pipeline.named_steps.keys()) == ["imputer", "scaler", "model"]


def test_odds_baseline_binary_predictions_excludes_draw():
    # ImpliedProbDraw is deliberately present but never read - Home/Away
    # scores don't sum to 1 on their own; Draw's share is excluded, not
    # redistributed.
    df = pd.DataFrame({
        "ImpliedProbHome": [0.6, 0.2, 0.34],
        "ImpliedProbDraw": [0.25, 0.3, 0.33],
        "ImpliedProbAway": [0.15, 0.5, 0.33],
    })
    predictions, proba = odds_baseline_binary_predictions(df)

    assert list(predictions) == [2, 0, 2]  # Home, Away, Home
    assert np.allclose(proba.sum(axis=1), 1.0)  # renormalized to a real 2-class distribution
    assert proba.shape == (3, 2)


def test_build_random_forest_classifier_has_imputer_and_model_steps_but_no_scaler():
    """Tree-based models are scale-invariant - a StandardScaler step would
    be a no-op, so it's deliberately absent (unlike the baseline pipelines)."""
    pipeline = build_random_forest_classifier()
    assert list(pipeline.named_steps.keys()) == ["imputer", "model"]


def test_build_random_forest_regressor_has_imputer_and_model_steps_but_no_scaler():
    pipeline = build_random_forest_regressor()
    assert list(pipeline.named_steps.keys()) == ["imputer", "model"]


def test_build_xgboost_classifier_has_no_imputer_and_no_scaler():
    """Unlike every other model type, XGBoost handles NaN natively - no
    imputer step at all, not even for consistency with the others."""
    pipeline = build_xgboost_classifier()
    assert list(pipeline.named_steps.keys()) == ["model"]


def test_build_xgboost_regressor_has_no_imputer_and_no_scaler():
    pipeline = build_xgboost_regressor()
    assert list(pipeline.named_steps.keys()) == ["model"]


def test_build_xgboost_classifier_handles_missing_values_without_an_imputer():
    """The whole point of skipping SimpleImputer: XGBoost must not error on
    NaN inputs on its own, unlike a bare LogisticRegression/RandomForest."""
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [2.0, 3.0, np.nan, 5.0]})
    y = [0, 1, 0, 1]
    pipeline = build_xgboost_classifier()
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == 4


def _toy_classification_data() -> pd.DataFrame:
    # All 3 classes present - matches classification_metrics' default
    # 3-class framing (labels=[0, 1, 2]), which train_and_save_classifier
    # uses unless told otherwise.
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "TargetResult": [0, 2, 1, 0, 2, 1],
    })


def test_train_and_save_classifier_fits_predicts_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluation.MODELS_PATH", tmp_path)
    monkeypatch.setattr("src.evaluation.RESULTS_LOG_PATH", tmp_path / "results_log.csv")

    data = _toy_classification_data()
    model = train_and_save_classifier(build_baseline_classifier(), "test_classifier", data, data, ["a", "b"])

    assert hasattr(model, "predict")
    assert (tmp_path / "test_classifier.pkl").exists()
    assert (tmp_path / "test_classifier.json").exists()
    assert (tmp_path / "results_log.csv").exists()


def test_train_and_save_regressor_fits_predicts_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluation.MODELS_PATH", tmp_path)
    monkeypatch.setattr("src.evaluation.RESULTS_LOG_PATH", tmp_path / "results_log.csv")

    data = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [4.0, 3.0, 2.0, 1.0],
        "TargetGoalDifference": [1, -1, 2, -2],
    })
    model = train_and_save_regressor(build_baseline_regressor(), "test_regressor", data, data, ["a", "b"])

    assert hasattr(model, "predict")
    assert (tmp_path / "test_regressor.pkl").exists()
    assert (tmp_path / "test_regressor.json").exists()
