import numpy as np
import pandas as pd

from src.models import (
    build_linear_regression_regressor,
    build_logistic_regression_classifier,
    build_neural_network_classifier,
    build_neural_network_regressor,
    build_pytorch_classifier,
    build_pytorch_regressor,
    build_random_forest_classifier,
    build_random_forest_regressor,
    build_svm_classifier,
    build_svm_regressor,
    build_xgboost_classifier,
    build_xgboost_regressor,
    load_tuned_params,
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


def test_build_logistic_regression_classifier_has_imputer_scaler_and_model_steps():
    pipeline = build_logistic_regression_classifier()
    assert list(pipeline.named_steps.keys()) == ["imputer", "scaler", "model"]


def test_build_logistic_regression_classifier_handles_missing_values():
    """The whole point of including SimpleImputer: fitting must not error
    on NaN inputs, which is exactly what the real feature table contains."""
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [2.0, 3.0, np.nan, 5.0]})
    y = [0, 1, 0, 1]
    pipeline = build_logistic_regression_classifier()
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == 4


def test_build_linear_regression_regressor_has_imputer_scaler_and_model_steps():
    pipeline = build_linear_regression_regressor()
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


def test_build_svm_classifier_has_imputer_scaler_and_model_steps():
    pipeline = build_svm_classifier()
    assert list(pipeline.named_steps.keys()) == ["imputer", "scaler", "model"]


def test_build_svm_regressor_has_imputer_scaler_and_model_steps():
    pipeline = build_svm_regressor()
    assert list(pipeline.named_steps.keys()) == ["imputer", "scaler", "model"]


def test_build_svm_classifier_supports_non_contiguous_binary_labels():
    """Unlike XGBoost, scikit-learn's SVC should handle our {0, 2}
    (Away, Home) 2-class encoding natively, with no remapping needed.

    Needs enough rows per class for CalibratedClassifierCV's default
    5-fold cross-validation to have something to split - a handful of
    rows per class isn't enough, unlike the plain-fit tests elsewhere.
    """
    X = pd.DataFrame({"a": list(range(20)), "b": list(range(20, 0, -1))}, dtype=float)
    y = [0, 2] * 10
    pipeline = build_svm_classifier()
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert set(predictions).issubset({0, 2})


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
    model = train_and_save_classifier(build_logistic_regression_classifier(), "test_classifier", data, data, ["a", "b"])

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
    model = train_and_save_regressor(build_linear_regression_regressor(), "test_regressor", data, data, ["a", "b"])

    assert hasattr(model, "predict")
    assert (tmp_path / "test_regressor.pkl").exists()
    assert (tmp_path / "test_regressor.json").exists()


def test_build_neural_network_classifier_has_imputer_scaler_and_model_steps():
    pipeline = build_neural_network_classifier()
    assert list(pipeline.named_steps.keys()) == ["imputer", "scaler", "model"]


def test_build_neural_network_regressor_has_imputer_scaler_and_model_steps():
    pipeline = build_neural_network_regressor()
    assert list(pipeline.named_steps.keys()) == ["imputer", "scaler", "model"]


def test_build_neural_network_classifier_uses_smaller_hidden_layer_than_default():
    """Deliberately smaller than scikit-learn's default (100 neurons),
    given our modest dataset size - confirms the choice wasn't dropped
    accidentally in a future refactor."""
    pipeline = build_neural_network_classifier()
    assert pipeline.named_steps["model"].hidden_layer_sizes == (64,)


def test_build_neural_network_classifier_supports_non_contiguous_binary_labels():
    """Unlike XGBoost, MLPClassifier should handle our {0, 2} (Away, Home)
    2-class encoding natively, with no remapping needed."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    y = [0, 2, 0, 2]
    pipeline = build_neural_network_classifier()
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert set(predictions).issubset({0, 2})


def test_build_pytorch_classifier_has_imputer_scaler_and_model_steps():
    pipeline = build_pytorch_classifier()
    assert list(pipeline.named_steps.keys()) == ["imputer", "scaler", "model"]


def test_build_pytorch_regressor_has_imputer_scaler_and_model_steps():
    pipeline = build_pytorch_regressor()
    assert list(pipeline.named_steps.keys()) == ["imputer", "scaler", "model"]


def test_load_tuned_params_returns_empty_dict_when_file_missing(tmp_path, monkeypatch):
    """Safe no-op before hyperparameter_tuning.py has ever been run -
    every train_*_models() function calls this unconditionally."""
    monkeypatch.setattr("src.models.MODELS_PATH", tmp_path)
    assert load_tuned_params("random_forest_classifier") == {}


def test_load_tuned_params_returns_empty_dict_for_untuned_model_name(tmp_path, monkeypatch):
    monkeypatch.setattr("src.models.MODELS_PATH", tmp_path)
    (tmp_path / "tuned_hyperparameters.json").write_text('{"random_forest_classifier": {"model__n_estimators": 300}}')

    assert load_tuned_params("svm_classifier") == {}


def test_load_tuned_params_returns_saved_params_and_applies_via_set_params(tmp_path, monkeypatch):
    """Keys must already be in Pipeline.set_params' "stepname__param"
    format - confirms a real pipeline accepts them with no reparsing."""
    monkeypatch.setattr("src.models.MODELS_PATH", tmp_path)
    (tmp_path / "tuned_hyperparameters.json").write_text(
        '{"random_forest_classifier": {"model__n_estimators": 300, "model__max_depth": 10}}'
    )

    params = load_tuned_params("random_forest_classifier")
    pipeline = build_random_forest_classifier().set_params(**params)

    assert pipeline.named_steps["model"].n_estimators == 300
    assert pipeline.named_steps["model"].max_depth == 10
