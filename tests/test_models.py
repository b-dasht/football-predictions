import numpy as np
import pandas as pd

from src.models import (
    build_baseline_classifier,
    build_baseline_regressor,
    odds_baseline_binary_predictions,
    odds_baseline_predictions,
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
