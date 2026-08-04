import numpy as np

from src.pytorch_models import TorchMLPClassifier, TorchMLPRegressor


def test_torch_mlp_classifier_fits_and_predicts():
    rng = np.random.default_rng(42)
    X = rng.random((40, 3))
    y = np.tile([0, 1, 2], 14)[:40]

    model = TorchMLPClassifier(hidden_size=8, epochs=20, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == 40
    assert set(predictions).issubset({0, 1, 2})


def test_torch_mlp_classifier_predict_proba_sums_to_one():
    rng = np.random.default_rng(42)
    X = rng.random((30, 3))
    y = np.tile([0, 1, 2], 10)

    model = TorchMLPClassifier(hidden_size=8, epochs=20, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (30, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_torch_mlp_classifier_supports_non_contiguous_labels():
    """Like every scikit-learn classifier except XGBoost, this should
    handle the {0, 2} (Away, Home) 2-class encoding with no remapping."""
    rng = np.random.default_rng(42)
    X = rng.random((20, 2))
    y = np.tile([0, 2], 10)

    model = TorchMLPClassifier(hidden_size=8, epochs=20, random_state=42)
    model.fit(X, y)

    assert list(model.classes_) == [0, 2]
    assert set(model.predict(X)).issubset({0, 2})


def test_torch_mlp_classifier_get_params_via_base_estimator():
    """Confirms scikit-learn's BaseEstimator machinery works here - needed
    for save_model_with_metadata's model.get_params() call."""
    model = TorchMLPClassifier(hidden_size=32, epochs=100, lr=0.01, random_state=7)
    params = model.get_params()

    assert params == {"hidden_size": 32, "epochs": 100, "lr": 0.01, "random_state": 7}


def test_torch_mlp_regressor_fits_and_predicts():
    rng = np.random.default_rng(42)
    X = rng.random((40, 3))
    y = rng.random(40) * 10 - 5

    model = TorchMLPRegressor(hidden_size=8, epochs=20, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (40,)
    assert np.isfinite(predictions).all()
