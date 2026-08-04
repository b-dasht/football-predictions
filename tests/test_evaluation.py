from src.evaluation import classification_metrics, regression_metrics


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
