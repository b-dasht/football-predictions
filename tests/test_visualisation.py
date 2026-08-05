import numpy as np
import pandas as pd

from src.visualisation import (
    _color_for,
    _load_model_results,
    plot_metric_comparison,
    plot_per_class_metric_comparison,
    plot_regression_distribution,
    plot_roc_curves,
    plot_tuning_progress,
    write_confusion_matrix_table,
)


def test_color_for_shares_family_color_across_no_odds_and_binary_variants():
    """A model's _no_odds and _binary variants must render in the same
    color as their primary sibling - they're always shown on a separate
    chart, or disambiguated by linestyle (plot_tuning_progress), so two
    variants can never collide, and sharing the hue keeps a model family
    identifiable across every chart it appears on."""
    assert _color_for("random_forest_classifier_no_odds") == _color_for("random_forest_classifier")
    assert _color_for("random_forest_classifier_binary") == _color_for("random_forest_classifier")
    assert _color_for("random_forest_classifier_no_odds") != "#52514e"  # not the gray fallback


def test_load_model_results_skips_tuned_hyperparameters_file(tmp_path, monkeypatch):
    """tuned_hyperparameters.json lives in models/ alongside per-model
    metadata but has no task/framing keys - must not crash or be mistaken
    for a model result."""
    monkeypatch.setattr("src.visualisation.MODELS_PATH", tmp_path)
    (tmp_path / "tuned_hyperparameters.json").write_text('{"random_forest_classifier": {"model__n_estimators": 300}}')
    (tmp_path / "random_forest_classifier.json").write_text(
        '{"task": "classification", "framing": "3-class", "metrics": {"accuracy": 0.5}}'
    )

    results = _load_model_results("classification", "3-class")

    assert list(results.keys()) == ["random_forest_classifier"]


def _toy_classification_results() -> dict[str, dict]:
    return {
        "model_a": {
            "accuracy": 0.55,
            "recall_per_class": {"Away": 0.5, "Draw": 0.1, "Home": 0.8},
            "confusion_matrix": pd.DataFrame(
                [[10, 2, 3], [4, 1, 5], [2, 3, 15]],
                index=["True_Away", "True_Draw", "True_Home"],
                columns=["Pred_Away", "Pred_Draw", "Pred_Home"],
            ),
        },
        "model_b": {
            "accuracy": 0.60,
            "recall_per_class": {"Away": 0.6, "Draw": 0.2, "Home": 0.7},
            "confusion_matrix": pd.DataFrame(
                [[12, 1, 2], [3, 2, 5], [1, 4, 15]],
                index=["True_Away", "True_Draw", "True_Home"],
                columns=["Pred_Away", "Pred_Draw", "Pred_Home"],
            ),
        },
    }


def test_write_confusion_matrix_table_writes_markdown_sections(tmp_path):
    results_by_framing = {"3-class": _toy_classification_results(), "2-class": {}}
    dest = tmp_path / "confusion_matrices.md"

    write_confusion_matrix_table(results_by_framing, dest)
    content = dest.read_text()

    assert "## 3-class" in content
    assert "### model_a" in content
    assert "### model_b" in content
    assert "| Actual \\ Predicted | Away | Draw | Home |" in content
    assert "2-class" not in content  # empty framing skipped entirely


def test_plot_metric_comparison_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.visualisation.FIGURES_PATH", tmp_path)
    results = _toy_classification_results()

    plot_metric_comparison(results, "accuracy", "Accuracy", "accuracy_3class.png", fmt="{:.1%}")

    assert (tmp_path / "accuracy_3class.png").exists()


def test_plot_per_class_metric_comparison_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.visualisation.FIGURES_PATH", tmp_path)
    results = _toy_classification_results()

    plot_per_class_metric_comparison(results, "recall_per_class", "Recall by Class", "recall_by_class_3class.png")

    assert (tmp_path / "recall_by_class_3class.png").exists()


def test_plot_regression_distribution_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.visualisation.FIGURES_PATH", tmp_path)
    rng = np.random.default_rng(42)
    y_true = rng.integers(-3, 4, size=50)
    predictions_by_model = {
        "model_a": (y_true, y_true + rng.normal(0, 0.5, size=50)),
        "model_b": (y_true, rng.normal(0, 1.5, size=50)),
    }

    plot_regression_distribution(predictions_by_model, "regression_distributions_with_odds.png")

    assert (tmp_path / "regression_distributions_with_odds.png").exists()


def test_plot_tuning_progress_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.visualisation.FIGURES_PATH", tmp_path)
    monkeypatch.setattr("src.visualisation.REPORTS_PATH", tmp_path)
    log = pd.DataFrame([
        {"timestamp": "2026-01-01T00:00:00", "model_name": "random_forest_classifier", "accuracy": 0.50},
        {"timestamp": "2026-01-02T00:00:00", "model_name": "random_forest_classifier", "accuracy": 0.53},
        {"timestamp": "2026-01-01T00:00:00", "model_name": "random_forest_classifier_no_odds", "accuracy": 0.48},
    ])
    log.to_csv(tmp_path / "results_log.csv", index=False)

    plot_tuning_progress(
        ["random_forest_classifier", "random_forest_classifier_no_odds"], "accuracy", "Accuracy", "tuning_progress_accuracy.png"
    )

    assert (tmp_path / "tuning_progress_accuracy.png").exists()


def test_plot_tuning_progress_skips_when_no_matching_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("src.visualisation.FIGURES_PATH", tmp_path)
    monkeypatch.setattr("src.visualisation.REPORTS_PATH", tmp_path)
    log = pd.DataFrame([{"timestamp": "2026-01-01T00:00:00", "model_name": "svm_classifier", "accuracy": 0.5}])
    log.to_csv(tmp_path / "results_log.csv", index=False)

    plot_tuning_progress(["random_forest_classifier"], "accuracy", "Accuracy", "tuning_progress_accuracy.png")

    assert not (tmp_path / "tuning_progress_accuracy.png").exists()


def test_plot_roc_curves_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.visualisation.FIGURES_PATH", tmp_path)
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=50).astype(bool)
    roc_data = {
        "model_a": (y_true, rng.random(50)),
        "bet365_odds_binary": (y_true, rng.random(50)),
    }

    plot_roc_curves(roc_data, "roc_curve_2class.png")

    assert (tmp_path / "roc_curve_2class.png").exists()
