import numpy as np
import pandas as pd

from src.visualisation import (
    plot_metric_comparison,
    plot_per_class_metric_comparison,
    plot_regression_distribution,
    plot_roc_curves,
    write_confusion_matrix_table,
)


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
