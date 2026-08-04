"""Plotting functions for model results. Reusable across every model-comparison stage."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap

from src.config import DATA_PROCESSED_PATH, MODELS_PATH, PROJECT_ROOT
from src.evaluation import RESULT_LABELS, classification_metrics, regression_metrics
from src.models import odds_baseline_predictions
from src.utils import get_feature_columns, split_by_season

FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"

# Fixed categorical colors - one per model, never reassigned/cycled, so a
# given model keeps the same color across every chart it appears in.
COLOR_MODEL_A = "#2a78d6"  # blue
COLOR_MODEL_B = "#eb6834"  # orange
COLOR_GRID = "#e1e0d9"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_MUTED = "#898781"

# Sequential single-hue ramp (light -> dark blue) for the confusion-matrix
# heatmaps - magnitude data should never use a rainbow/multi-hue colormap.
_BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential",
    ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)


def _style_axis(ax) -> None:
    """Recessive gridlines/spines shared by every chart in this module."""
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_MUTED)


def plot_classification_comparison(
    metrics_a: dict, metrics_b: dict, name_a: str, name_b: str, save_name: str = "classification_comparison.png"
) -> None:
    """Three panels: accuracy, log loss, and per-class recall - side by side
    rather than combined onto one axis, since accuracy/log loss are on
    different scales (never mix scales on one axis)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("#fcfcfb")

    # Panel 1: accuracy
    ax = axes[0]
    ax.bar([name_a, name_b], [metrics_a["accuracy"], metrics_b["accuracy"]],
           color=[COLOR_MODEL_A, COLOR_MODEL_B], width=0.5, zorder=3)
    for i, v in enumerate([metrics_a["accuracy"], metrics_b["accuracy"]]):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", color=COLOR_TEXT_PRIMARY, fontsize=10)
    ax.set_title("Accuracy", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_ylim(0, max(metrics_a["accuracy"], metrics_b["accuracy"]) * 1.25)
    _style_axis(ax)

    # Panel 2: log loss (lower is better - noted directly, since the
    # direction isn't obvious from the bar alone)
    ax = axes[1]
    ax.bar([name_a, name_b], [metrics_a["log_loss"], metrics_b["log_loss"]],
           color=[COLOR_MODEL_A, COLOR_MODEL_B], width=0.5, zorder=3)
    for i, v in enumerate([metrics_a["log_loss"], metrics_b["log_loss"]]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", color=COLOR_TEXT_PRIMARY, fontsize=10)
    ax.set_title("Log Loss (lower is better)", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_ylim(0, max(metrics_a["log_loss"], metrics_b["log_loss"]) * 1.25)
    _style_axis(ax)

    # Panel 3: per-class recall - grouped bars, the clearest way to show
    # the draw-recall gap between models
    ax = axes[2]
    x = np.arange(len(RESULT_LABELS))
    width = 0.35
    recall_a = [metrics_a["recall_per_class"][label] for label in RESULT_LABELS]
    recall_b = [metrics_b["recall_per_class"][label] for label in RESULT_LABELS]
    ax.bar(x - width / 2, recall_a, width, label=name_a, color=COLOR_MODEL_A, zorder=3)
    ax.bar(x + width / 2, recall_b, width, label=name_b, color=COLOR_MODEL_B, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(RESULT_LABELS, color=COLOR_TEXT_PRIMARY)
    ax.set_title("Recall by Class", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)
    _style_axis(ax)

    fig.tight_layout()
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    dest = FIGURES_PATH / save_name
    fig.savefig(dest, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved {dest}")


def plot_confusion_matrices(
    metrics_a: dict, metrics_b: dict, name_a: str, name_b: str, save_name: str = "confusion_matrices.png"
) -> None:
    """Side-by-side confusion-matrix heatmaps, one hue (sequential blue), light->dark."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.patch.set_facecolor("#fcfcfb")

    for ax, metrics, name in zip(axes, [metrics_a, metrics_b], [name_a, name_b]):
        cm = metrics["confusion_matrix"]
        im = ax.imshow(cm.to_numpy(), cmap=_BLUE_SEQUENTIAL)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(RESULT_LABELS, color=COLOR_TEXT_PRIMARY)
        ax.set_yticklabels(RESULT_LABELS, color=COLOR_TEXT_PRIMARY)
        ax.set_xlabel("Predicted", color=COLOR_TEXT_MUTED)
        ax.set_ylabel("Actual", color=COLOR_TEXT_MUTED)
        ax.set_title(name, color=COLOR_TEXT_PRIMARY, fontsize=11)
        # Annotate every cell - a 3x3 confusion matrix is exactly the kind
        # of small, structured grid where per-cell labels are the point.
        vmax = cm.to_numpy().max()
        for i in range(3):
            for j in range(3):
                value = cm.iloc[i, j]
                text_color = "white" if value > vmax / 2 else COLOR_TEXT_PRIMARY
                ax.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    dest = FIGURES_PATH / save_name
    fig.savefig(dest, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved {dest}")


def generate_baseline_report() -> None:
    """Reload the saved baseline models and validation data, then produce
    every baseline comparison plot. Reruns predictions rather than
    threading metrics through from training, since the saved pipelines
    are exactly what's meant to be reusable by later stages."""
    features = pd.read_csv(DATA_PROCESSED_PATH / "features.csv")
    _train, validation, _test = split_by_season(features)
    feature_cols = get_feature_columns(features)

    classifier = joblib.load(MODELS_PATH / "baseline_logistic_regression.pkl")
    regressor = joblib.load(MODELS_PATH / "baseline_linear_regression.pkl")

    val_predictions = classifier.predict(validation[feature_cols])
    val_proba = classifier.predict_proba(validation[feature_cols])
    logreg_metrics = classification_metrics(validation["TargetResult"], val_predictions, val_proba)

    odds_predictions, odds_proba = odds_baseline_predictions(validation)
    odds_metrics = classification_metrics(validation["TargetResult"], odds_predictions, odds_proba)

    plot_classification_comparison(logreg_metrics, odds_metrics, "LogisticRegression", "Bet365Odds")
    plot_confusion_matrices(logreg_metrics, odds_metrics, "LogisticRegression", "Bet365Odds")

    val_reg_predictions = regressor.predict(validation[feature_cols])
    plot_regression_diagnostic(validation["TargetGoalDifference"], val_reg_predictions, "LinearRegression")


def plot_regression_diagnostic(y_true, y_pred, model_name: str, save_name: str = "regression_diagnostic.png") -> None:
    """Actual vs. predicted scatter with a y=x reference line - the standard
    way to see whether a regression model's errors are unbiased/patterned."""
    metrics = regression_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor("#fcfcfb")

    ax.scatter(y_true, y_pred, color=COLOR_MODEL_A, alpha=0.35, s=18, zorder=3, edgecolors="none")
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, color=COLOR_TEXT_MUTED, linewidth=1.2, linestyle="-", zorder=2, label="Perfect prediction")
    ax.set_xlabel("Actual Goal Difference", color=COLOR_TEXT_PRIMARY)
    ax.set_ylabel("Predicted Goal Difference", color=COLOR_TEXT_PRIMARY)
    ax.set_title(model_name, color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY, loc="upper left")
    ax.text(
        0.97, 0.03, f"MAE={metrics['mae']:.2f}   RMSE={metrics['rmse']:.2f}   R²={metrics['r2']:.2f}",
        transform=ax.transAxes, ha="right", va="bottom", color=COLOR_TEXT_MUTED, fontsize=9,
    )
    _style_axis(ax)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8, zorder=0)

    fig.tight_layout()
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    dest = FIGURES_PATH / save_name
    fig.savefig(dest, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved {dest}")


if __name__ == "__main__":
    generate_baseline_report()
