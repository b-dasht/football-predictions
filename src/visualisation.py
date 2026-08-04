"""Plotting functions for model results. Reads from models/*.json (the
per-model metadata saved by evaluation.save_model_with_metadata) so every
chart automatically includes every model trained so far - no hardcoded
per-model list to keep in sync as new models are added.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap

from src.config import DATA_PROCESSED_PATH, FIGURES_PATH, MODELS_PATH
from src.evaluation import regression_metrics
from src.utils import get_feature_columns, split_by_season

# Fixed color per model family - fixed order, never reassigned by rank/
# position, so a model keeps the same color on every chart it appears in.
# A model's "_binary" (2-class) variant shares its 3-class sibling's color:
# the two framings are always shown on separate charts, never mixed
# together, so there's no collision risk in reusing the hue.
MODEL_COLORS = {
    "bet365_odds": "#2a78d6",                    # slot 1: blue
    "baseline_logistic_regression": "#eb6834",   # slot 2: orange
    "random_forest_classifier": "#1baf7a",       # slot 3: aqua
    "xgboost_classifier": "#eda100",             # slot 4: yellow
    "svm_classifier": "#e87ba4",                 # slot 5: magenta
    "neural_network_classifier": "#008300",      # slot 6: green
    "baseline_linear_regression": "#eb6834",
    "random_forest_regressor": "#1baf7a",
    "xgboost_regressor": "#eda100",
    "svm_regressor": "#e87ba4",
    "neural_network_regressor": "#008300",
}
_FALLBACK_COLOR = "#52514e"  # neutral gray, for any model not yet in the fixed mapping above

COLOR_GRID = "#e1e0d9"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_MUTED = "#898781"

# Sequential single-hue ramp (light -> dark blue) for the confusion-matrix
# heatmaps - magnitude data should never use a rainbow/multi-hue colormap.
_BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential",
    ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)


def _color_for(name: str) -> str:
    family = name.removesuffix("_binary")
    return MODEL_COLORS.get(family, _FALLBACK_COLOR)


def _rotate_model_labels(ax) -> None:
    """Rotate + right-align long model-name x-tick labels so they read
    cleanly instead of overlapping - matters more as more models get added
    (rotation alone, without ha="right", still lets adjacent labels collide)."""
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
        label.set_fontsize(8)


def _style_axis(ax) -> None:
    """Recessive gridlines/spines shared by every chart in this module."""
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_MUTED)


def _load_model_results(task: str, framing: str) -> dict[str, dict]:
    """Load every models/*.json matching this task/framing, deserializing
    the confusion matrix back into a DataFrame. Returns {model_name: metrics}."""
    results = {}
    for path in sorted(MODELS_PATH.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if data["task"] != task or data["framing"] != framing:
            continue
        metrics = dict(data["metrics"])
        if "confusion_matrix" in metrics:
            metrics["confusion_matrix"] = pd.DataFrame(metrics["confusion_matrix"])
        results[path.stem] = metrics
    return results


def plot_classification_comparison(results: dict[str, dict], save_name: str) -> None:
    """Three panels: accuracy, log loss, and per-class recall, across every
    model passed in - side by side rather than combined onto one axis,
    since accuracy/log loss are on different scales (never mix scales on
    one axis)."""
    names = list(results.keys())
    colors = [_color_for(name) for name in names]
    labels = list(next(iter(results.values()))["recall_per_class"].keys())

    fig, axes = plt.subplots(1, 3, figsize=(max(13, len(names) * 2.8), 5.2))
    fig.patch.set_facecolor("#fcfcfb")

    # Panel 1: accuracy
    ax = axes[0]
    values = [results[n]["accuracy"] for n in names]
    ax.bar(names, values, color=colors, width=0.6, zorder=3)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f"{v:.1%}", ha="center", color=COLOR_TEXT_PRIMARY, fontsize=9)
    ax.set_title("Accuracy", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_ylim(0, max(values) * 1.25)
    _rotate_model_labels(ax)
    _style_axis(ax)

    # Panel 2: log loss (lower is better - noted directly, since the
    # direction isn't obvious from the bar alone)
    ax = axes[1]
    values = [results[n].get("log_loss", 0) for n in names]
    ax.bar(names, values, color=colors, width=0.6, zorder=3)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f"{v:.3f}", ha="center", color=COLOR_TEXT_PRIMARY, fontsize=9)
    ax.set_title("Log Loss (lower is better)", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_ylim(0, max(values) * 1.25)
    _rotate_model_labels(ax)
    _style_axis(ax)

    # Panel 3: per-class recall - grouped bars, the clearest way to show
    # the draw-recall gap between models
    ax = axes[2]
    x = np.arange(len(labels))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        recall = [results[name]["recall_per_class"][label] for label in labels]
        offset = (i - (len(names) - 1) / 2) * width
        ax.bar(x + offset, recall, width, label=name, color=_color_for(name), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=COLOR_TEXT_PRIMARY)
    ax.set_title("Recall by Class", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.legend(
        frameon=False, labelcolor=COLOR_TEXT_PRIMARY, fontsize=8,
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=min(len(names), 3),
    )
    _style_axis(ax)

    fig.tight_layout()
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    dest = FIGURES_PATH / save_name
    fig.savefig(dest, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {dest}")


def plot_confusion_matrices(results: dict[str, dict], save_name: str) -> None:
    """Side-by-side confusion-matrix heatmaps for every model passed in,
    one hue (sequential blue), light->dark."""
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    fig.patch.set_facecolor("#fcfcfb")
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        cm = results[name]["confusion_matrix"]
        labels = [col.removeprefix("Pred_") for col in cm.columns]
        n = len(labels)
        ax.imshow(cm.to_numpy(), cmap=_BLUE_SEQUENTIAL)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, color=COLOR_TEXT_PRIMARY)
        ax.set_yticklabels(labels, color=COLOR_TEXT_PRIMARY)
        ax.set_xlabel("Predicted", color=COLOR_TEXT_MUTED)
        ax.set_ylabel("Actual", color=COLOR_TEXT_MUTED)
        ax.set_title(name, color=COLOR_TEXT_PRIMARY, fontsize=10)
        # Annotate every cell - a small, structured grid like this is
        # exactly the case where per-cell labels are the point.
        vmax = cm.to_numpy().max()
        for i in range(n):
            for j in range(n):
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


def plot_regression_comparison(results: dict[str, dict], save_name: str = "regression_comparison.png") -> None:
    """Three panels: MAE, RMSE, R² across every regression model - separate
    panels rather than one axis, since the metrics are on different scales
    (goals vs. a variance-explained ratio)."""
    names = list(results.keys())
    colors = [_color_for(name) for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(max(13, len(names) * 2.8), 5.2))
    fig.patch.set_facecolor("#fcfcfb")

    panels = [("mae", "MAE (lower is better)"), ("rmse", "RMSE (lower is better)"), ("r2", "R² (higher is better)")]
    for ax, (metric_key, title) in zip(axes, panels):
        values = [results[n][metric_key] for n in names]
        ax.bar(names, values, color=colors, width=0.6, zorder=3)
        span = max(values) - min(min(values), 0)
        for i, v in enumerate(values):
            ax.text(i, v + span * 0.03, f"{v:.2f}", ha="center", color=COLOR_TEXT_PRIMARY, fontsize=9)
        ax.set_title(title, color=COLOR_TEXT_PRIMARY, fontsize=11)
        _rotate_model_labels(ax)
        _style_axis(ax)

    fig.tight_layout()
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    dest = FIGURES_PATH / save_name
    fig.savefig(dest, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {dest}")


def plot_regression_diagnostic(y_true, y_pred, model_name: str, save_name: str) -> None:
    """Actual vs. predicted scatter with a y=x reference line - the standard
    way to see whether a regression model's errors are unbiased/patterned."""
    metrics = regression_metrics(y_true, y_pred)
    color = _color_for(model_name)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor("#fcfcfb")

    ax.scatter(y_true, y_pred, color=color, alpha=0.35, s=18, zorder=3, edgecolors="none")
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


def generate_model_report() -> None:
    """Build every comparison plot from whatever's currently saved in
    models/*.json - automatically includes every model trained so far.
    Re-run this any time a new model is trained to refresh all the plots.
    """
    for framing in ["3-class", "2-class"]:
        results = _load_model_results("classification", framing)
        if not results:
            continue
        suffix = framing.replace("-", "")
        plot_classification_comparison(results, f"classification_comparison_{suffix}.png")
        plot_confusion_matrices(results, f"confusion_matrices_{suffix}.png")

    regression_results = _load_model_results("regression", "regression")
    if regression_results:
        plot_regression_comparison(regression_results)

    # Scatter diagnostics need actual (y_true, y_pred) pairs, not just
    # aggregate metrics, so each regression model is reloaded and re-run -
    # only models with a saved .pkl (skips any regression baseline that
    # isn't an actual trained model, if one is ever added).
    if regression_results:
        features = pd.read_csv(DATA_PROCESSED_PATH / "features.csv")
        _train, validation, _test = split_by_season(features)
        feature_cols = get_feature_columns(features)
        feature_cols_no_odds = get_feature_columns(features, include_odds=False)
        for name in regression_results:
            model_path = MODELS_PATH / f"{name}.pkl"
            if not model_path.exists():
                continue
            model = joblib.load(model_path)
            # A "_no_odds" model was fit without the odds columns - predicting
            # with the full feature set would fail on a feature-name mismatch.
            cols = feature_cols_no_odds if name.endswith("_no_odds") else feature_cols
            predictions = model.predict(validation[cols])
            plot_regression_diagnostic(
                validation["TargetGoalDifference"], predictions, name, f"regression_diagnostic_{name}.png"
            )


if __name__ == "__main__":
    generate_model_report()
