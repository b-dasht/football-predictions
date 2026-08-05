"""Plotting functions for model results. Reads from models/*.json (the
per-model metadata saved by evaluation.save_model_with_metadata) so every
chart automatically includes every model trained so far - no hardcoded
per-model list to keep in sync as new models are added.

Each comparison metric gets its own single-purpose file (accuracy, log
loss, recall/precision/F1 by class, AUROC, MAE/RMSE/R²/outcome accuracy)
rather than being crammed into shared multi-panel figures - this keeps
each chart legible as the model count grows, and makes it easy to embed
or reference just the one metric that matters for a given comparison.
Confusion matrices are written as a single markdown table (reports/
confusion_matrices.md) instead of a PNG heatmap grid - the heatmap's width
scaled linearly with model count and was already unreadable at ~10 models.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: every chart here is saved to a file, never shown interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import DATA_PROCESSED_PATH, FIGURES_PATH, MODELS_PATH, REPORTS_PATH
from src.models import odds_baseline_binary_predictions
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


def _color_for(name: str) -> str:
    # A model's "_binary" (2-class) and "_no_odds" variants share their
    # primary sibling's color - the two never collide on one chart (2-class
    # and no-odds always live on separate charts, or are disambiguated by
    # linestyle, as in plot_tuning_progress), so reusing the hue is safe and
    # keeps a model family visually identifiable across every variant.
    family = name.removesuffix("_binary").removesuffix("_no_odds")
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
    the confusion matrix back into a DataFrame. Returns {model_name: metrics}.

    Skips models/tuned_hyperparameters.json - it lives in the same
    directory (src/hyperparameter_tuning.py's output) but isn't per-model
    metadata, so it has no "task"/"framing" keys to match against.
    """
    results = {}
    for path in sorted(MODELS_PATH.glob("*.json")):
        if path.stem == "tuned_hyperparameters":
            continue
        with open(path) as f:
            data = json.load(f)
        if data["task"] != task or data["framing"] != framing:
            continue
        metrics = dict(data["metrics"])
        if "confusion_matrix" in metrics:
            metrics["confusion_matrix"] = pd.DataFrame(metrics["confusion_matrix"])
        results[path.stem] = metrics
    return results


def _save_fig(fig, save_name: str) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    dest = FIGURES_PATH / save_name
    fig.savefig(dest, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {dest}")


def plot_metric_comparison(
    results: dict[str, dict], metric_key: str, title: str, save_name: str,
    fmt: str = "{:.3f}", higher_is_better: bool = True,
) -> None:
    """A single scalar metric (accuracy, log loss, AUROC, MAE, RMSE, R²,
    outcome accuracy) across every model passed in, as its own bar-chart
    file - one question per chart, rather than sharing panels/scales with
    other metrics."""
    names = list(results.keys())
    colors = [_color_for(n) for n in names]
    values = [results[n][metric_key] for n in names]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.9), 5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.bar(names, values, color=colors, width=0.6, zorder=3)
    span = max(values) - min(min(values), 0)
    for i, v in enumerate(values):
        ax.text(i, v + span * 0.03, fmt.format(v), ha="center", color=COLOR_TEXT_PRIMARY, fontsize=9)
    direction = "higher is better" if higher_is_better else "lower is better"
    ax.set_title(f"{title} ({direction})", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.axhline(0, color=COLOR_GRID, linewidth=0.8, zorder=1)
    _rotate_model_labels(ax)
    _style_axis(ax)

    fig.tight_layout()
    _save_fig(fig, save_name)


def plot_per_class_metric_comparison(results: dict[str, dict], metric_key: str, title: str, save_name: str) -> None:
    """Grouped bars, one group per class, one bar per model - used for
    recall/precision/F1 by class. Matters most for Draw, the minority class
    where aggregate accuracy alone can hide a near-zero recall."""
    names = list(results.keys())
    labels = list(next(iter(results.values()))[metric_key].keys())

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.1), 5.5))
    fig.patch.set_facecolor("#fcfcfb")
    x = np.arange(len(labels))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        values = [results[name][metric_key][label] for label in labels]
        offset = (i - (len(names) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=name, color=_color_for(name), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=COLOR_TEXT_PRIMARY)
    ax.set_title(title, color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.legend(
        frameon=False, labelcolor=COLOR_TEXT_PRIMARY, fontsize=8,
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(names), 4),
    )
    _style_axis(ax)

    fig.tight_layout()
    _save_fig(fig, save_name)


def write_confusion_matrix_table(results_by_framing: dict[str, dict[str, dict]], dest) -> None:
    """One markdown file, sectioned by framing then by model, each
    confusion matrix as a plain table. Replaces the old PNG heatmap grid,
    which scaled 4.2in wider per model - already an unwieldy ~40in-wide
    image at 10 classifiers. The raw counts already live in every
    models/*.json; this just makes them scannable without opening JSON.
    """
    lines = [
        "# Confusion Matrices",
        "",
        "Regenerated by `python -m src.visualisation` - do not edit by hand.",
        "",
    ]
    for framing, results in results_by_framing.items():
        if not results:
            continue
        lines.append(f"## {framing}")
        lines.append("")
        for name, metrics in results.items():
            cm = metrics["confusion_matrix"]
            class_labels = [col.removeprefix("Pred_") for col in cm.columns]
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Actual \\ Predicted | " + " | ".join(class_labels) + " |")
            lines.append("|" + "---|" * (len(class_labels) + 1))
            for idx, row in cm.iterrows():
                row_label = idx.removeprefix("True_")
                lines.append(f"| {row_label} | " + " | ".join(str(v) for v in row) + " |")
            lines.append("")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines))
    logger.info(f"Saved {dest}")


def plot_regression_distribution(predictions_by_model: dict[str, tuple[np.ndarray, np.ndarray]], save_name: str) -> None:
    """Small multiples: one box-plot panel per model, showing the spread of
    predicted goal difference grouped by the true (discrete integer) goal
    difference. Replaces the old per-model y=x scatter, 11 separate files -
    goal difference is a small set of discrete integers, not a continuous
    quantity, so a scatter against a continuous y=x line never actually
    forms a clean diagonal (true values stack into vertical columns
    instead). A box plot per true value respects that discreteness
    directly: a good model shows rising medians and tight boxes; a bad one
    shows flat, overlapping ones. The dashed line is the true y=x diagonal,
    "predicted = true" - the one genuinely useful part of the old diagonal
    reference, kept but now correctly aligned.

    Boxes sit at their actual integer true-goal-difference value, not at a
    sequential index - the validation set doesn't have an example of every
    possible value (e.g. no match finished exactly +6), and positioning by
    index instead of value would silently compress that gap, distorting the
    y=x line's slope right where the gap falls. Positioning by real value
    means a missing value just shows up honestly as empty space between
    boxes, and the diagonal stays a true straight line throughout.
    """
    names = list(predictions_by_model.keys())
    ncols = min(3, len(names))
    nrows = -(-len(names) // ncols)  # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.6, nrows * 4.2), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    axes_flat = axes.flatten()

    for ax, name in zip(axes_flat, names):
        y_true, y_pred = predictions_by_model[name]
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        low, high = int(y_true.min()), int(y_true.max())
        full_range = range(low, high + 1)
        groups, positions = [], []
        for v in full_range:
            values = y_pred[y_true == v]
            if len(values) == 0:  # no validation match landed on this exact value - leave a gap, don't compress it away
                continue
            groups.append(values)
            positions.append(v)
        color = _color_for(name)

        ax.boxplot(
            groups, positions=positions, widths=0.6, patch_artist=True,
            medianprops={"color": COLOR_TEXT_PRIMARY, "linewidth": 1.3},
            boxprops={"facecolor": color, "alpha": 0.35, "edgecolor": color},
            whiskerprops={"color": color}, capprops={"color": color},
            flierprops={"markeredgecolor": color, "markersize": 3, "alpha": 0.5},
        )
        ax.plot([low, high], [low, high], color=COLOR_TEXT_MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.set_xticks(list(full_range))
        ax.set_xticklabels([str(v) for v in full_range], fontsize=7)
        ax.set_xlim(low - 0.7, high + 0.7)
        ax.set_title(name, color=COLOR_TEXT_PRIMARY, fontsize=10)
        ax.set_xlabel("True Goal Difference", color=COLOR_TEXT_MUTED, fontsize=8)
        ax.set_ylabel("Predicted Goal Difference", color=COLOR_TEXT_MUTED, fontsize=8)
        _style_axis(ax)

    for ax in axes_flat[len(names):]:
        ax.set_visible(False)

    fig.text(0.5, -0.01, "Dashed line: predicted = true", ha="center", color=COLOR_TEXT_MUTED, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, save_name)


def plot_roc_curves(roc_data: dict[str, tuple[np.ndarray, np.ndarray]], save_name: str) -> None:
    """One chart, every 2-class model's ROC curve overlaid (including the
    Bet365 odds baseline) - the standard way to compare binary classifiers'
    ranking quality directly, independent of any specific decision
    threshold. Only meaningful for the 2-class (Home/Away) framing: AUROC
    is fundamentally binary, and a 3-class one-vs-rest extension would
    produce a single macro number that's less actionable than accuracy/log
    loss already give us - deliberately not attempted here.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.plot([0, 1], [0, 1], color=COLOR_TEXT_MUTED, linewidth=1, linestyle="--", zorder=2, label="Random (AUROC=0.500)")
    for name, (y_true, y_score) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color=_color_for(name), linewidth=1.6, zorder=3, label=f"{name} (AUROC={auroc:.3f})")
    ax.set_xlabel("False Positive Rate", color=COLOR_TEXT_PRIMARY)
    ax.set_ylabel("True Positive Rate", color=COLOR_TEXT_PRIMARY)
    ax.set_title("ROC Curve - 2-class (Home vs Away)", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY, fontsize=8, loc="lower right")
    _style_axis(ax)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8, zorder=0)

    fig.tight_layout()
    _save_fig(fig, save_name)


def plot_tuning_progress(model_names: list[str], metric_key: str, title: str, save_name: str) -> None:
    """Line chart: metric_key's value across every historical training run
    logged in reports/results_log.csv, one line per model name. The x-axis
    is each model's own run sequence (1st time it was trained, 2nd time,
    ...), not a shared calendar timestamp - different model types get
    retrained at different points in the project's history, so a shared
    time axis would misalign them.

    Every models/*.json only ever holds the current/latest version of a
    model (each retrain overwrites it by name) - results_log.csv is the
    only place "every attempt" actually accumulates, which is exactly what
    a tuning-progress view needs. A flat segment between two points just
    means nothing relevant changed in that stretch (e.g. a retrain for an
    unrelated reason, same hyperparameters) - itself useful signal, not a
    gap to be filled in.

    Color encodes model family (matching every other chart); linestyle
    encodes the with-odds/no-odds variant (solid/dashed) - the same
    2-dimensional color+style pattern used for the "_binary" suffix
    elsewhere, since the two dimensions (which model, which odds variant)
    need to be visually separable at a glance.
    """
    log = pd.read_csv(REPORTS_PATH / "results_log.csv")
    log = log[log["model_name"].isin(model_names) & log[metric_key].notna()]
    if log.empty:
        logger.warning(f"No results_log.csv rows found for {model_names} - skipping {save_name}")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor("#fcfcfb")
    max_runs = 1
    for name in model_names:
        rows = log[log["model_name"] == name].sort_values("timestamp")
        if rows.empty:
            continue
        linestyle = "--" if name.endswith("_no_odds") else "-"
        run_index = range(1, len(rows) + 1)
        max_runs = max(max_runs, len(rows))
        ax.plot(
            run_index, rows[metric_key], color=_color_for(name), linestyle=linestyle,
            marker="o", markersize=5, linewidth=1.6, label=name, zorder=3,
        )
    ax.set_xlabel("Training Run (chronological)", color=COLOR_TEXT_MUTED)
    ax.set_ylabel(title, color=COLOR_TEXT_PRIMARY)
    ax.set_title(f"{title} Across Training Runs", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_xticks(range(1, max_runs + 1))
    ax.legend(
        frameon=False, labelcolor=COLOR_TEXT_PRIMARY, fontsize=8,
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=min(len(model_names), 3),
    )
    _style_axis(ax)

    fig.tight_layout()
    _save_fig(fig, save_name)


# The 5 model types that actually get tuned (per copilot-instructions.md
# #16) - PyTorch is deliberately excluded, since it's out of scope for
# tuning and its line would just be flat, diluting the "did tuning help"
# story these two charts exist to tell.
_TUNING_PROGRESS_CLASSIFIERS = [
    "baseline_logistic_regression", "baseline_logistic_regression_no_odds",
    "random_forest_classifier", "random_forest_classifier_no_odds",
    "xgboost_classifier", "xgboost_classifier_no_odds",
    "svm_classifier", "svm_classifier_no_odds",
    "neural_network_classifier", "neural_network_classifier_no_odds",
]
_TUNING_PROGRESS_REGRESSORS = [
    "baseline_linear_regression", "baseline_linear_regression_no_odds",
    "random_forest_regressor", "random_forest_regressor_no_odds",
    "xgboost_regressor", "xgboost_regressor_no_odds",
    "svm_regressor", "svm_regressor_no_odds",
    "neural_network_regressor", "neural_network_regressor_no_odds",
]


_CLASSIFICATION_METRIC_PLOTS = [
    ("accuracy", "Accuracy", "{:.1%}", True, plot_metric_comparison),
    ("log_loss", "Log Loss", "{:.3f}", False, plot_metric_comparison),
    ("recall_per_class", "Recall by Class", None, None, plot_per_class_metric_comparison),
    ("precision_per_class", "Precision by Class", None, None, plot_per_class_metric_comparison),
    ("f1_per_class", "F1 Score by Class", None, None, plot_per_class_metric_comparison),
]

_REGRESSION_METRIC_PLOTS = [
    ("mae", "MAE", "{:.2f}", False),
    ("rmse", "RMSE", "{:.2f}", False),
    ("r2", "R²", "{:.2f}", True),
    ("outcome_accuracy", "Outcome Accuracy (sign of predicted goal difference)", "{:.1%}", True),
]


def generate_model_report() -> None:
    """Build every comparison artifact from whatever's currently saved in
    models/*.json - automatically includes every model trained so far.
    Re-run this any time a new model is trained to refresh everything.
    """
    confusion_matrices_by_framing = {}

    features = pd.read_csv(DATA_PROCESSED_PATH / "features.csv")
    _train, validation, _test = split_by_season(features)
    feature_cols = get_feature_columns(features)
    feature_cols_no_odds = get_feature_columns(features, include_odds=False)
    validation_binary = validation[validation["TargetResult"] != 1]

    for framing in ["3-class", "2-class"]:
        results = _load_model_results("classification", framing)
        confusion_matrices_by_framing[framing] = results
        if not results:
            continue
        suffix = framing.replace("-", "")
        for metric_key, title, fmt, higher_is_better, plot_fn in _CLASSIFICATION_METRIC_PLOTS:
            if plot_fn is plot_metric_comparison:
                plot_fn(results, metric_key, title, f"{metric_key}_{suffix}.png", fmt=fmt, higher_is_better=higher_is_better)
            else:
                plot_fn(results, metric_key, title, f"{metric_key.removesuffix('_per_class')}_by_class_{suffix}.png")
        if framing == "2-class" and all("auroc" in m for m in results.values()):
            plot_metric_comparison(results, "auroc", "AUROC", f"auroc_{suffix}.png", fmt="{:.3f}", higher_is_better=True)

    write_confusion_matrix_table(confusion_matrices_by_framing, REPORTS_PATH / "confusion_matrices.md")

    # ROC curves: need raw (y_true, predicted_probability) pairs, not just
    # the aggregate AUROC already in the JSON, so every 2-class model with a
    # saved .pkl is reloaded and re-scored. The Bet365 baseline has no .pkl
    # (it's not a trained model) - its probabilities are recomputed directly.
    # "true positive" is defined as TargetResult == 2 (Home) throughout, and
    # every model's predict_proba column 1 is that same class by construction
    # (labels are always passed as [Away, Home], in that order) - including
    # XGBoost's binary variant, which fits on a locally remapped {0,1}
    # encoding internally but preserves that same column order.
    two_class_results = confusion_matrices_by_framing.get("2-class", {})
    if two_class_results:
        y_true_binary = (validation_binary["TargetResult"] == 2).to_numpy()
        roc_data = {}
        for name in two_class_results:
            if name == "bet365_odds_binary":
                _, proba = odds_baseline_binary_predictions(validation_binary)
            else:
                model_path = MODELS_PATH / f"{name}.pkl"
                if not model_path.exists():
                    continue
                model = joblib.load(model_path)
                proba = model.predict_proba(validation_binary[feature_cols])
            roc_data[name] = (y_true_binary, proba[:, 1])
        if roc_data:
            plot_roc_curves(roc_data, "roc_curve_2class.png")

    regression_results = _load_model_results("regression", "regression")
    if regression_results:
        for metric_key, title, fmt, higher_is_better in _REGRESSION_METRIC_PLOTS:
            plot_metric_comparison(regression_results, metric_key, title, f"{metric_key}.png", fmt=fmt, higher_is_better=higher_is_better)

        # Distribution plots need actual (y_true, y_pred) pairs, so each
        # regression model with a saved .pkl is reloaded and re-run - split
        # into with/without-odds groups (rather than one grid of all 11) to
        # keep each grid a readable size, mirroring the classification
        # 3-class/2-class file split.
        with_odds, no_odds = {}, {}
        for name in regression_results:
            model_path = MODELS_PATH / f"{name}.pkl"
            if not model_path.exists():
                continue
            model = joblib.load(model_path)
            is_no_odds = name.endswith("_no_odds")
            cols = feature_cols_no_odds if is_no_odds else feature_cols
            predictions = model.predict(validation[cols])
            target = (no_odds if is_no_odds else with_odds)
            target[name] = (validation["TargetGoalDifference"].to_numpy(), predictions)
        if with_odds:
            plot_regression_distribution(with_odds, "regression_distributions_with_odds.png")
        if no_odds:
            plot_regression_distribution(no_odds, "regression_distributions_no_odds.png")

    plot_tuning_progress(_TUNING_PROGRESS_CLASSIFIERS, "accuracy", "Accuracy", "tuning_progress_accuracy.png")
    plot_tuning_progress(_TUNING_PROGRESS_REGRESSORS, "mae", "MAE (lower is better)", "tuning_progress_mae.png")


if __name__ == "__main__":
    generate_model_report()
