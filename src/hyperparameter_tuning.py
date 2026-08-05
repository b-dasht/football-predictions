"""Hyperparameter tuning (see .github/copilot-instructions.md #16).

Tunes each model *type* once (not once per with-odds/no-odds/2-class
variant - the architecture doesn't change between variants, only the input
columns/rows, so the same tuned hyperparameters are reused across all of
them by src/models.py). Uses the training data only, via a time-aware,
season-based expanding-window cross-validation - never the validation
season, per the binding rule.

RandomizedSearchCV (a fixed budget of randomly sampled combinations) is
used instead of an exhaustive GridSearchCV - it scales predictably across
every model type regardless of how large a given search space gets, with
no extra dependency.

Scoring: classification is scored on neg_log_loss rather than accuracy -
accuracy alone would let the search ignore Draw entirely (already the
project's known weak point - a model that always predicts Home/Away can
still score well on accuracy), whereas log loss rewards well-calibrated
probabilities across all 3 classes and is already one of the project's
headline classification metrics. Regression is scored on
neg_mean_absolute_error, matching MAE, the primary reported regression
metric.
"""

import json

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

from src.config import DATA_PROCESSED_PATH, MODELS_PATH, RANDOM_STATE
from src.models import (
    build_logistic_regression_classifier,
    build_neural_network_classifier,
    build_neural_network_regressor,
    build_random_forest_classifier,
    build_random_forest_regressor,
    build_svm_classifier,
    build_svm_regressor,
    build_xgboost_classifier,
    build_xgboost_regressor,
)
from src.utils import get_feature_columns, split_by_season

TUNED_PARAMS_PATH = MODELS_PATH / "tuned_hyperparameters.json"

CLASSIFICATION_SCORING = "neg_log_loss"
REGRESSION_SCORING = "neg_mean_absolute_error"

# Keyed to match each Pipeline's step name ("model", or "model__estimator"
# for SVM's CalibratedClassifierCV wrapper) - confirmed via
# build_svm_classifier().get_params(), so RandomizedSearchCV's best_params_
# can be applied straight back with pipeline.set_params(**params), no
# reparsing needed.
PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    },
    # Round 2 (2026-08-05): round 1's best values landed on n_estimators'
    # low boundary (50), reg_alpha's high boundary (1.0), and reg_lambda's
    # low boundary (0.5) - accuracy/R^2 both improved substantially in
    # round 1 (the largest jump of any model type), suggesting the true
    # optimum sits further in that same direction, not just at what round
    # 1 happened to offer. Range shifted accordingly: fewer estimators
    # still, more L1 regularization, less L2.
    "xgboost": {
        "model__n_estimators": [10, 20, 30, 50, 75],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
        "model__subsample": [0.6, 0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "model__reg_alpha": [0.5, 1.0, 2.0, 5.0],
        "model__reg_lambda": [0.1, 0.3, 0.5, 1.0],
    },
    "svm_classifier": {
        "model__estimator__C": loguniform(1e-2, 1e2),
        "model__estimator__gamma": loguniform(1e-3, 1),
    },
    "svm_regressor": {
        "model__C": loguniform(1e-2, 1e2),
        "model__gamma": loguniform(1e-3, 1),
    },
    # Round 2 (2026-08-05): round 1's best values landed on
    # hidden_layer_sizes' smallest offered option ((32,)) and near
    # learning_rate_init's low boundary - consistent with the known severe
    # overfitting finding (a smaller, slower-learning network should
    # overfit less). Range shifted smaller/slower rather than re-searching
    # the same space.
    "neural_network": {
        "model__hidden_layer_sizes": [(8,), (16,), (32,)],
        "model__alpha": loguniform(1e-3, 1e-1),
        "model__learning_rate_init": loguniform(1e-5, 1e-3),
    },
    # LinearRegression has no meaningful hyperparameters to tune - only
    # Logistic Regression's regularization strength gets a search here.
    "logistic_regression": {
        "model__C": loguniform(1e-3, 1e2),
    },
}


def season_expanding_splits(train_df: pd.DataFrame, min_train_seasons: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-aware CV folds within the training data only: fold i trains on
    every season up to and including season i, validates on season i+1 -
    mirrors the real train -> validation setup exactly, at a season
    granularity rather than an arbitrary row count.

    min_train_seasons skips the earliest folds, where the training set is
    tiny (a single season, ~380 matches) and unrepresentative of the final
    model's actual training size (~5,300 rows) - those folds would be
    noisy signal for choosing hyperparameters meant for the full-history fit.

    Returns positional (not label-based) index arrays, since that's what
    RandomizedSearchCV's cv parameter expects when passed a plain list of
    (train_idx, test_idx) pairs.
    """
    seasons = sorted(train_df["Season"].unique())
    season_order = {season: i for i, season in enumerate(seasons)}
    season_position = train_df["Season"].map(season_order).to_numpy()

    splits = []
    for i in range(min_train_seasons, len(seasons) - 1):
        train_idx = np.where(season_position <= i)[0]
        val_idx = np.where(season_position == i + 1)[0]
        splits.append((train_idx, val_idx))
    return splits


def _random_search(build_fn, param_distributions: dict, X, y, cv, scoring: str, n_iter: int) -> dict:
    search = RandomizedSearchCV(
        build_fn(), param_distributions, n_iter=n_iter, cv=cv, scoring=scoring,
        random_state=RANDOM_STATE, n_jobs=-1, refit=False,
    )
    search.fit(X, y)
    logger.info(f"[{build_fn.__name__}] best {scoring}={search.best_score_:.4f} params={search.best_params_}")
    return search.best_params_


def tune_all_models(n_iter: int = 25, only: list[str] | None = None) -> dict:
    """Run one RandomizedSearchCV per tunable model type and save every
    result to models/tuned_hyperparameters.json - src/models.py's
    train_*_models() functions read this file and apply the tuned params
    to every variant of that model type (3-class, no-odds, 2-class,
    regression, regression no-odds).

    only, if given, restricts the search to just those model-type names
    (e.g. ["xgboost_classifier", "xgboost_regressor"]) - for continuing to
    tune specific models further (a second, deeper round with a refined
    search space) without re-running every other model type, which would
    just waste time re-searching ones that already plateaued. Existing
    results for every other model type are preserved, not overwritten -
    the saved file is loaded first and only the requested entries are
    replaced.
    """
    features = pd.read_csv(DATA_PROCESSED_PATH / "features.csv")
    train, _validation, _test = split_by_season(features)
    feature_cols = get_feature_columns(features)
    cv = season_expanding_splits(train)
    logger.info(f"Time-aware CV: {len(cv)} folds (training data only, validation season never touched)")

    X_class, y_class = train[feature_cols], train["TargetResult"]
    X_reg, y_reg = train[feature_cols], train["TargetGoalDifference"]

    # name -> (build_fn, param_distributions, X, y, scoring)
    jobs = {
        "logistic_regression": (
            build_logistic_regression_classifier, PARAM_DISTRIBUTIONS["logistic_regression"],
            X_class, y_class, CLASSIFICATION_SCORING,
        ),
        "random_forest_classifier": (
            build_random_forest_classifier, PARAM_DISTRIBUTIONS["random_forest"],
            X_class, y_class, CLASSIFICATION_SCORING,
        ),
        "random_forest_regressor": (
            build_random_forest_regressor, PARAM_DISTRIBUTIONS["random_forest"],
            X_reg, y_reg, REGRESSION_SCORING,
        ),
        "xgboost_classifier": (
            build_xgboost_classifier, PARAM_DISTRIBUTIONS["xgboost"],
            X_class, y_class, CLASSIFICATION_SCORING,
        ),
        "xgboost_regressor": (
            build_xgboost_regressor, PARAM_DISTRIBUTIONS["xgboost"],
            X_reg, y_reg, REGRESSION_SCORING,
        ),
        "svm_classifier": (
            build_svm_classifier, PARAM_DISTRIBUTIONS["svm_classifier"],
            X_class, y_class, CLASSIFICATION_SCORING,
        ),
        "svm_regressor": (
            build_svm_regressor, PARAM_DISTRIBUTIONS["svm_regressor"],
            X_reg, y_reg, REGRESSION_SCORING,
        ),
        "neural_network_classifier": (
            build_neural_network_classifier, PARAM_DISTRIBUTIONS["neural_network"],
            X_class, y_class, CLASSIFICATION_SCORING,
        ),
        "neural_network_regressor": (
            build_neural_network_regressor, PARAM_DISTRIBUTIONS["neural_network"],
            X_reg, y_reg, REGRESSION_SCORING,
        ),
    }
    if only is not None:
        jobs = {name: job for name, job in jobs.items() if name in only}

    tuned = {}
    if TUNED_PARAMS_PATH.exists():
        with open(TUNED_PARAMS_PATH) as f:
            tuned = json.load(f)

    for name, (build_fn, param_distributions, X, y, scoring) in jobs.items():
        tuned[name] = _random_search(build_fn, param_distributions, X, y, cv, scoring, n_iter)

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    with open(TUNED_PARAMS_PATH, "w") as f:
        json.dump(tuned, f, indent=2, default=str)
    logger.info(f"Saved tuned hyperparameters -> {TUNED_PARAMS_PATH}")
    return tuned


if __name__ == "__main__":
    tune_all_models()
