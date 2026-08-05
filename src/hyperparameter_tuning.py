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
    build_baseline_classifier,
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
    # XGBoost's defaults are tuned for much larger datasets than our ~5,300
    # training rows (a known suspect for why it looked weakest untuned) -
    # the search deliberately includes smaller/shallower/more-regularized
    # options, not just a wider version of the defaults.
    "xgboost": {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_alpha": [0, 0.1, 0.5, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    },
    "svm_classifier": {
        "model__estimator__C": loguniform(1e-2, 1e2),
        "model__estimator__gamma": loguniform(1e-3, 1),
    },
    "svm_regressor": {
        "model__C": loguniform(1e-2, 1e2),
        "model__gamma": loguniform(1e-3, 1),
    },
    "neural_network": {
        "model__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)],
        "model__alpha": loguniform(1e-5, 1e-1),
        "model__learning_rate_init": loguniform(1e-4, 1e-2),
    },
    # LinearRegression has no meaningful hyperparameters to tune - only the
    # baseline classifier (Logistic Regression's regularization strength)
    # gets a search here.
    "baseline_classifier": {
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


def tune_all_models(n_iter: int = 25) -> dict:
    """Run one RandomizedSearchCV per tunable model type and save every
    result to models/tuned_hyperparameters.json - src/models.py's
    train_*_models() functions read this file and apply the tuned params
    to every variant of that model type (3-class, no-odds, 2-class,
    regression, regression no-odds).
    """
    features = pd.read_csv(DATA_PROCESSED_PATH / "features.csv")
    train, _validation, _test = split_by_season(features)
    feature_cols = get_feature_columns(features)
    cv = season_expanding_splits(train)
    logger.info(f"Time-aware CV: {len(cv)} folds (training data only, validation season never touched)")

    X_class, y_class = train[feature_cols], train["TargetResult"]
    X_reg, y_reg = train[feature_cols], train["TargetGoalDifference"]

    tuned = {
        "baseline_logistic_regression": _random_search(
            build_baseline_classifier, PARAM_DISTRIBUTIONS["baseline_classifier"],
            X_class, y_class, cv, CLASSIFICATION_SCORING, n_iter,
        ),
        "random_forest_classifier": _random_search(
            build_random_forest_classifier, PARAM_DISTRIBUTIONS["random_forest"],
            X_class, y_class, cv, CLASSIFICATION_SCORING, n_iter,
        ),
        "random_forest_regressor": _random_search(
            build_random_forest_regressor, PARAM_DISTRIBUTIONS["random_forest"],
            X_reg, y_reg, cv, REGRESSION_SCORING, n_iter,
        ),
        "xgboost_classifier": _random_search(
            build_xgboost_classifier, PARAM_DISTRIBUTIONS["xgboost"],
            X_class, y_class, cv, CLASSIFICATION_SCORING, n_iter,
        ),
        "xgboost_regressor": _random_search(
            build_xgboost_regressor, PARAM_DISTRIBUTIONS["xgboost"],
            X_reg, y_reg, cv, REGRESSION_SCORING, n_iter,
        ),
        "svm_classifier": _random_search(
            build_svm_classifier, PARAM_DISTRIBUTIONS["svm_classifier"],
            X_class, y_class, cv, CLASSIFICATION_SCORING, n_iter,
        ),
        "svm_regressor": _random_search(
            build_svm_regressor, PARAM_DISTRIBUTIONS["svm_regressor"],
            X_reg, y_reg, cv, REGRESSION_SCORING, n_iter,
        ),
        "neural_network_classifier": _random_search(
            build_neural_network_classifier, PARAM_DISTRIBUTIONS["neural_network"],
            X_class, y_class, cv, CLASSIFICATION_SCORING, n_iter,
        ),
        "neural_network_regressor": _random_search(
            build_neural_network_regressor, PARAM_DISTRIBUTIONS["neural_network"],
            X_reg, y_reg, cv, REGRESSION_SCORING, n_iter,
        ),
    }

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    with open(TUNED_PARAMS_PATH, "w") as f:
        json.dump(tuned, f, indent=2, default=str)
    logger.info(f"Saved tuned hyperparameters -> {TUNED_PARAMS_PATH}")
    return tuned


if __name__ == "__main__":
    tune_all_models()
