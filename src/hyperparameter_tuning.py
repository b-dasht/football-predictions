"""Hyperparameter tuning (see .github/copilot-instructions.md #16).

Tunes each model type twice - once for the with-odds feature set, once for
the no-odds feature set - as independent searches (see tune_all_models'
docstring for why), and reuses each result across that feature set's own
3-class/2-class/regression variants (the architecture doesn't change
between those, only the input columns/rows). Uses the training data only,
via a time-aware, season-based expanding-window cross-validation - never
the validation season, per the binding rule.

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
    # Round 3 (2026-08-05) was tried - round 2 again landed on max_depth's
    # low boundary (2) and learning_rate's high boundary (0.08), shifted
    # shallower/faster still - but round 3's CV score barely moved
    # (classifier neg_log_loss -0.9623 -> -0.9626, regressor
    # neg_mean_absolute_error -1.2892 -> -1.2909, both within noise) and
    # validation accuracy actually dipped slightly. Reverted to round 2's
    # hyperparameters (models/tuned_hyperparameters.json) and stopped
    # tuning XGBoost here rather than chase further noise - this grid is
    # left at its round-3 shape for reference, not because round 3 is
    # what's currently in use.
    "xgboost": {
        "model__n_estimators": [10, 20, 30, 50, 75],
        "model__max_depth": [1, 2, 3],
        "model__learning_rate": [0.05, 0.08, 0.12, 0.16],
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
    # Round 3 (2026-08-05): split into separate classifier/regressor
    # entries (previously shared) - round 2 picked opposite ends of the
    # same hidden_layer_sizes options for the two (classifier wanted the
    # smallest option, (32,); regressor wanted the largest), so one shared
    # space no longer fits both well. Each also now includes a couple of
    # 2-hidden-layer options - not tried before - to actually test whether
    # depth helps on a dataset this size (~5,300 rows), rather than only
    # searching single-layer width. classifier's alpha/learning_rate_init
    # both hit round 2's boundary (alpha near its high end, at 0.084 of a
    # 0.1 ceiling; learning_rate_init had room) - alpha's ceiling raised.
    # regressor's hidden_layer_sizes hit the high boundary and
    # learning_rate_init hit the low boundary - both shifted further out.
    #
    # Round 4 (2026-08-05), classifier only: round 3's CV score improved
    # meaningfully here (neg_log_loss -1.0135 -> -1.0103) specifically
    # because a 2-hidden-layer shape, (8, 4), won - genuine evidence that
    # depth helps for this task, not just noise (unlike round 3's XGBoost/
    # NN-regressor results, which were within CV noise of round 2 and were
    # reverted). Refined around that winning shape rather than the
    # original single-layer-biased grid, plus one 3-layer option to check
    # whether depth keeps paying off. alpha/learning_rate_init weren't at
    # a round-3 boundary, so left as-is.
    "neural_network_classifier": {
        "model__hidden_layer_sizes": [(4, 4), (8, 4), (8, 8), (16, 4), (16, 8), (8, 4, 2)],
        "model__alpha": loguniform(1e-3, 3e-1),
        "model__learning_rate_init": loguniform(1e-5, 1e-3),
    },
    # XGBoost (both) and this regressor stopped after round 2 - round 3
    # showed no real CV improvement for any of the three (all within
    # noise), so their round-2 hyperparameters were kept as final rather
    # than chasing further marginal, likely-noisy gains. Left at round 2's
    # range for reference; not re-run.
    "neural_network_regressor": {
        "model__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (16, 8)],
        "model__alpha": loguniform(1e-4, 1e-1),
        "model__learning_rate_init": loguniform(1e-6, 1e-3),
    },
    # Round 2 (2026-08-05): round 1's best C (0.00195) sat just above the
    # 1e-3 floor - shifted the range down to see if even more
    # regularization helps further. LinearRegression has no meaningful
    # hyperparameters to tune - only Logistic Regression's regularization
    # strength gets a search here.
    "logistic_regression": {
        "model__C": loguniform(1e-5, 1e-1),
    },
    # No-odds search space for the Neural Network classifier: deliberately
    # broader than the with-odds grid above (which narrowed toward
    # multi-layer shapes across rounds 3-4, tuned around what suited the
    # *with-odds* problem) - spans every depth tried so far, single-layer
    # through 3-layer, so the no-odds search can land wherever actually
    # fits best for this differently-shaped, odds-free problem rather than
    # inheriting a bias toward what won with the odds features included.
    "neural_network_classifier_no_odds": {
        "model__hidden_layer_sizes": [(16,), (32,), (64,), (8, 4), (16, 8), (32, 16), (8, 4, 2)],
        "model__alpha": loguniform(1e-4, 3e-1),
        "model__learning_rate_init": loguniform(1e-5, 1e-3),
    },
    "neural_network_regressor_no_odds": {
        "model__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (16, 8)],
        "model__alpha": loguniform(1e-4, 1e-1),
        "model__learning_rate_init": loguniform(1e-6, 1e-3),
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
    """Run one RandomizedSearchCV per tunable model/feature-set combination
    and save every result to models/tuned_hyperparameters.json -
    src/models.py's train_*_models() functions read this file and apply
    each entry to its matching variant.

    Every model type is tuned *twice* - once on the full (with-odds)
    feature set, once on the no-odds feature set - as two independent
    searches, not one search whose result gets reused for both. A
    hyperparameter set tuned around having the odds features available
    isn't necessarily the right shape once they're removed (e.g. a deeper
    architecture that exploits a strong odds signal may just overfit noise
    without it) - the 3-class/regression no-odds variants get their own
    "_no_odds"-suffixed entry (e.g. "xgboost_classifier_no_odds") rather
    than inheriting "xgboost_classifier"'s. The 2-class (_binary) variant
    still reuses the primary with-odds entry - it's always trained with
    odds included, per copilot-instructions.md #13.

    only, if given, restricts the search to just those names (e.g.
    ["xgboost_classifier", "xgboost_classifier_no_odds"]) - for continuing
    to tune specific models further (a second, deeper round with a refined
    search space) without re-running every other one, which would just
    waste time re-searching combinations that already plateaued. Existing
    results for every other entry are preserved, not overwritten - the
    saved file is loaded first and only the requested entries are replaced.
    """
    features = pd.read_csv(DATA_PROCESSED_PATH / "features.csv")
    train, _validation, _test = split_by_season(features)
    feature_cols = get_feature_columns(features)
    feature_cols_no_odds = get_feature_columns(features, include_odds=False)
    cv = season_expanding_splits(train)
    logger.info(f"Time-aware CV: {len(cv)} folds (training data only, validation season never touched)")

    X_class, y_class = train[feature_cols], train["TargetResult"]
    X_reg, y_reg = train[feature_cols], train["TargetGoalDifference"]
    X_class_no_odds = train[feature_cols_no_odds]
    X_reg_no_odds = train[feature_cols_no_odds]

    # name -> (build_fn, param_distributions, X, y, scoring)
    jobs = {
        "logistic_regression": (
            build_logistic_regression_classifier, PARAM_DISTRIBUTIONS["logistic_regression"],
            X_class, y_class, CLASSIFICATION_SCORING,
        ),
        "logistic_regression_no_odds": (
            build_logistic_regression_classifier, PARAM_DISTRIBUTIONS["logistic_regression"],
            X_class_no_odds, y_class, CLASSIFICATION_SCORING,
        ),
        "random_forest_classifier": (
            build_random_forest_classifier, PARAM_DISTRIBUTIONS["random_forest"],
            X_class, y_class, CLASSIFICATION_SCORING,
        ),
        "random_forest_classifier_no_odds": (
            build_random_forest_classifier, PARAM_DISTRIBUTIONS["random_forest"],
            X_class_no_odds, y_class, CLASSIFICATION_SCORING,
        ),
        "random_forest_regressor": (
            build_random_forest_regressor, PARAM_DISTRIBUTIONS["random_forest"],
            X_reg, y_reg, REGRESSION_SCORING,
        ),
        "random_forest_regressor_no_odds": (
            build_random_forest_regressor, PARAM_DISTRIBUTIONS["random_forest"],
            X_reg_no_odds, y_reg, REGRESSION_SCORING,
        ),
        "xgboost_classifier": (
            build_xgboost_classifier, PARAM_DISTRIBUTIONS["xgboost"],
            X_class, y_class, CLASSIFICATION_SCORING,
        ),
        "xgboost_classifier_no_odds": (
            build_xgboost_classifier, PARAM_DISTRIBUTIONS["xgboost"],
            X_class_no_odds, y_class, CLASSIFICATION_SCORING,
        ),
        "xgboost_regressor": (
            build_xgboost_regressor, PARAM_DISTRIBUTIONS["xgboost"],
            X_reg, y_reg, REGRESSION_SCORING,
        ),
        "xgboost_regressor_no_odds": (
            build_xgboost_regressor, PARAM_DISTRIBUTIONS["xgboost"],
            X_reg_no_odds, y_reg, REGRESSION_SCORING,
        ),
        "svm_classifier": (
            build_svm_classifier, PARAM_DISTRIBUTIONS["svm_classifier"],
            X_class, y_class, CLASSIFICATION_SCORING,
        ),
        "svm_classifier_no_odds": (
            build_svm_classifier, PARAM_DISTRIBUTIONS["svm_classifier"],
            X_class_no_odds, y_class, CLASSIFICATION_SCORING,
        ),
        "svm_regressor": (
            build_svm_regressor, PARAM_DISTRIBUTIONS["svm_regressor"],
            X_reg, y_reg, REGRESSION_SCORING,
        ),
        "svm_regressor_no_odds": (
            build_svm_regressor, PARAM_DISTRIBUTIONS["svm_regressor"],
            X_reg_no_odds, y_reg, REGRESSION_SCORING,
        ),
        "neural_network_classifier": (
            build_neural_network_classifier, PARAM_DISTRIBUTIONS["neural_network_classifier"],
            X_class, y_class, CLASSIFICATION_SCORING,
        ),
        "neural_network_classifier_no_odds": (
            build_neural_network_classifier, PARAM_DISTRIBUTIONS["neural_network_classifier_no_odds"],
            X_class_no_odds, y_class, CLASSIFICATION_SCORING,
        ),
        "neural_network_regressor": (
            build_neural_network_regressor, PARAM_DISTRIBUTIONS["neural_network_regressor"],
            X_reg, y_reg, REGRESSION_SCORING,
        ),
        "neural_network_regressor_no_odds": (
            build_neural_network_regressor, PARAM_DISTRIBUTIONS["neural_network_regressor_no_odds"],
            X_reg_no_odds, y_reg, REGRESSION_SCORING,
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
