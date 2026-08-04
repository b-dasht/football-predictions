"""Train and save every model: for each classifier type, a 3-class version,
a 3-class version with the Bet365 odds features excluded, and a separately
trained 2-class Home/Away-only variant; for each regressor type, a version
with and without odds. Plus the Bet365-odds baseline in the matching
framing (classification only - see docs/EDA_FINDINGS.md for why no
equivalent exists for goal difference among the kept columns).

See .github/copilot-instructions.md #13 for why every classification model
gets both framings, and why a no-odds variant exists for every model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.config import DATA_PROCESSED_PATH, RANDOM_STATE
from src.evaluation import (
    classification_metrics,
    log_classification_metrics,
    log_regression_metrics,
    regression_metrics,
    save_model_with_metadata,
)
from src.utils import get_feature_columns, split_by_season


def build_baseline_classifier() -> Pipeline:
    """Logistic Regression: impute missing rolling stats (median), scale, then classify."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def build_baseline_regressor() -> Pipeline:
    """Linear Regression: same imputation/scaling approach as the classifier."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def build_random_forest_classifier() -> Pipeline:
    """Random Forest: impute (still required - unlike XGBoost, scikit-learn's
    RandomForestClassifier doesn't accept NaN natively), no scaler - tree
    splits are scale-invariant, so standardizing would be a no-op here.

    n_estimators/max_depth are scikit-learn's sensible defaults, not yet
    tuned - that's Hyperparameter Tuning's job, later in the roadmap.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def build_random_forest_regressor() -> Pipeline:
    """Random Forest Regressor: same imputation approach, no scaler, as above."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def build_xgboost_classifier() -> Pipeline:
    """XGBoost: no imputer - unlike scikit-learn's tree models, XGBoost
    natively handles NaN by learning which direction to route a missing
    value at each split, which is arguably more principled than an
    arbitrary median fill. No scaler either (tree-based, scale-invariant).
    """
    return Pipeline([
        ("model", XGBClassifier(random_state=RANDOM_STATE)),
    ])


def build_xgboost_regressor() -> Pipeline:
    """XGBoost Regressor: same no-imputer, no-scaler approach as above."""
    return Pipeline([
        ("model", XGBRegressor(random_state=RANDOM_STATE)),
    ])


def odds_baseline_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class = whichever outcome Bet365's odds imply is most likely.

    Not a trained model - just reading off the bookmaker's own implied
    probabilities as the prediction, per the project's rule that models
    should be judged against this, not only against Logistic Regression.
    """
    proba = df[["ImpliedProbAway", "ImpliedProbDraw", "ImpliedProbHome"]].to_numpy()
    predictions = proba.argmax(axis=1)  # column order matches TargetResult's 0/1/2 encoding
    return predictions, proba


def odds_baseline_binary_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Bet365's Home vs Away call only, with Draw's implied-probability share
    excluded (not redistributed) and the remaining two renormalized to sum
    to 1. The fair comparison partner for a model that's also restricted to
    choosing only between Home and Away.
    """
    prob_home = df["ImpliedProbHome"].to_numpy()
    prob_away = df["ImpliedProbAway"].to_numpy()
    proba_home = prob_home / (prob_home + prob_away)
    predictions = np.where(proba_home > 0.5, 2, 0)  # 2=Home, 0=Away
    proba = np.column_stack([1 - proba_home, proba_home])  # columns: [Away, Home]
    return predictions, proba


def train_and_save_classifier(
    model: Pipeline,
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    framing: str = "3-class",
    labels: list[int] | None = None,
    label_names: list[str] | None = None,
) -> Pipeline:
    """Fit, evaluate on the validation split, log, and save one classifier.

    Shared by every model type (baseline, Random Forest, and beyond) so
    the fit->evaluate->log->save sequence is written once, not duplicated
    per model type. Returns the fitted pipeline, in case the caller needs
    it (e.g. for a diagnostic like coefficient/feature-importance inspection).
    """
    model.fit(train[feature_cols], train["TargetResult"])
    predictions = model.predict(validation[feature_cols])
    proba = model.predict_proba(validation[feature_cols])
    metrics = classification_metrics(validation["TargetResult"], predictions, proba, labels=labels, label_names=label_names)
    log_classification_metrics(f"{name} ({framing})", metrics, label_names=label_names)
    save_model_with_metadata(model, name, "classification", framing, metrics)
    return model


def train_and_save_regressor(
    model: Pipeline, name: str, train: pd.DataFrame, validation: pd.DataFrame, feature_cols: list[str]
) -> Pipeline:
    """Fit, evaluate, log, and save one regressor. Mirrors train_and_save_classifier."""
    model.fit(train[feature_cols], train["TargetGoalDifference"])
    predictions = model.predict(validation[feature_cols])
    metrics = regression_metrics(validation["TargetGoalDifference"], predictions)
    log_regression_metrics(name, metrics)
    save_model_with_metadata(model, name, "regression", "regression", metrics)
    return model


def train_baseline_models() -> None:
    """Train, evaluate, and save every baseline model and the Bet365 odds
    baseline, in every required framing (3-class, 3-class no-odds, 2-class;
    regression, regression no-odds).
    """
    features = pd.read_csv(DATA_PROCESSED_PATH / "features.csv")
    train, validation, _test = split_by_season(features)
    feature_cols = get_feature_columns(features)
    feature_cols_no_odds = get_feature_columns(features, include_odds=False)
    train_binary = train[train["TargetResult"] != 1]
    validation_binary = validation[validation["TargetResult"] != 1]

    train_and_save_classifier(
        build_baseline_classifier(), "baseline_logistic_regression", train, validation, feature_cols
    )
    train_and_save_classifier(
        build_baseline_classifier(), "baseline_logistic_regression_no_odds", train, validation, feature_cols_no_odds
    )
    train_and_save_classifier(
        build_baseline_classifier(), "baseline_logistic_regression_binary", train_binary, validation_binary,
        feature_cols, framing="2-class", labels=[0, 2], label_names=["Away", "Home"],
    )

    odds_predictions, odds_proba = odds_baseline_predictions(validation)
    odds_metrics = classification_metrics(validation["TargetResult"], odds_predictions, odds_proba)
    log_classification_metrics("bet365_odds (3-class)", odds_metrics)
    save_model_with_metadata(None, "bet365_odds", "classification", "3-class", odds_metrics)

    binary_odds_predictions, binary_odds_proba = odds_baseline_binary_predictions(validation_binary)
    binary_odds_metrics = classification_metrics(
        validation_binary["TargetResult"], binary_odds_predictions, binary_odds_proba,
        labels=[0, 2], label_names=["Away", "Home"],
    )
    log_classification_metrics("bet365_odds_binary (2-class)", binary_odds_metrics, label_names=["Away", "Home"])
    # "bet365_odds_binary" (not "bet365_odds") - a distinct filename, since
    # it would otherwise collide with the 3-class save above.
    save_model_with_metadata(None, "bet365_odds_binary", "classification", "2-class", binary_odds_metrics)

    train_and_save_regressor(build_baseline_regressor(), "baseline_linear_regression", train, validation, feature_cols)
    train_and_save_regressor(
        build_baseline_regressor(), "baseline_linear_regression_no_odds", train, validation, feature_cols_no_odds
    )


def train_random_forest_models() -> None:
    """Train, evaluate, and save every Random Forest variant (3-class,
    3-class no-odds, 2-class; regression, regression no-odds). Doesn't
    retrain/resave the Bet365 odds baseline - it's not affected by model
    type, so the copy saved by train_baseline_models() is reused as-is by
    every comparison plot automatically.
    """
    features = pd.read_csv(DATA_PROCESSED_PATH / "features.csv")
    train, validation, _test = split_by_season(features)
    feature_cols = get_feature_columns(features)
    feature_cols_no_odds = get_feature_columns(features, include_odds=False)
    train_binary = train[train["TargetResult"] != 1]
    validation_binary = validation[validation["TargetResult"] != 1]

    train_and_save_classifier(
        build_random_forest_classifier(), "random_forest_classifier", train, validation, feature_cols
    )
    train_and_save_classifier(
        build_random_forest_classifier(), "random_forest_classifier_no_odds", train, validation, feature_cols_no_odds
    )
    train_and_save_classifier(
        build_random_forest_classifier(), "random_forest_classifier_binary", train_binary, validation_binary,
        feature_cols, framing="2-class", labels=[0, 2], label_names=["Away", "Home"],
    )

    train_and_save_regressor(build_random_forest_regressor(), "random_forest_regressor", train, validation, feature_cols)
    train_and_save_regressor(
        build_random_forest_regressor(), "random_forest_regressor_no_odds", train, validation, feature_cols_no_odds
    )


def train_xgboost_models() -> None:
    """Train, evaluate, and save every XGBoost variant (3-class, 3-class
    no-odds, 2-class; regression, regression no-odds).

    The 2-class case needs one extra step other models don't: XGBoost
    requires class labels to be consecutive integers starting at 0, but
    our 2-class TargetResult uses {0, 2} (Away, Home) - valid for every
    scikit-learn classifier so far, but XGBoost raises an error on it
    (confirmed directly). Locally remap 2->1 just for this fit/evaluate
    call - labels=[0, 1] with label_names=["Away", "Home"] keeps every
    reported metric in human-readable terms, so nothing downstream needs
    to know the remapping happened at all.
    """
    features = pd.read_csv(DATA_PROCESSED_PATH / "features.csv")
    train, validation, _test = split_by_season(features)
    feature_cols = get_feature_columns(features)
    feature_cols_no_odds = get_feature_columns(features, include_odds=False)

    train_and_save_classifier(build_xgboost_classifier(), "xgboost_classifier", train, validation, feature_cols)
    train_and_save_classifier(
        build_xgboost_classifier(), "xgboost_classifier_no_odds", train, validation, feature_cols_no_odds
    )

    train_binary = train[train["TargetResult"] != 1].copy()
    validation_binary = validation[validation["TargetResult"] != 1].copy()
    train_binary["TargetResult"] = train_binary["TargetResult"].replace(2, 1)
    validation_binary["TargetResult"] = validation_binary["TargetResult"].replace(2, 1)
    train_and_save_classifier(
        build_xgboost_classifier(), "xgboost_classifier_binary", train_binary, validation_binary,
        feature_cols, framing="2-class", labels=[0, 1], label_names=["Away", "Home"],
    )

    train_and_save_regressor(build_xgboost_regressor(), "xgboost_regressor", train, validation, feature_cols)
    train_and_save_regressor(
        build_xgboost_regressor(), "xgboost_regressor_no_odds", train, validation, feature_cols_no_odds
    )


if __name__ == "__main__":
    train_baseline_models()
    train_random_forest_models()
    train_xgboost_models()
