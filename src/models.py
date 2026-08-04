"""Train and save baseline models: Logistic Regression (both a 3-class and a
separately trained 2-class Home/Away-only variant), Linear Regression, and
the Bet365-odds baseline in the matching framing (classification only - see
docs/EDA_FINDINGS.md for why no equivalent exists for goal difference among
the kept columns). See .github/copilot-instructions.md #13 for why every
classification model gets both framings, not just the 3-class one.
"""

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED_PATH, MODELS_PATH, RANDOM_STATE
from src.evaluation import (
    classification_metrics,
    log_classification_metrics,
    log_regression_metrics,
    regression_metrics,
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


def train_baseline_models() -> None:
    """Train, evaluate (on the validation season), and save baseline models -
    both the full 3-class setup (Home/Draw/Away) and a genuinely separate
    2-class model (trained only on non-draw matches, restricted to
    predicting Home or Away - never Draw), each compared against the
    Bet365 odds baseline in the matching framing. Both stay in place: every
    later model (Advanced Models onward) should be trained and compared in
    both framings too, not just this baseline pass.
    """
    features = pd.read_csv(DATA_PROCESSED_PATH / "features.csv")
    train, validation, _test = split_by_season(features)
    feature_cols = get_feature_columns(features)

    # --- 3-class: Logistic Regression vs. the full Bet365 odds baseline ---
    classifier = build_baseline_classifier()
    classifier.fit(train[feature_cols], train["TargetResult"])

    val_predictions = classifier.predict(validation[feature_cols])
    val_proba = classifier.predict_proba(validation[feature_cols])
    logreg_metrics = classification_metrics(validation["TargetResult"], val_predictions, val_proba)
    log_classification_metrics("LogisticRegression (3-class)", logreg_metrics)

    odds_predictions, odds_proba = odds_baseline_predictions(validation)
    odds_metrics = classification_metrics(validation["TargetResult"], odds_predictions, odds_proba)
    log_classification_metrics("Bet365Odds (3-class)", odds_metrics)

    # --- 2-class: a separately trained model, fit only on non-draw matches,
    # so it never has the option to predict Draw at all - not the 3-class
    # model's marginal Home/Away probabilities. Compared against Bet365's
    # odds restricted the same way, for a genuine like-for-like comparison. ---
    train_binary = train[train["TargetResult"] != 1]
    validation_binary = validation[validation["TargetResult"] != 1]

    binary_classifier = build_baseline_classifier()
    binary_classifier.fit(train_binary[feature_cols], train_binary["TargetResult"])

    binary_predictions = binary_classifier.predict(validation_binary[feature_cols])
    binary_proba = binary_classifier.predict_proba(validation_binary[feature_cols])  # columns: [Away, Home]
    binary_logreg_metrics = classification_metrics(
        validation_binary["TargetResult"], binary_predictions, binary_proba, labels=[0, 2], label_names=["Away", "Home"]
    )
    log_classification_metrics("LogisticRegression (2-class)", binary_logreg_metrics, label_names=["Away", "Home"])

    binary_odds_predictions, binary_odds_proba = odds_baseline_binary_predictions(validation_binary)
    binary_odds_metrics = classification_metrics(
        validation_binary["TargetResult"], binary_odds_predictions, binary_odds_proba,
        labels=[0, 2], label_names=["Away", "Home"],
    )
    log_classification_metrics("Bet365Odds (2-class)", binary_odds_metrics, label_names=["Away", "Home"])

    # --- Regression: Linear Regression ---
    regressor = build_baseline_regressor()
    regressor.fit(train[feature_cols], train["TargetGoalDifference"])

    val_reg_predictions = regressor.predict(validation[feature_cols])
    linreg_metrics = regression_metrics(validation["TargetGoalDifference"], val_reg_predictions)
    log_regression_metrics("LinearRegression", linreg_metrics)

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODELS_PATH / "baseline_logistic_regression.pkl")
    joblib.dump(binary_classifier, MODELS_PATH / "baseline_logistic_regression_binary.pkl")
    joblib.dump(regressor, MODELS_PATH / "baseline_linear_regression.pkl")
    logger.info(f"Saved baseline models to {MODELS_PATH}")


if __name__ == "__main__":
    train_baseline_models()
