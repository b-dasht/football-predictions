"""Train and save baseline models: Logistic Regression (a 3-class version, a
separately trained 2-class Home/Away-only variant, and a 3-class variant
with the Bet365 odds features excluded), Linear Regression (plus its own
odds-excluded variant), and the Bet365-odds baseline in the matching
framing (classification only - see docs/EDA_FINDINGS.md for why no
equivalent exists for goal difference among the kept columns). See
.github/copilot-instructions.md #13 for why every classification model
gets both framings, and why a no-odds variant exists.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    feature_cols_no_odds = get_feature_columns(features, include_odds=False)

    # One canonical name per model, used identically everywhere it's
    # referenced - console logs, the .pkl/.json filenames, and the results
    # log/plots - so there's never a mismatch between "what the log called
    # it" and "what file it's saved as".
    LOGREG_NAME = "baseline_logistic_regression"
    LOGREG_NO_ODDS_NAME = "baseline_logistic_regression_no_odds"
    LOGREG_BINARY_NAME = "baseline_logistic_regression_binary"
    LINREG_NAME = "baseline_linear_regression"
    LINREG_NO_ODDS_NAME = "baseline_linear_regression_no_odds"
    ODDS_NAME = "bet365_odds"
    ODDS_BINARY_NAME = "bet365_odds_binary"  # distinct filename - "bet365_odds" alone would collide between framings

    # --- 3-class: Logistic Regression vs. the full Bet365 odds baseline ---
    classifier = build_baseline_classifier()
    classifier.fit(train[feature_cols], train["TargetResult"])

    val_predictions = classifier.predict(validation[feature_cols])
    val_proba = classifier.predict_proba(validation[feature_cols])
    logreg_metrics = classification_metrics(validation["TargetResult"], val_predictions, val_proba)
    log_classification_metrics(f"{LOGREG_NAME} (3-class)", logreg_metrics)
    save_model_with_metadata(classifier, LOGREG_NAME, "classification", "3-class", logreg_metrics)

    # --- 3-class, odds excluded: how much does the model actually depend
    # on the Bet365 features specifically? (They turned out to be the two
    # single most influential features by coefficient magnitude in the
    # with-odds model - worth checking directly rather than assuming.) ---
    classifier_no_odds = build_baseline_classifier()
    classifier_no_odds.fit(train[feature_cols_no_odds], train["TargetResult"])

    no_odds_predictions = classifier_no_odds.predict(validation[feature_cols_no_odds])
    no_odds_proba = classifier_no_odds.predict_proba(validation[feature_cols_no_odds])
    logreg_no_odds_metrics = classification_metrics(validation["TargetResult"], no_odds_predictions, no_odds_proba)
    log_classification_metrics(f"{LOGREG_NO_ODDS_NAME} (3-class)", logreg_no_odds_metrics)
    save_model_with_metadata(classifier_no_odds, LOGREG_NO_ODDS_NAME, "classification", "3-class", logreg_no_odds_metrics)

    odds_predictions, odds_proba = odds_baseline_predictions(validation)
    odds_metrics = classification_metrics(validation["TargetResult"], odds_predictions, odds_proba)
    log_classification_metrics(f"{ODDS_NAME} (3-class)", odds_metrics)
    save_model_with_metadata(None, ODDS_NAME, "classification", "3-class", odds_metrics)

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
    log_classification_metrics(f"{LOGREG_BINARY_NAME} (2-class)", binary_logreg_metrics, label_names=["Away", "Home"])
    save_model_with_metadata(binary_classifier, LOGREG_BINARY_NAME, "classification", "2-class", binary_logreg_metrics)

    binary_odds_predictions, binary_odds_proba = odds_baseline_binary_predictions(validation_binary)
    binary_odds_metrics = classification_metrics(
        validation_binary["TargetResult"], binary_odds_predictions, binary_odds_proba,
        labels=[0, 2], label_names=["Away", "Home"],
    )
    log_classification_metrics(f"{ODDS_BINARY_NAME} (2-class)", binary_odds_metrics, label_names=["Away", "Home"])
    save_model_with_metadata(None, ODDS_BINARY_NAME, "classification", "2-class", binary_odds_metrics)

    # --- Regression: Linear Regression ---
    regressor = build_baseline_regressor()
    regressor.fit(train[feature_cols], train["TargetGoalDifference"])

    val_reg_predictions = regressor.predict(validation[feature_cols])
    linreg_metrics = regression_metrics(validation["TargetGoalDifference"], val_reg_predictions)
    log_regression_metrics(LINREG_NAME, linreg_metrics)
    save_model_with_metadata(regressor, LINREG_NAME, "regression", "regression", linreg_metrics)

    # --- Regression, odds excluded ---
    regressor_no_odds = build_baseline_regressor()
    regressor_no_odds.fit(train[feature_cols_no_odds], train["TargetGoalDifference"])

    no_odds_reg_predictions = regressor_no_odds.predict(validation[feature_cols_no_odds])
    linreg_no_odds_metrics = regression_metrics(validation["TargetGoalDifference"], no_odds_reg_predictions)
    log_regression_metrics(LINREG_NO_ODDS_NAME, linreg_no_odds_metrics)
    save_model_with_metadata(regressor_no_odds, LINREG_NO_ODDS_NAME, "regression", "regression", linreg_no_odds_metrics)


if __name__ == "__main__":
    train_baseline_models()
