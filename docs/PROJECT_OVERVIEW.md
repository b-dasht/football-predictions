# Project Overview — English Premier League Machine Learning Prediction System

This document covers the *why* and background context for the project. For binding coding rules, project structure, and how AI coding assistants should behave in this repo, see [copilot-instructions.md](../.github/copilot-instructions.md) — that file is the source of truth when the two overlap. For a glossary of the raw data columns, see [DATA_DICTIONARY.md](DATA_DICTIONARY.md). For exploration conclusions, see [EDA_FINDINGS.md](EDA_FINDINGS.md).

## 1. Project Purpose

The aim of this project is to build an end-to-end machine learning system that predicts English Premier League football matches using historical match data.
The project is intended to demonstrate a complete machine learning workflow rather than simply train a predictive model. The focus is on:

- robust data processing
- meaningful feature engineering
- preventing data leakage
- selecting appropriate machine learning algorithms
- comparing model performance
- understanding why different models perform differently
- producing reproducible and maintainable code

---

# 2. Main Objectives

The project will develop two separate predictive models using the same underlying football dataset.

## Model A — Match Outcome Prediction (Classification)
The first model will predict the result of a football match.

Target:
- Home win
- Draw
- Away win

This is a multiclass classification problem.
Example:
A match between Manchester City and Arsenal:

Prediction:
Home win probability: 0.55
Draw probability: 0.25
Away win probability: 0.20

The model should output both the predicted class and ideally the probability distribution across outcomes.

---

## Model B — Match Goal Difference Prediction (Regression)

The second model will predict the expected goal difference.

The target variable is:
Home Goals - Away Goals

Examples:
Liverpool 2 - 1 Chelsea

Target:
+1

Arsenal 0 - 3 Manchester United
Target:
-3

Tottenham 2 - 2 Newcastle

Target:
0

This regression output can then be interpreted as expected match dominance and converted into an expected outcome if required.

---

# 3. Dataset

The primary dataset will come from:

https://datahub.io/football/english-premier-league

The dataset contains historical Premier League fixtures and match statistics.

The project will initially focus on Premier League data from approximately:

2010/11 season onwards through to the 2025/26 season

The reason for limiting the timeframe is to reduce concept drift. Football has changed significantly over multiple decades due to tactical evolution, financial changes, and differences in data availability and how data is logged.

---

# 4. Validation Strategy

Football prediction is a time-dependent forecasting problem.

Random train/test splitting must not be used because it allows future information to influence past predictions.

The dataset will be split chronologically.

## Training Data

2010/11 → 2023/24
Used for:
- feature development
- model training
- cross-validation
- hyperparameter tuning

---

## Validation Data

2024/25 → 2025/26 (two seasons)
Used as genuinely unseen seasons to:
- compare models
- select the best approach
- evaluate generalisation

Two seasons rather than one: a single validation season only has ~380 matches, and split three ways by outcome the minority Draw class drops to roughly 100 examples — too few to trust a close model comparison as more than single-season noise. Two seasons (~760 matches) meaningfully reduces that noise, at the cost of one fewer training season - a reasonable trade since Hyperparameter Tuning gets its own robustness from time-aware cross-validation within the training set, not from a large validation set.

---

## Future Test Data

2026/27 season

Used as a future prediction benchmark once the final model has been selected.
The final model should be frozen before evaluation on future matches.

**Timeline currency note (as of 2026-08-04):** the 2025/26 season has already finished, and 2026/27 is starting imminently. This split needs to be re-checked against the calendar before every training/evaluation run. If the model is retrained after new results are available, re-confirm which seasons are actually playing the "unseen validation" role — a season stops being valid for that purpose the moment its matches were used in training or tuning.

---

# 5. Feature Engineering Approach

*(This is the rationale behind the feature list; [copilot-instructions.md §11](../.github/copilot-instructions.md) holds the canonical list plus the cold-start handling rule — update both if the feature set changes.)*

The most important part of the project is creating realistic pre-match features.
The model must only use information available before a match begins.
No post-match information can be used.
For every fixture, features should be generated from previous matches.

Potential features include:

Team Form

Examples:
- points gained in previous matches
- wins/losses/draws
- goals scored
- goals conceded
- goal difference
- rolling averages over recent matches

Possible rolling windows:
- last 5 matches
- last 10 matches

Team Strength

Examples:
- league position before the fixture
- historical performance
- rolling goal difference
- Elo rating (possible future enhancement)

Home and Away Performance

Examples:
Home team:
- home win percentage
- average home goals scored
- average home goals conceded

Away team:
- away win percentage
- average away goals scored
- average away goals conceded

Match Context

Potential features:
- days since previous fixture
- recent winning streak
- unbeaten streak
- promoted team indicator

---

# 6. Machine Learning Models

The project will compare multiple algorithms to understand their strengths and weaknesses.

If the raw data includes bookmaker odds (the football-data.co.uk source typically does), the implied probabilities from those odds are a strong external baseline — historically hard for a from-scratch model to beat. Comparing against them, not just against Logistic Regression, gives a more honest read on whether the modelling effort is adding real value.

Every classification model is trained and evaluated two ways: the full **3-class** task (Home/Draw/Away), and a separately trained **2-class** model (Home/Away only, fit only on non-draw matches, never able to predict Draw). Both are compared against the Bet365 odds baseline in the matching framing. This split matters because Draws are a genuinely hard minority class (see `docs/EDA_FINDINGS.md`) — a single 3-class comparison conflates "how good is this model at Home vs Away" with "how good is it at spotting Draws," and testing showed the odds baseline's apparent edge over Logistic Regression in the 3-class framing was partly, not entirely, a Draw-handling effect: the gap shrank substantially (though didn't disappear) once Draw was removed as an option for both.

Classification Models
Model A will compare:

Logistic Regression
Purpose:
- simple baseline model
- interpretable linear relationships

Support Vector Machine
Purpose:
- investigate nonlinear decision boundaries
- compare kernel methods

Random Forest

Purpose:
- investigate ensemble tree methods
- capture nonlinear feature interactions

XGBoost

Purpose:
- investigate gradient boosting performance on structured data

Neural Network

Purpose:
- investigate whether deep learning provides benefits for tabular football data

Regression Models

Model B will compare:

Linear Regression

Baseline continuous prediction model.

Support Vector Regression

Investigate kernel-based regression.

Random Forest Regression

Capture nonlinear relationships.

XGBoost Regression

Expected strong performance on structured tabular data.

Neural Network Regression

Investigate deep learning approaches.

---

# 7. Evaluation Strategy

Models should not only be judged by predictive accuracy.

The project should evaluate:

Classification

Metrics:

- accuracy
- precision
- recall
- F1 score
- confusion matrix
- log loss
- AUROC (2-class Home/Away framing only)

Regression

Metrics:

- mean absolute error (MAE)
- root mean squared error (RMSE)
- R²
- outcome accuracy (does the sign of the predicted goal difference match the true result?)

Goal difference is a small set of discrete integers, not a truly continuous quantity, so it's visualised as a box plot of predictions grouped by the true value rather than a scatter against a continuous y=x line — see `.github/copilot-instructions.md` §15.

---

# 8. Model Comparison Philosophy

The purpose is not simply to find the highest-performing model.

The project should analyse:

- why some models perform better
- whether increased complexity improves results
- trade-offs between performance and interpretability
- whether simpler models provide competitive results

For example:

XGBoost may perform strongly because football data is structured and contains nonlinear interactions, while logistic regression provides a useful baseline because it is simple and interpretable.

The project should test these assumptions rather than assume them.

The 3-class vs. 2-class comparison (§6) is a concrete instance of this: rather than assuming the odds baseline's 3-class edge over Logistic Regression meant it was simply the better model overall, isolating the Home/Away-only question showed the truth was more specific — real, but smaller, once Draw-handling was controlled for. See `.github/copilot-instructions.md` §13/§15 for the binding rule this became.

A second instance: the Bet365 odds features had the two highest-magnitude coefficients of all 83 features in the baseline Logistic Regression, which looked like the model was leaning heavily on them - but training a no-odds variant showed classification accuracy was essentially unaffected (49.1% → 49.2%), while the regression model's R² dropped meaningfully (0.187 → 0.126) without them. Coefficient size alone would have suggested "the model depends on odds" without qualification; the actual dependence turned out to differ by task. See `.github/copilot-instructions.md` §13 for the no-odds variant this became a standing requirement for.

---

# 9. Software Engineering Approach

The project should be developed as a maintainable software project rather than a collection of notebooks.

Principles:

- reusable Python modules
- clear separation of data processing and modelling
- reproducible experiments — every model trained gets a persistent, inspectable record: its fitted pipeline, its hyperparameters, and its evaluation results (see `.github/copilot-instructions.md` §17 for the mechanism, `reports/results_log.csv` for the running history across every model)
- configuration management
- logging
- testing
- version control discipline

Notebooks should mainly be used for:

- exploration
- visualisation
- reporting results

Core functionality should exist in Python scripts/modules.

---

# 10. Expected Project Structure

See [copilot-instructions.md §3](../.github/copilot-instructions.md) for the canonical directory layout — keep it defined in one place only.

---

# 11. Final Deliverable

The final project should contain:

- cleaned football dataset pipeline
- feature engineering pipeline
- multiple trained machine learning models
- comparison of model performance
- evaluation on unseen season data
- saved final models
- documentation explaining methodology and results

The final output should demonstrate the ability to design, implement, evaluate, and explain a complete machine learning system.