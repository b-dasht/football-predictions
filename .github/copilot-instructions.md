# Football Machine Learning Prediction Project — Development Guidelines

This file holds the binding coding rules and is the source of truth whenever it and [PROJECT_OVERVIEW.md](../docs/PROJECT_OVERVIEW.md) disagree. See PROJECT_OVERVIEW.md for the full rationale and background behind the project.

## 1. Project Objective

This project aims to develop and evaluate different machine learning models for English Premier League football predictions.

The project contains two prediction tasks:

### Model A — Match Outcome Classification

Predict:
- Home Win
- Draw
- Away Win

### Model B — Goal Difference Regression

Predict:
Home Goals - Away Goals

Examples:
2-1 → +1
0-3 → -3
2-2 → 0

The primary objective is not only predictive performance, but demonstrating a rigorous machine learning workflow:
- reproducible data processing
- prevention of data leakage
- appropriate model selection
- validation methodology
- interpretation of model performance

---

## 2. General Development Principles

## Write production-quality code

Code should be:
- readable
- maintainable
- modular
- reproducible

Avoid writing disposable scripts.
Notebooks are for exploration and reporting. Core functionality belongs in Python modules.

---

## Avoid unnecessary complexity

Prefer:
- simple solutions
- clear code
- reusable functions

Avoid:
- unnecessary abstractions
- excessive classes
- premature optimisation
- adding dependencies without need

---

## Work incrementally

Implement one component at a time.

Examples:
Good:
Add match data loader
Add rolling team statistics calculation

Avoid:
Rewrite entire pipeline and add new models simultaneously
Each change should have a clear purpose.

---

## 3. Project Structure

Maintain the following structure:
football-ml/

├── data/
│ ├── raw/
│ ├── interim/
│ └── processed/
│
├── notebooks/
│
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── models.py
│ ├── evaluation.py
│ ├── visualisation.py
│ └── utils.py
│
├── models/
│
├── reports/
│ └── figures/
│
├── logs/
│
├── tests/
│
├── requirements.txt
│
└── README.md

Rules:
- Raw data must never be modified.
- Notebooks should not contain reusable business logic.
- Avoid duplicate implementations.
- Keep functions inside appropriate modules.

---

## 4. Repository Discipline

Treat the repository as the single source of truth.

Do not create duplicate versions. Avoid:
feature_engineering_v2.py
feature_engineering_final.py
feature_engineering_new.py

Instead:
- update existing modules
- use Git history to track changes

Before creating a new file, ask:

> Does this represent a separate responsibility?

If not, modify an existing file.

---

## 5. Configuration Management

Use one configuration file: src/config.py


All constants and parameters should be stored here.

Examples:

DATA_PATH = "data/raw"
TRAIN_START_SEASON = "2010-11"
VALIDATION_START_SEASON = "2024-25"
VALIDATION_END_SEASON = "2025-26"
RANDOM_STATE = 42
ROLLING_WINDOWS = [5, 10]

Do not hard-code:
file paths
seasons
model parameters
random seeds

throughout the codebase.

Import configuration values:
from src.config import RANDOM_STATE

## 6. Logging

Use loguru for logging.

Each module should use: 
from loguru import logger

Use logging instead of print statements.

Examples:
logger.info("Loading match data")
logger.warning("Missing values detected")
logger.error("Failed to process dataset")

Logging should:
capture important events
aid debugging
avoid excessive output

Do not log inside large loops unless necessary.

## 7. Coding Style
Follow PEP8.

Use:
meaningful variable names
clear functions
type hints where useful

Example:
def calculate_goal_difference(
    home_goals: int,
    away_goals: int
) -> int:
    return home_goals - away_goals

Avoid:
def calc(x,y):

Comments should explain why something is done.

Avoid:
# Calculate average
average = values.mean()

Prefer:
# Rolling averages prevent future information leakage by only using previous matches
rolling_average = values.rolling(window=5).mean()

Do not add unnecessary comments explaining obvious code.

## 8. Data Handling Rules

Follow this pipeline:
Raw Data

↓

Clean Data

↓

Feature Dataset

↓

Model Input

Never modify raw files.

Always validate:
dataframe shape
column names
missing values
duplicates
data types

## 9. Prevent Data Leakage
Football prediction is a time-dependent problem.

All features must represent information available before kick-off.

Allowed:
previous results
previous goals
previous form
previous league position
previous team statistics

Not allowed:
final league position
season averages calculated using future matches
post-match statistics

Before adding a feature ask:
Would this information have been available immediately before the match started?

If not, it cannot be used.

## 10. Data Splitting Strategy
Do not randomly shuffle matches.

Use chronological splitting:
Training

2010/11 → 2023/24


Validation

2024/25 → 2025/26 (two seasons, not one — see rationale below)


Testing

2026/27

**Why two validation seasons, not one (as of 2026-08-04):** a single validation season (380 matches) split three ways by outcome leaves the minority Draw class with only ~100 examples — too few to trust a close model-comparison result (e.g. an accuracy gap of 1-2 percentage points) as more than single-season noise. Using two seasons (~760 matches) roughly halves that noise at the cost of one fewer training season (~7% less training data), which is a reasonable trade since Hyperparameter Tuning uses time-aware cross-validation *within* the training set regardless, so it doesn't depend on a large held-out validation set for its own robustness.

**Timeline currency note:** this split is still time-sensitive and needs periodic review, not one-time definition. The 2025/26 validation season has already completed, and 2026/27 is about to start. Before running any evaluation, confirm in `src/config.py` which seasons are genuinely "unseen" at the time the model was frozen — a season is only a valid validation/test set if none of its matches were available during training or tuning.

## 11. Feature Engineering

Features should be generated only from previous matches.

Potential features:
Team Form:
previous 5 match points
previous 5 goals scored
previous 5 goals conceded
win percentage:
Team Strength
league position
goal difference
Elo rating (future improvement)
Home/Away Performance:
home win percentage
away win percentage
average goals
Momentum:
winning streak
unbeaten streak
Rest
days since previous match

Cold-start handling:
Rolling features (e.g. previous 5/10 matches) are undefined for the first few fixtures of a team's season and for promoted teams with no top-flight history. Define an explicit rule (e.g. NaN + imputation flag, or fall back to previous-season figures) rather than silently dropping rows or filling with zeros, which would misrepresent a team's strength.

## 12. Machine Learning Workflow
Follow this order:
Data collection

↓

Cleaning

↓

Exploration

↓

Feature engineering

↓

Train/validation split

↓

Preprocessing pipeline

↓

Baseline model

↓

Advanced models

↓

Hyperparameter tuning

↓

Evaluation

↓

Final model

Do not skip baseline models.

## 13. Model Development

Classification models, compare:

Logistic Regression
Support Vector Machine
Random Forest
XGBoost
Neural Network

If the source data includes bookmaker odds columns (e.g. `B365H`/`B365D`/`B365A` from football-data.co.uk), derive implied-probability features and/or treat them as an additional baseline. Bookmaker odds are a strong, well-calibrated benchmark for match outcome prediction — models should be judged against this, not only against Logistic Regression.

Every classification model must be trained and evaluated in **two framings**, not just one:
- **3-class** (Home/Draw/Away) — the full task.
- **2-class** (Home/Away only) — a separately trained model, fit only on non-draw matches, never able to predict Draw at all. This is not the 3-class model's marginal Home/Away probabilities; it's a genuinely different model fit on a filtered training set.

Both framings get compared against the Bet365 odds baseline, restricted to the matching framing (all three outcomes for the 3-class comparison; Bet365's Home/Away implied probabilities renormalized, with Draw's share excluded, for the 2-class comparison). This exists because a single 3-class comparison conflates two different questions — "can the model call Home vs Away correctly" and "can it handle the Draw case" — and the odds baseline's apparent edge in the 3-class framing turned out to be partly (not entirely) attributable to Draw-handling specifically, only visible once the two questions are separated.

Every 3-class classification model and every regression model must also be trained in a **no-odds variant** (`get_feature_columns(df, include_odds=False)`, same model architecture, saved with a `_no_odds` suffix) — a direct check on how much the model actually depends on the Bet365 features, rather than assuming from coefficient/feature-importance magnitude alone. This matters because the two can disagree: in the baseline Logistic Regression, `ImpliedProbHome`/`ImpliedProbAway` were individually the two highest-magnitude coefficients of all 83 features, yet removing all three odds features barely changed 3-class accuracy (49.1% → 49.2%) — evidently redundant with the rolling-form features for *which team wins*. The regression model told a different story: removing odds meaningfully hurt it (R² 0.187 → 0.126) — the odds carry real information about match margin that the rolling stats don't fully capture. Not scoped to the 2-class framing too, to keep the comparison matrix from growing 4x per model — a no-odds check on the main (3-class/regression) framing is enough to answer the dependency question.

Regression models, compare:

Linear Regression
Support Vector Regression
Random Forest Regressor
XGBoost Regressor
Neural Network Regressor

## 14. Preprocessing

Use scikit-learn pipelines.

Example:
Pipeline(
    [
        ("preprocessor", preprocessing),
        ("model", model)
    ]
)

Rules:

preprocessing must be fitted only on training data
validation/test data must use the fitted transformer
save complete pipelines

## 15. Model Evaluation

**Classification**

Report:

accuracy
precision
recall
F1 score
confusion matrix
log loss

Draws are a minority class and historically the hardest outcome to predict. Overall accuracy alone can look good while draw recall is near zero — always report per-class metrics, not just the aggregate.

Report both the 3-class and 2-class (Home/Away-only) framings side by side per §13 — a model's 3-class accuracy alone doesn't reveal how much of its gap to a competing model is about Draw-handling versus genuine Home/Away discrimination.

**Regression**

Report:

MAE
RMSE
R²

Always compare models consistently.

## 16. Hyperparameter Tuning

Tune models only using training data.

Preferred approach:

Training data

↓

Time-aware cross validation

↓

Hyperparameter optimisation

↓

Final validation on unseen season

Do not tune against the validation season.

## 17. Model Saving

Save trained models using:
joblib

Store:

model
preprocessing pipeline
parameters
evaluation results

Example:
models/

├── xgboost_classifier.pkl

├── random_forest_regressor.pkl

└── preprocessing.pkl

A classification model saved in the 2-class framing (per §13) gets a `_binary` suffix (e.g. `baseline_logistic_regression_binary.pkl`), saved alongside its 3-class counterpart — never overwriting it.

**How this is actually implemented**: `evaluation.save_model_with_metadata(model, name, task, framing, metrics)` is the one place every model gets persisted from. It writes three things:
- `models/{name}.pkl` — the fitted pipeline (joblib). Omitted for a baseline that isn't an actual trained model (e.g. the Bet365 odds baseline) — pass `model=None`.
- `models/{name}.json` — the final estimator's hyperparameters (via `.get_params()`) and the full evaluation metrics (including the confusion matrix), human-readable. This is the answer to "what parameters produced this result?" for any specific model — just open its `.json`.
- A row appended to `reports/results_log.csv` — one row per model per training run, every time. This is the running performance history across every model ever trained, viewable directly (Data Wrangler, Excel, `pandas.read_csv`) without retraining or re-running anything.

`.pkl` files are gitignored (regenerable, large binaries); `.json` files and `results_log.csv` are **not** — both are small and diffable, so git history itself shows how a model's parameters/results changed over time.

`src/visualisation.py`'s comparison plots read directly from `models/*.json` (grouped by task/framing) — every model trained shows up automatically, with no per-model list to maintain by hand.

## 18. Notebook Guidelines
Notebooks should:

run from top to bottom
have clear sections
avoid duplicated code

Structure:
Imports
Load data
Explore data
Train model
Evaluate model
Visualise results

Reusable code belongs in src/.

## 19. Testing
Add tests for important functions.

Examples:

data loading
feature generation
target calculation

Example:
def test_goal_difference():
    assert calculate_goal_difference(3, 1) == 2
Do not write tests for trivial code.

## 20. Dependencies

Maintain:
requirements.txt

Only include required packages.

Preferred libraries:
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
loguru
joblib

## 21. Git Practices

Use clear commits.

Good:
Add rolling team form features
Implement XGBoost classification baseline

Bad:
Changes
Updates

Commit after completing logical units.

## 22. AI Coding Assistant Instructions
When generating or modifying code:
Inspect existing code before making changes.
Follow the current project structure.
Do not create unnecessary files.
Match the existing coding style.
Keep changes small and focused.
Explain major design decisions briefly.
Avoid rewriting working code without reason.
Prioritise correctness over complexity.

## 23. Project Philosophy
The goal is not to build the most complicated model.

The goal is to demonstrate:

rigorous machine learning methodology
good software engineering practices
careful validation
thoughtful feature engineering
understanding of model limitations

A simpler, well-designed model is preferable to a complex model with poor methodology.