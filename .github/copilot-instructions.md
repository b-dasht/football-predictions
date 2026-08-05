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
│ ├── pytorch_models.py
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

**Exception to the above**: the PyTorch Neural Network in `src/pytorch_models.py` (`pytorch_classifier`/`pytorch_regressor`) is an explicit exposure exercise to the library itself, not part of this comparison methodology — it gets a 3-class classifier and a regressor plus their `_no_odds` variants (the ablation is cheap and answers a real "does this model depend on the odds features" question, same as every other model type), but no 2-class variant — that's a whole separate training run, not a cheap ablation, and isn't needed for the exposure exercise's purpose. A hand-written training loop (not a wrapper library like `skorch`) is deliberate, since the point is exposure to PyTorch's own idioms (`nn.Module`, autograd, the manual forward/backward/optimizer-step loop). TensorFlow was also planned for the same purpose but has no wheel compatible with this project's Python version as of when this was tried.

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
AUROC (2-class framing only — see below)

Draws are a minority class and historically the hardest outcome to predict. Overall accuracy alone can look good while draw recall is near zero — always report per-class metrics, not just the aggregate.

Report both the 3-class and 2-class (Home/Away-only) framings side by side per §13 — a model's 3-class accuracy alone doesn't reveal how much of its gap to a competing model is about Draw-handling versus genuine Home/Away discrimination.

AUROC is computed only for the 2-class framing. It's fundamentally a binary metric; a one-vs-rest extension for 3-class would produce a single macro-averaged number that's less actionable than accuracy/log loss already give, so it's deliberately not computed there.

**Regression**

Report:

MAE
RMSE
R²
outcome accuracy (round the predicted goal difference, take its sign, compare to the true result's sign — Home win/Draw/Away win)

Goal difference is a small set of discrete integers (roughly -9 to +9), not a truly continuous quantity, and the sport's inherent randomness caps how high R² can realistically go — MAE/RMSE/R² alone can be hard to read as "is this actually a good model." Outcome accuracy answers that directly, in the same terms as the classification models, without changing what's being predicted.

Always compare models consistently.

**Visualisation (`src/visualisation.py`)**

Every model-count-sensitive comparison is one metric, one file — accuracy, log loss, AUROC, and each per-class score (recall/precision/F1) each get their own single-purpose bar chart, rather than being crammed into shared multi-panel figures that stop being legible as models are added.

Confusion matrices are written as a single markdown table (`reports/confusion_matrices.md`), not a PNG heatmap grid — a heatmap's width scales linearly with model count and becomes unreadable well before 10 models; the raw counts already live in every `models/*.json`.

Goal-difference regression predictions are visualised as box plots of predicted value grouped by the true (discrete integer) value — not a scatter against a continuous y=x line, which never forms a clean diagonal against a quantized x-axis. One grid image per with/without-odds group (small multiples, one panel per model), not one file per model.

The 2-class framing gets one ROC curve chart with every model overlaid (including the Bet365 odds baseline) — the standard way to compare binary classifiers' ranking quality independent of any decision threshold.

`models/*.json` (and therefore every comparison chart above) only ever holds each model's *current* version — a retrain overwrites the previous JSON/pickle by name, it never accumulates. `reports/results_log.csv` is the append-only history of every training run, and is what `plot_tuning_progress()` reads to chart a metric across a model's successive retrains (one line per model name, x-axis = that model's own run sequence, color = family, linestyle = with-odds/no-odds) — the way to see whether hyperparameter tuning (or any other change) actually helped.

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

**Implementation (`src/hyperparameter_tuning.py`)**

Time-aware cross validation is a season-based expanding window, not a random K-fold or sklearn's row-count-based `TimeSeriesSplit`: fold *i* trains on every training season up to and including season *i*, validates on season *i+1* (`season_expanding_splits`). This mirrors the real train→validation setup exactly, at a season granularity. The earliest folds (too little training history to be representative of the final ~5,300-row fit) are skipped via `min_train_seasons`.

`RandomizedSearchCV` (a fixed budget of randomly sampled combinations) is used instead of an exhaustive `GridSearchCV` — it scales predictably across every model type regardless of search-space size, with no extra dependency. Classification is scored on `neg_log_loss`, not accuracy — accuracy alone would let the search ignore Draw entirely (a model that always predicts Home/Away can still score well on it), whereas log loss rewards well-calibrated probabilities across all 3 classes. Regression is scored on `neg_mean_absolute_error`, matching MAE, the primary reported regression metric.

Each model *type* is tuned twice — once for the with-odds feature set, once for the no-odds feature set — as two fully independent `RandomizedSearchCV` runs, each saved under its own key in `models/tuned_hyperparameters.json` (tracked in git) (e.g. `xgboost_classifier` and `xgboost_classifier_no_odds`). This is deliberate, not an oversight: a hyperparameter set tuned around having the odds features available isn't necessarily the right shape once they're removed, and reusing one for both would quietly bias the no-odds variant toward whatever suited the odds-informed problem. The 2-class (`_binary`) variant is always trained with odds included, so it reuses the with-odds entry — no separate search needed there. `src/models.py`'s `load_tuned_params(name)` + `pipeline.set_params(**params)` applies each entry to its matching variant. `LinearRegression` has no meaningful hyperparameters, so the baseline regressor isn't tuned. The PyTorch exposure models are out of scope for tuning (per §13's exception).

**Iterating across multiple tuning rounds**: `tune_all_models(n_iter, only=[...])` can re-run just a subset of entries (loads any existing `tuned_hyperparameters.json` first and only overwrites the requested ones) — for continuing to refine a search space around where a previous round's best values landed on a boundary, or for running a variant's first-ever independent search. Judge each round by its **cross-validated score** where a prior CV score exists for comparison — a round that leaves the CV score essentially unchanged (within noise) but happens to shift the validation result is noise, not improvement, and should be reverted to the previous round's hyperparameters rather than kept; repeatedly chasing validation-set movements across rounds is a backdoor way of tuning against validation, which is exactly what the training-only CV process exists to prevent. When no prior CV score exists to compare against (e.g. a variant's first independent search, judged only against hyperparameters inherited from a different feature set), the actual validation-set result for that variant is the fairest available signal instead — revert to the inherited hyperparameters if the independent search's validation result is worse. Either way, revert by copying the kept hyperparameters into the losing entry's key, never by deleting it — a missing key falls back to raw library defaults via `load_tuned_params`'s `{}` default, not to whatever it should actually reuse. Stop iterating once further rounds stop moving the judging signal.

CV and validation can also *diverge* across rounds, not just plateau — watch for this, not just for a flat CV score. The Neural Network's no-odds regressor improved its CV score for two consecutive rounds (-1.3795 → -1.3707 → -1.3652) while its validation R² went the *opposite* way after the first round (0.074 → 0.107 → 0.049) — each deeper architecture increasingly overfit that search's own CV folds rather than generalizing. A CV score moving in one consistent direction is not on its own a reason to keep pushing; if validation stops tracking it, or reverses, that's the stronger signal, and the answer can legitimately be "no amount of further tuning here beats the inherited/reused hyperparameters" — a real, informative conclusion, not a gap to keep chasing with more rounds.

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

A classification model saved in the 2-class framing (per §13) gets a `_binary` suffix (e.g. `logistic_regression_binary.pkl`), saved alongside its 3-class counterpart — never overwriting it.

**How this is actually implemented**: `evaluation.save_model_with_metadata(model, name, task, framing, metrics)` is the one place every model gets persisted from. It writes three things:
- `models/{name}.pkl` — the fitted pipeline (joblib). Omitted for a baseline that isn't an actual trained model (e.g. the Bet365 odds baseline) — pass `model=None`.
- `models/{name}.json` — the final estimator's hyperparameters (via `.get_params()`) and the full evaluation metrics (including the confusion matrix), human-readable. This is the answer to "what parameters produced this result?" for any specific model — just open its `.json`.
- A row appended to `reports/results_log.csv` — the running performance history across every model ever trained, viewable directly (Data Wrangler, Excel, `pandas.read_csv`) without retraining or re-running anything. Skipped when a model's core metrics (accuracy+log_loss, or MAE+RMSE+R²) exactly match its immediately-preceding logged row — a retrain that didn't actually change anything (e.g. an unrelated project-wide retrain) shouldn't add a duplicate data point to `plot_tuning_progress()`'s trend lines.

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