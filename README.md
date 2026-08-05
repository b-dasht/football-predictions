# Football Predictions

Machine learning models predicting English Premier League match outcomes (Home/Draw/Away) and goal difference, built as a demonstration of rigorous ML methodology — reproducible data processing, leakage-free feature engineering, chronological validation, and honest comparison against a strong external baseline (Bet365's own odds) — as much as of raw predictive performance.

## What this is

Two prediction tasks, trained on 16 seasons (2010/11–2025/26) of Premier League results and Bet365 odds from [football-data.co.uk](https://www.football-data.co.uk/):

- **Match outcome** (classification): Home win / Draw / Away win, plus a separately-trained Home-vs-Away-only variant.
- **Goal difference** (regression): `Home Goals − Away Goals`.

Every model is trained and compared two ways: with Bet365's odds included as a feature, and without — to measure how much each model actually depends on the bookmaker's own information rather than assuming it from coefficients alone.

**Read [`docs/EVALUATION_FINDINGS.md`](docs/EVALUATION_FINDINGS.md) for the final results, model comparison tables, and what we'd try next.**

## Project structure

```
data/            raw, interim, and processed match data
notebooks/       exploratory analysis (run interactively)
src/             all reusable code — data loading, features, models, evaluation, tuning, plots
models/          trained pipelines (.pkl, gitignored) + metadata/metrics (.json, tracked)
reports/         results_log.csv (every training run's history) + figures/ (generated plots)
tests/           pytest suite
docs/            background, methodology rationale, and findings
.github/         binding coding rules (copilot-instructions.md) — the source of truth for how this repo works
```

## Running it

```bash
pip install -r requirements.txt

python -m src.data_loader          # download/cache raw season CSVs
python -m src.preprocessing        # clean into data/interim/matches.csv
python -m src.feature_engineering  # build data/processed/features.csv
python -m src.models                # train every model, all framings
python -m src.hyperparameter_tuning # re-run hyperparameter search (optional - already tuned)
python -m src.visualisation         # regenerate every comparison chart into reports/figures/

pytest                              # run the test suite
```

Each step reads the previous stage's output, so they're safe to re-run individually as the data or code changes.

## Where to look next

- [`docs/EVALUATION_FINDINGS.md`](docs/EVALUATION_FINDINGS.md) — final model comparison, key findings, recommended models per task, and future directions (more features, model ensembling).
- [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) — the "why": background, rationale, dataset context.
- [`docs/EDA_FINDINGS.md`](docs/EDA_FINDINGS.md) — exploratory data analysis findings that shaped the feature engineering.
- [`.github/copilot-instructions.md`](.github/copilot-instructions.md) — the binding methodology rules (data leakage prevention, validation strategy, model development, evaluation, hyperparameter tuning) — treat this as the source of truth for *how* the project works.
- `reports/confusion_matrices.md` and `reports/figures/` — generated comparison artifacts (regenerate with `python -m src.visualisation`).
