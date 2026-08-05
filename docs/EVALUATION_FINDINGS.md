# Evaluation Findings

Final results and conclusions from Advanced Models + Hyperparameter Tuning, kept here so they're easy to reference without re-reading `reports/results_log.csv`, `models/*.json`, or the chat history that produced them. Numbers below are read directly from `models/*.json` (the current, final version of every model) — regenerate `reports/figures/` via `python -m src.visualisation` to see them plotted.

## Final Model Comparison

**3-class (Home/Draw/Away), with odds — accuracy:**

| Model | Accuracy | Log Loss |
|---|---|---|
| Bet365 odds (baseline) | 51.6% | 0.995 |
| XGBoost | 51.2% | 0.999 |
| Logistic Regression | 51.2% | 1.010 |
| Random Forest | 50.8% | 0.998 |
| SVM | 49.9% | 1.020 |
| Neural Network | 49.6% | 1.032 |

**3-class, no odds — accuracy:**

| Model | Accuracy |
|---|---|
| Logistic Regression | 50.1% |
| SVM | 49.9% |
| XGBoost | 48.9% |
| Random Forest | 48.2% |
| Neural Network | 47.1% |

**2-class (Home/Away only) — accuracy and AUROC:**

| Model | Accuracy | AUROC |
|---|---|---|
| Bet365 odds (baseline) | 69.6% | 0.762 |
| XGBoost | 69.6% | 0.759 |
| Random Forest | 68.0% | **0.762** |
| Logistic Regression | 69.3% | 0.753 |
| SVM | 67.1% | 0.744 |
| Neural Network | 66.8% | 0.723 |

**Regression (goal difference), with odds:**

| Model | MAE | R² | Outcome Accuracy |
|---|---|---|---|
| Linear Regression | 1.251 | **0.187** | 45.1% |
| XGBoost | 1.255 | 0.184 | 45.3% |
| Random Forest | 1.261 | 0.179 | **46.8%** |
| SVM | 1.273 | 0.147 | 43.3% |
| Neural Network | 1.286 | 0.125 | 44.2% |

**Regression, no odds:**

| Model | MAE | R² | Outcome Accuracy |
|---|---|---|---|
| Linear Regression | 1.291 | **0.126** | **44.7%** |
| XGBoost / SVM / Neural Network | 1.28–1.31 | 0.113–0.117 | 40.7–43.7% |
| Random Forest | 1.284 | 0.119 | 43.2% |

## Key Findings

- **Draw is essentially unpredictable for every tuned model.** After hyperparameter tuning, Random Forest, SVM, and XGBoost all converged to **0% Draw recall** — they never predict a draw at all. Logistic Regression and the Neural Network do slightly better (1–3%), but still barely attempt it. The untuned PyTorch exposure model, oddly, has the *highest* Draw recall of any model (11–14%) — a side effect of never having been pushed toward a log-loss-optimal decision boundary the way the tuned models were. This is a real, worth-knowing tradeoff: tuning for log loss/accuracy pushes every model toward *never* gambling on the minority class, which looks good in aggregate metrics but means the project's models are, in practice, binary (Home vs. Away) classifiers wearing a 3-class label.
- **Bet365's odds are a genuinely tough, well-calibrated benchmark.** The best models now sit within 0.4 percentage points of it on the 3-class task, and XGBoost ties it exactly on the 2-class task. Random Forest's AUROC (0.762) actually matches Bet365's — its probability *rankings* are excellent even though its default-threshold accuracy trails.
- **Odds-dependency differs by task, confirming the original no-odds ablation rationale (§13).** Classification accuracy barely moves without odds (Logistic Regression: 51.2% → 50.1%); regression R² drops much more (Linear Regression: 0.187 → 0.126). The odds carry real information about *margin of victory* that the rolling-form features don't fully capture, but comparatively little extra information about *who wins*.
- **The simplest model wins the no-odds problem, on both tasks.** Logistic Regression is the best no-odds 3-class classifier (50.1%, beating XGBoost's 48.9%) and Linear Regression is the best no-odds regressor on both R² and outcome accuracy. With less signal available, a simpler, more regularized model generalizes better than a higher-capacity one — the opposite of the with-odds picture, where XGBoost and Random Forest lead.
- **Hyperparameter tuning helped exactly where it was expected to, and stopped helping exactly when the evidence said so.** XGBoost (untuned: 46.4% 3-class) and the Neural Network (untuned: 43.4%) both improved substantially — they were undertuned defaults, not near any real ceiling. Random Forest and SVM were already close to their with-odds ceiling and barely moved. Multiple further tuning rounds (documented in `.github/copilot-instructions.md` §16) showed cross-validation scores improving while validation performance plateaued or reversed — direct, repeated empirical evidence that further search stopped finding anything that generalizes, for both the with-odds and no-odds problems.
- **The no-odds ceiling is an information gap, not a tuning gap.** Independently tuning the no-odds variants (rather than reusing with-odds hyperparameters) gave mixed results — genuine improvements for XGBoost's classifier and two regressors, but no improvement anywhere the search was pushed further (the no-odds Neural Network, in particular, failed 5 independent tuning attempts across both tasks). No-odds models simply don't have access to information the with-odds models do (insider team news, market sentiment aggregated across thousands of bettors) — more tuning can't manufacture that.
- **Bet365's own ceiling is informative context, not just a target.** A professional bookmaker with far more information than this project has access to — team news, injuries, lineups, market activity — still only reaches ~51.6%/~69.6%. That's strong indirect evidence that a large share of Premier League match outcomes is genuinely irreducible (refereeing decisions, deflections, in-game injuries, moment-to-moment luck), not a sign that a better model or more tuning would close the gap to certainty.

## Recommended Final Model, Per Task

Not a single "winner" — the honest answer differs by what the model needs to do:

- **3-class match outcome, with odds**: **Logistic Regression** and **XGBoost** are statistically tied (51.2% each). Per the project's own philosophy (§23 — prefer the simpler model when performance is equal), Logistic Regression is the more defensible pick: fully interpretable coefficients, no hyperparameter-tuning fragility, and equal accuracy.
- **2-class Home/Away, with odds**: **XGBoost** — clear best on both accuracy (ties Bet365 exactly) and a strong AUROC.
- **3-class, no odds**: **Logistic Regression** — clear best, and the simplest model to boot.
- **Goal-difference regression, with or without odds**: **Linear Regression** for R², **Random Forest** for outcome accuracy (with odds only) — genuinely close enough that either is defensible; Linear Regression's simplicity breaks the tie under the same "prefer simple" principle.
- **PyTorch models**: not part of this comparison (exposure exercise, per §13) — but worth remembering their unusually high Draw recall as a design idea, not a production candidate.

## Future Directions

Hyperparameter tuning has been exhausted for the current feature set (see `.github/copilot-instructions.md` §16) — genuinely moving performance further from here requires new information sources or a different way of combining what's already been built, not more search.

### More Features

Roughly in order of how much new data-collection effort each would need:

**Already have the raw data — needs feature engineering, not new data:**
- **Referee tendencies.** `docs/EDA_FINDINGS.md`'s Referee Effect section already found a real spread (33%–54% home-win rate across referees with ≥50 matches) but flagged a real confound: senior referees tend to officiate bigger clubs' matches, which have higher home-win rates regardless of who's whistling. A usable feature would need to control for the teams involved (e.g. a referee's deviation from *that team's own* average, not the league average) before trusting it — the Man Utd/Howard Webb spot-check in EDA_FINDINGS.md found no effect once controlled this way, so this needs care, not just a raw referee-average column.

**Would need new data, likely via scraping (free sources exist, but come with rate-limit/terms-of-service considerations, and effort scales with how far back in history you want it):**
- **Starting lineups and injuries/suspensions.** Which specific players are missing matters more than a team's rolling-average form suggests — a team missing three regular starters is weaker than its history implies. `football-data.co.uk` (this project's only data source so far) has no player-level data at all. Historical lineups are available from sites like FBref (structured match-report pages, StatsBomb-powered stats) or Wikipedia match reports, but there's no clean CSV download — this means actual web scraping, page-by-page, respecting each site's terms of use and rate limits. Getting this right across 5,300+ historical matches is a substantial data-engineering project in its own right, not a quick add.
- **Manager identity and tenure.** The well-documented "new manager bounce" (a short-term results uplift right after a managerial change) isn't currently modeled at all. Manager history/tenure isn't in the existing data source; it would need scraping from something like Wikipedia's per-club managerial history tables. A relatively cheap feature to build once the appointment-date data exists: days since the current manager started, or a flag for "changed manager in the last N matches."
- **Squad market value** (a proxy for underlying team strength independent of recent form) — sites like Transfermarkt publish this, again scraping-only, no official free API, and their terms of use should be checked before any automated collection.
- **Attendance.** Already flagged as absent from this dataset in `docs/EDA_FINDINGS.md` (the 2020/21 COVID season's empty stadiums correlate with a home-advantage drop, but there's no per-match attendance figure to build a general feature from). Would need an external historical-attendance source.
- **Fixture congestion / true rest days.** `docs/EDA_FINDINGS.md` explicitly considered and rejected this already: computing genuine days-since-last-match requires matches from *other* competitions a team played in between league fixtures (Champions League, League Cup, FA Cup) — data this project doesn't have. Worth revisiting only alongside a source that covers all competitions a team plays in, not just the league.

**Would need a paid data provider — not realistically free:**
- **Professional-grade player tracking / advanced stats** (expected goals, pressing intensity, passing networks) from providers like Opta, StatsBomb, or Sportmonks — the data clubs and serious analytics operations actually use, but licensed and priced accordingly.
- **Historical odds movement ("steam").** How odds shift between opening and kickoff often reflects information the market has absorbed that a single closing-line snapshot (all this project currently has) misses. Betfair Exchange's historical data or OddsPortal both offer this, generally as paid historical datasets.

### Model Ensembling / Stacking

The models in this project make genuinely different kinds of errors (a linear model, tree ensembles, a neural network, and the market's own odds) — combining them is a natural next step that doesn't require any new data:

- **Simple averaging**, the cheapest option: average the predicted probabilities of the strongest few models (e.g. Logistic Regression + XGBoost + Random Forest). Averaging tends to reduce variance and can beat every individual input even without any learned weighting.
- **Stacking with a meta-learner**: train a second-stage model (a simple Logistic Regression is the usual choice) on the *out-of-fold* predictions of the base models. This needs real care to avoid leakage — the meta-learner must never see a base model's prediction on a row that model was trained on, or it will overfit to which base model happens to memorize best, not which one generalizes best. `hyperparameter_tuning.py`'s existing `season_expanding_splits()` CV scheme is the natural place to generate those out-of-fold predictions correctly.
- **Blending with Bet365 directly.** Since the odds already encode information this project's models don't have access to (see above), a weighted blend of "our model's probability" and "Bet365's implied probability" is a standard, practical technique in this domain — sometimes beating either input alone, since the market may slightly underweight patterns a model captures well from history.
- **Caveat worth testing empirically, not assuming:** every model here is trained on the *same* rolling-form and odds features. Their errors may be more correlated than truly independent models would be, which limits how much any ensembling approach can add — this needs to be measured (e.g. checking how correlated the models' validation-set errors actually are) before assuming a stacked model will clearly beat the best single one.

### Other ideas, briefly

- **More training data**: more seasons, or transferring patterns from other leagues — traded off against older seasons reflecting an increasingly different game (tactics, rules, squad rotation norms all drift over 15+ years).
- **A different target formulation**: a football-specific model like Dixon-Coles (a Poisson-based model for each team's goal-scoring rate) is a well-established alternative to a generic classifier/regressor and might capture the discrete, low-scoring nature of football more naturally than an off-the-shelf model built for general tabular data.
- **Calibration analysis**: checking whether each model's *predicted probabilities* are trustworthy (e.g. "of all matches predicted at 60% Home win, do about 60% actually finish as a Home win?"), not just whether its hard calls are accurate — complements outcome accuracy rather than replacing it.

## Limitations

- **Small by modern ML standards.** ~5,300 training rows is enough for the model types used here, but rules out most deep-learning approaches that need far more data to outperform simpler methods — consistent with the PyTorch/Neural Network models' struggles here relative to the classical ones.
- **Single competition, single country.** Everything here is Premier League only; patterns may not transfer to other leagues without re-validation.
- **The sport itself changes over 15+ years of training data** — tactics, transfer spending, squad rotation, and the offside/VAR rule changes all shift what "team form" means over time, in ways a single static feature-engineering pipeline can't fully account for.
- **One validation window.** Two seasons (2024/25–2025/26) is enough to make a single model comparison trustworthy (per `.github/copilot-instructions.md` §10's reasoning), but it's still one slice of history — a model's rank order here isn't guaranteed to hold in every future season.
