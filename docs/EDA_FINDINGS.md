# EDA Findings

Notable conclusions from `notebooks/01_exploration.ipynb`, kept here so they're easy to reference without re-running or re-reading the whole notebook.

## Target Distributions

- Match outcome split: Home 44.7%, Draw 24.2%, Away 31.0%. Consistent with well-known football knowledge (home wins most common, draws least) — draws are a minority but not rare (roughly 1 in 4 matches).
- Goal difference: mean ≈ +0.31 (home teams score more on average — the home advantage effect), skew ≈ 0.03 (essentially symmetric), kurtosis ≈ 0.75 (slightly more peaked than a normal distribution, but not dramatically). Range -9 to +9.

## Home Advantage Over Time

- Broadly stable/uniform across seasons, with one clear exception: 2020/21, the COVID-lockdown season played behind closed doors (no fans), where the home-win rate dropped noticeably.
- This suggests attendance is a real driver of home advantage. Attendance itself can't be a pre-match feature (unknown before kickoff), but it's fair game as a **weight on historical results** when computing rolling-form features for *future* matches — e.g. discount a home win that happened with a near-empty stadium when aggregating a team's recent form, since it reflects less "true" home advantage than a win in front of a full crowd (and the reverse for away results). Not leakage: it only reweights *already-known past* matches used to predict a *later* match, never the match being predicted itself.
- Weighting design isn't decided yet — this is deliberately left for Feature Engineering, not explored further here. One thing to consider when we get there: raw attendance alone isn't enough, since stadium sizes vary hugely across the league (a 25,000-crowd is a sellout at one ground, half-empty at another) — the weighting most likely needs to account for **both** the raw attendance figure **and** the stadium's total capacity (e.g. attendance as a fraction of capacity), not attendance number alone.

## Match Statistics (non-betting columns)

- `FTHG`/`FTAG`/`GoalDifference` strongly correlated — expected/mechanical, since `GoalDifference = FTHG - FTAG` by definition, not new information.
- `HST`↔`HS` and `AST`↔`AS` (shots on target vs. total shots) strongly correlated — real candidates for collapsing into a single "shot accuracy" ratio (`HST/HS`) rather than carrying both.

## Missingness (non-betting columns)

- Only `Time` has missing data among non-betting columns, and it's missing for the majority of the season history (2010/11 through 2018/19) — a strong candidate to drop rather than impute.

## Betting/Odds Columns

- 168 betting/odds columns total; only 9 have ≥90% coverage, 39 have <10% coverage — confirms most bookmaker columns are unusable across the full 16-season history.
- Bet365 vs. market Max vs. market Average odds correlate very strongly per outcome type (near 1.0) — supports treating them as redundant and using Bet365 alone as the baseline, per the project docs.
- Calibration check: actual home-win rate tracks closely with Bet365's implied probability across buckets — good evidence the odds are genuinely informative, not noise.

## Implications for Feature Engineering

- **Drop**: `Time` (too sparse); most betting/odds columns with <90% coverage (keep Bet365 as the primary odds source).
- **Collapse**: `HS`/`HST` and `AS`/`AST` into shot-accuracy ratios instead of keeping both raw counts.
- **New columns**: standard rolling-form features per the project rules (§11), plus the attendance-weighted rolling form idea above as something worth prototyping.
