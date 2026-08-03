# EDA Findings

Notable conclusions from `notebooks/01_exploration.ipynb`, kept here so they're easy to reference without re-running or re-reading the whole notebook.

## Target Distributions

- Match outcome split: Home 44.7%, Draw 24.2%, Away 31.0%. Consistent with well-known football knowledge (home wins most common, draws least) — draws are a minority but not rare (roughly 1 in 4 matches).
- Goal difference: mean ≈ +0.31 (home teams score more on average — the home advantage effect), skew ≈ 0.03 (essentially symmetric), kurtosis ≈ 0.75 (slightly more peaked than a normal distribution, but not dramatically). Range -9 to +9.

## Home Advantage Over Time

- Broadly stable/uniform across seasons, with one clear exception: 2020/21, the COVID-lockdown season played behind closed doors (no fans), where the home-win rate dropped noticeably.
- This suggests attendance is a real driver of home advantage. Attendance itself can't be a pre-match feature (unknown before kickoff), but it's usable as a **weighting on historical results** when computing rolling-form features for *future* matches — e.g. discount a home win that happened with a near-empty stadium when aggregating a team's recent form, since it reflects less "true" home advantage than a win in front of a full crowd (and the reverse for away results). No data leakage: it only reweights *already-known past* matches used to predict a *later* match, never the match being predicted itself.
- Weighting design isn't decided yet — this is deliberately left for later Feature Engineering, not explored further here. One thing to consider when we get there: raw attendance alone isn't enough, since stadium sizes vary hugely across the league (a 25,000-crowd is a sellout at one ground, half-empty at another) — the weighting most likely needs to account for **both** the raw attendance figure **and** the stadium's total capacity (e.g. attendance as a fraction of capacity), not attendance number alone.
- **Blocker found**: `Attendance` does not exist as a column anywhere in this dataset (checked directly — zero match). The weighting idea above isn't implementable with this data source as-is; it would need attendance sourced separately (a different dataset/API) if pursued. Until then, only the COVID season itself can be flagged via season/date logic, not a general attendance-based signal.

## Match Statistics (non-betting columns)

- `FTHG`/`FTAG`/`GoalDifference` strongly correlated — expected/mechanical, since `GoalDifference = FTHG - FTAG` by definition, not new information.
- `HST`↔`HS` and `AST`↔`AS` (shots on target vs. total shots) strongly correlated — real candidates for collapsing into a single "shot accuracy" ratio (`HST/HS`) rather than carrying both.
- **Important leakage flag**: `HS`, `AS`, `HST`, `AST`, `HC`, `AC`, `HF`, `AF`, `HY`, `AY`, `HR`, `AR`, `HTHG`, `HTAG`, `HTR` are all recorded *during or after* the match (shots, cards, corners, and the half-time score/result don't exist until the match is underway). None of these are usable as a direct input for predicting the match they describe — only as historical/rolling inputs for a *future* match. Directly ties to the project's data-leakage rule (§9); worth calling out explicitly since exploring these columns side-by-side with genuinely pre-match ones (like odds) could make it easy to forget the distinction once Feature Engineering starts.
- Sanity-checked the most extreme goal differences (±8/±9): all five real, well-known Premier League results (e.g. Southampton 0-9 Leicester 2019/20, Man United 9-0 Southampton 2020/21, Liverpool 9-0 Bournemouth 2022/23) — confirmed genuine outcomes, not data errors.

## Missingness (non-betting columns)

- Only `Time` has missing data among non-betting columns, and it's missing for the majority of the season history (2010/11 through 2018/19) — a strong candidate to drop rather than impute.

## Betting/Odds Columns

- 168 betting/odds columns total; only 9 have ≥90% coverage, 39 have <10% coverage — confirms most bookmaker columns are unusable across the full 16-season history.
- Bet365 vs. market Max vs. market Average odds correlate very strongly per outcome type (near 1.0) — supports treating them as redundant and using Bet365 alone as the baseline, per the project docs.
- Checked whether this high correlation was just an artifact of Bet365 being the only provider with full coverage: it isn't. `MaxH`/`AvgH` don't exist at all before 2019/20 (0% coverage), but pandas' `.corr()` already restricts each pairwise comparison to jointly-non-null rows — confirmed the heatmap's ~0.99 correlation is identical whether computed on all rows or explicitly filtered to the 2,660 rows where all three exist, and it's consistently ~0.99 in every individual season from 2019/20–2025/26 (not one season dominating). Bet365 and market average are genuinely near-interchangeable; Bet365 remains the better baseline choice specifically because it covers all 16 seasons vs. 7 for Max/Avg, without sacrificing signal quality.
- Calibration check: actual home-win rate tracks closely with Bet365's implied probability across buckets — good evidence the odds are genuinely informative, not noise.

## Not Yet Explored

Flagged as possible follow-ups, not yet investigated:

- Referee effect — untouched so far. Unlike the match-stat columns above, the referee is typically known pre-match, so (unlike shots/cards/corners) this would be a legitimate candidate feature, not a leakage risk.
- Team-level home-advantage variance — checked league-wide home advantage over time, but not whether some clubs benefit more from home advantage than others.
- Day-of-week/scheduling effects — not explored.
- Goal-scoring trend over time (beyond just win/draw/loss %) — attacking output could have drifted across 16 seasons independent of the result split.

## Implications for Feature Engineering

- **Drop**: `Time` (too sparse); most betting/odds columns with <90% coverage (keep Bet365 as the primary odds source, given as good as Max/Avg and far better season coverage).
- **Collapse**: `HS`/`HST` and `AS`/`AST` into shot-accuracy ratios instead of keeping both raw counts.
- **Never use directly**: `HS`/`AS`/`HST`/`AST`/`HC`/`AC`/`HF`/`AF`/`HY`/`AY`/`HR`/`AR`/`HTHG`/`HTAG`/`HTR` as pre-match features for the match they describe — recorded during/after the match, only valid as historical/rolling inputs.
- **New columns**: standard rolling-form features per the project rules (§11).
- **Blocked/needs more data**: the attendance-weighted rolling form idea — `Attendance` isn't in this dataset, so this needs an external data source before it's implementable.
