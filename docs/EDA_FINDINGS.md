# EDA Findings

Notable conclusions from `notebooks/01_exploration.ipynb`, kept here so they're easy to reference without re-running or re-reading the whole notebook.

## Target Distributions

- Match outcome split: Home 44.7%, Draw 24.2%, Away 31.0%. Consistent with well-known football knowledge (home wins most common, draws least) — draws are a minority but not rare (roughly 1 in 4 matches).
- Goal difference: mean ≈ +0.31 (home teams score more on average — the home advantage effect), skew ≈ 0.03 (essentially symmetric), kurtosis ≈ 0.75 (slightly more peaked than a normal distribution, but not dramatically). Range -9 to +9.

## Home Advantage Over Time

- Broadly stable/uniform across seasons, with one clear exception: 2020/21, the COVID-lockdown season played behind closed doors (no fans), where the home-win rate dropped noticeably.
- No attendance data exists in this dataset (checked directly — no `Attendance` column anywhere), so this can only be flagged as a season-level fact (2020/21 was behind closed doors), not built into a general attendance-based feature. Not pursuing further without an external data source.

## Goal-Scoring Trends Over Time

- Average total goals per match ranges from ≈2.57 (2014/15) to ≈3.28 (2023/24) across the 16 seasons — some fluctuation, no strong monotonic trend.
- Any drift here is confounded with manager/tactical changes at individual clubs (e.g. a single high-profile managerial appointment can shift a team's scoring rate independent of any league-wide era effect) — season-level aggregates alone can't separate the two.

## Match Statistics (non-betting columns)

- `FTHG`/`FTAG`/`GoalDifference` strongly correlated — expected/mechanical, since `GoalDifference = FTHG - FTAG` by definition, not new information.
- `HST`↔`HS` and `AST`↔`AS` (shots on target vs. total shots) strongly correlated — real candidates for collapsing into a single "shot accuracy" ratio (`HST/HS`) rather than carrying both.
- **Important leakage flag**: `HS`, `AS`, `HST`, `AST`, `HC`, `AC`, `HF`, `AF`, `HY`, `AY`, `HR`, `AR`, `HTHG`, `HTAG`, `HTR` are all recorded *during or after* the match (shots, cards, corners, and the half-time score/result don't exist until the match is underway). None of these are usable as a direct input for predicting the match they describe — they can only feed **prior/historical** information for a *future* match (e.g. a team's rolling average shots over its last 5 games), never the match being predicted itself. Directly ties to the project's data-leakage rule (§9).
- Sanity-checked the most extreme goal differences (±8/±9): all five real, well-known Premier League results (e.g. Southampton 0-9 Leicester 2019/20, Man United 9-0 Southampton 2020/21, Liverpool 9-0 Bournemouth 2022/23) — confirmed genuine outcomes, not data errors.

## Team-Specific Home/Away Stats

- Home advantage varies a lot by team, not just by league or season. Among teams with a reasonable sample (≥100 home matches), the gap between home win % and away win % ranges from **~23 points (Stoke)** down to **~3 points (Crystal Palace)**.
- Man City, Arsenal, and Man United also show large gaps (~18-19 points) — but for a different reason than Stoke: they're simply strong teams overall, so both their home *and* away win rates are high in absolute terms; the gap measures the *relative* home boost, not overall quality.
- Confirms this is worth building as a per-team feature (e.g. team-specific home/away win rate, goals scored/conceded splits) rather than assuming league-wide home advantage applies equally to every team.

## Head-to-Head Matchup Feasibility

- Checked whether the data supports a "last 5-10 head-to-head meetings" feature (not building it yet — that's Feature Engineering). 627 distinct team pairings exist across the 16 seasons; median 6 meetings per pairing, up to 32 for teams that stayed in the league the whole time (e.g. Man City vs Tottenham), but as few as 2 for pairings that only briefly shared the league.
- Man United vs Arsenal specifically: 32 meetings recorded, full match-by-match record available.
- **Caveat for Feature Engineering**: a "last 5-10 meetings" feature needs an explicit fallback for pairings with fewer than 5-10 prior meetings (common for newly-promoted teams) — can't just assume the history exists.
- Both framings raised are feasible: overall matchup (regardless of venue) and same-stadium-only (splitting by which team hosted) — the same underlying data supports either.

## Betting/Odds Columns

- 168 betting/odds columns total; only 9 have ≥90% coverage, 39 have <10% coverage — confirms most bookmaker columns are unusable across the full 16-season history.
- Bet365 vs. market Max vs. market Average odds correlate very strongly per outcome type (near 1.0) — supports treating them as redundant and using Bet365 alone as the baseline, per the project docs.
- Checked whether this high correlation was just an artifact of Bet365 being the only provider with full coverage: it isn't. `MaxH`/`AvgH` don't exist at all before 2019/20 (0% coverage), but pandas' `.corr()` already restricts each pairwise comparison to jointly-non-null rows — confirmed the heatmap's ~0.99 correlation is identical whether computed on all rows or explicitly filtered to the 2,660 rows where all three exist, and it's consistently ~0.99 in every individual season from 2019/20–2025/26 (not one season dominating). Bet365 and market average are genuinely near-interchangeable; Bet365 remains the better baseline choice specifically because it covers all 16 seasons vs. 7 for Max/Avg, without sacrificing signal quality.
- Calibration check: actual home-win rate tracks closely with Bet365's implied probability across buckets — good evidence the odds are genuinely informative, not noise.

## Missingness (non-betting columns)

- Only `Time` has missing data among non-betting columns, and it's missing for the majority of the season history (2010/11 through 2018/19) — a strong candidate to drop rather than impute.

## Considered and Rejected

- **Day-of-week/rest-day effects**: days-of-rest-since-last-match would be a more meaningful signal than day-of-week alone, but computing it properly requires match data from other competitions a team may have played in between league fixtures (e.g. Champions League, domestic cups) — not available in this dataset. Decided not to pursue rather than build a version that's silently wrong for European-competition teams.

## Referee Effect

- Real spread found: among referees with ≥50 matches, home win rate ranges from ~33.3% to ~53.9%, vs. 44.7% league-wide.
- **Confound**: likely reflects which fixtures a referee is assigned to (senior/experienced referees are more often given bigger clubs' matches, which have higher home win rates regardless of who's officiating) rather than a genuine referee bias. Not ruled out as a feature, but the raw spread alone isn't strong evidence of a true effect — would need controlling for the teams involved to say more.
- **Team-specific check** (partially controls for team strength): compared each team's points-per-game with a specific referee against that team's own overall average. The suggested illustrative case, Man United with Howard Webb (`H Webb`), showed essentially **no deviation** (1.867 vs. 1.842 PPG, a +0.025 difference over 15 matches) — a genuine null result for that specific pairing, not cherry-picked. Some other team/referee pairs *do* show large deviations (up to roughly ±1.1-1.2 PPG), but every one of these is based on only 5-15 matches — too small a sample to distinguish a real effect from noise without further statistical testing.

## Implications for Feature Engineering

- **Drop**: `Time` (too sparse); most betting/odds columns with <90% coverage (keep Bet365 as the primary odds source, given as good as Max/Avg and far better season coverage).
- **Collapse**: `HS`/`HST` and `AS`/`AST` into shot-accuracy ratios instead of keeping both raw counts.
- **Never use directly**: `HS`/`AS`/`HST`/`AST`/`HC`/`AC`/`HF`/`AF`/`HY`/`AY`/`HR`/`AR`/`HTHG`/`HTAG`/`HTR` as pre-match features for the match they describe — recorded during/after the match, only valid as historical/rolling inputs.
- **New columns**: standard rolling-form features per the project rules (§11); team-specific home/away splits (not just league-wide); head-to-head record over the last 5-10 meetings (with an explicit fallback for under-populated pairings), built both as overall matchup and same-stadium-only.
- **Deferred**: referee as a feature — the raw effect found is confounded with fixture assignment; not worth building until/unless there's a way to control for that.
