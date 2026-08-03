# EDA Findings

Notable conclusions from `notebooks/01_exploration.ipynb`, kept here so they're easy to reference without re-running or re-reading the whole notebook. Fill in as you go through each section.

## Target Distributions

- Match outcome split (Home/Draw/Away):
- Goal difference distribution (shape, range, anything unexpected):

## Home Advantage Over Time

- Has the home/draw/away split stayed stable across the 16 seasons, or drifted?
- Anything season-specific that jumps out (an unusual year, a clear trend)?

## Match Statistics (non-betting columns)

- Any pair of columns strongly correlated (candidates for collapsing into one feature)?
- Any column that looks low-value / redundant for prediction?

## Missingness (non-betting columns)

- Which columns have missing data, and is it season-specific?

## Betting/Odds Columns

- How many odds columns are actually well-populated vs. sparse?
- Do the different providers (Bet365 vs market max vs market average) agree closely enough to treat as redundant?
- Did the calibration check confirm the odds are actually predictive?

## Implications for Feature Engineering

- Columns to drop:
- Columns to collapse (and how):
- New columns to engineer:
