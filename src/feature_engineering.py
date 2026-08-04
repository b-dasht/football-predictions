"""Build pre-match features from the cleaned match table.

Turns data/interim/matches.csv into data/processed/features.csv: rolling
team form, head-to-head history, and odds-derived features - all computed
so that a match's features never use information from that match itself
or anything after it (see docs/EDA_FINDINGS.md for why the raw in-match
stat columns can't be used directly).
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DATA_INTERIM_PATH, DATA_PROCESSED_PATH, ROLLING_WINDOWS

# Metrics computed as rolling historical averages, per team perspective.
ROLLING_METRICS = ["Points", "GoalsScored", "GoalsConceded", "ShotAccuracy"]

# (group_cols, suffix): each defines one "history channel" a rolling
# average is computed over. All share the same ROLLING_METRICS/ROLLING_WINDOWS.
ROLLING_CHANNELS = [
    (["Team"], "Overall"),                    # last N matches, any venue
    (["Team", "Venue"], "Venue"),              # last N matches at this venue (home-only/away-only)
    (["Team", "Opponent"], "H2H"),             # last N meetings vs this opponent, any venue
    (["Team", "Opponent", "Venue"], "H2HVenue"),  # last N meetings vs this opponent, this venue only
]


def _build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (home + away), the shape rolling stats need."""

    def _shot_accuracy(shots_on_target: pd.Series, shots: pd.Series) -> pd.Series:
        return shots_on_target / shots.replace(0, np.nan)

    home_rows = pd.DataFrame({
        "match_id": df["match_id"],
        "Date": df["Date"],
        "Team": df["HomeTeam"],
        "Opponent": df["AwayTeam"],
        "Venue": "Home",
        "Points": df["FTR"].map({"H": 3, "D": 1, "A": 0}),
        "GoalsScored": df["FTHG"],
        "GoalsConceded": df["FTAG"],
        "ShotAccuracy": _shot_accuracy(df["HST"], df["HS"]),
    })
    away_rows = pd.DataFrame({
        "match_id": df["match_id"],
        "Date": df["Date"],
        "Team": df["AwayTeam"],
        "Opponent": df["HomeTeam"],
        "Venue": "Away",
        "Points": df["FTR"].map({"H": 0, "D": 1, "A": 3}),
        "GoalsScored": df["FTAG"],
        "GoalsConceded": df["FTHG"],
        "ShotAccuracy": _shot_accuracy(df["AST"], df["AS"]),
    })
    return pd.concat([home_rows, away_rows], ignore_index=True)


def _add_rolling_channel(long_df: pd.DataFrame, group_cols: list[str], suffix: str) -> pd.DataFrame:
    """Add rolling-average + has-enough-history columns for one history channel.

    shift(1) excludes the current row before the rolling window is taken,
    so a match's own result never leaks into its own features. min_periods
    equal to the window means an incomplete window produces NaN rather than
    a misleadingly-averaged partial value.
    """
    long_df = long_df.sort_values(group_cols + ["Date"])
    grouped = long_df.groupby(group_cols, sort=False)
    prior_count = grouped["Points"].transform(lambda s: s.shift(1).expanding().count())

    for window in ROLLING_WINDOWS:
        long_df[f"HasFullHistory_{suffix}_last{window}"] = prior_count >= window

        for metric in ROLLING_METRICS:
            col_name = f"{metric}_{suffix}_last{window}"
            long_df[col_name] = grouped[metric].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean()
            )

    return long_df


# Every column football-data.co.uk provides that isn't a betting/odds
# column (per docs/DATA_DICTIONARY.md's Match Info + Match Statistics
# sections) - anything not in this list is treated as a betting column.
MATCH_INFO_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "Season",
]

# Only Bet365 is kept among the raw odds columns (see docs/EDA_FINDINGS.md
# for why: same information as the market average, far better season
# coverage) - every other bookmaker's raw odds column is dropped.
ODDS_COLUMNS_TO_KEEP = ["B365H", "B365D", "B365A"]

# In-match/post-match columns - never valid as direct pre-match features
# (see docs/EDA_FINDINGS.md), only as inputs to the rolling stats above.
IN_MATCH_COLUMNS_TO_DROP = [
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "HTHG", "HTAG", "HTR",
]


def _columns_to_drop(raw_df: pd.DataFrame) -> list[str]:
    betting_columns = [c for c in raw_df.columns if c not in MATCH_INFO_COLUMNS]
    betting_columns_to_drop = [c for c in betting_columns if c not in ODDS_COLUMNS_TO_KEEP]
    return ["Div", "Time"] + IN_MATCH_COLUMNS_TO_DROP + betting_columns_to_drop


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Classification and regression targets, kept separate from feature columns.

    TargetResult: 0=Away win, 1=Draw, 2=Home win - integer-encoded because
    XGBoost's classifier requires consecutive non-negative integer labels;
    the mapping order is otherwise arbitrary (these are nominal classes,
    not an ordinal scale).
    TargetGoalDifference: Home goals - Away goals, the regression target.
    """
    df = df.copy()
    df["TargetResult"] = df["FTR"].map({"A": 0, "D": 1, "H": 2})
    df["TargetGoalDifference"] = df["FTHG"] - df["FTAG"]
    return df


def _add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bet365 implied probabilities, normalized to remove the bookmaker's overround."""
    df = df.copy()
    raw_home = 1 / df["B365H"]
    raw_draw = 1 / df["B365D"]
    raw_away = 1 / df["B365A"]
    overround = raw_home + raw_draw + raw_away

    df["ImpliedProbHome"] = raw_home / overround
    df["ImpliedProbDraw"] = raw_draw / overround
    df["ImpliedProbAway"] = raw_away / overround
    return df


def build_features() -> pd.DataFrame:
    """Load the cleaned match table and produce the model-ready feature table.

    Saves the result to data/processed/features.csv.
    """
    df = pd.read_csv(DATA_INTERIM_PATH / "matches.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["match_id"] = df.index

    long_df = _build_long_format(df)
    for group_cols, suffix in ROLLING_CHANNELS:
        long_df = _add_rolling_channel(long_df, group_cols, suffix)

    # Excludes the raw ROLLING_METRICS themselves (Points/GoalsScored/...),
    # since those are the CURRENT match's own result - only the derived
    # "_last5"/"_last10" rolling columns (built from shift(1), i.e. strictly
    # prior matches) are safe to keep as features.
    non_feature_cols = ["Date", "Team", "Opponent", "Venue"] + ROLLING_METRICS
    engineered_cols = [c for c in long_df.columns if c not in non_feature_cols]
    home_features = long_df[long_df["Venue"] == "Home"][engineered_cols]
    away_features = long_df[long_df["Venue"] == "Away"][engineered_cols]
    home_features = home_features.rename(columns={c: f"Home_{c}" for c in engineered_cols if c != "match_id"})
    away_features = away_features.rename(columns={c: f"Away_{c}" for c in engineered_cols if c != "match_id"})

    features = df.merge(home_features, on="match_id").merge(away_features, on="match_id")
    features = _add_odds_features(features)
    features = _add_targets(features)
    features = features.drop(columns=_columns_to_drop(df))

    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    dest = DATA_PROCESSED_PATH / "features.csv"
    features.to_csv(dest, index=False)
    logger.info(f"Saved {len(features)} rows, {len(features.columns)} columns to {dest}")
    return features


if __name__ == "__main__":
    build_features()
