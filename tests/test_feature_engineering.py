import pandas as pd

from src.config import ROLLING_WINDOWS
from src.feature_engineering import (
    _add_odds_features,
    _add_rolling_channel,
    _add_targets,
    _build_long_format,
    _columns_to_drop,
)


def _toy_matches() -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": [0, 1, 2],
        "Date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
        "HomeTeam": ["A", "B", "A"],
        "AwayTeam": ["B", "A", "C"],
        "FTHG": [2, 0, 1],
        "FTAG": [0, 1, 1],
        "FTR": ["H", "A", "D"],
        "HS": [10, 8, 12],
        "AS": [5, 9, 6],
        "HST": [5, 3, 6],
        "AST": [2, 4, 3],
    })


def test_build_long_format_has_one_row_per_team_per_match():
    long_df = _build_long_format(_toy_matches())
    assert len(long_df) == 6  # 3 matches x 2 teams each
    assert set(long_df["Team"]) == {"A", "B", "C"}


def test_rolling_channel_flags_insufficient_history():
    """Team A's only prior match by match_id=2 is match_id=0 - one match
    isn't enough for any of the configured rolling windows (5, 10)."""
    long_df = _build_long_format(_toy_matches())
    long_df = _add_rolling_channel(long_df, group_cols=["Team"], suffix="Overall")

    row = long_df[(long_df["Team"] == "A") & (long_df["match_id"] == 2)].iloc[0]
    window = ROLLING_WINDOWS[0]
    assert row[f"HasFullHistory_Overall_last{window}"] == False  # noqa: E712
    assert pd.isna(row[f"Points_Overall_last{window}"])


def test_rolling_channel_never_uses_the_current_match():
    """A run of matches with a deliberately extreme final result - if the
    rolling average for that final row is anywhere near it, shift(1) isn't
    working and the current match is leaking into its own features."""
    window = ROLLING_WINDOWS[0]
    n_matches = window + 1
    dates = pd.date_range("2020-01-01", periods=n_matches, freq="7D")

    df = pd.DataFrame({
        "match_id": range(n_matches),
        "Date": dates,
        "HomeTeam": ["A"] * n_matches,
        "AwayTeam": ["B"] * n_matches,
        "FTHG": [0] * (n_matches - 1) + [99],  # extreme, current-match-only value
        "FTAG": [0] * n_matches,
        "FTR": ["D"] * (n_matches - 1) + ["H"],
        "HS": [10] * n_matches,
        "AS": [10] * n_matches,
        "HST": [5] * n_matches,
        "AST": [5] * n_matches,
    })
    long_df = _build_long_format(df)
    long_df = _add_rolling_channel(long_df, group_cols=["Team"], suffix="Overall")

    last_row = long_df[(long_df["Team"] == "A") & (long_df["match_id"] == n_matches - 1)].iloc[0]
    # All `window` prior matches were draws (1 point each) - the average
    # must be exactly 1.0, unaffected by the current match's win (3 points).
    assert last_row[f"Points_Overall_last{window}"] == 1.0
    assert last_row[f"GoalsScored_Overall_last{window}"] == 0.0


def test_columns_to_drop_keeps_only_bet365_odds():
    raw_df = pd.DataFrame(columns=[
        "Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HTR", "Referee", "HS", "AS", "HST", "AST",
        "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "Season",
        "B365H", "B365D", "B365A", "MaxH", "AvgH",
    ])
    dropped = _columns_to_drop(raw_df)
    assert "MaxH" in dropped
    assert "AvgH" in dropped
    assert "B365H" not in dropped
    assert "Div" in dropped
    assert "Time" in dropped
    assert "HTHG" in dropped


def test_odds_features_sum_to_one():
    df = pd.DataFrame({"B365H": [2.0], "B365D": [3.5], "B365A": [4.0]})
    result = _add_odds_features(df)
    total = result["ImpliedProbHome"] + result["ImpliedProbDraw"] + result["ImpliedProbAway"]
    assert abs(total.iloc[0] - 1.0) < 1e-9


def test_add_targets_encodes_result_and_goal_difference():
    df = pd.DataFrame({
        "FTR": ["H", "D", "A"],
        "FTHG": [2, 1, 0],
        "FTAG": [0, 1, 3],
    })
    result = _add_targets(df)
    assert list(result["TargetResult"]) == [2, 1, 0]
    assert list(result["TargetGoalDifference"]) == [2, 0, -3]
