import pandas as pd

from src.utils import get_feature_columns, split_by_season


def test_split_by_season_partitions_correctly():
    df = pd.DataFrame({
        "Season": ["2010-11", "2023-24", "2024-25", "2025-26", "2026-27"],
        "match_id": [0, 1, 2, 3, 4],
    })
    train, validation, test = split_by_season(df)

    assert list(train["match_id"]) == [0, 1]
    assert list(validation["match_id"]) == [2, 3]
    assert list(test["match_id"]) == [4]


def test_split_by_season_handles_empty_test_split():
    df = pd.DataFrame({
        "Season": ["2010-11", "2025-26"],
        "match_id": [0, 1],
    })
    train, validation, test = split_by_season(df)

    assert len(test) == 0
    assert len(validation) == 1


def test_get_feature_columns_excludes_identifiers_and_targets():
    df = pd.DataFrame(columns=[
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season",
        "TargetResult", "TargetGoalDifference",
        "Home_Points_Overall_last5", "Away_Points_Overall_last5",
        "ImpliedProbHome", "ImpliedProbDraw", "ImpliedProbAway",
    ])
    result = get_feature_columns(df)

    assert set(result) == {
        "Home_Points_Overall_last5", "Away_Points_Overall_last5",
        "ImpliedProbHome", "ImpliedProbDraw", "ImpliedProbAway",
    }
    assert "HomeTeam" not in result
    assert "TargetResult" not in result
