import pandas as pd

from src.utils import split_by_season


def test_split_by_season_partitions_correctly():
    df = pd.DataFrame({
        "Season": ["2010-11", "2024-25", "2025-26", "2026-27"],
        "match_id": [0, 1, 2, 3],
    })
    train, validation, test = split_by_season(df)

    assert list(train["match_id"]) == [0, 1]
    assert list(validation["match_id"]) == [2]
    assert list(test["match_id"]) == [3]


def test_split_by_season_handles_empty_test_split():
    df = pd.DataFrame({
        "Season": ["2010-11", "2025-26"],
        "match_id": [0, 1],
    })
    train, validation, test = split_by_season(df)

    assert len(test) == 0
    assert len(validation) == 1
