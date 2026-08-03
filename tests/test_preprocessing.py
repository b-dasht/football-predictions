import pandas as pd

from src.preprocessing import _drop_blank_rows, _parse_dates


def test_drop_blank_rows_removes_fully_blank_row():
    df = pd.DataFrame({
        "Date": ["14/08/10", None],
        "HomeTeam": ["Arsenal", None],
        "AwayTeam": ["Chelsea", None],
        "FTHG": [1, None],
    })
    result = _drop_blank_rows(df)
    assert len(result) == 1


def test_drop_blank_rows_keeps_row_with_partial_missing_data():
    df = pd.DataFrame({
        "Date": ["14/08/10"],
        "HomeTeam": ["Arsenal"],
        "AwayTeam": ["Chelsea"],
        "FTHG": [None],
    })
    result = _drop_blank_rows(df)
    assert len(result) == 1


def test_parse_dates_handles_two_and_four_digit_years():
    df = pd.DataFrame({"Date": ["14/08/10", "08/08/2015"]})
    result = _parse_dates(df)
    assert result["Date"].iloc[0] == pd.Timestamp("2010-08-14")
    assert result["Date"].iloc[1] == pd.Timestamp("2015-08-08")
