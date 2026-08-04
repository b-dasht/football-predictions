from src.data_loader import _season_to_code, season_range


def test_season_to_code():
    assert _season_to_code("2010-11") == "1011"
    assert _season_to_code("2024-25") == "2425"


def test_season_range():
    assert season_range("2022-23", "2024-25") == ["2022-23", "2023-24", "2024-25"]


def test_season_range_single_season():
    assert season_range("2024-25", "2024-25") == ["2024-25"]
