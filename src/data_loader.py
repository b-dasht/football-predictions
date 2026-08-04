"""Download and load raw EPL match data from football-data.co.uk."""

from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from src.config import DATA_RAW_PATH, TEST_SEASON, TRAIN_START_SEASON, VALIDATION_SEASON

BASE_URL = "https://www.football-data.co.uk/mmz4281"
COMPETITION_CODE = "E0"  # football-data.co.uk's code for the English Premier League


def _season_to_code(season: str) -> str:
    """Convert '2010-11' -> '1011', the season code football-data.co.uk's URLs use."""
    start_year, end_year = season.split("-")
    return f"{start_year[-2:]}{end_year}"


def season_range(start_season: str, end_season: str) -> list[str]:
    """Generate consecutive seasons from start_season to end_season, inclusive."""
    start_year = int(start_season.split("-")[0])
    end_year = int(end_season.split("-")[0])
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year + 1)]


def download_season(season: str, force: bool = False) -> Path | None:
    """Download one season's raw CSV into data/raw/, unless already cached.

    Returns None if the season isn't available yet (a not-yet-started
    season returns HTTP 404 from the source - expected, not an error).
    """
    dest = DATA_RAW_PATH / f"{season}.csv"
    if dest.exists() and not force:
        logger.info(f"{season}: using cached file")
        return dest

    url = f"{BASE_URL}/{_season_to_code(season)}/{COMPETITION_CODE}.csv"
    response = requests.get(url, timeout=30)

    if response.status_code == 404:
        logger.info(f"{season}: not available yet (404) - {url}")
        return None
    response.raise_for_status()

    DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    logger.info(f"{season}: downloaded to {dest}")
    return dest


def download_all_seasons() -> None:
    """Download every configured season into data/raw/.

    Every season is cached after first download except the most recent
    (TEST_SEASON), which is always re-fetched since it may still be in
    progress. Seasons through VALIDATION_SEASON are required to exist;
    TEST_SEASON not existing yet is expected, not an error.

    Raises:
        RuntimeError: if any required season failed to download.
    """
    seasons = season_range(TRAIN_START_SEASON, TEST_SEASON)
    required_seasons = set(season_range(TRAIN_START_SEASON, VALIDATION_SEASON))

    missing_required = []
    for season in seasons:
        is_latest = season == seasons[-1]
        path = download_season(season, force=is_latest)
        if path is None and season in required_seasons:
            missing_required.append(season)

    if missing_required:
        raise RuntimeError(f"Missing required seasons: {missing_required}")
    logger.info(f"All {len(required_seasons)} required seasons present in {DATA_RAW_PATH}")


def load_raw_matches() -> pd.DataFrame:
    """Load every cached season CSV into one combined DataFrame, tagged by season.

    Older seasons have fewer columns than newer ones (e.g. fewer bookmaker
    odds fields) - concatenation fills the gaps with NaN rather than
    dropping any column, and mismatches are logged for visibility.
    """
    seasons = season_range(TRAIN_START_SEASON, TEST_SEASON)
    loaded: list[tuple[str, pd.DataFrame]] = []
    for season in seasons:
        path = DATA_RAW_PATH / f"{season}.csv"
        if not path.exists():
            logger.info(f"{season}: no cached file, skipping")
            continue
        df = pd.read_csv(path)
        df["Season"] = season
        loaded.append((season, df))

    if not loaded:
        raise FileNotFoundError(f"No season files found in {DATA_RAW_PATH}. Run download_all_seasons() first.")

    combined = pd.concat([df for _, df in loaded], ignore_index=True)
    logger.info(f"Loaded {len(combined)} matches across {len(loaded)} seasons")

    all_columns = set(combined.columns)
    for season, df in loaded:
        missing_columns = all_columns - set(df.columns) - {"Season"}
        if missing_columns:
            logger.warning(f"{season}: missing columns present in other seasons: {sorted(missing_columns)}")

    return combined


if __name__ == "__main__":
    download_all_seasons()
    matches = load_raw_matches()
    logger.info(f"Final combined shape: {matches.shape}")
