"""Clean raw EPL match data into a consistent, chronologically-ordered table."""

import pandas as pd
from loguru import logger

from src.config import DATA_INTERIM_PATH
from src.data_loader import load_raw_matches

KEY_COLUMNS = ["Date", "HomeTeam", "AwayTeam"]


def _drop_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are entirely blank (e.g. trailing blank lines in source CSVs)."""
    before = len(df)
    df = df.dropna(subset=KEY_COLUMNS, how="all")
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} blank row(s)")
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Date as a real datetime.

    Source CSVs mix dd/mm/yy and dd/mm/yyyy formats across seasons
    (and even within a couple of seasons) - dayfirst=True combined with
    pandas' per-element parsing handles both without needing to know
    which season uses which format.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df


def _check_duplicates(df: pd.DataFrame) -> None:
    """Log a warning if any fixture (Date+HomeTeam+AwayTeam) appears more than once."""
    dupes = df.duplicated(subset=KEY_COLUMNS, keep=False)
    if dupes.any():
        logger.warning(f"Found {dupes.sum()} duplicate fixture row(s):\n{df.loc[dupes, KEY_COLUMNS]}")
    else:
        logger.info("No duplicate fixtures found")


def clean_matches() -> pd.DataFrame:
    """Load raw matches and produce a clean, chronologically-sorted table.

    Saves the result to data/interim/matches.csv.
    """
    df = load_raw_matches()
    df = _drop_blank_rows(df)
    df = _parse_dates(df)
    _check_duplicates(df)
    df = df.sort_values("Date").reset_index(drop=True)

    DATA_INTERIM_PATH.mkdir(parents=True, exist_ok=True)
    dest = DATA_INTERIM_PATH / "matches.csv"
    df.to_csv(dest, index=False)
    logger.info(f"Saved {len(df)} cleaned matches to {dest}")
    return df


if __name__ == "__main__":
    clean_matches()
