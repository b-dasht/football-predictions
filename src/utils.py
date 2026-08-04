"""General-purpose helpers reused across modules."""

import pandas as pd
from loguru import logger

from src.config import TEST_SEASON, TRAIN_END_SEASON, TRAIN_START_SEASON, VALIDATION_SEASON
from src.data_loader import season_range


def split_by_season(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation/test split using config.py's season boundaries.

    Never shuffles - splits strictly by which season each match belongs to
    (see copilot-instructions.md #10 for why a random split isn't valid
    here). An empty split (e.g. TEST_SEASON hasn't started yet) is logged
    clearly rather than raising, since that's expected at certain points
    in the season calendar.
    """
    train_seasons = season_range(TRAIN_START_SEASON, TRAIN_END_SEASON)
    train = features_df[features_df["Season"].isin(train_seasons)]
    validation = features_df[features_df["Season"] == VALIDATION_SEASON]
    test = features_df[features_df["Season"] == TEST_SEASON]

    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        if split.empty:
            logger.warning(f"{name} split is empty (no matches found for its configured season(s))")
        else:
            logger.info(f"{name}: {len(split)} matches, seasons {sorted(split['Season'].unique())}")

    return train, validation, test
