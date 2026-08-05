import numpy as np
import pandas as pd

from src.hyperparameter_tuning import season_expanding_splits


def _toy_train_df() -> pd.DataFrame:
    # 5 seasons, 4 rows each - enough to exercise multiple folds.
    seasons = []
    for season in ["2010-11", "2011-12", "2012-13", "2013-14", "2014-15"]:
        seasons += [season] * 4
    return pd.DataFrame({"Season": seasons, "value": range(20)})


def test_season_expanding_splits_grows_training_set_each_fold():
    train_df = _toy_train_df()

    splits = season_expanding_splits(train_df, min_train_seasons=1)

    # min_train_seasons=1 -> first fold trains on season index 0..1 (2
    # seasons), validates on season index 2; last fold validates on the
    # final season - 5 seasons total gives 3 folds (validating on seasons
    # at index 2, 3, 4).
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        assert set(train_idx).isdisjoint(set(val_idx))


def test_season_expanding_splits_never_validates_before_its_training_data():
    """Every validation index must come from a season strictly after every
    training index's season - the whole point of a time-aware split."""
    train_df = _toy_train_df()
    seasons = sorted(train_df["Season"].unique())
    season_position = train_df["Season"].map({s: i for i, s in enumerate(seasons)}).to_numpy()

    splits = season_expanding_splits(train_df, min_train_seasons=1)

    for train_idx, val_idx in splits:
        assert season_position[train_idx].max() < season_position[val_idx].min()


def test_season_expanding_splits_respects_min_train_seasons():
    train_df = _toy_train_df()

    splits = season_expanding_splits(train_df, min_train_seasons=3)

    # Skips folds until at least 3 seasons (indices 0,1,2) are available to
    # train on - only 1 fold left (train on seasons 0-3, validate on season 4).
    assert len(splits) == 1
    train_idx, val_idx = splits[0]
    assert len(train_idx) == 16  # 4 seasons x 4 rows
    assert len(val_idx) == 4


def test_season_expanding_splits_indices_are_positional():
    """Splits must return positions into the DataFrame's row order, not
    label-based index values - RandomizedSearchCV's cv parameter expects
    positional indices even when the DataFrame's own index isn't 0..n-1."""
    train_df = _toy_train_df()
    train_df.index = train_df.index + 100  # a non-trivial, non-contiguous-looking index

    splits = season_expanding_splits(train_df, min_train_seasons=1)

    all_positions = np.concatenate([np.concatenate([t, v]) for t, v in splits])
    assert all_positions.max() < len(train_df)
