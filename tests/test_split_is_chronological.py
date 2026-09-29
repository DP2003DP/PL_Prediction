"""chronological_split and season_folds must never let a model see the future (contract section 4)."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.split import chronological_split, season_folds
from src.utils import resolve

KEY = ["player_id", "date"]


def _configured(cfg: dict) -> tuple[list[str], str, str]:
    s = cfg["split"]
    return list(s["train_seasons"]), s["val_season"], s["test_season"]


def _with_split(cfg: dict, **overrides) -> dict:
    new = copy.deepcopy(cfg)
    new["split"].update(overrides)
    return new


def _keys(df: pd.DataFrame) -> list[tuple]:
    return list(zip(df["player_id"], df["date"]))


def _assert_valid_split(df: pd.DataFrame, cfg: dict, train, val, test) -> None:
    train_s, val_s, test_s = _configured(cfg)
    seasons = [set(train["season"]), set(val["season"]), set(test["season"])]
    assert seasons == [set(train_s), {val_s}, {test_s}], f"split seasons {seasons} do not match the config"
    assert not (seasons[0] & seasons[1] or seasons[0] & seasons[2] or seasons[1] & seasons[2])
    assert train["date"].max() < val["date"].min(), "train reaches into the validation period"
    assert val["date"].max() < test["date"].min(), "validation reaches into the test period"
    in_cfg = df["season"].isin(train_s + [val_s, test_s])
    assert len(train) + len(val) + len(test) == int(in_cfg.sum())
    keys = _keys(train) + _keys(val) + _keys(test)
    assert len(set(keys)) == len(keys), "a row appears in more than one split"
    assert set(keys) == set(_keys(df[in_cfg]))


def test_split_seasons_disjoint_dates_ordered_counts_add_up(split_frame, cfg):
    before = split_frame.copy(deep=True)
    train, val, test = chronological_split(split_frame, cfg)
    pd.testing.assert_frame_equal(split_frame, before)
    _assert_valid_split(split_frame, cfg, train, val, test)
    for part in (train, val, test):
        assert set(split_frame.columns) <= set(part.columns)


def test_split_ignores_rows_from_unconfigured_seasons(split_frame, cfg):
    train_s, val_s, test_s = _configured(cfg)
    outside = split_frame[~split_frame["season"].isin(train_s + [val_s, test_s])]
    assert len(outside) > 0, "precondition: the synthetic frame has out-of-config seasons"
    train, val, test = chronological_split(split_frame, cfg)
    returned = set(_keys(train)) | set(_keys(val)) | set(_keys(test))
    assert not returned & set(_keys(outside))


def test_split_works_on_sorted_default_index(split_frame, cfg):
    df = split_frame.sort_values(KEY).reset_index(drop=True)
    _assert_valid_split(df, cfg, *chronological_split(df, cfg))


@pytest.mark.parametrize("overrides", [
    lambda tr, va, te: {"train_seasons": tr + [va]},
    lambda tr, va, te: {"train_seasons": tr + [te]},
    lambda tr, va, te: {"val_season": te},
], ids=["val_in_train", "test_in_train", "val_equals_test"])
def test_overlapping_seasons_raise(split_frame, cfg, overrides):
    with pytest.raises(ValueError):
        chronological_split(split_frame, _with_split(cfg, **overrides(*_configured(cfg))))


@pytest.mark.parametrize("overrides", [
    lambda tr, va, te: {"train_seasons": [va], "val_season": tr[-1]},
    lambda tr, va, te: {"val_season": te, "test_season": va},
], ids=["val_before_train", "test_before_val"])
def test_seasons_out_of_order_raise(split_frame, cfg, overrides):
    with pytest.raises(ValueError):
        chronological_split(split_frame, _with_split(cfg, **overrides(*_configured(cfg))))


@pytest.mark.parametrize("late, early", [("val", "train"), ("test", "val")])
def test_row_dated_across_a_boundary_raises(split_frame, cfg, late, early):
    train_s, val_s, test_s = _configured(cfg)
    season = {"train": train_s[-1], "val": val_s, "test": test_s}
    df = split_frame.copy()
    label = df.index[df["season"] == season[late]][0]
    df.loc[label, "date"] = df.loc[df["season"] == season[early], "date"].min()
    with pytest.raises(ValueError):
        chronological_split(df, cfg)


@pytest.mark.parametrize("which", ["train_seasons", "all_seasons"])
@pytest.mark.parametrize("layout", ["shuffled_offset_index", "sorted_range_index"])
def test_season_folds_expanding_and_positional(split_frame, cfg, which, layout):
    seasons = cfg["split"]["train_seasons"] if which == "train_seasons" else cfg["data"]["seasons"]
    df = split_frame if layout == "shuffled_offset_index" else split_frame.sort_values(KEY).reset_index(drop=True)
    folds = season_folds(df, seasons)
    assert len(folds) == len(seasons) - 1
    season_arr = df["season"].to_numpy()
    prev_train = np.array([], dtype=int)
    for i, (tr, va) in enumerate(folds, start=1):
        tr, va = np.asarray(tr), np.asarray(va)
        assert np.issubdtype(tr.dtype, np.integer) and np.issubdtype(va.dtype, np.integer)
        assert np.array_equal(np.sort(tr), np.flatnonzero(np.isin(season_arr, seasons[:i]))), f"fold {i} train"
        assert np.array_equal(np.sort(va), np.flatnonzero(season_arr == seasons[i])), f"fold {i} validation"
        assert set(df.iloc[va]["season"]) == {seasons[i]}
        assert df.iloc[tr]["date"].max() < df.iloc[va]["date"].min(), f"fold {i} trains on the future"
        assert np.isin(prev_train, tr).all() and len(tr) > len(prev_train), f"fold {i} is not expanding"
        prev_train = tr


@pytest.mark.parametrize("position", ["first", "last"])
def test_season_folds_raise_for_season_without_rows(split_frame, cfg, position):
    seasons = list(cfg["split"]["train_seasons"])
    seasons = ["2010-11"] + seasons if position == "first" else seasons + ["2030-31"]
    with pytest.raises(ValueError):
        season_folds(split_frame, seasons)


def test_real_features_split_is_chronological(cfg):
    path = resolve(cfg["paths"]["features"])
    if not path.exists():
        pytest.skip(f"{path} not built yet")
    df = pd.read_parquet(path)
    train, val, test = chronological_split(df, cfg)
    _assert_valid_split(df, cfg, train, val, test)
    train_s = cfg["split"]["train_seasons"]
    for i, (tr, va) in enumerate(season_folds(train, train_s), start=1):
        assert set(train.iloc[tr]["season"]) == set(train_s[:i])
        assert set(train.iloc[va]["season"]) == {train_s[i]}
        assert train.iloc[tr]["date"].max() < train.iloc[va]["date"].min()
