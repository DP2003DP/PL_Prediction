"""Chronological train/validation/test split by season, plus expanding-window season folds.

Run as `python -m src.split` to print a split summary and write reports/split_summary.json.
"""
import json

import numpy as np
import pandas as pd

from src.utils import load_config, resolve, get_logger

log = get_logger(__name__)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _as_utc(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates, utc=True)


def chronological_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into (train, val, test) by the seasons in `cfg["split"]`.

    Rows from seasons outside the split config are ignored. Each returned frame is a copy
    that keeps the input row order and the input index. Raises ValueError if a season is
    assigned to more than one set, if any set is empty, or if the sets overlap in time
    (max(train.date) >= min(val.date) or max(val.date) >= min(test.date)).
    """
    _require_columns(df, ["season", "date"])
    split_cfg = cfg["split"]
    train_seasons = [str(s) for s in split_cfg["train_seasons"]]
    val_season, test_season = str(split_cfg["val_season"]), str(split_cfg["test_season"])

    assigned = train_seasons + [val_season, test_season]
    duplicated = sorted({s for s in assigned if assigned.count(s) > 1})
    if duplicated:
        raise ValueError(f"Season(s) assigned to more than one split set: {duplicated}")

    season = df["season"].astype(str)
    parts = {
        "train": df.loc[season.isin(train_seasons)].copy(),
        "val": df.loc[season == val_season].copy(),
        "test": df.loc[season == test_season].copy(),
    }
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"Split set '{name}' is empty; check cfg['split'] seasons against the data")

    train, val, test = parts["train"], parts["val"], parts["test"]
    bounds = {name: (_as_utc(p["date"]).min(), _as_utc(p["date"]).max()) for name, p in parts.items()}
    if bounds["train"][1] >= bounds["val"][0]:
        raise ValueError(f"Train max date {bounds['train'][1]} is not before val min date {bounds['val'][0]}")
    if bounds["val"][1] >= bounds["test"][0]:
        raise ValueError(f"Val max date {bounds['val'][1]} is not before test min date {bounds['test'][0]}")
    return train, val, test


def season_folds(df: pd.DataFrame, seasons: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: fold i trains on seasons[:i] and validates on seasons[i].

    Indices are positional (for `df.iloc` / numpy arrays), independent of `df.index`, and
    ascending in `df` row order. Raises ValueError if fewer than two seasons are given, a
    season is repeated or has no rows, or (when `df` has a `date` column) a fold's
    validation season does not start strictly after its training seasons end.
    """
    _require_columns(df, ["season"])
    seasons = [str(s) for s in seasons]
    if len(seasons) < 2:
        raise ValueError(f"Need at least two seasons for season folds, got {seasons}")
    if len(set(seasons)) != len(seasons):
        raise ValueError(f"Seasons must be unique, got {seasons}")

    values = df["season"].astype(str).to_numpy()
    empty = [s for s in seasons if not (values == s).any()]
    if empty:
        raise ValueError(f"Season(s) with no rows: {empty}")

    dates = _as_utc(df["date"]).dt.tz_localize(None).to_numpy() if "date" in df.columns else None
    folds = []
    for i in range(1, len(seasons)):
        train_idx = np.flatnonzero(np.isin(values, seasons[:i]))
        val_idx = np.flatnonzero(values == seasons[i])
        if dates is not None and dates[train_idx].max() >= dates[val_idx].min():
            raise ValueError(f"Fold {i}: season {seasons[i]} does not start after {seasons[:i]} end")
        folds.append((train_idx, val_idx))
    return folds


def _describe(part: pd.DataFrame, target: str) -> dict:
    dates = _as_utc(part["date"])
    return {
        "seasons": sorted(part["season"].astype(str).unique().tolist()),
        "rows": int(len(part)),
        "players": int(part["player_id"].nunique()) if "player_id" in part.columns else None,
        "date_min": dates.min().isoformat(),
        "date_max": dates.max().isoformat(),
        "positives": int(part[target].sum()),
        "base_rate": round(float(part[target].mean()), 6),
    }


def main() -> None:
    cfg = load_config()
    target = cfg["features"]["target"]
    path = resolve(cfg["paths"]["features"])
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}. Run `python -m src.features` first.")
    df = pd.read_parquet(path)
    _require_columns(df, ["season", "date", target])

    train, val, test = chronological_split(df, cfg)
    summary = {name: _describe(part, target) for name, part in (("train", train), ("val", val), ("test", test))}
    summary["ignored_rows"] = int(len(df) - len(train) - len(val) - len(test))

    train_seasons = [str(s) for s in cfg["split"]["train_seasons"]]
    folds = season_folds(train, train_seasons)
    summary["cv_folds"] = [
        {"fold": i, "train_seasons": train_seasons[:i], "val_season": train_seasons[i],
         "train_rows": int(len(tr)), "val_rows": int(len(va)),
         "val_base_rate": round(float(train[target].iloc[va].mean()), 6)}
        for i, (tr, va) in enumerate(folds, start=1)
    ]

    for name in ("train", "val", "test"):
        s = summary[name]
        log.info("%-5s seasons=%s rows=%d players=%s dates=%s..%s base_rate=%.4f",
                 name, ",".join(s["seasons"]), s["rows"], s["players"], s["date_min"][:10],
                 s["date_max"][:10], s["base_rate"])
    for f in summary["cv_folds"]:
        log.info("cv fold %d: train=%d rows (%s) val=%d rows (%s) val_base_rate=%.4f", f["fold"],
                 f["train_rows"], ",".join(f["train_seasons"]), f["val_rows"], f["val_season"], f["val_base_rate"])
    if summary["ignored_rows"]:
        log.info("ignored %d rows from seasons outside the split config", summary["ignored_rows"])

    out = resolve(cfg["paths"]["reports_dir"]) / "split_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
