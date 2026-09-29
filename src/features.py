"""Leakage-safe feature engineering: clean player matches -> model table (eligible rows only).

Every rolling/expanding feature for row t is built from strictly earlier rows (shift(1) inside
the player, or inside the opponent team), so row t's own stats never reach its features.
"""
import time

import pandas as pd

from src.utils import get_logger, load_config, resolve

log = get_logger(__name__)

ROLL_STATS = ["goals", "assists", "ga", "minutes", "threat", "creativity", "xg", "xa"]
P90_STATS = ["goals", "assists", "threat"]
POSITIONS = ["DEF", "MID", "FWD"]
CLEAN_REQUIRED = [
    "player_id", "player_name", "date", "season", "matchweek", "fixture_id", "team", "opponent",
    "home_away", "position", "minutes", "goals", "assists", "threat", "creativity", "xg", "xa",
]
TEAM_REQUIRED = ["season", "fixture_id", "date", "team", "goals_against"]
JOIN_KEYS = ["season", "fixture_id", "opponent"]


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _player_form(df: pd.DataFrame, windows: list[int], p90_min: float) -> pd.DataFrame:
    """Rolling means and per-90 rates over each player's previous N appearances."""
    groups = df["player_id"]
    prev = df[ROLL_STATS].astype(float).groupby(groups, sort=False).shift(1)
    out = {}
    for n in windows:
        roll = prev.groupby(groups, sort=False).rolling(n, min_periods=1)
        means = roll.mean().droplevel(0)
        sums = roll.sum().droplevel(0)
        for stat in ROLL_STATS:
            out[f"{stat}_roll{n}"] = means[stat]
        minutes = sums["minutes"]
        for stat in P90_STATS:
            out[f"{stat}_p90_roll{n}"] = (90.0 * sums[stat] / minutes).where(minutes >= p90_min)
    return pd.DataFrame(out).reindex(df.index)


def _opponent_form(team_matches: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Each team's mean goals conceded over its previous N matches, keyed for the opponent join."""
    tm = team_matches.copy()
    tm["date"] = pd.to_datetime(tm["date"], utc=True)
    dupes = tm.duplicated(["season", "fixture_id", "team"])
    if dupes.any():
        raise ValueError(f"team_matches has {int(dupes.sum())} duplicate (season, fixture_id, team) rows")
    tm = tm.sort_values(["team", "date", "fixture_id"], kind="mergesort").reset_index(drop=True)
    groups = tm["team"]
    prev = tm["goals_against"].astype(float).groupby(groups, sort=False).shift(1)
    out = tm[["season", "fixture_id", "team"]].rename(columns={"team": "opponent"})
    for n in windows:
        roll = prev.groupby(groups, sort=False).rolling(n, min_periods=1).mean()
        out[f"opp_conceded_roll{n}"] = roll.droplevel(0).reindex(tm.index)
    return out


def _prior_rate(elig: pd.DataFrame, target: str, min_prior: int) -> pd.Series:
    """Expanding mean of the target over the player's previous eligible rows (NaN below min_prior)."""
    groups = elig["player_id"]
    prev_y = elig[target].astype(float).groupby(groups, sort=False).shift(1).fillna(0.0)
    prev_sum = prev_y.groupby(groups, sort=False).cumsum()
    prev_n = elig.groupby("player_id", sort=False).cumcount()
    return (prev_sum / prev_n).where(prev_n >= min_prior)


def build_features(clean: pd.DataFrame, team_matches: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pure function: clean appearances + team results -> eligible-row feature table."""
    fcfg = cfg["features"]
    windows = sorted(int(n) for n in fcfg["windows"])
    target = fcfg["target"]
    _require(clean, CLEAN_REQUIRED, "clean")
    _require(team_matches, TEAM_REQUIRED, "team_matches")

    df = clean.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["player_id", "date", "fixture_id"], kind="mergesort").reset_index(drop=True)
    df["ga"] = df["goals"] + df["assists"]
    df[target] = (df["ga"] >= 1).astype(int)
    df["eligible"] = df["minutes"] >= cfg["cleaning"]["min_minutes"]

    form = _player_form(df, windows, fcfg["p90_min_minutes"])
    df = pd.concat([df.drop(columns=form.columns, errors="ignore"), form], axis=1)
    by_player = df.groupby("player_id", sort=False)
    df["prior_matches"] = by_player.cumcount()
    df["matches_in_window"] = df["prior_matches"].clip(upper=max(windows))
    df["days_since_last_match"] = by_player["date"].diff() / pd.Timedelta(days=1)
    df["is_home"] = (df["home_away"] == "H").astype(int)
    for pos in POSITIONS:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)

    elig = df.loc[df["eligible"]].reset_index(drop=True)
    elig["player_prior_rate"] = _prior_rate(elig, target, int(fcfg["min_prior_matches"]))

    opp = _opponent_form(team_matches, windows)
    n_before = len(elig)
    merged = elig.merge(opp, on=JOIN_KEYS, how="left", validate="many_to_one", indicator=True)
    if len(merged) != n_before:
        raise RuntimeError(f"opponent join changed row count: {n_before} -> {len(merged)}")
    unmatched = int((merged["_merge"] == "left_only").sum())
    if unmatched:
        log.warning("%d eligible rows have no team_matches row for their opponent/fixture", unmatched)

    columns = list(fcfg["id_columns"]) + [target] + list(fcfg["feature_columns"])
    missing = [c for c in columns if c not in merged.columns]
    if missing:
        raise KeyError(f"config columns not produced by build_features: {missing}")
    out = merged[columns].reset_index(drop=True)
    non_numeric = [c for c in fcfg["feature_columns"] if not pd.api.types.is_numeric_dtype(out[c])]
    if non_numeric:
        raise TypeError(f"non-numeric feature columns: {non_numeric}")
    return out


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    clean_path, tm_path = resolve(paths["clean"]), resolve(paths["team_matches"])
    for p in (clean_path, tm_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run `python -m src.clean` first")

    t0 = time.perf_counter()
    clean = pd.read_parquet(clean_path)
    team_matches = pd.read_parquet(tm_path)
    log.info("Loaded clean: %d rows, team_matches: %d rows", len(clean), len(team_matches))
    feats = build_features(clean, team_matches, cfg)

    out_path = resolve(paths["features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)

    target = cfg["features"]["target"]
    log.info("Wrote %s: %d rows x %d cols in %.1fs", out_path, len(feats), feats.shape[1],
             time.perf_counter() - t0)
    log.info("Base rate (%s=1): %.4f", target, feats[target].mean())
    per_season = feats.groupby("season")[target].agg(rows="size", base_rate="mean")
    for season, row in per_season.iterrows():
        log.info("  %s: %6d rows, base rate %.4f", season, int(row["rows"]), row["base_rate"])
    nan_share = feats[cfg["features"]["feature_columns"]].isna().mean().sort_values(ascending=False)
    log.info("NaN share per feature:")
    for col, share in nan_share.items():
        log.info("  %-24s %.4f", col, share)


if __name__ == "__main__":
    main()
