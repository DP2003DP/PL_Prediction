"""Stage 2: clean raw player/team match rows into data/processed parquet files.

Run: python -m src.clean
"""
from pathlib import Path

import pandas as pd

from src.utils import get_logger, load_config, resolve

log = get_logger(__name__)

PLAYER_SCHEMA = {
    "player_id": "int", "fpl_element": "int", "player_name": "str", "date": "datetime",
    "season": "str", "matchweek": "int", "fixture_id": "int", "team": "str",
    "opponent": "str", "home_away": "str", "position": "str", "minutes": "int",
    "goals": "int", "assists": "int", "xg": "float", "xa": "float", "threat": "float",
    "creativity": "float", "influence": "float", "ict_index": "float", "bps": "int",
    "starts": "float", "team_goals": "int", "opp_goals": "int",
}
TEAM_SCHEMA = {
    "season": "str", "fixture_id": "int", "date": "datetime", "team": "str",
    "opponent": "str", "home_away": "str", "goals_for": "int", "goals_against": "int",
}
KEY_FIELDS = ["date", "player_id", "minutes"]
ZERO_FILL = ["goals", "assists", "threat", "creativity", "influence", "ict_index", "bps"]
VALID_POSITIONS = {"GK", "DEF", "MID", "FWD"}
LOG_COLUMNS = ["step", "description", "rows_before", "rows_after", "rows_removed"]
TEAM_KEY = ["season", "fixture_id", "team"]


def _to_int(s: pd.Series, name: str) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    frac = num.notna() & (num.astype("float64") % 1 != 0)
    if frac.any():
        raise ValueError(f"Column {name!r} has {int(frac.sum())} non-integer values, e.g. {num[frac].iloc[0]}")
    return num.astype("Int64")


def coerce_types(df: pd.DataFrame, schema: dict[str, str], label: str) -> pd.DataFrame:
    """Select schema columns in order and enforce dtypes; unparseable values become missing."""
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}")
    out = df[list(schema)].copy()
    for col, kind in schema.items():
        present = out[col].notna()
        if kind == "int":
            out[col] = _to_int(out[col], col)
        elif kind == "float":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        elif kind == "datetime":
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce", format="ISO8601")
        else:
            out[col] = out[col].astype("str").str.strip()
        bad = int((present & out[col].isna()).sum())
        if bad:
            log.warning("%s: %d unparseable values in %r set to missing", label, bad, col)
    return out


def _finalize_ints(df: pd.DataFrame, schema: dict[str, str], label: str) -> pd.DataFrame:
    casts = {}
    for col, kind in schema.items():
        if kind != "int":
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing:
            log.warning("%s: %r has %d missing values; kept as nullable Int64", label, col, n_missing)
        else:
            casts[col] = "int64"
    return df.astype(casts)


def _record(records: list[dict], step: str, description: str, before: int, after: int) -> None:
    rec = {"step": step, "description": description, "rows_before": int(before),
           "rows_after": int(after), "rows_removed": int(before) - int(after)}
    records.append(rec)
    log.info("step %-7s %8d -> %8d (-%d)  %s", step, before, after, rec["rows_removed"], description)


def _dedupe_player_date(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    key = ["player_id", "date"]
    keyed = df[key].notna().all(axis=1)
    dup = keyed & df.duplicated(key, keep=False)
    groups = df[dup].groupby(key)
    n_groups = groups.ngroups
    cross_fixture = int((groups["fixture_id"].nunique(dropna=False) > 1).sum()) if n_groups else 0
    cross_team = int((groups["team"].nunique(dropna=False) > 1).sum()) if n_groups else 0
    ranked = df.sort_values(["minutes", "bps"], ascending=False, na_position="last", kind="mergesort")
    drop = ranked.index[keyed.loc[ranked.index] & ranked.duplicated(key, keep="first")]
    desc = (f"dedupe on (player_id, date): {n_groups} duplicate groups ({int(dup.sum())} rows; "
            f"{cross_fixture} span >1 fixture_id, {cross_team} span >1 team); "
            "rule: keep the row with most minutes, then highest bps, then first in file order; "
            "team is kept per row (no merging across teams)")
    if n_groups:
        sample = df[dup].sort_values(key).head(6)[["player_id", "player_name", "date", "fixture_id", "team", "minutes"]]
        log.warning("(player_id, date) duplicates found; sample:\n%s", sample.to_string(index=False))
    return df.drop(index=drop).reset_index(drop=True), desc


def _gw_ranges(pairs: list[tuple[str, int]]) -> str:
    by_season: dict[str, list[int]] = {}
    for season, gw in sorted(pairs):
        by_season.setdefault(season, []).append(int(gw))
    parts = []
    for season, gws in by_season.items():
        runs, start = [], gws[0]
        for prev, cur in zip(gws, gws[1:] + [None]):
            if cur is None or cur != prev + 1:
                runs.append(str(start) if start == prev else f"{start}-{prev}")
                start = cur
        parts.append(f"{season} GW{','.join(runs)}")
    return "; ".join(parts) or "none"


def _null_unpublished_xg(df: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    """FPL stored xg/xa/starts as 0 before it published them; a gameweek where all are 0 means unavailable."""
    keys = [df["season"], df["matchweek"]]
    has_value = (df["xg"].notna() | df["xa"].notna()).groupby(keys, dropna=False).transform("any")
    nonzero = ((df["xg"].fillna(0) != 0) | (df["xa"].fillna(0) != 0)).groupby(keys, dropna=False).transform("any")
    mask = has_value & ~nonzero
    affected = _gw_ranges(list(set(zip(df.loc[mask, "season"], df.loc[mask, "matchweek"]))))
    df = df.copy()
    df.loc[mask, ["xg", "xa", "starts"]] = float("nan")
    return df, int(mask.sum()), affected


def clean_player_matches(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Apply cleaning steps 1-7 (spec section 4, contract section 2). Returns (clean, log records)."""
    records: list[dict] = []
    min_minutes = cfg["cleaning"]["min_minutes"]
    drop_positions = set(cfg["cleaning"].get("drop_positions", ["GK"]))

    df = coerce_types(raw, PLAYER_SCHEMA, "player_matches")
    _record(records, "0", "raw rows loaded (dtypes enforced)", len(df), len(df))

    n = len(df)
    df = df.drop_duplicates(ignore_index=True)
    _record(records, "1", "drop exact duplicate rows", n, len(df))

    n = len(df)
    df, desc = _dedupe_player_date(df)
    _record(records, "2", desc, n, len(df))

    n = len(df)
    missing = {c: int(df[c].isna().sum()) for c in KEY_FIELDS}
    df = df[df[KEY_FIELDS].notna().all(axis=1)].reset_index(drop=True)
    df = df.astype({"player_id": "int64", "minutes": "int64"})
    _record(records, "3", f"drop rows missing key fields (missing counts: {missing})", n, len(df))

    filled = {c: int(df[c].isna().sum()) for c in ZERO_FILL}
    df = df.fillna({c: 0 for c in ZERO_FILL}).astype({"goals": "int64", "assists": "int64", "bps": "int64"})
    _record(records, "4", f"fill missing count stats with 0 (filled: {filled}); xg/xa/starts left missing",
            len(df), len(df))

    df, n_nulled, affected = _null_unpublished_xg(df)
    _record(records, "4b", f"set xg/xa/starts to NaN in gameweeks where every non-null xg and xa is 0 "
            f"(not yet published): {n_nulled} rows in {affected}", len(df), len(df))

    n = len(df)
    df = df[~df["position"].isin(drop_positions)].reset_index(drop=True)
    _record(records, "5", f"drop positions {sorted(drop_positions)}", n, len(df))
    unknown = ~df["position"].isin(VALID_POSITIONS - drop_positions)
    if unknown.any():
        counts = df.loc[unknown, "position"].value_counts(dropna=False).to_dict()
        log.warning("Unrecognized position labels (scraper should normalize these): %s", counts)
        n = len(df)
        df = df[~unknown].reset_index(drop=True)
        _record(records, "5b", f"drop rows with unrecognized position labels {counts}", n, len(df))

    n = len(df)
    negative = int((df["minutes"] < 0).sum())
    df = df[df["minutes"] > 0].reset_index(drop=True)
    extra = f"; includes {negative} negative-minute rows" if negative else ""
    _record(records, "6", f"drop minutes == 0 (not an appearance){extra}", n, len(df))
    df = df.assign(eligible=df["minutes"] >= min_minutes)
    _record(records, "6b", f"eligible = minutes >= {min_minutes}: rows_after is the eligible count "
            "(flag only, nothing dropped; rows_removed = non-eligible appearances kept)",
            len(df), int(df["eligible"].sum()))

    n = len(df)
    df = df.sort_values(["player_id", "date", "fixture_id"], kind="mergesort").reset_index(drop=True)
    _record(records, "7", "sort by player_id, date (fixture_id as tie-break)", n, len(df))

    df = _finalize_ints(df, PLAYER_SCHEMA, "player_matches")
    return df[list(PLAYER_SCHEMA) + ["eligible"]], records


def clean_team_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Enforce dtypes, dedupe on (season, fixture_id, team), sort by team, date."""
    df = coerce_types(raw, TEAM_SCHEMA, "team_matches")
    n0 = len(df)
    incomplete = df[TEAM_KEY + ["date", "goals_for", "goals_against"]].isna().any(axis=1)
    if incomplete.any():
        log.warning("team_matches: dropping %d rows missing key/date/score", int(incomplete.sum()))
        df = df[~incomplete]
    exact = int(df.duplicated().sum())
    keyed = int(df.duplicated(TEAM_KEY).sum())
    if keyed > exact:
        log.warning("team_matches: %d conflicting (season, fixture_id, team) duplicates; kept first", keyed - exact)
    df = df.drop_duplicates(TEAM_KEY, keep="first")
    df = df.sort_values(["team", "date", "fixture_id"], kind="mergesort").reset_index(drop=True)
    df = _finalize_ints(df, TEAM_SCHEMA, "team_matches")
    log.info("team_matches: %d -> %d rows (%d duplicates, %d incomplete)", n0, len(df), keyed, int(incomplete.sum()))
    return df


def load_season_files(raw_dir: Path, seasons: list[str], prefix: str, schema: dict[str, str]) -> pd.DataFrame:
    paths = {s: raw_dir / f"{prefix}_{s}.csv" for s in seasons}
    missing = [str(p) for p in paths.values() if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty raw files (run `python -m src.scrape` first): {missing}")
    frames = []
    for season, path in paths.items():
        df = pd.read_csv(path, low_memory=False)
        cols = [c for c in schema if c not in df.columns]
        if cols:
            raise ValueError(f"{path}: missing required columns {cols}")
        if df.empty:
            raise ValueError(f"{path}: no data rows")
        off = int((df["season"].astype("str") != season).sum())
        if off:
            log.warning("%s: %d rows have a season value other than %s", path.name, off, season)
        log.info("loaded %s: %d rows", path.name, len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def season_summary(clean: pd.DataFrame, team: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for season, g in clean.groupby("season", sort=True):
        e = g[g["eligible"]]
        rate = float(((e["goals"] + e["assists"]) >= 1).mean()) if len(e) else float("nan")
        n_team = int((team["season"] == season).sum())
        if n_team != 760:
            log.warning("team_matches %s: %d rows (expected 760 = 380 fixtures x 2)", season, n_team)
        _record(records, "summary", f"{season}: appearances -> eligible rows; eligible base rate "
                f"(goals+assists>=1) = {rate:.4f}; players = {g['player_id'].nunique()}; "
                f"team_matches rows = {n_team}", len(g), len(e))
    e = clean[clean["eligible"]]
    rate = float(((e["goals"] + e["assists"]) >= 1).mean()) if len(e) else float("nan")
    _record(records, "summary", f"all seasons: appearances -> eligible rows; eligible base rate = {rate:.4f}",
            len(clean), len(e))
    return records


def consistency_checks(clean: pd.DataFrame, team: pd.DataFrame) -> None:
    """Console-only checks that features.py relies on (opponent join, transfers)."""
    keys = team[TEAM_KEY].rename(columns={"team": "opponent"}).assign(_opp=True)
    joined = clean.merge(keys, on=["season", "fixture_id", "opponent"], how="left")
    own = team[TEAM_KEY].assign(_own=True)
    joined = joined.merge(own, on=TEAM_KEY, how="left")
    opp_ok = joined["_opp"].eq(True)
    own_ok = joined["_own"].eq(True)
    log.info("join check: %d/%d rows match team_matches on opponent, %d/%d on own team",
             int(opp_ok.sum()), len(clean), int(own_ok.sum()), len(clean))
    if not (opp_ok.all() and own_ok.all()):
        bad = joined.loc[~(opp_ok & own_ok), ["season", "fixture_id", "team", "opponent"]].drop_duplicates()
        log.warning("unmatched (season, fixture_id, team/opponent), sample:\n%s", bad.head(10).to_string(index=False))
    teams_per = clean.groupby(["season", "player_id"])["team"].nunique()
    log.info("players with >1 team within a season (transfers, team kept per row): %d", int((teams_per > 1).sum()))


def main() -> None:
    cfg = load_config()
    seasons = cfg["data"]["seasons"]
    raw_dir = resolve(cfg["paths"]["raw_dir"])

    raw_players = load_season_files(raw_dir, seasons, "player_matches", PLAYER_SCHEMA)
    clean, records = clean_player_matches(raw_players, cfg)

    raw_team = load_season_files(raw_dir, seasons, "team_matches", TEAM_SCHEMA)
    team = clean_team_matches(raw_team)
    _record(records, "team", "team_matches: drop incomplete rows, dedupe on (season, fixture_id, team)",
            len(raw_team), len(team))

    records += season_summary(clean, team)
    consistency_checks(clean, team)

    paths = {k: resolve(cfg["paths"][k]) for k in ("clean", "team_matches", "cleaning_log")}
    paths["clean"].parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(paths["clean"], index=False)
    team.to_parquet(paths["team_matches"], index=False)
    log_df = pd.DataFrame(records, columns=LOG_COLUMNS)
    log_df.to_csv(paths["cleaning_log"], index=False)

    with pd.option_context("display.max_colwidth", 140, "display.width", 250):
        print(log_df.to_string(index=False))
    log.info("wrote %s (%d rows), %s (%d rows), %s", paths["clean"], len(clean),
             paths["team_matches"], len(team), paths["cleaning_log"])


if __name__ == "__main__":
    main()
