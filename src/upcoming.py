"""Predict P(goal or assist | plays >= 30 min) for the next gameweek of the in-progress season.

Run: python -m src.upcoming   (after `python -m src.live`)
Each upcoming fixture is scored through src.features.build_features via one placeholder row per
candidate. Because build_features only uses strictly earlier rows, a placeholder's features come
from real history alone, provided no player or team has two placeholders in the same call.
"""
import json
import time
from datetime import datetime, timezone

import pandas as pd
from xgboost import XGBClassifier

from src.clean import clean_player_matches, clean_team_matches
from src.evaluate import load_selection, to_matrix
from src.features import CLEAN_REQUIRED, TEAM_REQUIRED, build_features
from src.utils import get_logger, load_config, resolve

log = get_logger(__name__)

PLAYER_COLUMNS = ["player_id", "player_name", "team", "position", "status", "chance_of_playing_next_round", "news"]
FIXTURE_COLUMNS = ["fixture_id", "gameweek", "date", "home_team", "away_team"]
AVAILABILITY = ["status", "chance_of_playing_next_round", "news", "likely_starter"]
CANDIDATE_COLUMNS = ["player_id", "player_name", "date", "season", "matchweek", "fixture_id", "team", "opponent",
                     "home_away", "position"] + AVAILABILITY + ["prior_matches"]
PLACEHOLDER_STATS = {"minutes": 90, "goals": 0, "assists": 0, "threat": 0.0, "creativity": 0.0,
                     "xg": float("nan"), "xa": float("nan")}
PLACEHOLDER_GOALS_AGAINST = 0
KEY = ["player_id", "season", "fixture_id"]
FIXTURE_KEY = ["season", "fixture_id"]


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _fixture_sides(next_fixtures: pd.DataFrame) -> pd.DataFrame:
    fx = next_fixtures.assign(date=pd.to_datetime(next_fixtures["date"], utc=True))
    home = fx.rename(columns={"home_team": "team", "away_team": "opponent"}).assign(home_away="H")
    away = fx.rename(columns={"away_team": "team", "home_team": "opponent"}).assign(home_away="A")
    cols = ["fixture_id", "gameweek", "date", "team", "opponent", "home_away"]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def select_candidates(players: pd.DataFrame, next_fixtures: pd.DataFrame, history_clean: pd.DataFrame,
                      history_team_matches: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per (available outfield player, next-GW fixture of his current team)."""
    _require(players, PLAYER_COLUMNS, "players")
    _require(next_fixtures, FIXTURE_COLUMNS, "next_fixtures")
    if players["player_id"].duplicated().any():
        raise ValueError("players has duplicate player_id rows")
    ucfg = cfg["upcoming"]
    min_minutes = cfg["cleaning"]["min_minutes"]

    pool = players[~players["position"].isin(cfg["cleaning"]["drop_positions"])
                   & ~players["status"].isin(ucfg["exclude_status"])]
    cands = pool[PLAYER_COLUMNS].merge(_fixture_sides(next_fixtures), on="team", how="inner")
    cands = cands.rename(columns={"gameweek": "matchweek"}).assign(season=ucfg["season"])

    tm = history_team_matches.assign(date=pd.to_datetime(history_team_matches["date"], utc=True))
    tm = tm.sort_values(["team", "date", "fixture_id"], kind="mergesort")
    recent = tm.groupby("team", sort=False).tail(int(ucfg["likely_starter_lookback"]))[["season", "fixture_id", "team"]]
    played = history_clean.loc[history_clean["minutes"] >= min_minutes, ["player_id", "season", "fixture_id", "team"]]
    n_recent = (played.merge(recent, on=["season", "fixture_id", "team"])
                .groupby(["player_id", "team"]).size().rename("n_recent").reset_index())
    cands = cands.merge(n_recent, on=["player_id", "team"], how="left")
    cands["likely_starter"] = cands["n_recent"].fillna(0) >= int(ucfg["likely_starter_min_matches"])
    cands["prior_matches"] = cands["player_id"].map(history_clean.groupby("player_id").size()).fillna(0)
    cands = cands.astype({"prior_matches": "int64", "matchweek": "int64", "fixture_id": "int64",
                          "player_id": "int64", "likely_starter": bool})
    cands["news"] = cands["news"].fillna("").astype(str)
    cands["chance_of_playing_next_round"] = pd.to_numeric(cands["chance_of_playing_next_round"], errors="coerce")
    return (cands[CANDIDATE_COLUMNS].sort_values(["date", "fixture_id", "team", "player_id"], kind="mergesort")
            .reset_index(drop=True))


def _fixture_passes(candidates: pd.DataFrame) -> list[set[tuple]]:
    """Greedy partition of fixtures so that no team appears twice within one pass."""
    fx = candidates.drop_duplicates(FIXTURE_KEY).sort_values(["date", "fixture_id"], kind="mergesort")
    passes: list[tuple[set, set]] = []
    for season, fid, team, opp in fx[FIXTURE_KEY + ["team", "opponent"]].itertuples(index=False):
        for teams, fixtures in passes:
            if team not in teams and opp not in teams:
                teams.update((team, opp))
                fixtures.add((season, fid))
                break
        else:
            passes.append(({team, opp}, {(season, fid)}))
    return [fixtures for _, fixtures in passes]


def _placeholders(part: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = part[[c for c in CLEAN_REQUIRED if c not in PLACEHOLDER_STATS]].assign(**PLACEHOLDER_STATS)
    sides = pd.concat([part[["season", "fixture_id", "date", "team"]],
                       part[["season", "fixture_id", "date", "opponent"]].rename(columns={"opponent": "team"})])
    team_rows = sides.drop_duplicates(["season", "fixture_id", "team"]).assign(goals_against=PLACEHOLDER_GOALS_AGAINST)
    return rows[CLEAN_REQUIRED], team_rows[TEAM_REQUIRED]


def build_upcoming_features(history_clean: pd.DataFrame, history_team_matches: pd.DataFrame,
                            candidates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Features for each candidate fixture, computed from real history only (no placeholder influence)."""
    _require(candidates, CANDIDATE_COLUMNS, "candidates")
    if candidates.duplicated(KEY).any():
        raise ValueError("candidates has duplicate (player_id, season, fixture_id) rows")
    hist_keys = history_team_matches[FIXTURE_KEY].drop_duplicates()
    clash = candidates[FIXTURE_KEY].drop_duplicates().merge(hist_keys, on=FIXTURE_KEY)
    if len(clash):
        raise ValueError(f"upcoming fixtures already in history_team_matches: {clash.to_dict('records')}")

    fcfg = cfg["features"]
    id_cols = [c for c in fcfg["id_columns"] if c not in ("minutes", "goals", "assists")]
    hist = history_clean[CLEAN_REQUIRED]
    hist_tm = history_team_matches[TEAM_REQUIRED]
    fixture_keys = pd.MultiIndex.from_frame(candidates[FIXTURE_KEY])
    parts = []
    for fixtures in _fixture_passes(candidates):
        part = candidates[fixture_keys.isin(list(fixtures))]
        if part["player_id"].duplicated().any():
            raise RuntimeError("a player has two placeholder rows in one build_features pass")
        rows, team_rows = _placeholders(part)
        feats = build_features(pd.concat([hist, rows], ignore_index=True),
                               pd.concat([hist_tm, team_rows], ignore_index=True), cfg)
        got = feats.merge(part[KEY], on=KEY, how="inner")
        if len(got) != len(part):
            raise RuntimeError(f"expected {len(part)} placeholder rows back from build_features, got {len(got)}")
        parts.append(got)
    if not parts:
        return pd.DataFrame(columns=id_cols + list(fcfg["feature_columns"]) + AVAILABILITY)
    out = pd.concat(parts, ignore_index=True).merge(candidates[KEY + AVAILABILITY], on=KEY, validate="one_to_one")
    out = out[id_cols + list(fcfg["feature_columns"]) + AVAILABILITY]
    return out.sort_values(["date", "fixture_id", "team", "player_id"], kind="mergesort").reset_index(drop=True)


def _flag(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def select_target_fixtures(fixtures: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, dict]:
    """Unstarted fixtures to score: the current GW's remaining ones if any, else the next GW's."""
    _require(fixtures, FIXTURE_COLUMNS + ["started", "finished"], "fixtures")
    started, finished = _flag(fixtures["started"]), _flag(fixtures["finished"])
    kickoff = pd.to_datetime(fixtures["date"], utc=True, errors="coerce", format="ISO8601")
    gameweek = pd.to_numeric(fixtures["gameweek"], errors="coerce")
    open_ = ~started & ~finished & kickoff.notna()
    current, nxt = meta.get("current_gw"), meta.get("next_gw")
    if current is not None and (open_ & (gameweek == int(current))).any():
        target_gw, mode = int(current), "current_gw_remaining"
    elif nxt is not None:
        target_gw, mode = int(nxt), "next_gw"
    else:
        raise ValueError(f"no unstarted fixtures in current_gw and no next_gw (season over?): {meta}")
    in_gw = gameweek == target_gw
    chosen = in_gw & open_
    if not chosen.any():
        raise ValueError(f"GW{target_gw} has no unstarted fixtures with a kickoff time")
    scored_teams = set(fixtures.loc[chosen, "home_team"]) | set(fixtures.loc[chosen, "away_team"])
    plays_again = fixtures["home_team"].isin(scored_teams) | fixtures["away_team"].isin(scored_teams)
    pending = ~finished & ((gameweek < target_gw) | (in_gw & started & plays_again))
    teams = pd.concat([fixtures.loc[pending, "home_team"], fixtures.loc[pending, "away_team"]])
    info = {
        "target_gw": target_gw, "target_mode": mode, "pending_fixtures": int(pending.sum()),
        "pending_teams": sorted(teams.astype(str).unique()), "excluded_in_gw": int((in_gw & ~chosen).sum()),
        "deadline": meta.get("next_deadline") if mode == "next_gw" else None,
        "first_kickoff": _iso(kickoff[chosen].min()), "last_kickoff": _iso(kickoff[chosen].max()),
    }
    return fixtures[chosen].reset_index(drop=True), info


def _read(path, kind: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.live` first")
    if kind == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path, low_memory=False)


def _utc(iso: str) -> pd.Timestamp:
    ts = pd.Timestamp(iso)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_current_season(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    season = cfg["upcoming"]["season"]
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    raw_players = _read(raw_dir / f"player_matches_{season}.csv", "csv")
    raw_team = _read(raw_dir / f"team_matches_{season}.csv", "csv")
    unsplit = int(raw_players["threat"].isna().sum())
    if unsplit:
        log.warning("%d %s rows have no threat/creativity (double-gameweek split); clean_player_matches fills "
                    "them with 0, which biases those players' threat/creativity form down", unsplit, season)
    clean, _ = clean_player_matches(raw_players, cfg)
    team = clean_team_matches(raw_team)
    return clean, team


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    ucfg, paths = cfg["upcoming"], cfg["paths"]
    season = ucfg["season"]
    now = pd.Timestamp.now(tz="UTC")

    for key in ("clean", "team_matches"):
        if not resolve(paths[key]).exists():
            raise FileNotFoundError(f"{resolve(paths[key])} not found; run `python -m src.clean` first")
    clean = pd.read_parquet(resolve(paths["clean"]))
    team_matches = pd.read_parquet(resolve(paths["team_matches"]))
    if season in set(clean["season"]):
        raise ValueError(f"{season} is already in {paths['clean']}; the model seasons must not include it")
    cur_clean, cur_team = load_current_season(cfg)
    resolve(paths["current_clean"]).parent.mkdir(parents=True, exist_ok=True)
    cur_clean.to_parquet(resolve(paths["current_clean"]), index=False)
    cur_team.to_parquet(resolve(paths["current_team_matches"]), index=False)
    log.info("%s: %d clean appearances (%d eligible), %d team_matches rows", season, len(cur_clean),
             int(cur_clean["eligible"].sum()), len(cur_team))
    history = pd.concat([clean, cur_clean], ignore_index=True)
    team_history = pd.concat([team_matches, cur_team], ignore_index=True)

    live_dir = resolve(paths["live_dir"]) / season
    players = _read(live_dir / "players.csv", "csv")
    fixtures = _read(live_dir / "fixtures.csv", "csv")
    meta = _read(live_dir / "meta.json", "json")
    next_fixtures, target = select_target_fixtures(fixtures, meta)
    gw = target["target_gw"]
    log.info("target GW%d (%s): %d fixtures, %s to %s", gw, target["target_mode"], len(next_fixtures),
             target["first_kickoff"], target["last_kickoff"])
    if target["excluded_in_gw"]:
        log.warning("GW%d: %d fixtures already started or without a kickoff time are excluded",
                    gw, target["excluded_in_gw"])
    if target["pending_fixtures"]:
        log.warning("%d fixtures before GW%d are not finished (postponed or in play); history is incomplete "
                    "for %s", target["pending_fixtures"], gw, target["pending_teams"])
    unknown = sorted(set(next_fixtures["home_team"]).union(next_fixtures["away_team"]) - set(team_history["team"]))
    if unknown:
        log.warning("teams with no match history (opp_conceded will be NaN): %s", unknown)

    candidates = select_candidates(players, next_fixtures, history, team_history, cfg)
    if candidates.empty:
        raise RuntimeError(f"GW{gw}: no candidate players (check players.csv teams vs fixtures.csv)")
    feats = build_upcoming_features(history, team_history, candidates, cfg)

    selection = load_selection(resolve(paths["model_selection"]))
    variant = selection["xgb_variant"]
    threshold = float(selection["thresholds"][variant])
    model = XGBClassifier()
    model.load_model(resolve(paths["xgb_model"]))
    feats["prob"] = model.predict_proba(to_matrix(feats, selection["feature_columns"]))[:, 1].astype("float64")
    feats["call"] = feats["prob"] >= threshold
    feats = feats.sort_values(["prob", "player_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    feats.to_parquet(resolve(paths["upcoming"]), index=False)

    fetched = meta.get("fetched_at")
    fetched_age = (now - _utc(fetched)) / pd.Timedelta(hours=1) if fetched else None
    deadline = target["deadline"]
    upcoming_meta = {
        "season": season, "gameweek": gw, "target_gw": gw, "target_mode": target["target_mode"],
        "pending_fixtures": target["pending_fixtures"], "deadline": deadline,
        "first_kickoff": target["first_kickoff"], "last_kickoff": target["last_kickoff"],
        "generated_at": now.isoformat(timespec="seconds"), "data_through": meta.get("last_finished_kickoff"),
        "live_fetched_at": meta.get("fetched_at"), "n_fixtures": int(len(next_fixtures)),
        "n_candidates": int(len(feats)), "n_likely_starters": int(feats["likely_starter"].sum()),
        "model": {"variant": variant, "trained_at": selection.get("trained_at"), "threshold": threshold,
                  "train_seasons": selection["train_seasons"], "val_season": selection["val_season"],
                  "test_season": selection["test_season"]},
    }
    resolve(paths["upcoming_meta"]).write_text(json.dumps(upcoming_meta, indent=2), encoding="utf-8")

    if fetched_age is not None and fetched_age > ucfg["snapshot_max_age_hours"]:
        log.warning("live snapshot is %.1f h old (> %s h); rerun `python -m src.live --refresh`",
                    fetched_age, ucfg["snapshot_max_age_hours"])
    if target["target_mode"] == "next_gw":
        cutoff = _utc(deadline) if deadline else _utc(target["first_kickoff"])
        if now >= cutoff:
            log.warning("GW%d deadline %s has passed; rerun `python -m src.live --refresh`", gw, _iso(cutoff))
    elif now >= _utc(target["last_kickoff"]):
        log.warning("all targeted GW%d fixtures have kicked off (last %s); rerun `python -m src.live --refresh`",
                    gw, target["last_kickoff"])
    log.info("GW%d %s (deadline %s, data through %s): %d fixtures, %d candidates, %d likely starters, %d calls",
             gw, target["target_mode"], deadline, upcoming_meta["data_through"], len(next_fixtures), len(feats),
             upcoming_meta["n_likely_starters"], int(feats["call"].sum()))
    top = feats.head(10)[["player_name", "team", "opponent", "home_away", "position", "likely_starter", "prob"]]
    log.info("top 10 by prob:\n%s", top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log.info("wrote %s and %s in %.1fs", resolve(paths["upcoming"]), resolve(paths["upcoming_meta"]),
             time.perf_counter() - t0)


if __name__ == "__main__":
    main()
