"""Fetch the in-progress season from the live FPL API and normalize it to the raw schemas.

Source: the official Fantasy Premier League API (cfg["upcoming"]["api_base"]):
/bootstrap-static/, /fixtures/ and /event/{gw}/live/. Output schemas match src/scrape.py.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.scrape import (PLAYER_COLUMNS, PLAYER_FLOAT, PLAYER_INT, DownloadError, Downloader,
                        StructureError, build_team_matches, finished_fixtures, iso_utc)
from src.utils import get_logger, load_config, resolve

log = get_logger(__name__)

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
BOOTSTRAP_KEYS = {
    "events": ["id", "deadline_time", "finished", "data_checked", "is_current", "is_next"],
    "teams": ["id", "code", "name"],
    "elements": ["id", "code", "first_name", "second_name", "team", "element_type", "status",
                 "chance_of_playing_next_round", "news"],
    "element_types": ["id", "singular_name_short"],
}
FIXTURE_KEYS = ["id", "event", "kickoff_time", "team_h", "team_a", "team_h_score",
                "team_a_score", "started", "finished", "stats"]
LIVE_STAT_KEYS = ["minutes", "goals_scored", "assists", "influence", "creativity", "threat",
                  "ict_index", "bps", "starts", "expected_goals", "expected_assists"]
SPLIT_STATS = {"minutes": "minutes", "goals_scored": "goals", "assists": "assists"}
PROPORTIONAL = {"xg": "expected_goals", "xa": "expected_assists", "threat": "threat",
                "creativity": "creativity", "influence": "influence", "ict_index": "ict_index",
                "bps": "bps"}
PLAYERS_COLUMNS = ["player_id", "fpl_element", "player_name", "team", "position", "status",
                   "chance_of_playing_next_round", "news"]
FIXTURES_COLUMNS = ["fixture_id", "gameweek", "date", "home_team", "away_team", "started",
                    "finished", "home_goals", "away_goals"]


def load_json(path: Path, url: str, kind: type) -> dict | list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise StructureError(f"{url}: invalid JSON: {e}") from e
    if not isinstance(data, kind):
        raise StructureError(f"{url}: expected a JSON {kind.__name__}, got {type(data).__name__}")
    return data


def check_items(items, keys: list[str], where: str, allow_empty: bool = False) -> None:
    if not isinstance(items, list) or (not items and not allow_empty):
        raise StructureError(f"{where}: expected a non-empty list")
    for i, item in enumerate(items):
        missing = [k for k in keys if k not in item] if isinstance(item, dict) else keys
        if missing:
            raise StructureError(f"{where}[{i}]: missing keys {missing}")


def parse_bootstrap(path: Path, url: str) -> dict:
    data = load_json(path, url, dict)
    for key, keys in BOOTSTRAP_KEYS.items():
        check_items(data.get(key), keys, f"{url} {key}")
    return data


def parse_fixtures(path: Path, url: str) -> list:
    data = load_json(path, url, list)
    check_items(data, FIXTURE_KEYS, url)
    return data


def parse_event(path: Path, url: str) -> dict:
    data = load_json(path, url, dict)
    check_items(data.get("elements"), ["id", "stats", "explain"], f"{url} elements")
    for el in data["elements"]:
        where = f"{url} element {el['id']}"
        check_items([el["stats"]], LIVE_STAT_KEYS, f"{where} stats")
        check_items(el["explain"], ["fixture", "stats"], f"{where} explain", allow_empty=True)
        for ex in el["explain"]:
            check_items(ex["stats"], ["identifier", "value"], f"{where} explain", allow_empty=True)
    return data


def team_names(boot: dict, cfg: dict) -> pd.Series:
    """API team id -> name, keeping the name used in earlier seasons (matched on team code)."""
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    history: dict[int, str] = {}
    for season in cfg["data"]["seasons"]:
        path = raw_dir / "fpl" / season / "teams.csv"
        if path.exists():
            t = pd.read_csv(path)
            if {"code", "name"} <= set(t.columns):
                history.update(zip(t["code"].astype(int), t["name"]))
    if not history:
        raise FileNotFoundError(f"no model-season teams.csv (with code, name) under "
                                f"{raw_dir / 'fpl'}; run `python -m src.scrape` first")
    names = {}
    for t in boot["teams"]:
        old = history.get(int(t["code"]))
        if old is not None and old != t["name"]:
            log.warning("team code %s is %r in the API but %r in earlier seasons; using %r",
                        t["code"], t["name"], old, old)
        names[int(t["id"])] = old or t["name"]
    tm_path = resolve(cfg["paths"]["team_matches"])
    if tm_path.exists():
        known = set(pd.read_parquet(tm_path, columns=["team"])["team"])
        with_history = {n for n in names.values() if n in set(history.values())}
        missing = sorted(with_history - known)
        if missing:
            log.warning("teams with earlier-season history missing from %s: %s",
                        tm_path.name, missing)
        log.info("teams with history in model seasons: %d of %d; new: %s", len(with_history),
                 len(names), sorted(set(names.values()) - with_history))
    return pd.Series(names)


def build_players(boot: dict, names: pd.Series) -> pd.DataFrame:
    types = {int(t["id"]): t["singular_name_short"] for t in boot["element_types"]}
    expected = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    if any(types.get(k) != v for k, v in expected.items()):
        raise StructureError(f"unexpected element_types {types}")
    el = pd.DataFrame(boot["elements"])
    el["position"] = el["element_type"].map(POSITIONS)
    n_other = int(el["position"].isna().sum())
    if n_other:
        log.info("dropping %d non-player elements (element_type %s)", n_other,
                 sorted(el.loc[el["position"].isna(), "element_type"].unique().tolist()))
    el = el.loc[el["position"].notna()]
    return pd.DataFrame({
        "player_id": el["code"].astype("int64"),
        "fpl_element": el["id"].astype("int64"),
        "player_name": (el["first_name"].str.strip() + " " + el["second_name"].str.strip()),
        "team": el["team"].map(names),
        "team_id": el["team"].astype("int64"),
        "position": el["position"],
        "status": el["status"],
        "chance_of_playing_next_round": pd.to_numeric(el["chance_of_playing_next_round"]),
        "news": el["news"].fillna(""),
    }).sort_values(["team", "position", "player_name"], ignore_index=True)


def fixtures_frame(fixtures: list) -> pd.DataFrame:
    fx = pd.DataFrame(fixtures)[FIXTURE_KEYS[:-1]]
    for col in ["started", "finished"]:
        fx[col] = fx[col].fillna(False).astype(bool)
    return fx


def fixture_sides(fixtures: list) -> dict[tuple[int, int], str]:
    """(fixture, element) -> 'h'/'a' from the fixture's bps list (every player with bps)."""
    sides = {}
    for f in fixtures:
        for stat in f["stats"] or []:
            if stat.get("identifier") == "bps":
                for side in ("h", "a"):
                    for item in stat.get(side, []):
                        sides[(int(f["id"]), int(item["element"]))] = side
    return sides


def explain_rows(lives: dict[int, dict], finished_ids: set[int]) -> pd.DataFrame:
    """Rows per (element, finished fixture); double gameweeks split by explain-minutes share."""
    rows = []
    for gw, live in lives.items():
        for el in live["elements"]:
            st = el["stats"]
            exps = [(int(ex["fixture"]), {s["identifier"]: s["value"] for s in ex["stats"]})
                    for ex in el["explain"]]
            minutes = {fid: vals.get("minutes", 0) for fid, vals in exps}
            multi = len(minutes) > 1
            total = sum(minutes.values())
            for fid, vals in exps:
                if fid not in finished_ids:
                    continue
                row = {"fpl_element": int(el["id"]), "fixture_id": fid, "event": gw,
                       "multi": multi, "explain_minutes": vals.get("minutes", 0)}
                if multi:
                    share = minutes[fid] / total if total else 1 / len(minutes)
                    row |= {dst: vals.get(src, 0) for src, dst in SPLIT_STATS.items()}
                    row |= {dst: round(float(st[src]) * share, 2)
                            for dst, src in PROPORTIONAL.items()}
                    row |= {"bps": round(row["bps"]), "starts": np.nan}
                else:
                    row |= {dst: st[src] for src, dst in SPLIT_STATS.items()}
                    row |= {dst: st[src] for dst, src in PROPORTIONAL.items()}
                    row["starts"] = st["starts"]
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise StructureError("no player rows for finished fixtures in the event live data")
    if df["multi"].any():
        log.warning("%d rows are in double gameweeks: minutes/goals/assists split from explain; "
                    "%s approximated by minutes share; starts NaN", int(df["multi"].sum()),
                    list(PROPORTIONAL))
    mm = (~df["multi"]) & (df["minutes"] != df["explain_minutes"])
    log.info("explain rows: %d; minutes mismatches (stats vs explain): %d", len(df), int(mm.sum()))
    return df


def build_player_matches(season: str, lives: dict[int, dict], fixtures: list,
                         fx_done: pd.DataFrame, players: pd.DataFrame,
                         names: pd.Series) -> pd.DataFrame:
    df = explain_rows(lives, set(fx_done["id"]))
    pl = players.set_index("fpl_element")
    unknown = ~df["fpl_element"].isin(pl.index)
    if unknown.any():
        log.warning("dropping %d rows for elements not in bootstrap (or non-players)",
                    int(unknown.sum()))
        df = df.loc[~unknown]
    f = fx_done.set_index("id").reindex(df["fixture_id"]).set_index(df.index)
    boot_team = df["fpl_element"].map(pl["team_id"])
    sides = fixture_sides(fixtures)
    side = pd.Series([sides.get(k) for k in zip(df["fixture_id"], df["fpl_element"])],
                     index=df.index, dtype=object)
    side_team = f["team_h"].where(side == "h", f["team_a"].where(side == "a"))
    boot_ok = (boot_team == f["team_h"]) | (boot_team == f["team_a"])
    team_id = side_team.fillna(boot_team.where(boot_ok))
    moved = side_team.notna() & (side_team != boot_team)
    log.info("team from fixture bps side: %d rows (%d differ from current team, e.g. transfers); "
             "from current team: %d rows", int(side_team.notna().sum()), int(moved.sum()),
             int((side_team.isna() & boot_ok).sum()))
    ambiguous = team_id.isna()
    if ambiguous.any():
        log.warning("%s: dropping %d rows with ambiguous team (current team not in the fixture "
                    "and no bps entry); %d of them have minutes > 0", season,
                    int(ambiguous.sum()), int((df.loc[ambiguous, "minutes"] > 0).sum()))
    df, f = df.loc[~ambiguous], f.loc[~ambiguous]
    team_id = team_id.loc[~ambiguous].astype("int64")
    home = team_id == f["team_h"]
    bad_round = df["event"] != f["event"]
    if bad_round.any():
        log.warning("%d rows: event live gameweek differs from fixture event", int(bad_round.sum()))
    out = pd.DataFrame({
        "player_id": df["fpl_element"].map(pl["player_id"]),
        "fpl_element": df["fpl_element"],
        "player_name": df["fpl_element"].map(pl["player_name"]),
        "date": iso_utc(f["kickoff_time"]),
        "season": season,
        "matchweek": f["event"],
        "fixture_id": df["fixture_id"],
        "team": team_id.map(names),
        "opponent": f["team_a"].where(home, f["team_h"]).map(names),
        "home_away": np.where(home, "H", "A"),
        "position": df["fpl_element"].map(pl["position"]),
        "team_goals": f["team_h_score"].where(home, f["team_a_score"]),
        "opp_goals": f["team_a_score"].where(home, f["team_h_score"]),
    } | {c: df[c] for c in ["minutes", "goals", "assists", "xg", "xa", "threat", "creativity",
                            "influence", "ict_index", "bps", "starts"]})
    for col in PLAYER_INT + PLAYER_FLOAT:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    bad = out[PLAYER_INT].isna().any(axis=1)
    if bad.any():
        raise StructureError(f"{int(bad.sum())} player rows with missing integer fields")
    out = out.astype({c: "int64" for c in PLAYER_INT} | {c: "float64" for c in PLAYER_FLOAT})
    return out.sort_values(["date", "fixture_id", "player_id"], ignore_index=True)[PLAYER_COLUMNS]


def reconcile_goals(pm: pd.DataFrame, tm: pd.DataFrame, fixtures: list,
                    names: pd.Series) -> list[tuple[int, str]]:
    """Return (fixture_id, team) sides where player goals + opponent own goals != score."""
    own = {}
    for f in fixtures:
        for stat in f["stats"] or []:
            if stat.get("identifier") == "own_goals":
                for side, benefit in (("h", f["team_a"]), ("a", f["team_h"])):
                    key = (int(f["id"]), names[int(benefit)])
                    own[key] = own.get(key, 0) + sum(int(i["value"]) for i in stat.get(side, []))
    scored = pm.groupby(["fixture_id", "team"])["goals"].sum()
    t = tm.set_index(["fixture_id", "team"])
    total = scored.reindex(t.index, fill_value=0) + pd.Series(
        [own.get(k, 0) for k in t.index], index=t.index)
    bad = total != t["goals_for"]
    (log.warning if bad.any() else log.info)(
        "goal reconciliation (player goals + opponent own goals = team score): %d of %d team "
        "sides match", int((~bad).sum()), len(bad))
    if bad.any():
        log.warning("mismatched sides: %s", t.index[bad].tolist()[:10])
    return t.index[bad].tolist()


def normalize(season: str, boot: dict, fixtures: list, lives: dict[int, dict],
              names: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pure transform of API payloads -> (players, fixtures frame, player_matches, team_matches)."""
    players = build_players(boot, names)
    fx = fixtures_frame(fixtures)
    unknown = (set(fx["team_h"]) | set(fx["team_a"])) - set(names.index)
    if unknown:
        raise StructureError(f"fixtures reference unknown team ids {sorted(unknown)}")
    fx_done = finished_fixtures(fx, season)
    tm = build_team_matches(season, fx_done, names)
    pm = build_player_matches(season, lives, fixtures, fx_done, players, names)
    reconcile_goals(pm, tm, fixtures, names)
    return players, fx, pm, tm


def build_meta(season: str, boot: dict, fx: pd.DataFrame, snapshots: list[Path],
               base: str) -> dict:
    events = boot["events"]
    current = next((e for e in events if e["is_current"]), None)
    nxt = next((e for e in events if e["is_next"]), None)
    fetched = min(p.stat().st_mtime for p in snapshots)
    done = fx.loc[fx["finished"], "kickoff_time"]
    return {
        "season": season,
        "fetched_at": datetime.fromtimestamp(fetched, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "current_gw": current["id"] if current else None,
        "next_gw": nxt["id"] if nxt else None,
        "next_deadline": nxt["deadline_time"] if nxt else None,
        "finished_gws": [e["id"] for e in events if e["finished"]],
        "last_finished_kickoff": iso_utc(done).max() if len(done) else None,
        "source": f"Official Fantasy Premier League API ({base})",
    }


def fetch_all(cfg: dict, refresh: bool) -> tuple[dict, list, dict[int, dict], list[Path]]:
    up = cfg["upcoming"]
    base = up["api_base"].rstrip("/")
    live_dir = resolve(cfg["paths"]["live_dir"]) / up["season"]
    max_age = float(up["snapshot_max_age_hours"])
    dl = Downloader(cfg)
    snapshots = [live_dir / "bootstrap.json", live_dir / "fixtures.json"]
    stale = [p.name for p in snapshots if not p.exists() or p.stat().st_size == 0
             or time.time() - p.stat().st_mtime >= max_age * 3600]
    if stale and not refresh:
        log.info("snapshot(s) %s missing or older than %gh; refreshing both", stale, max_age)
    force = refresh or bool(stale)
    boot = dl.fetch(f"{base}/bootstrap-static/", snapshots[0], parse_bootstrap, force=force)
    fixtures = dl.fetch(f"{base}/fixtures/", snapshots[1], parse_fixtures, force=force)
    events = {int(e["id"]): e for e in boot["events"]}
    gws = sorted({int(f["event"]) for f in fixtures if f["finished"] and f["event"] is not None})
    missing = [gw for gw in gws if gw not in events]
    if missing:
        raise StructureError(f"fixtures reference gameweeks {missing} missing from bootstrap events")
    lives = {}
    for gw in gws:
        final = bool(events[gw]["finished"] and events[gw]["data_checked"])
        dest = live_dir / (f"event_{gw}_live.json" if final else f"event_{gw}_live.provisional.json")
        lives[gw] = dl.fetch(f"{base}/event/{gw}/live/", dest, parse_event,
                             max_age_hours=None if final else 0)
        if final:
            (live_dir / f"event_{gw}_live.provisional.json").unlink(missing_ok=True)
    log.info("network: %d downloaded, %d cached; finished gameweeks with data: %s",
             dl.downloaded, dl.cached, gws)
    return boot, fixtures, lives, snapshots


def run(cfg: dict, refresh: bool = False) -> dict:
    up = cfg["upcoming"]
    season = up["season"]
    if season in cfg["data"]["seasons"]:
        raise ValueError(f"{season} is a model season; the live season must be separate")
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    live_dir = resolve(cfg["paths"]["live_dir"]) / season
    boot, fixtures, lives, snapshots = fetch_all(cfg, refresh)

    names = team_names(boot, cfg)
    players, fx, pm, tm = normalize(season, boot, fixtures, lives, names)
    n_done = int(fx["finished"].sum())

    fixtures_out = pd.DataFrame({
        "fixture_id": fx["id"].astype("int64"),
        "gameweek": pd.to_numeric(fx["event"]).astype("Int64"),
        "date": iso_utc(fx["kickoff_time"]),
        "home_team": fx["team_h"].map(names),
        "away_team": fx["team_a"].map(names),
        "started": fx["started"],
        "finished": fx["finished"],
        "home_goals": pd.to_numeric(fx["team_h_score"]).astype("Int64"),
        "away_goals": pd.to_numeric(fx["team_a_score"]).astype("Int64"),
    }).sort_values(["date", "fixture_id"], ignore_index=True)
    meta = build_meta(season, boot, fx, snapshots, up["api_base"].rstrip("/"))

    pm.to_csv(raw_dir / f"player_matches_{season}.csv", index=False)
    tm.to_csv(raw_dir / f"team_matches_{season}.csv", index=False)
    players[PLAYERS_COLUMNS].to_csv(live_dir / "players.csv", index=False)
    fixtures_out.to_csv(live_dir / "fixtures.csv", index=False)
    (live_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("%s: %d player match rows (%d players, %d with minutes > 0), %d team match rows, "
             "%d finished fixtures, gameweeks %s, %d teams; %d players in snapshot",
             season, len(pm), pm["player_id"].nunique(), int((pm["minutes"] > 0).sum()),
             len(tm), n_done, sorted(pm["matchweek"].unique().tolist()),
             pm["team"].nunique(), len(players))
    log.info("meta: %s", meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the in-progress season from the FPL API.")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch bootstrap and fixtures snapshots even if they are fresh")
    args = parser.parse_args(argv)
    try:
        run(load_config(), refresh=args.refresh)
    except DownloadError as e:
        log.error("download failed: %s", e)
        return 1
    except StructureError as e:
        log.error("unexpected API structure: %s", e)
        return 1
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
