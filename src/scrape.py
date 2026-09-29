"""Download and normalize Premier League player match data.

Source: official Fantasy Premier League data, via the vaastav/Fantasy-Premier-League
GitHub archive of per-gameweek FPL API snapshots (cfg["data"]["base_url"]).
FBref is not scraped: it serves a Cloudflare interactive challenge to every request.
"""
import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from src.utils import get_logger, load_config, resolve

log = get_logger(__name__)

REQUIRED = {
    "merged_gw.csv": [
        "name", "position", "team", "element", "fixture", "kickoff_time", "round",
        "minutes", "goals_scored", "assists", "threat", "creativity", "influence",
        "ict_index", "bps", "opponent_team", "was_home", "team_h_score", "team_a_score",
    ],
    "players_raw.csv": ["id", "code"],
    "teams.csv": ["id", "name"],
    "fixtures.csv": [
        "id", "event", "kickoff_time", "team_h", "team_a",
        "team_h_score", "team_a_score", "finished",
    ],
}
PLAYER_COLUMNS = [
    "player_id", "fpl_element", "player_name", "date", "season", "matchweek", "fixture_id",
    "team", "opponent", "home_away", "position", "minutes", "goals", "assists", "xg", "xa",
    "threat", "creativity", "influence", "ict_index", "bps", "starts", "team_goals", "opp_goals",
]
PLAYER_INT = ["player_id", "fpl_element", "matchweek", "fixture_id", "minutes", "goals",
              "assists", "bps", "team_goals", "opp_goals"]
PLAYER_FLOAT = ["xg", "xa", "threat", "creativity", "influence", "ict_index", "starts"]
TEAM_COLUMNS = ["season", "fixture_id", "date", "team", "opponent", "home_away",
                "goals_for", "goals_against"]
LOG_COLUMNS = ["season", "player_match_rows", "team_match_rows", "fixtures", "unique_players",
               "downloaded_files", "cached_files", "status"]
POSITION_MAP = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
RETRYABLE = (requests.Timeout, requests.ConnectionError,
             requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError)
RETRY_AFTER_CAP_SECONDS = 120.0


class DownloadError(Exception):
    pass


class StructureError(Exception):
    pass


def normalize_season(value: str) -> str:
    m = re.fullmatch(r"(\d{4})[-/](\d{2}|\d{4})", value.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad season {value!r}; use 2021-22 or 2021-2022")
    start, end = int(m[1]), m[2]
    if int(end) != (start + 1 if len(end) == 4 else (start + 1) % 100):
        raise argparse.ArgumentTypeError(f"bad season {value!r}: years must be consecutive")
    return f"{start}-{(start + 1) % 100:02d}"


class Downloader:
    def __init__(self, cfg: dict):
        data = cfg["data"]
        self.delay = float(data["request_delay_seconds"])
        self.max_retries = int(data["max_retries"])
        self.timeout = float(data["timeout_seconds"])
        self.session = requests.Session()
        self.session.headers["User-Agent"] = data["user_agent"]
        self.last_request: float | None = None
        self.downloaded = 0
        self.cached = 0

    def _wait(self) -> None:
        if self.last_request is not None:
            remaining = self.delay - (time.monotonic() - self.last_request)
            if remaining > 0:
                time.sleep(remaining)

    def fetch(self, url: str, dest: Path, parse: Callable[[Path, str], Any],
              max_age_hours: float | None = None, force: bool = False) -> Any:
        """Return parse(dest, url), using the cached file unless it is stale, forced or invalid."""
        if dest.exists() and dest.stat().st_size > 0:
            age_hours = (time.time() - dest.stat().st_mtime) / 3600
            if force:
                log.info("refresh requested for %s", dest.name)
            elif max_age_hours is not None and age_hours >= max_age_hours:
                log.info("%s is %.1fh old (max %gh); re-fetching", dest.name, age_hours,
                         max_age_hours)
            else:
                try:
                    result = parse(dest, url)
                except StructureError as e:
                    bad = dest.with_name(dest.name + ".bad")
                    dest.replace(bad)
                    log.warning("cached %s is invalid (%s); moved to %s, re-downloading once",
                                dest, e, bad.name)
                else:
                    self.cached += 1
                    log.info("cache hit %s", dest)
                    return result
        self.download(url, dest)
        return parse(dest, url)

    def download(self, url: str, dest: Path) -> None:
        error = ""
        for attempt in range(self.max_retries + 1):
            self._wait()
            retry_after = None
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except RETRYABLE as e:
                error = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200 and resp.content:
                    self._write(dest, resp.content)
                    self.downloaded += 1
                    log.info("downloaded %s (%d bytes)", url, len(resp.content))
                    return
                if resp.status_code == 200:
                    error = "empty body (HTTP 200)"
                elif resp.status_code == 429 or resp.status_code >= 500:
                    error = f"HTTP {resp.status_code}"
                    retry_after = retry_after_seconds(resp.headers.get("Retry-After"))
                else:
                    raise DownloadError(f"HTTP {resp.status_code} for {url}")
            finally:
                self.last_request = time.monotonic()
            if attempt < self.max_retries:
                backoff = retry_after if retry_after is not None else self.delay * 2 ** attempt
                log.warning("%s for %s; retry %d/%d in %.1fs",
                            error, url, attempt + 1, self.max_retries, backoff)
                time.sleep(backoff)
        raise DownloadError(f"{error} for {url} after {self.max_retries} retries")

    @staticmethod
    def _write(dest: Path, content: bytes) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(content)
        tmp.replace(dest)


def retry_after_seconds(value: str | None) -> float | None:
    if value and value.strip().isdigit():
        return min(float(value), RETRY_AFTER_CAP_SECONDS)
    return None


def read_csv(path: Path, url: str) -> pd.DataFrame:
    """Parse and validate a raw file; raises StructureError naming the URL."""
    with open(path, "rb") as f:
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            raise StructureError(f"{url}: does not end with a newline (truncated?)")
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise StructureError(f"{url}: cannot parse: {e}") from e
    missing = [c for c in REQUIRED[path.name] if c not in df.columns]
    if missing:
        raise StructureError(f"{url}: missing columns {missing}")
    return df


def to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False,
                                                       "1": True, "0": False})


def iso_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="ISO8601").dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def drop_rows(df: pd.DataFrame, mask: pd.Series, season: str, reason: str,
              sample_col: str | None = None) -> pd.DataFrame:
    n = int(mask.sum())
    if n:
        sample = ""
        if sample_col:
            sample = f"; e.g. {sample_col}={sorted(df.loc[mask, sample_col].unique().tolist())[:10]}"
        log.warning("%s: dropping %d rows: %s%s", season, n, reason, sample)
    return df.loc[~mask]


def finished_fixtures(fixtures: pd.DataFrame, season: str) -> pd.DataFrame:
    fx = fixtures.copy()
    fx["finished"] = to_bool(fx["finished"]).fillna(False).astype(bool)
    for col in ["id", "event", "team_h", "team_a", "team_h_score", "team_a_score"]:
        fx[col] = pd.to_numeric(fx[col], errors="coerce")
    done = fx["finished"] & fx["team_h_score"].notna() & fx["team_a_score"].notna()
    if (~done).any():
        log.warning("%s: %d of %d fixtures not finished or missing scores",
                    season, int((~done).sum()), len(fx))
    return fx.loc[done].astype({c: "int64" for c in ["id", "event", "team_h", "team_a",
                                                      "team_h_score", "team_a_score"]})


def build_team_matches(season: str, fx: pd.DataFrame, team_names: pd.Series) -> pd.DataFrame:
    date = iso_utc(fx["kickoff_time"])
    sides = []
    for side, other, ha in [("h", "a", "H"), ("a", "h", "A")]:
        sides.append(pd.DataFrame({
            "season": season,
            "fixture_id": fx["id"],
            "date": date,
            "team": fx[f"team_{side}"].map(team_names),
            "opponent": fx[f"team_{other}"].map(team_names),
            "home_away": ha,
            "goals_for": fx[f"team_{side}_score"],
            "goals_against": fx[f"team_{other}_score"],
        }))
    out = pd.concat(sides, ignore_index=True)
    if out[["team", "opponent"]].isna().any().any():
        raise StructureError(f"{season}: fixtures reference team ids missing from teams.csv")
    return out.sort_values(["date", "fixture_id", "home_away"], ascending=[True, True, False],
                           ignore_index=True)[TEAM_COLUMNS]


def build_player_matches(season: str, gw: pd.DataFrame, players: pd.DataFrame,
                         team_names: pd.Series, fx: pd.DataFrame) -> pd.DataFrame:
    df = gw.copy()
    n_raw = len(df)
    for col in ["element", "fixture", "opponent_team", "team_h_score", "team_a_score", "round"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if players["id"].duplicated().any():
        raise StructureError(f"{season}: duplicate ids in players_raw.csv")
    df["player_id"] = df["element"].map(players.set_index("id")["code"])
    df = drop_rows(df, df["player_id"].isna(), season,
                   "element not found in players_raw id", "element")

    df["position_norm"] = df["position"].astype(str).str.strip().str.upper().map(POSITION_MAP)
    df = drop_rows(df, df["position_norm"].isna(), season,
                   "position not GK/DEF/MID/FWD (e.g. AM = assistant manager)", "position")

    known = set(team_names)
    bad_team = ~df["team"].isin(known)
    if bad_team.any():
        log.warning("%s: %d rows have merged_gw team not in teams.csv names: %s", season,
                    int(bad_team.sum()), sorted(df.loc[bad_team, "team"].astype(str).unique()))
    df["opponent"] = df["opponent_team"].map(team_names)
    df = drop_rows(df, df["opponent"].isna(), season,
                   "opponent_team id not in teams.csv", "opponent_team")

    df["was_home_b"] = to_bool(df["was_home"])
    df = drop_rows(df, df["was_home_b"].isna(), season, "unparseable was_home", "was_home")
    home = df["was_home_b"].astype(bool)

    fxi = fx.set_index("id")
    in_fx = df["fixture"].isin(fxi.index)
    df = drop_rows(df, ~in_fx, season, "fixture not among finished fixtures", "fixture")
    home = home.loc[df.index]
    f = fxi.reindex(df["fixture"]).set_index(df.index)

    fx_team = f["team_h"].where(home, f["team_a"]).map(team_names)
    fx_opp = f["team_a"].where(home, f["team_h"]).map(team_names)
    team_mm = df["team"] != fx_team
    opp_mm = df["opponent"] != fx_opp
    score_mm = (df["team_h_score"] != f["team_h_score"]) | (df["team_a_score"] != f["team_a_score"])
    round_mm = df["round"] != f["event"]
    kickoff_mm = iso_utc(df["kickoff_time"]) != iso_utc(f["kickoff_time"])
    log.info("%s: cross-check vs fixtures.csv: team mismatches=%d, opponent mismatches=%d, "
             "score mismatches=%d, round mismatches=%d, kickoff mismatches=%d "
             "(date is taken from fixtures.csv)", season,
             int(team_mm.sum()), int(opp_mm.sum()), int(score_mm.sum()),
             int(round_mm.sum()), int(kickoff_mm.sum()))
    if team_mm.any():
        pairs = (df.loc[team_mm, "team"].astype(str) + " -> " + fx_team[team_mm].astype(str))
        log.warning("%s: merged_gw team disagrees with fixture side (merged_gw -> fixture): %s",
                    season, pairs.value_counts().head(10).to_dict())

    th, ta = df["team_h_score"], df["team_a_score"]
    out = pd.DataFrame({
        "player_id": df["player_id"],
        "fpl_element": df["element"],
        "player_name": df["name"].astype(str).str.strip(),
        "date": iso_utc(f["kickoff_time"]),
        "season": season,
        "matchweek": df["round"],
        "fixture_id": df["fixture"],
        "team": df["team"].astype(str),
        "opponent": df["opponent"],
        "home_away": np.where(home, "H", "A"),
        "position": df["position_norm"],
        "minutes": df["minutes"],
        "goals": df["goals_scored"],
        "assists": df["assists"],
        "xg": df.get("expected_goals", np.nan),
        "xa": df.get("expected_assists", np.nan),
        "threat": df["threat"],
        "creativity": df["creativity"],
        "influence": df["influence"],
        "ict_index": df["ict_index"],
        "bps": df["bps"],
        "starts": df.get("starts", np.nan),
        "team_goals": th.where(home, ta),
        "opp_goals": ta.where(home, th),
    }, index=df.index)
    for col in PLAYER_INT + PLAYER_FLOAT:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["threat", "creativity", "influence", "ict_index"]:
        out = drop_rows(out, out[col].isna(), season, f"non-numeric {col}")
    for col in PLAYER_INT:
        out = drop_rows(out, out[col].isna(), season, f"missing or non-numeric {col}")
    out = out.astype({c: "int64" for c in PLAYER_INT} | {c: "float64" for c in PLAYER_FLOAT})
    log.info("%s: %d merged_gw rows -> %d player match rows", season, n_raw, len(out))
    return out.sort_values(["date", "fixture_id", "player_id"], ignore_index=True)[PLAYER_COLUMNS]


def sanity_check(season: str, fx: pd.DataFrame, n_teams: int, pm: pd.DataFrame) -> None:
    checks = {"finished fixtures": (len(fx), 380), "teams": (n_teams, 20),
              "fixture matchweeks": (fx["event"].nunique(), 38),
              "player-row matchweeks": (pm["matchweek"].nunique(), 38)}
    for name, (got, want) in checks.items():
        (log.info if got == want else log.warning)("%s: %s = %d (expected %d)",
                                                   season, name, got, want)
    empty = sorted(set(range(1, 39)) - set(fx["event"]))
    if empty:
        log.warning("%s: gameweeks with no fixtures (postponed/rescheduled): %s", season, empty)


def scrape_season(season: str, cfg: dict, dl: Downloader) -> dict:
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    base = cfg["data"]["base_url"].rstrip("/")
    row = dict.fromkeys(LOG_COLUMNS, 0) | {"season": season, "status": "ok"}
    dl0, cache0 = dl.downloaded, dl.cached
    try:
        frames = {}
        for rel in cfg["data"]["raw_files"]:
            url = f"{base}/{season}/{rel}"
            dest = raw_dir / "fpl" / season / Path(rel).name
            frames[dest.name] = dl.fetch(url, dest, read_csv)
        teams = frames["teams.csv"]
        team_names = teams.assign(id=pd.to_numeric(teams["id"])).set_index("id")["name"]
        fx = finished_fixtures(frames["fixtures.csv"], season)
        players = frames["players_raw.csv"].astype({"id": "int64", "code": "int64"})
        tm = build_team_matches(season, fx, team_names)
        pm = build_player_matches(season, frames["merged_gw.csv"], players, team_names, fx)
        sanity_check(season, fx, len(team_names), pm)
        pm.to_csv(raw_dir / f"player_matches_{season}.csv", index=False)
        tm.to_csv(raw_dir / f"team_matches_{season}.csv", index=False)
        row |= {"player_match_rows": len(pm), "team_match_rows": len(tm), "fixtures": len(fx),
                "unique_players": pm["player_id"].nunique()}
    except DownloadError as e:
        log.error("%s: download failed, skipping season: %s", season, e)
        row["status"] = "download_failed"
    except StructureError as e:
        log.error("%s: unexpected structure, skipping season: %s", season, e)
        row["status"] = "invalid_structure"
    except Exception:
        log.exception("%s: unexpected error, skipping season", season)
        row["status"] = "error"
    row["downloaded_files"] = dl.downloaded - dl0
    row["cached_files"] = dl.cached - cache0
    return row


def write_log(rows: list[dict], path: Path) -> pd.DataFrame:
    new = pd.DataFrame(rows, columns=LOG_COLUMNS)
    if path.exists():
        old = pd.read_csv(path, dtype={"season": str})
        new = pd.concat([old[~old["season"].isin(new["season"])], new], ignore_index=True)
    new = new.sort_values("season", ignore_index=True)[LOG_COLUMNS]
    new.to_csv(path, index=False)
    return new


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Download and normalize FPL match data.")
    parser.add_argument("--seasons", nargs="+", type=normalize_season,
                        default=cfg["data"]["seasons"], help="e.g. 2021-22 or 2021-2022")
    args = parser.parse_args(argv)
    seasons = list(dict.fromkeys(args.seasons))
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    dl = Downloader(cfg)
    rows = []
    for season in seasons:
        log.info("=== season %s ===", season)
        rows.append(scrape_season(season, cfg, dl))
    summary = write_log(rows, raw_dir / "scrape_log.csv")
    log.info("run done: %d downloaded, %d cached\n%s", dl.downloaded, dl.cached,
             summary.to_string(index=False))
    failed = [r["season"] for r in rows if r["status"] != "ok"]
    if failed:
        log.error("seasons not scraped: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
