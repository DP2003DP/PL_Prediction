"""Shared fixtures: real config plus a small synthetic league matching the clean/team_matches schema."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

N_MATCHWEEKS = 10
XG_FIRST_SEASON_IDX = 2  # FPL xG/xA exist from 2022-23 onward
OLD_TEAMS = ["Ashford", "Brookvale", "Carrow", "Dunmore"]
NEW_TEAMS = ["Ashford", "Brookvale", "Carrow", "Elmstead"]  # Dunmore relegated, Elmstead promoted
PAIRINGS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
LAST = (99, 99)

# spells: (team, (first_season_idx, first_mw), (last_season_idx, last_mw)), inclusive
PLAYERS = [
    dict(pid=101, name="Alan Archer", pos="FWD", play=0.92, sub=0.08, spells=[("Ashford", (0, 0), LAST)]),
    dict(pid=102, name="Ben Barker", pos="DEF", play=0.85, sub=0.05, spells=[("Ashford", (0, 0), LAST)]),
    dict(pid=201, name="Carl Cole", pos="MID", play=0.88, sub=0.10, spells=[("Brookvale", (0, 0), LAST)]),
    dict(pid=202, name="Dan Drake", pos="FWD", play=0.85, sub=0.10, spells=[("Brookvale", (0, 0), LAST)]),
    dict(pid=203, name="Sid Short", pos="FWD", play=0.80, sub=0.60, spells=[("Brookvale", (0, 0), LAST)]),
    dict(pid=301, name="Tom Travers", pos="MID", play=0.90, sub=0.10,
         spells=[("Carrow", (0, 0), (1, 4)), ("Ashford", (1, 5), LAST)]),
    dict(pid=302, name="Eli Evans", pos="DEF", play=0.85, sub=0.05, spells=[("Carrow", (0, 0), LAST)]),
    dict(pid=303, name="Carl Cole", pos="MID", play=0.80, sub=0.15, spells=[("Carrow", (0, 0), LAST)]),
    dict(pid=401, name="Finn Ford", pos="FWD", play=0.88, sub=0.10, spells=[("Dunmore", (0, 0), (1, 9))]),
    dict(pid=402, name="Gil Grant", pos="DEF", play=0.85, sub=0.05, spells=[("Dunmore", (0, 0), (1, 9))]),
    dict(pid=501, name="Gus Green", pos="MID", play=0.88, sub=0.10, spells=[("Elmstead", (2, 0), LAST)]),
    dict(pid=502, name="Hal Hunt", pos="FWD", play=0.85, sub=0.15, spells=[("Elmstead", (3, 0), LAST)]),
]
GOAL_RATE = {"FWD": 0.50, "MID": 0.25, "DEF": 0.06}
ASSIST_RATE = {"FWD": 0.20, "MID": 0.25, "DEF": 0.08}
THREAT_SCALE = {"FWD": 15.0, "MID": 9.0, "DEF": 4.0}
CREATIVITY_SCALE = {"FWD": 8.0, "MID": 14.0, "DEF": 6.0}

META = {
    "transfer_player": 301,
    "transfer_from": "Carrow",
    "transfer_to": "Ashford",
    "promoted_team": "Elmstead",
    "debut_player": 501,
    "sub_player": 203,
    "xg_first_season_idx": XG_FIRST_SEASON_IDX,
}


def _season_start(season: str) -> datetime:
    start = datetime(int(season[:4]), 9, 10, tzinfo=timezone.utc)
    return start + timedelta(days=(5 - start.weekday()) % 7)  # first Saturday on/after 10 Sep


def _kickoff(season: str, mw: int, slot: int) -> datetime:
    base = _season_start(season) + timedelta(weeks=mw)
    if slot == 0:
        return base + (timedelta(hours=14) if mw % 2 == 0 else timedelta(hours=11, minutes=30))
    return base + (timedelta(days=1, hours=16, minutes=30) if mw % 2 == 0 else timedelta(days=2, hours=19))


def _team_of(player: dict, s_idx: int, mw: int) -> str | None:
    for team, first, last in player["spells"]:
        if first <= (s_idx, mw) <= last:
            return team
    return None


def _player_row(rng, p, s_idx, season, mw, fixture_id, date, team, opp, ha, first_ever):
    pos = p["pos"]
    if first_ever:
        minutes = 90
    elif rng.random() < p["sub"]:
        minutes = int(rng.integers(1, 30))
    else:
        minutes = 90 if rng.random() < 0.6 else int(rng.integers(30, 90))
    share = minutes / 90
    goals = int(rng.poisson(GOAL_RATE[pos] * share))
    assists = int(rng.poisson(ASSIST_RATE[pos] * share))
    threat = round(float(rng.gamma(2.0, THREAT_SCALE[pos])) * share + 20 * goals, 1)
    creativity = round(float(rng.gamma(2.0, CREATIVITY_SCALE[pos])) * share + 15 * assists, 1)
    influence = round(float(rng.gamma(2.0, 8.0)) * share + 25 * goals, 1)
    has_xg = s_idx >= XG_FIRST_SEASON_IDX
    xg = round(float(rng.gamma(1.5, GOAL_RATE[pos] / 1.5)) * share + 0.3 * goals, 2) if has_xg else np.nan
    xa = round(float(rng.gamma(1.5, ASSIST_RATE[pos] / 1.5)) * share + 0.3 * assists, 2) if has_xg else np.nan
    return {
        "player_id": p["pid"], "fpl_element": (s_idx + 1) * 1000 + p["pid"] % 97,
        "player_name": p["name"], "date": date, "season": season, "matchweek": mw + 1,
        "fixture_id": fixture_id, "team": team, "opponent": opp, "home_away": ha, "position": pos,
        "minutes": minutes, "goals": goals, "assists": assists, "xg": xg, "xa": xa,
        "threat": threat, "creativity": creativity, "influence": influence,
        "ict_index": round((threat + creativity + influence) / 10, 1),
        "bps": int(rng.integers(0, 30)) + 12 * goals + 9 * assists,
        "starts": float(minutes >= 60) if has_xg else np.nan,
    }


def make_synthetic(seasons: list[str], min_minutes: int, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean, team_matches) frames shaped like data/processed/*.parquet."""
    rng = np.random.default_rng(seed)
    rows, fixtures, seen = [], [], set()
    for s_idx, season in enumerate(seasons):
        teams = OLD_TEAMS if s_idx < XG_FIRST_SEASON_IDX else NEW_TEAMS
        for mw in range(N_MATCHWEEKS):
            flip = (mw // len(PAIRINGS)) % 2 == 1
            for slot, (a, b) in enumerate(PAIRINGS[mw % len(PAIRINGS)]):
                home, away = (teams[b], teams[a]) if flip else (teams[a], teams[b])
                fixture_id, date = mw * 2 + slot + 1, _kickoff(season, mw, slot)
                goals, fixture_rows = {}, []
                for team, opp, ha in ((home, away, "H"), (away, home, "A")):
                    team_rows = []
                    for p in PLAYERS:
                        if _team_of(p, s_idx, mw) != team:
                            continue
                        first_ever = p["pid"] not in seen
                        if not first_ever and rng.random() > p["play"]:
                            continue
                        seen.add(p["pid"])
                        team_rows.append(_player_row(rng, p, s_idx, season, mw, fixture_id, date,
                                                     team, opp, ha, first_ever))
                    goals[team] = sum(r["goals"] for r in team_rows) + int(rng.poisson(0.9))
                    fixture_rows.extend(team_rows)
                for r in fixture_rows:
                    r["team_goals"], r["opp_goals"] = goals[r["team"]], goals[r["opponent"]]
                rows.extend(fixture_rows)
                for team, opp, ha in ((home, away, "H"), (away, home, "A")):
                    fixtures.append({"season": season, "fixture_id": fixture_id, "date": date, "team": team,
                                     "opponent": opp, "home_away": ha,
                                     "goals_for": goals[team], "goals_against": goals[opp]})
    clean = pd.DataFrame(rows)
    clean["date"] = pd.to_datetime(clean["date"], utc=True)
    clean["eligible"] = clean["minutes"] >= min_minutes
    clean = clean.sort_values(["player_id", "date"]).reset_index(drop=True)
    team_matches = pd.DataFrame(fixtures)
    team_matches["date"] = pd.to_datetime(team_matches["date"], utc=True)
    team_matches = team_matches.sort_values(["team", "date"]).reset_index(drop=True)
    return clean, team_matches


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def synthetic_meta() -> dict:
    return dict(META)


@pytest.fixture(scope="session")
def make_league():
    return make_synthetic


@pytest.fixture(scope="session")
def _synthetic(cfg):
    return make_synthetic(cfg["data"]["seasons"], cfg["cleaning"]["min_minutes"])


@pytest.fixture
def clean(_synthetic) -> pd.DataFrame:
    return _synthetic[0].copy(deep=True)


@pytest.fixture
def team_matches(_synthetic) -> pd.DataFrame:
    return _synthetic[1].copy(deep=True)


@pytest.fixture
def split_frame(_synthetic, cfg) -> pd.DataFrame:
    """Feature-table-like frame: config seasons plus one season before and one after, shuffled,
    with a non-positional integer index."""
    base = _synthetic[0]
    df = base.loc[base["eligible"]].copy()
    df["y"] = (df["goals"] + df["assists"] >= 1).astype(int)
    seasons = cfg["data"]["seasons"]
    before = df[df["season"] == seasons[0]].assign(season=_shift_season(seasons[0], -1))
    before["date"] = before["date"] - pd.Timedelta(days=364)
    after = df[df["season"] == seasons[-1]].head(25).assign(season=_shift_season(seasons[-1], 1))
    after["date"] = after["date"] + pd.Timedelta(days=364)
    df = pd.concat([before, df, after], ignore_index=True)
    rng = np.random.default_rng(3)
    df = df.iloc[rng.permutation(len(df))]
    df.index = pd.Index(np.arange(len(df)) * 7 + 1000)
    return df


def _shift_season(season: str, years: int) -> str:
    start = int(season[:4]) + years
    return f"{start}-{(start + 1) % 100:02d}"
