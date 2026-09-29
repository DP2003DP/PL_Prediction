"""Upcoming-fixture features (CONTRACT_UPCOMING.md stage 4): candidate selection, history-only features,
placeholder invariance, and double-gameweek independence. Synthetic data only, except the last test."""
import copy
import io
import json

import numpy as np
import pandas as pd
import pytest

import src.features as features_mod
import src.upcoming as upcoming_mod
from src.upcoming import build_upcoming_features, select_candidates, select_target_fixtures
from src.utils import resolve

REAL_BUILD = features_mod.build_features
KEY = ["player_id", "fixture_id"]
CANDIDATE_COLS = ["likely_starter", "status", "chance_of_playing_next_round", "news", "prior_matches"]
PLACEHOLDER_ID_COLS = ["player_id", "player_name", "date", "season", "matchweek", "fixture_id", "team",
                       "opponent", "home_away", "position"]
PLAYERS = [  # player_id, name, current team, position, status, chance_of_playing_next_round, news
    (101, "Alan Archer", "Ashford", "FWD", "a", np.nan, ""),
    (102, "Ben Barker", "Ashford", "DEF", "i", 0.0, "Hamstring injury - unknown return date"),
    (301, "Tom Travers", "Ashford", "MID", "a", np.nan, ""),
    (901, "Gary Gloves", "Ashford", "GK", "a", np.nan, ""),
    (201, "Carl Cole", "Brookvale", "MID", "d", 75.0, "Knock - 75% chance of playing"),
    (202, "Dan Drake", "Brookvale", "FWD", "s", 0.0, "Suspended until 25 Oct"),
    (203, "Sid Short", "Brookvale", "FWD", "a", np.nan, ""),
    (302, "Eli Evans", "Brookvale", "DEF", "a", np.nan, "Joined from Carrow"),
    (303, "Carl Cole", "Carrow", "MID", "u", 0.0, "Joined Dunmore on loan"),
    (601, "Nia Newman", "Carrow", "MID", "a", np.nan, ""),
    (902, "Kit Keeper", "Carrow", "GK", "d", 50.0, "Illness - 50% chance of playing"),
    (402, "Gil Grant", "Elmstead", "DEF", "a", np.nan, ""),
    (501, "Gus Green", "Elmstead", "MID", "a", np.nan, ""),
    (502, "Hal Hunt", "Elmstead", "FWD", "n", np.nan, ""),
]
# likely_starter traps under the default config: 101 has exactly min_minutes in one of Ashford's last 3 games;
# 203 has an appearance under min_minutes; 301 missed Ashford's last 2 games; 302 just moved from Carrow.
TRAPS = {101: True, 203: False, 301: False, 302: False}
SCENARIOS = {  # fixture ids per scenario (see _fixtures)
    "one_fixture": [51],
    "gw": [51, 52],
    "dgw": [51, 52, 53],  # 53 is a midweek Carrow v Ashford: both teams play twice
}
DGW_CASES = {  # fixtures, team whose players are all unavailable, DGW players, twice-met opponent and its fixtures
    "player_and_opponent": ([51, 52, 53], None, (101, 301, 601), "Ashford", {51, 53}),
    "opponent_only": ([52, 54], "Elmstead", (), "Elmstead", {52, 54}),  # Elmstead has no candidates of its own
}


def _fixtures(cutoff: pd.Timestamp) -> pd.DataFrame:
    base = cutoff.normalize() + pd.Timedelta(days=6)
    spec = [(51, 6, "0 days 14:00:00", "Ashford", "Brookvale"), (52, 6, "1 days 16:30:00", "Carrow", "Elmstead"),
            (53, 6, "4 days 19:45:00", "Carrow", "Ashford"), (54, 6, "3 days 19:45:00", "Elmstead", "Brookvale"),
            (61, 7, "7 days 14:00:00", "Brookvale", "Carrow"), (62, 7, "8 days 16:30:00", "Elmstead", "Ashford")]
    return pd.DataFrame([{"fixture_id": fid, "gameweek": gw, "date": base + pd.Timedelta(offset),
                          "home_team": home, "away_team": away, "started": False, "finished": False,
                          "home_goals": np.nan, "away_goals": np.nan} for fid, gw, offset, home, away in spec])


@pytest.fixture(scope="module")
def league(make_league, cfg):
    """History = model seasons + current season GW1-5, with likely_starter traps; plus players and fixtures."""
    season, mm = cfg["upcoming"]["season"], cfg["cleaning"]["min_minutes"]
    clean, tm = make_league(cfg["data"]["seasons"] + [season], mm)
    cutoff = clean.loc[(clean["season"] == season) & (clean["matchweek"] == 5), "date"].max()
    clean, tm = clean[clean["date"] <= cutoff].copy(), tm[tm["date"] <= cutoff].reset_index(drop=True)
    last3 = tm[tm["team"] == "Ashford"].sort_values("date").tail(3)
    in_last3 = clean["season"].eq(season) & clean["fixture_id"].isin(last3["fixture_id"])
    clean = clean[~(in_last3 & (clean["player_id"] == 301) & ~clean["fixture_id"].eq(last3["fixture_id"].iloc[0]))]
    first_101 = clean.index[in_last3.reindex(clean.index) & (clean["player_id"] == 101)][0]
    clean.loc[first_101, "minutes"] = mm
    clean["eligible"] = clean["minutes"] >= mm
    players = pd.DataFrame(PLAYERS, columns=["player_id", "player_name", "team", "position", "status",
                                             "chance_of_playing_next_round", "news"])
    players.insert(1, "fpl_element", np.arange(len(players)) + 1)
    return clean.reset_index(drop=True), tm, players, _fixtures(cutoff)


def _pick(fixtures: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    return fixtures[fixtures["fixture_id"].isin(ids)].reset_index(drop=True)


def _expected_candidates(players, fixtures, hist_clean, hist_tm, cfg) -> pd.DataFrame:
    """Independent implementation of the select_candidates definition, plus the placeholder id columns."""
    up, mm = cfg["upcoming"], cfg["cleaning"]["min_minutes"]
    keep = players[(players["position"] != "GK") & ~players["status"].isin(up["exclude_status"])]
    rows = []
    for p in keep.itertuples(index=False):
        last = hist_tm[hist_tm["team"] == p.team].sort_values("date").tail(up["likely_starter_lookback"])
        mine = hist_clean[(hist_clean["player_id"] == p.player_id) & (hist_clean["team"] == p.team)]
        played = mine.merge(last[["season", "fixture_id"]], on=["season", "fixture_id"])
        for f in fixtures.itertuples(index=False):
            if p.team not in (f.home_team, f.away_team):
                continue
            home = p.team == f.home_team
            rows.append({"player_id": p.player_id, "player_name": p.player_name, "date": f.date,
                         "season": up["season"], "matchweek": f.gameweek, "fixture_id": f.fixture_id,
                         "team": p.team, "opponent": f.away_team if home else f.home_team,
                         "home_away": "H" if home else "A", "position": p.position,
                         "likely_starter": int((played["minutes"] >= mm).sum()) >= up["likely_starter_min_matches"],
                         "status": p.status, "chance_of_playing_next_round": p.chance_of_playing_next_round,
                         "news": p.news, "prior_matches": int((hist_clean["player_id"] == p.player_id).sum())})
    return pd.DataFrame(rows).set_index(KEY).sort_index()


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in KEY if c not in df.columns]
    assert not missing, f"missing key columns {missing}"
    out = df.set_index(KEY).sort_index()
    assert out.index.is_unique, "duplicate (player_id, fixture_id) rows"
    return out


def _played(rows: pd.DataFrame, columns, rng) -> pd.DataFrame:
    """Clean rows for these candidates as if their fixture had been played with random stats."""
    n, add = len(rows), rows.reset_index()[PLACEHOLDER_ID_COLS].copy()
    add["fpl_element"] = 0
    for col, (lo, hi) in {"minutes": (30, 91), "goals": (0, 4), "assists": (0, 3), "bps": (0, 60),
                          "team_goals": (0, 6), "opp_goals": (0, 6)}.items():
        add[col] = rng.integers(lo, hi, n)
    for col, hi in {"threat": 150, "creativity": 150, "influence": 150, "ict_index": 30, "xg": 2, "xa": 2,
                    "starts": 1}.items():
        add[col] = rng.uniform(0, hi, n).round(2)
    add["eligible"] = True
    return add[list(columns)]


def _played_tm(f, season: str, rng) -> pd.DataFrame:
    h, a = (int(x) for x in rng.integers(0, 6, 2))
    base = {"season": season, "fixture_id": f.fixture_id, "date": f.date}
    return pd.DataFrame([{**base, "team": f.home_team, "opponent": f.away_team, "home_away": "H",
                          "goals_for": h, "goals_against": a},
                         {**base, "team": f.away_team, "opponent": f.home_team, "home_away": "A",
                          "goals_for": a, "goals_against": h}])


def _history_only(hist_clean, hist_tm, fixtures, expected, cfg, seed) -> pd.DataFrame:
    """Features build_features gives each fixture when it alone is appended to history as played."""
    rng, season, parts = np.random.default_rng(seed), cfg["upcoming"]["season"], []
    for f in fixtures.itertuples(index=False):
        rows = expected[expected.index.get_level_values("fixture_id") == f.fixture_id]
        if rows.empty:
            continue
        c = pd.concat([hist_clean, _played(rows, hist_clean.columns, rng)], ignore_index=True)
        tm = pd.concat([hist_tm, _played_tm(f, season, rng)], ignore_index=True)
        out = REAL_BUILD(c, tm, cfg)
        parts.append(out[(out["season"] == season) & (out["fixture_id"] == f.fixture_id)])
    return _keyed(pd.concat(parts, ignore_index=True))


def _assert_features_equal(actual: pd.DataFrame, expected: pd.DataFrame, cols: list[str]) -> None:
    assert actual.index.equals(expected.index), "different (player_id, fixture_id) rows"
    a = actual[cols].to_numpy(dtype="float64", na_value=np.nan)
    e = expected[cols].to_numpy(dtype="float64", na_value=np.nan)
    diff = ~np.isclose(a, e, rtol=1e-9, atol=1e-12, equal_nan=True)
    bad = [(actual.index[i], cols[j], a[i, j], e[i, j]) for i, j in zip(*np.nonzero(diff))]
    assert not bad, f"{len(bad)} feature values differ; first: {bad[:6]}"


def _run(league, cfg, ids):
    hist_clean, hist_tm, players, fixtures = league
    fx = _pick(fixtures, ids)
    cand = select_candidates(players, fx, hist_clean, hist_tm, cfg)
    return fx, cand, build_upcoming_features(hist_clean, hist_tm, cand, cfg)


class _BuildSpy:
    """Stands in for build_features: records placeholder rows per call and optionally randomizes them."""

    def __init__(self, hist_clean, hist_tm, seed=None):
        dates = hist_clean["date"].dt.as_unit("ns")
        self.hist = set(zip(hist_clean["player_id"], dates))
        self.hist_tm = set(zip(hist_tm["season"], hist_tm["fixture_id"], hist_tm["team"]))
        self.rng = None if seed is None else np.random.default_rng(seed)
        self.calls = []

    def __call__(self, clean, team_matches, cfg):
        c, tm = clean.copy(deep=True), team_matches.copy(deep=True)
        dates = pd.to_datetime(c["date"], utc=True).dt.as_unit("ns")
        new = np.array([k not in self.hist for k in zip(c["player_id"], dates)], dtype=bool)
        tnew = np.array([k not in self.hist_tm for k in zip(tm["season"], tm["fixture_id"], tm["team"])], dtype=bool)
        self.calls.append((c.loc[new, ["player_id", "fixture_id"]].copy(), tm.loc[tnew, ["team", "fixture_id"]].copy()))
        if self.rng is not None:
            n, tn = int(new.sum()), int(tnew.sum())
            for col, vals in {"minutes": self.rng.integers(30, 91, n), "goals": self.rng.integers(0, 5, n),
                              "assists": self.rng.integers(0, 5, n), "threat": self.rng.uniform(0, 150, n),
                              "creativity": self.rng.uniform(0, 150, n), "xg": self.rng.uniform(0, 2, n),
                              "xa": self.rng.uniform(0, 2, n), "team_goals": self.rng.integers(0, 6, n),
                              "opp_goals": self.rng.integers(0, 6, n)}.items():
                _assign(c, new, col, vals)
            for col in ("goals_for", "goals_against"):
                _assign(tm, tnew, col, self.rng.integers(0, 7, tn))
        return REAL_BUILD(c, tm, cfg)


def _assign(df: pd.DataFrame, mask: np.ndarray, col: str, values: np.ndarray) -> None:
    if col in df.columns and mask.any():
        if pd.api.types.is_integer_dtype(df[col].dtype):
            values = np.round(values).astype("int64")
        df.loc[mask, col] = values


def _install(monkeypatch, spy) -> None:
    monkeypatch.setattr(features_mod, "build_features", spy)
    for name, obj in list(vars(upcoming_mod).items()):
        if obj is REAL_BUILD:
            monkeypatch.setattr(upcoming_mod, name, spy)


# --- 4. candidate selection -----------------------------------------------------------------------

def _cfg_variant(cfg: dict, variant: str) -> dict:
    new = copy.deepcopy(cfg)
    if variant == "lookback5_min4_exclude_d":
        new["upcoming"].update(likely_starter_lookback=5, likely_starter_min_matches=4, exclude_status=["d"])
    elif variant == "lookback1_min1_exclude_none":
        new["upcoming"].update(likely_starter_lookback=1, likely_starter_min_matches=1, exclude_status=[])
    return new


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("variant", ["default", "lookback5_min4_exclude_d", "lookback1_min1_exclude_none"])
def test_select_candidates_follows_definition(league, cfg, scenario, variant):
    hist_clean, hist_tm, players, fixtures = league
    vcfg = _cfg_variant(cfg, variant)
    fx = _pick(fixtures, SCENARIOS[scenario])
    got = _keyed(select_candidates(players, fx, hist_clean, hist_tm, vcfg))
    exp = _expected_candidates(players, fx, hist_clean, hist_tm, vcfg)
    assert list(got.index) == list(exp.index), "wrong (player, fixture) candidate rows"
    missing = [c for c in CANDIDATE_COLS if c not in got.columns]
    assert not missing, f"select_candidates is missing {missing}"
    assert (got["likely_starter"].astype(bool) == exp["likely_starter"]).all(), (
        f"likely_starter differs for {list(got.index[got['likely_starter'].astype(bool) != exp['likely_starter']])}")
    assert (got["prior_matches"].astype(int) == exp["prior_matches"]).all()
    assert (got["status"].astype(str) == exp["status"]).all()
    assert np.allclose(got["chance_of_playing_next_round"].astype(float), exp["chance_of_playing_next_round"],
                       equal_nan=True)
    assert (got["news"].fillna("").astype(str) == exp["news"]).all()


def test_candidates_drop_gk_and_excluded_statuses_and_blank_teams(league, cfg):
    hist_clean, hist_tm, players, fixtures = league
    excluded = cfg["upcoming"]["exclude_status"]
    assert set(excluded) <= set(players["status"]), "precondition: every excluded status is represented"
    got = _keyed(select_candidates(players, _pick(fixtures, [51]), hist_clean, hist_tm, cfg))
    ids = set(got.index.get_level_values("player_id"))
    by_id = players.set_index("player_id")
    assert not ids & set(by_id.index[by_id["position"] == "GK"]), "goalkeepers must be dropped"
    assert not ids & set(by_id.index[by_id["status"].isin(excluded)]), "excluded statuses must be dropped"
    assert not ids & set(by_id.index[by_id["team"].isin(["Carrow", "Elmstead"])]), "blank-GW teams have no rows"
    assert ids == {101, 301, 201, 203, 302}


def test_likely_starter_traps(league, cfg):
    hist_clean, hist_tm, players, fixtures = league
    fx = _pick(fixtures, [51])
    exp = _expected_candidates(players, fx, hist_clean, hist_tm, cfg)
    got = _keyed(select_candidates(players, fx, hist_clean, hist_tm, cfg))
    mm = cfg["cleaning"]["min_minutes"]
    for pid, want in TRAPS.items():
        assert bool(exp.loc[(pid, 51), "likely_starter"]) == want, f"precondition: trap {pid} is not set up"
        assert bool(got.loc[(pid, 51), "likely_starter"]) == want, f"likely_starter wrong for trap player {pid}"
    own_last = hist_clean[hist_clean["player_id"].isin([301, 302])].sort_values("date").groupby("player_id").tail(3)
    assert (own_last.groupby("player_id")["minutes"].apply(lambda m: (m >= mm).sum()) >= 2).all(), (
        "precondition: 301/302 look like starters by their own last appearances")


# --- stage 2 output shape -------------------------------------------------------------------------

def test_upcoming_output_columns_and_id_values(league, cfg):
    hist_clean, hist_tm, players, _ = league
    fx, cand, out = _run(league, cfg, SCENARIOS["dgw"])
    f = cfg["features"]
    id_cols = [c for c in f["id_columns"] if c not in ("minutes", "goals", "assists")]
    missing = [c for c in id_cols + f["feature_columns"] + CANDIDATE_COLS if c not in out.columns]
    assert not missing, f"build_upcoming_features is missing {missing}"
    leaked = [c for c in ("minutes", "goals", "assists", f["target"]) if c in out.columns]
    assert not leaked, f"placeholder values must not be returned: {leaked}"
    got, exp = _keyed(out), _expected_candidates(players, fx, hist_clean, hist_tm, cfg)
    assert list(got.index) == list(exp.index) == list(_keyed(cand).index)
    for col in ("player_name", "season", "team", "opponent", "home_away", "position"):
        assert (got[col].astype(str) == exp[col].astype(str)).all(), f"{col} differs"
    assert (got["matchweek"].astype(int) == exp["matchweek"]).all()
    assert (pd.to_datetime(got["date"], utc=True) == pd.to_datetime(exp["date"], utc=True)).all()
    assert (got["likely_starter"].astype(bool) == exp["likely_starter"]).all()


# --- 1. consistency with a later-played fixture ---------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_features_equal_build_features_of_the_played_fixture(league, cfg, seed):
    hist_clean, hist_tm, players, _ = league
    fx, _, out = _run(league, cfg, SCENARIOS["gw"])
    exp = _expected_candidates(players, fx, hist_clean, hist_tm, cfg)
    ref = _history_only(hist_clean, hist_tm, fx, exp, cfg, seed)
    _assert_features_equal(_keyed(out), ref, cfg["features"]["feature_columns"])


# --- 2. placeholder invariance and extra future matches -------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_placeholder_values_do_not_matter(league, cfg, monkeypatch, seed):
    hist_clean, hist_tm, _, _ = league
    _, _, plain = _run(league, cfg, SCENARIOS["dgw"])
    spy = _BuildSpy(hist_clean, hist_tm, seed=seed)
    _install(monkeypatch, spy)
    _, _, perturbed = _run(league, cfg, SCENARIOS["dgw"])
    assert spy.calls, "build_upcoming_features did not call src.features.build_features"
    pd.testing.assert_frame_equal(_keyed(perturbed), _keyed(plain), check_like=True)


def test_extra_future_gameweek_does_not_change_features(league, cfg):
    hist_clean, hist_tm, players, fixtures = league
    fx6, fx7 = _pick(fixtures, [51, 52]), _pick(fixtures, [61, 62])
    cand = pd.concat([select_candidates(players, fx6, hist_clean, hist_tm, cfg),
                      select_candidates(players, fx7, hist_clean, hist_tm, cfg)], ignore_index=True)
    both = _keyed(build_upcoming_features(hist_clean, hist_tm, cand, cfg))
    _, _, only6 = _run(league, cfg, [51, 52])
    in6 = both.index.get_level_values("fixture_id").isin([51, 52])
    pd.testing.assert_frame_equal(both[in6], _keyed(only6), check_like=True)
    exp7 = _expected_candidates(players, fx7, hist_clean, hist_tm, cfg)
    _assert_features_equal(both[~in6], _history_only(hist_clean, hist_tm, fx7, exp7, cfg, 3),
                           cfg["features"]["feature_columns"])


# --- 3. double gameweek -----------------------------------------------------------------------------

@pytest.mark.parametrize("case", list(DGW_CASES))
def test_double_gameweek_fixtures_are_history_only(league, cfg, monkeypatch, case):
    ids, unavailable, dgw_players, opponent, opp_fixtures = DGW_CASES[case]
    hist_clean, hist_tm, players, fixtures = league
    if unavailable:
        players = players.assign(status=players["status"].where(players["team"] != unavailable, "i"))
    spy = _BuildSpy(hist_clean, hist_tm)
    _install(monkeypatch, spy)
    fx = _pick(fixtures, ids)
    got = _keyed(build_upcoming_features(hist_clean, hist_tm, select_candidates(players, fx, hist_clean, hist_tm, cfg),
                                         cfg))
    exp = _expected_candidates(players, fx, hist_clean, hist_tm, cfg)
    _assert_features_equal(got, _history_only(hist_clean, hist_tm, fx, exp, cfg, 5), cfg["features"]["feature_columns"])
    for placeholders, _ in spy.calls:
        dup = placeholders["player_id"][placeholders["player_id"].duplicated()].tolist()
        assert not dup, f"players {dup} had two placeholder rows in one build_features call"
    for pid in dgw_players:  # player DGW: both fixtures see exactly the real history
        assert list(got.loc[pid, "prior_matches"].astype(int)) == [int((hist_clean["player_id"] == pid).sum())] * 2
    conceded = hist_tm[hist_tm["team"] == opponent].sort_values("date")["goals_against"]
    vs_opp = got[got["opponent"] == opponent]  # opponent DGW: met by two different teams
    assert set(vs_opp.index.get_level_values("fixture_id")) == opp_fixtures
    for n in cfg["features"]["windows"]:
        assert np.allclose(vs_opp[f"opp_conceded_roll{n}"].astype(float), conceded.tail(n).mean())


# --- target gameweek selection ----------------------------------------------------------------------

LIVE_FIXTURES = [  # fixture_id, gameweek, kickoff, home, away, started, finished
    (41, 5, "2026-10-04T14:00:00Z", "Ashford", "Brookvale", True, True),
    (42, 5, "2026-10-04T16:30:00Z", "Carrow", "Elmstead", False, False),  # earlier GW, never played: pending
    (51, 6, "2026-10-17T11:30:00Z", "Brookvale", "Carrow", True, True),
    (52, 6, "2026-10-17T14:00:00Z", "Elmstead", "Ashford", True, False),  # in play
    (53, 6, "2026-10-18T16:30:00Z", "Ashford", "Carrow", False, False),
    (54, 6, None, "Brookvale", "Elmstead", False, False),  # no kickoff time yet
    (61, 7, "2026-10-24T14:00:00Z", "Carrow", "Ashford", False, False),
    (62, 7, "2026-10-25T16:30:00Z", "Elmstead", "Brookvale", False, False),
    (63, 7, None, "Ashford", "Elmstead", False, False),
    (99, None, None, "Carrow", "Brookvale", False, False),  # postponed, not rescheduled
]
# pending = unfinished earlier-GW fixtures, plus in-play target-GW fixtures of a team that is scored again
ALL4 = ["Ashford", "Brookvale", "Carrow", "Elmstead"]
TARGET_CASES = {  # meta, kicked off since, chosen ids, (target_gw, target_mode, pending_fixtures, pending_teams, excluded)
    "current_gw_remaining": ({"current_gw": 6, "next_gw": 7}, [], [53],  # pending: 42, 52
                             (6, "current_gw_remaining", 2, ["Ashford", "Carrow", "Elmstead"], 3)),
    "current_gw_exhausted": ({"current_gw": 6, "next_gw": 7}, [53], [61, 62], (7, "next_gw", 4, ALL4, 1)),
    "no_current_gw": ({"current_gw": None, "next_gw": 6}, [], [53], (6, "next_gw", 2, ["Ashford", "Carrow", "Elmstead"], 3)),
    "in_play_teams_not_scored": ({"current_gw": 7, "next_gw": 8}, [53, 61], [62],  # 61 in play, its teams not scored
                                 (7, "current_gw_remaining", 4, ALL4, 2)),
}
DEADLINE = "2026-10-24T10:00:00Z"


def _live_fixtures(kicked_off: list[int]) -> pd.DataFrame:
    fx = pd.DataFrame(LIVE_FIXTURES, columns=["fixture_id", "gameweek", "date", "home_team", "away_team",
                                              "started", "finished"]).assign(home_goals=np.nan, away_goals=np.nan)
    fx.loc[fx["fixture_id"].isin(kicked_off), "started"] = True
    return fx


@pytest.mark.parametrize("via_csv", [False, True], ids=["frame", "csv_roundtrip"])
@pytest.mark.parametrize("case", list(TARGET_CASES))
def test_select_target_fixtures(case, via_csv):
    meta, kicked_off, ids, expected = TARGET_CASES[case]
    fx = _live_fixtures(kicked_off)
    if via_csv:
        fx = pd.read_csv(io.StringIO(fx.to_csv(index=False)))
    chosen, info = select_target_fixtures(fx, {**meta, "next_deadline": DEADLINE})
    assert sorted(chosen["fixture_id"].astype(int)) == ids, "wrong fixtures (started/in-play/no-kickoff must go)"
    keys = ["target_gw", "target_mode", "pending_fixtures", "pending_teams", "excluded_in_gw"]
    assert tuple(info[k] for k in keys) == expected
    assert info["deadline"] == (DEADLINE if expected[1] == "next_gw" else None)


def test_select_target_fixtures_raises_when_season_is_over():
    with pytest.raises(ValueError):
        select_target_fixtures(_live_fixtures([53]), {"current_gw": 6, "next_gw": None})


# --- real artifact (skipped until `python -m src.upcoming` has run) -----------------------------------

def test_real_upcoming_parquet_is_consistent(cfg):
    path = resolve(cfg["paths"]["upcoming"])
    if not path.exists():
        pytest.skip(f"{path} not built yet")
    up, f = pd.read_parquet(path), cfg["features"]
    id_cols = [c for c in f["id_columns"] if c not in ("minutes", "goals", "assists")]
    missing = [c for c in id_cols + f["feature_columns"] + CANDIDATE_COLS + ["prob", "call"] if c not in up.columns]
    assert not missing, f"upcoming.parquet is missing {missing}"
    assert not up.duplicated(KEY).any()
    assert (up["position"] != "GK").all() and not up["status"].isin(cfg["upcoming"]["exclude_status"]).any()
    assert (up["season"] == cfg["upcoming"]["season"]).all() and up["prob"].between(0, 1).all()
    sel_path = resolve(cfg["paths"]["model_selection"])
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        assert (up["call"].astype(bool) == (up["prob"] >= sel["thresholds"][sel["xgb_variant"]])).all()
    meta_path = resolve(cfg["paths"]["upcoming_meta"])
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["n_candidates"] == len(up) and meta["season"] == cfg["upcoming"]["season"]
