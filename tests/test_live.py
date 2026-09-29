"""Unit tests for src/live.py parsing on small synthetic FPL API payloads. No network access."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

import src.live as live
from src.live import StructureError
from src.scrape import PLAYER_COLUMNS

SEASON = "2026-27"
NAMES = pd.Series({1: "Ashford", 2: "Brookvale", 3: "Carrow", 4: "Dunmore"})
PROPORTIONAL = {"xg": "expected_goals", "xa": "expected_assists", "threat": "threat", "creativity": "creativity",
                "influence": "influence", "ict_index": "ict_index"}
ELEMENTS = [  # id, code, first, second, current team, element_type
    (11, 1011, "Alan", "Archer", 1, 4), (12, 1012, "Gary", "Gloves", 1, 1), (13, 1013, "Ben", "Barker", 1, 2),
    (21, 1021, "Carl", "Cole", 2, 3),
    (31, 1031, "Tom", "Travers", 3, 3),  # played for Ashford v Carrow in GW1, then joined Carrow
    (32, 1032, "Eli", "Evans", 3, 2), (33, 1033, "Cara", "Cross", 3, 4),
    (34, 1034, "Lee", "Loaned", 2, 3),  # unused Carrow sub in GW1 (no bps entry), now at Brookvale
    (41, 1041, "Dave", "Dunn", 4, 4), (42, 1042, "Olly", "Owen", 4, 2),
    (90, 1090, "Mike", "Manager", 4, 5),  # assistant manager
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("tests/test_live.py must never touch the network")
    monkeypatch.setattr(live, "Downloader", refuse)
    monkeypatch.setattr("requests.sessions.Session.request", refuse)


def _boot() -> dict:
    return {
        "events": [{"id": gw, "deadline_time": f"2026-08-{7 * gw + 7:02d}T17:30:00Z", "finished": gw <= 2,
                    "data_checked": gw <= 2, "is_current": gw == 3, "is_next": gw == 4} for gw in range(1, 5)],
        "teams": [{"id": int(i), "code": 100 + int(i), "name": n} for i, n in NAMES.items()],
        "elements": [{"id": e, "code": c, "first_name": f, "second_name": s, "team": t, "element_type": et,
                      "status": "a", "chance_of_playing_next_round": None, "news": ""}
                     for e, c, f, s, t, et in ELEMENTS],
        "element_types": [{"id": i, "singular_name_short": s}
                          for i, s in enumerate(["GKP", "DEF", "MID", "FWD", "AM"], start=1)],
    }


def _fixture(fid, gw, kickoff, h, a, hs, as_, started, finished, goals=(), assists=(), own=(), bps=()):
    def stat(identifier, items):  # items: (side, element, value)
        return {"identifier": identifier, "h": [{"value": v, "element": e} for s, e, v in items if s == "h"],
                "a": [{"value": v, "element": e} for s, e, v in items if s == "a"]}
    stats = [stat("goals_scored", goals), stat("assists", assists), stat("own_goals", own), stat("bps", bps)]
    return {"id": fid, "event": gw, "kickoff_time": kickoff, "team_h": h, "team_a": a, "team_h_score": hs,
            "team_a_score": as_, "started": started, "finished": finished, "stats": stats if started else []}


def _fixtures() -> list:
    return [
        _fixture(1, 1, "2026-08-15T14:00:00Z", 1, 3, 2, 1, True, True, goals=[("h", 11, 1), ("a", 33, 1)],
                 assists=[("h", 31, 1)], own=[("a", 32, 1)],
                 bps=[("h", 11, 30), ("h", 12, 20), ("h", 13, 18), ("h", 31, 25), ("a", 32, 5), ("a", 33, 22)]),
        _fixture(2, 1, "2026-08-16T16:30:00Z", 2, 4, 0, 0, True, True, bps=[("h", 21, 12), ("a", 41, 10), ("a", 42, 14)]),
        _fixture(3, 2, "2026-08-22T14:00:00Z", 1, 2, 1, 0, True, True, goals=[("h", 11, 1)],
                 bps=[("h", 11, 28), ("a", 21, 9)]),
        _fixture(4, 2, "2026-08-25T19:00:00Z", 4, 1, 1, 2, True, True, goals=[("h", 41, 1), ("a", 11, 1)],
                 assists=[("a", 13, 1)], own=[("h", 42, 1)],
                 bps=[("h", 41, 24), ("h", 42, 3), ("a", 11, 26), ("a", 13, 21)]),
        _fixture(6, 3, "2026-08-29T14:00:00Z", 3, 1, 1, 1, True, True, goals=[("h", 33, 1), ("a", 11, 1)],
                 assists=[("h", 31, 1)],
                 bps=[("h", 31, 27), ("h", 32, 15), ("h", 33, 25), ("a", 11, 24), ("a", 12, 19)]),
        _fixture(7, 3, "2026-08-30T16:30:00Z", 2, 4, 0, 0, True, False, bps=[("h", 21, 8)]),  # in play
        _fixture(8, None, None, 2, 3, None, None, False, False),  # postponed, no date yet
    ]


def _el(eid, per_fixture, totals=None):
    """Live element: per_fixture = {fixture: {identifier: value}}; totals override the summed stats."""
    summed = {k: sum(v.get(k, 0) for v in per_fixture.values()) for k in ("minutes", "goals_scored", "assists")}
    stats = {"influence": "10.0", "creativity": "8.0", "threat": "12.0", "ict_index": "3.0", "bps": 15,
             "starts": int(summed["minutes"] > 0), "expected_goals": "0.20", "expected_assists": "0.10",
             **summed, **(totals or {})}
    explain = [{"fixture": fid, "stats": [{"identifier": k, "points": 0, "value": v} for k, v in vals.items()]}
               for fid, vals in per_fixture.items()]
    return {"id": eid, "stats": stats, "explain": explain}


ALAN_GW2 = {"minutes": 150, "goals_scored": 2, "assists": 0, "influence": "60.0", "creativity": "30.0",
            "threat": "75.0", "ict_index": "16.5", "bps": 54, "starts": 2, "expected_goals": "1.20",
            "expected_assists": "0.30"}
BEN_GW2 = {"minutes": 90, "goals_scored": 0, "assists": 1, "influence": "20.0", "creativity": "25.0",
           "threat": "5.0", "ict_index": "5.0", "bps": 21, "starts": 1, "expected_goals": "0.02",
           "expected_assists": "0.40"}


def _lives() -> dict:
    full = {"minutes": 90}
    am = _el(90, {2: {"mng_win": 1}}, {"minutes": 0})
    return {
        1: {"elements": [_el(11, {1: {"minutes": 90, "goals_scored": 1}}, {"threat": "45.0", "expected_goals": "0.61"}),
                         _el(12, {1: full}), _el(13, {1: full}), _el(31, {1: {"minutes": 90, "assists": 1}}),
                         _el(32, {1: full}), _el(33, {1: {"minutes": 90, "goals_scored": 1}}), _el(34, {1: {"minutes": 0}}),
                         _el(21, {2: full}), _el(41, {2: full}), _el(42, {2: full}), am]},
        2: {"elements": [_el(11, {3: {"minutes": 90, "goals_scored": 1}, 4: {"minutes": 60, "goals_scored": 1}}, ALAN_GW2),
                         _el(13, {3: {"minutes": 0}, 4: {"minutes": 90, "assists": 1}}, BEN_GW2),
                         _el(21, {3: full}), _el(41, {4: {"minutes": 90, "goals_scored": 1}}), _el(42, {4: full})]},
        3: {"elements": [_el(31, {6: {"minutes": 90, "assists": 1}}), _el(32, {6: full}),
                         _el(33, {6: {"minutes": 90, "goals_scored": 1}}), _el(11, {6: {"minutes": 90, "goals_scored": 1}}),
                         _el(12, {6: full}), _el(21, {7: {"minutes": 30}})]},
    }


def _normalize(boot=None, fixtures=None, lives=None):
    return live.normalize(SEASON, boot or _boot(), fixtures or _fixtures(), lives or _lives(), NAMES)


@pytest.fixture(scope="module")
def normalized():
    players, fx, pm, tm = _normalize()
    return players, pm.set_index(["fpl_element", "fixture_id"]).sort_index(), tm


def test_synthetic_payloads_pass_the_structure_checks(tmp_path):
    for name, payload, parse in (("bootstrap", _boot(), live.parse_bootstrap),
                                 ("fixtures", _fixtures(), live.parse_fixtures),
                                 *((f"event_{gw}", ev, live.parse_event) for gw, ev in _lives().items())):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert parse(path, f"test://{name}") == payload


def test_output_schema_and_only_finished_fixtures(normalized):
    players, pm, tm = normalized
    assert list(pm.reset_index()[PLAYER_COLUMNS].columns) == PLAYER_COLUMNS
    assert set(pm.index.get_level_values("fixture_id")) == {1, 2, 3, 4, 6}, "only finished fixtures"
    assert sorted(tm["fixture_id"].unique()) == [1, 2, 3, 4, 6] and len(tm) == 10
    assert (pm["season"] == SEASON).all() and pm["date"].str.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$").all()
    assert pm.loc[(31, 1), "player_name"] == "Tom Travers" and pm.loc[(31, 1), "player_id"] == 1031


def test_team_attribution_follows_bps_side_across_a_transfer(normalized):
    players, pm, _ = normalized
    assert players.set_index("fpl_element").loc[31, "team"] == "Carrow"
    tom_old = pm.loc[(31, 1)]  # Ashford v Carrow while still at Ashford: current club is also in the fixture
    assert (tom_old["team"], tom_old["opponent"], tom_old["home_away"]) == ("Ashford", "Carrow", "H")
    assert (tom_old["team_goals"], tom_old["opp_goals"]) == (2, 1)
    tom_new = pm.loc[(31, 6)]  # Carrow v Ashford after the move
    assert (tom_new["team"], tom_new["opponent"], tom_new["home_away"]) == ("Carrow", "Ashford", "H")
    ben_bench = pm.loc[(13, 3)]  # 0 minutes, no bps entry: falls back to the current club, which is in the fixture
    assert (ben_bench["team"], ben_bench["home_away"]) == ("Ashford", "H")
    assert (34, 1) not in pm.index, "no bps entry and current club not in the fixture: the row must be dropped"
    fx = {f["id"]: (NAMES[f["team_h"]], NAMES[f["team_a"]]) for f in _fixtures()}
    for (_, fid), row in pm.iterrows():
        home, away = fx[fid]
        expected = (home, away) if row["home_away"] == "H" else (away, home)
        assert (row["team"], row["opponent"]) == expected


def test_double_gameweek_split(normalized):
    _, pm, _ = normalized
    for eid, totals, minutes, goals, assists in ((11, ALAN_GW2, (90, 60), (1, 1), (0, 0)),
                                                  (13, BEN_GW2, (0, 90), (0, 0), (0, 1))):
        legs = pm.loc[[(eid, 3), (eid, 4)]]
        assert tuple(legs["minutes"]) == minutes and tuple(legs["goals"]) == goals
        assert tuple(legs["assists"]) == assists
        share = np.array(minutes) / sum(minutes)
        for col, key in PROPORTIONAL.items():
            vals = legs[col].to_numpy(dtype=float)
            assert np.allclose(vals, float(totals[key]) * share, atol=0.011), f"{eid} {col} not split by minutes"
            assert abs(vals.sum() - float(totals[key])) <= 0.02
        bps = legs["bps"].to_numpy()
        assert np.allclose(bps, totals["bps"] * share, atol=0.51) and abs(int(bps.sum()) - totals["bps"]) <= 1
        assert legs["starts"].isna().all(), "starts cannot be split across a double gameweek"
    single = pm.loc[(11, 1)]  # one fixture that gameweek: totals are used as they are
    assert (single["minutes"], single["threat"], single["xg"]) == (90, 45.0, 0.61)


def test_assistant_manager_dropped_and_goalkeeper_kept(normalized):
    players, pm, _ = normalized
    assert 1090 not in set(players["player_id"]) and 90 not in set(pm.index.get_level_values("fpl_element"))
    assert set(players["position"]) == {"GK", "DEF", "MID", "FWD"}
    assert pm.loc[(12, 1), "position"] == "GK"


def test_own_goals_reconcile_with_the_score(normalized):
    _, pm, tm = normalized
    assert live.reconcile_goals(pm.reset_index(), tm, _fixtures(), NAMES) == []
    ashford_f1 = int(pm.xs(1, level="fixture_id").query("team == 'Ashford'")["goals"].sum())
    assert ashford_f1 == 1, "Ashford's second goal in fixture 1 is a Carrow own goal"


@pytest.mark.parametrize("change, expected", [
    ("drop_own_goal", [(1, "Ashford")]),
    ("own_goal_on_wrong_side", [(1, "Ashford"), (1, "Carrow")]),
    ("wrong_score", [(4, "Ashford")]),
])
def test_reconcile_goals_flags_mismatches(normalized, change, expected):
    _, pm, tm = normalized
    fixtures, tm = _fixtures(), tm.copy()
    own = next(s for s in fixtures[0]["stats"] if s["identifier"] == "own_goals")
    if change == "drop_own_goal":
        own["a"] = []
    elif change == "own_goal_on_wrong_side":
        own["h"], own["a"] = own["a"], []
    else:
        tm.loc[(tm["fixture_id"] == 4) & (tm["team"] == "Ashford"), "goals_for"] = 3
    assert sorted(live.reconcile_goals(pm.reset_index(), tm, fixtures, NAMES)) == expected


def _bad_event(lives):
    del lives[1]["elements"][0]["explain"]
    return lives[1]


def _bad_explain(lives):
    del lives[2]["elements"][0]["explain"][0]["stats"][0]["value"]
    return lives[2]


@pytest.mark.parametrize("parse, payload", [
    ("parse_event", lambda: _bad_event(_lives())),
    ("parse_event", lambda: _bad_explain(_lives())),
    ("parse_event", lambda: {"elements": []}),
    ("parse_fixtures", lambda: {"fixtures": _fixtures()}),
    ("parse_fixtures", lambda: [{k: v for k, v in f.items() if k != "team_h"} for f in _fixtures()]),
    ("parse_bootstrap", lambda: {k: v for k, v in _boot().items() if k != "teams"}),
    ("parse_bootstrap", lambda: "<html>Just a moment...</html>"),
], ids=["element_without_explain", "explain_stat_without_value", "no_elements", "fixtures_not_a_list",
        "fixture_without_team_h", "bootstrap_without_teams", "html_instead_of_json"])
def test_malformed_payload_raises_structure_error(tmp_path, parse, payload):
    path = tmp_path / "payload.json"
    data = payload()
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    with pytest.raises(StructureError):
        getattr(live, parse)(path, "test://payload")


def test_unexpected_element_types_or_team_ids_raise_structure_error():
    boot = _boot()
    boot["element_types"][0]["singular_name_short"] = "GK"
    with pytest.raises(StructureError):
        _normalize(boot=boot)
    fixtures = copy.deepcopy(_fixtures())
    fixtures[1]["team_a"] = 9
    with pytest.raises(StructureError):
        _normalize(fixtures=fixtures)
