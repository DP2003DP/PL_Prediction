"""Smoke tests for app/app.py via Streamlit's AppTest, against the real artifacts (skipped when they are missing)."""
import json
import re

import pandas as pd
import pytest
import xgboost as xgb
from streamlit.testing.v1 import AppTest

import src.utils as utils
from src.utils import load_config, resolve

APP = str(resolve("app/app.py"))
ARTIFACTS = ["xgb_model", "model_selection", "features", "clean", "metrics"]
_missing = [load_config()["paths"][k] for k in ARTIFACTS if not resolve(load_config()["paths"][k]).exists()]
pytestmark = pytest.mark.skipif(bool(_missing), reason=f"app artifacts not built yet: {_missing}")

PROB_LABEL = "P(goal or assist)"
UPCOMING, HISTORICAL = "Upcoming gameweek", "Historical match"


def _pick_transfer(f: pd.DataFrame, sel: dict) -> pd.Series:
    prev = f.groupby("player_id")[["team", "season"]].shift()
    return (f["season"] == prev["season"]) & (f["team"] != prev["team"])


# each case picks the middle row of a subset of features.parquet
CASES = {
    "first_appearance": lambda f, sel: f["prior_matches"] == 0,
    "five_or_fewer_prior": lambda f, sel: f["prior_matches"].between(1, 5),
    "train_season": lambda f, sel: f["season"] == sel["train_seasons"][1],
    "val_season": lambda f, sel: f["season"] == sel["val_season"],
    "test_season_defender": lambda f, sel: (f["season"] == sel["test_season"]) & (f["position"] == "DEF"),
    "mid_season_transfer": _pick_transfer,
}


@pytest.fixture(scope="module")
def artifacts(cfg):
    paths = {k: resolve(cfg["paths"][k]) for k in ARTIFACTS}
    features = pd.read_parquet(paths["features"]).sort_values(["player_id", "date", "fixture_id"])
    booster = xgb.Booster()
    booster.load_model(str(paths["xgb_model"]))
    return {
        "features": features,
        "booster": booster,
        "sel": json.loads(paths["model_selection"].read_text(encoding="utf-8")),
        "metrics": json.loads(paths["metrics"].read_text(encoding="utf-8")),
    }


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.radio(key="mode").set_value(HISTORICAL).run()
    return at


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _prob_text(at: AppTest) -> str:
    return next(m.value for m in at.metric if m.label == PROB_LABEL)


def _choose(at: AppTest, row: pd.Series) -> None:
    pid = int(row["player_id"])
    at.selectbox(key="player").set_value(pid).run()
    at.selectbox(key=f"match_{pid}").set_value(row.name).run()


def test_default_render_has_no_exceptions(app):
    _assert_clean(app)
    assert app.title[0].value == "Premier League goal-or-assist predictor"
    assert len(app.selectbox(key="player").options) > 100
    assert {m.label for m in app.metric} >= {PROB_LABEL, "Model call", "Actual outcome"}
    assert [s.value for s in app.subheader] == ["Why this prediction", "Last 10 appearances before this match"]
    assert not app.warning, "SHAP fell back to gain importance"
    assert any("Not betting or fantasy advice" in c.value for c in app.caption)


def test_model_note_shows_test_season_metrics(app, artifacts):
    sel, metrics = artifacts["sel"], artifacts["metrics"]
    m = metrics["models"][sel["xgb_variant"]]
    note = app.caption[0].value
    assert f"seasons {sel['train_seasons'][0]} to {sel['val_season']}" in note
    assert f"test-season ({metrics['test_season']})" in note
    assert f"ROC-AUC = {m['roc_auc']:.3f}" in note
    assert f"PR-AUC = {m['pr_auc']:.3f}" in note


@pytest.mark.parametrize("case", list(CASES))
def test_switching_player_and_match(app, artifacts, case):
    f, sel, booster = artifacts["features"], artifacts["sel"], artifacts["booster"]
    subset = f[CASES[case](f, sel)]
    assert len(subset), f"no rows for case {case}"
    row = subset.iloc[len(subset) // 2]
    _choose(app, row)
    _assert_clean(app)
    assert app.selectbox(key="player").value == row["player_id"]
    assert app.selectbox(key=f"match_{int(row['player_id'])}").value == row.name

    # probability: shown within [0, 1] and equal to an independent prediction for the same row
    cols = booster.feature_names
    p = float(booster.predict(xgb.DMatrix(f.loc[[row.name], cols].astype(float)))[0])
    shown = _prob_text(app)
    assert 0.0 <= float(shown.rstrip("%")) / 100 <= 1.0
    assert shown == f"{p:.1%}"
    threshold = sel["thresholds"][sel["xgb_variant"]]
    assert next(m.value for m in app.metric if m.label == "Model call") == ("Yes" if p >= threshold else "No")

    # SHAP: base value + contributions reproduce the model output
    shap_note = next(c.value for c in app.caption if c.value.startswith("SHAP contributions"))
    assert re.search(r"log-odds = ([\d.]+%)\.", shap_note).group(1) == shown

    # in-sample / out-of-sample flag
    in_sample = row["season"] in sel["train_seasons"] or row["season"] == sel["val_season"]
    assert bool(app.info) == in_sample
    assert bool(app.success) == (row["season"] == sel["test_season"])

    # last 10 appearances: strictly before the match, newest first, as many as the history allows
    tables = [d.value for d in app.dataframe if "Opponent" in d.value.columns]
    expected = min(10, int(row["prior_matches"]))
    if expected == 0:
        assert not tables
        assert any("No earlier appearances" in m.value for m in app.markdown)
    else:
        dates = list(tables[0]["Date"])
        assert len(dates) == expected
        assert all(d < f"{row['date']:%Y-%m-%d}" for d in dates)
        assert dates == sorted(dates, reverse=True) and len(set(dates)) == len(dates)


def test_switching_away_and_back_resets_match_to_newest(app, artifacts):
    default = app.selectbox(key="player").value
    newest = app.selectbox(key=f"match_{default}").value
    oldest = artifacts["features"].query("player_id == @default").index[0]
    app.selectbox(key=f"match_{default}").set_value(oldest).run()
    assert app.selectbox(key=f"match_{default}").value == oldest != newest
    other = int(artifacts["features"].query("player_id != @default")["player_id"].iloc[0])
    app.selectbox(key="player").set_value(other).run()
    _assert_clean(app)
    app.selectbox(key="player").set_value(default).run()
    _assert_clean(app)
    assert app.selectbox(key=f"match_{default}").value == newest


def _hide(monkeypatch, target: str) -> None:
    """Make the app see `target` (a config path) as missing, without touching the file."""
    real = utils.resolve
    monkeypatch.setattr(utils, "resolve", lambda rel: real("missing") / rel if str(rel) == target else real(rel))


@pytest.mark.parametrize("artifact", ["xgb_model", "clean"])
def test_missing_artifact_shows_error_not_exception(cfg, monkeypatch, artifact):
    target = cfg["paths"][artifact]
    _hide(monkeypatch, target)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(target in e.value for e in at.error)


# --- upcoming-gameweek mode (skipped until src.upcoming has written its outputs) ---------------------------------

_no_upcoming = [load_config()["paths"][k] for k in ["upcoming", "upcoming_meta"]
                if not resolve(load_config()["paths"][k]).exists()]
needs_upcoming = pytest.mark.skipif(bool(_no_upcoming), reason=f"upcoming artifacts not built yet: {_no_upcoming}")


@pytest.fixture(scope="module")
def upcoming(cfg):
    up = pd.read_parquet(resolve(cfg["paths"]["upcoming"]))
    meta = json.loads(resolve(cfg["paths"]["upcoming_meta"]).read_text(encoding="utf-8"))
    return up, meta


@pytest.fixture
def up_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def _board(at: AppTest) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if {"Rank", "Player", "P(G/A)"} <= set(d.value.columns))


@needs_upcoming
def test_default_render_is_upcoming_view(up_app):
    _assert_clean(up_app)
    assert up_app.radio(key="mode").value == UPCOMING
    assert len(_board(up_app)) > 0
    assert up_app.selectbox(key="up_row").value is not None
    assert {m.label for m in up_app.metric} >= {PROB_LABEL, "Model call", "Fixture"}


@needs_upcoming
def test_leaderboard_header_shows_meta_gameweek(up_app, upcoming):
    meta = upcoming[1]
    assert up_app.subheader[0].value == f"Gameweek {meta['gameweek']} ({meta['season']})"


@needs_upcoming
def test_caveat_shows_short_minutes_band_calibration(up_app, artifacts, cfg):
    bands = artifacts["metrics"].get("by_minutes_band")
    if not bands:
        pytest.skip("reports/metrics.json has no by_minutes_band")
    key = next(k for k in bands if k.split("-")[0] == str(cfg["cleaning"]["min_minutes"]))
    band = bands[key]
    caveat = next(w.value for w in up_app.warning if w.value.startswith("**Read this first.**"))
    assert f"playing {key} minutes" in caveat
    assert f"about {band['mean_predicted'] / band['base_rate']:.1f}x" in caveat
    assert f"{band['mean_predicted']:.1%} vs {band['base_rate']:.1%} actual" in caveat


@needs_upcoming
def test_selected_row_probability_matches_stored_and_model(up_app, upcoming, artifacts):
    up, booster = upcoming[0], artifacts["booster"]
    up_app.toggle(key="up_starters").set_value(False).run()
    board = _board(up_app)
    for ridx in [board.index[0], board.index[len(board) // 2], board.index[-1]]:
        up_app.selectbox(key="up_row").set_value(ridx).run()
        _assert_clean(up_app)
        shown = board.loc[ridx]
        match = up[(up["player_name"] == shown["Player"]) & (up["team"] == shown["Team"])
                   & (up["opponent"] + " (" + up["home_away"] + ")" == shown["Opponent"])]
        assert len(match) == 1, f"leaderboard row {ridx} does not identify one upcoming.parquet row"
        stored = float(match["prob"].iloc[0])
        p = float(booster.predict(xgb.DMatrix(match[booster.feature_names].astype(float)))[0])
        assert 0.0 <= stored <= 1.0
        assert abs(p - stored) < 1e-6
        assert _prob_text(up_app) == f"{stored:.1%}"
        assert abs(shown["P(G/A)"] - 100 * stored) < 1e-6
        assert not [w for w in up_app.warning if "different model file" in w.value]


@needs_upcoming
def test_likely_starters_filter(up_app, upcoming):
    meta = upcoming[1]
    assert up_app.toggle(key="up_starters").value is True
    starters = len(_board(up_app))
    up_app.toggle(key="up_starters").set_value(False).run()
    _assert_clean(up_app)
    assert starters < len(_board(up_app)) == meta["n_candidates"]
    if "n_likely_starters" in meta:
        assert starters == meta["n_likely_starters"]


@needs_upcoming
def test_switching_mode_to_historical_and_back(up_app):
    up_app.radio(key="mode").set_value(HISTORICAL).run()
    _assert_clean(up_app)
    assert up_app.selectbox(key="player").value is not None
    assert not [s for s in up_app.selectbox if s.key == "up_row"]
    up_app.radio(key="mode").set_value(UPCOMING).run()
    _assert_clean(up_app)
    assert up_app.selectbox(key="up_row").value is not None
    assert len(_board(up_app)) > 0


def test_missing_upcoming_defaults_to_historical_and_shows_info(cfg, monkeypatch):
    _hide(monkeypatch, cfg["paths"]["upcoming"])
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    _assert_clean(at)
    assert at.radio(key="mode").value == HISTORICAL
    at.radio(key="mode").set_value(UPCOMING).run()
    _assert_clean(at)
    assert any("No upcoming-gameweek predictions yet" in i.value for i in at.info)
