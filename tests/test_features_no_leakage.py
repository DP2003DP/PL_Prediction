"""build_features must use only matches strictly before row t (contract section 3)."""
import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.utils import resolve

KEY = ["player_id", "date"]
OPP_COLS = ["opp_conceded_roll5", "opp_conceded_roll10"]
OWN_STAT_CASES = [
    ("goals", lambda v: v + 2, "goals_roll5"),
    ("assists", lambda v: v + 2, "assists_roll5"),
    ("minutes", lambda v: 31 if v >= 60 else 90, "minutes_roll5"),
    ("threat", lambda v: v + 50.0, "threat_roll5"),
    ("creativity", lambda v: v + 50.0, "creativity_roll5"),
    ("xg", lambda v: v + 1.5, "xg_roll5"),
    ("xa", lambda v: v + 1.5, "xa_roll5"),
]


def _as_float(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame.to_numpy(dtype="float64", na_value=np.nan), index=frame.index, columns=frame.columns)


def _keyed(out: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in KEY + cols if c not in out.columns]
    assert not missing, f"output is missing columns {missing}"
    keyed = _as_float(out.set_index(KEY)[cols]).sort_index()
    assert keyed.index.is_unique, "duplicate (player_id, date) rows in output"
    return keyed


def feature_diff(fn, clean, team_matches, cfg, mutate, cols) -> pd.DataFrame:
    """Bool frame keyed by (player_id, date): which feature cells change when mutate() is applied."""
    before = _keyed(fn(clean.copy(deep=True), team_matches.copy(deep=True), cfg), cols)
    after = _keyed(fn(*mutate(clean.copy(deep=True), team_matches.copy(deep=True)), cfg), cols)
    assert before.index.equals(after.index), "the perturbation changed which rows are output"
    same = np.isclose(before.to_numpy(), after.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
    return pd.DataFrame(~same, index=before.index, columns=cols)


def _key(row) -> tuple:
    return int(row["player_id"]), pd.Timestamp(row["date"])


def _changed_cols(diff: pd.DataFrame, key: tuple) -> list[str]:
    return list(diff.columns[diff.loc[key].to_numpy()])


def _changed_keys(diff: pd.DataFrame) -> list[tuple]:
    return list(diff.index[diff.any(axis=1).to_numpy()])


def _player_rows(clean: pd.DataFrame, pid: int) -> pd.DataFrame:
    return clean[clean["player_id"] == pid].sort_values(["date", "fixture_id"]).reset_index(drop=True)


def _set_row(row, changes: dict):
    pid, date = _key(row)

    def mutate(c, tm):
        m = (c["player_id"] == pid) & (c["date"] == date)
        assert m.sum() == 1, "perturbation must hit exactly one row"
        for col, value in changes.items():
            c.loc[m, col] = value
        return c, tm
    return mutate


def _score_change(row, extra: int):
    """Add `extra` goals for row's team in row's fixture, in team_matches and in clean."""
    def mutate(c, tm):
        fx = (tm["season"] == row["season"]) & (tm["fixture_id"] == row["fixture_id"])
        assert fx.sum() == 2, "fixture must have exactly two team_matches rows"
        tm.loc[fx & (tm["team"] == row["team"]), "goals_for"] += extra
        tm.loc[fx & (tm["team"] == row["opponent"]), "goals_against"] += extra
        cf = (c["season"] == row["season"]) & (c["fixture_id"] == row["fixture_id"])
        c.loc[cf & (c["team"] == row["team"]), "team_goals"] += extra
        c.loc[cf & (c["team"] == row["opponent"]), "opp_goals"] += extra
        return c, tm
    return mutate


def _pick_target(clean: pd.DataFrame, cfg: dict, pid: int):
    """Eligible mid-history row t with xG/xA, a non-NaN prior rate, and an eligible next appearance."""
    p = _player_rows(clean, pid)
    prior_eligible = p["eligible"].cumsum().shift(1, fill_value=0)
    ok = (p["eligible"] & (prior_eligible > cfg["features"]["min_prior_matches"])
          & p["xg"].notna() & p["xa"].notna()
          & p["eligible"].shift(-1, fill_value=False) & (p.index < len(p) - 2))
    idx = p.index[ok.to_numpy()]
    assert len(idx) > 0, f"no suitable target row for player {pid}"
    i = idx[len(idx) // 2]
    return p.loc[i], p.loc[i + 1]


def _later_row_vs_opponent(clean, team_matches, row):
    """First eligible row after row's fixture whose opponent is row's opponent, within its next 5 games."""
    games = team_matches[(team_matches["team"] == row["opponent"]) & (team_matches["date"] > row["date"])]
    games = games.sort_values("date").head(5)
    keys = set(zip(games["season"], games["fixture_id"]))
    cand = clean[clean["eligible"] & (clean["opponent"] == row["opponent"])]
    cand = cand[[k in keys for k in zip(cand["season"], cand["fixture_id"])]].sort_values("date")
    assert len(cand) > 0, "synthetic data has no later eligible row against this opponent"
    return cand.iloc[0]


def _goals_roll5(shift: bool):
    def fn(clean, team_matches, cfg):
        df = clean.sort_values(["player_id", "date", "fixture_id"]).reset_index(drop=True)
        src = df.groupby("player_id")["goals"].shift(1) if shift else df["goals"]
        df["goals_roll5"] = src.groupby(df["player_id"]).transform(lambda s: s.rolling(5, min_periods=1).mean())
        return df.loc[df["eligible"], KEY + ["goals_roll5"]]
    return fn


def _opp_conceded_roll5(shift: bool):
    def fn(clean, team_matches, cfg):
        tm = team_matches.sort_values(["team", "date"]).reset_index(drop=True)
        src = tm.groupby("team")["goals_against"].shift(1) if shift else tm["goals_against"]
        tm["opp_conceded_roll5"] = src.groupby(tm["team"]).transform(lambda s: s.rolling(5, min_periods=1).mean())
        opp = tm[["season", "fixture_id", "team", "opp_conceded_roll5"]].rename(columns={"team": "opponent"})
        rows = clean.loc[clean["eligible"], KEY + ["season", "fixture_id", "opponent"]]
        return rows.merge(opp, on=["season", "fixture_id", "opponent"], how="left")
    return fn


@pytest.fixture
def target(clean, cfg):
    return _pick_target(clean, cfg, 101)


@pytest.fixture(scope="module")
def features(_synthetic, cfg):
    return build_features(_synthetic[0].copy(deep=True), _synthetic[1].copy(deep=True), cfg)


# --- leakage: perturb row t, t's features must not move ---------------------------------------

@pytest.mark.parametrize("stat, perturb, sentinel", OWN_STAT_CASES, ids=[c[0] for c in OWN_STAT_CASES])
def test_own_stat_does_not_leak_into_row_features(clean, team_matches, cfg, target, stat, perturb, sentinel):
    t, nxt = target
    cols = cfg["features"]["feature_columns"]
    diff = feature_diff(build_features, clean, team_matches, cfg, _set_row(t, {stat: perturb(t[stat])}), cols)
    assert _changed_cols(diff, _key(t)) == [], f"changing row t's own {stat} changed its features"
    assert diff.loc[_key(nxt), sentinel], f"{sentinel} of the next appearance ignored t's {stat} (test not sensitive)"
    stray = [k for k in _changed_keys(diff) if k[0] != t["player_id"] or k[1] <= t["date"]]
    assert not stray, f"rows other than this player's later rows changed: {stray[:5]}"


def test_all_own_stats_at_once_do_not_leak(clean, team_matches, cfg, target):
    t, nxt = target
    changes = {stat: perturb(t[stat]) for stat, perturb, _ in OWN_STAT_CASES}
    changes.update(influence=t["influence"] + 40.0, ict_index=t["ict_index"] + 4.0, bps=t["bps"] + 30,
                   starts=1.0 - t["starts"], team_goals=t["team_goals"] + 3, opp_goals=t["opp_goals"] + 3)
    cols = cfg["features"]["feature_columns"]
    diff = feature_diff(build_features, clean, team_matches, cfg, _set_row(t, changes), cols)
    assert _changed_cols(diff, _key(t)) == []
    assert diff.loc[_key(nxt)].any()


def test_flipping_target_moves_only_later_prior_rates(clean, team_matches, cfg, target):
    t, nxt = target
    y_t = int(t["goals"] + t["assists"] >= 1)
    mutate = _set_row(t, {"goals": 0, "assists": 0} if y_t else {"goals": 1})
    diff = feature_diff(build_features, clean, team_matches, cfg, mutate, cfg["features"]["feature_columns"])
    assert _changed_cols(diff, _key(t)) == [], "flipping t's own target changed t's features"
    assert diff.loc[_key(nxt), "player_prior_rate"], "player_prior_rate of the next eligible row ignored t's y"
    out = build_features(*mutate(clean.copy(deep=True), team_matches.copy(deep=True)), cfg).set_index(KEY)
    assert int(out.loc[_key(t), cfg["features"]["target"]]) == 1 - y_t


def test_fixture_score_does_not_leak_into_opp_conceded(clean, team_matches, cfg, target):
    t, _ = target
    later = _later_row_vs_opponent(clean, team_matches, t)
    cols = cfg["features"]["feature_columns"]
    diff = feature_diff(build_features, clean, team_matches, cfg, _score_change(t, 4), cols)
    assert _changed_cols(diff, _key(t)) == [], "changing t's fixture score changed t's features"
    assert diff.loc[_key(later), "opp_conceded_roll5"], "a later match vs the same opponent ignored the new score"
    opponent_of = clean.set_index(KEY)["opponent"]
    stray = [k for k in _changed_keys(diff) if k[1] <= t["date"] or opponent_of.loc[k] != t["opponent"]]
    assert not stray, f"unexpected rows changed: {stray[:5]}"
    assert set(diff.columns[diff.any(axis=0).to_numpy()]) <= set(OPP_COLS)


@pytest.mark.parametrize("kind", ["player_stat", "fixture_score"])
def test_negative_control_helper_flags_unshifted_rolling(clean, team_matches, cfg, target, kind):
    t, nxt = target
    if kind == "player_stat":
        make, col, mutate, later = _goals_roll5, "goals_roll5", _set_row(t, {"goals": t["goals"] + 2}), nxt
    else:
        make, col, mutate = _opp_conceded_roll5, "opp_conceded_roll5", _score_change(t, 4)
        later = _later_row_vs_opponent(clean, team_matches, t)
    leaky = feature_diff(make(shift=False), clean, team_matches, cfg, mutate, [col])
    assert leaky.loc[_key(t), col], "perturbation helper missed a deliberately leaky feature"
    safe = feature_diff(make(shift=True), clean, team_matches, cfg, mutate, [col])
    assert not safe.loc[_key(t), col] and safe.loc[_key(later), col]


def test_short_appearances_feed_form_but_not_prior_rate(clean, team_matches, cfg):
    k = cfg["features"]["min_prior_matches"]
    found = None
    for pid in sorted(clean["player_id"].unique()):
        p = _player_rows(clean, pid)
        prior_eligible = p["eligible"].cumsum()
        for i in range(len(p) - 1):
            if not p.loc[i, "eligible"] and p.loc[i + 1, "eligible"] and prior_eligible[i] >= k:
                found = (p.loc[i], p.loc[i + 1])
                break
        if found:
            break
    assert found, "synthetic data has no short appearance followed by an eligible row"
    s, u = found
    y_s = int(s["goals"] + s["assists"] >= 1)
    mutate = _set_row(s, {"goals": 0, "assists": 0} if y_s else {"goals": 1})
    diff = feature_diff(build_features, clean, team_matches, cfg, mutate, cfg["features"]["feature_columns"])
    assert diff.loc[_key(u), "ga_roll5"], "appearances under min_minutes must still feed rolling form"
    assert not diff.loc[_key(u), "player_prior_rate"], "player_prior_rate must use eligible rows only"


# --- output shape and values ---------------------------------------------------------------------

def test_output_has_contract_columns_and_numeric_features(features, cfg):
    f = cfg["features"]
    missing = [c for c in f["id_columns"] + [f["target"]] + f["feature_columns"] if c not in features.columns]
    assert not missing, f"missing columns: {missing}"
    non_numeric = [c for c in f["feature_columns"] if not pd.api.types.is_numeric_dtype(features[c])]
    assert not non_numeric, f"non-numeric feature columns: {non_numeric}"


def test_output_is_exactly_the_eligible_rows(features, clean, cfg):
    assert not features.duplicated(KEY).any(), "duplicate (player_id, date) rows"
    expected = set(zip(clean.loc[clean["eligible"], "player_id"], clean.loc[clean["eligible"], "date"]))
    assert set(zip(features["player_id"], features["date"])) == expected
    assert len(features) == len(expected)
    assert (features["minutes"] >= cfg["cleaning"]["min_minutes"]).all()


def test_output_sorted_and_simple_columns_correct(features, cfg):
    keys = list(zip(features["player_id"], features["date"]))
    assert keys == sorted(keys), "output must be sorted by player_id, date"
    y = features[cfg["features"]["target"]].astype(int)
    assert (y == (features["goals"] + features["assists"] >= 1).astype(int)).all()
    assert (features["is_home"].astype(int) == (features["home_away"] == "H").astype(int)).all()
    for pos in ("DEF", "MID", "FWD"):
        assert (features[f"pos_{pos}"].astype(int) == (features["position"] == pos).astype(int)).all()


def test_build_features_is_pure_and_order_invariant(clean, team_matches, cfg):
    clean_copy, tm_copy = clean.copy(deep=True), team_matches.copy(deep=True)
    f = cfg["features"]
    cols = f["id_columns"] + [f["target"]] + f["feature_columns"]
    out = build_features(clean, team_matches, cfg)[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(clean, clean_copy)
    pd.testing.assert_frame_equal(team_matches, tm_copy)
    shuffled = build_features(clean.sample(frac=1, random_state=1), team_matches.sample(frac=1, random_state=2), cfg)
    pd.testing.assert_frame_equal(shuffled[cols].reset_index(drop=True), out)


def _reference_row(clean, team_matches, cfg, row) -> dict:
    f = cfg["features"]
    hist = clean[(clean["player_id"] == row.player_id) & (clean["date"] < row.date)].sort_values(["date", "fixture_id"])
    ref = {}
    for n in f["windows"]:
        w = hist.tail(n)
        for stat in ("goals", "assists", "minutes", "threat", "creativity", "xg", "xa"):
            ref[f"{stat}_roll{n}"] = w[stat].mean()
        ref[f"ga_roll{n}"] = (w["goals"] + w["assists"]).mean()
        mins = w["minutes"].sum()
        for stat in ("goals", "assists", "threat"):
            ref[f"{stat}_p90_roll{n}"] = 90 * w[stat].sum() / mins if mins >= f["p90_min_minutes"] else np.nan
    ref["matches_in_window"] = min(len(hist), max(f["windows"]))
    ref["prior_matches"] = len(hist)
    ref["days_since_last_match"] = (row.date - hist["date"].iloc[-1]) / pd.Timedelta(days=1) if len(hist) else np.nan
    ref["is_home"] = int(row.home_away == "H")
    for pos in ("DEF", "MID", "FWD"):
        ref[f"pos_{pos}"] = int(row.position == pos)
    elig = hist[hist["eligible"]]
    enough = len(elig) >= f["min_prior_matches"]
    ref["player_prior_rate"] = (elig["goals"] + elig["assists"] >= 1).mean() if enough else np.nan
    opp = team_matches[(team_matches["team"] == row.opponent) & (team_matches["date"] < row.date)]
    for n in f["windows"]:
        ref[f"opp_conceded_roll{n}"] = opp.sort_values("date")["goals_against"].tail(n).mean()
    return ref


def test_feature_values_match_contract_definitions(features, clean, team_matches, cfg):
    cols = cfg["features"]["feature_columns"]
    got = _keyed(features, cols)
    bad = []
    for row in clean[clean["eligible"]].itertuples(index=False):
        ref = _reference_row(clean, team_matches, cfg, row)
        actual = got.loc[(row.player_id, row.date)]
        for c in cols:
            a, e = float(actual[c]), float(ref[c])
            tol = 1.0 if c == "days_since_last_match" else 1e-6  # allow whole-day flooring
            if not ((np.isnan(a) and np.isnan(e)) or abs(a - e) < tol):
                bad.append((row.player_id, str(row.date), c, a, e))
    assert not bad, f"{len(bad)} feature values differ from the contract definition; first: {bad[:8]}"


def test_transfer_player_history_spans_teams(features, clean, synthetic_meta):
    pid = synthetic_meta["transfer_player"]
    p = _player_rows(clean, pid)
    first_new = p[(p["team"] == synthetic_meta["transfer_to"]) & p["eligible"]].iloc[0]
    earlier = p[p["date"] < first_new["date"]]
    assert len(earlier) > 0 and (earlier["team"] == synthetic_meta["transfer_from"]).any()
    row = features.set_index(KEY).loc[_key(first_new)]
    assert int(row["prior_matches"]) == len(earlier)
    assert int(row["matches_in_window"]) == min(len(earlier), 10)
    assert not np.isnan(float(row["goals_roll5"]))


def test_debut_row_has_no_history(features, clean, cfg, synthetic_meta):
    first = _player_rows(clean, synthetic_meta["debut_player"]).iloc[0]
    assert first["eligible"]
    row = features.set_index(KEY).loc[_key(first)]
    assert int(row["prior_matches"]) == 0 and int(row["matches_in_window"]) == 0
    empty = [c for c in cfg["features"]["feature_columns"] if "_roll" in c and not c.startswith("opp_")]
    values = row[empty + ["days_since_last_match", "player_prior_rate"]].to_numpy(dtype="float64", na_value=np.nan)
    assert np.isnan(values).all()


def test_opponent_history_spans_seasons_and_is_empty_for_promoted_team(features, team_matches, cfg, synthetic_meta):
    team = synthetic_meta["promoted_team"]
    games = team_matches[team_matches["team"] == team].sort_values("date")
    first, second = games.iloc[0], games.iloc[1]

    def vs(game):
        m = ((features["opponent"] == team) & (features["season"] == game["season"])
             & (features["fixture_id"] == game["fixture_id"]))
        assert m.any(), "synthetic data has no eligible row against the promoted team in this fixture"
        return features.loc[m]
    assert vs(first)[OPP_COLS].isna().all().all(), "a team with no prior matches must give NaN opp_conceded"
    assert np.allclose(vs(second)["opp_conceded_roll5"].astype(float), float(first["goals_against"]))
    opener = features[(features["season"] == cfg["data"]["seasons"][1]) & (features["matchweek"] == 1)
                      & (features["opponent"] != team)]
    assert len(opener) > 0 and opener["opp_conceded_roll5"].notna().all(), "opponent history must span seasons"


# --- real artifacts (skipped until the pipeline has produced them) --------------------------------

def _real(name: str, cfg: dict) -> pd.DataFrame:
    path = resolve(cfg["paths"][name])
    if not path.exists():
        pytest.skip(f"{path} not built yet")
    return pd.read_parquet(path)


def test_real_features_parquet_matches_contract(cfg):
    out = _real("features", cfg)
    f = cfg["features"]
    missing = [c for c in f["id_columns"] + [f["target"]] + f["feature_columns"] if c not in out.columns]
    assert not missing, f"features.parquet is missing {missing}"
    assert not out.duplicated(KEY).any()
    assert (out["minutes"] >= cfg["cleaning"]["min_minutes"]).all()
    assert set(out[f["target"]].unique()) <= {0, 1}
    clean_path = resolve(cfg["paths"]["clean"])
    if clean_path.exists():
        clean = pd.read_parquet(clean_path, columns=KEY + ["eligible"])
        assert len(out) == int(clean["eligible"].sum())


def test_real_data_no_leakage_on_player_subset(cfg):
    clean, team_matches = _real("clean", cfg), _real("team_matches", cfg)
    with_xg = clean[clean["eligible"] & clean["xg"].notna()]
    pids = with_xg["player_id"].value_counts().head(25).index
    sub = clean[clean["player_id"].isin(pids)].reset_index(drop=True)
    t, nxt = _pick_target(sub, cfg, int(pids[0]))
    changes = {stat: perturb(t[stat]) for stat, perturb, _ in OWN_STAT_CASES}
    diff = feature_diff(build_features, sub, team_matches, cfg, _set_row(t, changes), cfg["features"]["feature_columns"])
    assert _changed_cols(diff, _key(t)) == []
    assert diff.loc[_key(nxt)].any()
    diff = feature_diff(build_features, sub, team_matches, cfg, _score_change(t, 4), OPP_COLS)
    assert _changed_cols(diff, _key(t)) == []


TRUNCATION_CUTOFFS = ["2021-08-21", "2023-01-22", "2025-08-15"]  # early season, mid-season, test-season opener
RANDOMIZED_CUTOFF = "2023-01-22"


def _end_of_day(cutoff: str) -> pd.Timestamp:
    return pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)


@pytest.fixture(scope="module")
def real_full(cfg):
    clean, team_matches = _real("clean", cfg), _real("team_matches", cfg)
    return clean, team_matches, _keyed(build_features(clean, team_matches, cfg), cfg["features"]["feature_columns"])


def _assert_prefix_unchanged(full: pd.DataFrame, rebuilt: pd.DataFrame, end: pd.Timestamp) -> None:
    before = full[full.index.get_level_values("date") < end]
    after = rebuilt[rebuilt.index.get_level_values("date") < end]
    assert before.index.equals(after.index), "rows up to the cutoff differ between the two builds"
    same = np.isclose(before.to_numpy(), after.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
    bad = {c: int(n) for c, n in zip(before.columns, (~same).sum(axis=0)) if n}
    assert not bad, f"features dated on/before the cutoff depend on later matches: {bad}"


@pytest.mark.parametrize("cutoff", TRUNCATION_CUTOFFS)
def test_real_data_truncated_future_gives_same_features(real_full, cfg, cutoff):
    clean, team_matches, full = real_full
    end = _end_of_day(cutoff)
    dates = full.index.get_level_values("date")
    assert ((dates >= end - pd.Timedelta(days=1)) & (dates < end)).any(), f"{cutoff} is not a matchday"
    rebuilt = build_features(clean[clean["date"] < end], team_matches[team_matches["date"] < end], cfg)
    rebuilt = _keyed(rebuilt, cfg["features"]["feature_columns"])
    assert len(rebuilt) == int((dates < end).sum())
    _assert_prefix_unchanged(full, rebuilt, end)


def test_real_data_randomized_future_does_not_change_past(real_full, cfg):
    clean, team_matches, full = real_full
    end, rng = _end_of_day(RANDOMIZED_CUTOFF), np.random.default_rng(cfg["project"]["random_seed"])
    c, tm = clean.copy(deep=True), team_matches.copy(deep=True)
    fut = (c["date"] >= end).to_numpy()
    n, min_min = int(fut.sum()), cfg["cleaning"]["min_minutes"]
    eligible = c.loc[fut, "minutes"].to_numpy() >= min_min
    c.loc[fut, "minutes"] = np.where(eligible, rng.integers(min_min, 91, n), rng.integers(1, min_min, n))
    for col in ("goals", "assists", "team_goals", "opp_goals"):
        c.loc[fut, col] = rng.integers(0, 5, n)
    for col, high in (("threat", 150.0), ("creativity", 150.0), ("xg", 2.0), ("xa", 2.0)):
        c.loc[fut, col] = rng.uniform(0, high, n).round(2)
    tfut = (tm["date"] >= end).to_numpy()
    fixture = tm.loc[tfut].groupby(["season", "fixture_id"]).ngroup().to_numpy()
    scores = rng.integers(0, 6, (fixture.max() + 1, 2))
    home = (tm.loc[tfut, "home_away"] == "H").to_numpy()
    tm.loc[tfut, "goals_for"] = np.where(home, scores[fixture, 0], scores[fixture, 1])
    tm.loc[tfut, "goals_against"] = np.where(home, scores[fixture, 1], scores[fixture, 0])
    rebuilt = _keyed(build_features(c, tm, cfg), cfg["features"]["feature_columns"])
    assert rebuilt.index.equals(full.index), "randomizing later matches changed which rows are output"
    _assert_prefix_unchanged(full, rebuilt, end)
    later = (full.index.get_level_values("date") >= end)
    moved = ~np.isclose(full[later].to_numpy(), rebuilt[later].to_numpy(), equal_nan=True)
    assert moved.any(axis=0).sum() >= 10, "randomization barely moved later features (test not sensitive)"
