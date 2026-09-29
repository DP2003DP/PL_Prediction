"""Streamlit app: probability that a Premier League player scores or assists, for the next gameweek or a past match."""
import json
import re
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import upcoming_view  # noqa: E402
from src.utils import load_config, resolve  # noqa: E402

TOWARDS, AWAY, NEUTRAL = "#2166ac", "#d6604d", "#9e9e9e"
MODES = ["Upcoming gameweek", "Historical match"]
PATH_KEYS = ["xgb_model", "model_selection", "features", "clean", "metrics", "current_clean", "upcoming", "upcoming_meta"]
STAT_LABELS = {
    "goals": "Goals", "assists": "Assists", "ga": "Goals + assists", "minutes": "Minutes",
    "threat": "Threat", "creativity": "Creativity", "xg": "xG", "xa": "xA",
}
FIXED_LABELS = {
    "matches_in_window": "Appearances in last-10 window",
    "prior_matches": "Career appearances so far",
    "days_since_last_match": "Days since last appearance",
    "is_home": "Home match",
    "pos_DEF": "Position: defender",
    "pos_MID": "Position: midfielder",
    "pos_FWD": "Position: forward",
    "player_prior_rate": "Past goal-or-assist rate",
}
BINARY_FEATURES = {"is_home", "pos_DEF", "pos_MID", "pos_FWD"}
COUNT_FEATURES = {"matches_in_window", "prior_matches"}
# letters that NFKD does not decompose into base letter + combining mark
FOLD_EXTRA = str.maketrans({"Ø": "O", "ø": "o", "ß": "ss", "Æ": "AE", "æ": "ae", "ı": "i", "Ł": "L", "ł": "l",
                            "Đ": "D", "đ": "d", "’": "'"})


def ascii_fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(FOLD_EXTRA))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def feature_label(name: str) -> str:
    if name in FIXED_LABELS:
        return FIXED_LABELS[name]
    opp = re.fullmatch(r"opp_conceded_roll(\d+)", name)
    if opp:
        return f"Opponent goals conceded per match, last {opp.group(1)}"
    m = re.fullmatch(r"([a-z]+?)(_p90)?_roll(\d+)", name)
    if m and m.group(1) in STAT_LABELS:
        stat, p90, n = m.groups()
        return f"{STAT_LABELS[stat]} per {'90' if p90 else 'match'}, last {n}"
    return name


def format_value(name: str, value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if name in BINARY_FEATURES:
        return "yes" if value >= 0.5 else "no"
    if name in COUNT_FEATURES:
        return f"{int(value)}"
    if name == "player_prior_rate":
        return f"{value:.0%}"
    if name == "days_since_last_match":
        return f"{value:.0f} days"
    return f"{value:.2f}"


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Loading data...")
def load_frame(path: str, stamp: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


@st.cache_data
def load_json(path: str, stamp: float) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str, stamp: float):
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


@st.cache_resource(show_spinner="Building SHAP explainer...")
def load_explainer(path: str, stamp: float):
    import shap

    return shap.TreeExplainer(load_model(path, stamp))


@st.cache_data
def player_table(path: str, stamp: float, test_season: str, history_seasons: tuple) -> pd.DataFrame:
    features = load_frame(path, stamp)
    latest = features.sort_values(["date", "fixture_id"]).groupby("player_id").tail(1).set_index("player_id")
    players = latest[["player_name", "team", "position"]].copy()
    players["folded"] = players["player_name"].map(ascii_fold)
    alias = players["folded"].where(players["folded"] != players["player_name"])
    players["label"] = (
        players["player_name"] + alias.map(lambda a: f" [{a}]" if isinstance(a, str) else "")
        + " (" + players["team"] + ", " + players["position"] + ")"
    )
    dup = players["label"].duplicated(keep=False)
    players.loc[dup, "label"] = players.loc[dup, "label"] + " #" + players.index[dup].astype(str)
    test = features[features["season"] == test_season]
    players["test_rows"] = test.groupby("player_id").size().reindex(players.index).fillna(0).astype(int)
    hist = features[features["season"].isin(history_seasons)]
    players["hist_ga"] = (
        (hist["goals"] + hist["assists"]).groupby(hist["player_id"]).sum().reindex(players.index).fillna(0)
    )
    return players.sort_values("label", key=lambda s: s.map(ascii_fold).str.casefold())


def default_player(players: pd.DataFrame):
    """FWD with the most goals + assists before the test season, among players with test-season rows."""
    pool = players[(players["position"] == "FWD") & (players["test_rows"] > 0)]
    if pool.empty:
        pool = players
    return pool.sort_index().sort_values(["hist_ga", "test_rows"], ascending=False, kind="stable").index[0]


def match_label(row: pd.Series) -> str:
    return (
        f"{row['date']:%Y-%m-%d} · vs {row['opponent']} ({row['home_away']}) · "
        f"{row['season']} GW{int(row['matchweek'])}"
    )


def gauge(prob: float, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": TOWARDS if prob >= threshold else NEUTRAL},
            "steps": [{"range": [0, threshold * 100], "color": "rgba(158,158,158,0.15)"}],
            "threshold": {"line": {"color": AWAY, "width": 3}, "thickness": 0.85, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=230, margin={"l": 25, "r": 25, "t": 20, "b": 10})
    return fig


def explain(model_path: str, model_mtime: float, x: pd.DataFrame):
    """Return (per-feature SHAP series in log-odds, base value) for one row; raises if SHAP fails."""
    explainer = load_explainer(model_path, model_mtime)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values).reshape(-1)
    base = float(np.ravel(explainer.expected_value)[-1])
    return pd.Series(values, index=x.columns), base


def contribution_chart(top: pd.DataFrame) -> go.Figure:
    top = top.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["shap"],
        y=top["label"] + "  (" + top["value"] + ")",
        orientation="h",
        marker_color=[TOWARDS if v > 0 else AWAY for v in top["shap"]],
        text=[f"{v:+.3f}" for v in top["shap"]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>contribution %{x:+.3f} log-odds<extra></extra>",
    ))
    fig.update_layout(
        height=60 + 36 * len(top), margin={"l": 10, "r": 50, "t": 10, "b": 40},
        xaxis_title="Contribution (log-odds): ← away from G/A · towards G/A →",
    )
    fig.add_vline(x=0, line_color="#666", line_width=1)
    return fig


def form_chart(last: pd.DataFrame) -> go.Figure:
    last = last.iloc[::-1]
    x = [f"{d:%d %b %y}<br>{o} ({h})" for d, o, h in zip(last["date"], last["opponent"], last["home_away"])]
    fig = go.Figure()
    for col, color in [("goals", TOWARDS), ("assists", "#67a9cf")]:
        fig.add_bar(x=x, y=last[col], name=col.capitalize(), marker_color=color,
                    customdata=last["minutes"], hovertemplate="%{y} " + col + " in %{customdata} min<extra></extra>")
    fig.update_layout(barmode="stack", height=260, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      yaxis={"title": "Count", "dtick": 1, "rangemode": "tozero"},
                      legend={"orientation": "h", "y": 1.15})
    return fig


def model_note(metrics: dict | None, sel: dict, variant: str) -> str:
    train = sel.get("train_seasons") or []
    span = f"{train[0]} to {sel.get('val_season')}" if train else str(sel.get("val_season"))
    if not metrics:
        return f"Model trained on seasons {span}; test-season metrics unavailable (run `python -m src.evaluate`)."
    m = metrics.get("models", {}).get(variant) or metrics.get("models", {}).get("xgb") or {}
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"  # noqa: E731
    return (
        f"Model trained on seasons {span}; test-season ({metrics.get('test_season')}) "
        f"ROC-AUC = {fmt(m.get('roc_auc'))}, PR-AUC = {fmt(m.get('pr_auc'))}."
    )


def model_caveats(ctx) -> None:
    if ctx.sel.get("xgb_scale_pos_weight") is not None:
        st.caption(
            f"This XGBoost model was trained with class weighting (scale_pos_weight = "
            f"{float(ctx.sel['xgb_scale_pos_weight']):.2f}), so its probabilities are inflated relative to real "
            "frequencies. Compare them with the threshold rather than reading them as literal chances."
        )
    if ctx.metrics and ctx.metrics.get("selected_model") not in (None, ctx.variant):
        st.caption(
            f"Note: the model selected on validation was `{ctx.metrics['selected_model']}`; the app shows the "
            f"XGBoost model (`{ctx.variant}`) because it supports per-prediction SHAP explanations."
        )


def render_shap(ctx, x: pd.DataFrame) -> None:
    cols = ctx.feature_cols
    try:
        shap_values, base = explain(ctx.model_path, ctx.model_mtime, x)
        table = pd.DataFrame({
            "feature": cols,
            "label": [feature_label(c) for c in cols],
            "value": [format_value(c, x.iloc[0][c]) for c in cols],
            "shap": shap_values.reindex(cols).to_numpy(),
        })
        table = table.reindex(table["shap"].abs().sort_values(ascending=False).index)
        top, rest = table.head(8), table.iloc[8:]
        st.plotly_chart(contribution_chart(top), width="stretch", config={"displayModeBar": False})
        logit = base + table["shap"].sum()
        st.caption(
            f"SHAP contributions in log-odds. The base value {base:+.3f} (≈ {sigmoid(base):.1%}) is the model's "
            f"average output; blue bars push this prediction towards a goal or assist, red bars away from it. "
            f"Base + all {len(table)} contributions = {logit:+.3f} log-odds = {sigmoid(logit):.1%}. "
            f"The other {len(rest)} features add {rest['shap'].sum():+.3f} together. Values in brackets are this "
            "row's feature values (n/a = not available, e.g. no xG before 2022-23)."
        )
        with st.expander("All feature contributions"):
            st.dataframe(
                table.rename(columns={"label": "Feature", "value": "Value", "shap": "SHAP (log-odds)",
                                      "feature": "Column"}),
                hide_index=True, width="stretch",
                column_config={"SHAP (log-odds)": st.column_config.NumberColumn(format="%+.3f")},
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"SHAP explanation failed ({type(exc).__name__}: {exc}). Showing the model's global gain importance "
            "instead; it is not specific to this prediction."
        )
        gain = pd.Series(ctx.model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
        gain = gain.head(8)
        st.dataframe(
            pd.DataFrame({
                "Feature": [feature_label(c) for c in gain.index],
                "Value": [format_value(c, x.iloc[0][c]) if c in x.columns else "n/a" for c in gain.index],
                "Gain": gain.to_numpy(),
            }),
            hide_index=True, width="stretch",
        )


def render_last10(ctx, pid, before: pd.Timestamp, keys=("clean",)) -> None:
    if not ctx.paths["clean"].exists():
        st.error(
            f"`{ctx.cfg['paths']['clean']}` is missing, so recent appearances cannot be shown. Run `python -m src.clean`."
        )
        return
    if "current_clean" in keys and not ctx.paths["current_clean"].exists():
        st.warning(
            f"Current-season appearances unavailable (`{ctx.cfg['paths']['current_clean']}` is missing; run "
            "`python -m src.upcoming`). The list shows earlier seasons only."
        )
    frames = [load_frame(str(ctx.paths[k]), mtime(ctx.paths[k])) for k in keys if ctx.paths[k].exists()]
    hist = pd.concat([f[(f["player_id"] == pid) & (f["date"] < before)] for f in frames])
    last = hist.sort_values("date", ascending=False).head(10)
    if last.empty:
        st.write("No earlier appearances in the data (this is the player's first recorded match).")
        return
    t_col, c_col = st.columns([1, 1.2])
    with t_col:
        show = last[["date", "opponent", "home_away", "minutes", "goals", "assists"]].rename(columns={
            "date": "Date", "opponent": "Opponent", "home_away": "H/A", "minutes": "Min",
            "goals": "Goals", "assists": "Assists",
        })
        show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(show, hide_index=True, width="stretch")
    with c_col:
        st.plotly_chart(form_chart(last), width="stretch", config={"displayModeBar": False})
    st.caption(
        f"{int(last['goals'].sum())} goals and {int(last['assists'].sum())} assists in "
        f"{int(last['minutes'].sum())} minutes over these {len(last)} appearances "
        f"(includes short cameos under {ctx.min_minutes} minutes)."
    )


def historical_view(ctx) -> None:
    features, feature_cols, threshold = ctx.features, ctx.feature_cols, ctx.threshold
    train_seasons, val_season, test_season = ctx.train_seasons, ctx.val_season, ctx.test_season
    players = player_table(
        str(ctx.paths["features"]), mtime(ctx.paths["features"]), test_season, tuple(train_seasons) + (val_season,)
    )
    ids = players.index.tolist()
    left, right = st.columns(2)
    with left:
        pid = st.selectbox(
            "Player", ids, index=ids.index(default_player(players)), key="player",
            format_func=lambda i: players.at[i, "label"], placeholder="Type to search...",
        )
    rows = features[features["player_id"] == pid].sort_values(["date", "fixture_id"], ascending=False)
    with right:
        ridx = st.selectbox(
            "Match (newest first)", rows.index.tolist(), key=f"match_{pid}",
            format_func=lambda i: match_label(rows.loc[i]),
        )
    row = rows.loc[ridx]
    x = rows.loc[[ridx], feature_cols].astype(float)
    prob = float(ctx.model.predict_proba(x)[0, 1])
    call = prob >= threshold

    st.divider()
    in_sample = row["season"] in train_seasons or row["season"] == val_season
    if in_sample:
        st.info(
            f"In-sample: this {row['season']} match was in the model's training data "
            f"(seasons {train_seasons[0]} to {val_season}), so the prediction is likely optimistic."
        )
    elif row["season"] == test_season:
        st.success(f"Out-of-sample: {test_season} is the held-out test season, never used for training or tuning.")

    g_col, p_col, a_col = st.columns([1.3, 1, 1])
    with g_col:
        st.plotly_chart(gauge(prob, threshold), width="stretch", config={"displayModeBar": False})
    with p_col:
        st.metric("P(goal or assist)", f"{prob:.1%}")
        st.metric("Model call", "Yes" if call else "No", help="Yes if the probability is at or above the threshold.")
        st.caption(f"Frozen threshold {threshold:.1%} (F1-optimal on {val_season}, shown as the red line).")
    with a_col:
        actual = int(row[ctx.target]) == 1
        st.metric("Actual outcome", "Goal or assist" if actual else "No goal or assist")
        st.caption(
            f"{int(row['goals'])} goal(s), {int(row['assists'])} assist(s) in {int(row['minutes'])} minutes "
            f"for {row['team']}. The call was {'right' if call == actual else 'wrong'}."
        )
        if ctx.metrics and ctx.metrics.get("test_base_rate") is not None:
            st.caption(f"For reference, {ctx.metrics['test_base_rate']:.0%} of {test_season} rows had a goal or assist.")
    model_caveats(ctx)

    st.subheader("Why this prediction")
    render_shap(ctx, x)
    st.subheader("Last 10 appearances before this match")
    render_last10(ctx, pid, row["date"])


def main():
    st.set_page_config(page_title="Goal-or-assist predictor", layout="wide")
    cfg = load_config()
    paths = {k: resolve(cfg["paths"][k]) for k in PATH_KEYS}
    target, min_minutes = cfg["features"]["target"], cfg["cleaning"]["min_minutes"]

    st.title("Premier League goal-or-assist predictor")
    st.write(
        "The model's pre-match probability that a player scores or assists, given they play at least "
        f"{min_minutes} minutes: for the next gameweek's fixtures, or for any past match in the data."
    )

    missing = [k for k in ["xgb_model", "model_selection", "features"] if not paths[k].exists()]
    if missing:
        st.error(
            "Missing required artifact(s): "
            + ", ".join(f"`{paths[k].relative_to(ROOT).as_posix()}`" for k in missing)
            + ". Run the pipeline first: `python -m src.features`, `python -m src.train`, `python -m src.evaluate`."
        )
        st.stop()

    sel = load_json(str(paths["model_selection"]), mtime(paths["model_selection"]))
    metrics = load_json(str(paths["metrics"]), mtime(paths["metrics"])) if paths["metrics"].exists() else None
    variant = sel.get("xgb_variant", "xgb")
    model_path, model_mtime = str(paths["xgb_model"]), mtime(paths["xgb_model"])
    try:
        model = load_model(model_path, model_mtime)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load `{cfg['paths']['xgb_model']}`: {exc}")
        st.stop()

    features = load_frame(str(paths["features"]), mtime(paths["features"]))
    feature_cols = model.get_booster().feature_names or sel.get("feature_columns") or cfg["features"]["feature_columns"]
    needed = set(feature_cols) | set(cfg["features"]["id_columns"]) | {target}
    absent = sorted(needed - set(features.columns))
    if absent:
        st.error(f"`{cfg['paths']['features']}` is missing columns: {', '.join(absent)}. Re-run `python -m src.features`.")
        st.stop()

    ctx = SimpleNamespace(
        cfg=cfg, paths=paths, sel=sel, metrics=metrics, variant=variant, model=model, model_path=model_path,
        model_mtime=model_mtime, features=features, feature_cols=list(feature_cols), target=target,
        min_minutes=min_minutes,
        train_seasons=sel.get("train_seasons", cfg["split"]["train_seasons"]),
        val_season=sel.get("val_season", cfg["split"]["val_season"]),
        test_season=sel.get("test_season", cfg["split"]["test_season"]),
        threshold=float(sel.get("thresholds", {}).get(variant, (metrics or {}).get("threshold", 0.5))),
        load_frame=load_frame, load_json=load_json, mtime=mtime, ascii_fold=ascii_fold, gauge=gauge,
        render_shap=render_shap, render_last10=render_last10, model_caveats=model_caveats,
    )

    st.caption(model_note(metrics, sel, variant) + " Not betting or fantasy advice.")
    if metrics is None:
        st.error("`reports/metrics.json` is missing, so test-season metrics are not shown. Run `python -m src.evaluate`.")

    mode = st.radio(
        "View", MODES, index=0 if paths["upcoming"].exists() else 1, horizontal=True, key="mode",
        label_visibility="collapsed",
    )
    if mode == MODES[0]:
        upcoming_view.render(ctx)
    else:
        historical_view(ctx)

    st.divider()
    st.caption(
        "Data: official Fantasy Premier League statistics (vaastav/Fantasy-Premier-League archive, plus the live "
        "FPL API for the current season). Assists follow FPL's definition. Not betting or fantasy advice."
    )


main()
