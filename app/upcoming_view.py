"""Upcoming-gameweek view: leaderboard, filters and a per-player explanation for the next FPL gameweek."""
import numpy as np
import pandas as pd
import streamlit as st

STATUS = {"a": "Available", "d": "Doubtful", "i": "Injured", "s": "Suspended", "u": "Unavailable", "n": "Not in squad"}
POSITIONS = ["DEF", "MID", "FWD"]
REFRESH = "`python -m src.live --refresh` then `python -m src.upcoming`"
CANDIDATE_COLUMNS = ["player_id", "player_name", "date", "fixture_id", "team", "opponent", "home_away", "position",
                     "prob", "call", "likely_starter", "status", "chance_of_playing_next_round", "news"]


def _ts(value):
    ts = pd.to_datetime(value, utc=True, errors="coerce") if value else None
    return None if ts is None or pd.isna(ts) else ts


def _fmt(ts, fmt="%a %d %b %Y, %H:%M UTC") -> str:
    return ts.strftime(fmt) if ts is not None else "unknown"


def availability(row: pd.Series) -> str:
    status = str(row["status"])
    label = STATUS.get(status, status)
    chance, news = row["chance_of_playing_next_round"], row["news"]
    news = news.strip() if isinstance(news, str) else ""
    if status == "a" and not news:
        return label
    text = label + (f" ({chance:.0f}%)" if pd.notna(chance) else "")
    return f"{text}: {news}" if news else text


def header(ctx, meta: dict, up: pd.DataFrame) -> None:
    gw = meta.get("target_gw", meta.get("gameweek", "?"))
    remaining = meta.get("target_mode", "next_gw") == "current_gw_remaining"
    max_age = ctx.cfg["upcoming"]["snapshot_max_age_hours"]
    deadline, generated = _ts(meta.get("deadline")), _ts(meta.get("generated_at"))
    snapshots = [t for t in (generated, _ts(meta.get("live_fetched_at"))) if t is not None]
    title = f"Gameweek {gw}, remaining fixtures" if remaining else f"Gameweek {gw}"
    st.subheader(f"{title} ({meta.get('season', ctx.cfg['upcoming']['season'])})")
    st.markdown(
        f"**Deadline** {'passed' if remaining and deadline is None else _fmt(deadline)} · **Data through** {_fmt(_ts(meta.get('data_through')), '%a %d %b %Y')} "
        f"· **Generated** {_fmt(generated, '%d %b %Y, %H:%M UTC')}"
    )
    now, stale = pd.Timestamp.now(tz="UTC"), []
    if remaining:
        if not up.empty and up["date"].max() <= now:
            stale.append(f"every listed GW{gw} fixture has already kicked off")
    elif deadline is not None and now > deadline:
        stale.append(f"the GW{gw} deadline has passed")
    if snapshots and now - min(snapshots) > pd.Timedelta(hours=max_age):
        hours = (now - min(snapshots)) / pd.Timedelta(hours=1)
        stale.append(f"the data snapshot is {hours:.0f} hours old (limit {max_age} h), so team news may have changed")
    if not meta:
        stale.append(f"`{ctx.cfg['paths']['upcoming_meta']}` is missing, so freshness cannot be checked")
    if stale:
        st.warning("These predictions may be stale: " + "; ".join(stale) + f". Refresh with {REFRESH}.")
    pending = int(meta.get("pending_fixtures") or 0)
    if pending > 0:
        st.warning(
            f"{pending} earlier-gameweek fixture{'s are' if pending > 1 else ' is'} not finished yet, so the recent "
            "form of the teams involved may be incomplete."
        )


def caveat(ctx) -> None:
    bands = (ctx.metrics or {}).get("by_minutes_band") or {}
    key = next((k for k in bands if str(k).split("-")[0] == str(ctx.min_minutes)), None)
    text = (
        f"**Read this first.** Each probability assumes the player plays at least {ctx.min_minutes} minutes; "
        "it does not include the risk of being benched or substituted early."
    )
    band = bands.get(key) or {}
    if band.get("base_rate") and band.get("mean_predicted") is not None:
        text += (
            f" On the {ctx.test_season} test season, players who ended up playing {key} minutes were over-predicted "
            f"about {band['mean_predicted'] / band['base_rate']:.1f}x (average prediction "
            f"{band['mean_predicted']:.1%} vs {band['base_rate']:.1%} actual), so treat rotation risks with care."
        )
    st.warning(text + f" {ctx.season} fixtures are out-of-sample: the model was trained on {ctx.train_seasons[0]} to "
               f"{ctx.val_season} and has not been refit. Not betting or fantasy advice.")


def prepare(ctx, up: pd.DataFrame) -> pd.DataFrame:
    up = up.sort_values(["prob", "player_id", "fixture_id"], ascending=[False, True, True], kind="stable")
    up = up.reset_index(drop=True)
    up["rank"] = np.arange(1, len(up) + 1)
    up["likely_starter"] = up["likely_starter"].fillna(False).astype(bool)
    up["call"] = up["call"].fillna(False).astype(bool)
    up["availability"] = up.apply(availability, axis=1)
    up["folded"] = up["player_name"].map(ctx.ascii_fold)
    up["search"] = up["folded"].str.casefold()
    return up


def filters(up: pd.DataFrame, fold) -> pd.DataFrame:
    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.6, 1], vertical_alignment="bottom")
    with c1:
        positions = st.multiselect("Position", POSITIONS, default=POSITIONS, key="up_pos")
    with c2:
        teams = st.multiselect("Team", sorted(up["team"].unique()), key="up_team", placeholder="All teams")
    with c3:
        query = st.text_input("Search player", key="up_search", placeholder="e.g. Odegaard")
    with c4:
        starters = st.toggle("Likely starters only", value=True, key="up_starters")
    view = up[up["position"].isin(positions)]
    if teams:
        view = view[view["team"].isin(teams)]
    if starters:
        view = view[view["likely_starter"]]
    if query.strip():
        view = view[view["search"].str.contains(fold(query.strip()).casefold(), regex=False)]
    return view


def leaderboard(ctx, view: pd.DataFrame, total: int) -> None:
    table = pd.DataFrame({
        "Rank": view["rank"], "Player": view["player_name"], "Team": view["team"], "Pos": view["position"],
        "Opponent": view["opponent"] + " (" + view["home_away"] + ")",
        "Kickoff (UTC)": view["date"].dt.strftime("%a %d %b %H:%M"),
        "P(G/A)": view["prob"] * 100, "Call": np.where(view["call"], "Yes", "No"),
        "Likely starter": view["likely_starter"], "Prior apps": view["prior_matches"],
        "Availability": view["availability"],
    })
    st.dataframe(
        table, hide_index=True, width="stretch", height=min(430, 38 + 35 * max(len(table), 1)),
        column_config={
            "P(G/A)": st.column_config.ProgressColumn("P(G/A)", format="%.1f%%", min_value=0, max_value=100),
            "Rank": st.column_config.NumberColumn(help=f"Rank by probability among all {total} candidates"),
            "Prior apps": st.column_config.NumberColumn(format="%d", help="Premier League appearances in the data"),
            "Likely starter": st.column_config.CheckboxColumn(
                help=f"Played {ctx.min_minutes}+ minutes in at least {ctx.cfg['upcoming']['likely_starter_min_matches']}"
                     f" of the team's last {ctx.cfg['upcoming']['likely_starter_lookback']} matches"),
        },
    )
    st.caption(
        f"Showing {len(view)} of {total} player-fixtures. Call = Yes when the probability is at or above the "
        f"frozen threshold of {ctx.threshold:.1%}. Rank is across all {total} candidates."
    )


def details(ctx, view: pd.DataFrame) -> None:
    def label(i):
        r = view.loc[i]
        alias = f" [{r['folded']}]" if r["folded"] != r["player_name"] else ""
        return f"#{r['rank']} {r['player_name']}{alias} ({r['team']}, {r['position']}) vs {r['opponent']} ({r['home_away']})"

    ridx = st.selectbox("Player details", view.index.tolist(), key="up_row", format_func=label)
    row = view.loc[ridx]
    x = view.loc[[ridx], ctx.feature_cols].astype(float)
    prob = float(ctx.model.predict_proba(x)[0, 1])
    if abs(prob - float(row["prob"])) > 1e-4:
        st.warning(
            f"`{ctx.cfg['paths']['upcoming']}` was scored with a different model file ({float(row['prob']):.1%} "
            f"stored vs {prob:.1%} now). Re-run `python -m src.upcoming`."
        )
    g_col, p_col, f_col = st.columns([1.3, 1, 1])
    with g_col:
        st.plotly_chart(ctx.gauge(prob, ctx.threshold), width="stretch", config={"displayModeBar": False})
    with p_col:
        st.metric("P(goal or assist)", f"{prob:.1%}")
        st.metric("Model call", "Yes" if prob >= ctx.threshold else "No",
                  help="Yes if the probability is at or above the threshold.")
        st.caption(f"Frozen threshold {ctx.threshold:.1%} (F1-optimal on {ctx.val_season}, shown as the red line).")
    with f_col:
        venue = "home" if row["home_away"] == "H" else "away"
        st.metric("Fixture", f"vs {row['opponent']} ({row['home_away']})")
        st.caption(f"{row['team']} {venue}, kickoff {row['date']:%a %d %b, %H:%M} UTC.")
        st.markdown(
            f"**Likely starter:** {'yes' if row['likely_starter'] else 'no'}  \n"
            f"**Availability:** {row['availability']}  \n"
            f"**Premier League appearances in the data:** {int(row['prior_matches'])}"
        )
        if row["prior_matches"] < ctx.cfg["features"]["min_prior_matches"]:
            st.caption("Little Premier League history, so most form features are missing: treat with extra caution.")
    ctx.model_caveats(ctx)
    st.subheader("Why this prediction")
    ctx.render_shap(ctx, x)
    st.subheader("Last 10 appearances")
    ctx.render_last10(ctx, row["player_id"], row["date"], keys=("clean", "current_clean"))


def render(ctx) -> None:
    ctx.season = ctx.cfg["upcoming"]["season"]
    paths = ctx.paths
    if not paths["upcoming"].exists():
        st.info(
            "No upcoming-gameweek predictions yet. Fetch the live FPL data and score the next gameweek with "
            "`python -m src.live` and then `python -m src.upcoming`, or switch to **Historical match**."
        )
        return
    up = ctx.load_frame(str(paths["upcoming"]), ctx.mtime(paths["upcoming"]))
    meta = ctx.load_json(str(paths["upcoming_meta"]), ctx.mtime(paths["upcoming_meta"])) \
        if paths["upcoming_meta"].exists() else {}
    absent = sorted(set(CANDIDATE_COLUMNS + ctx.feature_cols) - set(up.columns))
    if absent:
        st.error(f"`{ctx.cfg['paths']['upcoming']}` is missing columns: {', '.join(absent)}. "
                 "Re-run `python -m src.upcoming`.")
        return
    ctx.season = meta.get("season", ctx.season)
    header(ctx, meta, up)
    caveat(ctx)
    if up.empty:
        st.info(f"No upcoming fixtures to predict. Refresh with {REFRESH}.")
        return
    up = prepare(ctx, up)
    view = filters(up, ctx.ascii_fold)
    leaderboard(ctx, view, len(up))
    if view.empty:
        st.info("No players match these filters.")
        return
    details(ctx, view)
