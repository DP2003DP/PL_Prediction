"""Top-k ranking checks and matchweek cluster-bootstrap intervals, used by src.evaluate."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
CHUNK = 200
MINUTES_BANDS = (("30-59", 30, 59), ("60-89", 60, 89), ("90+", 90, np.inf))


def _floats(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype="float64", na_value=np.nan)


def matchweek_pool(test: pd.DataFrame, target: str, p: np.ndarray) -> pd.DataFrame:
    """One row per (matchweek, player): the player's highest-probability fixture in that matchweek."""
    frame = pd.DataFrame({
        "mw": test["matchweek"].to_numpy(), "player_id": test["player_id"].to_numpy(),
        "position": test["position"].to_numpy(), "prior": _floats(test["player_prior_rate"]),
        "y": test[target].to_numpy(dtype=int), "p": np.asarray(p, dtype=np.float64),
    })
    frame = frame.sort_values(["mw", "player_id", "p"], ascending=[True, True, False])
    return frame.drop_duplicates(["mw", "player_id"], keep="first").reset_index(drop=True)


def _top_k_rate(pool: pd.DataFrame, score: str, k: int) -> pd.Series:
    ranked = pool.sort_values(["mw", score, "player_id"], ascending=[True, False, True], na_position="last")
    return ranked[ranked.groupby("mw").cumcount() < k].groupby("mw")["y"].mean()


def top_k_check(test: pd.DataFrame, target: str, p: np.ndarray, k: int) -> tuple[dict, pd.DataFrame]:
    """Mean over matchweeks of the top-k hit rate for the model and each comparator, on the deduped pool."""
    pool = matchweek_pool(test, target, p)
    rates = pd.DataFrame({
        "model": _top_k_rate(pool, "p", k),
        "random": pool.groupby("mw")["y"].mean(),
        "random_fwd": pool[pool["position"] == "FWD"].groupby("mw")["y"].mean(),
        "prior_rate_ranker": _top_k_rate(pool, "prior", k),
    }).sort_index()
    mean = rates.mean()
    model = float(mean["model"])

    def lift(col: str) -> float | None:
        return model / float(mean[col]) if mean[col] > 0 else None

    summary = {
        "k": k, "n_matchweeks": int(len(rates)),
        "pool": "one row per player per matchweek (the fixture with the highest model probability)",
        "n_rows_deduped": int(len(test) - len(pool)),
        "model_hit_rate": model, "random_hit_rate": float(mean["random"]),
        "random_fwd_hit_rate": float(mean["random_fwd"]),
        "prior_rate_ranker_hit_rate": float(mean["prior_rate_ranker"]),
        "lift": lift("random"), "lift_vs_random_fwd": lift("random_fwd"),
        "lift_vs_prior_rate_ranker": lift("prior_rate_ranker"),
        "lift_definition": "model_hit_rate / comparator hit rate; `lift` is versus random_hit_rate",
        "prior_rate_ranker": "top k by player_prior_rate, NaN last, ties broken by player_id",
    }
    return summary, rates


def minutes_band_metrics(test: pd.DataFrame, target: str, p: np.ndarray) -> dict:
    """Post-match diagnostic: how the model does by minutes actually played (not a feature)."""
    minutes, y = _floats(test["minutes"]), test[target].to_numpy(dtype=int)
    out = {}
    for label, low, high in MINUTES_BANDS:
        mask = (minutes >= low) & (minutes <= high)
        yb, pb = y[mask], p[mask]
        both = len(np.unique(yb)) == 2
        out[label] = {"n": int(mask.sum()), "base_rate": float(yb.mean()) if mask.any() else None,
                      "mean_predicted": float(pb.mean()) if mask.any() else None,
                      "pr_auc": average_precision_score(yb, pb) if both else None,
                      "roc_auc": roc_auc_score(yb, pb) if both else None}
    return out


def cluster_counts(n_clusters: int, n_reps: int, seed: int) -> np.ndarray:
    """(n_reps, n_clusters) matrix: how many times each cluster is drawn in each resample."""
    draws = np.random.default_rng(seed).integers(0, n_clusters, size=(n_reps, n_clusters))
    counts = np.zeros((n_reps, n_clusters))
    np.add.at(counts, (np.repeat(np.arange(n_reps), n_clusters), draws.ravel()), 1)
    return counts


def weighted_aucs(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC-AUC and average precision, computed as sklearn does, for each row of `weights` (reps x rows)."""
    order = np.argsort(-p, kind="mergesort")
    y, p, w = y[order].astype(float), p[order], weights[:, order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(w * y, axis=1)[:, last]
    fp = np.cumsum(w * (1.0 - y), axis=1)[:, last]
    zeros = np.zeros((len(w), 1))
    tp0, fp0 = np.hstack([zeros, tp]), np.hstack([zeros, fp])
    pos, neg = tp[:, -1], fp[:, -1]
    roc = np.sum(np.diff(fp0, axis=1) * (tp0[:, 1:] + tp0[:, :-1]), axis=1) / (2.0 * pos * neg)
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    ap = np.sum(np.diff(tp0, axis=1) * precision, axis=1) / pos
    return roc, ap


def _interval(estimate: float, samples: np.ndarray) -> dict:
    tail = 100 * (1 - CI_LEVEL) / 2
    low, high = np.percentile(samples, [tail, 100 - tail])
    return {"estimate": float(estimate), "ci_low": float(low), "ci_high": float(high)}


def bootstrap(y: np.ndarray, matchweek: np.ndarray, probs: dict, xgb_name: str, rates: pd.DataFrame,
              top_k_model: str, seed: int, n_reps: int = N_BOOTSTRAP) -> dict:
    """Percentile CIs from resampling test matchweeks with replacement (same resamples for every model)."""
    codes, clusters = pd.factorize(pd.Series(matchweek), sort=True)
    counts = cluster_counts(len(clusters), n_reps, seed)
    ones = np.ones((1, len(y)))
    point, draws = {}, {}
    for name, p in probs.items():
        roc, ap = weighted_aucs(y, p, ones)
        point[name] = {"roc_auc": roc[0], "pr_auc": ap[0]}
        chunks = [weighted_aucs(y, p, counts[i:i + CHUNK][:, codes]) for i in range(0, n_reps, CHUNK)]
        draws[name] = {"roc_auc": np.concatenate([c[0] for c in chunks]),
                       "pr_auc": np.concatenate([c[1] for c in chunks])}

    metrics = ("pr_auc", "roc_auc")
    models = {name: {m: _interval(point[name][m], draws[name][m]) for m in metrics} for name in probs}
    paired = {f"{xgb_name} - {other}": {m: _interval(point[xgb_name][m] - point[other][m],
                                                     draws[xgb_name][m] - draws[other][m]) for m in metrics}
              for other in probs if other != xgb_name}

    rates = rates.reindex(clusters)
    n = len(clusters)
    model_rate = counts @ rates["model"].to_numpy() / n
    random_rate = counts @ rates["random"].to_numpy() / n
    top_k = {"model": top_k_model,
             "model_hit_rate": _interval(rates["model"].mean(), model_rate),
             "lift_vs_random": _interval(rates["model"].mean() / rates["random"].mean(), model_rate / random_rate)}
    return {"method": "cluster bootstrap over test matchweeks (resampled with replacement); percentile intervals",
            "cluster_unit": "matchweek", "n_clusters": int(n), "n_reps": int(n_reps), "seed": int(seed),
            "ci_level": CI_LEVEL, "models": models, "paired_differences": paired, "top_k": top_k}


def _ci(stat: dict, digits: int = 3, signed: bool = False) -> tuple[str, str]:
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(stat["estimate"]), f"[{fmt.format(stat['ci_low'])}, {fmt.format(stat['ci_high'])}]"


def minutes_band_markdown(bands: dict, selected: str) -> list[str]:
    def fmt(value) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [f"### Diagnostic: by minutes played ({selected})", "",
             "Diagnostic only: minutes are known only after the match, so they are not a model feature and nothing "
             "here is tuned. It shows where conditioning on 30+ minutes matters: the model does not know whether a "
             "player will come off the bench or play the full match.", "",
             "| Minutes played | n | Base rate | Mean predicted | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|"]
    for label, m in bands.items():
        lines.append(f"| {label} | {m['n']:,} | {fmt(m['base_rate'])} | {fmt(m['mean_predicted'])} | "
                     f"{fmt(m['roc_auc'])} | {fmt(m['pr_auc'])} |")
    return lines + [""]


def top_k_markdown(tk: dict, selected: str) -> list[str]:
    rows = [(f"Model top {tk['k']} ({selected})", tk["model_hit_rate"], None),
            (f"Top {tk['k']} by player_prior_rate", tk["prior_rate_ranker_hit_rate"], tk["lift_vs_prior_rate_ranker"]),
            ("Random forward", tk["random_fwd_hit_rate"], tk["lift_vs_random_fwd"]),
            ("Random player", tk["random_hit_rate"], tk["lift"])]
    lines = [f"### Top-{tk['k']} check ({selected})", "",
             f"Each matchweek, the top {tk['k']} players are compared with simpler pickers on the same pool: one row "
             "per player per matchweek (for a player with two fixtures in a matchweek, the higher-probability "
             f"fixture). Hit rate = share of picks with a goal or assist, averaged over {tk['n_matchweeks']} "
             "matchweeks.", "",
             "| Picker | Hit rate | Model lift over this picker |", "|---|---|---|"]
    for label, rate, lift in rows:
        lines.append(f"| {label} | {rate:.3f} | {'-' if lift is None else f'{lift:.2f}x'} |")
    return lines + [""]


def bootstrap_markdown(bs: dict) -> list[str]:
    lines = ["### Uncertainty (95% CI, matchweek bootstrap)", "",
             f"{bs['n_reps']:,} resamples of the {bs['n_clusters']} test matchweeks with replacement (seed "
             f"{bs['seed']}); percentile intervals. Differences use the same resamples for both models.", "",
             "| Quantity | Estimate | 95% CI |", "|---|---|---|"]
    for name, stats in bs["models"].items():
        for metric, label in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")):
            lines.append(f"| {name} {label} | {' | '.join(_ci(stats[metric]))} |")
    for pair, stats in bs["paired_differences"].items():
        for metric, label in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")):
            lines.append(f"| {pair} {label} | {' | '.join(_ci(stats[metric], 4, signed=True))} |")
    tk = bs["top_k"]
    lines.append(f"| {tk['model']} top-k hit rate | {' | '.join(_ci(tk['model_hit_rate']))} |")
    est, ci = _ci(tk["lift_vs_random"], 2)
    lines.append(f"| {tk['model']} top-k lift vs random | {est}x | {ci} |")
    return lines + [""]
