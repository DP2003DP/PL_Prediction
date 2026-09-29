"""Evaluate the frozen models once on the held-out test season.

Run as `python -m src.evaluate`. Scores the test-season rows of features.parquet with the models and
thresholds frozen by src.train (nothing is tuned here) and writes reports/metrics.json,
reports/metrics_table.md, reports/test_predictions.parquet and the figures in reports/figures/.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier  # noqa: E402

from src.eval_stats import (  # noqa: E402
    bootstrap, bootstrap_markdown, minutes_band_markdown, minutes_band_metrics, top_k_check, top_k_markdown,
)
from src.split import chronological_split  # noqa: E402
from src.utils import get_logger, load_config, resolve  # noqa: E402

log = get_logger(__name__)

POSITIONS = ["FWD", "MID", "DEF"]
SELECTION_KEYS = ["selected_model", "xgb_variant", "thresholds", "feature_columns", "logreg_features",
                  "train_seasons", "val_season", "test_season"]
POSITION_KEYS = ["accuracy", "lift_over_baseline", "precision", "recall", "f1", "roc_auc", "pr_auc"]

INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"logreg": "#2a78d6", "rf": "#eb6834", "xgb": "#1baf7a"}
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlesize": 11, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "legend.frameon": False, "legend.fontsize": 9,
}


def load_selection(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model selection not found: {path}. Run `python -m src.train` first.")
    selection = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in SELECTION_KEYS if k not in selection]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    return selection


def load_models(cfg: dict, selection: dict) -> dict:
    paths = {"logreg": resolve(cfg["paths"]["logreg_model"]), "rf": resolve(cfg["paths"]["rf_model"]),
             selection["xgb_variant"]: resolve(cfg["paths"]["xgb_model"])}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Model file(s) not found: {missing}. Run `python -m src.train` first.")
    xgb = XGBClassifier()
    xgb.load_model(paths[selection["xgb_variant"]])
    return {"logreg": joblib.load(paths["logreg"]), "rf": joblib.load(paths["rf"]), selection["xgb_variant"]: xgb}


def to_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Same float64 matrix src.train fits on, so test inputs match training dtypes exactly."""
    return pd.DataFrame(df[cols].to_numpy(dtype="float64", na_value=np.nan), columns=cols, index=df.index)


def score_models(test: pd.DataFrame, models: dict, selection: dict, train_majority: int) -> dict:
    """Test-set scores per model, with the baseline scores src.train uses (constant class, 0/1 rule)."""
    xgb_name, X = selection["xgb_variant"], to_matrix(test, selection["feature_columns"])
    probs = {
        "majority": np.full(len(test), float(train_majority)),
        "heuristic": (test["ga_roll5"].fillna(0) > 0).to_numpy(),
        "logreg": models["logreg"].predict_proba(to_matrix(test, selection["logreg_features"]))[:, 1],
        "rf": models["rf"].predict_proba(X)[:, 1],
        xgb_name: models[xgb_name].predict_proba(X)[:, 1],
    }
    return {name: np.asarray(p, dtype=np.float64) for name, p in probs.items()}


def frozen_threshold(name: str, selection: dict) -> float:
    if name not in selection["thresholds"]:
        raise ValueError(f"No frozen threshold for model '{name}' in model_selection.json")
    return float(selection["thresholds"][name])


def binary_metrics(y: np.ndarray, p: np.ndarray, y_pred: np.ndarray, baseline_accuracy: float,
                   threshold: float | None) -> dict:
    both_classes = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y, y_pred)
    return {
        "accuracy": accuracy,
        "lift_over_baseline": accuracy - baseline_accuracy,
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if both_classes else None,
        "pr_auc": average_precision_score(y, p) if both_classes else None,
        "brier": brier_score_loss(y, p),
        "threshold": threshold,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def majority_class(y: pd.Series) -> int:
    return int(y.mean() >= 0.5) if len(y) else 0


def position_metrics(train: pd.DataFrame, test: pd.DataFrame, target: str, p: np.ndarray,
                     y_pred: np.ndarray) -> dict:
    """Per-position test metrics; each position's baseline predicts that position's train-majority class."""
    out = {}
    for pos in POSITIONS:
        mask = (test["position"] == pos).to_numpy()
        if not mask.any():
            log.warning("no test rows for position %s", pos)
            continue
        y = test[target].to_numpy()[mask]
        baseline = float(np.mean(y == majority_class(train.loc[train["position"] == pos, target])))
        m = binary_metrics(y, p[mask], y_pred[mask], baseline, None)
        out[pos] = {"n": int(mask.sum()), "base_rate": float(y.mean()), "majority_baseline_accuracy": baseline,
                    **{k: m[k] for k in POSITION_KEYS}}
    return out


def calibration(y: np.ndarray, probs: dict, n_bins: int) -> dict:
    out = {}
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        edges = np.percentile(p, np.linspace(0, 100, n_bins + 1))
        counts = np.bincount(np.searchsorted(edges[1:-1], p), minlength=n_bins)
        counts = counts[counts > 0]
        out[name] = {"bin_mean_pred": mean_pred.tolist(), "bin_frac_pos": frac_pos.tolist(),
                     "bin_count": counts.tolist() if len(counts) == len(frac_pos) else None}
    return out


def evaluate(train: pd.DataFrame, test: pd.DataFrame, models: dict, selection: dict,
             cfg: dict) -> tuple[dict, dict]:
    """Return (metrics dict, {model: test probabilities}). Pure: no file I/O."""
    target = cfg["features"]["target"]
    y = test[target].to_numpy(dtype=int)
    train_base_rate = float(train[target].mean())
    recorded = (selection.get("base_rate") or {}).get("train")
    if recorded is not None and abs(float(recorded) - train_base_rate) > 1e-6:
        log.warning("train base rate %.6f differs from model_selection.json (%.6f)", train_base_rate, recorded)
    train_majority = majority_class(train[target])
    baseline_accuracy = float(np.mean(y == train_majority))

    probs = score_models(test, models, selection, train_majority)
    selected = selection["selected_model"]
    if selected not in models:
        raise ValueError(f"selected_model '{selected}' is not one of the saved models {list(models)}")

    results, labels = {}, {}
    for name, p in probs.items():
        threshold = frozen_threshold(name, selection)
        labels[name] = (p >= threshold).astype(int)
        results[name] = binary_metrics(y, p, labels[name], baseline_accuracy, threshold)
        accuracy_05 = accuracy_score(y, (p >= 0.5).astype(int))
        results[name].update(accuracy_at_0_5=accuracy_05, lift_at_0_5=accuracy_05 - baseline_accuracy)

    prob_models = ["logreg", "rf", selection["xgb_variant"]]
    top_k, top_k_rates = top_k_check(test, target, probs[selected], int(cfg["model"]["top_k"]))
    metrics = {
        "test_season": selection["test_season"],
        "n_test_rows": int(len(test)),
        "test_base_rate": float(y.mean()),
        "majority_baseline_accuracy": baseline_accuracy,
        "selected_model": selected,
        "xgb_variant": selection["xgb_variant"],
        "threshold": results[selected]["threshold"],
        "train_seasons": list(selection["train_seasons"]),
        "val_season": selection["val_season"],
        "models": results,
        "by_position": position_metrics(train, test, target, probs[selected], labels[selected]),
        "by_minutes_band": minutes_band_metrics(test, target, probs[selected]),
        "top_k": top_k,
        "calibration": calibration(y, {m: probs[m] for m in prob_models}, int(cfg["model"]["calibration_bins"])),
        "bootstrap": bootstrap(y, test["matchweek"].to_numpy(), {m: probs[m] for m in prob_models},
                               selection["xgb_variant"], top_k_rates, selected, int(cfg["project"]["random_seed"])),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return metrics, probs


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 6) if math.isfinite(obj) else None
    return obj


def _series_color(name: str) -> str:
    return SERIES["xgb"] if name.startswith("xgb") else SERIES[name]


def _label(name: str, selected: str) -> str:
    return f"{name} (selected)" if name == selected else name


def _style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def plot_calibration(metrics: dict, probs: dict, path: Path) -> None:
    calib, selected = metrics["calibration"], metrics["selected_model"]
    top = max(max(max(c["bin_mean_pred"]), max(c["bin_frac_pos"])) for c in calib.values())
    top = max(top, max(float(np.quantile(probs[m], 0.995)) for m in calib))
    lim = min(1.0, math.ceil(top * 1.05 * 10) / 10)
    with plt.rc_context(STYLE):
        fig, (ax, axh) = plt.subplots(2, 1, figsize=(7, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1, label="Perfectly calibrated")
        for name, c in calib.items():
            brier = metrics["models"][name]["brier"]
            ax.plot(c["bin_mean_pred"], c["bin_frac_pos"], color=_series_color(name), linewidth=2, marker="o",
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"{_label(name, selected)}, Brier {brier:.3f}")
            axh.hist(probs[name], bins=np.linspace(0, lim, 41), histtype="step", linewidth=1.5,
                     color=_series_color(name))
        ax.set(xlim=(0, lim), ylim=(0, lim), ylabel="Observed share with a goal or assist")
        ax.set_title(f"Calibration on the {metrics['test_season']} test season "
                     f"({len(next(iter(calib.values()))['bin_mean_pred'])} equal-count bins)", loc="left")
        ax.legend(loc="upper left")
        axh.set(xlabel="Predicted probability of a goal or assist", ylabel="Test rows")
        for a in (ax, axh):
            _style_axes(a)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def plot_confusion(metrics: dict, path: Path) -> None:
    selected = metrics["selected_model"]
    cm = np.array(metrics["models"][selected]["confusion_matrix"])
    row_share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    names = ["No goal/assist (0)", "Goal or assist (1)"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("ramp", BLUE_RAMP), vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{row_share[i, j]:.1%} of actual {i}", ha="center", va="center",
                        fontsize=11, color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], names)
        ax.set_yticks([0, 1], names, rotation=90, va="center")
        ax.set(xlabel="Predicted", ylabel="Actual")
        ax.set_title(f"{selected} on the {metrics['test_season']} test season\n"
                     f"frozen threshold {metrics['threshold']:.3f}, n = {cm.sum():,}", loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def plot_pr(metrics: dict, probs: dict, y: np.ndarray, path: Path) -> None:
    selected = metrics["selected_model"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.axhline(metrics["test_base_rate"], color=MUTED, linewidth=1,
                   label=f"Random (base rate {metrics['test_base_rate']:.3f})")
        for name in metrics["calibration"]:
            m, color = metrics["models"][name], _series_color(name)
            precision, recall, _ = precision_recall_curve(y, probs[name])
            ax.plot(recall, precision, color=color, linewidth=2, drawstyle="steps-post",
                    label=f"{_label(name, selected)}, PR-AUC {m['pr_auc']:.3f}")
            ax.plot(m["recall"], m["precision"], marker="o", markersize=9, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
        h = metrics["models"]["heuristic"]
        ax.plot(h["recall"], h["precision"], marker="D", markersize=8, color=INK_2, markeredgecolor=SURFACE,
                markeredgewidth=2, linestyle="none", label="Heuristic: 5-match G+A > 0")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
        ax.set_title(f"Precision-recall on the {metrics['test_season']} test season\n"
                     "dots mark each model's threshold frozen on validation", loc="left")
        ax.legend(loc="upper right")
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def metrics_table(metrics: dict) -> str:
    selected, season = metrics["selected_model"], metrics["test_season"]
    lines = [
        f"## Test-season results ({season})",
        "",
        f"Test set: {metrics['n_test_rows']:,} player-matches with at least 30 minutes played. "
        f"Base rate (share with a goal or assist): {_fmt(metrics['test_base_rate'])}. "
        f"Majority-class baseline accuracy (always predict \"no goal or assist\"): "
        f"{_fmt(metrics['majority_baseline_accuracy'])}. Models were trained on "
        f"{metrics['train_seasons'][0]} to {metrics['val_season']}; thresholds were frozen on the "
        f"{metrics['val_season']} validation season before the test season was scored.",
        "",
        "| Model | Accuracy | Lift | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in metrics["models"].items():
        label = f"**{name} (selected)**" if name == selected else name
        lines.append(f"| {label} | {_fmt(m['accuracy'])} | {m['lift_over_baseline']:+.3f} | {_fmt(m['precision'])} | "
                     f"{_fmt(m['recall'])} | {_fmt(m['f1'])} | {_fmt(m['roc_auc'])} | {_fmt(m['pr_auc'])} | "
                     f"{_fmt(m['brier'])} |")
    lines += [
        "",
        "Lift = accuracy minus the majority-class baseline accuracy. Accuracy, precision, recall and F1 use each "
        "model's threshold frozen on validation (predict 1 if score >= threshold). The majority baseline scores a "
        "constant 0 and the heuristic scores 1 if the player's 5-match goals + assists > 0, so their ROC-AUC, "
        "PR-AUC and Brier describe those hard calls.",
        "",
        "### Accuracy at a fixed 0.5 cutoff (context only)",
        "",
        "| Model | Accuracy | Lift |",
        "|---|---|---|",
    ]
    for name, m in metrics["models"].items():
        label = f"**{name} (selected)**" if name == selected else name
        lines.append(f"| {label} | {_fmt(m['accuracy_at_0_5'])} | {m['lift_at_0_5']:+.3f} |")
    lines += [
        "",
        "Predict 1 if score >= 0.5: a conventional cutoff, not tuned. The headline table above uses the frozen "
        "validation thresholds. The 0/1 baselines already use 0.5, so their rows match the headline table.",
        "",
        f"### By position ({selected})",
        "",
        "| Position | n | Base rate | Majority baseline | Accuracy | Lift | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pos, m in metrics["by_position"].items():
        lines.append(f"| {pos} | {m['n']:,} | {_fmt(m['base_rate'])} | {_fmt(m['majority_baseline_accuracy'])} | "
                     f"{_fmt(m['accuracy'])} | {m['lift_over_baseline']:+.3f} | {_fmt(m['precision'])} | "
                     f"{_fmt(m['recall'])} | {_fmt(m['f1'])} | {_fmt(m['roc_auc'])} | {_fmt(m['pr_auc'])} |")
    cm = metrics["models"][selected]["confusion_matrix"]
    lines += [""] + minutes_band_markdown(metrics["by_minutes_band"], selected)
    lines += top_k_markdown(metrics["top_k"], selected) + bootstrap_markdown(metrics["bootstrap"])
    lines += [
        f"### Confusion matrix ({selected}, threshold {_fmt(metrics['threshold'])})",
        "",
        "| | Predicted 0 | Predicted 1 |",
        "|---|---|---|",
        f"| **Actual 0** | {cm[0][0]:,} | {cm[0][1]:,} |",
        f"| **Actual 1** | {cm[1][0]:,} | {cm[1][1]:,} |",
        "",
    ]
    return "\n".join(lines)


def run(cfg: dict) -> dict:
    paths, target = cfg["paths"], cfg["features"]["target"]
    selection = load_selection(resolve(paths["model_selection"]))
    if selection["test_season"] != cfg["split"]["test_season"]:
        raise ValueError(f"model_selection test_season {selection['test_season']} != config "
                         f"{cfg['split']['test_season']}")
    features_path = resolve(paths["features"])
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}. Run `python -m src.features` first.")
    df = pd.read_parquet(features_path)
    id_columns = list(cfg["features"]["id_columns"])
    needed = set(id_columns + [target, "ga_roll5", "player_prior_rate"] + selection["feature_columns"]
                 + selection["logreg_features"])
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"{features_path} is missing columns: {missing}")

    train, _, test = chronological_split(df, cfg)
    log.info("test season %s: %d rows, base rate %.4f", selection["test_season"], len(test), test[target].mean())
    metrics, probs = evaluate(train, test, load_models(cfg, selection), selection, cfg)
    metrics = json_safe(metrics)

    reports, figures = resolve(paths["reports_dir"]), resolve(paths["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)
    resolve(paths["metrics"]).write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (reports / "metrics_table.md").write_text(metrics_table(metrics), encoding="utf-8")
    predictions = test[id_columns + [target]].reset_index(drop=True).assign(
        p_logreg=probs["logreg"], p_rf=probs["rf"], p_xgb=probs[selection["xgb_variant"]])
    predictions.to_parquet(resolve(paths["test_predictions"]), index=False)

    y = test[target].to_numpy(dtype=int)
    plot_calibration(metrics, probs, figures / "calibration.png")
    plot_confusion(metrics, figures / "confusion_matrix.png")
    plot_pr(metrics, probs, y, figures / "pr_curve.png")

    log.info("majority baseline accuracy %.4f; selected %s (threshold %.4f)",
             metrics["majority_baseline_accuracy"], metrics["selected_model"], metrics["threshold"])
    for name, m in metrics["models"].items():
        log.info("%-15s acc=%.4f lift=%+.4f prec=%.4f rec=%.4f f1=%.4f roc=%s pr=%s brier=%.4f", name,
                 m["accuracy"], m["lift_over_baseline"], m["precision"], m["recall"], m["f1"],
                 _fmt(m["roc_auc"], 4), _fmt(m["pr_auc"], 4), m["brier"])
    tk = metrics["top_k"]
    log.info("top-%d hit rate %.4f vs random %.4f, random FWD %.4f, prior-rate ranker %.4f over %d matchweeks "
             "(%d duplicate rows deduped)", tk["k"], tk["model_hit_rate"], tk["random_hit_rate"],
             tk["random_fwd_hit_rate"], tk["prior_rate_ranker_hit_rate"], tk["n_matchweeks"], tk["n_rows_deduped"])
    for name, stats in metrics["bootstrap"]["models"].items():
        log.info("bootstrap %s PR-AUC %.4f [%.4f, %.4f]", name, stats["pr_auc"]["estimate"],
                 stats["pr_auc"]["ci_low"], stats["pr_auc"]["ci_high"])
    log.info("wrote %s, %s, %s and figures in %s", resolve(paths["metrics"]), reports / "metrics_table.md",
             resolve(paths["test_predictions"]), figures)
    return metrics


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()
