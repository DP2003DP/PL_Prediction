"""Train baselines, logistic regression, random forest, and XGBoost with season-based CV.

Run as `python -m src.train`. Hyperparameters are chosen by expanding-window CV over the train
seasons; the validation season supplies early stopping, metrics, frozen thresholds, and model
selection. Chosen models are refit on train + validation and saved. Test-season rows are never used.
"""
import itertools
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.split import chronological_split, season_folds
from src.utils import get_logger, load_config, resolve

log = get_logger(__name__)

MODEL_NAMES = ["majority", "heuristic", "logreg", "rf", "xgb_unweighted", "xgb_weighted"]
CANDIDATES = ["logreg", "rf", "xgb_unweighted", "xgb_weighted"]
METRICS = ["pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1", "brier"]
LOGREG_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
RF_N_CONFIGS = 8
RF_SPACE = {
    "n_estimators": [300],
    "max_depth": [6, 10, 16, None],
    "min_samples_leaf": [20, 50, 100, 200],
    "max_features": ["sqrt", 0.33, 0.5],
}
XGB_MAX_ROUNDS = 2000
LEAKAGE_ROC_AUC = 0.85


@contextmanager
def phase(name: str, timings: dict):
    log.info("phase start: %s", name)
    t0 = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - t0, 2)
    log.info("phase done: %s (%.1fs)", name, timings[name])


def load_features(path: Path, cfg: dict) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}. Run `python -m src.features` first.")
    df = pd.read_parquet(path)
    fcfg = cfg["features"]
    required = ["season", "date", fcfg["target"], "ga_roll5"] + fcfg["feature_columns"] + fcfg["logreg_features"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"features.parquet is missing columns: {missing}")
    y = df[fcfg["target"]]
    if y.isna().any() or not set(pd.unique(y)) <= {0, 1}:
        raise ValueError(f"Target '{fcfg['target']}' must be 0/1 with no missing values")
    return df


def to_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(df[cols].to_numpy(dtype="float64", na_value=np.nan), columns=cols, index=df.index)


def neg_pos_ratio(y: np.ndarray) -> float:
    pos = int(y.sum())
    return float((len(y) - pos) / pos)


def f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    p_, r_ = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p_ * r_, p_ + r_, out=np.zeros_like(p_), where=(p_ + r_) > 0)
    return float(thresholds[int(np.argmax(f1))])


def score(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }


def make_logreg(C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed)),
    ])


def make_rf(params: dict, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)


def make_xgb(params: dict, seed: int, n_estimators: int, spw: float | None = None,
             early_stopping_rounds: int | None = None) -> XGBClassifier:
    kw = dict(params, n_estimators=n_estimators, objective="binary:logistic", tree_method="hist",
              random_state=seed, n_jobs=-1)
    if spw is not None:
        kw["scale_pos_weight"] = spw
    if early_stopping_rounds:
        kw.update(early_stopping_rounds=early_stopping_rounds, eval_metric="aucpr")
    return XGBClassifier(**kw)


def fit_xgb_es(params, X_fit, y_fit, X_eval, y_eval, seed, rounds, spw) -> XGBClassifier:
    model = make_xgb(params, seed, XGB_MAX_ROUNDS, spw, rounds)
    model.fit(X_fit, y_fit, eval_set=[(X_eval, y_eval)], verbose=False)
    return model


def sample_grid(space: dict, n: int, rng: np.random.Generator) -> list[dict]:
    combos = [dict(zip(space, values)) for values in itertools.product(*space.values())]
    picks = rng.choice(len(combos), size=min(n, len(combos)), replace=False)
    return [combos[i] for i in sorted(picks)]


def sample_xgb_configs(n: int, rng: np.random.Generator) -> list[dict]:
    return [{
        "learning_rate": round(float(10 ** rng.uniform(-2, np.log10(0.3))), 4),
        "max_depth": int(rng.integers(2, 9)),
        "subsample": round(float(rng.uniform(0.5, 1.0)), 3),
        "colsample_bytree": round(float(rng.uniform(0.4, 1.0)), 3),
        "min_child_weight": round(float(10 ** rng.uniform(0, 2)), 2),
    } for _ in range(n)]


def run_search(model: str, configs: list[dict], evaluate) -> tuple[dict, list[dict], float]:
    """evaluate(params) -> (fold PR-AUCs, fold best iterations or None). Returns best params."""
    rows = []
    for i, params in enumerate(configs):
        fold_scores, fold_iters = evaluate(params)
        row = {"model": model, "config_id": i, "params": json.dumps(params),
               "mean_pr_auc": float(np.mean(fold_scores)), "std_pr_auc": float(np.std(fold_scores))}
        row.update({f"fold{k}_pr_auc": float(s) for k, s in enumerate(fold_scores, start=1)})
        row["fold_best_iterations"] = None if fold_iters is None else ",".join(map(str, fold_iters))
        rows.append(row)
        log.info("%s config %d/%d mean PR-AUC=%.4f %s", model, i + 1, len(configs), row["mean_pr_auc"], params)
    best = int(np.argmax([r["mean_pr_auc"] for r in rows]))
    for i, r in enumerate(rows):
        r["selected"] = i == best
    return configs[best], rows, rows[best]["mean_pr_auc"]


def sklearn_cv(make_model, X: pd.DataFrame, y: np.ndarray, folds):
    def evaluate(params):
        scores = []
        for tr, va in folds:
            model = make_model(params).fit(X.iloc[tr], y[tr])
            scores.append(average_precision_score(y[va], model.predict_proba(X.iloc[va])[:, 1]))
        return scores, None
    return evaluate


def xgb_cv(X: pd.DataFrame, y: np.ndarray, folds, seed: int, rounds: int, weighted: bool):
    def evaluate(params):
        scores, iters = [], []
        for tr, va in folds:
            spw = neg_pos_ratio(y[tr]) if weighted else None
            model = fit_xgb_es(params, X.iloc[tr], y[tr], X.iloc[va], y[va], seed, rounds, spw)
            scores.append(average_precision_score(y[va], model.predict_proba(X.iloc[va])[:, 1]))
            iters.append(int(model.best_iteration))
        return scores, iters
    return evaluate


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Not JSON serializable: {type(o).__name__}")


def run(df: pd.DataFrame, cfg: dict) -> dict:
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    seed = int(cfg["project"]["random_seed"])
    rng = np.random.default_rng(seed)
    fcfg, scfg, mcfg = cfg["features"], cfg["split"], cfg["model"]
    feats, lr_feats, target = fcfg["feature_columns"], fcfg["logreg_features"], fcfg["target"]
    train_seasons = [str(s) for s in scfg["train_seasons"]]
    rounds = int(mcfg["early_stopping_rounds"])

    with phase("split", timings):
        train, val, _test = chronological_split(df, cfg)
        del _test
        train, val = train.reset_index(drop=True), val.reset_index(drop=True)
        folds = season_folds(train, train_seasons)
        X_tr, X_va = to_matrix(train, feats), to_matrix(val, feats)
        y_tr, y_va = train[target].to_numpy(dtype=int), val[target].to_numpy(dtype=int)
        spw_train = neg_pos_ratio(y_tr)
    log.info("train rows=%d base_rate=%.4f | val rows=%d base_rate=%.4f | %d CV folds | neg/pos=%.3f",
             len(train), y_tr.mean(), len(val), y_va.mean(), len(folds), spw_train)
    fold_info = [{"fold": i, "train_seasons": train_seasons[:i], "val_season": train_seasons[i],
                  "n_train": int(len(tr)), "n_val": int(len(va))} for i, (tr, va) in enumerate(folds, start=1)]

    preds: dict[str, np.ndarray] = {}
    cv_rows: list[dict] = []
    cv_best: dict[str, float] = {}

    with phase("baselines", timings):
        majority_class = int(y_tr.mean() >= 0.5)
        if majority_class != 0:
            log.warning("majority class on train is 1, not 0; the majority baseline predicts 1")
        preds["majority"] = np.full(len(val), float(majority_class))
        preds["heuristic"] = (X_va["ga_roll5"].fillna(0) > 0).to_numpy(dtype=float)

    with phase("logreg_cv", timings):
        X_tr_lr, X_va_lr = X_tr[lr_feats], X_va[lr_feats]
        lr_best, rows, cv_best["logreg"] = run_search(
            "logreg", [{"C": c} for c in LOGREG_C_GRID],
            sklearn_cv(lambda p: make_logreg(p["C"], seed), X_tr_lr, y_tr, folds))
        cv_rows += rows
        logreg = make_logreg(lr_best["C"], seed).fit(X_tr_lr, y_tr)
        preds["logreg"] = logreg.predict_proba(X_va_lr)[:, 1]

    with phase("rf_cv", timings):
        rf_best, rows, cv_best["rf"] = run_search(
            "rf", sample_grid(RF_SPACE, RF_N_CONFIGS, rng),
            sklearn_cv(lambda p: make_rf(p, seed), X_tr, y_tr, folds))
        cv_rows += rows
        rf = make_rf(rf_best, seed).fit(X_tr, y_tr)
        preds["rf"] = rf.predict_proba(X_va)[:, 1]

    xgb_configs = sample_xgb_configs(int(mcfg["xgb_search_iterations"]), rng)
    xgb_info: dict[str, dict] = {}
    for variant, weighted in (("xgb_unweighted", False), ("xgb_weighted", True)):
        with phase(f"{variant}_cv", timings):
            best, rows, cv_best[variant] = run_search(
                variant, xgb_configs, xgb_cv(X_tr, y_tr, folds, seed, rounds, weighted))
            cv_rows += rows
            spw = spw_train if weighted else None
            model = fit_xgb_es(best, X_tr, y_tr, X_va, y_va, seed, rounds, spw)
            preds[variant] = model.predict_proba(X_va)[:, 1]
            xgb_info[variant] = {"params": best, "best_iteration": int(model.best_iteration),
                                 "scale_pos_weight": spw}
            log.info("%s best_iteration=%d on val early stopping", variant, model.best_iteration)

    with phase("thresholds_and_metrics", timings):
        thresholds = {m: 0.5 if m in ("majority", "heuristic") else f1_threshold(y_va, preds[m])
                      for m in MODEL_NAMES}
        val_metrics = {m: score(y_va, preds[m], thresholds[m]) for m in MODEL_NAMES}
        selected = max(CANDIDATES, key=lambda m: val_metrics[m]["pr_auc"])
        xgb_variant = max(("xgb_unweighted", "xgb_weighted"), key=lambda m: val_metrics[m]["pr_auc"])
        suspicious = [m for m in CANDIDATES if val_metrics[m]["roc_auc"] > LEAKAGE_ROC_AUC]
        if suspicious:
            log.warning("val ROC-AUC > %.2f for %s: implausible for form features, check for leakage",
                        LEAKAGE_ROC_AUC, suspicious)

    with phase("refit_train_val", timings):
        X_all = pd.concat([X_tr, X_va], ignore_index=True)
        y_all = np.concatenate([y_tr, y_va])
        logreg_final = make_logreg(lr_best["C"], seed).fit(X_all[lr_feats], y_all)
        rf_final = make_rf(rf_best, seed).fit(X_all, y_all)
        xi = xgb_info[xgb_variant]
        xgb_final = make_xgb(xi["params"], seed, xi["best_iteration"] + 1, xi["scale_pos_weight"])
        xgb_final.fit(X_all, y_all, verbose=False)

    with phase("save", timings):
        out = {k: resolve(cfg["paths"][k]) for k in
               ("xgb_model", "rf_model", "logreg_model", "model_selection", "validation_metrics")}
        out["cv_results"] = resolve(cfg["paths"]["reports_dir"]) / "cv_results.csv"
        for path in out.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        xgb_final.save_model(str(out["xgb_model"]))
        joblib.dump(rf_final, out["rf_model"], compress=3)
        joblib.dump(logreg_final, out["logreg_model"])

        val_df = pd.DataFrame([{"model": m, "threshold": thresholds[m], **val_metrics[m],
                                "cv_pr_auc": cv_best.get(m), "n_val": len(val),
                                "val_base_rate": float(y_va.mean()), "selected": m == selected}
                               for m in MODEL_NAMES])
        val_df.to_csv(out["validation_metrics"], index=False)
        pd.DataFrame(cv_rows).to_csv(out["cv_results"], index=False)

    timings["total"] = round(time.perf_counter() - t_start, 2)
    selection = {
        "selected_model": selected,
        "xgb_variant": xgb_variant,
        "threshold_rule": "predict 1 if probability >= threshold",
        "thresholds": thresholds,
        "val_metrics": val_metrics,
        "xgb_params": {**xi["params"], "objective": "binary:logistic", "tree_method": "hist",
                       "random_state": seed},
        "xgb_best_n_estimators": xi["best_iteration"] + 1,
        "xgb_scale_pos_weight": xi["scale_pos_weight"],
        "xgb_variants": {v: {"params": info["params"], "best_n_estimators": info["best_iteration"] + 1,
                             "scale_pos_weight": info["scale_pos_weight"], "cv_pr_auc": cv_best[v],
                             "val_pr_auc": val_metrics[v]["pr_auc"]} for v, info in xgb_info.items()},
        "rf_params": {**rf_best, "random_state": seed},
        "logreg_params": {"C": lr_best["C"], "max_iter": 2000, "imputer": "median", "scaler": "standard"},
        "baselines": {"majority": f"constant score {majority_class}",
                      "heuristic": "score 1 if ga_roll5 > 0 (NaN counts as 0), else 0"},
        "majority_class": majority_class,
        "cv": {"method": "expanding-window by season (src.split.season_folds); score = mean PR-AUC",
               "folds": fold_info,
               "n_configs": {"logreg": len(LOGREG_C_GRID), "rf": RF_N_CONFIGS,
                             "xgb": len(xgb_configs), "xgb_variants": 2},
               "best_pr_auc": cv_best,
               "xgb_early_stopping_rounds": rounds, "xgb_max_rounds": XGB_MAX_ROUNDS},
        "refit": {"data": "train + val", "n_rows": int(len(y_all)), "models": ["logreg", "rf", xgb_variant]},
        "train_seasons": train_seasons,
        "val_season": str(scfg["val_season"]),
        "test_season": str(scfg["test_season"]),
        "feature_columns": feats,
        "logreg_features": lr_feats,
        "n_rows": {"train": int(len(train)), "val": int(len(val))},
        "base_rate": {"train": float(y_tr.mean()), "val": float(y_va.mean())},
        "random_seed": seed,
        "timings_seconds": timings,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out["model_selection"].write_text(
        json.dumps(selection, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")

    log.info("validation metrics (fit on train, scored on %s):\n%s", scfg["val_season"],
             val_df.set_index("model")[["threshold"] + METRICS + ["cv_pr_auc"]].round(4).to_string())
    log.info("selected_model=%s xgb_variant=%s xgb n_estimators=%d total runtime %.1fs",
             selected, xgb_variant, xi["best_iteration"] + 1, timings["total"])
    return selection


def main() -> None:
    cfg = load_config()
    df = load_features(resolve(cfg["paths"]["features"]), cfg)
    run(df, cfg)


if __name__ == "__main__":
    main()
