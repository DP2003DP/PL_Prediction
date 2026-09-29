## Test-season results (2025-26)

Test set: 8,107 player-matches with at least 30 minutes played. Base rate (share with a goal or assist): 0.184. Majority-class baseline accuracy (always predict "no goal or assist"): 0.816. Models were trained on 2020-21 to 2024-25; thresholds were frozen on the 2024-25 validation season before the test season was scored.

| Model | Accuracy | Lift | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| majority | 0.816 | +0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.184 | 0.184 |
| heuristic | 0.513 | -0.303 | 0.222 | 0.662 | 0.333 | 0.570 | 0.209 | 0.487 |
| logreg | 0.691 | -0.125 | 0.307 | 0.543 | 0.392 | 0.684 | 0.308 | 0.141 |
| rf | 0.682 | -0.134 | 0.302 | 0.557 | 0.391 | 0.684 | 0.314 | 0.140 |
| **xgb_unweighted (selected)** | 0.701 | -0.115 | 0.311 | 0.516 | 0.388 | 0.689 | 0.318 | 0.140 |

Lift = accuracy minus the majority-class baseline accuracy. Accuracy, precision, recall and F1 use each model's threshold frozen on validation (predict 1 if score >= threshold). The majority baseline scores a constant 0 and the heuristic scores 1 if the player's 5-match goals + assists > 0, so their ROC-AUC, PR-AUC and Brier describe those hard calls.

### Accuracy at a fixed 0.5 cutoff (context only)

| Model | Accuracy | Lift |
|---|---|---|
| majority | 0.816 | +0.000 |
| heuristic | 0.513 | -0.303 |
| logreg | 0.815 | -0.002 |
| rf | 0.817 | +0.001 |
| **xgb_unweighted (selected)** | 0.815 | -0.001 |

Predict 1 if score >= 0.5: a conventional cutoff, not tuned. The headline table above uses the frozen validation thresholds. The 0/1 baselines already use 0.5, so their rows match the headline table.

### By position (xgb_unweighted)

| Position | n | Base rate | Majority baseline | Accuracy | Lift | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| FWD | 911 | 0.323 | 0.677 | 0.431 | -0.246 | 0.346 | 0.857 | 0.493 | 0.589 | 0.391 |
| MID | 3,876 | 0.224 | 0.776 | 0.597 | -0.179 | 0.298 | 0.591 | 0.396 | 0.635 | 0.325 |
| DEF | 3,320 | 0.099 | 0.901 | 0.897 | -0.004 | 0.182 | 0.012 | 0.023 | 0.615 | 0.147 |

### Diagnostic: by minutes played (xgb_unweighted)

Diagnostic only: minutes are known only after the match, so they are not a model feature and nothing here is tuned. It shows where conditioning on 30+ minutes matters: the model does not know whether a player will come off the bench or play the full match.

| Minutes played | n | Base rate | Mean predicted | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 30-59 | 1,051 | 0.081 | 0.196 | 0.694 | 0.174 |
| 60-89 | 2,401 | 0.214 | 0.219 | 0.670 | 0.335 |
| 90+ | 4,655 | 0.191 | 0.171 | 0.706 | 0.347 |

### Top-20 check (xgb_unweighted)

Each matchweek, the top 20 players are compared with simpler pickers on the same pool: one row per player per matchweek (for a player with two fixtures in a matchweek, the higher-probability fixture). Hit rate = share of picks with a goal or assist, averaged over 38 matchweeks.

| Picker | Hit rate | Model lift over this picker |
|---|---|---|
| Model top 20 (xgb_unweighted) | 0.387 | - |
| Top 20 by player_prior_rate | 0.372 | 1.04x |
| Random forward | 0.324 | 1.20x |
| Random player | 0.183 | 2.11x |

### Uncertainty (95% CI, matchweek bootstrap)

1,000 resamples of the 38 test matchweeks with replacement (seed 42); percentile intervals. Differences use the same resamples for both models.

| Quantity | Estimate | 95% CI |
|---|---|---|
| logreg PR-AUC | 0.308 | [0.286, 0.331] |
| logreg ROC-AUC | 0.684 | [0.668, 0.701] |
| rf PR-AUC | 0.314 | [0.293, 0.334] |
| rf ROC-AUC | 0.684 | [0.668, 0.698] |
| xgb_unweighted PR-AUC | 0.318 | [0.298, 0.339] |
| xgb_unweighted ROC-AUC | 0.689 | [0.673, 0.703] |
| xgb_unweighted - logreg PR-AUC | +0.0107 | [+0.0014, +0.0199] |
| xgb_unweighted - logreg ROC-AUC | +0.0044 | [-0.0018, +0.0108] |
| xgb_unweighted - rf PR-AUC | +0.0045 | [-0.0016, +0.0101] |
| xgb_unweighted - rf ROC-AUC | +0.0046 | [+0.0012, +0.0079] |
| xgb_unweighted top-k hit rate | 0.387 | [0.346, 0.426] |
| xgb_unweighted top-k lift vs random | 2.11x | [1.91, 2.30] |

### Confusion matrix (xgb_unweighted, threshold 0.241)

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 4,917 | 1,701 |
| **Actual 1** | 721 | 768 |
