"""Analysis of the learning-curve draws (development data only):

* paired differences against the logistic-regression baseline at every size,
* training notes needed to reach 2% and 1% mean error, with bootstrap intervals,
* which hyperparameters cross-validation picked at each size,
* a fuller random-forest check on all development rows (bigger grid, OOB vs trees).

Run after learning_curves.py:  .venv/bin/python src/analyse.py
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from common import FEATURES, N_FOLDS, RESULTS, SEED, TARGET, load_dev

BASE = "Logistic regression"
TARGETS = [0.02, 0.01]
N_BOOT = 2000


def paired(df):
    wide = df.pivot_table(index=["n_train", "draw"], columns="method", values="error")
    rows = []
    for m in wide.columns:
        if m == BASE:
            continue
        d = (wide[m] - wide[BASE]).groupby(level="n_train")
        for n, v in d:
            rows.append({"method": m, "n_train": n, "mean_diff": v.mean(),
                         "se": v.std(ddof=1) / np.sqrt(len(v)),
                         "wins": int((v < 0).sum()), "ties": int((v == 0).sum()),
                         "losses": int((v > 0).sum()), "draws": len(v)})
    return pd.DataFrame(rows)


def first_crossing(sizes, means, t):
    """n at which the mean curve first drops to t, interpolated in log-log space."""
    below = np.flatnonzero(means <= t)
    if not len(below):
        return np.inf
    i = below[0]
    if i == 0:
        return float(sizes[0])
    x0, x1 = np.log(sizes[i - 1]), np.log(sizes[i])
    y0, y1 = np.log(means[i - 1]), np.log(max(means[i], 1e-9))
    return float(np.exp(x0 + (np.log(t) - y0) * (x1 - x0) / (y1 - y0)))


def n_to_reach(df):
    rng = np.random.default_rng(SEED)
    rows = []
    for m, g in df.groupby("method"):
        e = g.pivot(index="n_train", columns="draw", values="error")
        sizes, E = e.index.values, e.values
        idx = rng.integers(0, E.shape[1], size=(N_BOOT,) + E.shape)
        boot = np.take_along_axis(np.broadcast_to(E, idx.shape), idx, axis=2).mean(axis=2)
        for t in TARGETS:
            est = first_crossing(sizes, E.mean(axis=1), t)
            bs = np.array([first_crossing(sizes, b, t) for b in boot])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"method": m, "target": t, "n_needed": est, "lo95": lo, "hi95": hi,
                         "share_boot_not_reached": float(np.isinf(bs).mean())})
    return pd.DataFrame(rows)


def chosen_params(df):
    rows = []
    for (m, n), g in df.groupby(["method", "n_train"]):
        vc = g["params"].value_counts()
        rows.append({"method": m, "n_train": n, "most_common": vc.index[0],
                     "share": vc.iloc[0] / len(g)})
    return pd.DataFrame(rows)


def forest_check():
    dev = load_dev()
    X, y = dev[FEATURES].values, dev[TARGET].values
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = {}
    for label, grid in [("small (used in curves)",
                         {"max_features": [1, 2, 4], "min_samples_leaf": [1, 3],
                          "n_estimators": [200]}),
                        ("large", {"max_features": [1, 2, 3, 4],
                                   "min_samples_leaf": [1, 2, 3, 5],
                                   "n_estimators": [200, 500],
                                   "criterion": ["gini", "entropy"]})]:
        gs = GridSearchCV(RandomForestClassifier(random_state=SEED), grid, cv=cv,
                          scoring="accuracy", n_jobs=-1).fit(X, y)
        i = gs.best_index_
        folds = [1 - gs.cv_results_[f"split{k}_test_score"][i] for k in range(N_FOLDS)]
        out[label] = {"best": {a: str(b) for a, b in gs.best_params_.items()},
                      "cv_error": float(np.mean(folds)),
                      "se": float(np.std(folds, ddof=1) / np.sqrt(N_FOLDS)),
                      "n_configs": int(len(gs.cv_results_["params"]))}
    oob = []
    for t in [25, 50, 100, 200, 500, 1000]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = RandomForestClassifier(n_estimators=t, max_features=2, oob_score=True,
                                        random_state=SEED, n_jobs=-1).fit(X, y)
        oob.append({"trees": t, "oob_error": float(1 - rf.oob_score_)})
    out["oob_vs_trees"] = oob
    return out


def main():
    df = pd.read_csv(RESULTS / "learning_curves_draws.csv")
    p = paired(df)
    p.to_csv(RESULTS / "paired_vs_lr.csv", index=False)
    r = n_to_reach(df)
    r.to_csv(RESULTS / "n_to_reach.csv", index=False)
    c = chosen_params(df)
    c.to_csv(RESULTS / "chosen_params.csv", index=False)
    f = forest_check()
    (RESULTS / "forest_check.json").write_text(json.dumps(f, indent=2))
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(p.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(r.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        print(c.to_string(index=False))
    print(json.dumps(f, indent=2))


if __name__ == "__main__":
    main()
