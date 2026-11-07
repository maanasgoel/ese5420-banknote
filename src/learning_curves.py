"""Learning curves (main question after proposal feedback): how many training notes
does each method need to reach a given error rate?

For each training size n and each of 50 random draws: take a stratified sample of n
development rows, tune each method by stratified 5-fold CV inside that sample, and
score it on all remaining development rows. All methods see the same draws, so the
comparison is paired. The test set is not used.

A second, "grouped" run repeats this for the two core methods with training notes
drawn as whole 8-row blocks of the original file order (our stand-in for "all images
of one specimen"), so no evaluation row shares a block with a training row.

Run:  .venv/bin/python src/learning_curves.py            # both runs
      .venv/bin/python src/learning_curves.py random     # only the main run
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)

from common import (CORE, FEATURES, N_FOLDS, RESULTS, SEED, TARGET, all_models,
                    knn_grid_for, load_dev, pseudo_specimen_groups)

SIZES = [10, 20, 30, 50, 100, 150, 200, 300, 400, 600, 800]
GROUPED_SIZES = [50, 100, 200, 400, 800]  # below 50 there are too few blocks for 5 folds
N_DRAWS = 50


def draw_seed(n, draw, tag=None):
    """Same seeds as the baseline-checkpoint run for the random draws (tag=None)."""
    key = [SEED, n, draw] + ([tag] if tag is not None else [])
    return int(np.random.SeedSequence(key).generate_state(1)[0])


def fit_and_score(X, y, tr, ev, n, cv, names, groups=None):
    rows = []
    for name, (pipe, grid) in all_models().items():
        if name not in names:
            continue
        grid = dict(grid)
        if "clf__n_neighbors" in grid:
            grid["clf__n_neighbors"] = knn_grid_for(n, N_FOLDS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy")
            gs.fit(X[tr], y[tr], **({"groups": groups} if groups is not None else {}))
        pred = gs.predict(X[ev])
        score = gs.predict_proba(X[ev])[:, 1]
        rows.append({"method": name, "n_train": n, "n_eval": len(ev),
                     "error": float((pred != y[ev]).mean()),
                     "missed_forgeries": int(((pred == 0) & (y[ev] == 1)).sum()),
                     "false_alarms": int(((pred == 1) & (y[ev] == 0)).sum()),
                     "auc": float(roc_auc_score(y[ev], score)),
                     "params": str(gs.best_params_)})
    return rows


def random_draw(X, y, n, draw, names):
    seed = draw_seed(n, draw)
    tr, ev = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    inner = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return [dict(r, draw=draw) for r in fit_and_score(X, y, tr, ev, n, inner, names)]


def grouped_draw(X, y, groups, n, draw, names):
    """Pick whole blocks per class until the class quota is met; evaluation rows come
    only from blocks that were not picked (rows left over in a cut block are dropped)."""
    seed = draw_seed(n, draw, tag=1)
    rng = np.random.default_rng(seed)
    n1 = int(round(n * y.mean()))
    tr, used = [], set()
    for c, quota in ((0, n - n1), (1, n1)):
        blocks = rng.permutation(np.unique(groups[y == c]))
        taken = []
        for b in blocks:
            if len(taken) >= quota:
                break
            used.add(b)
            taken.extend(rng.permutation(np.flatnonzero(groups == b)).tolist())
        tr.extend(taken[:quota])
    tr = np.array(tr)
    ev = np.flatnonzero(~np.isin(groups, list(used)))
    inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return [dict(r, draw=draw) for r in
            fit_and_score(X, y, tr, ev, n, inner, names, groups=groups[tr])]


def summarise(df):
    g = df.groupby(["method", "n_train"])["error"]
    s = g.agg(mean="mean", sd="std", q10=lambda e: e.quantile(0.10),
              q90=lambda e: e.quantile(0.90), median="median", draws="count").reset_index()
    s["se"] = s["sd"] / np.sqrt(s["draws"])
    auc = df.groupby(["method", "n_train"])["auc"].mean().rename("auc").reset_index()
    return s.merge(auc, on=["method", "n_train"])


def run(kind, X, y, groups):
    if kind == "random":
        names, sizes = list(all_models()), SIZES
        task = lambda n, d: random_draw(X, y, n, d, names)  # noqa: E731
    else:
        names, sizes = CORE, GROUPED_SIZES
        task = lambda n, d: grouped_draw(X, y, groups, n, d, names)  # noqa: E731
    t0 = time.time()
    jobs = [(n, d) for n in sizes for d in range(N_DRAWS)]
    out = Parallel(n_jobs=-1)(delayed(task)(n, d) for n, d in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    stem = "learning_curves" if kind == "random" else "learning_curves_grouped"
    df.to_csv(RESULTS / f"{stem}_draws.csv", index=False)
    s = summarise(df)
    s.to_csv(RESULTS / f"{stem}_summary.csv", index=False)
    print(f"== {kind} ==")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"{len(jobs)} draws in {time.time() - t0:.0f}s\n")


def main():
    kinds = sys.argv[1:] or ["random", "grouped"]
    dev = load_dev()
    X, y = dev[FEATURES].values, dev[TARGET].values
    groups = pseudo_specimen_groups(dev, block=8)
    RESULTS.mkdir(exist_ok=True)
    for kind in kinds:
        run(kind, X, y, groups)


if __name__ == "__main__":
    main()
