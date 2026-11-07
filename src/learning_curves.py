"""Learning curves (main question after proposal feedback): how many training notes
does each method need to reach a given error rate?

For each training size n and each of 50 random draws: take a stratified sample of n
development rows, tune each method by stratified 5-fold CV inside that sample, and
score it on all remaining development rows. All three methods see the same draws,
so the comparison is paired. The test set is not used.

Run:  .venv/bin/python src/learning_curves.py
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from common import FEATURES, N_FOLDS, RESULTS, SEED, TARGET, load_dev, models

SIZES = [20, 50, 100, 200, 400, 800]
N_DRAWS = 50


def one_draw(X, y, n, draw):
    seed = int(np.random.SeedSequence([SEED, n, draw]).generate_state(1)[0])
    tr, ev = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    inner = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rows = []
    for name, (pipe, grid) in models().items():
        grid = dict(grid)
        if "clf__n_neighbors" in grid:  # k cannot exceed an inner training fold
            max_k = int(n * (N_FOLDS - 1) / N_FOLDS) - 1
            grid["clf__n_neighbors"] = [k for k in grid["clf__n_neighbors"] if k <= max_k]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="accuracy").fit(X[tr], y[tr])
        pred = gs.predict(X[ev])
        score = gs.predict_proba(X[ev])[:, 1]
        rows.append({"method": name, "n_train": n, "draw": draw, "n_eval": len(ev),
                     "error": float((pred != y[ev]).mean()),
                     "missed_forgeries": int(((pred == 0) & (y[ev] == 1)).sum()),
                     "false_alarms": int(((pred == 1) & (y[ev] == 0)).sum()),
                     "auc": float(roc_auc_score(y[ev], score)),
                     "params": str(gs.best_params_)})
    return rows


def summarise(df):
    g = df.groupby(["method", "n_train"])["error"]
    s = g.agg(mean="mean", sd="std", q10=lambda e: e.quantile(0.10),
              q90=lambda e: e.quantile(0.90), median="median").reset_index()
    s["se"] = s["sd"] / np.sqrt(N_DRAWS)
    return s


def main():
    sizes = [int(a) for a in sys.argv[1:]] or SIZES
    dev = load_dev()
    X, y = dev[FEATURES].values, dev[TARGET].values
    t0 = time.time()
    jobs = [(n, d) for n in sizes for d in range(N_DRAWS)]
    out = Parallel(n_jobs=-1, verbose=0)(delayed(one_draw)(X, y, n, d) for n, d in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "learning_curves_draws.csv", index=False)
    s = summarise(df)
    s.to_csv(RESULTS / "learning_curves_summary.csv", index=False)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n{len(jobs)} draws in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
