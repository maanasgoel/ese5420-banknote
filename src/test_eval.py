"""One-time evaluation on the 270 held-out test rows.

Every method is tuned by stratified 5-fold CV on all 1,078 development rows (same
grids as the learning curves), refitted on the development rows, and scored once on
the test rows. Pairwise comparisons against the logistic-regression baseline use an
exact McNemar test on the notes where the two methods disagree.

Run:  .venv/bin/python src/test_eval.py
"""
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from baseline import wilson
from common import (FEATURES, N_FOLDS, RESULTS, SEED, TARGET, all_models, knn_grid_for,
                    load_dev, load_test)


def main():
    dev, test = load_dev(), load_test()
    assert not set(dev.row_id) & set(test.row_id)
    Xd, yd = dev[FEATURES].values, dev[TARGET].values
    Xt, yt = test[FEATURES].values, test[TARGET].values
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    out = {"run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_dev": int(len(yd)), "n_test": int(len(yt)), "forged_test": int(yt.sum()),
           "methods": {}}
    preds = pd.DataFrame({"row_id": test.row_id, "y": yt})
    roc = []
    for name, (pipe, grid) in all_models().items():
        grid = dict(grid)
        if "clf__n_neighbors" in grid:
            grid["clf__n_neighbors"] = knn_grid_for(len(yd), N_FOLDS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1).fit(Xd, yd)
        p = gs.predict(Xt)
        s = gs.predict_proba(Xt)[:, 1]
        preds[name] = p
        preds[name + " score"] = s
        tn, fp, fn, tp = confusion_matrix(yt, p, labels=[0, 1]).ravel()
        k = int((p != yt).sum())
        out["methods"][name] = {
            "params": {a: (float(b) if isinstance(b, (int, float, np.floating)) else str(b))
                       for a, b in gs.best_params_.items()},
            "dev_cv_error": float(1 - gs.best_score_),
            "test_errors": k, "test_error": k / len(yt),
            "wilson95": list(wilson(k, len(yt))),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "missed_forgeries": int(fn), "false_alarms": int(fp),
            "auc": float(roc_auc_score(yt, s))}
        fpr, tpr, _ = roc_curve(yt, s)
        roc += [{"method": name, "fpr": a, "tpr": b} for a, b in zip(fpr, tpr)]

    base = preds["Logistic regression"] == yt
    for name in out["methods"]:
        if name == "Logistic regression":
            continue
        other = preds[name] == yt
        b, c = int((base & ~other).sum()), int((~base & other).sum())
        out["methods"][name]["mcnemar_vs_lr"] = {
            "lr_right_other_wrong": b, "lr_wrong_other_right": c,
            "p_exact": float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "test_eval.json").write_text(json.dumps(out, indent=2))
    preds.to_csv(RESULTS / "test_predictions.csv", index=False)
    pd.DataFrame(roc).to_csv(RESULTS / "test_roc.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
