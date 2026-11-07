"""Baseline checkpoint: data summary, majority-class reference and the logistic
regression baseline tuned by stratified 5-fold CV. Uses development data only;
the test rows in data/split.csv are not loaded here.

Run:  .venv/bin/python src/baseline.py
"""
import json

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score, cross_validate)

from common import (FEATURES, N_FOLDS, RESULTS, SEED, SPLIT_FILE, TARGET, load_clean,
                    load_dev, make_split, models, pseudo_specimen_groups)


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def fold_summary(errors):
    errors = np.asarray(errors)
    return {"folds": errors.round(5).tolist(), "mean": float(errors.mean()),
            "sd": float(errors.std(ddof=1)), "se": float(errors.std(ddof=1) / np.sqrt(len(errors)))}


def main():
    RESULTS.mkdir(exist_ok=True)
    df, n_dup = load_clean()
    if not SPLIT_FILE.exists():
        make_split(df)
    dev = load_dev()
    X, y = dev[FEATURES].values, dev[TARGET].values
    out = {
        "n_raw": int(len(df) + n_dup), "n_duplicates_dropped": n_dup, "n_clean": int(len(df)),
        "forged_share_clean": float(df[TARGET].mean()),
        "n_dev": int(len(dev)), "n_test": int(len(df) - len(dev)),
        "forged_share_dev": float(y.mean()),
        "class_counts_clean": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Majority-class reference
    maj = cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=cv, scoring="accuracy")
    out["majority_cv_error"] = fold_summary(1 - maj)

    # Logistic regression: tune C over a log grid by stratified 5-fold CV
    pipe, grid = models()["Logistic regression"]
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1).fit(X, y)
    best = gs.best_index_
    fold_err = [1 - gs.cv_results_[f"split{i}_test_score"][best] for i in range(N_FOLDS)]
    out["lr_best_C"] = float(gs.best_params_["clf__C"])
    out["lr_cv_error_at_best_C"] = fold_summary(fold_err)
    out["lr_cv_error_by_C"] = {
        # sklearn's std uses ddof=0; rescale to ddof=1 to match fold_summary
        f"{c:g}": {"mean": float(1 - m),
                   "se": float(s * np.sqrt(N_FOLDS / (N_FOLDS - 1)) / np.sqrt(N_FOLDS))}
        for c, m, s in zip(gs.cv_results_["param_clf__C"], gs.cv_results_["mean_test_score"],
                           gs.cv_results_["std_test_score"])}

    # Pooled out-of-fold errors at the chosen C (count + Wilson 95% interval)
    best_pipe = clone(pipe).set_params(clf__C=out["lr_best_C"])
    oof = cross_val_predict(best_pipe, X, y, cv=cv)
    k = int((oof != y).sum())
    fp = int(((oof == 1) & (y == 0)).sum())
    fn = int(((oof == 0) & (y == 1)).sum())
    out["lr_oof"] = {"errors": k, "n": int(len(y)), "error": k / len(y),
                     "wilson95": list(wilson(k, len(y))),
                     "false_forged_alarms": fp, "missed_forgeries": fn}

    # Nested CV: honest estimate of the whole "tune C then fit" procedure
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 1)
    inner = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 2)
    nested = cross_validate(GridSearchCV(pipe, grid, cv=inner, scoring="accuracy"),
                            X, y, cv=outer, scoring=["accuracy", "roc_auc"], n_jobs=-1)
    out["lr_nested_cv_error"] = fold_summary(1 - nested["test_accuracy"])
    out["lr_nested_cv_auc"] = fold_summary(nested["test_roc_auc"])

    # Seed stability: all 10 repeats of 5-fold CV at the chosen C (not the best one)
    reps = []
    for r in range(10):
        cv_r = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 100 + r)
        reps.append(1 - cross_val_score(best_pipe, X, y, cv=cv_r, scoring="accuracy").mean())
    out["lr_repeated_cv_error"] = {"per_repeat": np.round(reps, 5).tolist(),
                                   "mean": float(np.mean(reps)), "min": float(np.min(reps)),
                                   "max": float(np.max(reps))}

    # Leakage sensitivity: keep 8-row runs of the original file order in one fold
    groups = pseudo_specimen_groups(dev, block=8)
    gcv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    leak = {}
    for name, (p, g) in models().items():
        if name == "Random forest":  # untuned here (forests come later in the course)
            p, g = clone(p).set_params(max_features=2), None
        for label, splitter in [("random", cv), ("grouped", gcv)]:
            accs = []
            split_kw = {"groups": groups} if label == "grouped" else {}
            for tr, te in splitter.split(X, y, **split_kw):
                if g is None:
                    m = clone(p).fit(X[tr], y[tr])
                else:  # tune inside the outer fold, using the same kind of folds
                    m = GridSearchCV(p, g, cv=splitter, scoring="accuracy")
                    m.fit(X[tr], y[tr], **({"groups": groups[tr]} if split_kw else {}))
                accs.append(m.score(X[te], y[te]))
            leak.setdefault(name, {})[label] = fold_summary(1 - np.array(accs))
    out["leakage_check_block8"] = leak

    # kNN with k = 1 is the most sensitive to near-copies of the same specimen
    knn1 = clone(models()["kNN"][0]).set_params(clf__n_neighbors=1)
    out["knn1_random_vs_grouped"] = {
        "random": fold_summary(1 - cross_val_score(knn1, X, y, cv=cv)),
        "grouped": fold_summary(1 - cross_val_score(knn1, X, y, cv=gcv, groups=groups))}

    (RESULTS / "baseline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
