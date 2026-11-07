"""Write LaTeX tables (and numbers.tex, one macro per number quoted in the text) for the
paper from results/. Never hand-edit the outputs.

Run:  .venv/bin/python src/make_tables.py
"""
import ast
import json
import math

import numpy as np
import pandas as pd

from common import FEATURES, RESULTS, ROOT, TARGET, load_clean, load_dev, load_raw

OUT = ROOT / "paper" / "tables"


def pct(v, d=2):
    return f"{100 * v:.{d}f}\\%"


def data_table():
    raw = load_raw()
    clean, n_dup = load_clean()
    dev = load_dev()
    n_test = len(clean) - len(dev)
    rows = [("Raw UCI file", raw.shape, raw[TARGET].mean()),
            ("After removing exact duplicates", clean.shape, clean[TARGET].mean()),
            ("Development set (80\\%)", dev.shape, dev[TARGET].mean())]
    test_share = (clean[TARGET].sum() - dev[TARGET].sum()) / n_test
    body = "\n".join(f"{name} & {r} & {c - 2} & {pct(s, 1)} \\\\"
                     for name, (r, c), s in rows)
    body += f"\nTest set (20\\%, held out) & {n_test} & 4 & {pct(test_share, 1)} \\\\"
    return (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Stage & Rows & Inputs & Forged \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        f"% {n_dup} duplicate rows dropped\n"
    )


def head_table():
    clean, _ = load_clean()
    h = clean.groupby(TARGET).head(2).sort_values(TARGET)
    lines = "\n".join(" & ".join(f"{v:.3f}" for v in r[FEATURES]) + f" & {int(r[TARGET])} \\\\"
                      for _, r in h.iterrows())
    return ("\\begin{tabular}{rrrrc}\n\\toprule\n"
            "Variance & Skewness & Kurtosis & Entropy & Class \\\\\n\\midrule\n"
            f"{lines}\n\\bottomrule\n\\end{{tabular}}\n")


def baseline_table(b):
    cv, nest, oof = b["lr_cv_error_at_best_C"], b["lr_nested_cv_error"], b["lr_oof"]
    rep = b["lr_repeated_cv_error"]
    lo, hi = oof["wilson95"]
    rows = [
        ("Majority class", pct(b["majority_cv_error"]["mean"], 1), "--", "--"),
        (f"Logistic regression, $C={b['lr_best_C']:g}$ (5-fold CV)",
         pct(cv["mean"]), pct(cv["se"]), f"{oof['errors']}/{oof['n']}; 95\\% CI {pct(lo)}--{pct(hi)}"),
        ("Logistic regression, nested 5$\\times$5 CV", pct(nest["mean"]), pct(nest["se"]),
         f"AUC {b['lr_nested_cv_auc']['mean']:.4f}"),
        ("Logistic regression, 10 repeats of 5-fold", pct(rep["mean"]), "--",
         f"range {pct(rep['min'])}--{pct(rep['max'])}"),
    ]
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return ("\\begin{tabular}{lrrl}\n\\toprule\n"
            "Model / estimate & Error & SE & Notes \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n")


def learning_table():
    """Early-draft table: the sizes and methods of the 16 November draft (same seeds,
    so the numbers are unchanged by the later, larger run)."""
    s = pd.read_csv(RESULTS / "learning_curves_summary.csv")
    s = s[s.n_train.isin([20, 50, 100, 200, 400, 800])]
    order = ["Logistic regression", "kNN", "Random forest"]
    piv = s.pivot(index="n_train", columns="method")
    lines = []
    for n, r in piv.iterrows():
        best = min(order, key=lambda m: r[("mean", m)])
        cells = []
        for m in order:
            txt = f"{100 * r[('mean', m)]:.2f} $\\pm$ {100 * r[('se', m)]:.2f}"
            cells.append(f"\\textbf{{{txt}}}" if m == best else txt)
        lines.append(f"{n} & " + " & ".join(cells) + " \\\\")
    return ("\\begin{tabular}{rccc}\n\\toprule\n"
            "$n$ & Logistic regression & $k$NN & Random forest$^\\dagger$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


ORDER = ["Logistic regression", "kNN", "Random forest", "Gradient boosting", "Neural network"]
EXT = {"Random forest", "Gradient boosting", "Neural network"}


def label(m):
    return {"kNN": "$k$NN"}.get(m, m) + ("$^\\dagger$" if m in EXT else "")


def tabular(spec, header, lines):
    return (f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def learning_full_table():
    s = pd.read_csv(RESULTS / "learning_curves_summary.csv")
    piv = s.pivot(index="n_train", columns="method")
    lines = []
    for n, r in piv.iterrows():
        best = min(ORDER, key=lambda m: r[("mean", m)])
        cells = []
        for m in ORDER:
            txt = f"{100 * r[('mean', m)]:.2f} $\\pm$ {100 * r[('se', m)]:.2f}"
            cells.append(f"\\textbf{{{txt}}}" if m == best else txt)
        lines.append(f"{n} & " + " & ".join(cells) + " \\\\")
    return tabular("r" + "c" * len(ORDER), "$n$ & " + " & ".join(label(m) for m in ORDER), lines)


def fmt_n(v):
    if math.isinf(v):
        return "$>$800"
    return f"{v:.0f}"


def reach_table():
    r = pd.read_csv(RESULTS / "n_to_reach.csv")
    lines = []
    for m in ORDER:
        cells = []
        for t in (0.02, 0.01):
            x = r[(r.method == m) & np.isclose(r.target, t)].iloc[0]
            ci = f"{fmt_n(x.lo95)}--{fmt_n(x.hi95)}"
            cells += [fmt_n(x.n_needed), ci]
        lines.append(f"{label(m)} & " + " & ".join(cells) + " \\\\")
    return tabular("lrcrc", "Method & \\multicolumn{2}{c}{2\\% error} & \\multicolumn{2}{c}{1\\% error} \\\\\n"
                   " & $n$ & 95\\% CI & $n$ & 95\\% CI", lines)


def paired_table(sizes=(10, 20, 50, 100, 200, 400, 800)):
    p = pd.read_csv(RESULTS / "paired_vs_lr.csv")
    lines = []
    for m in ORDER[1:]:
        cells = []
        for n in sizes:
            x = p[(p.method == m) & (p.n_train == n)].iloc[0]
            d, se = 100 * x.mean_diff, 100 * x.se
            txt = f"{d:+.2f}".replace("-", "$-$")
            if abs(d) > 2 * se:
                txt = f"\\textbf{{{txt}}}"
            cells.append(f"{txt} ({x.wins}/{x.losses})")
        lines.append(f"{label(m)} & " + " & ".join(cells) + " \\\\")
    return tabular("l" + "c" * len(sizes), "$n$ & " + " & ".join(str(n) for n in sizes), lines)


def test_table(te):
    lines = []
    for m in ORDER:
        x = te["methods"][m]
        lo, hi = x["wilson95"]
        mc = x.get("mcnemar_vs_lr")
        pv = "--" if mc is None else (f"{mc['p_exact']:.2f}" if mc["p_exact"] < 0.995 else "1.00")
        lines.append(f"{label(m)} & {100 * x['dev_cv_error']:.2f} & {x['test_errors']}/{te['n_test']} & "
                     f"{100 * x['test_error']:.2f} & {100 * lo:.1f}--{100 * hi:.1f} & "
                     f"{x['missed_forgeries']} & {x['false_alarms']} & {x['auc']:.4f} & {pv} \\\\")
    return tabular("lrrrcrrcr", "Method & Dev CV (\\%) & Errors & Test (\\%) & 95\\% CI & Missed & False al. "
                   "& AUC & $p$ vs LR", lines)


def grouped_table():
    s = pd.read_csv(RESULTS / "learning_curves_summary.csv")
    g = pd.read_csv(RESULTS / "learning_curves_grouped_summary.csv")
    lines = []
    for n in sorted(g.n_train.unique()):
        cells = []
        for m in ["Logistic regression", "kNN"]:
            a = s[(s.method == m) & (s.n_train == n)].iloc[0]
            b = g[(g.method == m) & (g.n_train == n)].iloc[0]
            cells += [f"{100 * a['mean']:.2f} $\\pm$ {100 * a.se:.2f}",
                      f"{100 * b['mean']:.2f} $\\pm$ {100 * b.se:.2f}"]
        lines.append(f"{n} & " + " & ".join(cells) + " \\\\")
    return tabular("rcccc", " & \\multicolumn{2}{c}{Logistic regression} & \\multicolumn{2}{c}{$k$NN} \\\\\n"
                   "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n$n$ & random & grouped & random & grouped", lines)


def pretty_params(m, d):
    d = ast.literal_eval(d.replace("np.float64(", "(").replace("np.int64(", "("))
    if m == "Logistic regression":
        return f"$C={d['clf__C']:.3g}$"
    if m == "kNN":
        return f"$k={d['clf__n_neighbors']}$"
    if m == "Random forest":
        return f"$m={d['max_features']}$, leaf {d['min_samples_leaf']}"
    if m == "Gradient boosting":
        return f"$\\eta={d['learning_rate']:g}$, depth {d['max_depth']}"
    return f"{d['clf__hidden_layer_sizes'][0]} units, $\\alpha={d['clf__alpha']:g}$"


def params_table(sizes=(10, 50, 200, 800)):
    c = pd.read_csv(RESULTS / "chosen_params.csv")
    lines = []
    for m in ORDER:
        cells = []
        for n in sizes:
            x = c[(c.method == m) & (c.n_train == n)].iloc[0]
            cells.append(f"{pretty_params(m, x.most_common)} ({100 * x.share:.0f}\\%)")
        lines.append(f"{label(m)} & " + " & ".join(cells) + " \\\\")
    return tabular("l" + "l" * len(sizes), "Method & " + " & ".join(f"$n={n}$" for n in sizes), lines)


def forest_setting(b):
    out = f"$m={b['max_features']}$, leaf {b['min_samples_leaf']}, {b['n_estimators']} trees"
    return out + (f", {b['criterion']}" if "criterion" in b else "")


def forest_table(f):
    lines = [f"{k} & {v['n_configs']} & {100 * v['cv_error']:.2f} $\\pm$ {100 * v['se']:.2f} & "
             + forest_setting(v["best"]) + " \\\\"
             for k, v in f.items() if k != "oob_vs_trees"]
    grid = tabular("lrcl", "Grid & Configs & CV error (\\%) & Best setting", lines)
    oob = tabular("l" + "r" * len(f["oob_vs_trees"]),
                  "Trees & " + " & ".join(str(o["trees"]) for o in f["oob_vs_trees"]),
                  ["OOB error (\\%) & " + " & ".join(f"{100 * o['oob_error']:.2f}"
                                                   for o in f["oob_vs_trees"]) + " \\\\"])
    return grid + "\n\\vspace{4pt}\n" + oob


def numbers(b, te):
    """Macros for numbers quoted in running text, so the prose can never drift."""
    s = pd.read_csv(RESULTS / "learning_curves_summary.csv")
    p = pd.read_csv(RESULTS / "paired_vs_lr.csv")
    r = pd.read_csv(RESULTS / "n_to_reach.csv")
    tag = {"Logistic regression": "LR", "kNN": "KNN", "Random forest": "RF",
           "Gradient boosting": "GB", "Neural network": "NN"}
    word = {10: "Ten", 20: "Twenty", 30: "Thirty", 50: "Fifty", 100: "Hundred", 150: "OneFifty",
            200: "TwoHundred", 300: "ThreeHundred", 400: "FourHundred", 600: "SixHundred",
            800: "EightHundred"}
    m = {"LRcv": pct(b["lr_cv_error_at_best_C"]["mean"]), "LRcvse": pct(b["lr_cv_error_at_best_C"]["se"]),
         "LRnested": pct(b["lr_nested_cv_error"]["mean"]), "LRnestedse": pct(b["lr_nested_cv_error"]["se"])}
    for _, x in s.iterrows():
        m[f"err{tag[x.method]}{word[x.n_train]}"] = f"{100 * x['mean']:.2f}\\%"
    for _, x in p.iterrows():
        m[f"diff{tag[x.method]}{word[x.n_train]}"] = f"{100 * x.mean_diff:+.2f}"
        m[f"se{tag[x.method]}{word[x.n_train]}"] = f"{100 * x.se:.2f}"
        m[f"wins{tag[x.method]}{word[x.n_train]}"] = str(x.wins)
    for _, x in r.iterrows():
        m[f"reach{tag[x.method]}{'Two' if np.isclose(x.target, 0.02) else 'One'}"] = fmt_n(x.n_needed)
    g = pd.read_csv(RESULTS / "learning_curves_grouped_summary.csv")
    for _, x in g.iterrows():
        m[f"gerr{tag[x.method]}{word[x.n_train]}"] = f"{100 * x['mean']:.2f}\\%"
    for k, x in te["methods"].items():
        m[f"test{tag[k]}"] = f"{100 * x['test_error']:.2f}\\%"
        m[f"testerr{tag[k]}"] = str(x["test_errors"])
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = json.loads((RESULTS / "baseline.json").read_text())
    te = json.loads((RESULTS / "test_eval.json").read_text())
    f = json.loads((RESULTS / "forest_check.json").read_text())
    for name, tex in [("data", data_table()), ("head", head_table()),
                      ("baseline", baseline_table(b)), ("learning", learning_table()),
                      ("learning_full", learning_full_table()), ("reach", reach_table()),
                      ("paired", paired_table()), ("test", test_table(te)),
                      ("grouped", grouped_table()), ("params", params_table()),
                      ("forest", forest_table(f)), ("numbers", numbers(b, te))]:
        (OUT / f"{name}.tex").write_text("% generated by src/make_tables.py\n" + tex)
        print("wrote", OUT / f"{name}.tex")


if __name__ == "__main__":
    main()
