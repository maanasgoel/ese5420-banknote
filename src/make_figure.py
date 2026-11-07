"""baseline.png: (a) logistic-regression CV error across the C grid,
(b) learning curves for all three methods on held-out development rows.

Run:  .venv/bin/python src/make_figure.py
"""
import json

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from common import FIGURES, N_FOLDS, RESULTS  # noqa: E402

COLORS = {"Logistic regression": "#2a78d6", "kNN": "#eb6834", "Random forest": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
pct = FuncFormatter(lambda v, _: f"{v:g}%")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    base = json.loads((RESULTS / "baseline.json").read_text())
    lc = pd.read_csv(RESULTS / "learning_curves_summary.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})

    # (a) C grid
    byc = base["lr_cv_error_by_C"]
    cs = [float(c) for c in byc]
    m = [100 * byc[c]["mean"] for c in byc]
    se = [100 * byc[c]["se"] for c in byc]
    col = COLORS["Logistic regression"]
    a.errorbar(cs, m, yerr=se, color=col, lw=2, marker="o", ms=6, capsize=3,
               mec="white", mew=1.2)
    best = base["lr_best_C"]
    a.axvline(best, color=INK2, lw=1, ls=":")
    a.annotate(f"chosen C = {best:g}\n{100 * base['lr_cv_error_at_best_C']['mean']:.2f}% "
               f"± {100 * base['lr_cv_error_at_best_C']['se']:.2f}% (SE)",
               xy=(best, 100 * base["lr_cv_error_at_best_C"]["mean"]),
               xytext=(0.4, 2.2), textcoords="data", fontsize=9, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a.axhline(100 * base["majority_cv_error"]["mean"], color=INK2, lw=1, ls="--")
    a.text(1.2e-3, 100 * base["majority_cv_error"]["mean"] * 0.8, "majority class (45.3%)",
           fontsize=8.5, color=INK2, va="top")
    a.set_xscale("log"); a.set_yscale("log")
    a.yaxis.set_major_formatter(pct)
    a.set_xlabel("Inverse regularisation strength C (log scale)")
    a.set_ylabel(f"{N_FOLDS}-fold CV error (log scale)")
    a.set_title("(a) Logistic regression baseline: tuning C", loc="left", fontsize=11)
    style(a)

    # (b) learning curves
    floor = 0.05  # % ; draws with zero errors are drawn at the axis floor
    dodge = {"Logistic regression": 0.94, "kNN": 1.0, "Random forest": 1.06}
    label_dy = {"Logistic regression": 9, "kNN": 0, "Random forest": -9}  # points
    for name, c in COLORS.items():
        g = lc[lc.method == name].sort_values("n_train")
        x = g.n_train * dodge[name]
        lo = (100 * g.q10).clip(lower=floor)
        b.vlines(x, lo, 100 * g.q90, color=c, lw=1.2, alpha=0.55)
        b.plot(x, 100 * g["mean"], color=c, lw=2, marker="o", ms=6,
               mec="white", mew=1.2, label=name)
        last = g.iloc[-1]
        b.annotate(name, (last.n_train * dodge[name], 100 * last["mean"]),
                   xytext=(9, label_dy[name]), textcoords="offset points",
                   color=INK, fontsize=9, va="center")
    b.axhline(1, color=INK2, lw=1, ls="--")
    b.text(21, 1.07, "1% error", fontsize=8.5, color=INK2, va="bottom")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xticks(sorted(lc.n_train.unique()))
    b.set_xticklabels([str(n) for n in sorted(lc.n_train.unique())])
    b.minorticks_off()
    b.yaxis.set_major_formatter(pct)
    b.set_xlim(17, 1900)
    b.set_ylim(floor, 30)
    b.set_xlabel("Training notes (log scale)")
    b.set_ylabel("Error on remaining dev rows (log scale)")
    b.set_title("(b) Learning curves: mean of 50 draws, bars = 10th-90th pct",
                loc="left", fontsize=11)
    b.legend(frameon=False, fontsize=9, loc="lower left")
    style(b)

    fig.text(0.01, 0.005, "UCI Banknote Authentication (Lohweg 2012, CC BY 4.0); 1,078 "
             "development rows after de-duplication; test set (270 rows) unused.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "baseline.png", dpi=200)
    print("wrote", FIGURES / "baseline.png")


if __name__ == "__main__":
    main()
