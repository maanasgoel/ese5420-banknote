"""Figures for the final paper (figures/*.pdf):

* learning_curves.pdf: (a) mean error vs training notes for all five methods,
  (b) paired difference from logistic regression, mean +/- 2 SE over 50 draws.
* grouped.pdf: random vs block-grouped draws for the two core methods.
* test_roc.pdf: ROC curves on the 270 test notes (appendix).

Run:  .venv/bin/python src/make_paper_figures.py
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402

ORDER = ["Logistic regression", "kNN", "Random forest", "Gradient boosting", "Neural network"]
SHORT = {"Logistic regression": "Logistic reg.", "kNN": "kNN", "Random forest": "Random forest",
         "Gradient boosting": "Grad. boosting", "Neural network": "Neural net"}
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKERS = dict(zip(ORDER, ["o", "s", "^", "D", "v"]))
CORE = ORDER[:2]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
pct = FuncFormatter(lambda v, _: f"{v:g}%")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


LABELLED = [10, 20, 50, 100, 200, 400, 800]


def log_n_axis(ax, sizes):
    """Ticks at every size, labels only at roughly doubling sizes so they do not collide."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(n) if n in LABELLED else "" for n in sizes])


def line_kw(m):
    core = m in CORE
    return dict(color=COLORS[m], lw=1.9 if core else 1.3, ls="-" if core else (0, (4, 2)),
                marker=MARKERS[m], ms=4.5 if core else 3.8, mec="white", mew=0.8,
                zorder=3 if core else 2)


def learning_figure(s, p):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    sizes = sorted(s.n_train.unique())
    floor = 0.05  # %: a mean of exactly 0 cannot sit on a log axis
    for m in ORDER:
        g = s[s.method == m].sort_values("n_train")
        a.plot(g.n_train, 100 * g["mean"], label=SHORT[m], **line_kw(m))
        zero = g[g["mean"] == 0]
        for n in zero.n_train:
            a.annotate("", xy=(n, floor), xytext=(n, floor * 1.9),
                       arrowprops=dict(arrowstyle="-|>", color=COLORS[m], lw=1.1))
    a.set_ylim(floor, 35)
    a.axhline(1, color=INK2, lw=0.8, ls=":")
    a.text(10.5, 1.06, "1%", fontsize=7, color=INK2, va="bottom")
    a.set_yscale("log")
    log_n_axis(a, sizes)
    a.yaxis.set_major_formatter(pct)
    a.set_xlabel("Training notes $n$ (log scale)", fontsize=8.5)
    a.set_ylabel("Mean error, 50 draws (log)", fontsize=8.5)
    a.set_title("(a) Learning curves", loc="left", fontsize=9)
    a.legend(frameon=False, fontsize=7, loc="lower left", handlelength=2.6)
    style(a)

    b.axhline(0, color=INK2, lw=0.9)
    for m in ORDER[1:]:
        g = p[p.method == m].sort_values("n_train")
        b.errorbar(g.n_train, 100 * g.mean_diff, yerr=200 * g.se, capsize=1.8,
                   elinewidth=0.9, **line_kw(m))
    b.set_yscale("symlog", linthresh=1, linscale=1.2)
    b.yaxis.set_major_locator(FixedLocator([-3, -1, -0.5, 0, 0.5, 1, 3, 10]))
    b.yaxis.set_minor_locator(NullLocator())
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+g}" if v else "0"))
    log_n_axis(b, sizes)
    b.set_xlabel("Training notes $n$ (log scale)", fontsize=8.5)
    b.set_ylabel("Error minus logistic reg. (pp)", fontsize=8.5)
    b.set_title("(b) Paired difference from baseline, $\\pm$2 SE", loc="left", fontsize=9)
    b.text(0.98, 0.96, "above 0: worse than baseline", transform=b.transAxes, ha="right",
           va="top", fontsize=7, color=INK2)
    b.text(0.98, 0.04, "below 0: better", transform=b.transAxes, ha="right", va="bottom",
           fontsize=7, color=INK2)
    style(b)
    for ax in (a, b):
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(w_pad=1.5)
    return fig


def grouped_figure(s, sg):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m in CORE:
        g = s[(s.method == m) & s.n_train.isin(sg.n_train.unique())].sort_values("n_train")
        h = sg[sg.method == m].sort_values("n_train")
        ax.plot(g.n_train, 100 * g["mean"], color=COLORS[m], lw=1.8, marker=MARKERS[m], ms=4.5,
                mec="white", mew=0.8, label=f"{SHORT[m]}, random draws")
        ax.errorbar(h.n_train, 100 * h["mean"], yerr=200 * h.se, color=COLORS[m], lw=1.3,
                    ls=(0, (4, 2)), marker=MARKERS[m], ms=4.5, mfc="white", mew=1.0,
                    capsize=1.8, label=f"{SHORT[m]}, grouped draws")
    ax.set_yscale("log")
    log_n_axis(ax, sorted(sg.n_train.unique()))
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Training notes $n$ (log scale)", fontsize=8.5)
    ax.set_ylabel("Mean error (log)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    style(ax)
    fig.tight_layout()
    return fig


def roc_figure(roc, te):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in ORDER:
        g = roc[roc.method == m]
        auc = te["methods"][m]["auc"]
        ax.step(100 * g.fpr, 100 * g.tpr, where="post", label=f"{SHORT[m]} (AUC {auc:.4f})",
                color=COLORS[m], lw=1.6 if m in CORE else 1.1,
                ls="-" if m in CORE else (0, (4, 2)))
    ax.set_xlim(-0.5, 10); ax.set_ylim(85, 100.5)
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("False-alarm rate (zoomed to 0-10%)", fontsize=8.5)
    ax.set_ylabel("Forgeries caught", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    style(ax)
    fig.tight_layout()
    return fig


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "pdf.fonttype": 42})
    s = pd.read_csv(RESULTS / "learning_curves_summary.csv")
    p = pd.read_csv(RESULTS / "paired_vs_lr.csv")
    FIGURES.mkdir(exist_ok=True)
    figs = {"learning_curves": learning_figure(s, p)}
    gpath = RESULTS / "learning_curves_grouped_summary.csv"
    if gpath.exists():
        figs["grouped"] = grouped_figure(s, pd.read_csv(gpath))
    if (RESULTS / "test_eval.json").exists():
        te = json.loads((RESULTS / "test_eval.json").read_text())
        figs["test_roc"] = roc_figure(pd.read_csv(RESULTS / "test_roc.csv"), te)
    for name, fig in figs.items():
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES / f"{name}.{ext}", dpi=250)
        print("wrote", FIGURES / f"{name}.pdf")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
