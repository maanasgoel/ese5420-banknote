# ESE 5420 final project: forged-banknote detection

Maanas and Advit. Compares logistic regression (baseline), kNN and random forests on the
UCI Banknote Authentication data. Because logistic regression is already below 1% error,
the main question (following the proposal feedback) is **how many training notes each
method needs to reach a given error rate**.

## Layout

| Path | What |
|---|---|
| `data/data_banknote_authentication.txt` | Raw UCI file (Lohweg 2012, CC BY 4.0) |
| `data/split.csv` | Fixed dev/test assignment per original row (`row_id` = 0-based line in raw file) |
| `src/common.py` | Loading, de-duplication, split, model pipelines and grids |
| `src/baseline.py` | Baseline checkpoint numbers -> `results/baseline.json` |
| `src/learning_curves.py` | 50 draws x training sizes x 3 methods -> `results/learning_curves_*.csv` |
| `src/make_figure.py` | `figures/baseline.png` |
| `src/make_tables.py` | LaTeX tables for the paper -> `paper/tables/` |
| `paper/` | NeurIPS-style LaTeX (`early_draft.tex`); build with `tectonic -X compile early_draft.tex` |
| `submissions/` | Canvas uploads (`baseline.txt`, `baseline.png`, `early_draft.pdf`) |

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/baseline.py
.venv/bin/python src/learning_curves.py      # ~5-10 min on 8 cores
.venv/bin/python src/make_figure.py
```

## Rules we keep

- The 270-row test set in `data/split.csv` is **not touched** until the final evaluation.
  `load_dev()` is the only loader the analysis scripts use, and it never returns test rows.
- Scaling lives inside each sklearn `Pipeline`, so it is re-fitted on every training fold.
- Every reported number is the mean over all folds/draws, never the best seed.

## Data credit

Volker Lohweg (2012), *Banknote Authentication*, UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/267/banknote+authentication, licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). We removed 24 exact duplicate rows;
all figures are derived from the data.
