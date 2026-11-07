# ESE 5420 final project: forged-banknote detection

Maanas and Advit. Compares logistic regression (baseline) and kNN, with random forests,
gradient boosting and a small neural network as later-course extensions, on the
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
| `src/learning_curves.py` | 50 draws x 11 training sizes x 5 methods, plus block-grouped draws -> `results/learning_curves_*.csv` |
| `src/analyse.py` | Paired differences, notes needed to reach 2%/1%, chosen hyperparameters, forest check |
| `src/test_eval.py` | The one-time test-set evaluation -> `results/test_*.{json,csv}` |
| `src/make_figure.py` | `figures/baseline.png` (baseline checkpoint) |
| `src/make_paper_figures.py` | `figures/{learning_curves,grouped,test_roc}.pdf` |
| `src/make_tables.py` | LaTeX tables and `numbers.tex` (every number quoted in the text) -> `paper/tables/` |
| `paper/` | NeurIPS-style LaTeX: `main.tex` (final paper), `early_draft.tex` (16 Nov checkpoint) |
| `submissions/` | Canvas uploads (`baseline.txt`, `baseline.png`, `early_draft.pdf`, `final_paper.pdf`) |

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/baseline.py
.venv/bin/python src/learning_curves.py      # ~20 min on 8 cores
.venv/bin/python src/analyse.py
.venv/bin/python src/test_eval.py            # touches the test set; run once, last
.venv/bin/python src/make_figure.py
.venv/bin/python src/make_paper_figures.py
.venv/bin/python src/make_tables.py
cd paper && tectonic -X compile main.tex
```

## Rules we keep

- The 270-row test set in `data/split.csv` is used **once**, by `src/test_eval.py`, after
  every modelling choice was fixed on the development rows (first run: 7 Nov 2026).
  Every other script uses `load_dev()`, which never returns test rows.
- Scaling lives inside each sklearn `Pipeline`, so it is re-fitted on every training fold.
- Every reported number is the mean over all folds/draws, never the best seed.

## Data credit

Volker Lohweg (2012), *Banknote Authentication*, UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/267/banknote+authentication, licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). We removed 24 exact duplicate rows;
all figures are derived from the data.
