# EMTAS v1.0

Evaluation Metrics–Triad Analysis System (EMTAS) — a Python desktop application for inter‑rater reliability, agreement, and binary classification diagnostics.

## Features

- ICC(2,1) and ICC(2,k) with bootstrap confidence intervals.
- Fleiss’ kappa (multi‑rater), Cohen’s kappa (two raters), Krippendorff’s alpha.
- Confusion‑matrix suite: Accuracy, Sensitivity, Specificity, PPV, NPV, F1, Balanced Accuracy, Youden's J, MCC.
- Optional ROC AUC and Average Precision when prediction scores are supplied.
- CSV import with auto‑population and header mapping.
- Export results to `.txt` and `.json`; save figures (ROC/PR/confusion matrix).
- Batch processing of parallel templates for group/team summaries.
- Editable name/group fields for clearer multi‑group reports.

## Install

- Python 3.9+
- Recommended packages: `PyQt5`, `numpy`, `scipy`, `matplotlib`

Example (venv shown for clarity):

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install PyQt5 numpy scipy matplotlib
```

## Run

```bash
python EMTAS_v1.0.py
```

## Usage notes

- Import a CSV to auto‑populate the tables.
- Supply continuous scores to enable ROC/PR plots; with labels only, AUC and AP are omitted by design.
- Use **Batch** to summarize folders of CSV templates (e.g., multiple groups).

## License

See `LICENSE` (MIT).

## Citation

See `CITATION.cff`. The conceptual background aligns with OQPM and related work listed in `paper.bib`.
