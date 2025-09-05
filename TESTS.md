# Tests

This repository includes basic unit tests covering the core computational routines exposed at module level in `EMTAS_v1.0.py`. Tests are designed to be independent of the GUI and require only standard scientific Python packages.

## What is covered
- Cohen’s kappa (`cohens_kappa`) on a small synthetic example with a known analytic result.
- Krippendorff’s alpha (nominal) (`krippendorff_alpha_nominal`) on perfect‑agreement data and on data with missing values.
- Spearman–Brown reliability averaging (`spearman_brown`) and ICC average‑measure helper (`icc_avg`).

## Running the test suite

Install dependencies and run with `pytest`:

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy pytest
pytest -q
```

The suite executes quickly and contains no GUI operations.
