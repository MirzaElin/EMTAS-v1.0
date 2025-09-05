---
title: "EMTAS v1.0: Evaluation Metrics–Triad Analysis System for reliability and binary classification"
tags:
  - psychometrics
  - reliability
  - agreement
  - classification
  - Python
authors:
  - name: Mirza Niaz Zaman Elin
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Kharkiv National Medical University, Kharkiv, Ukraine
    index: 1
date: 2025-09-05
---

# Summary

Evaluation Metrics–Triad Analysis System (EMTAS) is a Python desktop toolkit for psychometric evaluation and classification diagnostics. The software focuses on three complementary perspectives that frequently arise in organizational and behavioral measurement: inter‑rater reliability, agreement and consistency, and binary classification performance. EMTAS provides ready‑to‑run workflows for intra‑class correlation (ICC), Fleiss’ kappa, Cohen’s kappa, Krippendorff’s alpha, and a confusion‑matrix suite including accuracy, sensitivity, specificity, predictive values, F1, balanced accuracy, Youden’s J, and Matthews correlation. When continuous prediction scores are present, EMTAS also produces ROC and precision–recall curves with AUC and average precision.

The conceptual backdrop stems from Organizational Quantum Psychometric Modelling (OQPM), which frames leader–staff relations as coupled systems that can display superposition, entanglement, and synchronization. EMTAS operationalizes a practical subset of that vision by delivering coherent reliability and diagnostic tooling for OQPM‑style paired or entangled group assessments [@Elin2025].

# Statement of need

Workflows in organizational psychology, adjudication studies, medical annotation, and human‑in‑the‑loop pipelines often require (i) multiple reliability coefficients within a single interface, (ii) side‑by‑side visual diagnostics, and (iii) repeatable batch runs over parallel templates representing teams or time windows. Fragmented tooling increases friction and produces inconsistent reporting. EMTAS addresses these needs with a cross‑platform application that reads CSV templates, auto‑populates tables, computes the requested indices, saves figures to disk, and exports results to plain text and JSON for downstream analysis.

# State of the field and related work

Classical psychometrics offers long‑standing perspectives on reliability and item evaluation [@Cappelleri2014]. Organizational psychology continues to motivate rigorous measurement and interpretation of human systems [@Schein2015]. Quantum‑inspired thinking has opened alternative formalisms for cognition and decision modeling [@Aerts2010; @Wang2013; @HavenKhrennikov2013; @Haven2018], which in turn motivates tools capable of aligning traditional indices with scenarios emphasized by OQPM [@Elin2025]. EMTAS contributes a unified interface that brings these measurement routines together and emphasizes reproducible summaries and figures.

# Software description

**Key capabilities**

* **Agreement and reliability**: ICC(2,1) and ICC(2,k) with bootstrap confidence intervals; Fleiss’ kappa (multi‑rater); Cohen’s kappa (two raters); Krippendorff’s alpha (general‑purpose reliability).
* **Binary classification**: Accuracy, Sensitivity, Specificity, PPV, NPV, F1, Balanced Accuracy, Youden’s J, and Matthews correlation; ROC AUC and Average Precision when continuous scores are supplied.
* **Visualization**: On‑demand ROC and PR curves saved as image files; optional confusion‑matrix plot.
* **Data I/O**: CSV import with header mapping and auto‑population; results export to TXT and JSON; batch processing over folders of parallel templates to summarize groups or teams.
* **Usability**: Editable group/name fields so multiple groups appear in summaries; templates for ICC and Fleiss workflows; consistent file‑save dialogs to select destinations explicitly.

# Design and implementation

EMTAS is implemented in Python with a Qt GUI. Computation modules are organized separately from UI code, enabling straightforward extension. Batch processing reads a directory of CSV templates and returns a concise table of per‑file metrics, enabling leader–staff or team‑level comparisons consistent with scenarios described in OQPM (§6.2) [@Elin2025].

# Validation

Synthetic and real‑world‑like templates are used to verify each statistic, with spot checks against closed‑form expectations and bootstrap intervals for ICC. The visualization layer reproduces ROC/PR behavior consistent with score separability. When only hard labels are present, AUC and average precision are intentionally withheld to avoid misleading estimates. Edge‑case handling includes empty rows, partial columns, and non‑numeric entries, all reported via GUI messages to support careful data entry.

# Availability

Source code, example templates, and documentation are included in the public repository. The project is distributed under an OSI‑approved license to meet JOSS requirements. The software design aligns with open, reproducible analysis for organizational research and neighboring domains.

# Acknowledgements

Conceptual alignment with OQPM and the EMTAS construct family is grounded in a published monograph that articulates superposition, entanglement, and synchronization in organizational settings [@Elin2025].

# References
