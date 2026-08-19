# Submission package — Track 2, AI for MEMS

Four required deliverables, mapped to the submission requirements.

| # | Requirement | File | Status |
|---|---|---|---|
| 1 | One-Page Pitch | `01_one_page_pitch.md` | ready |
| 2 | Technical Report (max 3 pages) | `02_technical_report.tex` | **needs compiling** |
| 3 | Results & Validation | `03_results_and_validation.md` | ready |
| 4 | Code & Data Availability | `04_code_and_data_statement.md` | ready |

## Building the technical report

`pdflatex` is **not installed in this environment**, so the report has not been
compiled here. It is written for `IEEEtran` (conference, two-column) and pulls
figures from `../figures/` and tables from `../figures/tables.tex`.

Easiest route — Overleaf:

1. New project → upload `02_technical_report.tex`
2. Upload the five PDFs from `../figures/` plus `tables.tex`
3. Overleaf provides `IEEEtran.cls` automatically; compile with pdfLaTeX (twice)

Locally, if you have TeX Live:

```bash
cd submission
pdflatex 02_technical_report.tex && pdflatex 02_technical_report.tex
```

Required packages: `IEEEtran`, `booktabs`, `graphicx`, `amsmath`, `balance`.

## Figures used

| Figure | File | Panels |
|---|---|---|
| Fig. 1 | `Fig1_solver_and_inverse_design.pdf` | profile rediscovery; exponent reversal ×2 |
| Fig. 2 | `Fig2_ai_design_and_control.pdf` | RL adaptivity; warm start; method comparison |
| Fig. 3 | `Fig3_fold_structure.pdf` | the fold; deflection mode at pull-in |
| Fig. 4 | `Fig4_validation.pdf` | error across sources; 2025 measured device |
| Fig. 5 | `Fig5_amortized_sweep.pdf` | spec sweep; constraint-layer error reduction |

All are IEEE-width vector PDFs (7.16 in double column), 8 pt fonts, ticks
inward, colour-blind-safe palette, Type-42 fonts.

## Before submitting — two things to check

1. **Figure 2(a) shows ρ = +0.96; the text elsewhere quotes ρ = +0.991.** They
   come from different training runs — the figure used 1024 envs × 1200
   iterations because the full 4096 × 2000 configuration exhausted the 8 GB
   GPU. Either caption the figure with the configuration used, or re-run at
   full scale on CPU so the two agree. Do not quote +0.991 beside a figure
   reading +0.96.
2. **Table III mixes cost types** — 15 s of optimization, 594 s of surrogate
   *training*, 0.04 ms of *inference*. The caption should distinguish one-time
   training cost from per-design cost, or it reads as unfair to the surrogates.

## Page budget

The report currently runs slightly over three pages with all five figures. If
trimming is needed, Fig. 5 is the most expendable — its key numbers (0.75% spec
error, 0.04 ms) already appear in the text and in Table III.
