# Submission package — ISMC 2026, Track 2 "AI for MEMS"

The 2nd International EDS MEMS Student Challenge, organised by the IEEE EDS
MEMS Technical Committee.

| # | Requirement | File | Status |
|---|---|---|---|
| 1 | One-Page Pitch | `01_one_page_pitch.md` | ready |
| 2 | Technical Report (max 3 pages) | `02_technical_report.tex` | **needs compiling** |
| 3 | Results & Validation (+ demo/notebook encouraged) | `03_results_and_validation.md`, `../sim/demo.ipynb` | ready |
| 4 | Code & Data Availability | `04_code_and_data_statement.md` | ready |

## Framing (read this before editing any of the four)

The judges are the MEMS Technical Committee — device people, presenting at a
device conference — not an ML audience. Track 2's rubric is Track 1's rubric
renamed slot for slot, and the slot now called *Relevance & Impact* (25%) was
previously *Novel Functional Capability*: what the **device** can now do that
it could not before.

Everything therefore leads with the device outcome and treats the method as the
mechanism that produced it:

* **41% lower drive voltage** from electrode geometry alone — no new material,
  process step or mask.
* **+12.8% usable travel** at **0.0% devices destroyed**.
* The method (differentiating through the fold) carries the *AI/Methodological
  Novelty* 25% and appears immediately after.

Do not revert to a method-first opening; it lands the impact argument in the
wrong rubric slot.

The report's section names — Methodology, Datasets, Model architecture,
Results — are taken verbatim from the call, in that order, so a reviewer
working from a checklist can tick every item from the headings alone. Keep them.

## Building the technical report

`pdflatex` is **not installed in this environment**, so the report has not been
compiled here. It is written for `IEEEtran` (conference, two-column) and pulls
figures from `../figures/` and tables from `../figures/tables.tex`.

Easiest route — Overleaf:

1. New project → upload `02_technical_report.tex`
2. Upload the six PDFs from `../figures/` plus `tables.tex`
3. Overleaf provides `IEEEtran.cls` automatically; compile with pdfLaTeX (twice)

Locally, with TeX Live:

```bash
pdflatex 02_technical_report.tex && pdflatex 02_technical_report.tex
```

Required packages: `IEEEtran`, `booktabs`, `graphicx`, `amsmath`, `amssymb`,
`balance`.

## Figures

| Figure | File | Role |
|---|---|---|
| Fig. 0 | `Fig0_architecture.pdf` | system architecture; also the pitch hero image |
| Fig. 1 | `Fig1_solver_and_inverse_design.pdf` | profile rediscovery; exponent reversal ×2 |
| Fig. 2 | `Fig2_ai_design_and_control.pdf` | RL adaptivity; warm start; method comparison |
| Fig. 3 | `Fig3_fold_structure.pdf` | the fold; deflection mode at pull-in |
| Fig. 4 | `Fig4_validation.pdf` | error across sources; 2025 measured device |
| Fig. 5 | `Fig5_amortized_sweep.pdf` | spec sweep; constraint-layer error reduction |

The report uses Figs. 0, 1, 2 and 4. Figs. 3 and 5 live in the Results &
Validation document, which has no page limit. All are IEEE-width vector PDFs,
7.16 in double column. Regenerate with `python make_figures.py` and
`python make_arch.py`.

**Figs. 1–5** are matplotlib plots: 8 pt fonts, ticks inward on all four sides,
colour-blind-safe Okabe-Ito palette, Type-42 fonts.

**Fig. 0 is different** — it is a block diagram, not a plot, so its source of
truth is a hand-authored SVG, `figures/Fig0_architecture.svg`. Edit that file;
`make_arch.py` only converts it to PDF and PNG. Two traps if you edit it:

* Do **not** nest `<tspan>` with `dx`/`dy` inside a `text-anchor="middle"`
  element. cairosvg ignores the tspan advance when computing the anchor and
  silently overlaps the glyphs. Give every string its own `<text>` with an
  explicit `x`.
* SVG collapses runs of whitespace, so you cannot pad items apart inside one
  `<text>`. Position them separately (as the three fold-system rows are).

Only DejaVu Sans is guaranteed to the converter, so check glyph coverage before
introducing a new symbol.

## Before submitting

1. **Eligibility — you have to do this one personally.** The 1st edition
   required teams of 1–3 students, **each with a faculty or industry mentor who
   is IEEE-affiliated**. Confirm the 2026 rules and line up a mentor early;
   nothing in this repository can substitute for it. Also confirm the 2026
   deadline and the submission address.
2. **Fig. 2(a) shows ρ = +0.96, and every document now says +0.96 to match.**
   An earlier draft quoted +0.991, which came from a different training run —
   the figure used 1024 envs × 1200 iterations because the full 4096 × 2000
   configuration exhausted the 8 GB GPU. Either caption the figure with the
   configuration used, or re-run at full scale on CPU. Do not reintroduce
   +0.991 beside a figure reading +0.96.
3. **Table III mixes cost types** — 15 s of optimization, 594 s of surrogate
   *training*, 0.04 ms of *inference*. The caption should distinguish one-time
   training cost from per-design cost, or it reads as unfair to the surrogates.
4. **Page budget.** The report runs slightly over three pages with four
   figures. Fig. 4 is the first to cut — the validation table carries the same
   information. Cut Fig. 0 last; it is doing the framing work.

## Numbers that appear in more than one place

Kept deliberately consistent; if you change one, change all:

| Quantity | Value | Source of truth |
|---|---|---|
| Baseline uniform-gap cantilever | 22.86 V | `inverse_design.py` at N=60 |
| Best design found | 13.60 V (−41%) | `inverse_design.py` |
| Amortized network | 13.636 V, 0.04 ms | `amortized_vs_converged.py` |
| Amortized spec error | 0.56% mean / 2.14% max, 17-point sweep | `gen_figdata2.py`, Fig. 5(b) |
| RL headroom recovered | 80.4 ± 4.9% (= +12.8% travel) | `analyze_rl2dof.py` |
| RL adaptation correlation | ρ = +0.96 | Fig. 2(a) run |

Note on the spec error: the coarser 5-point evaluation inside
`amortized_hard.py` reports 0.75%. The submission quotes the denser 17-point
sweep (0.56%) because that is what Fig. 5(b) plots. Both are stated in
`03_results_and_validation.md`, so the two numbers are traceable rather than
contradictory.

**Grid consistency matters.** Every headline voltage is measured at N=60
(`inverse_design.N_FINAL`). Comparing a design fitted at N=40 against one
fitted at N=60 once flipped the sign of a conclusion here. `demo.ipynb` uses
N=60 for the same reason.
