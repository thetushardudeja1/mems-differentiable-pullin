# Submission package — ISMC 2026, Track 2 "AI for MEMS"

The 2nd International EDS MEMS Student Challenge, organised by the IEEE EDS
MEMS Technical Committee.

| # | Requirement | File | Status |
|---|---|---|---|
| 1 | One-Page Pitch | `01_one_page_pitch.tex` → `.pdf` | **1 page, compiled** |
| 2 | Technical Report (max 3 pages) | `02_technical_report.tex` → `.pdf` | **3 pages, compiled** |
| 3 | Results & Validation (+ demo/notebook encouraged) | `03_results_and_validation.md`, `../notebooks/MEMS_differentiable_pullin.ipynb` | ready |
| 4 | Code & Data Availability | `04_code_and_data_statement.md` | ready — repo is public |

Both PDFs build with `tectonic <file>.tex` (installed in the `tex` conda env)
or any pdfLaTeX. Page counts are hard limits and there is no slack in either —
check with `pypdf` after any edit.

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

## Building the documents

Both PDFs are compiled and committed. The LaTeX sources
(`01_one_page_pitch.tex`, `02_technical_report.tex`) and the generated table
fragments (`../figures/tables*.tex`) are kept **locally but not published** —
the repository ships the deliverables, not the toolchain. They are still in git
history:

```bash
git log --diff-filter=D --name-only -- submission/02_technical_report.tex
git checkout <sha> -- submission/02_technical_report.tex
```

To rebuild after editing, from this directory:

```bash
tectonic 02_technical_report.tex
```

`tectonic` is installed in the `tex` conda env and fetches `IEEEtran.cls` and
the rest on first run. Any pdfLaTeX with `IEEEtran`, `booktabs`, `graphicx`,
`amsmath`, `amssymb` and `balance` works too — run it twice for references.

## Figures

| Figure | File | Role |
|---|---|---|
| Fig. 0 | `Fig0_architecture.pdf` | system architecture; also the pitch hero image |
| Fig. 1 | `Fig1_solver_and_inverse_design.pdf` | profile rediscovery; exponent reversal ×2 |
| Fig. 2 | `Fig2_ai_design_and_control.pdf` | RL adaptivity; warm start; method comparison |
| Fig. 3 | `Fig3_fold_structure.pdf` | the fold; deflection mode at pull-in |
| Fig. 4 | `Fig4_validation.pdf` | error across sources; 2025 measured device |
| Fig. 5 | `Fig5_amortized_sweep.pdf` | spec sweep; constraint-layer error reduction |

**Where each one is actually used** (checked against the sources, not from
memory):

* **Report** — Figs. 1, 2 and 5.
* **Pitch** — `Fig0b_pitch_strip` as the hero image, plus Fig. 3.
* **Repository README** — `Fig0_architecture.png`.
* **Results & Validation** (no page limit) — Figs. 3, 4 and 5.

`Fig0c_architecture_1col` is unused by anything and is no longer published.
Figures are single-column panel grids (3.5 in) except Fig. 0, which is
double-column at 7.16 in. Regenerate with `python make_figures.py` and
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
2. **RESOLVED — the report's RL numbers are the full-scale run.**
   **80.4 ± 4.9%** headroom (seeds: 81.5 / 73.9 / 85.6), travel 0.6399 ± 0.0044,
   0.0% destroyed, **ρ = +0.991**, from 3 seeds × 2000 iters × 4096 envs.
   Recorded in `sim/run_final.log`, which is now committed so the headline
   traces to a file. Reproduce: `python analyze_rl2dof.py 3` (~8 min, GPU).

   Fig. 2(a) and `sim/rl_policy.pkl` come from a shorter single-seed run
   (1200 × 1024) that recovers 65.8% at ρ = +0.958 — kept because it evaluates
   in under a second in the Colab demo. `analyze_rl2dof.py` now saves
   `figdata_rl3.npz` and `rl_policy_best.pkl`, and `make_figures.py` prefers
   them, so re-running it once regenerates Fig. 2(a) from the full-scale run.
3. **Table III mixes cost types** — 15 s of optimization, 594 s of surrogate
   *training*, 0.04 ms of *inference*. The caption should distinguish one-time
   training cost from per-design cost, or it reads as unfair to the surrogates.
4. **Page budget — currently exactly 3 pages, with no slack.** Re-check with
   `pypdf` after any edit. The binding constraint is float *placement*, not word
   count: a `figure*` can only sit at a page top, so declaring one at its
   citation point can push it past the last page and add a fourth. Move the
   declaration earlier before cutting prose.

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
fitted at N=60 once flipped the sign of a conclusion here. The notebook uses
N=60 for the same reason.
