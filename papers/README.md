# Reference library

Papers backing the differentiable MEMS pull-in solver and inverse-design work.
Each entry notes **what we actually use it for**, so validation claims in the
report can be traced to a source.

---

## 01-pullin-physics — governing equations and validation targets

| file | used for |
|---|---|
| `Osterberg-Senturia-1997-M-TEST.pdf` | **Primary experimental benchmark.** Governing ODEs (their Table I) for cantilever / fixed-fixed / diaphragm; closed-form `V_PI` (Tables III–IV); MTEST-03 wafer data: t₀=2.94±0.07 µm, g₀=1.05±0.01 µm, Ẽ=168±6 GPa, σ̃=10±1 MPa, and a measured 500 µm fixed-fixed beam at **V_PI = 11.1 ± 0.1 V**. Also the source of the destructive-characterization motivation for the RL idea. |
| `Flores-2016-dynamic-pullin-mass-spring.pdf` | Exact closed-form thresholds for the lumped model `ẍ + αẋ + x = −λ/(1+x)²`: static **λ\* = 4/27**, dynamic **λ_d(0) = 1/8**. Used to validate `sim/pullin.py` to 0.001%. |
| `Gomez-Moulton-Vella-2017-delayed-pullin-overdamped.pdf` | Independent derivation of the same model (`Q²X″ + X′ + X = λ/(1−X)²`), same λ_fold=4/27, X_fold=1/3 — cross-validation. Source of the **critical slowing down / bottleneck** result `t_PI ∝ ε^(−1/2)`, which underpins the non-destructive-characterization RL idea. |
| `Haluzan-2010-reducing-pullin-by-gap-shape.pdf` | **Inverse-design benchmark.** Sample geometry (E=169 GPa, ν=0.32, h=100 µm, l=1000 µm, w=10 µm, d_min=1 µm); hand-scanned results — cantilever 23.66→14.86→14.14 V, fixed-fixed 178.26→125.48→123.94 V; Table 7 documents the **n≠1-is-worse reversal for fixed-fixed**. States the conjecture we quantified (0.14% cantilever / 1.45% fixed-fixed). |
| `Ballestra-2008-RF-MEMS-pullin-residual-stress.pdf` | Second independent experimental dataset (gold RF-MEMS, Tables 1–2: full geometry + measured V_PI); residual stress shifts V_PI 29→57 V — motivates the `N_t` tension term. |
| `Ghoussoub-Guo-2005-PDEs-electrostatic-MEMS.pdf` | Rigorous PDE theory for `Δu = λf(x)/(1+u)²`: existence, bounds on λ\*, multiplicity. Theoretical backing for the fold structure. |
| `Sadeghian-2007-GDQ-pullin-MEMS-switches.pdf` | Alternative numerical method (generalized differential quadrature) for the same problem — independent numbers to check against. |
| `MaaniMiandoab-2017-closed-form-static-pullin-midplane-stretching.pdf` | Closed-form pull-in **including mid-plane stretching and axial stress** — leading candidate explanation for our consistent ~2% low bias on fixed-fixed (we set `N_t = 0`; their ANSYS runs had `NLGEOM` on). |
| `Review-2010-modeling-electrostatic-MEMS.pdf` | Broad survey of electrostatic MEMS modeling. |

## 02-classical-control — does a classical method already do this?

Read **before** committing to any RL angle: a plain PI controller already achieved
zero steady-state error on the naive calibration task, so RL needs a problem
classical control does not already solve.

| file | used for |
|---|---|
| `Seeger-Boser-charge-control-tipin-instability.pdf` | **Read in full.** Charge control stabilises past pull-in: 33% → 83% of gap, in CMOS silicon — so "RL to stabilise past pull-in" is a solved problem. But tip-in is explicitly *not* fixable by control ("the rotation mode is uncontrollable and unobservable... the only way to prevent tipping is through mechanical design"), which opens a co-design target. Source of eqs 21/22/24/27/28/40/41 implemented in `sim/model2dof.py`, and Tables I–III used to validate it (3/3 instability mechanisms predicted correctly). |
| `Hung-Senturia-1999-extending-travel-range.pdf` | Extending travel range of analog-tuned actuators. *Not yet read.* |
| `Chan-Dutton-extended-range-of-travel.pdf` | Extended-travel electrostatic actuator. *Not yet read.* |
| `Lu-Fedder-position-control-probe-storage.pdf` | Closed-loop position control of exactly our plant — highest priority of the unread ones, may close off remaining control angles. *Not yet read.* |
| `NadalGuardia-2002-current-drive-beyond-pullin.pdf` | Current-drive methods beyond the voltage pull-in point. *Not yet read.* |

**Still to obtain:** Gupta & Senturia, "Pull-in time dynamics as a measure of
absolute pressure," MEMS 1997 — prior art for using pull-in *timing* as a
sensing mechanism; needed before claiming novelty for non-destructive
characterization.

## 03-squeeze-film-damping — only needed if we extend to dynamics

| file | used for |
|---|---|
| `Bao-Yang-2007-squeeze-film-air-damping-review.pdf` | The standard review; Reynolds equation and reduced forms. |
| `ROM-squeeze-film-test-structures.pdf` | Reduced-order models of squeeze-film damping. |
| `Piecewise-linear-ROM-squeeze-film.pdf` | ROM including large-displacement effects. |
| `Complete-squeeze-film-damping-model-2021.pdf` | Recent complete damping model. |

*None read in full — this set is only needed if the dynamic/damping extension happens.*

## 04-differentiable-simulation — novelty positioning

| file | used for |
|---|---|
| `JAX-FEM-differentiable-GPU-FEM.pdf` | Closest prior art in spirit. Establishes differentiable physics exists but **not for MEMS electrostatic pull-in**, and does not differentiate through a bifurcation. |
| `JAX-MPM-differentiable-meshfree.pdf` | Differentiable meshfree solver; same positioning argument. |

## 05-ml-rl-for-mems — prior art for the AI claim

| file | used for |
|---|---|
| `Pau-2025-online-MEMS-IMU-calibration-RBF.pdf` | Closest AI-for-MEMS-calibration prior art. **RBF networks, forward-only, no backprop, no RL**, adapting one device's drift — confirms nobody trains a policy across a *distribution* of fabrication variance. |
| `Smart-calibration-AI-MEMS-inertial-2024.pdf` | AI calibration of MEMS inertial sensors. |

## 06-reference-manuals

| file | used for |
|---|---|
| `COMSOL-MEMS-Module-Users-Guide-6.1.pdf` | COMSOL's own physics: `−∇·(ε∇V)=ρ`, Maxwell stress tensor, `Mü + Dů + Ku = F`, and their **reduced-order modelling** section (pp. 37–42) giving the single-mode `m_eff ä + c_eff ȧ + k_eff a = ηV` form — justifies our reduced-order approach as *their own recommended practice*, not a shortcut. Table 2-1 scaling table. |

## 07-sequential-design-safe-exploration — method + baselines for the RL angle

| file | used for |
|---|---|
| `Blau-2022-sequential-experimental-design-deep-RL.pdf` | **Read in full. The method.** Casts sequential experimental design as a Hidden-Parameter MDP; reward (their eq 13) = each experiment's marginal contribution to cumulative EIG; Theorem 2 proves the RL return equals the sPCE bound on EIG. Their ablations show RL beats DAD exactly where our problem sits: **discrete design spaces** and **non-differentiable likelihoods**. States this is the first work applying generic RL to optimal experiment design → our novelty must be the APPLICATION, not the method. Code: `github.com/csiro-mlai/RL-BOED`. Precedent for honest reporting: on prey-population, RL loses to myopic SMC (4.456 vs 4.521) and they say so. |
| `Foster-2021-deep-adaptive-design.pdf` | The amortized-gradient alternative (DAD). Justifies choosing RL: DAD cannot handle discrete designs and needs a differentiable likelihood. |
| `Rainforth-2023-modern-bayesian-experimental-design.pdf` | Survey; source for the EIG formalism and the myopic/greedy baselines to beat. |
| `Sui-2015-SafeOpt-safe-exploration-GP.pdf` | Safe optimisation with GPs — closest formalism for "never cross the threshold". |
| `Turchetta-2016-safe-exploration-finite-MDPs-GP.pdf` | Safe exploration in MDPs with irreversible states. |
| `Berkenkamp-2017-safe-model-based-RL-stability.pdf` | Safe model-based RL with stability guarantees. |

**The gap we fill:** none of Blau's three problems (source location, CES, prey
population) involves *irreversible* measurement — every system there can be
re-measured. In M-TEST characterization each pull-in test **destroys the
device**, so a bad probe permanently removes a data source. That is what couples
the sequential-design literature (Blau/Foster) to the safe-exploration
literature (Sui/Turchetta/Berkenkamp), and it is addressed by neither alone.

---

## 08-recent-applications — 2024-2026 context and the head-to-head baseline

| file | used for |
|---|---|
| `Zhang-2025-MicrosystNanoeng-ML-metastructure-linearization.pdf` | **The baseline our comparison table is built against.** Zhang *et al.*, *Microsyst. Nanoeng.* **11**:214 (2025), DOI 10.1038/s41378-025-01065-4. Open Access (CC BY-NC-ND 4.0). The canonical "standard practice" pipeline: generate a large FEM dataset, fit an MLP surrogate, then inverse-design *through the surrogate*. Verified numbers quoted in our report — **"approximately 48,000 data points were generated and divided equally into three groups of 16,000 each"** (p.6); the MLP runs **"in ~0.01 s per design, in place of a full FEA solve (~22 s per design)"** (p.6); FEA is used "only to generate the training data and to verify final optimized geometries". Devices fabricated via PiezoMUMPs and measured in SEM, giving ~85% linearity improvement. **Different task** (electrothermal linearisation, not pull-in), so we compare methodology, never head-to-head performance. |
| `Nazemi-2025-microbridge-reduced-pullin-preserved-frequency.pdf` | Measured 2025 microbridge; our normalised $V_{\rm PI}$-reduction validation target. |
| `AlHadi-2025-SciRep-bifurcation-vacuum-pressure-sensor.pdf` | 2025 bifurcation-based MEMS sensing; confirms fold behaviour is of current interest. |
| `Persano-2024-pullin-stress-fixed-fixed-RF-MEMS.pdf` | Residual-stress effect on fixed-fixed pull-in. |
| `LiWang-2026-distributed-static-model-capacitive-MEMS.pdf` | Recent distributed static modelling of capacitive MEMS. |

---

## Still to obtain

- **Gupta & Senturia 1997**, "Pull-in time dynamics as a measure of absolute
  pressure" — prior art check for the non-destructive-characterization idea.
- **P5 numerical methods**, to cite for the solver: Keller 1977 (pseudo-arclength
  continuation); Seydel, *Practical Bifurcation and Stability Analysis*
  (extended systems for turning points — what `pullin_fold` implements);
  Govaerts, SIAM 2000.
- Chaloner & Verdinelli 1995 (Fisher-information / D-optimal baseline).

## Notes

- `_duplicates/` holds byte-identical copies (SHA256-verified) of the Flores and
  squeeze-film-2021 papers. Safe to delete.
- Read in full so far: M-TEST, Flores, Gomez, Haluzan, Ballestra, Maani
  Miandoab, Seeger & Boser, Blau, plus the COMSOL reduced-order chapter.
