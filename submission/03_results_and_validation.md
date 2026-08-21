# Results & Validation

**ISMC 2026 · Track 2, AI for MEMS** · Quantitative metrics with baseline
comparisons

Every number below was produced by a named script in `sim/` and can be
regenerated from a clean checkout. Results that went against us are included.

## Headline device outcomes

| Outcome | Result | Baseline it is measured against |
|---|---|---|
| **Drive voltage, same process and mask count** | **22.86 → 13.60 V (−41%)** | uniform gap, identical travel spec, matched grid |
| **Usable travel recovered from tolerance margin** | **+12.8%**, at **0.0% devices destroyed** | best safe fixed-gain target |
| **Design turnaround** | **15 s**, laptop CPU, no licence | FEM campaign over a hand-picked family |
| **Per-design cost once trained** | **0.04 ms**, spec met to 0.56% | ~30 s of optimization |

**Interactive demo:** `sim/demo.ipynb` runs items 1 and 3 of that table live on
a laptop CPU in ~90 s, with outputs already stored in the file. It reproduces
22.864 V for the baseline, verifies the exact gradient against the analytic
`c³` law to **0.000000%**, and drives 22.86 → 14.04 V in 150 gradient steps
while holding travel pinned at 2.0000 µm.

---

## 1. Solver validation — 8 independent sources, 1967–2025

| # | Source | Quantity | Ours | Reference | Error | Script |
|---|---|---|---|---|---|---|
| 1 | Flores 2016; Gomez 2017 | λ\* lumped | 0.148150 | 4/27 = 0.148148 | **0.001%** | `pullin.py` |
| 2 | Standard | Λ_PI cantilever | 1.6799 | 1.6790 | **0.05%** | `test_fold.py` |
| 3 | Standard | Λ_PI fixed–fixed | 69.935 | 71.167 | 1.73% | `test_fold.py` |
| 4 | Analytic `c³` law | Σ ∂Λ/∂D | 5.0398 | 5.0398 | **0.00%** | `test_fold.py` |
| 5 | Seeger & Boser 2003 | tip-in, design #1 | 19.0% | 19% | **exact** | `model2dof.py` |
| 6 | Seeger & Boser 2003 | tip-in, design #3 | 85.1% | 85% | **0.12%** | `model2dof.py` |
| 7 | Seeger & Boser 2003 | instability mechanism | **3/3 correct** | — | — | `model2dof.py` |
| 8 | Nazemi 2025 (**measured**) | V_PI reduction | 13.2% | 16.0% | 2.8 pp | `validate_nazemi.py` |
| 9 | M-TEST 1997 (**measured**) | V_PI, 500 µm beam | 11.70 V | 11.1 ± 0.1 V | +5.4% | `validate_mtest.py` |

**Discriminating test.** Designs #3 and #4 of Seeger & Boser are mechanically
identical and differ only in parasitic capacitance. The model correctly
predicts that this single electrical change flips the failure mode from tip-in
to charge pull-in and costs 20% of travel — a mechanism prediction, not a
fitted number.

---

## 2. Inverse design vs. published hand-tuned study

| Design | Cantilever (ours / paper) | Fixed–fixed (ours / paper) |
|---|---|---|
| Uniform gap | 22.86 / 23.66 V | 175.83 / 178.26 V |
| Linear (n=1) | 14.31 / 14.86 V | 123.01 / 125.48 V |
| Best published family | 13.61 / 14.14 V | 121.79 / 123.94 V |
| **Free-form (ours)** | **13.60 V** | **120.03 V** |
| Optimal exponent n\* | **1.29** / 4/3 | **1.00** / 1.00 |

Independently matched, never fitted: **d_max = 6.571 µm** vs published 6.57 µm;
flattened-bottom ratio **0.80** vs 0.78.

**Positive controls.** Started from a uniform gap with no knowledge of the
published family, the optimizer descends **22.86 → 13.87 V** (cantilever) and
**175.8 → 123.5 V** (fixed–fixed) unaided, converging on the published profile
shape. This rules out the result being an artifact of a favourable
initialization.

**Open question answered.** Haluzan *et al.* conjectured further gains beyond
simple shapes "are believed to be minimal." Measured: **0.14%** (cantilever),
**1.45%** (fixed–fixed).

---

## 3. AI method comparison — equal budget of 400 true solves

| Method | V_PI | Training samples | Design time |
|---|---|---|---|
| MLP surrogate + active learning | 17.203 V | 400 | 71 s |
| FNO surrogate + active learning | 14.245 V | 400 | 594 s |
| Direct differentiable (400 solves) | 13.783 V | 0 | 15 s |
| Direct, converged (4 restarts) | 13.665 V | 0 | 170 s |
| **Amortized network** | **13.636 V** | **0** | **0.04 ms** |
| **Network + 100 steps** | **13.599 V** | 0 | 17 s |

Baselines were strengthened until they stopped being strawmen: the surrogates
were given active learning, a trust region, and feasible restarts. **The FNO is
competitive.** Our advantage is zero training data, no architecture search and
40× less wall-clock — not superior accuracy.

**Amortized network.** Trained through the solver with **zero stored
simulations** (1,577 s, 6,000 live solves). Replacing a soft constraint penalty
with a differentiable enforcement layer cut specification error from ~5% to
**0.56% mean / 2.14% worst case** over a 17-point sweep of the full range
T = 1–3 µm (`gen_figdata2.py`, plotted in Fig. 5(b)); the coarser 5-point
evaluation in `amortized_hard.py` gives 0.75% mean. Warm-starting from the network
reaches a better design in **5 optimization steps than a cold start does in
500**.

---

## 4. Reinforcement learning under unmeasurable variance

The tip-in ceiling is not measurable at test time, so a fixed-gain controller
must target the worst-case device.

| Policy | Mean travel | Devices destroyed |
|---|---|---|
| Best safe fixed target | 0.5674 | 0.0% |
| **Adaptive RL policy** | **0.6399 ± 0.0044** | **0.0 ± 0.0%** |
| Oracle (knows the ceiling) | 0.6577 | 0.0% |

**80.4 ± 4.9% of available headroom recovered**, +12.8% usable travel, across
3 seeds × 512 held-out devices (1,536 episodes), with **zero devices
destroyed**.

**Proof of adaptation, not just reduced conservatism:** correlation between
achieved travel and the device's own ceiling is **ρ = +0.96** for the policy
versus **−0.03** for fixed gain. The policy beats the baseline in *every*
ceiling bin, including the most fragile.

---

## 5. Throughput

| Workload | Rate |
|---|---|
| Batched design evaluation (CPU, fp64) | 1,430 designs/s, 100% convergence |
| RL training (GPU, fp32) | 9.1 M env-steps/s (8× CPU) |
| Lumped dynamics (GPU, fp32) | 179 M steps/s |
| Amortized inference | 0.04 ms/design |

---

## 6. Results that went against us

* **The amortized network does not beat converged optimization** — 13.636 vs
  13.665 V is inside the 0.28% spread across optimizer restarts. It *matches*.
* **A well-built FNO surrogate is competitive** (14.245 V), contradicting our
  initial hypothesis that a fold point would be structurally hard to
  interpolate. It predicted its own optimum to 13.178 vs 13.162 V actual.
* **No GPU speed-up for the static solver.** fp32 does not converge
  (|R| = 2.0 vs 1e-8); consumer GPUs run fp64 at ~1/64 rate.
* **Absolute voltages are systematically high** — +5.4% (M-TEST), ~+85%
  (Nazemi microbridge), traced to anchor compliance absent from our model.
* **An earlier reported "106% of headroom recovered"** was an artifact of a
  mislabelled oracle and a handicapped baseline; corrected to **82%**.

## Reproducing

```bash
conda env create -f environment.yml && conda activate mems && cd sim
jupyter lab demo.ipynb          # 90 s, laptop CPU, no licence, no GPU
python test_fold.py && python validate_beam.py && python model2dof.py
python validate_nazemi.py && python validate_mtest.py
python gen_figdata.py && python gen_figdata2.py
python make_figures.py && python make_tables.py && python make_arch.py
```
