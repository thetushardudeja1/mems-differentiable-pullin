# Cutting MEMS Drive Voltage by 41% — Inverse Design Through the Pull-In Bifurcation

![System architecture](figures/Fig0_architecture.png)

Electrostatic MEMS actuators are held back by two limits that share one cause.
Drive voltages are too high for a CMOS supply, so the device carries a charge
pump. And only a third of the gap is usable, because the exact snap threshold
moves with fabrication tolerance and **cannot be measured without destroying
the device** — so every die is driven for the worst case.

Both limits are pull-in. This repository attacks it directly.

| Device-level outcome | Result |
|---|---|
| **Drive voltage, same process and mask count** | **22.86 V → 13.60 V (−41%)**, electrode geometry only |
| **Usable travel recovered from tolerance margin** | **+12.8%**, at **0.0% of devices destroyed** |
| **Design turnaround** | **15 s** on a laptop CPU, no licence |
| **Per-design cost once trained** | **0.04 ms**, spec met to 0.56% |

The technical point that makes it work: pull-in **is** a saddle-node
bifurcation, where the tangent stiffness is singular and an ordinary
differentiable solver breaks down. So we solve the extended fold system

```
R(Y, Λ) = 0 ,   J(Y, Λ) v = 0 ,   vᵀv − 1 = 0
```

directly (Keller 1977; Seydel) and take **exact gradients through the
bifurcation itself**, rather than differentiating a forward solve or fitting a
surrogate to ~10⁴ archived simulations. No COMSOL, no ANSYS, no MATLAB, and
**no training dataset anywhere**.

## See it in 90 seconds

[`sim/demo.ipynb`](sim/demo.ipynb) runs end to end on a laptop CPU with its
outputs already stored, so you can read it without running it. It sizes the
baseline device (22.864 V), shows that pull-in is a fold, verifies the exact
gradient against the analytic `c³` law to **0.000000%**, and then drives
**22.86 → 14.04 V** in 150 gradient steps while holding the travel
specification pinned at 2.0000 µm.

---

## Validation and method results

| Result | Value |
|---|---|
| Pull-in threshold vs. closed form (lumped) | **0.001%** error |
| Λ_PI cantilever / fixed–fixed vs. literature | **0.05%** / 1.73% |
| Gradient vs. analytic `c³` scaling law | **0.00%** error |
| Instability mechanism, Seeger & Boser designs | **3 / 3** correct |
| Optimal exponent, cantilever vs. fixed–fixed | **n\*=1.29 / n\*=1.00** (reversal reproduced) |
| Amortized design, inference | **0.04 ms** (vs ~30 s optimization) |
| Amortized spec error (hard-constraint layer) | **0.56%** mean, 2.14% max (soft penalty: ~5%) |
| RL headroom recovered / devices destroyed | **80.4 ± 4.9%** / **0.0%** |
| Best design found for the Haluzan benchmark | **13.599 V** |

Validated against **8 independent sources spanning 1967–2025**, including one
directly measured voltage.

---

## Quick start

```bash
conda env create -f environment.yml
conda activate mems
cd sim

jupyter lab demo.ipynb       # 90-second interactive demo, CPU only

python test_fold.py          # solver + exact-gradient validation
python validate_beam.py      # distributed beam vs. literature
python model2dof.py          # 2-DOF tip-in vs. Seeger & Boser
python validate_nazemi.py    # vs. a 2025 measured microbridge
python validate_mtest.py     # vs. a 1997 measured pull-in voltage
```

Reproduce every figure from saved arrays (no experiment re-run):

```bash
python gen_figdata.py && python gen_figdata2.py
python make_figures.py && python make_tables.py && python make_arch.py
```

---

## What is where

| File | Purpose |
|---|---|
| `sim/beam.py` | Distributed beam solver; fold system; exact gradients |
| `sim/pullin.py` | Lumped dynamic model (GPU-batched) |
| `sim/inverse_design.py` | Gradient-based gap-profile design, both BCs |
| `sim/amortized_hard.py` | Amortized design net + hard-constraint layer |
| `sim/env2dof.py`, `train_rl2dof.py` | 2-DOF RL environment and policy |
| `sim/surrogate_fair.py`, `fno_surrogate.py` | MLP / FNO surrogate baselines |
| `sim/model2dof.py` | Classical charge-control baseline |
| `sim/demo.ipynb` | 90-second interactive demo (outputs included) |
| `sim/make_arch.py` | System architecture diagram (Fig. 0) |
| `papers/README.md` | Reference library index (what each paper is used for) |

---

## Honest limitations

These are stated because they are load-bearing for interpreting the numbers.

* **Absolute voltages carry a systematic offset.** M-TEST is +5.4% high
  ([010] modulus) and the uncertainty intervals do not overlap the
  measurement; the Nazemi microbridge is ~+85% high in absolute terms.
  Evidence points to **anchor compliance** (an effective length of ~162 µm
  against a drawn 120 µm), which our ideal-clamp 1D model does not represent.
  Relative and trend predictions — which is what the design claims rest on —
  agree to within 2.8 percentage points.
* **No GPU speed-up for the static solver.** The 4th-order stencil is
  cancellation-limited and requires float64; fp32 does not converge
  (|res| = 2.0 vs 1e-8). Consumer GPUs run fp64 at ~1/64 rate, so this
  workload is CPU-bound (1430 designs/s batched). GPU *is* used for the fp32
  dynamics and RL training.
* **The amortized network matches, it does not beat, converged optimization.**
  13.636 V vs 13.665 V is within the spread across optimizer restarts (0.28%).
  The genuine win is cost: 0.04 ms and zero training data.
* **A well-built FNO surrogate is competitive** (14.245 V vs our 13.783 V at
  equal budget). Our advantage is no training data, no architecture search,
  and 40× less wall-clock — not superior accuracy.
* **Fixed–fixed designs with exponent n ≳ 1.8 are infeasible**, not merely
  high-voltage: travel saturates below the 2 µm specification at any scale.
  They are excluded from the sweeps rather than plotted.

## References

Reference library and per-paper usage notes: [`papers/README.md`](papers/README.md).
Primary benchmarks: Osterberg & Senturia (M-TEST, *JMEMS* 1997);
Haluzan *et al.* (*Micromachines* 2010); Seeger & Boser (*JMEMS* 2003);
Nazemi *et al.* (*J. Sens. Sens. Syst.* 2025); Flores (2016) and
Gomez, Moulton & Vella (2017).
