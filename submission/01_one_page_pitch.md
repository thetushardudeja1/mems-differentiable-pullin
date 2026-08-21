# Cutting MEMS Drive Voltage by 41% — by Differentiating Through Pull-In

**ISMC 2026 · Track 2, AI for MEMS** · One-Page Pitch

![System architecture](../figures/Fig0_architecture.png)

---

## Problem statement

Electrostatic actuation is the workhorse of commercial MEMS — micromirror
arrays, RF switches, tunable capacitors, optical attenuators. It is also stuck
against two hard limits, and **both are the same instability**:

1. **Drive voltage is too high.** A device needing tens of volts cannot run off
   the CMOS supply, so it needs an on-chip charge pump — die area, power, and
   an extra reliability failure mode.
2. **Only a third of the gap is usable.** Past that, pull-in snaps the
   structure onto the electrode. Because the exact snap point shifts with
   fabrication tolerance and **cannot be measured on a device without
   destroying it**, drive schedules are set for the worst-case die. Every good
   die on the wafer is then driven conservatively.

Designing around pull-in today means hand-picking a geometry family and running
FEM campaigns one case at a time, in licensed tools that **expose no
gradients**. Current AI practice inherits exactly that limitation: fit a neural
surrogate to ~10⁴ FEM runs, then optimise through the surrogate's approximation
error.

## The AI approach

**We make the physics differentiable — including the instability itself.**

Pull-in is not a maximum of a smooth function; it is a **saddle-node
bifurcation**, where the tangent stiffness goes singular and a plain
differentiable solver breaks down. So we do not differentiate a forward solve.
We solve the extended fold system directly,

> **R(Y,Λ) = 0**,  **J(Y,Λ)·v = 0**,  **vᵀv − 1 = 0**

and obtain **∂V_PI/∂d(x) exactly**, via the implicit function theorem. Because
that gradient is exact and cheap, three capabilities follow that need **no
training data at all**:

1. **Inverse design** — descend on the electrode gap profile against the real
   pull-in voltage, not a proxy.
2. **An amortized design network** — specification → geometry, trained
   end-to-end *through the solver*. Zero stored simulations; every gradient
   comes from a live solve.
3. **A safe-operation RL policy** — infers each device's own unmeasurable
   ceiling from its response and drives closer to it.

## Novelty

* **Nobody differentiates through the MEMS pull-in bifurcation.** JAX-FEM,
  JAX-MPM and JaxSSO give differentiable FEM / meshfree / structural solvers;
  none targets electrostatic pull-in, and none differentiates a fold point.
* **Zero-dataset AI design.** The network is trained against exact physics, not
  against 10⁴ archived simulations — the inverse of current practice. A soft
  constraint penalty is structurally biased here, because cutting travel also
  cuts V_PI; replacing it with a differentiable **hard-constraint enforcement
  layer** cut specification error from ~5% to **0.56%**.
* **RL against an irreversible, destructive measurement.** The
  sequential-experimental-design and safe-exploration literatures each solve
  half of this; neither addresses information gathering when measuring destroys
  the device.

## Impact on MEMS

| Device-level outcome | Result |
|---|---|
| **Drive voltage, same process and mask count** | **22.86 V → 13.60 V (−41%)**, electrode geometry only |
| **Usable travel recovered from tolerance margin** | **+12.8%**, at **0.0% of devices destroyed** |
| **Design turnaround** | FEM campaign → **15 s on a laptop CPU**, no licence |
| **Per-design cost once trained** | **0.04 ms**, meeting spec to **0.56%** |

The 41% is **free in manufacturing terms**: it comes from reshaping the
electrode gap, not from a new material, a new process step, or an extra mask.
Starting from a *uniform* gap with no knowledge of any prior design family, the
optimiser rediscovers the hand-tuned optimum of Haluzan *et al.* (2010)
unaided — and settles their open conjecture that further gains "are believed to
be minimal": we measured what is actually left, **0.14%** (cantilever) and
**1.45%** (fixed–fixed).

## Evidence it is right

Validated against **8 independent sources, 1967–2025**: the closed-form pull-in
threshold to **0.001%**; the exact gradient against the analytic `c³` scaling
law to **0.00%**; **3 / 3** instability mechanisms of Seeger & Boser (2003)
predicted correctly, including a case where a purely *electrical* change flips
the failure mode; and measured devices from M-TEST (1997) and a 2025 microbridge
reproduced.

The decisive test: the optimal gap exponent **reverses** between geometries —
n\*≈4/3 for cantilevers, n\*=1 for fixed–fixed beams. We reproduce **both**.
Opposite trends in two geometries cannot be curve-fitted.

> **Stated plainly:** our absolute voltages carry a systematic offset traced to
> anchor compliance our 1D model omits; a well-built Fourier Neural Operator
> surrogate is competitive on design quality; and the static solver is CPU-bound
> because it requires float64. All three are quantified in the report.

**Everything is open** — solver, validation suite, RL environment, surrogate
baselines, demo notebook and figure pipeline: a licence-free benchmark the
MEMS + AI community can build on.
