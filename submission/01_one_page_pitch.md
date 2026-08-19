# Differentiable MEMS: Inverse Design Through the Pull-In Bifurcation

**Track 2 — AI for MEMS** · One-Page Pitch

---

## Problem

Electrostatic MEMS actuators are limited by **pull-in**, a snap-through
instability that caps usable travel at 33% of the gap and destroys devices when
crossed. Designing around it is slow and manual: engineers hand-pick geometry
families and run FEM campaigns one case at a time, in licensed tools
(COMSOL/ANSYS) that expose no gradients. The state of the art in AI-assisted
MEMS design inherits that limitation — the standard recipe (e.g. *Nature
Microsystems & Nanoengineering*, Nov 2025) is to run **~10⁴ FEM simulations**,
fit a neural surrogate, then optimise through the surrogate and its
approximation error.

## The AI approach

**We make the physics itself differentiable, including the instability.**

Pull-in *is* a saddle-node bifurcation. Rather than differentiating a forward
solve, we solve the extended fold system directly,

> **R(Y,Λ) = 0**,  **J(Y,Λ)·v = 0**,  **vᵀv − 1 = 0**

and take exact gradients through the bifurcation via the implicit function
theorem. This unlocks three AI capabilities that need **no training data at
all**:

1. **Gradient-based inverse design** — optimise the electrode gap profile
   against the pull-in voltage directly, not a proxy.
2. **An amortized design network** — a network mapping *specification →
   geometry*, trained end-to-end **through the solver**. Zero stored
   simulations; every gradient comes from a live solve.
3. **A safe-operation RL policy** — infers each device's unmeasurable
   fabrication ceiling from its own response and pushes closer to it.

## Novelty

* **Nobody differentiates through the MEMS pull-in bifurcation.** JAX-FEM,
  JAX-MPM and JaxSSO provide differentiable FEM/meshfree/structural solvers;
  none targets electrostatic pull-in, and none differentiates a fold point.
* **Zero-dataset AI design.** Our network is trained against exact physics, not
  against 10⁴ archived simulations — the opposite of current practice.
* **RL applied to irreversible MEMS failure.** The sequential-design and
  safe-exploration literatures each solve half of this; neither addresses
  information gathering under destructive measurement.

## Evidence it is right

Validated against **8 independent sources spanning 1967–2025**:

| Check | Result |
|---|---|
| Closed-form pull-in threshold | **0.001%** error |
| Λ_PI, cantilever / fixed–fixed | **0.05%** / 1.73% |
| Exact gradient vs. analytic `c³` law | **0.00%** error |
| Instability mechanism (Seeger & Boser) | **3 / 3** designs correct |
| Measured microbridge, 2025 device | 13.2% vs **16.0%** measured reduction |

The decisive test: the optimal gap exponent **reverses** between geometries —
n\*≈4/3 for cantilevers, n\*=1 for fixed–fixed beams. We reproduce **both**.
Opposite trends in two geometries cannot be curve-fitted.

## Impact on MEMS

* **A manual FEM campaign becomes 15 seconds on a laptop CPU — no licence.**
  From a uniform gap, the optimiser rediscovers a published hand-tuned optimum
  unaided.
* **Design in 0.04 ms.** The amortized network emits a geometry ~10⁶× faster
  than optimisation, meeting the specification to **0.75%**, and pays back its
  one-time training after ~53 designs.
* **19% of actuator travel is recoverable.** Because the tip-in ceiling is not
  measurable, fixed-gain control must target the worst-case device. Our policy
  recovers **80.4 ± 4.9%** of that lost range with **0.0% of devices
  destroyed** across 1,536 test episodes.
* **We answered an open published question.** Haluzan *et al.* conjectured that
  gains beyond simple gap shapes "are believed to be minimal" but could not
  test it. We measured it: **0.14%** (cantilever), **1.45%** (fixed–fixed).

## Open tools

Everything is released: solver, validation suite, RL environment, surrogate
baselines and figure pipeline — a reproducible, licence-free benchmark for the
MEMS + AI community.

> **Stated plainly:** our absolute voltages carry a systematic offset traced to
> anchor compliance our 1D model omits; a well-built FNO surrogate is
> competitive on quality; and the static solver is CPU-bound because it needs
> float64. All three are quantified in the report.
