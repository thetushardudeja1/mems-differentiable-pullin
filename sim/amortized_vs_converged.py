"""
Can the amortized network actually BEAT direct optimisation?

THREE THINGS ARE BEING SEPARATED
1. Is our "direct" reference even converged? Earlier scripts ran 120-150 steps
   from one start, giving 14.300 V at T=2.0, while inverse_design.py reached
   13.60 V with 200 steps. So the -4.1% "win" reported by amortized_hard.py is
   against an under-converged baseline and means nothing. Run direct to
   convergence with restarts to get the TRUE reference.
2. Where does the network alone land against that true reference? Amortisation
   theory says a finite-capacity net generalising over specs should not
   systematically beat per-instance optimisation -- the amortisation gap. We
   expect to match, not beat.
3. HYBRID: use the network's output as the INITIALISATION for a short direct
   optimisation. Learned warm-starting is the standard way amortisation
   actually wins, because it starts in a good basin. This should beat
   direct-from-scratch at equal step count.

Reported at T = 2.0 um, the spec used throughout the project.
"""

import time
import pickle
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, jit
import numpy as np

from amortized_design import P_BASE, S_BASE, solve, D_of
import amortized_hard as AH

T_SPEC = 2.0


def optimise_from(p0, sg0, steps, lr=0.02, T=T_SPEC):
    """Direct optimisation from a given start; returns best feasible V_PI."""
    def travel_of(p, s):
        return solve(p, s)[1]

    @jit
    def snewt(p, s):
        tr, d = jax.value_and_grad(lambda x: travel_of(p, x))(s)
        return s - (tr - T) / d, d

    def obj(p, s, d):
        tr = travel_of(p, s)
        sc = s - (tr - T) / jax.lax.stop_gradient(d)
        v, tr2, r = solve(p, sc)
        return v, (tr2, sc, r)

    og = jit(jax.value_and_grad(obj, has_aux=True))
    p, sg = p0, sg0
    for _ in range(5):
        sg, _ = snewt(p, sg)
    m = jnp.zeros_like(p)
    vv = jnp.zeros_like(p)
    best = jnp.inf
    hist = []
    for t in range(1, steps + 1):
        sg, d = snewt(jax.lax.stop_gradient(p), sg)
        (v, (tr2, sc, r)), g = og(p, sg, d)
        sg = sc
        if abs(float(tr2) - T) < 1e-3 and float(r) < 1e-6:
            best = min(best, float(v))
        m = 0.9 * m + 0.1 * g
        vv = 0.999 * vv + 0.001 * g ** 2
        p = p - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
        hist.append(float(best))
    return float(best), hist


if __name__ == "__main__":
    with open("amort_hard_theta.pkl", "rb") as f:
        theta = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    Tj = jnp.array(T_SPEC)
    p_net = AH.shape_head(theta, Tj)
    s_net = AH.scale_head(theta, Tj)
    s_enf = AH.enforce_sigma(p_net, Tj, s_net)
    v_net, tr_net, _ = solve(p_net, s_enf)
    print(f"spec: travel = {T_SPEC} um\n")
    print(f"[NETWORK alone]  V_PI = {float(v_net):.4f} V  "
          f"(travel {float(tr_net):.4f})  inference ~0.04 ms\n")

    # ---- 1. is direct converged? run long, with restarts ----
    print(f"[DIRECT to convergence, multiple restarts]")
    print(f"  {'start':>22}{'steps':>8}{'V_PI':>10}{'time':>9}")
    key = random.PRNGKey(0)
    starts = [("P_BASE (default)", P_BASE, jnp.array(S_BASE))]
    for i in range(3):
        key, k = random.split(key)
        starts.append((f"random #{i+1}",
                       P_BASE + 0.3 * random.normal(k, P_BASE.shape)
                       * jnp.exp(-0.35 * jnp.arange(P_BASE.shape[0])),
                       jnp.array(S_BASE)))
    best_direct = jnp.inf
    for name, p0, s0 in starts:
        t0 = time.perf_counter()
        v, hist = optimise_from(p0, s0, 500)
        dt = time.perf_counter() - t0
        best_direct = min(best_direct, v)
        print(f"  {name:>22}{500:>8}{v:>10.4f}{dt:>8.0f}s")
    print(f"  --> converged direct reference: {float(best_direct):.4f} V\n")

    # ---- 2. network vs the true reference ----
    gap = 100.0 * (float(v_net) - float(best_direct)) / float(best_direct)
    verdict = ("network WINS" if gap < -0.05 else
               "network matches" if gap < 0.5 else "amortisation gap")
    print(f"[NETWORK vs converged direct]  {float(v_net):.4f} vs "
          f"{float(best_direct):.4f}  ->  {gap:+.2f}%   ({verdict})\n")

    # ---- 3. hybrid: network as warm start ----
    print(f"[HYBRID: network output as the initialisation]")
    print(f"  {'extra steps':>12}{'from net':>11}{'from P_BASE':>14}"
          f"{'net advantage':>15}")
    for k_steps in [0, 5, 10, 25, 50, 100]:
        if k_steps == 0:
            v_h = float(v_net)
        else:
            v_h, _ = optimise_from(p_net, s_enf, k_steps)
        v_s, _ = optimise_from(P_BASE, jnp.array(S_BASE), max(k_steps, 1))
        adv = 100.0 * (v_h - v_s) / v_s
        print(f"  {k_steps:>12}{v_h:>11.4f}{v_s:>14.4f}{adv:>+14.2f}%")

    print(f"\n  A negative 'net advantage' means the learned initialisation")
    print(f"  reaches a better design than starting from scratch with the same")
    print(f"  number of optimisation steps.")
