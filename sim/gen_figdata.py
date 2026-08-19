"""
Regenerate and SAVE the arrays behind figures F2-F4.

Earlier runs printed these numbers instead of saving them, so the plots cannot
be rebuilt without re-running the experiments. Everything here writes .npz so
figures are reproducible from disk.

  F2  trend reversal : V_PI vs exponent n, cantilever and fixed-fixed
  F3  RL adaptivity  : per-device achieved travel vs that device's ceiling
  F4  warm start     : V_PI vs optimisation steps, net-init vs cold-start

Usage:  python gen_figdata.py [f2] [f3] [f4]     (default: all)
"""

import sys
import pickle
import time
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import beam as B

OUT = "figdata_"


# --------------------------------- F2 ---------------------------------------
def gen_f2():
    """V_PI vs polynomial exponent n, for both boundary conditions."""
    import inverse_design as ID
    print("[F2] trend reversal scan")
    res = {}
    for bc, label, ns in [
        (B.CANTILEVER, "cantilever",
         [1.0, 1.0833, 1.1666, 1.25, 1.2916, 1.3333, 1.375, 1.5, 1.6666, 1.8, 2.0]),
        (B.FIXED_FIXED, "fixed_fixed",
         [1.0, 1.0833, 1.1666, 1.25, 1.3333, 1.5, 1.6666, 1.8, 2.0]),
    ]:
        xi = B.node_xi(ID.N_OPT, bc)
        vs = []
        for n in ns:
            dm = float(ID.size_for_travel(lambda d: ID.D_poly(d, n, xi, bc),
                                          1.05, 30.0, ID.N_OPT, bc))
            lam, tr, r, _ = ID.evaluate(ID.D_poly(dm, n, xi, bc), ID.N_OPT, bc)
            # FEASIBILITY GATE. For large n on a fixed-fixed beam the gap
            # concentrates near midspan while most of the span sits at d_min,
            # so the near-clamp regions pull in first and travel SATURATES
            # below the 2 um spec at any d_max (verified: widening the
            # bisection bracket from 30 to 80 does not change the answer).
            # Those designs cannot meet the specification at all, so listing
            # their V_PI next to feasible ones would compare different specs.
            feasible = (abs(float(tr) - ID.TRAVEL_REQ) < 0.01
                        and float(r) < 1e-6)
            vs.append(float(ID.volts(lam)) if feasible else np.nan)
            flag = "" if feasible else f"  INFEASIBLE (travel {float(tr):.3f})"
            print(f"   {label} n={n:.4f}  V={vs[-1]:8.3f}  "
                  f"|res|={float(r):.0e}{flag}")
        res[f"{label}_n"] = np.array(ns)
        res[f"{label}_v"] = np.array(vs)
    np.savez(OUT + "f2.npz", **res)
    print("   -> figdata_f2.npz")


# --------------------------------- F3 ---------------------------------------
def gen_f3():
    """Per-device travel for the RL policy vs the fixed-gain baseline."""
    import env2dof as E
    import train_rl2dof as T
    import analyze_rl2dof as A
    print("[F3] RL adaptivity (training one seed)")
    key = random.PRNGKey(12345)
    key, dk, ek = random.split(key, 3)
    devs = T.sample_devices(dk, 512)
    r_ti = E.r_tipin(devs["beta"])

    t0 = time.perf_counter()
    params = A.train_one(0, n_iters=1200, n_envs=1024, lr=2e-3)   # 4096 envs OOMed the 8GB GPU here
    print(f"   trained in {time.perf_counter()-t0:.0f}s")
    with open("rl_policy.pkl", "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)

    tr_rl, d_rl = A.per_device_travel(params, devs, ek)
    worst = float(jnp.min(r_ti))
    tr_c, d_c = A.per_device_travel(None, devs, ek,
                                    policy=T.fixed_target_policy(1.02 * worst))
    tr_o, d_o = A.per_device_travel(None, devs, ek,
                                    policy=T.oracle_policy(margin=1.01))
    np.savez(OUT + "f3.npz",
             r_ti=np.asarray(r_ti), rl=np.asarray(tr_rl),
             fixed=np.asarray(tr_c), oracle=np.asarray(tr_o),
             dead_rl=np.asarray(d_rl), dead_fixed=np.asarray(d_c))
    print(f"   corr RL={float(jnp.corrcoef(r_ti, tr_rl)[0,1]):+.3f}  "
          f"fixed={float(jnp.corrcoef(r_ti, tr_c)[0,1]):+.3f}")
    print("   -> figdata_f3.npz, rl_policy.pkl")


# --------------------------------- F4 ---------------------------------------
def gen_f4():
    """V_PI vs optimisation steps from the network start vs a cold start."""
    from amortized_design import P_BASE, S_BASE, solve
    import amortized_hard as AH
    from amortized_vs_converged import optimise_from
    print("[F4] warm-start curves")
    with open("amort_hard_theta.pkl", "rb") as f:
        theta = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    T = 2.0
    Tj = jnp.array(T)
    p_net = AH.shape_head(theta, Tj)
    s_net = AH.scale_head(theta, Tj)
    s_enf = AH.enforce_sigma(p_net, Tj, s_net)
    v_net = float(solve(p_net, s_enf)[0])

    steps = 200
    _, hist_net = optimise_from(p_net, s_enf, steps)
    _, hist_cold = optimise_from(P_BASE, jnp.array(S_BASE), steps)
    hist_net = [v_net] + hist_net
    hist_cold = [float("inf")] + hist_cold
    np.savez(OUT + "f4.npz",
             steps=np.arange(len(hist_net)),
             net=np.array(hist_net), cold=np.array(hist_cold),
             v_net_only=v_net, converged_direct=13.6649)
    print(f"   net-only {v_net:.4f}, net+{steps} {hist_net[-1]:.4f}, "
          f"cold+{steps} {hist_cold[-1]:.4f}")
    print("   -> figdata_f4.npz")


if __name__ == "__main__":
    which = [a.lower() for a in sys.argv[1:]] or ["f2", "f3", "f4"]
    if "f2" in which:
        gen_f2()
    if "f4" in which:
        gen_f4()
    if "f3" in which:
        gen_f3()
    print("done")
