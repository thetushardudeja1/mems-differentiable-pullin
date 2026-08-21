"""
Does the policy actually ADAPT per device, or did it merely discover that the
fixed baseline was too conservative and shift everyone up by a constant?

These are very different claims:
  - ADAPTIVE  : achieved travel tracks each device's own ceiling r_ti(beta),
                so achieved/r_ti is roughly constant while achieved varies.
  - UNIFORM   : achieved travel is the same for every device regardless of
                its ceiling -- no inference happening, just less conservatism.

Also runs multiple seeds, because a single training run proves nothing.
"""

import sys
import time
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, vmap

import env2dof as E
import train_rl2dof as T


def train_one(seed, n_iters=2000, n_envs=4096, lr=2e-3, verbose=False):
    key = random.PRNGKey(seed)
    key, pk = random.split(key)
    params = T.init_policy(pk)
    b1, b2 = 0.9, 0.999
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    vv = jax.tree_util.tree_map(jnp.zeros_like, params)
    for it in range(1, n_iters + 1):
        key, dks, rks = random.split(key, 3)
        devs = T.sample_devices(dks, n_envs)
        keys = random.split(rks, n_envs)
        (loss, aux), g = T.grad_fn(params, devs, keys)
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        vv = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b ** 2, vv, g)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** it), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** it), vv)
        params = jax.tree_util.tree_map(
            lambda p, a, b: p - lr * a / (jnp.sqrt(b) + 1e-8), params, mh, vh)
        if verbose and it % 100 == 0:
            print(f"    seed {seed} it {it}: return={float(aux[0]):+.4f}")
    return params


def per_device_travel(params, devices, key, policy=None):
    n = devices["beta"].shape[0]
    keys = random.split(key, n)
    rets, travels, dead, _ = vmap(
        lambda b, q, a, k: T.rollout(params, b, q, a, k,
                                     stochastic=False, policy=policy)
    )(devices["beta"], devices["Q"], devices["asym"], keys)
    return travels, dead


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    key = random.PRNGKey(12345)
    key, dk, ek = random.split(key, 3)
    eval_devices = T.sample_devices(dk, 512)
    r_ti = E.r_tipin(eval_devices["beta"])

    worst = float(jnp.min(r_ti))
    cons_pol = T.fixed_target_policy(1.02 * worst)   # best SAFE fixed target
    tr_c, d_c = per_device_travel(None, eval_devices, ek, policy=cons_pol)
    tr_o, d_o = per_device_travel(None, eval_devices, ek, policy=T.oracle_policy(margin=1.01))

    print(f"held-out set: 512 devices, ceiling r_ti in "
          f"[{float(jnp.min(r_ti)):.3f}, {float(jnp.max(r_ti)):.3f}]")
    print(f"  conservative: travel={float(jnp.mean(tr_c)):.4f} "
          f"destroyed={float(jnp.mean(d_c)) * 100:.1f}%")
    print(f"  oracle      : travel={float(jnp.mean(tr_o)):.4f} "
          f"destroyed={float(jnp.mean(d_o)) * 100:.1f}%")
    headroom = float(jnp.mean(tr_o) - jnp.mean(tr_c))

    print(f"\n=== {n_seeds} independent training runs ===")
    results = []
    all_params = []
    for s in range(n_seeds):
        t0 = time.perf_counter()
        p = train_one(s)
        tr, d = per_device_travel(p, eval_devices, ek)
        rec = 100 * (float(jnp.mean(tr)) - float(jnp.mean(tr_c))) / headroom
        results.append((float(jnp.mean(tr)), float(jnp.mean(d)), rec))
        all_params.append(p)
        print(f"  seed {s}: travel={float(jnp.mean(tr)):.4f}  "
              f"destroyed={float(jnp.mean(d)) * 100:5.1f}%  "
              f"recovered={rec:5.1f}%  ({time.perf_counter() - t0:.0f}s)")

    trs = jnp.array([r[0] for r in results])
    ds = jnp.array([r[1] for r in results])
    recs = jnp.array([r[2] for r in results])
    print(f"\n  across seeds: travel={float(jnp.mean(trs)):.4f}"
          f" +/- {float(jnp.std(trs)):.4f}   "
          f"destroyed={float(jnp.mean(ds)) * 100:.1f}%"
          f" +/- {float(jnp.std(ds)) * 100:.1f}%   "
          f"recovered={float(jnp.mean(recs)):.1f}% +/- {float(jnp.std(recs)):.1f}%")

    # ---------------- the decisive test: adaptive or uniform? ----------------
    print(f"\n=== Is it adapting? (best seed, binned by device ceiling) ===")
    best = int(jnp.argmax(trs))
    tr_rl, d_rl = per_device_travel(all_params[best], eval_devices, ek)

    edges = jnp.quantile(r_ti, jnp.linspace(0, 1, 6))
    print(f"  {'r_ti bin':>16}{'n':>5}{'conservative':>14}{'RL':>10}"
          f"{'oracle':>9}{'RL/r_ti':>10}")
    ratios = []
    for i in range(5):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (r_ti >= lo) & (r_ti <= hi if i == 4 else r_ti < hi)
        if int(jnp.sum(m)) == 0:
            continue
        rl_b = float(jnp.mean(tr_rl[m]))
        ratio = rl_b / float(jnp.mean(r_ti[m]))
        ratios.append(ratio)
        print(f"  [{lo:.3f},{hi:.3f}]{int(jnp.sum(m)):>5}"
              f"{float(jnp.mean(tr_c[m])):>14.4f}{rl_b:>10.4f}"
              f"{float(jnp.mean(tr_o[m])):>9.4f}{ratio:>10.3f}")

    # correlation between a device's ceiling and what the policy achieves on it
    cc = float(jnp.corrcoef(r_ti, tr_rl)[0, 1])
    cc_cons = float(jnp.corrcoef(r_ti, tr_c)[0, 1])
    print(f"\n  corr(r_ti, travel):  RL={cc:+.3f}   conservative={cc_cons:+.3f}")
    spread_rl = float(jnp.std(tr_rl))
    spread_c = float(jnp.std(tr_c))
    print(f"  std(travel) across devices:  RL={spread_rl:.4f}  "
          f"conservative={spread_c:.4f}")

    print()
    if cc > 0.5 and spread_rl > 2 * spread_c:
        print("  VERDICT: ADAPTIVE -- travel tracks each device's own ceiling,")
        print("           so the policy is inferring beta from the response.")
    elif cc > 0.25:
        print("  VERDICT: PARTIALLY adaptive -- some tracking of the ceiling,")
        print("           but much of the gain is uniform de-conservatism.")
    else:
        print("  VERDICT: NOT adaptive -- the policy shifted every device up by")
        print("           roughly a constant. The honest claim is then only")
        print("           'the fixed baseline was too conservative', NOT that")
        print("           the policy infers unobservable parameters.")

    # ------------------------- persist the evidence -------------------------
    # This script used to print and discard. The reported headline (80.4% of
    # headroom, rho=+0.99) therefore lived only in a console log, while the
    # figure and the released checkpoint came from the SMALLER run in
    # gen_figdata.gen_f3 (1200 iters x 1024 envs) and showed ~66% and +0.958.
    # Two runs, one claim, no way to tell them apart. Saving the arrays here
    # makes the headline traceable to a file, and lets the notebook display
    # the same run the report quotes.
    np.savez(
        "figdata_rl3.npz",
        r_ti=np.asarray(r_ti),
        fixed=np.asarray(tr_c), oracle=np.asarray(tr_o),
        dead_fixed=np.asarray(d_c), dead_oracle=np.asarray(d_o),
        rl_best=np.asarray(tr_rl), dead_rl_best=np.asarray(d_rl),
        seed_travel=np.asarray(trs), seed_dead=np.asarray(ds),
        seed_recovered=np.asarray(recs),
        best_seed=np.asarray(best), n_seeds=np.asarray(n_seeds),
        n_iters=np.asarray(2000), n_envs=np.asarray(4096),
        corr_rl=np.asarray(cc), corr_fixed=np.asarray(cc_cons),
    )
    with open("rl_policy_best.pkl", "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, all_params[best]), f)
    print(f"\n  -> figdata_rl3.npz, rl_policy_best.pkl (seed {best}, "
          f"{float(recs[best]):.1f}% of headroom)")
