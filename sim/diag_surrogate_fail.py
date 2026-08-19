"""
Are the surrogate's designs genuinely infeasible, or is my projection failing?

All four surrogate optima came back "could not be put on the constraint". That
is a strong claim in our favour, and this same sigma-bisection has previously
failed for SOLVER reasons (cold-start fold solve diverging at large sigma) and
returned nonsense. So sweep sigma explicitly and look at travel and |res|
before believing it.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import surrogate_vs_direct as S


if __name__ == "__main__":
    key = random.PRNGKey(0)
    key, k0 = random.split(key)
    p0, s0 = S.sample_designs(k0, 1)
    p0, s0 = p0[0], s0[0]

    key, ks, kt = random.split(key, 3)
    ps, ss = S.sample_designs(ks, 1000)
    vs, trs, ress = S.true_eval_batch(ps, ss)
    ok = (ress < 1e-6) & jnp.isfinite(vs) & jnp.isfinite(trs)
    Xtr = jnp.concatenate([ps[ok], ss[ok][:, None]], axis=1)
    Ytr = jnp.stack([vs[ok], trs[ok]], axis=1)
    sur, mse = S.train_surrogate(kt, Xtr, Ytr)
    print(f"surrogate trained on {int(jnp.sum(ok))} samples, fit MSE {mse:.2e}")

    def sur_obj(z):
        pred = S.surrogate_predict(sur, z[:-1], z[-1])
        return pred[0] + 60.0 * (S.TRAVEL_REQ - pred[1]) ** 2

    p_opt, s_opt = S.optimise_through(sur_obj, p0, s0, steps=400)
    print(f"surrogate optimum: sigma={float(s_opt):.4f}")
    D = S.D_of(p_opt, s_opt)
    print(f"its gap profile: min={float(jnp.min(D)):.3f}  max={float(jnp.max(D)):.3f}"
          f"  max/min={float(jnp.max(D)/jnp.min(D)):.1f}")

    print(f"\nsweep sigma with the TRUE solver:")
    print(f"  {'sigma':>9}{'V_PI':>10}{'travel':>10}{'|res|':>11}{'max gap':>10}")
    best = None
    for sig in [0.05, 0.2, 0.5, 1.0, float(s_opt), 2.0, 4.0, 8.0, 16.0, 32.0]:
        v, tr, res = S.true_eval(p_opt, jnp.array(sig))
        Dg = S.D_of(p_opt, jnp.array(sig))
        mark = ""
        if float(res) < 1e-6 and abs(float(tr) - S.TRAVEL_REQ) < 0.05:
            mark = "  <- ON CONSTRAINT"
            best = sig
        print(f"  {sig:>9.3f}{float(v):>10.3f}{float(tr):>10.4f}"
              f"{float(res):>11.1e}{float(jnp.max(Dg)):>10.2f}{mark}")

    print()
    if best is None:
        conv = [float(S.true_eval(p_opt, jnp.array(s))[2]) < 1e-6
                for s in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]]
        if not any(conv):
            print("VERDICT: the fold solver fails on this shape at EVERY sigma ->")
            print("  the 'FAILED' rows are a SOLVER limitation, not evidence that")
            print("  the surrogate produced an infeasible design. Do not claim it.")
        else:
            print("VERDICT: solver converges but travel never reaches 2 um ->")
            print("  the surrogate's shape is genuinely infeasible at any scale.")
    else:
        print(f"VERDICT: feasible at sigma={best:.3f} -> my projection bracket was")
        print(f"  too narrow. The 'FAILED' rows were MY bug, not a surrogate failure.")
