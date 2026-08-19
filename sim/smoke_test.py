"""
Cheap pre-flight checks. No training. Verifies the math and that analytic
policy gradients are actually viable before committing to a full run.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap

import env as E


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def bisect_pullin_first_order():
    """Does the first-order model still give lambda_fold = 4/27?"""
    def touches(lam):
        def body(X, _):
            Xc = jnp.clip(X, 0.0, E.TOUCHDOWN)
            return jnp.clip(X + E.DT * (lam / (1 - Xc) ** 2 - Xc), 0.0, 1.0), None
        Xf, _ = jax.lax.scan(body, 0.0, None, length=20000)
        return Xf > E.TOUCHDOWN

    lo, hi = jnp.array(0.0), jnp.array(0.25)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        pulled = touches(mid)
        hi = jnp.where(pulled, mid, hi)
        lo = jnp.where(pulled, lo, mid)
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    key = random.PRNGKey(0)
    N = 64
    all_ok = True

    key, gk, tk = random.split(key, 3)
    gs = random.uniform(gk, (N,), minval=E.G_LO, maxval=E.G_HI)
    X_ts = random.uniform(tk, (N,), minval=E.X_TGT_LO, maxval=E.X_TGT_HI)

    print("\n--- 1. Analytic identities ---")
    lam_num = float(bisect_pullin_first_order())
    err = 100 * abs(lam_num - E.LAMBDA_FOLD) / E.LAMBDA_FOLD
    all_ok &= check("static pull-in = 4/27 in first-order model",
                    err < 0.5, f"sim={lam_num:.6f} exact={E.LAMBDA_FOLD:.6f} err={err:.3f}%")

    lam_at_fold = float(E.lambda_for_target(jnp.array(E.X_FOLD)))
    # float32 tolerance: JAX defaults to fp32, so 4/27 is only representable
    # to ~1e-7. This is a precision bound, not a physics discrepancy.
    all_ok &= check("lambda(X_fold) = 4/27 (fp32)",
                    abs(lam_at_fold - E.LAMBDA_FOLD) < 1e-6,
                    f"{lam_at_fold:.8f} vs {E.LAMBDA_FOLD:.8f}")

    print("\n--- 1b. Predicted vs measured relaxation time constant ---")
    for xe in [0.05, 0.15, 0.25]:
        tau_pred = 1.0 / (1.0 - 2 * xe / (1 - xe))
        Xs_t, _ = E.rollout_oracle(jnp.array(1.0), jnp.array(xe))
        # time to reach 1-1/e of the way to target
        thresh = xe * (1 - 1 / jnp.e)
        idx = int(jnp.argmax(Xs_t > thresh))
        tau_meas = idx * E.DT
        print(f"    X_e={xe:<5} tau_pred={tau_pred:5.2f}  tau_meas={tau_meas:5.2f}")

    print("\n--- 2. Oracle recovers its target (validates the inverse map) ---")
    Xs_o, t_o = E.batch_oracle(gs, X_ts)
    sse_o, pi_o = E.batch_metrics(Xs_o, t_o, X_ts)
    all_ok &= check("oracle steady-state error ~ 0",
                    float(jnp.max(sse_o)) < 2e-3,
                    f"max|X-X_t|={float(jnp.max(sse_o)):.2e}")
    all_ok &= check("oracle never pulls in", int(jnp.sum(pi_o)) == 0,
                    f"{int(jnp.sum(pi_o))}/{N}")

    print("\n--- 3. Task is non-trivial: no fixed command works ---")
    best = None
    for u in jnp.linspace(0.0, 1.0, 41):
        Xs, t = E.batch_fixed(u, gs, X_ts)
        s, p = E.batch_metrics(Xs, t, X_ts)
        m = float(jnp.mean(s))
        if best is None or m < best[0]:
            best = (m, float(u), int(jnp.sum(p)))
    all_ok &= check("best fixed command still has large error",
                    best[0] > 10 * float(jnp.mean(sse_o)),
                    f"err={best[0]:.4f} at u={best[1]:.3f} (oracle {float(jnp.mean(sse_o)):.2e}), "
                    f"pull-ins={best[2]}/{N}")

    print("\n--- 4. Classical PI baseline (the honest comparison) ---")
    for kp, ki in [(1.0, 2.0), (2.0, 5.0), (0.5, 1.0)]:
        Xs_p, t_p = E.batch_pi((kp, ki)[0], (kp, ki)[1], gs, X_ts)
        s_p, p_p = E.batch_metrics(Xs_p, t_p, X_ts)
        print(f"    kp={kp:<4} ki={ki:<4} err={float(jnp.mean(s_p)):.5f} "
              f"pull-ins={int(jnp.sum(p_p))}/{N}")

    print("\n--- 5. Analytic policy gradient viability ---")
    key, pk = random.split(key)
    params = E.init_policy(pk)

    def loss_fn(p, g, xt):
        Xs, _ = E.batch_policy(p, g, xt)
        return jnp.mean(vmap(E.tracking_loss)(Xs, xt))

    loss_val, grads = jax.value_and_grad(loss_fn)(params, gs, X_ts)
    gnorm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads))))
    all_ok &= check("loss is finite", bool(jnp.isfinite(loss_val)), f"loss={float(loss_val):.5f}")
    all_ok &= check("gradient is finite (no explosion through 300 steps)",
                    bool(jnp.isfinite(gnorm)), f"|grad|={gnorm:.4e}")
    all_ok &= check("gradient is non-zero (signal reaches params)",
                    gnorm > 1e-8, f"|grad|={gnorm:.4e}")

    print("\n--- 6. No NaNs in any rollout ---")
    all_ok &= check("oracle rollout clean", bool(jnp.all(jnp.isfinite(Xs_o))))
    Xs_pol, _ = E.batch_policy(params, gs, X_ts)
    all_ok &= check("untrained policy rollout clean", bool(jnp.all(jnp.isfinite(Xs_pol))))

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
