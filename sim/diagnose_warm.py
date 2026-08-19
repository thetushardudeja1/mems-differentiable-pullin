"""
Fixed-fixed continuation stalls at step 2 even though step 1 succeeds from a
far worse (cold) start. Test the discriminating question directly:

  does the SAME delta converge from a cold start but fail from a warm start?

If cold succeeds where warm fails, the warm start is being passed incorrectly
or is poisoned. If both fail, step 1 is succeeding for an unrelated reason
(e.g. it is the only delta small enough to be easy) and the solver simply
needs more iterations.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import beam as B

ALPHA = 0.42 * 1e-6 / 100e-6
N = 40
BC = B.FIXED_FIXED
M = B._unknown_count(N, BC)
iref = B._ref_index(N, BC)
xi = B.node_xi(N, BC)


def D_ff(d_max, n=1.0):
    u = jnp.minimum(xi, 1.0 - xi) / 0.5
    return jnp.maximum(1.0, d_max * u ** n)


def run(D, delta, Yu0, Lam0, n_newton, label):
    Yu, Lam = B.solve_at_delta(delta, Yu0, Lam0, D, ALPHA, 0.0, N, BC,
                               n_newton=n_newton)
    R = B._beam_residual(Yu, Lam, D, ALPHA, 0.0, N, BC)
    rn = float(jnp.linalg.norm(R))
    print(f"    {label:<34} iters={n_newton:3d}  Lambda={float(Lam):10.2f}  "
          f"|R|={rn:9.1e}  {'ok' if rn < 1e-6 else 'FAIL'}")
    return Yu, Lam


if __name__ == "__main__":
    D = D_ff(4.774)
    Dmid = float(D[iref])
    cold_Y, cold_L = jnp.zeros(M), jnp.array(B.LAMBDA_PI_REF[BC] * 0.1)

    print(f"FF d_max=4.774, N={N}, D_mid={Dmid:.3f}")

    print("\n  step 1 (s=0.05) -- known to work:")
    Yu1, Lam1 = run(D, 0.05 * Dmid, cold_Y, cold_L, 40, "cold start")

    print("\n  step 2 (s=0.10) -- warm vs cold, same target delta:")
    run(D, 0.10 * Dmid, Yu1, Lam1, 40, "WARM start (from step 1)")
    run(D, 0.10 * Dmid, cold_Y, cold_L, 40, "COLD start")

    print("\n  step 2 with more Newton iterations:")
    for it in (80, 200, 400):
        run(D, 0.10 * Dmid, Yu1, Lam1, it, f"warm, {it} iters")

    print("\n  a few deltas, all COLD, 200 iters:")
    for s in (0.05, 0.10, 0.20, 0.30, 0.40):
        run(D, s * Dmid, cold_Y, cold_L, 200, f"s={s:.2f}")

    print("\n  same on the CANTILEVER (which works) for contrast:")
    Nc, bcc = 40, B.CANTILEVER
    xic = B.node_xi(Nc, bcc)
    Mc = B._unknown_count(Nc, bcc)
    Dc = jnp.maximum(1.0, 4.774 * xic)
    irc = B._ref_index(Nc, bcc)
    for s in (0.05, 0.10, 0.30):
        Yu, Lam = B.solve_at_delta(s * float(Dc[irc]), jnp.zeros(Mc),
                                   jnp.array(B.LAMBDA_PI_REF[bcc] * 0.1),
                                   Dc, ALPHA, 0.0, Nc, bcc, n_newton=40)
        R = B._beam_residual(Yu, Lam, Dc, ALPHA, 0.0, Nc, bcc)
        rn = float(jnp.linalg.norm(R))
        print(f"    cantilever s={s:.2f} cold        Lambda={float(Lam):10.2f}  "
              f"|R|={rn:9.1e}  {'ok' if rn < 1e-6 else 'FAIL'}")
