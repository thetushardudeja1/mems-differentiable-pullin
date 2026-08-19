"""
Smoke test for the extended fold-point system: does it land on the same fold
the continuation sweep finds, and are its gradients usable for optimization?
"""

import jax
import jax.numpy as jnp

import beam as B


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


if __name__ == "__main__":
    all_ok = True
    N = 60

    print("--- 1. Fold system vs continuation sweep (uniform gap) ---")
    for bc in [B.CANTILEVER, B.FIXED_FIXED]:
        M = B._unknown_count(N, bc)
        D = jnp.ones(M)

        lam_sweep, travel_sweep, _, _ = B.pullin_lambda(D, N=N, bc=bc)
        lam_fold, travel_fold, res, _ = B.pullin_fold(D, N=N, bc=bc)
        ref = B.LAMBDA_PI_REF[bc]

        print(f"  {bc}:")
        print(f"    sweep : Lambda={float(lam_sweep):9.4f}  travel={float(travel_sweep):.4f}")
        print(f"    fold  : Lambda={float(lam_fold):9.4f}  travel={float(travel_fold):.4f}  "
              f"|res|={float(res):.2e}")
        print(f"    ref   : Lambda={ref:9.4f}")

        all_ok &= check(f"{bc}: fold residual converged", float(res) < 1e-6,
                        f"|res|={float(res):.2e}")
        all_ok &= check(f"{bc}: fold Lambda within 2.5% of published ref",
                        100 * abs(float(lam_fold) - ref) / ref < 2.5,
                        f"err={100 * abs(float(lam_fold) - ref) / ref:.2f}%")
        all_ok &= check(f"{bc}: fold agrees with sweep (<1%)",
                        100 * abs(float(lam_fold) - float(lam_sweep)) / float(lam_sweep) < 1.0)

    print("\n--- 2. Known physics: fold deflection ratio > 1/3 (distributed compliance) ---")
    for bc, lo, hi in [(B.CANTILEVER, 0.40, 0.50), (B.FIXED_FIXED, 0.35, 0.45)]:
        D = jnp.ones(B._unknown_count(N, bc))
        _, travel, _, _ = B.pullin_fold(D, N=N, bc=bc)
        r = float(travel)
        all_ok &= check(f"{bc}: fold at {r:.3f} of gap, in [{lo},{hi}]", lo < r < hi)

    print("\n--- 3. Gradient of Lambda_PI w.r.t. the gap profile ---")
    bc = B.CANTILEVER
    M = B._unknown_count(N, bc)

    def lam_of_D(D):
        lam, _, _, _ = B.pullin_fold(D, N=N, bc=bc)
        return lam

    D = jnp.ones(M)
    g = jax.grad(lam_of_D)(D)
    gn = float(jnp.linalg.norm(g))
    all_ok &= check("gradient finite", bool(jnp.all(jnp.isfinite(g))), f"|g|={gn:.4e}")
    all_ok &= check("gradient non-zero", gn > 1e-8, f"|g|={gn:.4e}")

    # Analytic scaling check: for a UNIFORM gap D = c, Lambda_PI scales as c^3
    # (since the load term is Lambda/G^2 with G ~ c and Y ~ c). So the
    # directional derivative along the uniform direction must satisfy
    #   sum_i dLambda/dD_i = dLambda/dc = 3 * Lambda_PI / c   at c = 1.
    dir_deriv = float(jnp.sum(g))
    expected = 3.0 * float(lam_of_D(D))
    all_ok &= check("gradient matches analytic c^3 scaling law",
                    abs(dir_deriv - expected) / expected < 0.02,
                    f"sum(dL/dD)={dir_deriv:.4f} vs 3*Lambda={expected:.4f} "
                    f"({100 * abs(dir_deriv - expected) / expected:.2f}%)")

    print("\n--- 4. Finite-difference check of the c^3 scaling ---")
    for c in [1.0, 2.0, 4.415]:
        lam_c = float(lam_of_D(jnp.full(M, c)))
        pred = float(lam_of_D(jnp.ones(M))) * c ** 3
        print(f"    c={c:6.3f}  Lambda={lam_c:10.4f}  c^3*Lambda(1)={pred:10.4f}  "
              f"err={100 * abs(lam_c - pred) / pred:5.2f}%")

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
