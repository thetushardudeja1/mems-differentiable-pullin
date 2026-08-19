"""
Why does the fixed-fixed bisection fail to reach travel = 2 um?

Hypothesis: pullin_fold fails to converge for the shaped FF profiles that would
give 2 um of travel, so size_for_travel (which now distrusts non-converged
points) retreats to a solvable-but-infeasible small scale.

Prints travel / Lambda / residual vs d_max directly, at both grids, so we can
see whether the solve fails and where the feasible d_max actually lies.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import beam as B

ALPHA = 0.42 * 1e-6 / 100e-6


def D_poly_ff(d_max, n, xi):
    u = jnp.minimum(xi, 1.0 - xi) / 0.5
    return jnp.maximum(1.0, d_max * u ** n)


def probe(N, n_exp, d_list, n_coarse):
    xi = B.node_xi(N, B.FIXED_FIXED)
    print(f"\n  N={N}, n={n_exp}, n_coarse={n_coarse}")
    print(f"    {'d_max':>7}{'Lambda':>12}{'travel':>9}{'|res|':>10}{'min gap':>9}")
    for dm in d_list:
        D = D_poly_ff(dm, n_exp, xi)
        lam, tr, res, _ = B.pullin_fold(D, alpha=ALPHA, N_t=0.0, N=N,
                                        bc=B.FIXED_FIXED, n_coarse=n_coarse)
        flag = "  <-- FAILED" if float(res) > 1e-6 else ""
        print(f"    {dm:7.3f}{float(lam):12.2f}{float(tr):9.4f}{float(res):10.1e}"
              f"{float(jnp.min(D)):9.3f}{flag}")


def branch(d_max, n_exp, N, n_delta=18, s_max=0.9):
    """Walk the displacement-controlled branch and show exactly where
    solve_at_delta stops converging."""
    xi = B.node_xi(N, B.FIXED_FIXED)
    D = D_poly_ff(d_max, n_exp, xi)
    M = B._unknown_count(N, B.FIXED_FIXED)
    iref = B._ref_index(N, B.FIXED_FIXED)
    Yu = jnp.zeros(M)
    Lam = jnp.array(B.LAMBDA_PI_REF[B.FIXED_FIXED] * 0.1)
    print(f"\n  branch: FF d_max={d_max}, n={n_exp}, N={N}  "
          f"(D_mid={float(D[iref]):.2f})")
    print(f"    {'s':>6}{'delta':>9}{'Lambda':>12}{'min gap':>9}{'argmin':>8}{'|R|':>10}")
    for k in range(n_delta):
        s = s_max * (k + 1) / n_delta
        Yu, Lam = B.solve_at_delta(s * D[iref], Yu, Lam, D, ALPHA, 0.0,
                                   N, B.FIXED_FIXED)
        R = B._beam_residual(Yu, Lam, D, ALPHA, 0.0, N, B.FIXED_FIXED)
        mg = float(jnp.min(D - Yu))
        rn = float(jnp.linalg.norm(R))
        flag = "  <-- diverged" if rn > 1e-6 else ""
        print(f"    {s:6.3f}{s * float(D[iref]):9.4f}{float(Lam):12.2f}{mg:9.4f}"
              f"{float(jnp.argmin(D - Yu)):8.0f}{rn:10.1e}{flag}")


if __name__ == "__main__":
    # uniform works (travel=2 at d=5.04); does the LINEAR family?
    print("=== fixed-fixed, linear family (n=1), sweeping d_max ===")
    probe(40, 1.0, [3.0, 4.0, 4.774, 6.0, 8.0, 10.0, 14.0], n_coarse=15)

    print("\n=== does a finer bracketing sweep rescue it? ===")
    probe(40, 1.0, [6.0, 8.0, 10.0, 14.0], n_coarse=40)

    print("\n=== same at the reporting grid N=60 ===")
    probe(60, 1.0, [4.774, 6.0, 8.0, 10.0], n_coarse=40)

    print("\n=== where does the branch actually break? ===")
    branch(4.774, 1.0, 40)   # this one converges
    branch(8.000, 1.0, 40)   # this one fails

    print("\n=== uniform control (known good) ===")
    for N in (40, 60):
        M = B._unknown_count(N, B.FIXED_FIXED)
        for c in [5.038, 7.0]:
            lam, tr, res, _ = B.pullin_fold(jnp.full(M, c), alpha=ALPHA, N_t=0.0,
                                            N=N, bc=B.FIXED_FIXED, n_coarse=15)
            print(f"    N={N} uniform d={c:.3f}  Lambda={float(lam):10.2f}  "
                  f"travel={float(tr):.4f}  |res|={float(res):.1e}")
