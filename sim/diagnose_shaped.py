"""
Diagnostic: why does the fold solve fail on strongly shaped gap profiles?

Hypothesis: the residual clips the local gap at 1e-6, so the beam is allowed to
pass THROUGH the electrode rather than the solve failing. Displacement control
referenced to the TIP gap then drives the sweep into states where the beam has
already collided somewhere else along its length, producing spurious folds.

Prints the whole Lambda(delta) branch together with the minimum gap along the
beam, so we can see exactly where it stops being physical.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import beam as B

N = 40
BC = B.CANTILEVER
M = B._unknown_count(N, BC)
xi = B.node_xi(N, BC)


def D_poly(d_max, n):
    return jnp.maximum(1.0, d_max * xi ** n)


def branch(D, n_delta=30, s_max=0.9):
    iref = B._ref_index(N, BC)
    Yu = jnp.zeros(M)
    Lam = jnp.array(0.17)
    rows = []
    for k in range(n_delta):
        s = s_max * (k + 1) / n_delta
        delta = s * D[iref]
        Yu, Lam = B.solve_at_delta(delta, Yu, Lam, D, 0.0, 0.0, N, BC)
        R = B._beam_residual(Yu, Lam, D, 0.0, 0.0, N, BC)
        rows.append((float(s), float(delta), float(Lam),
                     float(jnp.min(D - Yu)), float(jnp.argmin(D - Yu)),
                     float(jnp.linalg.norm(R))))
    return rows


if __name__ == "__main__":
    print("=== uniform gap D=1 (known good, fold at Lambda=1.68, travel=0.446) ===")
    print(f"  {'s':>6}{'delta':>9}{'Lambda':>12}{'min gap':>10}{'argmin':>8}{'|R|':>10}")
    for s, d, lam, mg, am, r in branch(jnp.ones(M), n_delta=12):
        print(f"  {s:6.3f}{d:9.4f}{lam:12.5f}{mg:10.4f}{am:8.0f}{r:10.1e}")

    for n, dm in [(1.0, 6.0), (1.0, 30.0), (1.6666, 7.37)]:
        D = D_poly(dm, n)
        print(f"\n=== poly n={n}, d_max={dm}  (D range {float(jnp.min(D)):.2f} .. "
              f"{float(jnp.max(D)):.2f}) ===")
        print(f"  {'s':>6}{'delta':>9}{'Lambda':>12}{'min gap':>10}{'argmin':>8}{'|R|':>10}")
        for s, d, lam, mg, am, r in branch(D, n_delta=15):
            flag = "  <-- NONPHYSICAL (beam through electrode)" if mg <= 1e-5 else ""
            print(f"  {s:6.3f}{d:9.4f}{lam:12.5f}{mg:10.4f}{am:8.0f}{r:10.1e}{flag}")
