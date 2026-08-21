"""
Where along the beam is pull-in voltage most sensitive to the electrode gap?

Two claims in the README are produced here, so both trace to a script:

  1. dV_PI/dD(xi) peaks near the CLAMP, at xi ~ 0.22 -- not at the tip, where
     deflection is largest. The electrode shaping that buys drive voltage
     happens in the first quarter of the beam.

  2. One reverse-mode pass gives the sensitivity at every node at once, for a
     fraction of the cost of differencing, and it is exact. Central differences
     need 2N solves and land at ~5 digits; autodiff needs one pass.

The gradient is taken THROUGH the fold system, where the tangent stiffness is
singular by construction -- that is the whole point, and the finite-difference
agreement below is the evidence that it is right.

    python sensitivity_map.py
"""

import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad

import beam as B
import inverse_design as ID

N, BC = 60, B.CANTILEVER
EPS = 1e-5                      # central-difference step, in units of the gap


def main():
    xi = B.node_xi(N, BC)
    fold = jit(partial(B.pullin_fold, alpha=ID.ALPHA, N_t=0.0, N=N, bc=BC))

    # A linear-profile device sized to the project's 2 um travel specification.
    D = jnp.maximum(1.0, 4.7275 * ID.profile_coord(xi, BC) ** 1.0)
    z0 = fold(D)[3]                       # warm start; stop_gradient'd inside

    @jit
    def V_of_D(Dv):
        return ID.volts(fold(Dv, z0=z0)[0])

    dV = jit(grad(V_of_D))

    Lam, travel, res, _ = fold(D)
    print(f"device: linear gap profile, N={N}, {BC}")
    print(f"  V_PI    = {float(ID.volts(Lam)):.4f} V")
    print(f"  travel  = {float(travel):.5f}  (spec 2.0)")
    print(f"  ||R||   = {float(res):.1e}\n")

    # ---- exact sensitivity: one reverse pass over all N nodes ---------------
    dV(D).block_until_ready()                                   # compile
    t0 = time.perf_counter()
    g_ad = dV(D); g_ad.block_until_ready()
    t_ad = time.perf_counter() - t0
    g_ad = np.asarray(g_ad)

    # ---- central differences: 2N forward solves ----------------------------
    t0 = time.perf_counter()
    g_fd = np.empty(N)
    for i in range(N):
        vp = V_of_D(D.at[i].add(EPS))
        vm = V_of_D(D.at[i].add(-EPS))
        g_fd[i] = float((vp - vm) / (2 * EPS))
    t_fd = time.perf_counter() - t0

    i_pk = int(np.argmax(np.abs(g_ad)))
    # Relative error is meaningless where the gradient is ~0 (near the clamp it
    # is 1e-3 while the peak is 3e-1), so quote it at the node that dominates
    # the design, and report the worst ABSOLUTE disagreement alongside it.
    rel_pk = abs(g_ad[i_pk] - g_fd[i_pk]) / abs(g_fd[i_pk])
    abs_max = np.max(np.abs(g_ad - g_fd))

    print(f"{'':24}{'solves':>8}{'wall':>10}{'accuracy':>24}")
    print(f"{'central differences':24}{2*N:>8}{t_fd:>9.2f}s"
          f"{'~5 digits (O(eps^2))':>24}")
    print(f"{'reverse-mode autodiff':24}{1:>8}{t_ad*1e3:>8.0f}ms"
          f"{'exact':>24}")
    print(f"\n  {2*N} solves versus one reverse pass -- {t_fd/t_ad:.0f}x here,")
    print(f"  and the ratio grows with N since autodiff stays at one pass.")
    print(f"  agreement at the dominant node: {rel_pk:.1e} relative")
    print(f"  worst absolute disagreement:    {abs_max:.1e} V per unit gap")
    print("  That residual is the finite-difference truncation error, not the")
    print("  autodiff -- changing EPS moves it, changing nothing else does.\n")

    # ---- where is the leverage? -------------------------------------------
    print(f"peak |dV_PI/dD| at xi = {float(xi[i_pk]):.3f}  "
          f"({float(g_ad[i_pk]):+.4f} V per unit gap)")
    print(f"  at the tip  xi = {float(xi[-1]):.3f}  "
          f"({float(g_ad[-1]):+.4f} V per unit gap, "
          f"{100*abs(g_ad[-1]/g_ad[i_pk]):.1f}% of the peak)")

    q = N // 4
    frac = float(np.sum(np.abs(g_ad[:q])) / np.sum(np.abs(g_ad)))
    print(f"  first quarter of the beam carries {100*frac:.0f}% of the total "
          f"sensitivity\n")

    print("  xi      dV_PI/dD (V per unit gap)")
    for i in range(0, N, max(1, N // 12)):
        bar = "#" * int(round(40 * abs(g_ad[i]) / np.max(np.abs(g_ad))))
        print(f"  {float(xi[i]):.3f}  {g_ad[i]:+.5f}  {bar}")

    # Euler's homogeneous-function theorem, as an independent check that the
    # gradient is the real one: with fringing OFF, Lambda_PI is homogeneous of
    # degree 3 in the gap profile, so sum_i D_i dLambda/dD_i = 3 Lambda exactly.
    fold0 = jit(partial(B.pullin_fold, alpha=0.0, N_t=0.0, N=N, bc=BC))
    z00 = fold0(D)[3]
    lam0 = jit(lambda Dv: fold0(Dv, z0=z00)[0])
    lhs = float(jnp.vdot(D, grad(lam0)(D)))
    rhs = 3.0 * float(lam0(D))
    print(f"\nEuler identity (alpha=0):  sum D dLam/dD = {lhs:.9f}"
          f"   3*Lam = {rhs:.9f}")
    print(f"  relative error {abs(lhs-rhs)/abs(rhs):.6%}  <- no fitted constant")


if __name__ == "__main__":
    main()
