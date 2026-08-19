"""
Gradient-based inverse design of the electrostatic gap profile d(x),
benchmarked against published hand-tuned results, for BOTH boundary conditions.

TASK (exactly Haluzan et al., Micromachines 2010, 1, 68-81):
  minimize pull-in voltage V_PI
  subject to (a) 2 um of stable travel before pull-in
             (b) gap >= d_min = 1 um everywhere (process limit)
  geometry: E=169 GPa, nu=0.32, h=100 um, l=1000 um, w=10 um

PUBLISHED RESULTS
  cantilever : uniform 23.66 -> linear 14.86 -> best poly n=4/3: 14.14 V
  fixed-fixed: uniform 178.26 -> linear 125.48 -> flattened bottom: 123.94 V
  Note the trend REVERSES between the two: for fixed-fixed beams every n != 1
  is worse than linear (their Table 7: n=1 -> 125.48, n=4/3 -> 137.68,
  n=3/2 -> 151.20). Reproducing that reversal is a real test of the solver.

PROFILE FAMILIES (their definitions)
  cantilever : d(x) = max(d_min, d_max * u^n),          u = x/l
  fixed-fixed: same but mirrored about mid-span,        u = min(x, l-x)/(l/2)
  flattened  : d(x) = max(d_min, min(d_flat, d_lin*u))  (their best for FF)

METHOD
  Every design -- theirs and ours -- is scaled so travel == 2 um exactly on the
  SAME grid before comparison, so the comparison is apples-to-apples.
  Free-form profiles are optimized by gradients taken through the fold-point
  condition, and a POSITIVE CONTROL run starts from a uniform gap so we can
  tell "no improvement" apart from "optimizer failure".
"""

import time
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
import numpy as np

import beam as B

E, NU = 169e9, 0.32
E_TILDE = E / (1 - NU ** 2)
H, L, W, D0 = 100e-6, 1000e-6, 10e-6, 1e-6
ALPHA = 0.42 * D0 / H
TRAVEL_REQ = 2.0
EPS0 = 8.8541878128e-12
V_SCALE = float(jnp.sqrt(E_TILDE * W ** 3 * D0 ** 3 / (6.0 * EPS0 * L ** 4)))

N_OPT, N_FINAL = 40, 60
K = 12

PAPER = {
    B.CANTILEVER: dict(uniform=23.66, linear=14.86, best=14.14,
                       best_label="poly n=4/3", n_scan=[1.0, 1.0833, 1.1666,
                                                       1.2916, 1.3333, 1.375,
                                                       1.5, 1.6666]),
    B.FIXED_FIXED: dict(uniform=178.26, linear=125.48, best=123.94,
                        best_label="flattened bottom",
                        n_scan=[0.8333, 0.9166, 1.0, 1.0833, 1.1666, 1.3333, 1.5]),
}


def volts(Lam):
    return V_SCALE * jnp.sqrt(Lam)


def profile_coord(xi, bc):
    """u: 0 at the clamp(s), 1 at the point of maximum deflection."""
    if bc == B.CANTILEVER:
        return xi
    return jnp.minimum(xi, 1.0 - xi) / 0.5      # mirrored about mid-span


def D_poly(d_max, n, xi, bc):
    return jnp.maximum(1.0, d_max * profile_coord(xi, bc) ** n)


def D_flat(d_lin, ratio, xi, bc):
    """Their 'flattened bottom' family: linear ramp capped at d_flat."""
    u = profile_coord(xi, bc)
    return jnp.maximum(1.0, jnp.minimum(ratio * d_lin, d_lin * u))


def basis(xi, K):
    ks = jnp.arange(K)
    return jnp.cos(jnp.pi * ks[None, :] * xi[:, None])


def shape_norm(p, x):
    s = jax.nn.softplus(basis(x, K) @ p)
    return s / jnp.mean(s)


def fit_freeform(D_target, xi):
    y = jnp.clip(D_target - 1.0, 1e-3, None)
    A = basis(xi, K)
    params, *_ = jnp.linalg.lstsq(A, jnp.log(jnp.expm1(y)), rcond=None)
    return params


def evaluate(D, n, bc, z0=None):
    return B.pullin_fold(D, alpha=ALPHA, N_t=0.0, N=n, bc=bc, z0=z0, n_coarse=15)


def size_for_travel(make_D, lo, hi, n, bc, iters=22, tol=1e-6):
    """Bisect a scalar scale so travel == TRAVEL_REQ (monotone in the scale).
    A failed fold solve returns a spuriously SMALL travel, which would be
    misread as 'too small' and march the bracket to its ceiling; failures occur
    at the large-scale end, so non-converged points shrink the bracket."""
    lo, hi = jnp.array(lo), jnp.array(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, travel, res, _ = evaluate(make_D(mid), n, bc)
        too_small = jnp.logical_and(travel < TRAVEL_REQ, res < tol)
        lo = jnp.where(too_small, mid, lo)
        hi = jnp.where(too_small, hi, mid)
    return 0.5 * (lo + hi)


def run(bc):
    ref = PAPER[bc]
    xi = B.node_xi(N_OPT, bc)
    M = B._unknown_count(N_OPT, bc)
    print("=" * 78)
    print(f"  {bc.upper()}")
    print("=" * 78)

    # ---------------- baselines (their families, our solver) ----------------
    print(f"\n--- Baselines on the search grid (N={N_OPT}) ---")
    _, tr1, _, _ = evaluate(jnp.ones(M), N_OPT, bc)
    c_req = TRAVEL_REQ / float(tr1)
    lam_u, _, _, _ = evaluate(jnp.full(M, c_req), N_OPT, bc)
    print(f"  uniform   d={c_req:6.3f} um  V_PI={float(volts(lam_u)):8.3f} V   "
          f"(paper {ref['uniform']:.2f} V)")

    print(f"  polynomial scan  d(x)=max(d_min, d_max*u^n):")
    best = None
    for n_exp in ref["n_scan"]:
        dm = float(size_for_travel(lambda d: D_poly(d, n_exp, xi, bc),
                                   1.05, 25.0, N_OPT, bc))
        lam_p, tr_p, res_p, _ = evaluate(D_poly(dm, n_exp, xi, bc), N_OPT, bc)
        v = float(volts(lam_p))
        mark = ""
        if best is None or v < best[0]:
            best, mark = (v, n_exp, dm), "  <--"
        print(f"    n={n_exp:.4f}  d_max={dm:6.3f} um  V_PI={v:8.3f} V  "
              f"travel={float(tr_p):.4f}{mark}")
    V_poly, n_best, dm_best = best
    print(f"  best polynomial: n={n_best:.4f}, V_PI={V_poly:.3f} V")

    # their flattened-bottom family (their overall best for fixed-fixed)
    V_flat, flat_par = None, None
    if bc == B.FIXED_FIXED:
        print(f"  flattened-bottom scan  d(x)=max(d_min, min(d_flat, d_lin*u)):")
        for ratio in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
            dl = float(size_for_travel(lambda d: D_flat(d, ratio, xi, bc),
                                       1.05, 30.0, N_OPT, bc))
            lam_f, tr_f, _, _ = evaluate(D_flat(dl, ratio, xi, bc), N_OPT, bc)
            v = float(volts(lam_f))
            mark = ""
            if V_flat is None or v < V_flat:
                V_flat, flat_par, mark = v, (dl, ratio), "  <--"
            print(f"    d_flat/d_lin={ratio:.2f}  d_lin={dl:6.3f} um  "
                  f"V_PI={v:8.3f} V  travel={float(tr_f):.4f}{mark}")

    # ---------------- free-form optimization ----------------
    def D_of(p, sigma):
        return 1.0 + sigma * shape_norm(p, xi)

    def travel_of(p, sigma, z0):
        _, tr, _, _ = evaluate(D_of(p, sigma), N_OPT, bc)
        return tr

    @jit
    def sigma_newton(p, sigma, z0):
        tr, dtr = jax.value_and_grad(lambda s: travel_of(p, s, z0))(sigma)
        return sigma - (tr - TRAVEL_REQ) / dtr, dtr

    def objective(p, sigma, dtr, z0):
        tr = travel_of(p, sigma, z0)          # keeps p-dependence live
        sig_c = sigma - (tr - TRAVEL_REQ) / jax.lax.stop_gradient(dtr)
        lam, tr2, res, z = evaluate(D_of(p, sig_c), N_OPT, bc, z0=z0)
        return volts(lam), (tr2, res, z, sig_c)

    obj_and_grad = jit(jax.value_and_grad(objective, has_aux=True))

    def optimize(p_init, label, n_iter=200, lr=0.02):
        p, sigma = p_init, jnp.array(1.0)
        zz = B.fold_initial_guess(D_of(p, sigma), ALPHA, 0.0, N_OPT, bc, n_coarse=15)
        for _ in range(6):
            sigma, dtr = sigma_newton(p, sigma, zz)
        m = jnp.zeros_like(p)
        vv = jnp.zeros_like(p)
        print(f"\n  --- optimizing from: {label} ---")
        t0 = time.perf_counter()
        for it in range(1, n_iter + 1):
            sigma, dtr = sigma_newton(jax.lax.stop_gradient(p), sigma, zz)
            (v, (tr, res, z_new, sig_c)), g = obj_and_grad(p, sigma, dtr, zz)
            zz = jnp.where(res < 1e-6, z_new, zz)
            sigma = sig_c
            m = 0.9 * m + 0.1 * g
            vv = 0.999 * vv + 0.001 * g ** 2
            p = p - lr * (m / (1 - 0.9 ** it)) / (jnp.sqrt(vv / (1 - 0.999 ** it)) + 1e-9)
            if it % 50 == 0 or it == 1:
                print(f"    it {it:4d}  V_PI={float(v):8.3f} V  travel={float(tr):.4f}  "
                      f"|grad|={float(jnp.linalg.norm(g)):.2e}  |res|={float(res):.1e}")
        print(f"    time: {time.perf_counter() - t0:.1f}s")
        return p, float(sigma)

    p_ref, s_ref = optimize(fit_freeform(D_poly(dm_best, n_best, xi, bc), xi),
                            "best published-family design")
    p_ctl, s_ctl = optimize(fit_freeform(jnp.full(M, 2.0), xi),
                            "uniform gap (POSITIVE CONTROL)")

    # ---------------- final, grid-matched comparison ----------------
    print(f"\n--- All designs re-fitted to travel=2 um at N={N_FINAL} ---")
    xi_f = B.node_xi(N_FINAL, bc)
    M_f = B._unknown_count(N_FINAL, bc)

    def report(make_D, lo, hi, label):
        sc = float(size_for_travel(make_D, lo, hi, N_FINAL, bc))
        Df = make_D(sc)
        lam, tr, res, _ = evaluate(Df, N_FINAL, bc)
        V = float(volts(lam))
        ok = float(res) < 1e-6 and abs(float(tr) - TRAVEL_REQ) < 5e-3
        print(f"  {label:<36} V_PI={V:8.3f} V  travel={float(tr):.4f}  "
              f"gap {float(jnp.min(Df)):.2f}-{float(jnp.max(Df)):.2f} um  "
              f"{'OK' if ok else '<-- UNRELIABLE'}")
        return V, Df, ok

    def ff(p):
        return lambda sig: 1.0 + sig * shape_norm(p, xi_f)

    V_u, _, _ = report(lambda c: jnp.full(M_f, c), 1.0, 12.0, "uniform gap")
    V_l, _, _ = report(lambda d: D_poly(d, 1.0, xi_f, bc), 1.05, 25.0, "linear (n=1)")
    V_p, D_p, _ = report(lambda d: D_poly(d, n_best, xi_f, bc), 1.05, 25.0,
                         f"best polynomial (n={n_best:.4f})")
    V_fl = None
    if flat_par is not None:
        V_fl, _, _ = report(lambda d: D_flat(d, flat_par[1], xi_f, bc), 1.05, 30.0,
                            f"flattened bottom (r={flat_par[1]:.2f})")
    V_r, D_r, ok_r = report(ff(p_ref), 0.5 * s_ref, 2.0 * s_ref, "free-form (from best family)")
    V_c, D_c, ok_c = report(ff(p_ctl), 0.5 * s_ctl, 2.0 * s_ctl, "free-form (from uniform, CONTROL)")

    print(f"\n  gap profiles d(x) [um]:")
    idx = [0, len(xi_f) // 6, len(xi_f) // 3, len(xi_f) // 2,
           2 * len(xi_f) // 3, 5 * len(xi_f) // 6, len(xi_f) - 1]
    print(f"    {'x/l':>9}" + "".join(f"{float(xi_f[i]):9.3f}" for i in idx))
    print(f"    {'published':>9}" + "".join(f"{float(D_p[i]):9.3f}" for i in idx))
    print(f"    {'free':>9}" + "".join(f"{float(D_r[i]):9.3f}" for i in idx))
    print(f"    {'control':>9}" + "".join(f"{float(D_c[i]):9.3f}" for i in idx))

    best_pub = V_fl if V_fl is not None else V_p
    print(f"\n  SUMMARY ({bc}, all at N={N_FINAL})")
    print(f"    {'design':<36}{'ours':>10}{'paper':>10}")
    print(f"    {'uniform gap':<36}{V_u:>9.2f}V{ref['uniform']:>9.2f}V")
    print(f"    {'linear gap (n=1)':<36}{V_l:>9.2f}V{ref['linear']:>9.2f}V")
    print(f"    {'best published family':<36}{best_pub:>9.2f}V{ref['best']:>9.2f}V"
          f"   [{ref['best_label']}]")
    print(f"    {'free-form (ours)':<36}{V_r:>9.2f}V{'--':>10}")
    print(f"    {'free-form from uniform (control)':<36}{V_c:>9.2f}V{'--':>10}")

    if ok_c and V_c < 0.75 * V_u:
        print(f"    control descended {V_u:.1f} -> {V_c:.1f} V unaided "
              f"({100 * (V_c - best_pub) / best_pub:+.1f}% vs best published)")
    else:
        print(f"    CONTROL FAILED -- treat 'no improvement' as an optimizer artifact")
    print(f"    free-form vs best published family: "
          f"{100 * (1 - V_r / best_pub):+.2f}% improvement")

    np.save(f"D_free_{bc}.npy", np.asarray(D_r))
    np.save(f"D_pub_{bc}.npy", np.asarray(D_p))
    np.save(f"xi_{bc}.npy", np.asarray(xi_f))
    return dict(bc=bc, uniform=V_u, linear=V_l, published=best_pub,
                free=V_r, control=V_c)


if __name__ == "__main__":
    print(f"V_PI = {V_SCALE:.4f} * sqrt(Lambda)   [Haluzan geometry, d0=d_min=1 um]\n")
    out = [run(B.CANTILEVER), run(B.FIXED_FIXED)]

    print("\n" + "=" * 78)
    print("  COMBINED")
    print("=" * 78)
    print(f"  {'case':<14}{'uniform':>10}{'linear':>10}{'published':>11}"
          f"{'free-form':>11}{'control':>10}")
    for r in out:
        print(f"  {r['bc']:<14}{r['uniform']:>9.1f}V{r['linear']:>9.1f}V"
              f"{r['published']:>10.1f}V{r['free']:>10.1f}V{r['control']:>9.1f}V")
