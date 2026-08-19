"""
Differentiable 1-D electrostatic beam solver with SHAPED GAP PROFILE.
JAX, float64, finite differences + Newton, displacement-controlled continuation.

GOVERNING EQUATION (from M-TEST Table I / Osterberg & Senturia 1997, and the
form used by Haluzan et al., Micromachines 2010):

    E~ I y'''' - T_b y'' = eps0 h V^2 / (2 g^2) * (1 + 0.42 g/h),   g = d(x) - y

Nondimensionalized with xi = x/l, Y = y/d0, G = D(xi) - Y, I = h w^3 / 12:

    Y'''' - N_t Y'' = Lambda * (1 + alpha G) / G^2

    Lambda = 6 eps0 V^2 l^4 / (E~ w^3 d0^3)     (dimensionless voltage^2)
    alpha  = 0.42 d0 / h                        (fringing-field parameter)
    N_t    = T_b l^2 / (E~ I)                   (residual-stress tension)

VALIDATION TARGETS, derived from the published closed forms that Haluzan et al.
quote from Osterberg (their Eqs. 1-2), V_pi = C sqrt(E~ w^3 d^3/(eps0 l^4 ...)):

    Lambda_PI(cantilever)  = 6 * 0.529^2 = 1.679
    Lambda_PI(fixed-fixed) = 6 * 3.444^2 = 71.20

PULL-IN EXTRACTION: displacement control. delta(V) folds back at pull-in so it
is not invertible, but V(delta) is single-valued and smooth through the fold.
We solve the augmented system [beam residual ; Y(xi_ref) - delta] for
(Y, Lambda) and take Lambda_PI = max over delta. Differentiable throughout.
"""

import os
import jax

# fp64 by default: the 4th-order stencil (Y[i-2]-4Y[i-1]+6Y[i]-4Y[i+1]+Y[i+2])
# is a cancellation-heavy sum scaled by h^-4 ~ 2.6e6, so precision matters.
# Set MEMS_X64=0 to benchmark fp32 (consumer GPUs run fp64 at ~1/64 rate).
jax.config.update("jax_enable_x64", os.environ.get("MEMS_X64", "1") == "1")

import jax.numpy as jnp
from jax import jit, jacfwd
from functools import partial

CANTILEVER = "cantilever"
FIXED_FIXED = "fixed_fixed"

# Published dimensionless pull-in targets (see module docstring).
LAMBDA_PI_REF = {CANTILEVER: 6 * 0.529 ** 2, FIXED_FIXED: 6 * 3.444 ** 2}


def _assemble_cantilever(Yu):
    """Clamped at xi=0, free at xi=1.
    BCs: Y(0)=0, Y'(0)=0, Y''(1)=0, Y'''(1)=0 (ghost nodes).
    Yu = Y[1..N]. Returns array indexed so that Y[i] -> full[i+1], covering
    Y[-1] .. Y[N+2]."""
    Ym1 = Yu[0]                                    # Y'(0)=0  => Y[-1]=Y[1]
    YN, YNm1, YNm2 = Yu[-1], Yu[-2], Yu[-3]
    YNp1 = 2 * YN - YNm1                           # Y''(1)=0
    YNp2 = 2 * YNp1 - 2 * YNm1 + YNm2              # Y'''(1)=0
    return jnp.concatenate([
        jnp.array([Ym1]), jnp.array([0.0]), Yu,
        jnp.array([YNp1]), jnp.array([YNp2]),
    ])


def _assemble_fixed_fixed(Yu):
    """Clamped both ends. BCs: Y(0)=Y'(0)=0, Y(1)=Y'(1)=0. Yu = Y[1..N-1]."""
    Ym1 = Yu[0]        # Y'(0)=0 => Y[-1]=Y[1]
    YNp1 = Yu[-1]      # Y'(1)=0 => Y[N+1]=Y[N-1]
    return jnp.concatenate([
        jnp.array([Ym1]), jnp.array([0.0]), Yu,
        jnp.array([0.0]), jnp.array([YNp1]),
    ])


def _beam_residual(Yu, Lam, D, alpha, N_t, N, bc):
    """Residual of the ODE at each unknown node. D is the gap profile at the
    unknown nodes (same length as Yu)."""
    h = 1.0 / N
    full = _assemble_cantilever(Yu) if bc == CANTILEVER else _assemble_fixed_fixed(Yu)
    n_eq = Yu.shape[0]
    i = jnp.arange(1, n_eq + 1)          # node indices of the equations
    j = i + 1                            # offset into `full`

    d4 = (full[j - 2] - 4 * full[j - 1] + 6 * full[j]
          - 4 * full[j + 1] + full[j + 2]) / h ** 4
    d2 = (full[j - 1] - 2 * full[j] + full[j + 1]) / h ** 2

    # Local gap. Floor is 1e-12, NOT 1e-6: a penetrating state must produce an
    # astronomically large residual so it is rejected, never a smooth solvable
    # branch. (A 1e-6 floor let Newton "converge" to states with the beam
    # passed through the electrode.) Iterates are kept physical by the
    # fraction-to-the-boundary rule in the Newton solvers, so this floor should
    # never be active at a genuine solution.
    G = jnp.where(D - Yu > 1e-12, D - Yu, 1e-12)
    load = Lam * (1.0 + alpha * G) / G ** 2
    return d4 - N_t * d2 - load


def min_gap(Yu, D):
    """Signed minimum gap along the beam. <= 0 means non-physical penetration."""
    return jnp.min(D - Yu)


# --------------------------- Newton globalization ----------------------------
# Two standard ingredients, replacing the crude absolute step cap that was
# capping Newton at |dz|_inf <= 0.5 and preventing convergence entirely:
#   1. Armijo backtracking line search on ||R||.
#   2. Fraction-to-the-boundary rule (from interior-point methods) so an
#      iterate can never step through the electrode.

def _frac_to_boundary(dY, G, tau=0.95):
    """Largest t <= 1 with G - t*dY >= (1-tau)*G, i.e. gap stays positive."""
    ratios = jnp.where(dY > 0, G / jnp.where(dY > 0, dY, 1.0), jnp.inf)
    return jnp.minimum(1.0, tau * jnp.min(ratios))


def _solve_equilibrated(J, R):
    """Newton step with column equilibration.

    The unknowns are badly scaled (Y ~ O(1), null vector v ~ O(1), but
    Lambda ~ O(10^4) for a fixed-fixed beam), so equilibrating columns before
    the solve is good practice and kept here.

    MEASURED: this made NO difference to the fixed-fixed convergence failures
    (results were bit-for-bit identical), so ill-conditioning of the linear
    solve is NOT the cause of those -- LAPACK's pivoting already handled it.
    Retained as defensive hygiene only; the real cause is elsewhere.
    """
    cn = jnp.linalg.norm(J, axis=0)
    cn = jnp.where(cn > 1e-300, cn, 1.0)
    w = jnp.linalg.solve(J / cn[None, :], -R)
    return w / cn


def _row_scales(J):
    """Row norms of J, used to make the residual components commensurable.

    The beam-equation rows carry Y/h^4 ~ 1e6 and Lambda/G^2 ~ 1e3, while the
    displacement-control row is O(1). An unscaled ||R|| therefore compares
    incommensurable quantities: at a warm start the beam rows are ~0 and ||R||
    is just the O(0.24) constraint violation, so a full Newton step -- which
    satisfies the constraint but leaves an O(10-100) nonlinear beam error --
    looks catastrophic and the line search backtracks to nearly zero.
    Measured symptom: Lambda crawled 573 -> 655 in 40 warm iterations while a
    cold start reached 1019 in under 40.
    """
    rn = jnp.linalg.norm(J, axis=1)
    return jnp.where(rn > 1e-300, rn, 1.0)


def _armijo(residual_fn, z, dz, r0, t_init, rn, n_ls=20, c=1e-4):
    """Backtracking line search on the ROW-SCALED residual norm."""
    def body(carry, _):
        t, done = carry
        r_try = jnp.linalg.norm(residual_fn(z + t * dz) / rn)
        ok = r_try < (1.0 - c * t) * r0
        done_new = jnp.logical_or(done, ok)
        t_new = jnp.where(done_new, t, 0.5 * t)
        return (t_new, done_new), None

    (t_star, _), _ = jax.lax.scan(body, (t_init, False), None, length=n_ls)
    return t_star


def _unknown_count(N, bc):
    return N if bc == CANTILEVER else N - 1


def _ref_index(N, bc):
    """Index (within Yu) of the node whose deflection we control:
    tip for a cantilever, midpoint for a fixed-fixed beam."""
    return N - 1 if bc == CANTILEVER else (N // 2) - 1


@partial(jit, static_argnames=("N", "bc", "n_newton"))
def solve_at_delta(delta, Yu0, Lam0, D, alpha, N_t, N, bc, n_newton=40):
    """Displacement-controlled solve: find (Y, Lambda) with Y(xi_ref) = delta.
    Globalized with Armijo backtracking + fraction-to-the-boundary."""
    iref = _ref_index(N, bc)
    M = Yu0.shape[0]

    def aug_residual(z):
        Yu, Lam = z[:-1], z[-1]
        R = _beam_residual(Yu, Lam, D, alpha, N_t, N, bc)
        return jnp.concatenate([R, jnp.array([Yu[iref] - delta])])

    def body(z, _):
        R = aug_residual(z)
        J = jacfwd(aug_residual)(z)
        dz = _solve_equilibrated(J, R)
        # Row scaling and step length are ALGORITHMIC choices, not part of the
        # mathematical solution, so they are held fixed under differentiation.
        # Leaving them live pollutes the unrolled-Newton gradient with
        # second-order terms (rn is built from J): it broke the analytic
        # c^3 scaling check, 0.00% -> 64.6% error, while the forward solve
        # stayed exact. At convergence dz -> 0, so freezing t is harmless and
        # the gradient is set by the converged residual, as implicit
        # differentiation requires.
        t_fb = _frac_to_boundary(dz[:M], D - z[:M])
        rn = jax.lax.stop_gradient(_row_scales(J))
        r0 = jnp.linalg.norm(jax.lax.stop_gradient(R) / rn)
        t = jax.lax.stop_gradient(
            _armijo(aug_residual, jax.lax.stop_gradient(z),
                    jax.lax.stop_gradient(dz), r0, t_fb, rn))
        return z + t * dz, None

    z0 = jnp.concatenate([Yu0, jnp.array([Lam0])])
    z, _ = jax.lax.scan(body, z0, None, length=n_newton)
    return z[:-1], z[-1]


def pullin_lambda(D, alpha=0.0, N_t=0.0, N=60, bc=CANTILEVER,
                  s_max=0.9, n_delta=60, tol=1e-6):
    """Lambda at pull-in via displacement-controlled continuation sweep.

    D: gap profile at the unknown nodes, normalized by d0.
       D = ones(...) is the uniform-gap case.
    Sweeps the controlled deflection as delta = s * D[i_ref] so the sweep stays
    feasible (deflection cannot exceed the local gap) for any shaped profile.
    Returns (Lambda_PI, travel_at_fold, s_grid, Lambda_curve).
    """
    M = _unknown_count(N, bc)
    iref = _ref_index(N, bc)
    ss = jnp.linspace(s_max / n_delta, s_max, n_delta)
    Yu = jnp.zeros(M)
    Lam = jnp.array(LAMBDA_PI_REF[bc] * 0.1)

    lams, dels, oks = [], [], []
    for k in range(n_delta):
        delta = ss[k] * D[iref]
        # continuation: warm-start each solve from the previous solution
        Yu, Lam = solve_at_delta(delta, Yu, Lam, D, alpha, N_t, N, bc)
        R = _beam_residual(Yu, Lam, D, alpha, N_t, N, bc)
        # only trust a sweep point if Newton actually converged AND the beam
        # has not penetrated the electrode
        ok = jnp.logical_and(jnp.linalg.norm(R) < tol, min_gap(Yu, D) > 0.0)
        lams.append(Lam)
        dels.append(delta)
        oks.append(ok)
    lams, dels, oks = jnp.stack(lams), jnp.stack(dels), jnp.stack(oks)
    kmax = jnp.argmax(jnp.where(oks, lams, -jnp.inf))
    return lams[kmax], dels[kmax], ss, jnp.where(oks, lams, jnp.nan)


# ------------------- extended fold-point (turning point) system ---------------
# At a saddle-node fold the tangent stiffness J = dR/dY is singular. Solving the
# extended system  [ R ; J v ; v'v - 1 ]  for (Y, Lambda, v) lands exactly on the
# fold in one Newton solve -- no continuation sweep, and cheap exact gradients.

def _jac_Y(Yu, Lam, D, alpha, N_t, N, bc):
    return jacfwd(lambda Y: _beam_residual(Y, Lam, D, alpha, N_t, N, bc))(Yu)


def _fold_residual(z, D, alpha, N_t, N, bc, M):
    Yu, Lam, v = z[:M], z[M], z[M + 1:]
    R = _beam_residual(Yu, Lam, D, alpha, N_t, N, bc)
    J = _jac_Y(Yu, Lam, D, alpha, N_t, N, bc)
    return jnp.concatenate([R, J @ v, jnp.array([v @ v - 1.0])])


@partial(jit, static_argnames=("N", "bc", "n_newton"))
def _fold_newton(z0, D, alpha, N_t, N, bc, n_newton=8):
    M = _unknown_count(N, bc)

    def res_fn(zz):
        return _fold_residual(zz, D, alpha, N_t, N, bc, M)

    def body(z, _):
        R = res_fn(z)
        J = jacfwd(res_fn)(z)
        dz = _solve_equilibrated(J, R)
        # NO row scaling here. The fold system's rows are [R ; J v ; v'v - 1],
        # a different structure from the displacement-controlled system, and
        # row-scaling it broke convergence for the cantilever (|res| 1.6e-08
        # -> 4.5e-01) which in turn corrupted the gradient check. Row scaling
        # is applied only in solve_at_delta, where the pathology (an O(1)
        # constraint row against O(1e6) beam rows) was actually measured.
        t_fb = _frac_to_boundary(dz[:M], D - z[:M])
        t = _armijo(res_fn, z, dz, jnp.linalg.norm(R), t_fb,
                    jnp.ones_like(R), n_ls=12)
        return z + t * dz, None

    z, _ = jax.lax.scan(body, z0, None, length=n_newton)
    return z


def fold_initial_guess(D, alpha=0.0, N_t=0.0, N=60, bc=CANTILEVER, n_coarse=25):
    """Coarse continuation sweep to bracket the fold, then extract the null
    vector. Gradient-free: this only produces a Newton starting point."""
    M = _unknown_count(N, bc)
    iref = _ref_index(N, bc)
    Dg = jax.lax.stop_gradient(D)

    ss = jnp.linspace(0.9 / n_coarse, 0.9, n_coarse)
    Yu = jnp.zeros(M)
    Lam = jnp.array(LAMBDA_PI_REF[bc] * 0.1)
    best_lam, best_Lam, best_Yu = -jnp.inf, Lam, Yu
    for k in range(n_coarse):
        Yu, Lam = solve_at_delta(ss[k] * Dg[iref], Yu, Lam, Dg, alpha, N_t, N, bc)
        R = _beam_residual(Yu, Lam, Dg, alpha, N_t, N, bc)
        ok = jnp.logical_and(jnp.linalg.norm(R) < 1e-6, min_gap(Yu, Dg) > 0.0)
        better = jnp.logical_and(ok, Lam > best_lam)
        best_lam = jnp.where(better, Lam, best_lam)
        best_Lam = jnp.where(better, Lam, best_Lam)
        best_Yu = jnp.where(better, Yu, best_Yu)

    # null vector of J at the bracketed fold: smallest right singular vector
    J0 = _jac_Y(best_Yu, best_Lam, Dg, alpha, N_t, N, bc)
    _, _, Vh = jnp.linalg.svd(J0)
    v0 = Vh[-1]
    return jax.lax.stop_gradient(
        jnp.concatenate([best_Yu, jnp.array([best_Lam]), v0]))


def pullin_fold(D, alpha=0.0, N_t=0.0, N=60, bc=CANTILEVER, n_coarse=25, z0=None):
    """Differentiable pull-in via the extended fold system.

    Returns (Lambda_PI, travel_at_fold, residual_norm, z). Pass z0 from a
    previous call to warm-start (skips the bracketing sweep) -- the guess is
    stop_gradient'd either way, so gradients are unaffected.
    """
    M = _unknown_count(N, bc)
    iref = _ref_index(N, bc)

    if z0 is None:
        z0 = fold_initial_guess(D, alpha, N_t, N, bc, n_coarse)
    else:
        z0 = jax.lax.stop_gradient(z0)

    z = _fold_newton(z0, D, alpha, N_t, N, bc)
    Yu, Lam = z[:M], z[M]
    res = jnp.linalg.norm(_fold_residual(z, D, alpha, N_t, N, bc, M))
    return Lam, Yu[iref], res, z


def node_xi(N, bc):
    """xi coordinates of the unknown nodes."""
    if bc == CANTILEVER:
        return jnp.arange(1, N + 1) / N
    return jnp.arange(1, N) / N


def lambda_to_voltage(Lam, E_tilde, w, d0, l):
    """Invert the nondimensionalization: V = sqrt(Lam * E~ w^3 d0^3/(6 eps0 l^4))."""
    eps0 = 8.8541878128e-12
    return jnp.sqrt(Lam * E_tilde * w ** 3 * d0 ** 3 / (6.0 * eps0 * l ** 4))


def voltage_to_lambda(V, E_tilde, w, d0, l):
    eps0 = 8.8541878128e-12
    return 6.0 * eps0 * V ** 2 * l ** 4 / (E_tilde * w ** 3 * d0 ** 3)
