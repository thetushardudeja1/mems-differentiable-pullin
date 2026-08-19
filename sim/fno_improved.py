"""
Improved FNO pipeline + a dimensionality sweep.

TWO FIXES OVER fno_surrogate.py
1. CORRECT SELECTION. The previous run returned the surrogate's optimum
   (14.245 V) even though the pipeline had already EVALUATED a better feasible
   design (13.749 V) with the true solver. No practitioner would do that: you
   return the best design you actually measured, not the model's extrapolation.
   With correct selection the FNO reports 13.749 V, which beats direct's
   13.783 V. Reporting it the old way flattered us.
2. DIMENSIONALITY. At K=12 basis coefficients the two methods are effectively
   tied. Surrogate sample-complexity is expected to grow with design dimension
   while gradient methods are far less sensitive, so the interesting question
   is whether the gap opens at larger K. That is the honest way to find where
   (and whether) direct differentiation actually wins.

Both methods now report best-feasible-evaluated, at an equal budget of true
fold solves, scored by projection onto travel = 2 um.
"""

import sys
import time
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, vmap, jit
import beam as B
import fno_surrogate as FNO

E, NU = 169e9, 0.32
E_TILDE = E / (1 - NU ** 2)
H, L, W, D0 = 100e-6, 1000e-6, 10e-6, 1e-6
ALPHA = 0.42 * D0 / H
EPS0 = 8.8541878128e-12
V_SCALE = float(jnp.sqrt(E_TILDE * W ** 3 * D0 ** 3 / (6.0 * EPS0 * L ** 4)))
TRAVEL_REQ = 2.0
N, BC = 40, B.CANTILEVER
XI = B.node_xi(N, BC)
BUDGET = 400


def make_basis(K):
    return jnp.cos(jnp.pi * jnp.arange(K)[None, :] * XI[:, None])


def make_D_of(K):
    Bs = make_basis(K)

    def D_of(p, sigma):
        s = jax.nn.softplus(Bs @ p)
        return 1.0 + sigma * s / jnp.mean(s)
    return D_of


@jit
def _solve_D(D):
    lam, travel, res, _ = B.pullin_fold(D, alpha=ALPHA, N_t=0.0, N=N, bc=BC,
                                        n_coarse=15)
    return V_SCALE * jnp.sqrt(lam), travel, res


def sample(key, n, K, centre=None, spread=1.0):
    kp, ks = random.split(key)
    decay = jnp.exp(-0.35 * jnp.arange(K))[None, :]
    if centre is None:
        p = 0.8 * spread * random.normal(kp, (n, K)) * decay
        p = p.at[:, 0].add(1.2)
        sg = random.uniform(ks, (n,), minval=0.6, maxval=3.0)
    else:
        pc, sc = centre
        p = pc[None, :] + 0.25 * spread * random.normal(kp, (n, K)) * decay
        sg = jnp.clip(sc * jnp.exp(0.25 * spread * random.normal(ks, (n,))),
                      0.2, 6.0)
    return p, sg


def project(D_of, p, sigma, iters=26):
    lo, hi = 0.05 * float(sigma), 8.0 * float(sigma)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v, tr, res = _solve_D(D_of(p, mid))
        if float(res) < 1e-6 and jnp.isfinite(tr) and float(tr) < TRAVEL_REQ:
            lo = mid
        else:
            hi = mid
    sg = 0.5 * (lo + hi)
    v, tr, res = _solve_D(D_of(p, sg))
    if float(res) > 1e-6 or abs(float(tr) - TRAVEL_REQ) > 0.02:
        return None
    return float(v)


def run_fno(key, K, rounds=10, n_init=200, per_round=20):
    D_of = make_D_of(K)
    solve_batch = jit(vmap(lambda p, s: _solve_D(D_of(p, s))))
    key, ki = random.split(key)
    ps, ss = sample(ki, n_init, K)
    vs, trs, res = solve_batch(ps, ss)
    ok = (res < 1e-6) & jnp.isfinite(vs) & jnp.isfinite(trs)
    Xp, Xs = ps[ok], ss[ok]
    Y = jnp.stack([vs[ok], trs[ok]], axis=1)
    Dp = vmap(D_of)(Xp, Xs)
    p_cur, s_cur = Xp[0], Xs[0]
    spent = n_init

    for rd in range(rounds):
        key, kf, ke = random.split(key, 3)
        sur, _ = FNO.fit_fno(kf, Dp, Y, steps=1200)
        lo = jnp.concatenate([jnp.min(Xp, 0), jnp.array([jnp.min(Xs)])])
        hi = jnp.concatenate([jnp.max(Xp, 0), jnp.array([jnp.max(Xs)])])

        def sobj(z):
            Dq = D_of(z[:-1], z[-1])
            pr = FNO.fno_forward(sur[0], (Dq - sur[1]) / sur[2]) * sur[4] + sur[3]
            return pr[0] + 60.0 * (TRAVEL_REQ - pr[1]) ** 2

        z0 = jnp.clip(jnp.concatenate([p_cur, jnp.array([s_cur])]), lo, hi)
        z = adam(sobj, z0, 250, lo, hi)
        p_cur, s_cur = z[:-1], z[-1]

        pe, se = sample(ke, per_round - 1, K, centre=(p_cur, s_cur))
        pe = jnp.concatenate([p_cur[None, :], pe])
        se = jnp.concatenate([jnp.array([s_cur]), se])
        ve, tre, rese = solve_batch(pe, se)
        spent += per_round
        oke = (rese < 1e-6) & jnp.isfinite(ve) & jnp.isfinite(tre)
        Xp = jnp.concatenate([Xp, pe[oke]])
        Xs = jnp.concatenate([Xs, se[oke]])
        Y = jnp.concatenate([Y, jnp.stack([ve[oke], tre[oke]], axis=1)])
        Dp = jnp.concatenate([Dp, vmap(D_of)(pe[oke], se[oke])])
        fe = jnp.abs(Y[:, 1] - TRAVEL_REQ) < 0.10
        if bool(jnp.any(fe)):
            idx = jnp.argmin(jnp.where(fe, Y[:, 0], jnp.inf))
            p_cur, s_cur = Xp[idx], Xs[idx]

    # CORRECT SELECTION: best design actually evaluated on the constraint
    fe = jnp.abs(Y[:, 1] - TRAVEL_REQ) < 0.05
    if not bool(jnp.any(fe)):
        return None, spent
    idx = jnp.argmin(jnp.where(fe, Y[:, 0], jnp.inf))
    return project(D_of, Xp[idx], Xs[idx]), spent


def adam(obj, z, steps, lo=None, hi=None, lr=0.02):
    gl = jit(jax.value_and_grad(obj))
    m = jnp.zeros_like(z)
    v = jnp.zeros_like(z)
    for t in range(1, steps + 1):
        l, g = gl(z)
        g = jnp.nan_to_num(g)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        z = z - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if lo is not None:
            z = jnp.clip(z, lo, hi)
    return z


def run_direct(key, K, budget=BUDGET):
    D_of = make_D_of(K)
    key, k0 = random.split(key)
    ps, ss = sample(k0, 1, K)
    p, sg = ps[0], ss[0]

    def travel_of(p, s):
        return _solve_D(D_of(p, s))[1]

    @jit
    def snewt(p, s):
        tr, d = jax.value_and_grad(lambda x: travel_of(p, x))(s)
        return s - (tr - TRAVEL_REQ) / d, d

    def obj(p, s, d):
        tr = travel_of(p, s)
        sc = s - (tr - TRAVEL_REQ) / jax.lax.stop_gradient(d)
        v, tr2, r = _solve_D(D_of(p, sc))
        return v, (tr2, sc)

    og = jit(jax.value_and_grad(obj, has_aux=True))
    for _ in range(4):
        sg, d = snewt(p, sg)
    m = jnp.zeros_like(p)
    vv = jnp.zeros_like(p)
    best = jnp.inf
    steps = budget // 3
    for t in range(1, steps + 1):
        sg, d = snewt(jax.lax.stop_gradient(p), sg)
        (v, (tr2, sc)), g = og(p, sg, d)
        sg = sc
        if abs(float(tr2) - TRAVEL_REQ) < 0.05:
            best = min(best, float(v))
        m = 0.9 * m + 0.1 * g
        vv = 0.999 * vv + 0.001 * g ** 2
        p = p - 0.02 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
    proj = project(D_of, p, sg)
    return (min(best, proj) if proj else best), steps * 3


if __name__ == "__main__":
    Ks = [int(x) for x in sys.argv[1:]] or [12, 30]
    print(f"budget {BUDGET} true fold solves per method; both report the BEST")
    print(f"FEASIBLE design they actually evaluated, scored at travel = 2 um\n")
    print(f"{'K (dims)':>9}{'FNO':>12}{'direct':>12}{'direct wins by':>16}"
          f"{'FNO time':>11}{'direct time':>13}")
    print("-" * 74)
    for K in Ks:
        key = random.PRNGKey(0)
        t0 = time.perf_counter()
        v_f, sp_f = run_fno(key, K)
        t_f = time.perf_counter() - t0
        t0 = time.perf_counter()
        v_d, sp_d = run_direct(key, K)
        t_d = time.perf_counter() - t0
        if v_f is None:
            print(f"{K:>9}{'INFEASIBLE':>12}{v_d:>12.3f}{'-':>16}"
                  f"{t_f:>10.0f}s{t_d:>12.0f}s")
        else:
            gap = 100.0 * (v_f - v_d) / v_d
            print(f"{K:>9}{v_f:>12.3f}{v_d:>12.3f}{gap:>+15.1f}%"
                  f"{t_f:>10.0f}s{t_d:>12.0f}s")
    print(f"\nreference: hand-tuned polynomial 13.62 V (our Haluzan reproduction)")
