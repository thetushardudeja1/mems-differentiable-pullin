"""
Budget-matched, best-effort comparison: neural surrogate vs direct
differentiable physics on the same MEMS inverse-design task.

WHAT WAS WRONG WITH THE FIRST ATTEMPT (surrogate_vs_direct.py)
  1. The surrogate was a plain MLP on a fixed random sample -- no active
     sampling near the constraint, i.e. a weak strawman.
  2. The two methods used DIFFERENT constraint handling. Direct used a weak
     penalty for parity, which cost it (14.40 V here vs 13.60 V with the
     feasible-by-construction projection it normally uses).
  3. The simulator budgets were not equal.

FIXES APPLIED HERE
  * EQUAL BUDGET: both methods get the same number of true fold solves
    (BUDGET), counted explicitly by a shared counter. Nothing is free.
  * ACTIVE LEARNING for the surrogate: alternate between fitting, optimising
    through the fit, and re-sampling AROUND the current optimum -- the standard
    adaptive workflow, which concentrates data where the optimiser actually
    goes. This is a much stronger baseline than one-shot random sampling.
  * FEASIBLE-BY-CONSTRUCTION for direct: sigma is solved each step so that
    travel == 2 um exactly, with the implicit-function correction that keeps
    gradients exact. This is the method as it is actually meant to be run.
  * IDENTICAL SCORING: both final designs are projected onto travel = 2 um
    with the true solver before being compared.
"""

import time
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, vmap, jit
import beam as B

E, NU = 169e9, 0.32
E_TILDE = E / (1 - NU ** 2)
H, L, W, D0 = 100e-6, 1000e-6, 10e-6, 1e-6
ALPHA = 0.42 * D0 / H
EPS0 = 8.8541878128e-12
V_SCALE = float(jnp.sqrt(E_TILDE * W ** 3 * D0 ** 3 / (6.0 * EPS0 * L ** 4)))
TRAVEL_REQ = 2.0
N, BC, K = 40, B.CANTILEVER, 12
XI = B.node_xi(N, BC)
BUDGET = 400


class Counter:
    def __init__(self):
        self.n = 0

    def spend(self, k=1):
        self.n += k
        return self.n <= BUDGET


CTR = Counter()


def basis(x):
    return jnp.cos(jnp.pi * jnp.arange(K)[None, :] * x[:, None])


def D_of(p, sigma):
    s = jax.nn.softplus(basis(XI) @ p)
    return 1.0 + sigma * s / jnp.mean(s)


@jit
def _solve(p, sigma):
    lam, travel, res, _ = B.pullin_fold(D_of(p, sigma), alpha=ALPHA, N_t=0.0,
                                        N=N, bc=BC, n_coarse=15)
    return V_SCALE * jnp.sqrt(lam), travel, res


def true_eval(p, sigma, count=True):
    if count:
        CTR.spend()
    return _solve(p, sigma)


_solve_batch = jit(vmap(_solve))


def sample_designs(key, n, centre=None, spread=1.0):
    kp, ks = random.split(key)
    decay = jnp.exp(-0.35 * jnp.arange(K))[None, :]
    if centre is None:
        p = 0.8 * spread * random.normal(kp, (n, K)) * decay
        p = p.at[:, 0].add(1.2)
        sigma = random.uniform(ks, (n,), minval=0.6, maxval=3.0)
    else:
        pc, sc = centre
        p = pc[None, :] + 0.25 * spread * random.normal(kp, (n, K)) * decay
        sigma = jnp.clip(sc * jnp.exp(0.25 * spread * random.normal(ks, (n,))),
                         0.2, 6.0)
    return p, sigma


# --------------------------------- surrogate --------------------------------
def init_mlp(key, d_in, hidden=128, d_out=2):
    ks = random.split(key, 3)
    return [random.normal(ks[0], (d_in, hidden)) / jnp.sqrt(d_in), jnp.zeros(hidden),
            random.normal(ks[1], (hidden, hidden)) / jnp.sqrt(hidden), jnp.zeros(hidden),
            random.normal(ks[2], (hidden, d_out)) / jnp.sqrt(hidden), jnp.zeros(d_out)]


def mlp(w, x):
    h = jnp.tanh(x @ w[0] + w[1])
    h = jnp.tanh(h @ w[2] + w[3])
    return h @ w[4] + w[5]


def fit(key, X, Y, steps=3000, lr=2e-3):
    mx, sx = X.mean(0), X.std(0) + 1e-8
    my, sy = Y.mean(0), Y.std(0) + 1e-8
    Xn, Yn = (X - mx) / sx, (Y - my) / sy
    w = init_mlp(key, X.shape[1])
    gl = jit(jax.value_and_grad(lambda w: jnp.mean((mlp(w, Xn) - Yn) ** 2)))
    m = [jnp.zeros_like(a) for a in w]
    v = [jnp.zeros_like(a) for a in w]
    for t in range(1, steps + 1):
        l, g = gl(w)
        m = [0.9 * a + 0.1 * b for a, b in zip(m, g)]
        v = [0.999 * a + 0.001 * b ** 2 for a, b in zip(v, g)]
        w = [a - lr * (mm / (1 - 0.9 ** t)) / (jnp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
             for a, mm, vv in zip(w, m, v)]
    return (w, mx, sx, my, sy), float(l)


def predict(sur, p, sigma):
    w, mx, sx, my, sy = sur
    x = jnp.concatenate([p, jnp.array([sigma])])
    return mlp(w, (x - mx) / sx) * sy + my


def adam_opt(obj, z0, steps, lr=0.02, lo=None, hi=None):
    """Adam, optionally confined to a TRUST REGION [lo, hi].

    Without bounds the surrogate optimiser walks far outside its training
    distribution and proposes designs the true solver cannot even evaluate
    (observed: travel = -639458, V = 1.3e7 -- non-converged solves). Every
    practical surrogate-optimisation method constrains the search to the
    sampled domain, so omitting that would make this a strawman baseline
    rather than a fair one.
    """
    z = z0
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


def project_true(p, sigma, iters=26):
    """Put a design on travel == 2 um using the true solver (scoring only)."""
    lo, hi = 0.05 * float(sigma), 8.0 * float(sigma)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v, tr, res = true_eval(p, mid, count=False)
        if float(res) < 1e-6 and jnp.isfinite(tr) and float(tr) < TRAVEL_REQ:
            lo = mid
        else:
            hi = mid
    sig = 0.5 * (lo + hi)
    v, tr, res = true_eval(p, sig, count=False)
    if float(res) > 1e-6 or abs(float(tr) - TRAVEL_REQ) > 0.02:
        return None
    return float(v), float(tr)


if __name__ == "__main__":
    key = random.PRNGKey(0)
    print(f"task: Haluzan cantilever, minimise V_PI subject to travel = 2 um")
    print(f"budget: {BUDGET} true fold solves for EACH method\n")

    key, k0 = random.split(key)
    p0s, s0s = sample_designs(k0, 1)
    p0, s0 = p0s[0], s0s[0]

    # ================= A. surrogate with ACTIVE sampling =================
    CTR.n = 0
    t0 = time.perf_counter()
    n_init = 200
    key, ki = random.split(key)
    ps, ss = sample_designs(ki, n_init)
    vs, trs, ress = _solve_batch(ps, ss)
    CTR.spend(n_init)
    ok = (ress < 1e-6) & jnp.isfinite(vs) & jnp.isfinite(trs)
    X = jnp.concatenate([ps[ok], ss[ok][:, None]], axis=1)
    Y = jnp.stack([vs[ok], trs[ok]], axis=1)

    print(f"[SURROGATE + ACTIVE LEARNING]")
    print(f"  {'round':>6}{'n_data':>8}{'fit MSE':>11}{'pred V':>9}"
          f"{'true V':>9}{'true travel':>13}{'budget':>9}")
    p_cur, s_cur = p0, s0
    n_rounds, per_round = 10, 20
    for rd in range(n_rounds):
        key, kf = random.split(key)
        sur, mse = fit(kf, X, Y)

        def sobj(z):
            pr = predict(sur, z[:-1], z[-1])
            return pr[0] + 60.0 * (TRAVEL_REQ - pr[1]) ** 2

        # trust region = the box actually covered by the training data
        lo = jnp.min(X, axis=0)
        hi = jnp.max(X, axis=0)
        z = adam_opt(sobj, jnp.clip(jnp.concatenate([p_cur, jnp.array([s_cur])]),
                                    lo, hi), 300, lo=lo, hi=hi)
        p_cur, s_cur = z[:-1], z[-1]
        pr = predict(sur, p_cur, s_cur)

        # spend the round's budget: the candidate plus local exploration of it
        key, ke = random.split(key)
        pe, se = sample_designs(ke, per_round - 1, centre=(p_cur, s_cur))
        pe = jnp.concatenate([p_cur[None, :], pe], axis=0)
        se = jnp.concatenate([jnp.array([s_cur]), se])
        ve, tre, rese = _solve_batch(pe, se)
        CTR.spend(per_round)
        oke = (rese < 1e-6) & jnp.isfinite(ve) & jnp.isfinite(tre)
        X = jnp.concatenate([X, jnp.concatenate([pe[oke], se[oke][:, None]], axis=1)])
        Y = jnp.concatenate([Y, jnp.stack([ve[oke], tre[oke]], axis=1)])
        cand = (f"{float(ve[0]):>9.3f}{float(tre[0]):>13.4f}"
                if bool(oke[0]) else f"{'diverged':>9}{'-':>13}")
        print(f"  {rd+1:>6}{X.shape[0]:>8}{mse:>11.2e}{float(pr[0]):>9.3f}"
              f"{cand}{CTR.n:>9}")
        # restart the next round from the best FEASIBLE design found so far,
        # not from an unconstrained surrogate optimum
        fe = jnp.abs(Y[:, 1] - TRAVEL_REQ) < 0.10
        if bool(jnp.any(fe)):
            idx = jnp.argmin(jnp.where(fe, Y[:, 0], jnp.inf))
            p_cur, s_cur = X[idx, :-1], X[idx, -1]
    t_sur = time.perf_counter() - t0

    # best design the surrogate pipeline actually found, scored honestly
    res_s = project_true(p_cur, s_cur)
    # also score the best FEASIBLE point it ever sampled
    feas = jnp.abs(Y[:, 1] - TRAVEL_REQ) < 0.05
    v_sampled = float(jnp.min(Y[feas, 0])) if bool(jnp.any(feas)) else float("nan")
    print(f"  time {t_sur:.0f}s   final design -> "
          f"{'V_PI = %.3f V at travel %.4f' % res_s if res_s else 'INFEASIBLE at any scale'}")
    print(f"  best feasible design it ever SAMPLED: "
          f"{v_sampled:.3f} V" if v_sampled == v_sampled else "  none sampled on-constraint")

    # ============ B. direct, feasible-by-construction (as intended) ============
    CTR.n = 0
    t0 = time.perf_counter()

    def travel_of(p, sg):
        return _solve(p, sg)[1]

    @jit
    def sigma_newton(p, sg):
        tr, dtr = jax.value_and_grad(lambda s: travel_of(p, s))(sg)
        return sg - (tr - TRAVEL_REQ) / dtr, dtr

    def objective(p, sg, dtr):
        tr = travel_of(p, sg)
        sg_c = sg - (tr - TRAVEL_REQ) / jax.lax.stop_gradient(dtr)
        v, tr2, res = _solve(p, sg_c)
        return v, (tr2, sg_c)

    og = jit(jax.value_and_grad(objective, has_aux=True))
    p, sg = p0, s0
    for _ in range(4):
        sg, dtr = sigma_newton(p, sg)
    m = jnp.zeros_like(p)
    vv = jnp.zeros_like(p)
    steps = BUDGET // 3          # ~3 true solves per gradient step
    print(f"\n[DIRECT, feasible-by-construction]  {steps} steps x ~3 solves = "
          f"~{steps*3} (budget {BUDGET})")
    for t in range(1, steps + 1):
        sg, dtr = sigma_newton(jax.lax.stop_gradient(p), sg)
        (v, (tr2, sg_c)), g = og(p, sg, dtr)
        CTR.spend(3)
        sg = sg_c
        m = 0.9 * m + 0.1 * g
        vv = 0.999 * vv + 0.001 * g ** 2
        p = p - 0.02 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
        if t % 30 == 0 or t == 1:
            print(f"  step {t:>4}  V_PI={float(v):.3f}  travel={float(tr2):.4f}"
                  f"  budget={CTR.n}")
    t_dir = time.perf_counter() - t0
    res_d = project_true(p, sg)
    print(f"  time {t_dir:.0f}s   final design -> "
          f"{'V_PI = %.3f V at travel %.4f' % res_d if res_d else 'INFEASIBLE'}")

    print(f"\n{'='*66}\nSUMMARY (equal budget of {BUDGET} true fold solves, "
          f"both scored at travel = 2 um)")
    print(f"  surrogate + active learning : "
          f"{('%.3f V' % res_s[0]) if res_s else 'INFEASIBLE'}")
    print(f"  direct differentiable       : "
          f"{('%.3f V' % res_d[0]) if res_d else 'INFEASIBLE'}")
    print(f"  hand-tuned polynomial (Haluzan reproduction) : 13.62 V")
    if res_s and res_d:
        print(f"  direct is {100*(res_s[0]-res_d[0])/res_d[0]:+.1f}% better than "
              f"the surrogate pipeline")
