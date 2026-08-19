"""
Head-to-head: neural surrogate vs direct differentiable physics, on the SAME
inverse-design task.

WHY THIS EXPERIMENT
"AI-accelerated multiphysics modeling and surrogate simulation" is normally done
one way: sample the simulator, fit a neural surrogate, optimise through the
surrogate. Zhang et al. (Microsystems & Nanoengineering, Nov 2025) train on
~10^4 FEM runs to do exactly this for MEMS actuators. Reported training-set
sizes for neural-operator surrogates run from ~10^2 (easy problems) to
10^4-4x10^4 (harder ones), with typical prediction errors of 5-10%.

We differentiate the physics directly instead, so the training set is empty.
This script measures what that is worth, rather than asserting it.

THE PHYSICS REASON IT SHOULD MATTER HERE
The quantity being optimised, V_PI, IS a fold (saddle-node) point. A surrogate
regressing V_PI from geometry has no representation of that bifurcation -- it
interpolates values whose defining condition is a singular Jacobian. The
travel = 2 um constraint puts the optimum exactly ON that fold, i.e. in the
stiffest part of the response surface, which is where an interpolant is least
reliable. So this is not a generic "surrogates need data" argument; there is a
specific structural reason to expect trouble.

PROTOCOL (identical objective, identical optimiser, identical basis)
  A. SURROGATE  : draw N designs, evaluate with the true solver, train an MLP
                  (params -> V_PI, travel), optimise through the MLP with Adam,
                  then score the resulting design with the TRUE solver.
  B. DIRECT     : differentiate the fold solver, optimise with the same Adam.

Scoring the surrogate's answer with the true solver is the crucial step: a
surrogate optimum that only looks good to the surrogate is worthless, and that
failure mode is invisible unless you re-simulate.
"""

import time
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, vmap, jit
import beam as B

# ---- Haluzan cantilever benchmark (same geometry validated in inverse_design)
E, NU = 169e9, 0.32
E_TILDE = E / (1 - NU ** 2)
H, L, W, D0 = 100e-6, 1000e-6, 10e-6, 1e-6
ALPHA = 0.42 * D0 / H
EPS0 = 8.8541878128e-12
V_SCALE = float(jnp.sqrt(E_TILDE * W ** 3 * D0 ** 3 / (6.0 * EPS0 * L ** 4)))
TRAVEL_REQ = 2.0

N = 40
BC = B.CANTILEVER
K = 12
XI = B.node_xi(N, BC)


def basis(x, K=K):
    ks = jnp.arange(K)
    return jnp.cos(jnp.pi * ks[None, :] * x[:, None])


def D_of(p, sigma):
    s = jax.nn.softplus(basis(XI) @ p)
    return 1.0 + sigma * s / jnp.mean(s)


def true_eval(p, sigma):
    """Ground truth: V_PI and travel from the fold solver."""
    lam, travel, res, _ = B.pullin_fold(D_of(p, sigma), alpha=ALPHA, N_t=0.0,
                                        N=N, bc=BC, n_coarse=15)
    return V_SCALE * jnp.sqrt(lam), travel, res


true_eval_batch = jit(vmap(true_eval))


def sample_designs(key, n):
    kp, ks = random.split(key)
    p = 0.8 * random.normal(kp, (n, K)) * jnp.exp(-0.35 * jnp.arange(K))[None, :]
    p = p.at[:, 0].add(1.2)
    sigma = random.uniform(ks, (n,), minval=0.6, maxval=3.0)
    return p, sigma


# ------------------------------- surrogate ----------------------------------
def init_mlp(key, d_in, hidden=128, d_out=2):
    k1, k2, k3 = random.split(key, 3)
    return [random.normal(k1, (d_in, hidden)) / jnp.sqrt(d_in), jnp.zeros(hidden),
            random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden), jnp.zeros(hidden),
            random.normal(k3, (hidden, d_out)) / jnp.sqrt(hidden), jnp.zeros(d_out)]


def mlp(w, x):
    h = jnp.tanh(x @ w[0] + w[1])
    h = jnp.tanh(h @ w[2] + w[3])
    return h @ w[4] + w[5]


def train_surrogate(key, X, Y, steps=4000, lr=2e-3):
    mu_x, sd_x = X.mean(0), X.std(0) + 1e-8
    mu_y, sd_y = Y.mean(0), Y.std(0) + 1e-8
    Xn, Yn = (X - mu_x) / sd_x, (Y - mu_y) / sd_y
    w = init_mlp(key, X.shape[1])

    def loss(w):
        return jnp.mean((mlp(w, Xn) - Yn) ** 2)

    gl = jit(jax.value_and_grad(loss))
    m = [jnp.zeros_like(a) for a in w]
    v = [jnp.zeros_like(a) for a in w]
    for t in range(1, steps + 1):
        l, g = gl(w)
        m = [0.9 * a + 0.1 * b for a, b in zip(m, g)]
        v = [0.999 * a + 0.001 * b ** 2 for a, b in zip(v, g)]
        w = [a - lr * (mm / (1 - 0.9 ** t)) / (jnp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
             for a, mm, vv in zip(w, m, v)]
    return (w, mu_x, sd_x, mu_y, sd_y), float(l)


def surrogate_predict(sur, p, sigma):
    w, mu_x, sd_x, mu_y, sd_y = sur
    x = jnp.concatenate([p, jnp.array([sigma])])
    return mlp(w, (x - mu_x) / sd_x) * sd_y + mu_y


def project_to_travel(p, sigma, target=TRAVEL_REQ, iters=26):
    """Rescale a design with the TRUE solver until travel == target.

    Without this the comparison is meaningless: a penalty term lets either
    method buy a low V_PI simply by shrinking the gap (first run: direct
    landed at travel 1.92 and the surrogates at 0.74-1.26, making their V_PI
    look far better than it was). travel is monotone in sigma, so bisection
    puts every design on the SAME constraint before scoring.
    Returns (sigma, V_PI, travel, residual) or None if it cannot be solved.
    """
    lo, hi = 0.05 * float(sigma), 6.0 * float(sigma)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v, tr, res = true_eval(p, mid)
        good = float(res) < 1e-6 and jnp.isfinite(tr)
        if good and float(tr) < target:
            lo = mid
        else:
            hi = mid
    sig = 0.5 * (lo + hi)
    v, tr, res = true_eval(p, sig)
    if float(res) > 1e-6 or abs(float(tr) - target) > 0.02:
        return None
    return sig, float(v), float(tr), float(res)


def optimise_through(objective, p0, sigma0, steps=400, lr=0.02):
    """Same Adam settings for both methods."""
    z = jnp.concatenate([p0, jnp.array([sigma0])])
    gl = jit(jax.value_and_grad(objective))
    m = jnp.zeros_like(z)
    v = jnp.zeros_like(z)
    for t in range(1, steps + 1):
        l, g = gl(z)
        g = jnp.nan_to_num(g)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        z = z - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return z[:-1], z[-1]


if __name__ == "__main__":
    key = random.PRNGKey(0)
    PENALTY = 60.0
    print(f"task: Haluzan cantilever, minimise V_PI s.t. travel = {TRAVEL_REQ} um")
    print(f"design space: {K} cosine coefficients + scale = {K+1} dims")
    print(f"reference: hand-tuned polynomial 13.62 V, direct-differentiable 13.60 V\n")

    key, k0 = random.split(key)
    p0, s0 = sample_designs(k0, 1)
    p0, s0 = p0[0], s0[0]

    # ---------------- B. direct differentiable physics ----------------
    def direct_obj(z):
        p, sg = z[:-1], z[-1]
        lam, travel, res, _ = B.pullin_fold(D_of(p, sg), alpha=ALPHA, N_t=0.0,
                                            N=N, bc=BC, n_coarse=15)
        v = V_SCALE * jnp.sqrt(lam)
        return v + PENALTY * (TRAVEL_REQ - travel) ** 2

    t0 = time.perf_counter()
    pd_, sd_ = optimise_through(direct_obj, p0, s0, steps=400)
    t_direct = time.perf_counter() - t0
    proj = project_to_travel(pd_, sd_)
    v_d = proj[1] if proj else float("nan")
    print(f"[DIRECT]  V_PI={v_d:.3f} V at travel={proj[2]:.4f} "
          f"(after projection to the constraint)   time={t_direct:.1f}s")
    print(f"          simulator solves: 400 (one per gradient step), "
          f"training samples: 0\n")

    # ---------------- A. surrogate, swept over training-set size ----------------
    print(f"[SURROGATE] standard practice: sample -> fit MLP -> optimise through it")
    print(f"  every design is projected to travel = {TRAVEL_REQ} um with the true")
    print(f"  solver before scoring, so all rows are compared at equal constraint.")
    print(f"  {'N_train':>8}{'converged':>11}{'fit MSE':>11}"
          f"{'surrogate says':>16}{'TRUE V_PI':>11}{'vs direct':>11}{'gen time':>10}")
    for n_train in [100, 300, 1000, 3000]:
        key, ks, kt = random.split(key, 3)
        ps, ss = sample_designs(ks, n_train)
        t0 = time.perf_counter()
        vs, trs, ress = true_eval_batch(ps, ss)
        t_gen = time.perf_counter() - t0

        ok = (ress < 1e-6) & jnp.isfinite(vs) & jnp.isfinite(trs)
        Xtr = jnp.concatenate([ps[ok], ss[ok][:, None]], axis=1)
        Ytr = jnp.stack([vs[ok], trs[ok]], axis=1)
        sur, fit_mse = train_surrogate(kt, Xtr, Ytr)

        def sur_obj(z):
            pred = surrogate_predict(sur, z[:-1], z[-1])
            return pred[0] + PENALTY * (TRAVEL_REQ - pred[1]) ** 2

        ps_opt, ss_opt = optimise_through(sur_obj, p0, s0, steps=400)
        pred = surrogate_predict(sur, ps_opt, ss_opt)
        proj_s = project_to_travel(ps_opt, ss_opt)
        if proj_s is None:
            print(f"  {n_train:>8}{int(jnp.sum(ok)):>11}{fit_mse:>11.2e}"
                  f"{float(pred[0]):>16.3f}{'FAILED':>11}{'-':>11}"
                  f"{t_gen:>9.1f}s   (design could not be put on the constraint)")
            continue
        v_t = proj_s[1]
        gap = 100.0 * (v_t - v_d) / v_d
        print(f"  {n_train:>8}{int(jnp.sum(ok)):>11}{fit_mse:>11.2e}"
              f"{float(pred[0]):>16.3f}{v_t:>11.3f}{gap:>+10.1f}%"
              f"{t_gen:>9.1f}s")

    print(f"\nNote: 'surrogate says' vs 'TRUE V_PI' is the honesty check -- a design")
    print(f"that only looks good to the surrogate is worthless, and you cannot see")
    print(f"that failure without re-simulating. Constraint satisfaction (travel)")
    print(f"is reported for the same reason.")
