"""
Amortized inverse design: a network that emits a MEMS geometry from a spec,
trained end-to-end THROUGH the differentiable physics with no dataset.

THE IDEA
Direct optimisation solves a fresh problem for every specification (~15 s).
Instead, train one network  spec -> geometry  by backpropagating through the
fold solver itself. There is no training set: the gradient signal is the
physics. At inference the design is a single forward pass.

WHY THIS IS ONLY POSSIBLE HERE
The standard route (e.g. Zhang et al., Nature Microsystems & Nanoengineering,
Nov 2025) trains a surrogate on ~10^4 FEM runs and then optimises through the
surrogate, inheriting its approximation error. Because our solver is
differentiable -- including through the saddle-node fold that DEFINES the
pull-in voltage -- the network can be trained directly against exact physics.
Zero simulations are stored; every gradient comes from a live solve.

TASK
  spec  : required stable travel T (um), varied over [1.0, 3.0]
  output: gap profile d(xi) (cosine coefficients + overall scale)
  goal  : minimise pull-in voltage V_PI subject to travel == T, gap >= 1 um

PARAMETERISATION
The net predicts a RESIDUAL around a known-good design (the fitted polynomial
family). Training therefore starts inside the feasible region instead of in a
part of design space where the fold solver diverges -- which is how earlier
experiments in this project wasted runs.

HONEST QUESTION THIS ANSWERS
Amortisation usually costs accuracy. We measure that cost directly: for each
spec we compare the network's one-shot design against a full direct
optimisation of the same spec.
"""

import time
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, vmap, jit
import numpy as np
import beam as B

# ---- Haluzan cantilever benchmark (identical to inverse_design.py) ----------
E, NU = 169e9, 0.32
E_TILDE = E / (1 - NU ** 2)
H, L, W, D0 = 100e-6, 1000e-6, 10e-6, 1e-6
ALPHA = 0.42 * D0 / H
EPS0 = 8.8541878128e-12
V_SCALE = float(jnp.sqrt(E_TILDE * W ** 3 * D0 ** 3 / (6.0 * EPS0 * L ** 4)))
N, BC, K = 40, B.CANTILEVER, 12
XI = B.node_xi(N, BC)
BASIS = jnp.cos(jnp.pi * jnp.arange(K)[None, :] * XI[:, None])

T_LO, T_HI = 1.0, 3.0          # specification range (um of stable travel)

# known-good starting design: fit of the best polynomial family
P_BASE = jnp.zeros(K).at[0].set(1.2)
S_BASE = 1.55


def D_of(p, sigma):
    s = jax.nn.softplus(BASIS @ p)
    return 1.0 + sigma * s / jnp.mean(s)


@jit
def solve(p, sigma):
    lam, travel, res, _ = B.pullin_fold(D_of(p, sigma), alpha=ALPHA, N_t=0.0,
                                        N=N, bc=BC, n_coarse=15)
    return V_SCALE * jnp.sqrt(lam), travel, res


# ------------------------------- the network --------------------------------
def init_net(key, hidden=64):
    k1, k2, k3 = random.split(key, 3)
    return dict(
        W1=random.normal(k1, (2, hidden)) * 0.5, b1=jnp.zeros(hidden),
        W2=random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden),
        b2=jnp.zeros(hidden),
        # small last layer -> starts as (almost) the known-good design
        W3=random.normal(k3, (hidden, K + 1)) * 1e-3, b3=jnp.zeros(K + 1),
    )


def net_design(theta, T):
    """spec T -> (shape coefficients, scale). Residual around a good design."""
    x = jnp.array([(T - T_LO) / (T_HI - T_LO), jnp.log(T)])
    h = jnp.tanh(x @ theta["W1"] + theta["b1"])
    h = jnp.tanh(h @ theta["W2"] + theta["b2"])
    out = h @ theta["W3"] + theta["b3"]
    p = P_BASE + out[:K]
    sigma = S_BASE * jnp.exp(jnp.clip(out[K], -1.5, 1.5)) * (T / 2.0)
    return p, sigma


PENALTY = 40.0


def loss_one(theta, T):
    p, sigma = net_design(theta, T)
    v, travel, res = solve(p, sigma)
    return v + PENALTY * (travel - T) ** 2, (v, travel, res)


def loss_batch(theta, Ts):
    ls, aux = vmap(lambda T: loss_one(theta, T))(Ts)
    return jnp.mean(ls), aux


grad_fn = jit(jax.value_and_grad(loss_batch, has_aux=True))


# --------------------- direct optimisation, for comparison ------------------
def direct_optimise(T, steps=120, lr=0.02):
    """Full per-spec optimisation: the ~15 s gold standard."""
    def travel_of(p, s):
        return solve(p, s)[1]

    @jit
    def snewt(p, s):
        tr, d = jax.value_and_grad(lambda x: travel_of(p, x))(s)
        return s - (tr - T) / d, d

    def obj(p, s, d):
        tr = travel_of(p, s)
        sc = s - (tr - T) / jax.lax.stop_gradient(d)
        v, tr2, r = solve(p, sc)
        return v, (tr2, sc)

    og = jit(jax.value_and_grad(obj, has_aux=True))
    p, sg = P_BASE, jnp.array(S_BASE * T / 2.0)
    for _ in range(5):
        sg, d = snewt(p, sg)
    m = jnp.zeros_like(p)
    vv = jnp.zeros_like(p)
    for t in range(1, steps + 1):
        sg, d = snewt(jax.lax.stop_gradient(p), sg)
        (v, (tr2, sc)), g = og(p, sg, d)
        sg = sc
        m = 0.9 * m + 0.1 * g
        vv = 0.999 * vv + 0.001 * g ** 2
        p = p - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
    v, tr, res = solve(p, sg)
    return float(v), float(tr), float(res)


def snap_to_spec(p, sigma, T, n=3):
    """One-shot design + a few Newton steps on the scale to hit travel exactly.
    Cheap (a handful of solves) and reported separately from the pure
    feed-forward number."""
    sg = sigma
    for _ in range(n):
        tr, d = jax.value_and_grad(lambda s: solve(p, s)[1])(sg)
        sg = sg - (tr - T) / d
    v, tr, res = solve(p, sg)
    return float(v), float(tr), float(res)


if __name__ == "__main__":
    key = random.PRNGKey(0)
    key, nk = random.split(key)
    theta = init_net(nk)

    BATCH, ITERS, LR = 24, 400, 3e-3
    print(f"Amortized inverse design: spec (required travel) -> gap profile")
    print(f"trained through the fold solver, ZERO stored simulations")
    print(f"spec range {T_LO}-{T_HI} um, batch {BATCH}, {ITERS} iters\n")

    b1, b2 = 0.9, 0.999
    m = jax.tree_util.tree_map(jnp.zeros_like, theta)
    vv = jax.tree_util.tree_map(jnp.zeros_like, theta)
    t0 = time.perf_counter()
    for it in range(1, ITERS + 1):
        key, tk = random.split(key)
        Ts = random.uniform(tk, (BATCH,), minval=T_LO, maxval=T_HI)
        (l, (v, tr, res)), g = grad_fn(theta, Ts)
        g = jax.tree_util.tree_map(jnp.nan_to_num, g)
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        vv = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b ** 2, vv, g)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** it), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** it), vv)
        theta = jax.tree_util.tree_map(
            lambda w, a, b: w - LR * a / (jnp.sqrt(b) + 1e-8), theta, mh, vh)
        if it % 50 == 0 or it == 1:
            print(f"  it {it:4d}  loss={float(l):7.3f}  meanV={float(jnp.mean(v)):7.3f}"
                  f"  travel_err={float(jnp.mean(jnp.abs(tr - Ts))):.4f}"
                  f"  |res|={float(jnp.max(res)):.1e}")
    t_train = time.perf_counter() - t0
    n_solves = BATCH * ITERS
    print(f"  training: {t_train:.0f}s, {n_solves:,} live solves, "
          f"0 stored samples")

    import pickle
    with open("amort_theta.pkl", "wb") as f:
        pickle.dump(jax.tree_util.tree_map(lambda a: np.asarray(a), theta), f)
    print("  saved weights -> amort_theta.pkl")

    # ---------------------- inference speed ----------------------
    fwd = jit(lambda T: net_design(theta, T))
    _ = fwd(jnp.array(2.0))[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(1000):
        out = fwd(jnp.array(2.0))
    out[0].block_until_ready()
    t_infer = (time.perf_counter() - t0) / 1000
    print(f"  inference: {t_infer*1e3:.4f} ms per design (forward pass only)\n")

    # ---------------------- quality vs direct optimisation ----------------------
    print(f"=== one-shot network vs full per-spec optimisation ===")
    print(f"  {'spec T':>8}{'net V_PI':>10}{'net travel':>12}"
          f"{'+snap V_PI':>12}{'direct V_PI':>13}{'cost':>8}{'direct t':>10}")
    rows = []
    for T in [1.0, 1.5, 2.0, 2.5, 3.0]:
        p, sg = net_design(theta, jnp.array(T))
        v_n, tr_n, res_n = solve(p, sg)
        v_s, tr_s, res_s = snap_to_spec(p, sg, T)
        t0 = time.perf_counter()
        v_d, tr_d, res_d = direct_optimise(T)
        t_d = time.perf_counter() - t0
        cost = 100.0 * (v_s - v_d) / v_d
        rows.append((T, float(v_n), float(tr_n), v_s, v_d, cost, t_d))
        print(f"  {T:>8.2f}{float(v_n):>10.3f}{float(tr_n):>12.4f}"
              f"{v_s:>12.3f}{v_d:>13.3f}{cost:>+7.1f}%{t_d:>9.1f}s")

    mean_cost = float(np.mean([r[5] for r in rows]))
    mean_direct_t = float(np.mean([r[6] for r in rows]))
    print(f"\n  amortisation cost: {mean_cost:+.1f}% mean V_PI vs full optimisation")
    print(f"  speedup: {mean_direct_t/t_infer:,.0f}x "
          f"({mean_direct_t:.1f}s -> {t_infer*1e3:.4f}ms)")
    print(f"  NOTE: the +snap column adds ~3 solves to pin travel exactly;")
    print(f"        the pure forward pass is the {t_infer*1e3:.4f} ms number.")
    np.save("amortized_rows.npy", np.array([r[:6] for r in rows]))
