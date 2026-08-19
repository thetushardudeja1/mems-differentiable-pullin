"""
Amortized inverse design with a HARD constraint layer.

PROBLEM WITH THE FIRST VERSION (amortized_design.py)
It enforced "travel == T" with a soft penalty, and the one-shot design came out
systematically UNDER spec: travel 1.895 for T = 2.0, a 5% miss. That is the
textbook failure of penalty methods -- lowering travel also lowers V_PI, so the
optimiser trades constraint violation for objective and settles biased low.

THE STANDARD FIX (and it is standard)
The amortized-optimisation literature calls this soft- vs hard-constraint
handling, and the remedy is a differentiable ENFORCEMENT LAYER that satisfies
the constraint by construction, differentiated via the implicit function
theorem (OptNet; HardNet, Azizan et al.; PiNet; FSNet). We already had exactly
that machinery -- the sigma-Newton with implicit-gradient correction used by
inverse_design.py -- it simply was not wired into the network.

ARCHITECTURE
    T --> shape head      --> p (cosine coefficients)
    (p, T) --> ENFORCEMENT LAYER: solve sigma s.t. travel(p, sigma) == T
               exactly, by Newton; gradients via the implicit function theorem
    loss = V_PI(p, sigma*)          <-- no penalty term at all
    T --> scale head      --> sigma_hat, regressed onto stop_grad(sigma*)

The scale head exists so that INFERENCE is still a single forward pass: at
deployment we use (p(T), sigma_hat(T)) with no solver in the loop. Its target
is exact, and T -> sigma* is smooth and one-dimensional, so it should fit
tightly.

REPORTED HONESTLY
  * feed-forward     : both heads, no solver -- the millisecond number
  * + enforcement    : shape head + Newton layer -- exact spec, a few solves
  * direct optimise  : full per-spec optimisation -- the quality reference
"""

import time
import pickle
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, vmap, jit
import numpy as np

from amortized_design import (V_SCALE, N, BC, K, BASIS, T_LO, T_HI,
                              P_BASE, S_BASE, D_of, solve, direct_optimise)

W_SCALE_HEAD = 30.0        # weight on the scale-head regression


def init_net(key, hidden=64):
    k1, k2, k3, k4 = random.split(key, 4)
    return dict(
        W1=random.normal(k1, (2, hidden)) * 0.5, b1=jnp.zeros(hidden),
        W2=random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden),
        b2=jnp.zeros(hidden),
        Wp=random.normal(k3, (hidden, K)) * 1e-3, bp=jnp.zeros(K),
        Ws=random.normal(k4, (hidden, 1)) * 1e-3, bs=jnp.zeros(1),
    )


def trunk(theta, T):
    x = jnp.array([(T - T_LO) / (T_HI - T_LO), jnp.log(T)])
    h = jnp.tanh(x @ theta["W1"] + theta["b1"])
    return jnp.tanh(h @ theta["W2"] + theta["b2"])


def shape_head(theta, T):
    return P_BASE + trunk(theta, T) @ theta["Wp"] + theta["bp"]


def scale_head(theta, T):
    raw = (trunk(theta, T) @ theta["Ws"] + theta["bs"])[0]
    return S_BASE * jnp.exp(jnp.clip(raw, -2.0, 2.0)) * (T / 2.0)


def enforce_sigma(p, T, sigma0, n_newton=4):
    """Differentiable constraint layer: solve travel(p, sigma) == T.

    Newton on sigma, then one final correction step whose p-dependence is kept
    live while d(travel)/d(sigma) is frozen. At convergence that reproduces the
    implicit-function-theorem gradient  d sigma*/dp = -F_p / F_sigma, so the
    constraint is satisfied exactly AND the shape head receives correct
    gradients -- the same trick used for the fold system itself.
    """
    def travel_of(s):
        return solve(p, s)[1]

    sg = jax.lax.stop_gradient(sigma0)
    for _ in range(n_newton):
        tr, d = jax.value_and_grad(
            lambda s: solve(jax.lax.stop_gradient(p), s)[1])(sg)
        sg = sg - (tr - T) / d
    sg = jax.lax.stop_gradient(sg)
    tr, d = jax.value_and_grad(travel_of)(sg)          # p-dependence live here
    return sg - (tr - T) / jax.lax.stop_gradient(d)


def loss_one(theta, T):
    p = shape_head(theta, T)
    s_hat = scale_head(theta, T)
    s_star = enforce_sigma(p, T, s_hat)
    v, travel, res = solve(p, s_star)
    # scale head learns to reproduce the enforcement layer's answer
    l_scale = (jnp.log(s_hat) - jnp.log(jax.lax.stop_gradient(s_star))) ** 2
    return v + W_SCALE_HEAD * l_scale, (v, travel, res, s_star, s_hat)


def loss_batch(theta, Ts):
    ls, aux = vmap(lambda T: loss_one(theta, T))(Ts)
    return jnp.mean(ls), aux


grad_fn = jit(jax.value_and_grad(loss_batch, has_aux=True))


if __name__ == "__main__":
    key = random.PRNGKey(0)
    key, nk = random.split(key)
    theta = init_net(nk)
    BATCH, ITERS, LR = 20, 300, 3e-3

    print("Amortized design with a HARD constraint (enforcement) layer")
    print("loss has NO penalty term; travel is satisfied by construction\n")
    b1, b2 = 0.9, 0.999
    m = jax.tree_util.tree_map(jnp.zeros_like, theta)
    vv = jax.tree_util.tree_map(jnp.zeros_like, theta)
    t0 = time.perf_counter()
    for it in range(1, ITERS + 1):
        key, tk = random.split(key)
        Ts = random.uniform(tk, (BATCH,), minval=T_LO, maxval=T_HI)
        (l, (v, tr, res, ss, sh)), g = grad_fn(theta, Ts)
        g = jax.tree_util.tree_map(jnp.nan_to_num, g)
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        vv = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b ** 2, vv, g)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** it), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** it), vv)
        theta = jax.tree_util.tree_map(
            lambda w, a, b: w - LR * a / (jnp.sqrt(b) + 1e-8), theta, mh, vh)
        if it % 50 == 0 or it == 1:
            print(f"  it {it:4d}  meanV={float(jnp.mean(v)):7.3f}  "
                  f"travel_err={float(jnp.mean(jnp.abs(tr - Ts))):.5f}  "
                  f"scale_head_err={float(jnp.mean(jnp.abs(sh/ss - 1))*100):5.2f}%"
                  f"  |res|={float(jnp.max(res)):.1e}")
    t_train = time.perf_counter() - t0
    print(f"  training {t_train:.0f}s, {BATCH*ITERS:,} live solves, "
          f"0 stored samples")
    with open("amort_hard_theta.pkl", "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, theta), f)

    fwd = jit(lambda T: (shape_head(theta, T), scale_head(theta, T)))
    _ = fwd(jnp.array(2.0))[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(2000):
        o = fwd(jnp.array(2.0))
    o[0].block_until_ready()
    t_inf = (time.perf_counter() - t0) / 2000

    print(f"\n  feed-forward inference: {t_inf*1e3:.4f} ms\n")
    print(f"=== quality and spec accuracy ===")
    print(f"  {'T':>6}{'ff V_PI':>10}{'ff travel':>11}{'ff err':>9}"
          f"{'enf V_PI':>10}{'enf travel':>12}{'direct':>9}{'gap':>8}")
    rows = []
    for T in [1.0, 1.5, 2.0, 2.5, 3.0]:
        Tj = jnp.array(T)
        p, s_hat = fwd(Tj)
        v_ff, tr_ff, _ = solve(p, s_hat)
        s_star = enforce_sigma(p, Tj, s_hat)
        v_e, tr_e, _ = solve(p, s_star)
        v_d, tr_d, _ = direct_optimise(T)
        gap = 100.0 * (float(v_e) - v_d) / v_d
        err = 100.0 * abs(float(tr_ff) - T) / T
        rows.append((T, float(v_ff), float(tr_ff), err, float(v_e), v_d, gap))
        print(f"  {T:>6.2f}{float(v_ff):>10.3f}{float(tr_ff):>11.4f}"
              f"{err:>8.2f}%{float(v_e):>10.3f}{float(tr_e):>12.4f}"
              f"{v_d:>9.3f}{gap:>+7.1f}%")

    mean_err = float(np.mean([r[3] for r in rows]))
    mean_gap = float(np.mean([r[6] for r in rows]))
    print(f"\n  feed-forward spec error : {mean_err:.2f}%  "
          f"(penalty version was ~5%)")
    print(f"  quality vs direct       : {mean_gap:+.1f}%")
    np.save("amort_hard_rows.npy", np.array(rows))
