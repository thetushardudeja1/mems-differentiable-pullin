"""
Fourier Neural Operator as the surrogate, everything else held identical.

WHY
The previous baseline was an MLP on the 13 design coefficients. The stronger,
more standard choice for "AI-accelerated multiphysics / surrogate simulation"
is a neural operator that consumes the PHYSICAL FIELD -- the gap profile D(xi)
-- rather than its parameterisation. FNOs are resolution-invariant and capture
spatial structure the MLP cannot see, so this is the fairest strong baseline.

ARCHITECTURE (Li et al. FNO, 1D)
    lift            D(xi) [+ xi coordinate]  ->  width channels
    L spectral blocks:  y = sigma( IFFT(R . FFT(x))[:modes] + W x )
    pool + head     -> (V_PI, travel)

Everything outside the architecture is unchanged from surrogate_fair.py: the
same 400-solve budget, the same active-learning rounds, the same trust region,
the same feasible restarts, and the same final scoring by projecting onto
travel = 2 um with the true solver.
"""

import time
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, vmap, jit

import surrogate_fair as F     # reuse task, solver wrapper, budget counter

M_NODES = F.XI.shape[0]
MODES = 12          # kept OUT of the parameter pytree: an int inside it
                    # makes jax.grad fail ("grad requires real- or complex-valued
                    # inputs ... but got int64").


# --------------------------------- FNO --------------------------------------
def init_fno(key, width=32, modes=MODES, depth=4, d_in=2, d_out=2):
    ks = random.split(key, 3 + 3 * depth)
    p = {"lift": (random.normal(ks[0], (d_in, width)) / jnp.sqrt(d_in),
                  jnp.zeros(width)),
         "head": (random.normal(ks[1], (width, 64)) / jnp.sqrt(width),
                  jnp.zeros(64),
                  random.normal(ks[2], (64, d_out)) / jnp.sqrt(64),
                  jnp.zeros(d_out)),
         "blocks": []}
    for i in range(depth):
        k1, k2, k3 = ks[3 + 3 * i], ks[4 + 3 * i], ks[5 + 3 * i]
        scale = 1.0 / (width * width)
        p["blocks"].append({
            # spectral weights, real and imaginary stored separately
            "Rr": scale * random.normal(k1, (modes, width, width)),
            "Ri": scale * random.normal(k2, (modes, width, width)),
            "W": random.normal(k3, (width, width)) / jnp.sqrt(width),
            "b": jnp.zeros(width),
        })
    return p


def spectral(block, x, modes):
    """x: (n, width) -> (n, width) via truncated Fourier multiplication."""
    n = x.shape[0]
    xh = jnp.fft.rfft(x, axis=0)                      # (n//2+1, width)
    m = min(modes, xh.shape[0])
    R = block["Rr"][:m] + 1j * block["Ri"][:m]        # (m, width, width)
    outh = jnp.zeros_like(xh)
    outh = outh.at[:m].set(jnp.einsum("mi,mio->mo", xh[:m], R))
    return jnp.fft.irfft(outh, n=n, axis=0)


def fno_forward(p, D, modes=MODES):
    """D: (n,) gap profile -> (2,) = (V_PI, travel)."""
    xi = jnp.linspace(0.0, 1.0, D.shape[0])
    x = jnp.stack([D, xi], axis=1)                    # (n, 2)
    Wl, bl = p["lift"]
    x = x @ Wl + bl
    for blk in p["blocks"]:
        x = jax.nn.gelu(spectral(blk, x, modes) + x @ blk["W"] + blk["b"])
    x = jnp.mean(x, axis=0)                           # global pooling
    W1, b1, W2, b2 = p["head"]
    return jax.nn.gelu(x @ W1 + b1) @ W2 + b2


fno_batch = vmap(fno_forward, in_axes=(None, 0))


def fit_fno(key, Dprof, Y, steps=1500, lr=1e-3):
    """Dprof: (n_samples, n_nodes) gap profiles. Y: (n_samples, 2)."""
    mu_d, sd_d = Dprof.mean(), Dprof.std() + 1e-8
    my, sy = Y.mean(0), Y.std(0) + 1e-8
    Dn, Yn = (Dprof - mu_d) / sd_d, (Y - my) / sy
    p = init_fno(key)

    def loss(p):
        return jnp.mean((fno_batch(p, Dn) - Yn) ** 2)

    gl = jit(jax.value_and_grad(loss))
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    l = jnp.inf
    for t in range(1, steps + 1):
        l, g = gl(p)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b ** 2, v, g)
        p = jax.tree_util.tree_map(
            lambda w, mm, vv: w - lr * (mm / (1 - 0.9 ** t))
            / (jnp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8), p, m, v)
    return (p, mu_d, sd_d, my, sy), float(l)


def fno_predict(sur, pdesign, sigma):
    p, mu_d, sd_d, my, sy = sur
    D = F.D_of(pdesign, sigma)
    return fno_forward(p, (D - mu_d) / sd_d) * sy + my


if __name__ == "__main__":
    key = random.PRNGKey(0)
    print("FNO surrogate vs direct differentiable physics")
    print(f"budget: {F.BUDGET} true fold solves;  gap profile sampled at "
          f"{M_NODES} nodes")
    print(f"FNO: width 32, 12 Fourier modes, 4 blocks, field input D(xi)\n")

    key, k0 = random.split(key)
    p0s, s0s = F.sample_designs(k0, 1)
    p0, s0 = p0s[0], s0s[0]

    F.CTR.n = 0
    t0 = time.perf_counter()
    n_init = 200
    key, ki = random.split(key)
    ps, ss = F.sample_designs(ki, n_init)
    vs, trs, ress = F._solve_batch(ps, ss)
    F.CTR.spend(n_init)
    ok = (ress < 1e-6) & jnp.isfinite(vs) & jnp.isfinite(trs)
    Xp, Xs = ps[ok], ss[ok]
    Y = jnp.stack([vs[ok], trs[ok]], axis=1)
    Dp = vmap(F.D_of)(Xp, Xs)

    print(f"  {'round':>6}{'n_data':>8}{'fit MSE':>11}{'pred V':>9}"
          f"{'true V':>9}{'true travel':>13}{'budget':>9}")
    p_cur, s_cur = p0, s0
    for rd in range(10):
        key, kf = random.split(key)
        sur, mse = fit_fno(kf, Dp, Y)

        lo = jnp.concatenate([jnp.min(Xp, 0), jnp.array([jnp.min(Xs)])])
        hi = jnp.concatenate([jnp.max(Xp, 0), jnp.array([jnp.max(Xs)])])

        def sobj(z):
            pr = fno_predict(sur, z[:-1], z[-1])
            return pr[0] + 60.0 * (F.TRAVEL_REQ - pr[1]) ** 2

        z = F.adam_opt(sobj, jnp.clip(jnp.concatenate([p_cur, jnp.array([s_cur])]),
                                      lo, hi), 250, lo=lo, hi=hi)
        p_cur, s_cur = z[:-1], z[-1]
        pr = fno_predict(sur, p_cur, s_cur)

        key, ke = random.split(key)
        pe, se = F.sample_designs(ke, 19, centre=(p_cur, s_cur))
        pe = jnp.concatenate([p_cur[None, :], pe], axis=0)
        se = jnp.concatenate([jnp.array([s_cur]), se])
        ve, tre, rese = F._solve_batch(pe, se)
        F.CTR.spend(20)
        oke = (rese < 1e-6) & jnp.isfinite(ve) & jnp.isfinite(tre)
        Xp = jnp.concatenate([Xp, pe[oke]])
        Xs = jnp.concatenate([Xs, se[oke]])
        Y = jnp.concatenate([Y, jnp.stack([ve[oke], tre[oke]], axis=1)])
        Dp = jnp.concatenate([Dp, vmap(F.D_of)(pe[oke], se[oke])])
        cand = (f"{float(ve[0]):>9.3f}{float(tre[0]):>13.4f}"
                if bool(oke[0]) else f"{'diverged':>9}{'-':>13}")
        print(f"  {rd+1:>6}{Y.shape[0]:>8}{mse:>11.2e}{float(pr[0]):>9.3f}"
              f"{cand}{F.CTR.n:>9}")

        fe = jnp.abs(Y[:, 1] - F.TRAVEL_REQ) < 0.10
        if bool(jnp.any(fe)):
            idx = jnp.argmin(jnp.where(fe, Y[:, 0], jnp.inf))
            p_cur, s_cur = Xp[idx], Xs[idx]

    t_fno = time.perf_counter() - t0
    res = F.project_true(p_cur, s_cur)
    fe = jnp.abs(Y[:, 1] - F.TRAVEL_REQ) < 0.05
    best_samp = float(jnp.min(Y[fe, 0])) if bool(jnp.any(fe)) else float("nan")
    print(f"  time {t_fno:.0f}s")
    print(f"  final design -> "
          f"{'V_PI = %.3f V at travel %.4f' % res if res else 'INFEASIBLE'}")
    print(f"  best feasible design ever sampled: {best_samp:.3f} V")

    print(f"\n{'='*62}\nSUMMARY at {F.BUDGET} true solves, scored at travel = 2 um")
    print(f"  FNO surrogate + active learning : "
          f"{('%.3f V' % res[0]) if res else 'INFEASIBLE'}")
    print(f"  MLP surrogate + active learning : 17.203 V   (surrogate_fair.py)")
    print(f"  direct differentiable physics   : 13.783 V   (surrogate_fair.py)")
    print(f"  hand-tuned polynomial           : 13.62  V   (Haluzan reproduction)")
    if res:
        print(f"  direct vs FNO: {100*(res[0]-13.783)/13.783:+.1f}%")
