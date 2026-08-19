"""
Adaptive operating-point selection under unknown fabrication parameters.

THE PROBLEM (quantified in test_env2dof.py)
The tip-in ceiling r_ti = 1/(1+beta) varies device-to-device because
beta = k_x L_c^2 / (6 k_theta) depends on fabrication. beta is NOT measurable
at test time. So a fixed-gain controller must target the WORST-CASE device
(r = 0.635 over our distribution) while per-device oracle tuning would average
0.784 -- a 19% loss of usable travel, purely to conservatism.

WHAT THE POLICY DOES
It chooses the operating point adaptively: each step it nudges the target
deflection, watching the device's own tilt response to infer how close it is to
its ceiling. The low-level PD tracker is UNCHANGED from the classical baseline,
so the comparison isolates target selection -- the thing conservatism costs us.

IDENTIFIABILITY (why the tilt is the informative signal)
A persistent fabrication asymmetry produces a static tilt
    th_ss = asym / (wr^2 * k_eff),      k_eff = 1 - beta*r/(1-r)
which DIVERGES as r -> r_ti. Measuring th at known r therefore reveals k_eff
and hence beta. But the signal is small far from the ceiling: at beta=0.1747,
th_ss is 1.7e-3 at r=0.5 and 4.6e-3 at r=0.8, against measurement noise 5e-3.
A memoryless policy cannot see it. The observation therefore carries an
exponential moving average of |th| -- a denoised tilt estimate, which is what a
real averaged/lock-in capacitive readout provides.

ALGORITHM
REINFORCE with a batch-mean baseline. Score-function rather than pathwise
gradients on purpose: the episode contains a hard, irreversible termination
(device destroyed), and Suh et al. (ICML 2022) show differentiable simulators
give poor policy gradients through exactly that kind of discontinuity.
"""

import time
import jax
import jax.numpy as jnp
from jax import random, vmap, jit

import env2dof as E

T_STEPS = 600
TH_NOISE = 5e-3          # tilt readout noise
R_NOISE = 2e-3           # position readout noise
EMA = 0.02               # tilt averaging constant
DEATH_PENALTY = float(__import__("os").environ.get("DEATH_PENALTY", "1.0"))  # same scale as mean-travel return
SIGMA = 0.10             # exploration std on the target increment
DTARGET_MAX = 0.01       # per-step target slew limit

# fabrication distribution (beta spans the range probed in test_env2dof.py)
BETA_LO, BETA_HI = 0.10, 0.60
Q_LO, Q_HI = 3.0, 20.0
ASYM_LO, ASYM_HI = 2e-3, 5e-3


def sample_devices(key, n):
    k1, k2, k3 = random.split(key, 3)
    return dict(
        beta=random.uniform(k1, (n,), minval=BETA_LO, maxval=BETA_HI),
        Q=random.uniform(k2, (n,), minval=Q_LO, maxval=Q_HI),
        asym=random.uniform(k3, (n,), minval=ASYM_LO, maxval=ASYM_HI),
    )


def init_policy(key, obs_dim=7, hidden=64):
    k1, k2, k3 = random.split(key, 3)
    return (random.normal(k1, (obs_dim, hidden)) * (1.0 / jnp.sqrt(obs_dim)),
            jnp.zeros(hidden),
            random.normal(k2, (hidden, hidden)) * (1.0 / jnp.sqrt(hidden)),
            jnp.zeros(hidden),
            random.normal(k3, (hidden, 1)) * 0.01,
            jnp.zeros(1))


def policy_mean(params, obs):
    W1, b1, W2, b2, W3, b3 = params
    h = jnp.tanh(obs @ W1 + b1)
    h = jnp.tanh(h @ W2 + b2)
    return jnp.tanh(h @ W3 + b3)[..., 0]


def _device_params(beta, Q, asym):
    p = E.make_params(Q=Q, beta=beta, wu=20.0, asym=asym)
    return p


def rollout(params, beta, Q, asym, key, stochastic=True, policy=None):
    """One device. Returns (return, mean_travel_when_alive, destroyed, log_prob)."""
    p = _device_params(beta, Q, asym)

    def body(carry, k):
        state, r_t, th_ema, logp = carry
        r, dr, th, dth, v, dead = state

        k1, k2, k3 = random.split(k, 3)
        r_meas = jnp.clip(r + R_NOISE * random.normal(k1), 0.0, 1.0)
        th_meas = th + TH_NOISE * random.normal(k2)
        th_ema_new = (1 - EMA) * th_ema + EMA * jnp.abs(th_meas)

        # Proximity-to-ceiling proxy. th_ss = asym/(wr^2 k_eff) diverges as the
        # rotational stiffness vanishes, so a LARGE averaged tilt means "close
        # to tip-in". Taking log keeps it bounded and well-scaled; the raw
        # ratio th_ema/(1 - r) blew up to ~1e4 on dead devices (r frozen at
        # exactly 1.0) and to +-inf when readout noise pushed r_meas past 1,
        # which is what produced NaNs from iteration 2 onward.
        prox = jnp.log10(th_ema_new + 1e-4) - jnp.log10(1e-4)

        obs = jnp.array([r_meas, jnp.clip(dr, -5.0, 5.0), th_ema_new,
                         jnp.abs(th_meas), v, r_t, prox])

        if policy is None:
            mean_a = policy_mean(params, obs)
        else:
            mean_a = policy(obs, r_t, beta)

        if stochastic:
            eps = random.normal(k3)
            a = jax.lax.stop_gradient(mean_a + SIGMA * eps)
            logp = logp + (-0.5 * ((a - mean_a) / SIGMA) ** 2
                           - jnp.log(SIGMA) - 0.5 * jnp.log(2 * jnp.pi))
        else:
            a = mean_a

        r_t_new = jnp.clip(r_t + DTARGET_MAX * jnp.clip(a, -1.0, 1.0), 0.05, 0.95)

        # unchanged low-level tracker
        pd_obs = jnp.array([r_meas, dr, th_meas, dth, v, r_t_new,
                            r_t_new - r_meas])
        v_cmd = E.classical_policy(pd_obs, p)

        pk = dict(p)
        pk["kick"] = E.TH_NOISE * random.normal(jax.random.fold_in(k, 11))
        new_state = E.step(state, v_cmd, pk)
        return (new_state, r_t_new, th_ema_new, logp), (new_state[0], new_state[5])

    s0 = (0.0, 0.0, 0.0, 0.0, 0.0, False)
    keys = random.split(key, T_STEPS)
    (_, _, _, logp), (rs, deads) = jax.lax.scan(
        body, (s0, jnp.array(0.10), jnp.array(0.0), jnp.array(0.0)), keys)

    destroyed = deads[-1]
    alive_mask = jnp.logical_not(deads)
    travel = jnp.sum(rs * alive_mask) / T_STEPS
    ret = travel - DEATH_PENALTY * destroyed
    return ret, travel, destroyed, logp


# ------------------------------- baselines ----------------------------------

def fixed_target_policy(r_fixed):
    """Conservative fixed operating point -- what you must use when beta is
    unknown and you cannot adapt."""
    def pol(obs, r_t, beta):
        return jnp.clip((r_fixed - r_t) / DTARGET_MAX, -1.0, 1.0)
    return pol


def oracle_policy(margin=1.01):  # 1.01 = best SAFE margin (oracle_frontier.py);
    """Cheats: knows beta, so targets just under this device's own ceiling.
    NOT an upper bound -- it is a reference at a chosen safety margin. At
    margin=0.97 a trained policy legitimately beat it (reported as a nonsense
    "106% of headroom recovered"). margin=1.01 is the best margin that still
    destroys 0% of devices; 1.03 collapses to 43% destruction."""
    def pol(obs, r_t, beta):
        r_goal = margin * E.r_tipin(beta)
        return jnp.clip((r_goal - r_t) / DTARGET_MAX, -1.0, 1.0)
    return pol


def evaluate(params, devices, key, policy=None, n=None):
    n = devices["beta"].shape[0] if n is None else n
    keys = random.split(key, n)
    rets, travels, dead, _ = vmap(
        lambda b, q, a, k: rollout(params, b, q, a, k,
                                   stochastic=False, policy=policy)
    )(devices["beta"][:n], devices["Q"][:n], devices["asym"][:n], keys)
    return (float(jnp.mean(travels)), float(jnp.mean(dead)),
            float(jnp.mean(rets)))


def reinforce_loss(params, devices, keys):
    rets, travels, dead, logps = vmap(
        lambda b, q, a, k: rollout(params, b, q, a, k, stochastic=True)
    )(devices["beta"], devices["Q"], devices["asym"], keys)
    adv = rets - jnp.mean(rets)
    adv = adv / (jnp.std(adv) + 1e-6)
    return -jnp.mean(adv * logps), (jnp.mean(rets), jnp.mean(travels),
                                    jnp.mean(dead))


grad_fn = jit(jax.value_and_grad(reinforce_loss, has_aux=True))


if __name__ == "__main__":
    print("device:", jax.devices()[0])
    key = random.PRNGKey(0)
    key, pk, dk = random.split(key, 3)
    params = init_policy(pk)

    # held-out evaluation set, fixed for every method
    eval_devices = sample_devices(dk, 512)
    key, ek = random.split(key)

    print("\n=== baselines on the held-out device set ===")
    r_ti_all = E.r_tipin(eval_devices["beta"])
    print(f"  ceiling r_ti: min={float(jnp.min(r_ti_all)):.3f} "
          f"mean={float(jnp.mean(r_ti_all)):.3f} "
          f"max={float(jnp.max(r_ti_all)):.3f}")

    worst_ceiling = float(jnp.min(r_ti_all))
    # Best fixed target that still destroys nothing (swept in oracle_frontier.py).
    # Comparing against a handicapped 0.95*worst inflated our gain to +20.4%.
    r_conservative = 1.02 * worst_ceiling
    tr_c, d_c, _ = evaluate(params, eval_devices, ek,
                            policy=fixed_target_policy(r_conservative))
    print(f"  fixed conservative (r={r_conservative:.3f}): "
          f"travel={tr_c:.4f}  destroyed={d_c * 100:.1f}%")

    tr_o, d_o, _ = evaluate(params, eval_devices, ek, policy=oracle_policy(margin=1.01))
    print(f"  oracle (knows beta):                 "
          f"travel={tr_o:.4f}  destroyed={d_o * 100:.1f}%")
    headroom = tr_o - tr_c
    print(f"  headroom available to adaptation: {headroom:.4f} "
          f"({100 * headroom / tr_c:.1f}% of conservative)")

    print("\n=== training ===")
    import sys
    N_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    N_ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-3
    print(f"  n_envs={N_ENVS}  n_iters={N_ITERS}  lr={LR}  "
          f"death_penalty={DEATH_PENALTY}")
    b1, b2 = 0.9, 0.999
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    vv = jax.tree_util.tree_map(jnp.zeros_like, params)

    t0 = time.perf_counter()
    for it in range(1, N_ITERS + 1):
        key, dks, rks = random.split(key, 3)
        devs = sample_devices(dks, N_ENVS)
        keys = random.split(rks, N_ENVS)
        (loss, (mret, mtrav, mdead)), g = grad_fn(params, devs, keys)
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        vv = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b ** 2, vv, g)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** it), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** it), vv)
        params = jax.tree_util.tree_map(
            lambda p, a, b: p - LR * a / (jnp.sqrt(b) + 1e-8), params, mh, vh)
        if it % 30 == 0 or it == 1:
            print(f"  it {it:4d}  return={float(mret):+.4f}  "
                  f"travel={float(mtrav):.4f}  destroyed={float(mdead) * 100:5.1f}%")
    print(f"  training time: {time.perf_counter() - t0:.1f}s")

    print("\n=== held-out evaluation ===")
    tr_rl, d_rl, _ = evaluate(params, eval_devices, ek)
    print(f"  fixed conservative : travel={tr_c:.4f}  destroyed={d_c * 100:.1f}%")
    print(f"  RL adaptive        : travel={tr_rl:.4f}  destroyed={d_rl * 100:.1f}%")
    print(f"  oracle (upper bnd) : travel={tr_o:.4f}  destroyed={d_o * 100:.1f}%")
    if headroom > 1e-6:
        rec = 100 * (tr_rl - tr_c) / headroom
        print(f"\n  headroom recovered: {rec:.1f}%")
    print(f"  travel vs conservative: {100 * (tr_rl / tr_c - 1):+.1f}%")
    if d_rl > d_c + 0.02:
        print(f"  WARNING: RL destroys {d_rl * 100:.1f}% of devices vs "
              f"{d_c * 100:.1f}% for the baseline -- extra travel bought with "
              f"broken parts does not count.")
