"""
2-DOF electrostatic actuator as an RL environment, with IRREVERSIBLE failure.
JAX, fp32-friendly, vmap-batchable.

PLANT (nondimensional, tau = omega_n t, r = x/g0), from Seeger & Boser 2003:

  translation   r'' + (1/Q) r' + r = lam(v, r) = LAM_SCALE * v^2 / (1-r)^2
  rotation      th'' + (wr/Qth) th' + wr^2 (1 - beta*r/(1-r)) th = noise
  actuator      v'  = (w_u/w_n) (v_cmd - v)        <-- finite amplifier bandwidth

with beta = k_x L_c^2 / (6 k_th), so the rotational stiffness vanishes at
r_ti = 1/(1+beta): their eq 22, validated in model2dof.py against their Table I
(design #1 -> 19.0% vs 19% published, design #3 -> 85.1% vs 85%).

WHY THE ACTUATOR LAG IS THE POINT
Their eq 41 requires amplifier bandwidth w_u > 2 w_n Q (Cs+Ctop+Cin)/C0 for
classical charge control to remain stable, i.e. bandwidth must scale with Q.
They state: "In a vacuum, the amplifier bandwidth requirement might be
prohibitively large." Modelling w_u explicitly means the agent faces exactly
that limitation rather than a proxy for it. Reproduced collapse (model2dof.py,
design #3): 85.1% of gap at Q<=10 but only 9.4% at Q=1000, against a mechanical
tip-in ceiling of 85.1%.

FAILURE IS IRREVERSIBLE (latching), which is what makes this a safety problem
rather than a tracking problem:
  contact  r >= 1
  tip-in   |th| >= TH_MAX
Both terminate the episode; a destroyed device does not recover.

Rotation is seeded with small process noise: th = 0 is an equilibrium, so a
perfectly symmetric device would never tip even when the mode is unstable.
Real devices tip because of fabrication asymmetry -- Seeger & Boser attribute
their large-deflection discrepancies to exactly that.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit

LAM_FOLD = 4.0 / 27.0
LAM_SCALE = 0.30          # v=1 over-drives past pull-in, so the bound matters
DT = 0.02
T_STEPS = 600
TH_MAX = 1.0              # normalized tip-in angle = destroyed
TH_NOISE = 2e-3           # fabrication asymmetry / thermal kick


def r_tipin(beta):
    return 1.0 / (1.0 + beta)


def stiffness_factor(r, beta):
    """1 + k_theta_e/k_theta. Negative => rotation mode unstable => tip-in."""
    rc = jnp.clip(r, 0.0, 0.995)
    return 1.0 - beta * rc / (1.0 - rc)


def step(state, v_cmd, p):
    """One explicit-Euler step. state = (r, dr, th, dth, v, dead), p = params."""
    r, dr, th, dth, v, dead = state

    # Actuator: finite amplifier bandwidth, integrated EXACTLY.
    # v' = wu (v_cmd - v) is linear, so v(t) = v_cmd + (v0 - v_cmd) exp(-wu t).
    # Explicit Euler would need DT*wu < 2 and blew up at wu >= 100 (DT=0.02),
    # making reachable travel collapse to zero at HIGH bandwidth -- backwards.
    vc = jnp.clip(v_cmd, 0.0, 1.0)
    v_new = jnp.clip(vc + (v - vc) * jnp.exp(-DT * p["wu"]), 0.0, 1.0)

    rc = jnp.clip(r, 0.0, 0.98)
    lam = LAM_SCALE * v_new ** 2 / (1.0 - rc) ** 2
    ddr = lam - rc - dr / p["Q"]
    dr_new = dr + DT * ddr
    r_new = jnp.clip(r + DT * dr_new, 0.0, 1.0)

    # Rotation. The forcing is a PERSISTENT per-device asymmetry (p["asym"])
    # plus a small stochastic kick. The asymmetry is what makes tip-in
    # decisive: a static offset torque produces a static tilt
    # th_ss = asym / (wr^2 k_eff) that DIVERGES as the rotational stiffness
    # k_eff -> 0, so crossing r_tipin destroys the device promptly. With
    # zero-mean noise alone the instability needed tau ~ 11 to grow out of the
    # noise floor, comparable to the episode length, so devices sometimes
    # survived above the ceiling. Seeger & Boser attribute their own
    # large-deflection discrepancies to fabrication asymmetry.
    k_eff = stiffness_factor(r_new, p["beta"])
    ddth = (-(p["wr"] ** 2) * k_eff * th - (p["wr"] / p["Qth"]) * dth
            + p["asym"] + p["kick"])
    dth_new = dth + DT * ddth
    th_new = jnp.clip(th + DT * dth_new, -2.0, 2.0)

    failed = jnp.logical_or(r_new >= 0.999, jnp.abs(th_new) >= TH_MAX)
    dead_new = jnp.logical_or(dead, failed)

    # a destroyed device is frozen: irreversible
    r_out = jnp.where(dead_new, 1.0, r_new)
    return (r_out, jnp.where(dead_new, 0.0, dr_new),
            jnp.where(dead_new, jnp.sign(th_new) * TH_MAX, th_new),
            jnp.where(dead_new, 0.0, dth_new),
            v_new, dead_new)


def observe(state, r_target, key):
    """Capacitive readout: position and angle, both noisy. beta / Q / bandwidth
    are NOT observable -- they are unknown fabrication parameters."""
    r, dr, th, dth, v, dead = state
    k1, k2 = jax.random.split(key)
    r_meas = r + 0.002 * jax.random.normal(k1)
    th_meas = th + 0.01 * jax.random.normal(k2)
    return jnp.array([r_meas, dr, th_meas, dth, v, r_target, r_target - r_meas])


def rollout(policy_fn, p, r_target, key, T=T_STEPS):
    """Run one device. Returns (mean tracking error over the last third,
    destroyed?, max r reached)."""
    def body(carry, k):
        state, = carry
        obs = observe(state, r_target, k)
        v_cmd = policy_fn(obs, p)
        kick = TH_NOISE * jax.random.normal(jax.random.fold_in(k, 7))
        pk = dict(p); pk["kick"] = kick
        new_state = step(state, v_cmd, pk)
        return (new_state,), (new_state[0], new_state[5])

    s0 = (0.0, 0.0, 0.0, 0.0, 0.0, False)
    keys = jax.random.split(key, T)
    (_,), (rs, deads) = jax.lax.scan(body, (s0,), keys)
    tail = T // 3
    alive = jnp.logical_not(deads[-1])
    err = jnp.mean(jnp.abs(rs[-tail:] - r_target))
    return err, deads[-1], jnp.max(rs), alive


# --------------------------- classical baseline -----------------------------

def classical_policy(obs, p):
    """Bandwidth-limited PD position feedback -- the classical stabilisation
    scheme. Its achievable travel is what collapses at high Q, because the
    loop must react faster than the actuator can (their eq 41)."""
    r_meas, dr, th_meas, dth, v, r_t, err = obs
    # Equilibrium of r'' + r'/Q + r = LAM_SCALE v^2/(1-r)^2 is
    #     LAM_SCALE v^2 = r_t (1 - r_t)^2,
    # so the feed-forward is sqrt(r_t (1-r_t)^2 / LAM_SCALE). (An earlier
    # version carried a spurious extra (1-r_t)^2, which made the controller
    # undershoot badly -- 0.24 error on a 0.60 target -- and masked tip-in
    # because the target was never actually reached.)
    v_ff = jnp.sqrt(jnp.clip(r_t * (1.0 - r_t) ** 2 / LAM_SCALE, 0.0, 1.0))
    u = v_ff + p["kp"] * err - p["kd"] * dr
    return jnp.clip(u, 0.0, 1.0)


def make_params(Q, beta, wu, wr=1.47, Qth=5.0, kp=2.0, kd=0.5, asym=3e-3):
    return dict(Q=Q, beta=beta, wu=wu, wr=wr, Qth=Qth, kp=kp, kd=kd,
                asym=asym, kick=0.0)


batch_rollout = jit(vmap(lambda p, rt, k: rollout(classical_policy, p, rt, k),
                         in_axes=({"Q": 0, "beta": 0, "wu": 0, "wr": None,
                                   "Qth": None, "kp": None, "kd": None,
                                   "kick": None}, 0, 0)))
