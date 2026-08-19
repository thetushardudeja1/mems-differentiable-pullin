"""
MEMS auto-calibration environment under fabrication variance.
Differentiable, GPU-batchable, JAX.

PHYSICS (overdamped first-order reduction, no inertia -> non-stiff):

    dX/dT + X = lambda / (1 - X)^2

This is exactly the reduction used in the literature for the overdamped regime:
  - Flores (arXiv:1603.02060), gamma = 0 case
  - Gomez, Moulton & Vella (arXiv:1710.10485), Q << 1
Both prove the dynamic pull-in threshold coincides with the static one here.

Equilibrium / validation identities (derived, exact):
    lambda(X) = X (1 - X)^2          <- steady state
    d lambda/dX = (1-X)(1-3X) = 0    ->  X_fold = 1/3,  lambda_fold = 4/27

FABRICATION VARIANCE:
Physically lambda = eps0 * A * V^2 / (2 k d0^3). Unit-to-unit variation in
spring constant k, electrode area A, and gap d0 therefore appears as an unknown
MULTIPLICATIVE GAIN g on the commanded V^2. (The gap enters cubically -- M-TEST
notes its S,B parameters scale with g0^3, which is why they do flatness
metrology. So gain variance is the physically dominant variance.)

    lambda_applied = g * LAMBDA_MAX * u^2,      u in [0, 1] is the command

Because g is unknown and unobservable, the steady-state command needed to hold
a target X_t,
    u* = sqrt( X_t (1 - X_t)^2 / (g * LAMBDA_MAX) ),
depends on g -> NO fixed open-loop voltage works for all devices, and
closed-loop feedback is genuinely required. Pull-in is a live risk: the safe
command ceiling sqrt(lambda_fold / (g*LAMBDA_MAX)) is 0.59 for g=1.4 vs 0.91
for g=0.6.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit

LAMBDA_FOLD = 4.0 / 27.0
X_FOLD = 1.0 / 3.0

LAMBDA_MAX = 0.30        # command scale: u=1 can over-drive past pull-in
G_LO, G_HI = 0.6, 1.4    # fabrication gain spread
X_TGT_LO, X_TGT_HI = 0.05, 0.25   # all targets < X_FOLD, so all are stable
TOUCHDOWN = 0.9          # collapsed (latching: stiction, does not recover)

# Horizon is set by CRITICAL SLOWING DOWN near the saddle-node, not by taste.
# Linearizing about equilibrium X_e (with lambda = X_e(1-X_e)^2) gives
#     d/dX[lambda/(1-X)^2 - X] = 2*X_e/(1-X_e) - 1,
# so the relaxation time constant is tau = 1 / (1 - 2*X_e/(1-X_e)), which
# DIVERGES as X_e -> X_FOLD = 1/3. Concretely: tau=1.12 at X_e=0.05 but
# tau=7.0 at X_e=0.30 and tau->inf at 1/3. This is the same slowing-down that
# Gomez, Moulton & Vella analyse (their t_PI ~ eps^-1/2 scaling).
# Worst case here is X_e=0.25 -> tau=3.0, so T_final=20 gives ~6.7 tau.
DT = 0.02
T_STEPS = 1000           # T_final = 20.0


def lambda_for_target(X_t):
    """Steady-state lambda needed to hold X_t. Exact inverse of the equilibrium."""
    return X_t * (1.0 - X_t) ** 2


def oracle_command(X_t, g):
    """u* for a device whose gain g is KNOWN. Upper bound on achievable control."""
    u_sq = lambda_for_target(X_t) / (g * LAMBDA_MAX)
    return jnp.clip(jnp.sqrt(jnp.clip(u_sq, 0.0, jnp.inf)), 0.0, 1.0)


def pullin_command_ceiling(g):
    """Command above which this device collapses."""
    return jnp.clip(jnp.sqrt(LAMBDA_FOLD / (g * LAMBDA_MAX)), 0.0, 1.0)


def step(X, latched, u, g):
    """One explicit-Euler step of the first-order dynamics. Differentiable."""
    lam = g * LAMBDA_MAX * jnp.clip(u, 0.0, 1.0) ** 2
    Xc = jnp.clip(X, 0.0, TOUCHDOWN)          # bound the 1/(1-X)^2 singularity
    dX = lam / (1.0 - Xc) ** 2 - Xc
    X_new = jnp.clip(X + DT * dX, 0.0, 1.0)
    touched = jnp.logical_or(latched, X_new > TOUCHDOWN)
    X_out = jnp.where(touched, 1.0, X_new)     # latch collapsed devices
    return X_out, touched


# ----------------------------- controllers -----------------------------------

def rollout_fixed(u_const, g, X_t):
    """Open-loop: one fixed command for every device (the 'no calibration' case)."""
    def body(carry, _):
        X, latched, I = carry
        X_new, touched = step(X, latched, u_const, g)
        return (X_new, touched, I), (X_new, touched)

    (_, _, _), (Xs, touches) = jax.lax.scan(body, (0.0, False, 0.0), None, length=T_STEPS)
    return Xs, touches


def rollout_pi(kp, ki, g, X_t):
    """Classical closed-loop PI controller -- the honest control-theory baseline."""
    def body(carry, _):
        X, latched, I = carry
        e = X_t - X
        I_new = I + e * DT
        u = jnp.clip(kp * e + ki * I_new, 0.0, 1.0)
        X_new, touched = step(X, latched, u, g)
        return (X_new, touched, I_new), (X_new, touched)

    (_, _, _), (Xs, touches) = jax.lax.scan(body, (0.0, False, 0.0), None, length=T_STEPS)
    return Xs, touches


def rollout_oracle(g, X_t):
    """Knows g. Upper bound."""
    return rollout_fixed(oracle_command(X_t, g), g, X_t)


# ----------------------------- learned policy --------------------------------

def init_policy(key, obs_dim=4, hidden=64):
    k1, k2, k3 = jax.random.split(key, 3)
    W1 = jax.random.normal(k1, (obs_dim, hidden)) * jnp.sqrt(2.0 / obs_dim)
    b1 = jnp.zeros(hidden)
    W2 = jax.random.normal(k2, (hidden, hidden)) * jnp.sqrt(2.0 / hidden)
    b2 = jnp.zeros(hidden)
    W3 = jax.random.normal(k3, (hidden, 1)) * 0.01
    b3 = jnp.zeros(1)
    return (W1, b1, W2, b2, W3, b3)


def policy_command(params, obs):
    W1, b1, W2, b2, W3, b3 = params
    h = jnp.tanh(obs @ W1 + b1)
    h = jnp.tanh(h @ W2 + b2)
    # sigmoid -> u in (0,1) natively: no clipping, so no clipping-induced
    # gradient bias, and the action bound is respected by construction.
    return jax.nn.sigmoid(h @ W3 + b3)[..., 0]


def rollout_policy(params, g, X_t):
    """Deterministic closed-loop rollout. Fully differentiable end-to-end:
    gradients flow through the physics, which is the point of the project."""
    def body(carry, _):
        X, latched, I = carry
        e = X_t - X
        I_new = I + e * DT
        obs = jnp.array([X, X_t, e, I_new])
        u = policy_command(params, obs)
        X_new, touched = step(X, latched, u, g)
        return (X_new, touched, I_new), (X_new, touched)

    (_, _, _), (Xs, touches) = jax.lax.scan(body, (0.0, False, 0.0), None, length=T_STEPS)
    return Xs, touches


# ----------------------------- metrics ---------------------------------------

TAIL = T_STEPS // 3   # judge steady state on the final third


def metrics(Xs, touches, X_t):
    """Physically interpretable: steady-state calibration error + collapse flag."""
    sse = jnp.mean(jnp.abs(Xs[-TAIL:] - X_t))
    pulled_in = touches[-1]
    return sse, pulled_in


def tracking_loss(Xs, X_t):
    """Differentiable objective: steady-state tracking + explicit pull-in barrier."""
    track = jnp.mean((Xs[-TAIL:] - X_t) ** 2)
    barrier = jnp.mean(jax.nn.relu(Xs - X_FOLD) ** 2)  # discourage crossing X_fold
    return track + 10.0 * barrier


batch_policy = jit(vmap(rollout_policy, in_axes=(None, 0, 0)))
batch_oracle = jit(vmap(rollout_oracle, in_axes=(0, 0)))
batch_fixed = jit(vmap(rollout_fixed, in_axes=(None, 0, 0)))
batch_pi = jit(vmap(rollout_pi, in_axes=(None, None, 0, 0)))
batch_metrics = jit(vmap(metrics, in_axes=(0, 0, 0)))
