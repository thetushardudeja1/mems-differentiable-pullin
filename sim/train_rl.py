"""
RL auto-calibration policy for electrostatic MEMS devices under fabrication
variance, trained against the differentiable pull-in simulator (sim/pullin.py).

Task: hold a target displacement X_target despite each simulated device having
a randomized (unknown to the policy) quality factor Q -- standing in for
unit-to-unit fabrication variance in damping/spring/geometry. The policy only
observes (X, V, X_target); it must adapt its control purely through closed-loop
feedback, the same way a real controller would since Q can't be measured
directly on a fabricated device.

Trained with REINFORCE (policy gradient), batched across thousands of parallel
randomized-variance devices on GPU via vmap -- this is the "JAX launches
thousands of models" story applied to training, not just simulation.
"""

import time
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit

from pullin import rhs, LAMBDA_FOLD

LAMBDA_MAX = LAMBDA_FOLD * 0.98  # policy may command right up near instability
DT = 5e-3
T_STEPS = 600  # steps per episode


# ---------------- Policy: small MLP, obs -> action in [-1, 1] ----------------
def init_policy(key, obs_dim=3, hidden=32):
    k1, k2 = random.split(key)
    W1 = random.normal(k1, (obs_dim, hidden)) * (1.0 / jnp.sqrt(obs_dim))
    b1 = jnp.zeros(hidden)
    W2 = random.normal(k2, (hidden, 1)) * (1.0 / jnp.sqrt(hidden))
    b2 = jnp.zeros(1)
    return (W1, b1, W2, b2)


def policy_mean(params, obs):
    W1, b1, W2, b2 = params
    h = jnp.tanh(obs @ W1 + b1)
    out = jnp.tanh(h @ W2 + b2)[..., 0]
    return out  # in [-1, 1]


def action_to_lambda(a):
    return (a + 1.0) * 0.5 * LAMBDA_MAX


# ---------------- Single-episode rollout for one randomized device ----------------
def rollout(params, key, Q, X_target):
    def obs_fn(X, V):
        return jnp.array([X, V, X_target])

    def step(carry, key_t):
        X, V = carry
        obs = obs_fn(X, V)
        mean_a = policy_mean(params, obs)
        eps = random.normal(key_t)
        # Detach the sampled action before computing log-prob: REINFORCE needs
        # the score function d/dtheta[log pi(a|s)] with `a` treated as a fixed
        # observed sample, not differentiated through the sampling itself.
        a_sampled = jax.lax.stop_gradient(mean_a + SIGMA * eps)
        a = jnp.clip(a_sampled, -1.0, 1.0)
        log_prob = -0.5 * ((a_sampled - mean_a) / SIGMA) ** 2 - jnp.log(SIGMA) - 0.5 * jnp.log(2 * jnp.pi)

        lam = action_to_lambda(a)
        Xc = jnp.minimum(X, 0.999)
        dXdT = V
        dVdT = (lam / (1.0 - Xc) ** 2 - Xc - V) / (Q ** 2)
        X_new = jnp.clip(X + DT * dXdT, 0.0, 1.0)
        V_new = V + DT * dVdT

        touchdown_penalty = 50.0 * jax.nn.relu(X_new - 0.97)
        reward = -((X_new - X_target) ** 2) - touchdown_penalty

        return (X_new, V_new), (reward, log_prob)

    keys = random.split(key, T_STEPS)
    (_, _), (rewards, log_probs) = jax.lax.scan(step, (0.0, 0.0), keys)
    return jnp.sum(rewards), jnp.sum(log_probs)


rollout_batch = jit(vmap(rollout, in_axes=(None, 0, 0, 0)))


def reinforce_loss(params, keys, Qs, X_targets):
    returns, log_probs = rollout_batch(params, keys, Qs, X_targets)
    baseline = jnp.mean(returns)
    advantage = returns - baseline
    loss = -jnp.mean(advantage * log_probs)
    return loss, returns


grad_fn = jit(jax.value_and_grad(reinforce_loss, has_aux=True))


def fixed_lambda_baseline_return(Q, X_target, lam_const):
    def step(carry, _):
        X, V = carry
        Xc = jnp.minimum(X, 0.999)
        dXdT = V
        dVdT = (lam_const / (1.0 - Xc) ** 2 - Xc - V) / (Q ** 2)
        X_new = jnp.clip(X + DT * dXdT, 0.0, 1.0)
        V_new = V + DT * dVdT
        touchdown_penalty = 50.0 * jax.nn.relu(X_new - 0.97)
        reward = -((X_new - X_target) ** 2) - touchdown_penalty
        return (X_new, V_new), reward

    (_, _), rewards = jax.lax.scan(step, (0.0, 0.0), None, length=T_STEPS)
    return jnp.sum(rewards)


fixed_lambda_batch = jit(vmap(fixed_lambda_baseline_return, in_axes=(0, 0, None)))


if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    key = random.PRNGKey(0)
    key, pkey = random.split(key)
    params = init_policy(pkey)

    N_ENVS = 4096
    N_ITERS = 150
    LR = 0.05

    key, qkey, tkey = random.split(key, 3)
    Qs_fixed_eval = random.uniform(qkey, (N_ENVS,), minval=0.08, maxval=0.3)  # fabrication variance
    X_targets_fixed_eval = random.uniform(tkey, (N_ENVS,), minval=0.05, maxval=0.30)

    print(f"\nTraining RL policy: {N_ENVS} parallel randomized-variance devices "
          f"(Q in [0.02, 0.5]), {N_ITERS} iterations")

    t0 = time.perf_counter()
    for it in range(N_ITERS):
        key, qkey, tkey, rkey = random.split(key, 4)
        Qs = random.uniform(qkey, (N_ENVS,), minval=0.08, maxval=0.3)
        X_targets = random.uniform(tkey, (N_ENVS,), minval=0.05, maxval=0.30)
        keys = random.split(rkey, N_ENVS)

        (loss, returns), grads = grad_fn(params, keys, Qs, X_targets)
        params = jax.tree_util.tree_map(lambda p, g: p - LR * g, params, grads)

        if it % 15 == 0 or it == N_ITERS - 1:
            print(f"  iter {it:4d}  mean_return={float(jnp.mean(returns)):.4f}  "
                  f"loss={float(loss):.4f}")
    t1 = time.perf_counter()
    print(f"Training time: {t1 - t0:.2f}s "
          f"({N_ENVS * T_STEPS * N_ITERS / (t1 - t0):,.0f} env-steps/sec)")

    # ---------------- Compare trained policy vs fixed-voltage classical baseline ----------------
    key, rkey = random.split(key)
    eval_keys = random.split(rkey, N_ENVS)
    trained_returns, _ = rollout_batch(params, eval_keys, Qs_fixed_eval, X_targets_fixed_eval)

    best_fixed_return = None
    best_lam = None
    for lam_const in jnp.linspace(0.02, LAMBDA_MAX, 15):
        r = fixed_lambda_batch(Qs_fixed_eval, X_targets_fixed_eval, lam_const)
        m = float(jnp.mean(r))
        if best_fixed_return is None or m > best_fixed_return:
            best_fixed_return, best_lam = m, float(lam_const)

    print(f"\n[Comparison on held-out fabrication-variance distribution, N={N_ENVS}]")
    print(f"  RL policy (adapts per device):        mean return = {float(jnp.mean(trained_returns)):.4f}")
    print(f"  Best single fixed voltage (classical, "
          f"lambda={best_lam:.4f}): mean return = {best_fixed_return:.4f}")
    print(f"  Improvement: {100.0 * (float(jnp.mean(trained_returns)) - best_fixed_return) / abs(best_fixed_return):.1f}%")
