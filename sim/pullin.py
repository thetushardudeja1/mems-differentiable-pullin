"""
Differentiable electrostatic MEMS pull-in simulator (JAX, GPU-batchable).

Governing equation (nondimensional lumped mass-spring-damper model), matching:
  - Flores, "On the dynamic pull-in instability in a mass-spring model of
    electrostatically actuated MEMS devices" (arXiv:1603.02060)
  - Gomez, Moulton & Vella, "Delayed pull-in transitions in overdamped MEMS
    devices" (arXiv:1710.10485)

    Q^2 * d2X/dT2 + dX/dT + X = lambda / (1 - X)^2

X(T) in [0, 1) is normalized gap-closure displacement (X=1 is touchdown),
lambda ~ V^2 is the normalized actuation voltage, Q = sqrt(m k) / b is the
quality factor (Q << 1 = overdamped, Q >> 1 = underdamped).

Known closed-form validation targets (exact, not fitted):
  - Static pull-in:  lambda_fold = 4/27,  X_fold = 1/3
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial

LAMBDA_FOLD = 4.0 / 27.0
X_FOLD = 1.0 / 3.0


def rhs(state, lam, Q):
    """State = [X, V] where V = dX/dT. Returns [dX/dT, dV/dT]."""
    X, V = state
    Xc = jnp.minimum(X, 0.999)  # avoid singularity at touchdown
    dXdT = V
    dVdT = (lam / (1.0 - Xc) ** 2 - Xc - V) / (Q ** 2)
    return jnp.array([dXdT, dVdT])


def rk4_step(state, lam, Q, dt):
    k1 = rhs(state, lam, Q)
    k2 = rhs(state + 0.5 * dt * k1, lam, Q)
    k3 = rhs(state + 0.5 * dt * k2, lam, Q)
    k4 = rhs(state + dt * k3, lam, Q)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@partial(jit, static_argnames=("n_steps",))
def simulate(lam, Q, dt, n_steps):
    """Integrate from rest (X=0, V=0). Returns trajectory of X, shape (n_steps+1,)."""
    state0 = jnp.array([0.0, 0.0])

    def body(state, _):
        new_state = rk4_step(state, lam, Q, dt)
        new_state = new_state.at[0].set(jnp.minimum(new_state[0], 1.0))
        return new_state, new_state[0]

    _, xs = jax.lax.scan(body, state0, None, length=n_steps)
    return jnp.concatenate([jnp.array([0.0]), xs])


# Batched across many (lambda, Q) pairs at once -- this is the GPU-parallel win.
simulate_batch = jit(vmap(simulate, in_axes=(0, 0, None, None)), static_argnames=("n_steps",))


def find_static_pullin_numerically(Q=0.05, dt=5e-3, n_steps=400_000, lo=0.0, hi=4.0 / 27.0 * 1.5, tol=1e-6):
    """Bisection on lambda: does the trajectory touch down (X -> 1) or settle?
    Uses a small Q (overdamped), matching the regime analyzed in Gomez, Moulton
    & Vella -- in this regime the dynamic pull-in threshold converges to the
    static threshold lambda* = 4/27, and the relaxation timescale is governed
    by 1/Q (not 1/Q^2 as in the underdamped case), so it settles within a
    tractable simulated time instead of requiring a very fine dt to resolve
    fast inertial oscillations.
    """

    def touches_down(lam):
        xs = simulate(lam, Q, dt, n_steps)
        return jnp.max(xs) > 0.999

    lo, hi = jnp.array(lo), jnp.array(hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        pulled_in = touches_down(mid)
        hi = jnp.where(pulled_in, mid, hi)
        lo = jnp.where(pulled_in, lo, mid)
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("JAX devices:", jax.devices())

    # --- Validation 1: static pull-in threshold lambda* = 4/27 ---
    lam_numeric = find_static_pullin_numerically()
    lam_exact = LAMBDA_FOLD
    err_pct = 100.0 * abs(float(lam_numeric) - lam_exact) / lam_exact
    print(f"\n[Validation 1] Static pull-in voltage (lambda*)")
    print(f"  Exact (Flores/Gomez, closed-form): {lam_exact:.6f}")
    print(f"  Simulated (JAX RK4, bisection):    {float(lam_numeric):.6f}")
    print(f"  Error: {err_pct:.4f}%")

    # --- Validation 2: GPU-batched sweep across many (lambda, Q) pairs at once ---
    n_devices = 2000
    lambdas = jnp.linspace(0.01, LAMBDA_FOLD * 0.99, n_devices)
    Qs = jnp.full(n_devices, 0.05)

    import time

    dt, n_steps = 5e-3, 100_000
    final_state_batch = jit(vmap(lambda lam, Q: simulate(lam, Q, dt, n_steps)[-1], in_axes=(0, 0)))
    # warmup (compile)
    _ = final_state_batch(lambdas, Qs).block_until_ready()
    t0 = time.perf_counter()
    final_X = final_state_batch(lambdas, Qs).block_until_ready()
    t1 = time.perf_counter()
    print(f"\n[Validation 2] GPU-batched simulation")
    print(f"  Simulated {n_devices} devices x {n_steps} steps each in {t1 - t0:.4f}s "
          f"({n_devices * n_steps / (t1 - t0):,.0f} steps/sec)")

    print(f"  Max final X across sweep (should stay < 1.0, no pull-in below threshold): "
          f"{float(jnp.max(final_X)):.4f}")
