"""
What is the TRUE achievable ceiling for a beta-knowing policy?

The long run produced RL travel = 0.6418 against an "oracle" of 0.6355, i.e.
"106% of headroom recovered" -- which is impossible if the oracle were really an
upper bound. It is not: the oracle targets margin * r_ti(beta) with margin=0.97,
an arbitrary 3% safety cushion. Sweeping the margin gives the actual frontier
of what perfect knowledge of beta can buy, and therefore an honest denominator.

A margin that is too aggressive starts destroying devices, so the frontier is
the best travel achievable at ~0% destruction.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap

import env2dof as E
import train_rl2dof as T


if __name__ == "__main__":
    key = random.PRNGKey(12345)
    key, dk, ek = random.split(key, 3)
    devices = T.sample_devices(dk, 512)
    r_ti = E.r_tipin(devices["beta"])

    def run(policy):
        keys = random.split(ek, devices["beta"].shape[0])
        rets, tr, dead, _ = vmap(
            lambda b, q, a, k: T.rollout(None, b, q, a, k,
                                         stochastic=False, policy=policy)
        )(devices["beta"], devices["Q"], devices["asym"], keys)
        return float(jnp.mean(tr)), float(jnp.mean(dead))

    worst = float(jnp.min(r_ti))
    print(f"device ceilings r_ti in [{worst:.3f}, {float(jnp.max(r_ti)):.3f}]")

    print("\n=== fixed conservative target, swept ===")
    print(f"  {'target':>8}{'travel':>10}{'destroyed':>12}")
    best_safe_fixed = (0.0, 0.0)
    for frac in [0.85, 0.90, 0.95, 0.98, 1.00, 1.02]:
        t = frac * worst
        tr, d = run(T.fixed_target_policy(t))
        flag = ""
        if d <= 0.001 and tr > best_safe_fixed[0]:
            best_safe_fixed = (tr, t)
            flag = "  <- best safe"
        print(f"  {t:>8.3f}{tr:>10.4f}{d * 100:>11.1f}%{flag}")

    print("\n=== oracle (knows beta), margin swept ===")
    print(f"  {'margin':>8}{'travel':>10}{'destroyed':>12}")
    best_safe_oracle = (0.0, 0.0)
    for m in [0.90, 0.95, 0.97, 0.99, 1.00, 1.01, 1.03]:
        tr, d = run(T.oracle_policy(margin=m))
        flag = ""
        if d <= 0.001 and tr > best_safe_oracle[0]:
            best_safe_oracle = (tr, m)
            flag = "  <- best safe"
        print(f"  {m:>8.2f}{tr:>10.4f}{d * 100:>11.1f}%{flag}")

    print(f"\n=== honest reference points ===")
    print(f"  best SAFE fixed target      : travel={best_safe_fixed[0]:.4f} "
          f"(r={best_safe_fixed[1]:.3f})")
    print(f"  best SAFE oracle            : travel={best_safe_oracle[0]:.4f} "
          f"(margin={best_safe_oracle[1]:.2f})")
    gap = best_safe_oracle[0] - best_safe_fixed[0]
    print(f"  true headroom from knowing beta: {gap:.4f} "
          f"({100 * gap / best_safe_fixed[0]:.1f}% of the fixed baseline)")
    print(f"\n  Use these as the denominators. The previously reported "
          f"'oracle' used margin=0.97,")
    print(f"  which is a safety cushion, not a bound -- which is why a trained "
          f"policy could exceed it.")
