"""
Pre-flight checks on the 2-DOF RL environment. No learning yet.

The environment is only worth training against if it reproduces physics we
already validated independently in model2dof.py:
  1. tip-in occurs at r = 1/(1+beta)  and destroys the device
  2. below that, the rotation mode is stable and the device survives
  3. the classical controller's reachable travel COLLAPSES as Q rises
     (their Fig. 20 / eq 41) -- this is the gap a learned policy must attack
  4. nothing blows up numerically in fp32
"""

import jax
import jax.numpy as jnp
from jax import random, vmap

import env2dof as E


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def sweep_reachable(Q, beta, wu, n_t=25, n_dev=64, key=None):
    """Largest target the classical controller can hold without destroying the
    device (averaged over a batch of noise realisations)."""
    targets = jnp.linspace(0.05, min(0.95, E.r_tipin(beta) * 1.15), n_t)
    best = 0.0
    for rt in targets:
        keys = random.split(key, n_dev)
        p = E.make_params(Q, beta, wu)
        errs, dead, rmax, alive = vmap(
            lambda k: E.rollout(E.classical_policy, p, rt, k))(keys)
        survived = float(jnp.mean(alive))
        tracked = float(jnp.mean(jnp.where(alive, errs, 1.0)))
        if survived > 0.9 and tracked < 0.05:
            best = float(rt)
    return best


if __name__ == "__main__":
    print("JAX devices:", jax.devices(), " x64:", jax.config.jax_enable_x64)
    key = random.PRNGKey(0)
    all_ok = True

    beta3 = 8.4 * (269e-6) ** 2 / (6 * 0.58e-6)     # design #3 -> r_ti = 0.851
    print(f"\ndesign #3: beta={beta3:.4f}  r_tipin={E.r_tipin(beta3):.3f}")

    print("\n--- 1. Rotational stiffness changes sign exactly at r_tipin ---")
    r_ti = E.r_tipin(beta3)
    k_below = float(E.stiffness_factor(r_ti - 0.05, beta3))
    k_above = float(E.stiffness_factor(r_ti + 0.05, beta3))
    all_ok &= check("stable below r_tipin", k_below > 0, f"k_eff={k_below:.4f}")
    all_ok &= check("unstable above r_tipin", k_above < 0, f"k_eff={k_above:.4f}")

    print("\n--- 2. Device survives below r_tipin, is destroyed above ---")
    key, k1, k2 = random.split(key, 3)
    p = E.make_params(Q=5.0, beta=beta3, wu=50.0)
    for rt, expect_alive in [(0.40, True), (0.60, True), (0.95, False)]:
        keys = random.split(k1, 64)
        errs, dead, rmax, alive = vmap(
            lambda k: E.rollout(E.classical_policy, p, rt, k))(keys)
        frac = float(jnp.mean(alive))
        all_ok &= check(f"target r={rt:.2f} -> survives {frac * 100:.0f}%"
                        f" (expect {'yes' if expect_alive else 'no'})",
                        (frac > 0.9) == expect_alive,
                        f"mean|err|={float(jnp.mean(jnp.where(alive, errs, jnp.nan))):.4f}"
                        if frac > 0 else "")

    print("\n--- 3. Where does actuator bandwidth actually bite HERE? ---")
    # NOTE: this is NOT a reproduction of Seeger & Boser's Fig. 20. Their
    # collapse arises inside the charge-control CIRCUIT (a capacitive-divider
    # loop needing w_u > 2 w_n Q (Cs+Ctop+Cin)/C0, a ratio of ~33 for design
    # #3, so w_u > 6600 w_n at Q=100). This environment models a different
    # architecture -- voltage drive with a first-order lag plus PD position
    # feedback -- in which the derivative term supplies damping regardless of
    # Q. So we characterise THIS architecture's own limit honestly rather than
    # claiming their curve.
    print(f"    {'wu':>8}{'reachable r':>14}   (mech. tip-in ceiling"
          f" {r_ti:.3f})")
    key, sk = random.split(key)
    rows = []
    for wu in [0.3, 1.0, 3.0, 10.0, 50.0]:
        r_reach = sweep_reachable(5.0, beta3, wu, key=sk)
        rows.append((wu, r_reach))
        print(f"    {wu:>8.1f}{r_reach:>14.3f}")
    all_ok &= check("low actuator bandwidth genuinely limits travel",
                    rows[0][1] < 0.75 * rows[-1][1],
                    f"wu=0.3 -> {rows[0][1]:.3f} vs wu=50 -> {rows[-1][1]:.3f}")
    all_ok &= check("ample bandwidth reaches the mechanical ceiling",
                    rows[-1][1] > 0.9 * r_ti,
                    f"{rows[-1][1]:.3f} vs {r_ti:.3f}")

    print("\n--- 3b. Cost of parameter uncertainty (the opening for RL) ---")
    # A fixed-gain controller must be tuned for the WORST-CASE device, because
    # beta (hence the tip-in ceiling) is not observable. Quantify what that
    # conservatism costs against per-device oracle tuning.
    key, sk_u = random.split(key)
    betas = [0.10, 0.1747, 0.35, 0.60]
    print(f"    {'beta':>8}{'r_tipin':>10}{'reachable':>12}{'headroom lost':>15}")
    per_device = []
    for b in betas:
        rr = sweep_reachable(5.0, b, 10.0, key=sk_u)
        per_device.append(rr)
        print(f"    {b:>8.3f}{E.r_tipin(b):>10.3f}{rr:>12.3f}"
              f"{E.r_tipin(b) - rr:>15.3f}")
    worst_case_target = min(per_device)
    oracle_mean = sum(per_device) / len(per_device)
    print(f"    a single fixed target safe for ALL devices: {worst_case_target:.3f}")
    print(f"    per-device (oracle) mean reachable:         {oracle_mean:.3f}")
    print(f"    -> conservatism costs {100 * (1 - worst_case_target / oracle_mean):.1f}%"
          f" of travel; that is what an adaptive policy could recover")

    print("\n--- 4. More amplifier bandwidth recovers travel (confirms the "
          "mechanism is bandwidth, not something else) ---")
    print(f"    {'wu':>8}{'reachable r at Q=100':>24}")
    prev_r = -1.0
    increasing = True
    for wu in [20.0, 50.0, 200.0, 800.0]:
        key, sk2 = random.split(key)
        r_reach = sweep_reachable(100.0, beta3, wu, key=sk2)
        print(f"    {wu:>8.0f}{r_reach:>24.3f}")
        increasing &= r_reach >= prev_r - 1e-9
        prev_r = r_reach
    all_ok &= check("reachable travel increases with amplifier bandwidth", increasing)

    print("\n--- 5. Numerical health ---")
    key, sk3 = random.split(key)
    keys = random.split(sk3, 256)
    p = E.make_params(Q=50.0, beta=beta3, wu=50.0)
    errs, dead, rmax, alive = vmap(
        lambda k: E.rollout(E.classical_policy, p, 0.5, k))(keys)
    all_ok &= check("no NaNs across 256 devices",
                    bool(jnp.all(jnp.isfinite(errs)) and jnp.all(jnp.isfinite(rmax))))
    all_ok &= check("r stays within [0,1]", bool(jnp.all(rmax <= 1.0 + 1e-6)))

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
