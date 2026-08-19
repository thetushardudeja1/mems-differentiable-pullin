"""
2-DOF electrostatic parallel-plate actuator: translation + rotation (tip-in),
with voltage control, charge control, and the amplifier-bandwidth stability
limit -- i.e. the classical baselines we must beat (or honestly lose to).

All equations from:
  Seeger & Boser, "Charge Control of Parallel-Plate, Electrostatic Actuators
  and the Tip-In Instability", J. Microelectromech. Syst. 12(5), 2003.

DYNAMICS (their 3a, 3b)
    m x'' + b_x x' + k_x x = (V^2/2) dC/dx + F
    J theta'' + b_th theta' + k_th theta = (V^2/2) dC/dtheta + Gamma

ELECTROSTATIC SPRING CONSTANTS (their 21, 24, 27), with r = x/g0:
    voltage-controlled translation : k_v   = -k_x * (2r) / (1 - r)
    charge-controlled  translation : k_q   = -k_x * (2r) / (1 + C0/Cp - r)
    rotation                       : k_the = -(k_x Lc^2 / 6) * r / (1 - r)

STABILITY LIMITS (their 22, 25, 28)
    voltage pull-in : r = 1/3                       (k_x + k_v <= 0)
    charge pull-in  : r = (1/3)(1 + C0/Cp)          (k_x + k_q <= 0)
    tip-in          : r = 1 / (1 + k_x Lc^2/(6 k_th))   (k_th + k_the <= 0)
  and the achievable travel is the MINIMUM of the applicable limits.

BANDWIDTH LIMIT (their 40/41) -- the opening for a learned controller:
    |A F| ~ (w_u / (2 w_n Q)) * C0 [ (1-r) Cp/C0 + 1 ]^2
            / [ (Cs + Ctop(r) + Cin) * r ]  > 1
  with Ctop(r) = Ctop0/(1-r) + Ctopp   (their 42).
  The required amplifier bandwidth scales as Q*w_n, so the stable range
  COLLAPSES at high Q (vacuum). Their words: "In a vacuum, the amplifier
  bandwidth requirement might be prohibitively large."
"""

import numpy as np

PF = 1e-12
UM = 1e-6


# ----------------------------- static limits --------------------------------

def r_voltage_pullin():
    """Voltage-controlled pull-in, as a fraction of the gap."""
    return 1.0 / 3.0


def r_charge_pullin(C0, Cp):
    """Charge-controlled pull-in (their eq 28). Returns >1 (i.e. no charge
    pull-in before contact) when Cp < C0/2, exactly as they state."""
    if Cp <= 0:
        return np.inf
    return (1.0 / 3.0) * (1.0 + C0 / Cp)


def r_tipin(k_x, L_c, k_theta):
    """Rotation-mode (tip-in) limit, their eq 22."""
    beta = k_x * L_c ** 2 / (6.0 * k_theta)
    return 1.0 / (1.0 + beta)


def max_travel(design, control="charge"):
    """Achievable fraction of the gap under a given control scheme."""
    r_ti = r_tipin(design["k_x"], design["L_c"], design["k_theta"])
    if control == "voltage":
        return min(r_ti, r_voltage_pullin()), r_ti
    return min(r_ti, r_charge_pullin(design["C0"], design["Cp"])), r_ti


# --------------------- bandwidth-limited stability (Fig 20) -----------------

def loop_gain(r, Q, d):
    """Their eq 40. Stable while this exceeds 1."""
    C_top = d["Ctop0"] / (1.0 - r) + d["Ctopp"]
    denom = (d["Cs"] + C_top + d["Cin"]) * r
    num = d["C0"] * ((1.0 - r) * d["Cp"] / d["C0"] + 1.0) ** 2
    return (d["w_u"] / (2.0 * d["w_n"] * Q)) * num / denom


def max_stable_r_bandwidth(Q, d, n=4000):
    """Largest deflection that remains stable given finite amplifier bandwidth,
    also capped by tip-in. This is the curve in their Fig. 20."""
    r_ti = r_tipin(d["k_x"], d["L_c"], d["k_theta"])
    rs = np.linspace(1e-4, min(0.999, r_ti), n)
    ok = np.array([loop_gain(r, Q, d) > 1.0 for r in rs])
    return rs[ok].max() if ok.any() else 0.0


# ------------------------- published designs (Tables I-III) ------------------
# Table I (structure), Table II (electronics), Table III (measured/fitted).

DESIGNS = {
    "#1": dict(x_ti_pub=0.19, g0=1.45 * UM, k_x=32.0, k_theta=0.45e-6,
               L_c=600 * UM, L_s=205 * UM, w_n=2 * np.pi * 12.6e3,
               C0=2.2 * PF, Cp=0.0, Cs=8.5 * PF, Ctop0=0.0, Ctopp=0.2 * PF,
               Cin=0.1 * PF, w_u=2 * np.pi * 5e6, x_max_pub=0.20,
               instability_pub="tip-in"),
    "#3": dict(x_ti_pub=0.85, g0=1.45 * UM, k_x=8.4, k_theta=0.58e-6,
               L_c=269 * UM, L_s=320 * UM, w_n=2 * np.pi * 6.0e3,
               C0=0.44 * PF, Cp=0.097 * 0.44 * PF, Cs=2.0 * PF,
               Ctop0=2.1 * PF, Ctopp=0.2 * PF, Cin=0.1 * PF,
               w_u=2 * np.pi * 10e6, x_max_pub=0.83, instability_pub="tip-in"),
    "#4": dict(x_ti_pub=0.85, g0=1.45 * UM, k_x=8.4, k_theta=0.58e-6,
               L_c=269 * UM, L_s=320 * UM, w_n=2 * np.pi * 6.0e3,
               C0=0.44 * PF, Cp=0.97 * 0.44 * PF, Cs=2.0 * PF,
               Ctop0=2.1 * PF, Ctopp=0.2 * PF, Cin=0.1 * PF,
               w_u=2 * np.pi * 10e6, x_max_pub=0.64, instability_pub="q pull-in"),
}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


if __name__ == "__main__":
    all_ok = True
    print("=== 1. Tip-in formula vs their Table I ===")
    for k, d in DESIGNS.items():
        r_ti = r_tipin(d["k_x"], d["L_c"], d["k_theta"])
        err = abs(r_ti - d["x_ti_pub"])
        all_ok &= check(f"design {k}: x_ti/g0 = {r_ti * 100:.1f}%  "
                        f"(published {d['x_ti_pub'] * 100:.0f}%)", err < 0.01)

    print("\n=== 2. Voltage vs charge control ceiling ===")
    all_ok &= check(f"voltage pull-in at {r_voltage_pullin() * 100:.1f}% of gap",
                    abs(r_voltage_pullin() - 1 / 3) < 1e-9)
    # their stated condition: no charge pull-in when Cp < C0/2
    all_ok &= check("Cp = C0/4 -> no charge pull-in before contact",
                    r_charge_pullin(1.0, 0.25) > 1.0,
                    f"r_qpi = {r_charge_pullin(1.0, 0.25):.2f}")
    all_ok &= check("Cp = C0   -> charge pull-in at 2/3 of gap",
                    abs(r_charge_pullin(1.0, 1.0) - 2 / 3) < 1e-9,
                    f"r_qpi = {r_charge_pullin(1.0, 1.0):.3f}")

    print("\n=== 3. Which instability limits each design? ===")
    for k, d in DESIGNS.items():
        r_q = r_charge_pullin(d["C0"], d["Cp"])
        r_ti = r_tipin(d["k_x"], d["L_c"], d["k_theta"])
        lim = "tip-in" if r_ti < r_q else "q pull-in"
        r_max = min(r_ti, r_q)
        print(f"  design {k}: tip-in {r_ti * 100:5.1f}%  q-pull-in "
              f"{min(r_q, 9.99) * 100:5.1f}%  -> limited by {lim:10s} at "
              f"{r_max * 100:5.1f}%   (published: {d['instability_pub']}, "
              f"{d['x_max_pub'] * 100:.0f}%)")
        all_ok &= check(f"  design {k} instability mechanism",
                        lim == d["instability_pub"])

    print("\n=== 4. Bandwidth collapse at high Q (their Fig. 20, design #3) ===")
    d3 = DESIGNS["#3"]
    print(f"  {'Q':>8}{'max stable x/g0':>18}")
    prev = 1.1
    monotone = True
    for Q in [1, 3, 10, 30, 100, 300, 1000, 3000]:
        r = max_stable_r_bandwidth(Q, d3)
        print(f"  {Q:>8}{r * 100:>17.1f}%")
        monotone &= (r <= prev + 1e-9)
        prev = r
    all_ok &= check("max stable deflection decreases monotonically with Q", monotone)
    all_ok &= check("collapses at high Q (Q=3000 well below tip-in ceiling)",
                    max_stable_r_bandwidth(3000, d3) < 0.5 * d3["x_ti_pub"])
    all_ok &= check("full travel available at low Q",
                    max_stable_r_bandwidth(1, d3) > 0.8 * d3["x_ti_pub"])

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
