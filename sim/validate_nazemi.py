"""
Reproduce Nazemi, Schembri, Elnemr & Emadi (2025),
"Development and analysis of microbridge resonators for reduced pull-in voltage
and preserved resonant frequency", J. Sens. Sens. Syst. 14, 219-225.
https://doi.org/10.5194/jsss-14-219-2025   (EUROSENSORS 2024 special issue)

THEIR IDEA
A clamped-clamped microbridge over a bottom electrode that spans only part of
the span. The design parameter is the electrode-to-bridge length ratio
r = w/l. Increasing r lowers the pull-in voltage while leaving the resonant
frequency nearly unchanged.

THEIR DEVICE (their Sec. 3-5)
    bridge      l = 120 um, b = 40 um, t = 1.5 um   (polysilicon, PolyMUMPs)
    cavity      g0 = 750 nm
    material    E = 160 GPa, nu = 0.22
    electrodes  w = 42 um (r = 35%) and w = 82 um (r = 68%), both 40 um wide

THEIR MEASUREMENTS (their Sec. 5)
    V_PI = 25 V at r = 35%,  V_PI = 21 V at r = 68%   -> 16% reduction
    (Fig. 8b additionally gives a swept curve, normalised by FEA at r = 90%.)

HOW WE MODEL THE PARTIAL ELECTRODE
No change to the solver. The load term is Lambda (1 + alpha G)/G^2, so pushing
the gap to a large value outside the electrode makes the force vanish there --
which is precisely the physics of an electrode that is not present. With
G_out = 1e3 the residual force is ~1e-6 of the electrode-region value.

WHY THE RATIO IS THE PRIMARY TARGET
Absolute V_PI depends on quantities the paper does not report: PolyMUMPs
residual stress, anchor compliance, and the 520 nm gold top layer (a composite
beam). The RATIO between electrode configurations divides most of that out,
and it is also how the authors themselves present Fig. 8b (normalised). So the
headline comparison is V_PI(68%)/V_PI(35%), with absolute values reported
alongside and their offset discussed honestly.
"""

import jax.numpy as jnp
import beam as B

EPS0 = 8.8541878128e-12
UM = 1e-6

L = 120 * UM
BW = 40 * UM
T = 1.5 * UM
G0 = 750e-9
E = 160e9
NU = 0.22
E_TILDE = E / (1 - NU ** 2)          # wide beam (b = 27 t >> 5 t) -> plate modulus
ALPHA = 0.65 * G0 / BW               # fringing, M-TEST convention
G_OUT = 1.0e3                        # "no electrode here"

N = 80
BC = B.FIXED_FIXED

V_SCALE = float(jnp.sqrt(E_TILDE * T ** 3 * G0 ** 3 / (6.0 * EPS0 * L ** 4)))

MEAS = {0.35: 25.0, 0.68: 21.0}      # measured pull-in voltages, volts


def gap_profile(ratio, N=N, bc=BC):
    """Unity gap under the centred electrode, effectively infinite elsewhere."""
    xi = B.node_xi(N, bc)
    lo, hi = 0.5 * (1.0 - ratio), 0.5 * (1.0 + ratio)
    inside = jnp.logical_and(xi >= lo, xi <= hi)
    return jnp.where(inside, 1.0, G_OUT)


def pullin_volts(ratio, N_t=0.0, N=N, bc=BC):
    D = gap_profile(ratio, N, bc)
    lam, travel, res, _ = B.pullin_fold(D, alpha=ALPHA, N_t=N_t, N=N, bc=bc,
                                        n_coarse=30)
    return V_SCALE * float(jnp.sqrt(lam)), float(lam), float(res), float(travel)


if __name__ == "__main__":
    print(f"Nazemi et al. 2025 microbridge: l={L/UM:.0f} um, b={BW/UM:.0f} um, "
          f"t={T/UM:.1f} um, g0={G0*1e9:.0f} nm")
    print(f"E~ = E/(1-nu^2) = {E_TILDE/1e9:.1f} GPa,  alpha = {ALPHA:.5f}")
    print(f"V_PI = {V_SCALE:.4f} * sqrt(Lambda)")

    print(f"\n=== sanity: full electrode (r=1) must match the uniform-gap case ===")
    v_full, lam_full, res_full, _ = pullin_volts(1.0)
    lam_ref = B.LAMBDA_PI_REF[BC]
    print(f"  r=1.00: Lambda={lam_full:.3f}  (uniform-gap reference "
          f"{lam_ref:.3f}, {100*abs(lam_full-lam_ref)/lam_ref:.2f}%)  "
          f"|res|={res_full:.1e}")

    print(f"\n=== pull-in voltage vs electrode length ratio ===")
    print(f"  {'r':>6}{'Lambda_PI':>12}{'V_PI (V)':>11}{'V/V(r=1)':>11}"
          f"{'|res|':>10}   measured")
    rows = {}
    for r in [0.10, 0.20, 0.35, 0.50, 0.68, 0.80, 0.90, 1.00]:
        v, lam, res, travel = pullin_volts(r)
        rows[r] = v
        meas = f"   {MEAS[r]:.0f} V" if r in MEAS else ""
        print(f"  {r:>6.2f}{lam:>12.3f}{v:>11.3f}{v/v_full:>11.3f}"
              f"{res:>10.1e}{meas}")

    print(f"\n=== the headline comparison: ratio between the two fabricated devices ===")
    ours = rows[0.68] / rows[0.35]
    theirs = MEAS[0.68] / MEAS[0.35]
    print(f"  V_PI(68%)/V_PI(35%)   ours = {ours:.4f}   measured = {theirs:.4f}")
    print(f"  reduction             ours = {100*(1-ours):.1f}%   "
          f"measured = {100*(1-theirs):.1f}%")
    print(f"  error on the reduction: {abs(100*(1-ours) - 100*(1-theirs)):.1f} "
          f"percentage points")

    print(f"\n=== absolute values (expected to be offset; see module docstring) ===")
    for r, m in MEAS.items():
        print(f"  r={r:.2f}: ours {rows[r]:6.2f} V vs measured {m:.0f} V "
              f"({100*(rows[r]-m)/m:+.0f}%)")

    print(f"\n=== does compressive residual stress explain the offset? ===")
    print(f"  PolyMUMPs polysilicon is typically slightly compressive, which")
    print(f"  softens a clamped-clamped bridge and lowers V_PI.")
    print(f"  {'sigma (MPa)':>12}{'N_t':>9}{'V(35%)':>10}{'V(68%)':>10}"
          f"{'ratio':>9}")
    for sig_mpa in [0.0, -5.0, -10.0, -15.0, -20.0]:
        N_t = 12.0 * (sig_mpa * 1e6) * L ** 2 / (E_TILDE * T ** 2)
        v35, _, r35, _ = pullin_volts(0.35, N_t=N_t)
        v68, _, r68, _ = pullin_volts(0.68, N_t=N_t)
        ok = "" if max(r35, r68) < 1e-6 else "  (unconverged)"
        print(f"  {sig_mpa:>12.0f}{N_t:>9.2f}{v35:>10.2f}{v68:>10.2f}"
              f"{v68/v35:>9.4f}{ok}")
    print(f"  measured:                      {MEAS[0.35]:>10.0f}"
          f"{MEAS[0.68]:>10.0f}{theirs:>9.4f}")
