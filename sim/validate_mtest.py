"""
Predict a MEASURED pull-in voltage from first principles.

Every validation so far has been against theory or another simulation. This one
targets a real fabricated device: Osterberg & Senturia (M-TEST, JMEMS 1997)
report, in their Fig. 7, a 500 um x 50 um fixed-fixed beam from wafer MTEST-03
pulling in at

    V_PI = 11.1 +/- 0.1 V   (measured, HP4145B, 100 mV voltage steps)

with independently measured wafer parameters (their Section V and Table VII):
    t0 = 2.94 +/- 0.07 um     thickness    (Dektak profilometry)
    g0 = 1.05 +/- 0.01 um     gap          (ellipsometry)
    E~ = 168 +/- 6 GPa        plate modulus, [011]  (or 138 +/- 4 for [010])
    s~ = 10 +/- 1 MPa         residual stress
    w  = 50 um                beam width

NONDIMENSIONALISATION (bending about the thickness t, toward the ground plane)
    E~ I g'''' - T_b g'' = -eps0 V^2 w / (2 g^2) * (1 + 0.65 g/w),  I = w t^3/12

  with x = l xi, g = d0 G this gives the solver's form
    Y'''' - N_t Y'' = Lambda (1 + alpha G)/G^2
    Lambda = 6 eps0 V^2 l^4 / (E~ t^3 d0^3)
    N_t    = T_b l^2/(E~ I) = 12 s~ l^2 / (E~ t^2)
    alpha  = 0.65 d0 / w                      (their fringing correction)

  so  V_PI = sqrt( Lambda_PI * E~ t^3 d0^3 / (6 eps0 l^4) ).

Residual stress matters enormously here: N_t ~ 20, which is NOT a perturbation.
Predicting 11.1 V therefore tests the tension term, the fringing correction and
the fold solver together, against a number nobody tuned.
"""

import itertools
import jax.numpy as jnp
import beam as B

EPS0 = 8.8541878128e-12
UM = 1e-6

# --- measured device + wafer parameters (M-TEST Fig. 7, Sec. V, Table VII) ---
L = 500 * UM
W = 50 * UM
T0, T0_ERR = 2.94 * UM, 0.07 * UM
G0, G0_ERR = 1.05 * UM, 0.01 * UM
E_011, E_011_ERR = 168e9, 6e9
E_010, E_010_ERR = 138e9, 4e9
SIG, SIG_ERR = 10e6, 1e6
V_MEAS, V_MEAS_ERR = 11.1, 0.1

N_GRID = 80


def predict(E, t, g0, sig, l=L, w=W, N=N_GRID):
    """Pull-in voltage in volts for a uniform-gap fixed-fixed beam."""
    N_t = 12.0 * sig * l ** 2 / (E * t ** 2)
    alpha = 0.65 * g0 / w
    M = B._unknown_count(N, B.FIXED_FIXED)
    D = jnp.ones(M)
    lam, travel, res, _ = B.pullin_fold(D, alpha=alpha, N_t=N_t, N=N,
                                        bc=B.FIXED_FIXED, n_coarse=25)
    v_scale = (E * t ** 3 * g0 ** 3 / (6.0 * EPS0 * l ** 4)) ** 0.5
    return (float(v_scale * jnp.sqrt(lam)), float(lam), float(N_t),
            float(alpha), float(res), float(travel))


if __name__ == "__main__":
    print(__doc__.split("NONDIMENSIONAL")[0].strip())
    print("\n" + "=" * 74)

    for name, E in [("[011], E~=168 GPa", E_011), ("[010], E~=138 GPa", E_010)]:
        v, lam, N_t, alpha, res, travel = predict(E, T0, G0, SIG)
        err = 100 * (v - V_MEAS) / V_MEAS
        print(f"\n{name}")
        print(f"  N_t   = {N_t:8.3f}   (residual-stress tension, dimensionless)")
        print(f"  alpha = {alpha:8.5f}   (fringing)")
        print(f"  Lambda_PI = {lam:8.3f}   travel at fold = {travel:.4f}"
              f"   |res| = {res:.1e}")
        print(f"  PREDICTED V_PI = {v:6.3f} V")
        print(f"  MEASURED  V_PI = {V_MEAS:6.3f} +/- {V_MEAS_ERR} V"
              f"      error = {err:+.2f}%")

    # --- how much of the answer is the residual stress? ---
    v_nostress, lam_ns, _, _, _, _ = predict(E_011, T0, G0, 0.0)
    print(f"\nwith residual stress set to zero: V_PI = {v_nostress:.3f} V "
          f"({100 * (v_nostress - V_MEAS) / V_MEAS:+.1f}%)")
    print(f"  -> tension supplies "
          f"{100 * (1 - v_nostress / predict(E_011, T0, G0, SIG)[0]):.0f}% of the"
          f" predicted voltage; this is not a small correction.")

    # --- propagate the reported measurement uncertainties ---
    print(f"\n=== uncertainty propagation (E~ [011]) ===")
    lo = hi = None
    for dE, dt, dg, ds in itertools.product((-1, 1), repeat=4):
        v, *_ = predict(E_011 + dE * E_011_ERR, T0 + dt * T0_ERR,
                        G0 + dg * G0_ERR, SIG + ds * SIG_ERR)
        lo = v if lo is None else min(lo, v)
        hi = v if hi is None else max(hi, v)
    print(f"  parameter uncertainties give V_PI in [{lo:.3f}, {hi:.3f}] V")
    print(f"  measurement                          "
          f"{V_MEAS - V_MEAS_ERR:.3f} - {V_MEAS + V_MEAS_ERR:.3f} V")
    overlap = not (hi < V_MEAS - V_MEAS_ERR or lo > V_MEAS + V_MEAS_ERR)
    print(f"  intervals overlap: {'YES' if overlap else 'NO'}")
    if overlap:
        print("  -> prediction is consistent with the measurement within the"
              " reported wafer-parameter uncertainties.")
