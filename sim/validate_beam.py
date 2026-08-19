"""
Stage-1 validation of the distributed beam solver against PUBLISHED numbers.

Reference chain:
  Osterberg & Senturia 1997 (M-TEST) closed forms, as quoted by
  Haluzan et al., Micromachines 2010, 1, 68-81, Eqs. (1)-(2) and Table 3:
      V_pi = 0.529 sqrt(E~ w^3 d^3 / (eps0 l^4 (1 + 0.42 d/h)))   cantilever
      V_pi = 3.444 sqrt(E~ w^3 d^3 / (eps0 l^4 (1 + 0.42 d/h)))   fixed-fixed
  Their Table 1 sample problem: E=169 GPa, nu=0.32, h=100um, l=1000um,
  w=10um, d_min=1um.
  Their Table 3 (min constant gap): cantilever 2.44 V (ANSYS) / 2.43 V
  (Osterberg closed form); fixed-fixed 15.77 V (ANSYS) / 15.85 V (closed form).
"""

import jax
import jax.numpy as jnp

import beam as B

E = 169e9
NU = 0.32
E_TILDE = E / (1 - NU ** 2)      # tall beam -> plate modulus
H = 100e-6
L = 1000e-6
W = 10e-6
D0 = 1e-6
ALPHA = 0.42 * D0 / H            # = 0.0042, negligible but included


def closed_form_V(coeff):
    eps0 = 8.8541878128e-12
    return coeff * jnp.sqrt(E_TILDE * W ** 3 * D0 ** 3
                            / (eps0 * L ** 4 * (1 + 0.42 * D0 / H)))


if __name__ == "__main__":
    print("x64 enabled:", jax.config.jax_enable_x64)

    print("\n--- 0. Dimensional closed form reproduces Haluzan Table 3 ---")
    v_c = float(closed_form_V(0.529))
    v_f = float(closed_form_V(3.444))
    print(f"  cantilever  closed form = {v_c:6.3f} V   (paper: 2.43 V closed form / 2.44 ANSYS)")
    print(f"  fixed-fixed closed form = {v_f:6.3f} V   (paper: 15.85 V closed form / 15.77 ANSYS)")

    print("\n--- 1. Dimensionless pull-in from the FD solver (uniform gap) ---")
    for bc, npts in [(B.CANTILEVER, 60), (B.FIXED_FIXED, 60)]:
        M = B._unknown_count(npts, bc)
        D = jnp.ones(M)
        lam_pi, travel, deltas, lams = B.pullin_lambda(D, alpha=0.0, N_t=0.0, N=npts, bc=bc)
        ref = B.LAMBDA_PI_REF[bc]
        err = 100 * abs(float(lam_pi) - ref) / ref
        k = int(jnp.argmax(lams))
        print(f"  {bc:12s} Lambda_PI = {float(lam_pi):8.4f}   ref = {ref:8.4f}   "
              f"err = {err:5.2f}%   (fold at delta = {float(deltas[k]):.3f})")

    print("\n--- 2. Round-trip to volts, uniform gap, Haluzan geometry ---")
    for bc, coeff in [(B.CANTILEVER, 0.529), (B.FIXED_FIXED, 3.444)]:
        M = B._unknown_count(60, bc)
        D = jnp.ones(M)
        lam_pi, _, _, _ = B.pullin_lambda(D, alpha=ALPHA, N_t=0.0, N=60, bc=bc)
        V_sim = float(B.lambda_to_voltage(lam_pi, E_TILDE, W, D0, L))
        V_ref = float(closed_form_V(coeff))
        print(f"  {bc:12s} V_PI(sim) = {V_sim:7.3f} V   V_PI(closed form) = {V_ref:7.3f} V   "
              f"err = {100 * abs(V_sim - V_ref) / V_ref:5.2f}%")

    print("\n--- 3. Grid convergence (cantilever) ---")
    for npts in [30, 45, 60, 90]:
        D = jnp.ones(B._unknown_count(npts, B.CANTILEVER))
        lam_pi, _, _, _ = B.pullin_lambda(D, alpha=0.0, N_t=0.0, N=npts, bc=B.CANTILEVER)
        print(f"  N={npts:3d}  Lambda_PI = {float(lam_pi):8.5f}")
