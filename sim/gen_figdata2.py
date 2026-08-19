"""
Data for the second batch of figure panels (Fig. 3 - Fig. 5).

  f5a  fold structure   : Lambda vs controlled deflection, turning point marked
  f5b  mode shapes      : deflection profile at pull-in, uniform vs shaped gap
  f6   Nazemi 2025      : V_PI vs bottom-electrode length ratio + measurements
  f7   validation bars  : relative error against every source we checked
  f8   amortized sweep  : network across the whole specification range

Usage: python gen_figdata2.py [fold] [nazemi] [valid] [amort]
"""

import sys
import pickle
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import beam as B


# ------------------------------ fold structure ------------------------------
def gen_fold():
    """Lambda(delta) along the displacement-controlled branch. The maximum IS
    the pull-in point: beyond it no equilibrium exists at higher voltage, which
    is the saddle-node our solver targets directly."""
    print("[fold] equilibrium branch + turning point")
    out = {}
    for bc, tag in [(B.CANTILEVER, "cant"), (B.FIXED_FIXED, "ff")]:
        M = B._unknown_count(60, bc)
        D = jnp.ones(M)
        lam_pi, travel, ss, lams = B.pullin_lambda(D, N=60, bc=bc, n_delta=70)
        lam_f, tr_f, res, _ = B.pullin_fold(D, N=60, bc=bc)
        out[f"{tag}_s"] = np.asarray(ss)
        out[f"{tag}_lam"] = np.asarray(lams)
        out[f"{tag}_fold"] = np.array([float(tr_f), float(lam_f)])
        print(f"   {tag}: fold at delta={float(tr_f):.4f}, "
              f"Lambda={float(lam_f):.4f}, |res|={float(res):.1e}")

        # deflection profile at the fold, uniform vs shaped gap
        z = B.fold_initial_guess(D, 0.0, 0.0, 60, bc, 25)
        zz = B._fold_newton(z, D, 0.0, 0.0, 60, bc)
        out[f"{tag}_Y_uniform"] = np.asarray(zz[:M])
        out[f"{tag}_xi"] = np.asarray(B.node_xi(60, bc))
    np.savez("figdata_fold.npz", **out)
    print("   -> figdata_fold.npz")


# --------------------------------- Nazemi -----------------------------------
def gen_nazemi():
    """Pull-in voltage vs bottom-electrode length ratio (Nazemi et al. 2025)."""
    import validate_nazemi as NZ
    print("[nazemi] electrode-ratio sweep")
    rs = [0.10, 0.20, 0.30, 0.35, 0.45, 0.55, 0.68, 0.80, 0.90, 1.00]
    vs, res = [], []
    for r in rs:
        v, lam, rr, tr = NZ.pullin_volts(r)
        vs.append(v)
        res.append(rr)
        print(f"   r={r:.2f}  V={v:8.3f}  |res|={rr:.0e}")
    np.savez("figdata_nazemi.npz",
             r=np.array(rs), v=np.array(vs), res=np.array(res),
             meas_r=np.array([0.35, 0.68]), meas_v=np.array([25.0, 21.0]))
    print("   -> figdata_nazemi.npz")


# ------------------------------- validation ---------------------------------
def gen_valid():
    """Relative error of every quantity we validated, for a summary bar chart.
    Values are transcribed from the validation scripts named beside each entry."""
    print("[valid] assembling validation summary")
    rows = [
        (r"$\lambda^*$ lumped",      0.001,  "test_fold / pullin"),
        (r"$\Lambda_{PI}$ cant.",    0.05,   "test_fold"),
        (r"$\Lambda_{PI}$ fix-fix",  1.73,   "test_fold"),
        (r"$\partial\Lambda/\partial D$", 0.00, "test_fold"),
        (r"tip-in \#1",              0.0,    "model2dof"),
        (r"tip-in \#3",              0.12,   "model2dof"),
        (r"Nazemi reduction",        2.8,    "validate_nazemi"),
        (r"M-TEST $V_{PI}$",         5.4,    "validate_mtest"),
    ]
    np.savez("figdata_valid.npz",
             labels=np.array([r[0] for r in rows]),
             err=np.array([r[1] for r in rows]),
             src=np.array([r[2] for r in rows]))
    print("   -> figdata_valid.npz")


# ------------------------------- amortized ----------------------------------
def gen_amort():
    """Network output across the whole specification range."""
    import amortized_hard as AH
    from amortized_design import solve
    print("[amort] specification sweep")
    with open("amort_hard_theta.pkl", "rb") as f:
        theta = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    Ts = np.linspace(1.0, 3.0, 17)
    v_ff, tr_ff, v_en = [], [], []
    for T in Ts:
        Tj = jnp.array(float(T))
        p = AH.shape_head(theta, Tj)
        sh = AH.scale_head(theta, Tj)
        v1, t1, _ = solve(p, sh)
        se = AH.enforce_sigma(p, Tj, sh)
        v2, t2, _ = solve(p, se)
        v_ff.append(float(v1))
        tr_ff.append(float(t1))
        v_en.append(float(v2))
        print(f"   T={T:.2f}  ff V={float(v1):7.3f} travel={float(t1):.4f}  "
              f"enf V={float(v2):7.3f}")
    np.savez("figdata_amort.npz", T=Ts, v_ff=np.array(v_ff),
             tr_ff=np.array(tr_ff), v_en=np.array(v_en))
    print("   -> figdata_amort.npz")


if __name__ == "__main__":
    which = [a.lower() for a in sys.argv[1:]] or ["fold", "nazemi", "valid", "amort"]
    if "fold" in which:
        gen_fold()
    if "nazemi" in which:
        gen_nazemi()
    if "valid" in which:
        gen_valid()
    if "amort" in which:
        gen_amort()
    print("done")
