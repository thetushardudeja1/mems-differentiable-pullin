"""
Fig. 6 -- the exact shape sensitivity, and what it costs.

This is the object the whole method exists to produce: dV_PI/dD(xi), the
derivative of the bifurcation with respect to the electrode gap at every point
along the beam, from a single reverse pass.

Two panels:
  (a) the sensitivity curve, peak marked. It peaks near the CLAMP, not at the
      free tip where deflection is largest -- the electrode shaping that buys
      drive voltage happens in the first quarter of the beam.
  (b) cost of obtaining it: one reverse pass versus 2N forward solves for
      central differences, and the agreement between the two.

Matches the house style of make_figures.py. Writes ../figures/Fig6_sensitivity.pdf
and .png.

    python make_fig_sensitivity.py
"""

import os
import time
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit, grad

import beam as B
import inverse_design as ID

COL1 = 3.5                       # IEEE single-column width [in]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.6, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "lines.linewidth": 1.6, "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_OURS, C_ALT, C_GREY = "#0072B2", "#D55E00", "#666666"
OUT = "../figures"
N, BC, EPS = 60, B.CANTILEVER, 1e-5


def main():
    xi = B.node_xi(N, BC)
    fold = jit(partial(B.pullin_fold, alpha=ID.ALPHA, N_t=0.0, N=N, bc=BC))

    D = jnp.maximum(1.0, 4.7275 * ID.profile_coord(xi, BC) ** 1.0)
    z0 = fold(D)[3]

    @jit
    def V_of_D(Dv):
        return ID.volts(fold(Dv, z0=z0)[0])

    dV = jit(grad(V_of_D))
    dV(D).block_until_ready()

    t0 = time.perf_counter()
    g = np.asarray(dV(D)); g_ready = float(g[0])
    t_ad = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_fd = np.empty(N)
    for i in range(N):
        g_fd[i] = float((V_of_D(D.at[i].add(EPS)) - V_of_D(D.at[i].add(-EPS)))
                        / (2 * EPS))
    t_fd = time.perf_counter() - t0

    x = np.asarray(xi)
    i_pk = int(np.argmax(np.abs(g)))
    rel_pk = abs(g[i_pk] - g_fd[i_pk]) / abs(g_fd[i_pk])

    fig, (a, b) = plt.subplots(1, 2, figsize=(COL1, 1.62))
    fig.subplots_adjust(wspace=0.52)

    # ---- (a) the sensitivity curve ----------------------------------------
    a.fill_between(x, 0, g, color=C_OURS, alpha=0.18)
    a.plot(x, g, color=C_OURS)
    a.plot(x[i_pk], g[i_pk], "o", color=C_ALT, ms=4.5, zorder=5)
    a.annotate(r"$\xi^*$=%.2f" % x[i_pk], xy=(x[i_pk], g[i_pk]),
               xytext=(x[i_pk] + 0.20, g[i_pk] * 0.60), fontsize=6.6,
               color=C_ALT,
               arrowprops=dict(arrowstyle="->", lw=0.8, color=C_ALT))
    a.text(0.98, 0.06, "tip: %.0f%% of peak" % (100 * g[-1] / g[i_pk]),
           transform=a.transAxes, ha="right", fontsize=6.2, color=C_GREY)
    a.set_xlabel(r"position $\xi = x/l$")
    a.set_ylabel(r"$\partial V_{\rm PI}/\partial D$")
    a.set_title("(a)  exact sensitivity", loc="left")

    # ---- (b) cost -----------------------------------------------------------
    names = ["central\ndiff.", "reverse\nautodiff"]
    solves = [2 * N, 1]
    bars = b.bar(names, solves, color=[C_GREY, C_OURS], width=0.58)
    b.set_yscale("log")
    b.set_ylim(0.5, 400)
    b.set_ylabel("solves required")
    for bar, s, t in zip(bars, solves, [t_fd, t_ad]):
        b.text(bar.get_x() + bar.get_width() / 2, s * 1.35,
               "%d\n%.0f ms" % (s, 1e3 * t), ha="center", fontsize=6.2)
    b.set_title("(b)  cost, $N=%d$" % N, loc="left")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "Fig6_sensitivity." + ext))
    plt.close(fig)

    print("Fig6_sensitivity  peak at xi=%.3f  tip/peak=%.1f%%" %
          (x[i_pk], 100 * g[-1] / g[i_pk]))
    print("  autodiff %.0f ms (1 pass) vs central differences %.0f ms (%d solves)"
          % (1e3 * t_ad, 1e3 * t_fd, 2 * N))
    print("  speed-up %.0fx; agreement at the dominant node %.1e relative"
          % (t_fd / t_ad, rel_pk))


if __name__ == "__main__":
    main()
