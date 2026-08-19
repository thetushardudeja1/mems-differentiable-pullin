"""
Render IEEE figures as combined multi-panel plates.

STYLE RULES ENFORCED HERE
  * No ambiguous wording. "published" never appears; the actual source is named
    (Haluzan et al. 2010) and "this work" is stated explicitly, so a reviewer
    can tell at a glance which curve is whose.
  * Every axis carries a quantity AND its unit.
  * Panel letters (a), (b), (c) in bold, top-left, outside the data area.
  * Bolder strokes/markers than a single-panel figure would need, because
    panels are reduced when placed side by side.
  * Ticks inward on all four sides, minor ticks on.

  Fig. 1 (double column, 3 panels) -- solver validation and inverse design
  Fig. 2 (double column, 3 panels) -- AI-driven design and control

Rebuilds entirely from saved .npz/.npy; no experiment is re-run.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL1, COL2 = 3.5, 7.16          # IEEE single / double column width [in]

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.6,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT = "../figures"
os.makedirs(OUT, exist_ok=True)

# colour-blind-safe (Okabe-Ito)
C_REF = "#000000"      # reference / literature
C_OURS = "#0072B2"     # this work, main
C_ALT = "#D55E00"      # this work, alternative start / baseline
C_GREY = "#8C8C8C"
C_GREEN = "#009E73"


def panel_label(ax, s):
    ax.text(-0.20, 1.06, s, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=400)
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf  (+.png)")


# ============================== FIGURE 1 ====================================
def figure1():
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.15))
    a, b, c = axes

    # ---- (a) gap profiles -------------------------------------------------
    xi = np.load("xi_opt.npy")
    a.plot(xi, np.load("D_poly_best.npy"), "-", color=C_REF, lw=1.9,
           label="Haluzan et al. 2010\n(hand-tuned, $n{=}4/3$)")
    a.plot(xi, np.load("D_opt.npy"), "--", color=C_OURS, lw=1.7,
           label="This work: free-form\n(start: Haluzan design)")
    a.plot(xi, np.load("D_ctrl.npy"), ":", color=C_ALT, lw=1.9,
           label="This work: free-form\n(start: uniform gap)")
    a.set_xlabel(r"Normalised position $x/l$")
    a.set_ylabel(r"Electrode gap $d(x)$  [$\mu$m]")
    a.set_xlim(0, 1)
    a.set_ylim(0.5, 6.2)
    a.legend(loc="upper left", handlelength=1.7, labelspacing=0.35,
             borderpad=0.2)
    panel_label(a, "(a)")

    # ---- (b),(c) trend reversal ------------------------------------------
    d = np.load("figdata_f2.npz")
    # NB: matplotlib renders "--" literally, so use an explicit en-dash.
    for ax, pre, ttl in [(b, "cantilever", "Cantilever"),
                         (c, "fixed_fixed", "Fixed–fixed")]:
        n, v = d[f"{pre}_n"], d[f"{pre}_v"]
        ok = np.isfinite(v)
        ax.plot(n[ok], v[ok], "o-", color=C_OURS, ms=3.6, mfc="white",
                mew=1.3, label="This work")
        i = int(np.nanargmin(v))
        ax.plot(n[i], v[i], "*", color=C_ALT, ms=11, zorder=6,
                label=f"Optimum $n^*={n[i]:.2f}$")
        ax.set_xlabel(r"Gap-profile exponent $n$")
        ax.set_title(ttl, pad=3)
        ax.legend(loc="upper center", handlelength=1.4)
        # pad the x-range so a marker sitting at an end point is not clipped
        span = n[ok].max() - n[ok].min()
        ax.set_xlim(n[ok].min() - 0.06 * span, n[ok].max() + 0.06 * span)
        vspan = np.nanmax(v) - np.nanmin(v)
        ax.set_ylim(np.nanmin(v) - 0.10 * vspan, np.nanmax(v) + 0.22 * vspan)
    b.set_ylabel(r"Pull-in voltage $V_{\rm PI}$  [V]")
    c.set_ylabel(r"Pull-in voltage $V_{\rm PI}$  [V]")
    panel_label(b, "(b)")
    panel_label(c, "(c)")

    fig.tight_layout(pad=0.35, w_pad=1.5)
    save(fig, "Fig1_solver_and_inverse_design")


# ============================== FIGURE 2 ====================================
def figure2():
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.15))
    a, b, c = axes

    # ---- (a) RL adaptivity ------------------------------------------------
    d = np.load("figdata_f3.npz")
    r, rl, fx = d["r_ti"], d["rl"], d["fixed"]
    a.plot(r, r, "-", color=C_GREY, lw=1.2,
           label="Device tip-in ceiling")
    a.plot(r, rl, ".", color=C_OURS, ms=2.6, alpha=0.75,
           label=f"RL policy ($\\rho={np.corrcoef(r, rl)[0,1]:+.2f}$)")
    a.plot(r, fx, ".", color=C_ALT, ms=2.6, alpha=0.75,
           label=f"Fixed gain ($\\rho={np.corrcoef(r, fx)[0,1]:+.2f}$)")
    a.set_xlabel(r"Fabrication ceiling $r_{\rm ti}$ (unmeasurable)")
    a.set_ylabel(r"Achieved travel  [gap fraction]")
    a.legend(loc="upper left", handlelength=1.4, labelspacing=0.3)
    panel_label(a, "(a)")

    # ---- (b) warm start ---------------------------------------------------
    d = np.load("figdata_f4.npz")
    s, net, cold = d["steps"], d["net"], d["cold"]
    ref = float(d["converged_direct"])
    v0 = float(d["v_net_only"])
    # Excess over the BEST design found anywhere, on a log axis.
    # Referencing "converged direct" (13.665 V) was wrong: the network reaches
    # 13.636 V, so the excess went NEGATIVE and both the blue curve and the
    # star vanished below the axis. The best value found in the whole study is
    # the only valid zero for this plot.
    best = float(min(np.nanmin(net), np.nanmin(cold[1:]), v0))
    FLOOR = 1.2e-2          # keep converged curves visible on the log axis
    b.semilogy(s[1:], np.maximum(cold[1:] - best, FLOOR), "-", color=C_ALT,
               lw=1.7, label="From uniform gap")
    b.semilogy(s[1:], np.maximum(net[1:] - best, FLOOR), "-", color=C_OURS,
               lw=1.7, label="From network output")
    b.axhline(ref - best, color=C_GREY, ls="--", lw=1.1,
              label=f"Converged direct ({ref:.2f} V)")
    b.plot(0.6, max(v0 - best, FLOOR), "*", color=C_GREEN, ms=12, zorder=6)
    b.annotate("Network alone\n(0.04 ms)", (0.6, max(v0 - best, FLOOR)),
               textcoords="offset points", xytext=(12, 4), fontsize=6.2,
               color=C_GREEN, fontweight="bold")
    b.set_xlabel("Physics-solver optimisation steps")
    b.set_ylabel(r"$V_{\rm PI}$ above best design found  [V]")
    b.set_xlim(-6, 200)
    b.set_ylim(FLOOR * 0.75, 60)
    # legend OUTSIDE the data region: placed inside, its star marker reads as
    # a data point (it did, at ~(25, 7)).
    b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4,
             labelspacing=0.25, borderpad=0.15, fontsize=6.2)
    panel_label(b, "(b)")

    # ---- (c) method comparison -------------------------------------------
    # measured in surrogate_fair.py / fno_surrogate.py / amortized_hard.py
    labels = ["MLP surrogate", "FNO surrogate", "Direct (400)",
              "Direct converged", "Amortized net", "Net + 100 steps"]
    vals = [17.203, 14.245, 13.783, 13.665, 13.636, 13.599]
    cols = [C_ALT, C_ALT, C_OURS, C_OURS, C_GREEN, C_GREEN]
    y = np.arange(len(vals))[::-1]
    c.barh(y, np.array(vals) - 13.0, left=13.0, color=cols, height=0.66,
           edgecolor="black", lw=0.6)
    # value labels outside the bar end, with x-range padded so none is clipped
    for yi, v in zip(y, vals):
        c.text(v + 0.10, yi, f"{v:.2f}", va="center", fontsize=6.4)
    c.axvline(14.14, color=C_REF, ls="--", lw=1.2, zorder=5)
    # Reference label sits ABOVE the top bar. Placing it below or rotating it
    # along the line put it on top of the "Net + 100 steps" bar and its value.
    c.text(14.14, len(vals) - 0.28, "Haluzan et al. 14.14 V", fontsize=6.2,
           ha="center", va="bottom", color=C_REF)
    c.set_yticks(y)
    c.set_yticklabels(labels, fontsize=6.6)
    c.set_xlabel(r"Pull-in voltage $V_{\rm PI}$  [V]")
    c.set_xlim(13.0, 19.4)
    c.set_ylim(-0.62, len(vals) + 0.32)
    c.set_title("Equal budget: 400 solves", pad=3)
    panel_label(c, "(c)")

    fig.tight_layout(pad=0.35, w_pad=1.6)
    save(fig, "Fig2_ai_design_and_control")


# ============================== FIGURE 3 ====================================
def figure3():
    """The saddle-node fold that defines pull-in, and the mode at that point."""
    d = np.load("figdata_fold.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2 * 0.68, 2.15))

    for tag, name, col in [("cant", "Cantilever", C_OURS),
                           ("ff", "Fixed–fixed", C_ALT)]:
        s, lam = d[f"{tag}_s"], d[f"{tag}_lam"]
        dfold, lfold = d[f"{tag}_fold"]
        ok = np.isfinite(lam)
        a.plot(s[ok], lam[ok] / lfold, "-", color=col, label=name)
        a.plot(dfold / s.max() * s.max(), 1.0, "*", color=col, ms=12, zorder=6)
        b.plot(d[f"{tag}_xi"], d[f"{tag}_Y_uniform"], "-", color=col,
               label=name)
    a.axhline(1.0, color=C_GREY, ls=":", lw=1.0)
    a.text(0.03, 1.02, "pull-in (fold)", fontsize=6.4, color=C_GREY)
    a.set_xlabel(r"Controlled deflection $\delta / d_0$")
    a.set_ylabel(r"Normalised load $\Lambda / \Lambda_{\rm PI}$")
    a.set_ylim(0, 1.18)
    a.legend(loc="lower center", handlelength=1.4)
    panel_label(a, "(a)")

    b.set_xlabel(r"Normalised position $x/l$")
    b.set_ylabel(r"Deflection at pull-in $y/d_0$")
    b.set_xlim(0, 1)
    b.legend(loc="upper left", handlelength=1.4)
    panel_label(b, "(b)")

    fig.tight_layout(pad=0.35, w_pad=1.5)
    save(fig, "Fig3_fold_structure")


# ============================== FIGURE 4 ====================================
def figure4():
    """Validation against every independent source we checked."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2 * 0.78, 2.25))

    v = np.load("figdata_valid.npz", allow_pickle=True)
    lab, err = list(v["labels"]), np.array(v["err"], dtype=float)
    y = np.arange(len(err))[::-1]
    colr = [C_GREEN if e < 1 else (C_OURS if e < 3 else C_ALT) for e in err]
    a.barh(y, np.maximum(err, 0.004), color=colr, height=0.66,
           edgecolor="black", lw=0.6)
    for yi, e in zip(y, err):
        # 2-dp rounding printed the 0.001% lumped check as "0.00%"
        txt = "<0.01%" if e < 0.01 else f"{e:.2f}%"
        a.text(max(e, 0.004) * 1.25, yi, txt, va="center", fontsize=6.2)
    a.set_xscale("log")
    a.set_yticks(y)
    # strip LaTeX escaping: matplotlib mathtext renders "\#" literally
    a.set_yticklabels([s.replace(r"\#", "#") for s in lab], fontsize=6.4)
    a.set_xlim(3e-3, 40)
    a.set_xlabel("Relative error vs. literature  [%]")
    a.set_title("Solver validation", pad=3)
    panel_label(a, "(a)")

    # NORMALISED to the 35% device. Absolute voltages are offset (~+85%) by
    # anchor compliance that neither model captures -- our 1D beam assumes
    # ideal clamps -- so plotting raw volts against the measurements would
    # misrepresent what is actually being compared. The paper's own Fig. 8
    # is normalised for the same reason, and the claim under test is the
    # RELATIVE reduction with electrode ratio.
    n = np.load("figdata_nazemi.npz")
    r, v = n["r"] * 100, n["v"]
    v0 = float(np.interp(35.0, r, v))
    b.plot(r, v / v0, "o-", color=C_OURS, ms=3.6, mfc="white", mew=1.3,
           label="This work (simulated)")
    b.plot(n["meas_r"] * 100, n["meas_v"] / n["meas_v"][0], "s", color=C_REF,
           ms=5.5, label="Nazemi et al. 2025\n(measured)")
    # matplotlib mathtext is not usetex here, so "\%" would print literally
    b.annotate(f"{100*(1-v[6]/v0):.1f}% (ours)", (68, v[6] / v0),
               textcoords="offset points", xytext=(4, 10), fontsize=6.2,
               color=C_OURS)
    b.annotate("16.0% (measured)", (68, n["meas_v"][1] / n["meas_v"][0]),
               textcoords="offset points", xytext=(-62, -12), fontsize=6.2,
               color=C_REF)
    b.set_xlabel("Bottom-electrode length ratio  [%]")
    b.set_ylabel(r"$V_{\rm PI}$ normalised to 35% device")
    b.set_title("Microbridge (Nazemi 2025)", pad=3)
    b.legend(loc="upper right", handlelength=1.4, labelspacing=0.3)
    panel_label(b, "(b)")

    fig.tight_layout(pad=0.35, w_pad=1.6)
    save(fig, "Fig4_validation")


# ============================== FIGURE 5 ====================================
def figure5():
    """Amortized network across the entire specification range."""
    d = np.load("figdata_amort.npz")
    T, v_ff, v_en, tr = d["T"], d["v_ff"], d["v_en"], d["tr_ff"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2 * 0.68, 2.15))

    a.plot(T, v_ff, "-", color=C_GREEN, lw=1.8,
           label="Feed-forward (0.04 ms)")
    a.plot(T, v_en, "--", color=C_OURS, lw=1.5,
           label="+ constraint layer")
    a.set_xlabel(r"Required travel specification  [$\mu$m]")
    a.set_ylabel(r"Pull-in voltage $V_{\rm PI}$  [V]")
    a.legend(loc="upper left", handlelength=1.5)
    panel_label(a, "(a)")

    err = 100.0 * np.abs(tr - T) / T
    b.plot(T, err, "o-", color=C_GREEN, ms=3.4, mfc="white", mew=1.2,
           label="Hard constraint layer")
    b.axhline(5.0, color=C_ALT, ls="--", lw=1.3,
              label="Soft penalty (previous)")
    b.axhline(np.mean(err), color=C_GREY, ls=":", lw=1.1)
    b.text(2.95, np.mean(err) * 1.35, f"mean {np.mean(err):.2f}%",
           fontsize=6.2, ha="right", color=C_GREY)
    b.set_xlabel(r"Required travel specification  [$\mu$m]")
    b.set_ylabel("Specification error  [%]")
    b.set_ylim(0, 6.2)
    b.legend(loc="upper center", handlelength=1.5)
    panel_label(b, "(b)")

    fig.tight_layout(pad=0.35, w_pad=1.5)
    save(fig, "Fig5_amortized_sweep")


if __name__ == "__main__":
    print("rendering IEEE multi-panel figures -> ../figures/")
    for fn, nm in [(figure1, "Fig1"), (figure2, "Fig2"), (figure3, "Fig3"),
                   (figure4, "Fig4"), (figure5, "Fig5")]:
        try:
            fn()
        except Exception as e:
            print(f"  {nm} FAILED: {type(e).__name__}: {e}")
    print("done")
