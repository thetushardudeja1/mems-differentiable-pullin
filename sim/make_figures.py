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

  Fig. 1 (SINGLE column, 3 panels) -- solver validation and inverse design
  Fig. 2 (SINGLE column, 2 panels) -- AI-driven design and control

Figs. 1 and 2 are single-column on purpose. A figure* blocks the full page
width; a single-column figure blocks one column and lets text flow past it,
which is what a hard 3-page limit needs. Only Fig. 0 (the architecture
diagram) and Table III stay full width, because neither is legible narrower.

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


def panel_label(ax, s, dx=-0.20, dy=1.06):
    # dx/dy are overridable because in the single-column stacked layouts the
    # default position lands on top of a tall y-axis label.
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=400)
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf  (+.png)")


# ============================== FIGURE 1 ====================================
def figure1():
    """SINGLE COLUMN. A figure* blocks the whole page width; a single-column
    figure blocks one column and lets text flow past it, which is what the
    3-page limit needs. Panel (a) spans the column, (b) and (c) share the row
    below it -- the EDL/TED convention of grouping panels inside one column."""
    fig = plt.figure(figsize=(COL1, 2.40))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92],
                          hspace=0.62, wspace=0.42)
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1])

    # ---- (a) gap profiles -------------------------------------------------
    xi = np.load("xi_opt.npy")
    a.plot(xi, np.load("D_poly_best.npy"), "-", color=C_REF, lw=1.7,
           label="Haluzan et al. 2010 (hand-tuned, $n{=}4/3$)")
    a.plot(xi, np.load("D_opt.npy"), "--", color=C_OURS, lw=1.6,
           label="This work (start: Haluzan design)")
    a.plot(xi, np.load("D_ctrl.npy"), ":", color=C_ALT, lw=1.8,
           label="This work (start: uniform gap)")
    a.set_xlabel(r"Normalised position $x/l$", labelpad=1.5)
    a.set_ylabel(r"Gap $d(x)$  [$\mu$m]", labelpad=1.5)
    a.set_xlim(0, 1)
    a.set_ylim(0.5, 7.6)
    a.legend(loc="upper left", handlelength=1.5, labelspacing=0.22,
             borderpad=0.15, fontsize=7.4)
    panel_label(a, "(a)", dx=-0.155, dy=1.10)

    # ---- (b),(c) trend reversal ------------------------------------------
    d = np.load("figdata_f2.npz")
    # NB: matplotlib renders "--" literally, so use an explicit en-dash.
    for ax, pre, ttl in [(b, "cantilever", "Cantilever"),
                         (c, "fixed_fixed", "Fixed–fixed")]:
        n, v = d[f"{pre}_n"], d[f"{pre}_v"]
        ok = np.isfinite(v)
        ax.plot(n[ok], v[ok], "o-", color=C_OURS, ms=2.8, mfc="white", mew=1.1)
        i = int(np.nanargmin(v))
        ax.plot(n[i], v[i], "*", color=C_ALT, ms=9, zorder=6)
        # optimum stated as text, not a legend entry: at 1.5 in wide a legend
        # box covers the curve it is describing.
        ax.set_title(f"{ttl}, $n^*\\!=\\!{n[i]:.2f}$", pad=2.5, fontsize=8.0)
        ax.set_xlabel(r"Exponent $n$", labelpad=1.5)
        ax.tick_params(labelsize=7.4)
        span = n[ok].max() - n[ok].min()
        ax.set_xlim(n[ok].min() - 0.06 * span, n[ok].max() + 0.06 * span)
        vspan = np.nanmax(v) - np.nanmin(v)
        ax.set_ylim(np.nanmin(v) - 0.12 * vspan, np.nanmax(v) + 0.12 * vspan)
    b.set_ylabel(r"$V_{\rm PI}$  [V]", labelpad=1.5)
    c.set_ylabel(r"$V_{\rm PI}$  [V]", labelpad=1.5)
    panel_label(b, "(b)")
    panel_label(c, "(c)")

    save(fig, "Fig1_solver_and_inverse_design")


# ============================== FIGURE 2 ====================================
def figure2():
    """SINGLE COLUMN, two panels. The old panel (c) -- the equal-budget bar
    chart -- is gone: Table III now carries every one of those numbers as rows,
    and repeating them cost a third of a page against a hard 3-page limit."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL1, 1.62))
    fig.subplots_adjust(wspace=0.52)

    # ---- (a) RL adaptivity ------------------------------------------------
    # Prefer the full-scale 3-seed run (analyze_rl2dof.py: 2000 iters x 4096
    # envs) that the report's 80.4% and rho are quoted from. figdata_f3.npz is
    # the REDUCED single-seed run from gen_figdata (1200 x 1024) and shows ~66%
    # and rho=+0.958 -- plotting that beside the full-scale text numbers is the
    # inconsistency this fallback exists to make visible rather than silent.
    if os.path.exists("figdata_rl3.npz"):
        d = np.load("figdata_rl3.npz")
        r, rl, fx = d["r_ti"], d["rl_best"], d["fixed"]
    else:
        print("  NOTE: figdata_rl3.npz missing -- Fig. 2(a) falls back to the "
              "reduced run.\n        Run 'python analyze_rl2dof.py 3' to match "
              "the reported numbers.")
        d = np.load("figdata_f3.npz")
        r, rl, fx = d["r_ti"], d["rl"], d["fixed"]
    rho_rl = np.corrcoef(r, rl)[0, 1]
    rho_fx = np.corrcoef(r, fx)[0, 1]
    a.plot(r, r, "-", color=C_GREY, lw=1.0)
    a.plot(r, rl, ".", color=C_OURS, ms=1.8, alpha=0.8)
    a.plot(r, fx, ".", color=C_ALT, ms=1.8, alpha=0.8)
    # Curves labelled in place. At 1.5 in wide a legend box covers the data it
    # is describing, so the rho values sit next to their own point clouds.
    a.text(0.97, 0.97, "device ceiling", transform=a.transAxes, fontsize=7.4,
           color=C_GREY, ha="right", va="top")
    a.text(0.97, 0.50, f"RL  $\\rho={rho_rl:+.2f}$", transform=a.transAxes,
           fontsize=7.4, color=C_OURS, ha="right", va="top", fontweight="bold")
    a.text(0.97, 0.15, f"fixed  $\\rho={rho_fx:+.2f}$", transform=a.transAxes,
           fontsize=7.4, color=C_ALT, ha="right", va="top", fontweight="bold")
    a.set_xlabel(r"Ceiling $r_{\rm ti}$ (unmeasurable)", labelpad=1.2,
                 fontsize=7.3)
    a.set_ylabel("Achieved travel", labelpad=1.2, fontsize=7.3)
    a.tick_params(labelsize=7.4)
    panel_label(a, "(a)", dx=-0.30, dy=1.04)

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
               lw=1.3)
    b.semilogy(s[1:], np.maximum(net[1:] - best, FLOOR), "-", color=C_OURS,
               lw=1.3)
    b.axhline(ref - best, color=C_GREY, ls="--", lw=0.9)
    b.plot(0.6, max(v0 - best, FLOOR), "*", color=C_GREEN, ms=8, zorder=6)
    # Four labels in a 1.5 in panel. The previous set overlapped each other and
    # the y-axis: "converged direct (13.66 V)" alone was 26 characters, wider
    # than the axes. Labels are short now and the value moves to the caption.
    b.text(0.94, 0.97, "cold start", transform=b.transAxes, fontsize=7.0,
           color=C_ALT, ha="right", va="top", fontweight="bold")
    b.text(0.06, 0.55, "direct", transform=b.transAxes,
           fontsize=7.0, color=C_GREY, ha="left", va="bottom")
    b.text(0.94, 0.20, "warm start", transform=b.transAxes, fontsize=7.0,
           color=C_OURS, ha="right", va="top", fontweight="bold")
    b.text(0.06, 0.30, "network", transform=b.transAxes, fontsize=7.0,
           color=C_GREEN, ha="left", va="top", fontweight="bold")
    b.set_xlabel("Solver steps", labelpad=1.2, fontsize=7.3)
    b.set_ylabel(r"$V_{\rm PI}$ above best [V]", labelpad=1.2, fontsize=7.3)
    b.set_xlim(-6, 200)
    b.set_ylim(FLOOR * 0.75, 60)
    b.set_xticks([0, 100, 200])
    b.tick_params(labelsize=7.4)
    # No legend box: at 1.5 in wide it covered the curves, and its star marker
    # read as a data point. Curves are labelled in place instead.
    panel_label(b, "(b)", dx=-0.34, dy=1.04)

    save(fig, "Fig2_ai_design_and_control")


# ============================== FIGURE 3 ====================================
def figure3():
    """The saddle-node fold that defines pull-in, and the mode at that point."""
    # SINGLE COLUMN. Sized at COL1 rather than scaled down from COL2 in LaTeX:
    # \includegraphics shrinks the type along with the axes, and at 3.5 in a
    # COL2 figure's 8 pt labels land at ~5.8 pt.
    d = np.load("figdata_fold.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL1, 1.62))
    fig.subplots_adjust(wspace=0.54)

    for tag, name, col in [("cant", "Cantilever", C_OURS),
                           ("ff", "Fixed–fixed", C_ALT)]:
        s, lam = d[f"{tag}_s"], d[f"{tag}_lam"]
        dfold, lfold = d[f"{tag}_fold"]
        ok = np.isfinite(lam)
        a.plot(s[ok], lam[ok] / lfold, "-", color=col, lw=1.3, label=name)
        a.plot(dfold / s.max() * s.max(), 1.0, "*", color=col, ms=8, zorder=6)
        b.plot(d[f"{tag}_xi"], d[f"{tag}_Y_uniform"], "-", color=col, lw=1.3,
               label=name)
    a.axhline(1.0, color=C_GREY, ls=":", lw=0.9)
    a.text(0.03, 1.03, "pull-in (fold)", fontsize=7.4, color=C_GREY)
    a.set_xlabel(r"Deflection $\delta / d_0$", labelpad=1.2, fontsize=7.3)
    a.set_ylabel(r"$\Lambda / \Lambda_{\rm PI}$", labelpad=1.2, fontsize=7.3)
    a.set_ylim(0, 1.20)
    a.tick_params(labelsize=7.4)
    a.legend(loc="lower center", handlelength=1.1, fontsize=7.4,
             labelspacing=0.2, borderpad=0.15)
    panel_label(a, "(a)", dx=-0.30, dy=1.04)

    b.set_xlabel(r"Position $x/l$", labelpad=1.2, fontsize=7.3)
    b.set_ylabel(r"$y/d_0$ at pull-in", labelpad=1.2, fontsize=7.3)
    b.set_xlim(0, 1)
    b.tick_params(labelsize=7.4)
    b.legend(loc="upper left", handlelength=1.1, fontsize=7.4,
             labelspacing=0.2, borderpad=0.15)
    panel_label(b, "(b)", dx=-0.30, dy=1.04)

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
        a.text(max(e, 0.004) * 1.25, yi, txt, va="center", fontsize=7.3)
    a.set_xscale("log")
    a.set_yticks(y)
    # strip LaTeX escaping: matplotlib mathtext renders "\#" literally
    a.set_yticklabels([s.replace(r"\#", "#") for s in lab], fontsize=7.4)
    a.set_xlim(3e-3, 40)
    a.set_xlabel("Relative error vs. literature  [%]")
    a.set_title("Solver validation", pad=3)
    panel_label(a, "(a)", dx=-0.155, dy=1.10)

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
               textcoords="offset points", xytext=(4, 10), fontsize=7.4,
               color=C_OURS)
    # Offset upward, not downward: at (-62,-12) this label sat on top of the
    # x-axis title and the two collided.
    b.annotate("16.0% (measured)", (68, n["meas_v"][1] / n["meas_v"][0]),
               textcoords="offset points", xytext=(-70, 6), fontsize=7.4,
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
    # SINGLE COLUMN, sized at COL1 for the same reason as figure3().
    d = np.load("figdata_amort.npz")
    T, v_ff, v_en, tr = d["T"], d["v_ff"], d["v_en"], d["tr_ff"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL1, 1.62))
    fig.subplots_adjust(wspace=0.56)

    # A legend box does not fit a 1.6 in panel -- it overflowed the axes frame.
    # The two curves also lie on top of each other, which is itself the result:
    # the constraint layer costs essentially nothing in drive voltage. So they
    # are labelled in place and the caption states the finding.
    a.plot(T, v_ff, "-", color=C_GREEN, lw=1.6)
    a.plot(T, v_en, "--", color=C_OURS, lw=1.2)
    a.set_xlabel(r"Travel spec  [$\mu$m]", labelpad=1.2, fontsize=7.3)
    a.set_ylabel(r"$V_{\rm PI}$  [V]", labelpad=1.2, fontsize=7.3)
    a.tick_params(labelsize=7.4)
    a.text(0.04, 0.96, "both modes", transform=a.transAxes, fontsize=7.0,
           color=C_GREY, ha="left", va="top")
    a.text(0.04, 0.84, "coincide", transform=a.transAxes, fontsize=7.0,
           color=C_GREY, ha="left", va="top")
    panel_label(a, "(a)", dx=-0.30, dy=1.04)

    err = 100.0 * np.abs(tr - T) / T
    b.plot(T, err, "o-", color=C_GREEN, ms=2.4, mfc="white", mew=0.9)
    b.axhline(5.0, color=C_ALT, ls="--", lw=1.1)
    b.axhline(np.mean(err), color=C_GREY, ls=":", lw=1.0)
    # Labelled in place -- a legend box covered the error curve. Text is kept
    # short and left-anchored: the previous right-anchored string was wider
    # than the panel and was clipped at the frame.
    b.text(0.05, 0.95, "soft penalty", transform=b.transAxes, fontsize=7.2,
           color=C_ALT, ha="left", va="top", fontweight="bold")
    b.text(0.05, 0.28, f"ours: {np.mean(err):.2f}%", transform=b.transAxes,
           fontsize=7.2, color="#00715A", ha="left", va="top",
           fontweight="bold")
    b.set_xlabel(r"Travel spec  [$\mu$m]", labelpad=1.2, fontsize=7.3)
    b.set_ylabel("Spec error  [%]", labelpad=1.2, fontsize=7.3)
    b.set_ylim(0, 6.2)
    b.tick_params(labelsize=7.4)
    panel_label(b, "(b)", dx=-0.30, dy=1.04)

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
