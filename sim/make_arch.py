"""
Render the system architecture diagram (Fig. 1 of the report / pitch hero).

STYLE RULES (same as make_figures.py)
  * No ambiguous wording. Every claim on the page names its number or its
    source; "published" never appears unattributed.
  * Colour-blind-safe Okabe-Ito palette, Type-42 fonts, IEEE column widths.
  * Square corners and 1.1 pt strokes: an engineering block diagram, not a
    marketing slide.

This figure carries the framing: the top strip contrasts standard practice
(fit a surrogate to ~10^4 simulations, then optimise through its error) with
this work (exact gradients, no stored simulations), and the three branches
below are the device-level capabilities that follow.

LAYOUT NOTE
  The axes is full-bleed (add_axes([0,0,1,1])) and the figure is saved with
  bbox_inches=None on purpose. With the default subplot margins the drawing
  area was only 0.775 x 0.77 of the figure, so 100 data units did not map to
  the IEEE column width and every text budget below was wrong; overflowing
  text then made bbox="tight" grow the canvas sideways and squash the plate.
  With the axes full-bleed, 100 x-units == W inches and 100 y-units == H
  inches exactly, which is what the character-count budgets here assume.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

W, H = 7.16, 3.9                 # IEEE double-column width [in]

plt.rcParams.update({
    "font.size": 8,
    "figure.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT = "../figures"
os.makedirs(OUT, exist_ok=True)

C_OURS = "#0072B2"     # this work
C_ALT = "#D55E00"      # contrast / standard practice
C_GREEN = "#009E73"    # capabilities
C_DGREEN = "#00715A"
C_GREY = "#8C8C8C"
C_INK = "#000000"

F_OURS = "#E3F0F8"     # tints
F_ALT = "#FBEAE0"
F_GREEN = "#E1F3EE"
F_GREY = "#F2F2F2"


def box(ax, x0, y0, w, h, ec, fc, lw=1.1, z=2):
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, zorder=z))


def txt(ax, x, y, s, size=7.0, weight="normal", color=C_INK,
        ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, style=style, zorder=5, linespacing=1.45)


def arrow(ax, p0, p1, color=C_INK, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0, zorder=4))


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---------------------------------------------------- the framing strip
    box(ax, 1, 86, 47, 13.5, C_ALT, F_ALT)
    txt(ax, 3.2, 96.2, "STANDARD PRACTICE", 6.6, "bold", C_ALT, ha="left")
    txt(ax, 3.2, 90.2,
        r"$\sim$10$^4$ FEM simulations $\rightarrow$ fit a surrogate"
        "\n"
        r"$\rightarrow$ optimise through its approximation error",
        6.9, ha="left")

    box(ax, 52, 86, 47, 13.5, C_OURS, F_OURS)
    txt(ax, 54.2, 96.2, "THIS WORK", 6.6, "bold", C_OURS, ha="left")
    txt(ax, 54.2, 90.2,
        r"0 stored simulations $\rightarrow$ differentiate the"
        "\n"
        r"physics itself $\rightarrow$ exact gradients, no surrogate",
        6.9, ha="left")

    # ------------------------------------------------------------ the input
    box(ax, 1, 58, 19, 22, C_GREY, F_GREY)
    txt(ax, 10.5, 77.4, "DEVICE UNDER DESIGN", 6.2, "bold", "#5A5A5A")
    txt(ax, 10.5, 66.5,
        "Electrode gap\nprofile $d(x)$\n\nTravel spec $T$,\n"
        "boundary condition", 6.9)

    # ------------------------------------------------------------- the core
    box(ax, 30, 57, 44, 25, C_OURS, F_OURS, lw=1.6)
    txt(ax, 52, 79.0, "DIFFERENTIABLE PULL-IN SOLVER", 7.4, "bold", C_OURS)
    txt(ax, 52, 74.3,
        "pull-in is a saddle-node bifurcation $-$ we solve for it directly",
        6.2, style="italic", color="#3A3A3A")
    txt(ax, 52, 68.3,
        r"$R(Y,\Lambda)=0$     $J(Y,\Lambda)\,v=0$     "
        r"$v^{\top}v-1=0$", 8.0)
    txt(ax, 52, 61.5,
        "extended fold system (Keller 1977)\n"
        r"returns $V_{\rm PI}$, stable travel, deflection mode $v$",
        6.4)

    # forward solve / reverse gradient
    arrow(ax, (20.4, 72.0), (29.6, 72.0), C_INK, 1.3)
    txt(ax, 25.0, 74.0, "solve", 6.2)
    arrow(ax, (29.6, 63.0), (20.4, 63.0), C_OURS, 1.3)
    txt(ax, 25.0, 60.8, "gradient", 6.2, color=C_OURS)

    # ------------------------------- the load-bearing claim of the whole set
    arrow(ax, (52, 56.6), (52, 53.4), C_OURS, 1.4)
    box(ax, 5, 46, 90, 7.2, C_OURS, "#FFFFFF", lw=1.3)
    txt(ax, 50, 49.6,
        r"exact  $\partial V_{\rm PI}/\partial d(x)$  by the implicit "
        r"function theorem $-$ differentiated $\it{through}$ the instability",
        7.2, "bold", C_OURS)

    # ---------------------------------------------------- the three outputs
    bw, gap = 31.0, 3.5
    xs = [1.0, 1.0 + bw + gap, 1.0 + 2 * (bw + gap)]
    cx = [x + bw / 2 for x in xs]

    # distribution bus, so no arrow ever crosses a text box
    arrow(ax, (50, 45.6), (50, 44.0), C_GREEN, 1.2)
    ax.plot([cx[0], cx[2]], [44.0, 44.0], color=C_GREEN, lw=1.2, zorder=4)
    for c in cx:
        arrow(ax, (c, 44.0), (c, 41.4), C_GREEN, 1.2)

    heads = ["1   INVERSE DESIGN",
             "2   AMORTIZED NETWORK",
             "3   SAFE-OPERATION RL"]
    bodies = ["Gradient descent on the\nfree-form gap profile,\n"
              "every iterate feasible",
              r"Maps spec $\rightarrow$ geometry," "\ntrained end-to-end\n"
              "through the solver",
              "Infers each device's own\nfabrication ceiling,\n"
              "which is not measurable"]
    results = ["22.86 V $\\rightarrow$ 13.60 V ($-$41%)\ngeometry only, in 15 s",
               "0 stored simulations\n0.04 ms per design",
               "+12.8% usable travel\n0.0% of devices destroyed"]

    for x0, hd, bd, rs in zip(xs, heads, bodies, results):
        box(ax, x0, 11, bw, 30, C_GREEN, "#FFFFFF")
        box(ax, x0, 36, bw, 5.0, C_GREEN, F_GREEN)
        txt(ax, x0 + bw / 2, 38.5, hd, 6.8, "bold", C_DGREEN)
        txt(ax, x0 + bw / 2, 29.0, bd, 6.9)
        box(ax, x0 + 2.0, 13.0, bw - 4.0, 8.5, C_GREEN, F_GREEN, lw=0.9)
        txt(ax, x0 + bw / 2, 17.2, rs, 7.0, "bold", C_DGREEN)

    # ------------------------------------------------------ validation rail
    box(ax, 1, 0.8, 98, 8.0, C_INK, "#FFFFFF", lw=1.0)
    txt(ax, 50, 4.8,
        "VALIDATED AGAINST 8 INDEPENDENT SOURCES, 1967-2025\n"
        r"exact gradient matches the analytic $c^{3}$ law to 0.00%   "
        r"$\bullet$   optimal-exponent reversal between cantilever "
        "and fixed-fixed beams reproduced",
        6.3, "bold")

    fig.savefig(f"{OUT}/Fig0_architecture.pdf", bbox_inches=None)
    fig.savefig(f"{OUT}/Fig0_architecture.png", dpi=400, bbox_inches=None)
    plt.close(fig)
    print(f"  wrote {OUT}/Fig0_architecture.pdf  (+.png)")


if __name__ == "__main__":
    main()
