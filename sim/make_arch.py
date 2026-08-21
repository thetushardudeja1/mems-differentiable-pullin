"""Render the architecture diagram (Fig. 0) from its hand-authored SVG.

The source of truth is ``figures/Fig0_architecture.svg``, written by hand
rather than emitted by matplotlib: it is a block diagram, not a plot, and hand
authoring gives exact control over box geometry, text baselines and the
distribution bus, none of which matplotlib expresses naturally. This script
only converts it to the PDF that LaTeX needs and a PNG for the READMEs.

Text is intentionally left as text in the SVG (not converted to paths) so it
stays selectable and searchable in the compiled report.

    python make_arch.py
"""

import os
import sys

# Each entry: (svg, stem, target width in inches).
# cairosvg's output_width is in CSS pixels at 96 dpi, NOT points. Passing
# inches*72 silently produced a 5.37 in PDF instead of 7.16 in.
#
# Three variants exist because one drawing cannot serve all three slots at a
# legible type size:
#   Fig0        7.16 in  double column, the full diagram
#   Fig0b       6.90 in  wide strip for the one-page pitch header
#   Fig0c       3.50 in  IEEE SINGLE column, for the technical report, whose
#                        3-page limit cannot afford a full-width float
FIGS = [
    ("../figures/Fig0_architecture.svg",       "Fig0_architecture",       7.16),
    ("../figures/Fig0b_pitch_strip.svg",       "Fig0b_pitch_strip",       6.90),
    ("../figures/Fig0c_architecture_1col.svg", "Fig0c_architecture_1col", 3.50),
]

def main():
    try:
        import cairosvg
    except ImportError:
        sys.exit("cairosvg is required:  pip install cairosvg\n"
                 "(alternatively: rsvg-convert or inkscape on the SVG in "
                 "figures/)")

    for src, stem, width_in in FIGS:
        if not os.path.exists(src):
            print(f"  SKIP {src} (missing)")
            continue
        pdf = f"../figures/{stem}.pdf"
        png = f"../figures/{stem}.png"
        cairosvg.svg2pdf(url=src, write_to=pdf, output_width=width_in * 96.0)
        cairosvg.svg2png(url=src, write_to=png, output_width=1900)
        try:
            from pypdf import PdfReader
            b = PdfReader(pdf).pages[0].mediabox
            print(f"  {stem}: {float(b.width)/72:.2f} x "
                  f"{float(b.height)/72:.2f} in  (target {width_in} in)")
        except ImportError:
            print(f"  wrote {pdf}")


if __name__ == "__main__":
    main()
