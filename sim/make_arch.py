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

SRC = "../figures/Fig0_architecture.svg"
OUT_PDF = "../figures/Fig0_architecture.pdf"
OUT_PNG = "../figures/Fig0_architecture.png"

# IEEE double-column width. The SVG viewBox is 972 units wide -- 940 of content
# plus a 16-unit margin on each side -- and that is the scale the font sizes in
# it were budgeted against (15 units -> ~8 pt).
WIDTH_IN = 7.16
VIEWBOX_W = 972

# cairosvg's output_width is in CSS pixels at 96 dpi, NOT points. Passing
# inches*72 silently produced a 5.37 in PDF instead of 7.16 in.
WIDTH_PX = WIDTH_IN * 96.0

def main():
    try:
        import cairosvg
    except ImportError:
        sys.exit("cairosvg is required:  pip install cairosvg\n"
                 "(alternatively: rsvg-convert or inkscape on the SVG in "
                 "figures/)")

    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC}")

    cairosvg.svg2pdf(url=SRC, write_to=OUT_PDF, output_width=WIDTH_PX)
    cairosvg.svg2png(url=SRC, write_to=OUT_PNG, output_width=VIEWBOX_W * 2)

    for f in (OUT_PDF, OUT_PNG):
        print(f"  wrote {f}  ({os.path.getsize(f) / 1024:.0f} kB)")

    try:
        from pypdf import PdfReader
        b = PdfReader(OUT_PDF).pages[0].mediabox
        print(f"  PDF page: {float(b.width)/72:.2f} x "
              f"{float(b.height)/72:.2f} in  (IEEE double column = "
              f"{WIDTH_IN} in)")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
