"""Publication style: Springer Nature column widths, one sans-serif family at 7 pt, colour-blind-safe system colours.

Every figure is written as vector PDF (TrueType fonts embedded), SVG (text kept as text) and 600-dpi PNG.
"""
import shutil
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from . import layout as L

MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM
FONT_SIZE = 7
N_NOTE = "n = 3 independent replicas × 100 ns"
FIGS = L.OUT / "figures"
RESULTS_FIGS = L.REPO / "analysis" / "results" / "figures"


def _family():
    """Liberation Sans (metric-compatible with Arial) when installed; Arial; else DejaVu Sans with a warning."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Liberation Sans", "Arial", "Helvetica"):
        if fam in names:
            return fam
    for d in ("/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/liberation2"):
        for f in Path(d).glob("LiberationSans-*.ttf"):
            font_manager.fontManager.addfont(str(f))
    if "Liberation Sans" in {f.name for f in font_manager.fontManager.ttflist}:
        return "Liberation Sans"
    warnings.warn("Liberation Sans/Arial not found; falling back to DejaVu Sans")
    return "DejaVu Sans"


def apply():
    fam = _family()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [fam], "mathtext.fontset": "custom",
        "mathtext.rm": fam, "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold", "mathtext.sf": fam,
        "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE, "axes.titleweight": "normal",
        "axes.linewidth": 0.6, "lines.linewidth": 0.9, "patch.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0, "axes.titlepad": 3.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.unicode_minus": True, "image.cmap": "viridis",
    })
    return fam


def panel_label(ax, letter, x=-0.02, y=1.0, fig=None):
    """Bold 8-pt panel letter at the top-left of an axes (in figure coordinates relative to the axes box)."""
    ax.text(x, y, letter, transform=ax.transAxes if fig is None else fig.transFigure, fontsize=8, fontweight="bold",
            va="bottom", ha="right")


def colour(key):
    return L.BY_KEY[key].color


def save(fig, name, show=True):
    out = []
    for d in (L.ensure(FIGS), L.ensure(RESULTS_FIGS)):
        for ext in ("pdf", "svg", "png"):
            p = d / f"{name}.{ext}"
            fig.savefig(p)
            out.append(p)
    if not show:
        plt.close(fig)
    return out
