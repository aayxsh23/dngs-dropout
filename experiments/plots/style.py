"""Shared look for the report's figures: one fixed colour and marker per variant, recessive axes.

Colours are the first five slots of a categorical palette checked for colour-blind separation
(dataviz validate_palette.js, light mode: all checks pass). Three of them are below 3:1 contrast
on white, so every line also gets its own marker and a direct label, never colour alone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = dict(A="#2a78d6", B="#eb6834", C="#1baf7a", D="#eda100", E="#e87ba4")   # fixed per variant, never re-ordered
MARKERS = dict(A="o", B="s", C="^", D="D", E="v")
NAMES = dict(A="A: none", B="B: depth", C="C: dropout", D="D: depth + dropout (shared)", E="E: depth + dropout (clean)")
INK, INK_2, GRID = "#1f1f1e", "#5f5e58", "#e4e3df"


def apply():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def end_labels(ax, items, min_gap=0.07):
    """Direct labels at the right end of lines, in text ink (the line beside each carries the colour).
    items: [(x, y, text)]. Labels are nudged apart vertically so they never overlap."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    items = sorted(items, key=lambda t: t[1])
    ys = [items[0][1]]
    for _, y, _ in items[1:]:
        ys.append(max(y, ys[-1] + gap))
    shift = sum(y for _, y, _ in items) / len(items) - sum(ys) / len(ys)   # re-centre on the lines
    ys = [y + shift for y in ys]
    for (x, _, text), yl in zip(items, ys):
        ax.text(x, yl, "  " + text, va="center", ha="left", fontsize=7, color=INK, clip_on=False)


def save(fig, figures_dir, name):
    import os
    os.makedirs(figures_dir, exist_ok=True)
    paths = [os.path.join(figures_dir, name + ext) for ext in (".pdf", ".png")]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    print("wrote", " and ".join(paths))
