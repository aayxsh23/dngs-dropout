#!/usr/bin/env python
"""F6: forest plot of every T1 and pass contrast (PLAN_V2 §5, §9).

Example:
    python experiments/stats.py --runs results/tables/runs.csv --tables results/tables ...
    python experiments/plots/forest.py --tables results/tables

Reads stats.py's paired_<n>v.csv (and, for PSNR, paired_per_scene_<n>v.csv).
One row per contrast, grouped as primary (Holm across the three), secondary and pass localization (Holm across
the two): the mean of the per-scene differences with its 95% t-interval over scenes, the per-scene differences as
faint dots, and the shaded ±δ band ("equivalent to 0" when the whole CI lies inside it). The right margin prints
stats.py's class for each row, so the figure and the verdict table can't disagree.

Writes <figures>/fig_forest.pdf and .png.
Needs matplotlib >= 3.3 (Colab, or any Python >= 3.8 environment), not the laptop's Python 3.7 training env.
"""
import argparse, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import style  # noqa: E402
from stats import DELTA, HIGHER_IS_BETTER  # noqa: E402

FAMILIES = [("primary", "Primary (Holm across 3)"), ("secondary", "Secondary"), ("pass", "Pass localization (Holm across 2)")]
CLASS_TEXT = {">0": "> 0", "<0": "< 0", "≈0": "≈ 0 (within ±δ)", "inconclusive": "inconclusive", "?": "too few scenes"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables", help="folder with stats.py's paired_<n>v.csv")
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--metric", default="psnr", choices=["psnr", "ssim", "lpips"])
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--name", default="fig_forest")
    a = ap.parse_args()

    path = os.path.join(a.tables, f"paired_{a.n_views}v.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run stats.py first")
    res = pd.read_csv(path)
    res = res[res["metric"] == a.metric]
    if res.empty:
        sys.exit(f"No {a.metric} rows in {path}")
    per_scene = None
    ps_path = os.path.join(a.tables, f"paired_per_scene_{a.n_views}v.csv")
    if a.metric == "psnr" and os.path.exists(ps_path):   # stats.py writes per-scene deltas for PSNR only
        per_scene = pd.read_csv(ps_path, index_col=0)

    rows = []   # (y label, result row or None for a group header)
    for fam, title in FAMILIES:
        g = res[res["family"] == fam]
        if g.empty:
            continue
        rows.append((title, None))
        rows += [(r["label"], r) for _, r in g.iterrows()]

    style.apply()
    fig, ax = style.plt.subplots(figsize=(5.2, 0.24 * len(rows) + 0.9), constrained_layout=True)
    style.margin_band(ax, DELTA.get(a.metric), axis="x")
    ys = list(range(len(rows)))[::-1]
    labels = []
    for y, (label, r) in zip(ys, rows):
        if r is None:
            labels.append(label)
            continue
        labels.append("   " + label)
        if per_scene is not None and r["label"] in per_scene:
            d = per_scene[r["label"]].dropna()
            ax.plot(d, [y] * len(d), "o", color=style.INK_2, alpha=0.25, markersize=3, markeredgewidth=0, zorder=2)
        if pd.notna(r["ci_lo"]):
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=style.INK, linewidth=1.4, zorder=3)
        decided = r["class"] in (">0", "<0", "≈0")
        ax.plot(r["mean"], y, "s", markersize=5, zorder=4, color=style.INK,
                markerfacecolor=style.INK if decided else "white")
        cls = CLASS_TEXT.get(r["class"], r["class"])
        if r.get("zero_excluded") is True or str(r.get("zero_excluded")) == "True":
            cls += ", CI excludes 0"
        ax.annotate(f"{cls}; first term better in {int(r['wins'])}/{int(r['n'])}", xy=(1.01, y), xycoords=("axes fraction", "data"),
                    va="center", fontsize=6.5, color=style.INK_2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    for t, (_, r) in zip(ax.get_yticklabels(), rows):
        if r is None:
            t.set_fontweight("bold")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    unit = " (dB)" if a.metric == "psnr" else ""
    better = "higher is better" if HIGHER_IS_BETTER[a.metric] else "lower is better"
    delta = DELTA.get(a.metric)
    ax.set_xlabel(f"Δ{a.metric.upper()}{unit}, mean of per-scene differences, 95% CI ({better})"
                  + (f"; band ±δ = {delta:g}" if delta else ""))
    n = int(res["n"].max())
    ax.set_title(f"{a.n_views}-view contrasts, paired over {n} scenes (filled square: decided; dots: scenes)",
                 loc="left")
    style.save(fig, a.figures, a.name if a.metric == "psnr" else f"{a.name}_{a.metric}")


if __name__ == "__main__":
    main()
