#!/usr/bin/env python
"""F8: pass localization. Which depth pass carries the cost of the shared mask? (PLAN_V2 §2.3 H2, §5)

Example:
    python experiments/plots/pass_loc.py --tables results/tables

Reads stats.py's paired_<n>v.csv and paired_per_scene_<n>v.csv (stats.py run with --pass_runs).
Three bars, each a variant minus E (the mask kept off both depth passes), paired by scene, mean with 95% CI:
    D − E        the mask on both depth passes (T1, the sign-flipped primary contrast E − D)
    D_hard − E   the mask on the hard depth pass only (block P)
    D_soft − E   the mask on the soft depth pass only (block P)
H2 (the opacity cap) predicts D_hard − E ≈ D − E < 0 and D_soft − E ≈ 0. Per-scene differences are drawn as dots
and the ±δ band is shaded. The label under each bar is stats.py's class: D − E takes the primary family's
(Holm across 3, as E − D), D_hard − E and D_soft − E the pass family's (Holm across 2).

Writes <figures>/fig_pass.pdf and .png.
Needs matplotlib >= 3.3 (Colab, or any Python >= 3.8 environment), not the laptop's Python 3.7 training env.
"""
import argparse, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import style  # noqa: E402
from stats import DELTA  # noqa: E402

FLIP = {">0": "<0", "<0": ">0"}
CLASS_TEXT = {">0": "> 0", "<0": "< 0", "≈0": "≈ 0", "inconclusive": "inconclusive", "?": "too few scenes"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables", help="folder with stats.py's paired_<n>v.csv")
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--name", default="fig_pass")
    a = ap.parse_args()

    path = os.path.join(a.tables, f"paired_{a.n_views}v.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run stats.py first")
    res = pd.read_csv(path)
    res = res[res["metric"] == "psnr"].set_index("comparison")
    if not {"D_hard-E", "D_soft-E"} <= set(res.index):
        sys.exit("No pass-localization rows in %s: run stats.py with --pass_runs" % path)
    ps_path = os.path.join(a.tables, f"paired_per_scene_{a.n_views}v.csv")
    per_scene = pd.read_csv(ps_path, index_col=0) if os.path.exists(ps_path) else pd.DataFrame()

    ed = res.loc["E-D"]   # T1 primary, E − D: flip it so all three bars read "X − E"
    bars = [("D − E\n(both passes)", dict(mean=-ed["mean"], ci_lo=-ed["ci_hi"], ci_hi=-ed["ci_lo"], n=ed["n"],
                                           cls=FLIP.get(ed["class"], ed["class"])), "E − D", -1, "D"),
            ("D_hard − E\n(hard pass only)", dict(res.loc["D_hard-E"], cls=res.loc["D_hard-E"]["class"]), "D_hard − E", 1, "D"),
            ("D_soft − E\n(soft pass only)", dict(res.loc["D_soft-E"], cls=res.loc["D_soft-E"]["class"]), "D_soft − E", 1, "D")]

    style.apply()
    fig, ax = style.plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    style.margin_band(ax, DELTA["psnr"])
    for x, (label, r, col, sign, colour) in enumerate(bars):
        ax.bar(x, r["mean"], width=0.55, color=style.COLORS[colour], alpha=0.85 if x == 0 else 0.55,
               edgecolor="white", linewidth=1, zorder=2)
        if pd.notna(r["ci_lo"]):
            ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], color=style.INK,
                        capsize=3, elinewidth=1, linewidth=0, zorder=4)
        if col in per_scene:
            d = sign * per_scene[col].dropna()
            ax.plot([x + 0.18] * len(d), d, "o", color=style.INK_2, alpha=0.35, markersize=3, markeredgewidth=0, zorder=3)
        ax.annotate(f"{r['mean']:+.2f} dB, {CLASS_TEXT.get(r['cls'], r['cls'])}", xy=(x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -26), textcoords="offset points", ha="center", va="top", fontsize=6.5, color=style.INK_2)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("ΔPSNR vs E (dB)")
    ax.set_title(f"Where the shared mask costs ({int(res['n'].max())} scenes; ±δ = {DELTA['psnr']:g} dB shaded)",
                 loc="left")
    style.save(fig, a.figures, a.name)


if __name__ == "__main__":
    main()
