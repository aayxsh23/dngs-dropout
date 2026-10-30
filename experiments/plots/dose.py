#!/usr/bin/env python
"""F7: dose response of the primary contrasts (PLAN_V2 §2.2 RQ3, §5).

Example:
    python experiments/plots/dose.py --tables results/tables

Reads stats.py's dose_<n>v.csv and dose_per_scene_<n>v.csv (stats.py run with --dose_runs).
One panel per primary contrast (E − D, I_D, I_E): the mean of the per-scene differences at γ = 0.1, 0.2 and 1/3
with its 95% t-interval, one faint line per scene, and the shaded ±δ band. The x axis is the drop rate reached at
iteration 6000 (γ · 6000 / 10000: 0.06, 0.12, 0.20). Every point is seed 0, so the γ = 0.2 point is T1 seed 0, not
the 3-seed T1 mean. The depth-supervision claim needs E − D larger at γ = 1/3 than at 0.1 in ≥ 6/8 scenes; the
panel title of E − D gives that count.

Writes <figures>/fig_dose.pdf and .png.
Needs matplotlib >= 3.3 (Colab, or any Python >= 3.8 environment), not the laptop's Python 3.7 training env.
"""
import argparse, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import style  # noqa: E402
from stats import DELTA, PRIMARY  # noqa: E402

PEAK = 6000 / 10000   # D1: rate(t) = γ · t / 10000, and training stops at t = 6000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables", help="folder with stats.py's dose_<n>v.csv")
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--name", default="fig_dose")
    a = ap.parse_args()

    path = os.path.join(a.tables, f"dose_{a.n_views}v.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run stats.py with --dose_runs")
    res = pd.read_csv(path)
    res = res[res["metric"] == "psnr"]
    ps_path = os.path.join(a.tables, f"dose_per_scene_{a.n_views}v.csv")
    per_scene = pd.read_csv(ps_path) if os.path.exists(ps_path) else pd.DataFrame(columns=["gamma", "comparison", "scene"])

    style.apply()
    fig, axes = style.plt.subplots(1, len(PRIMARY), figsize=(7.0, 2.4), sharey=True, constrained_layout=True)
    for ax, (key, label, _, _) in zip(axes, PRIMARY):
        style.margin_band(ax, DELTA["psnr"])
        g = res[res["comparison"] == key].sort_values("gamma")
        s = per_scene[per_scene["comparison"] == label].pivot_table(index="scene", columns="gamma", values="psnr_delta")
        for _, row in s.iterrows():
            row = row.dropna()
            ax.plot(row.index * PEAK, row.values, color=style.INK_2, alpha=0.25, linewidth=0.8, zorder=2)
        x = g["gamma"] * PEAK
        ax.errorbar(x, g["mean"], yerr=[g["mean"] - g["ci_lo"], g["ci_hi"] - g["mean"]], color=style.INK,
                    marker="s", markersize=4, capsize=3, elinewidth=1, linewidth=1.6, zorder=4)
        ax.set_title(label.split(" = ")[0], loc="left")
        if key == "E-D" and len(s.columns) >= 2:   # lowest vs highest dose: 0.1 vs 1/3 (mechanism condition 3)
            lo, hi = s.columns.min(), s.columns.max()
            both = s[[lo, hi]].dropna()
            ax.text(0.98, 0.04, f"larger at γ = 1/3 than at 0.1\nin {int((both[hi] > both[lo]).sum())}/{len(both)} scenes",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=style.INK_2)
        ax.set_xticks([gm * PEAK for gm in sorted(res["gamma"].unique())])
        ax.set_xticklabels([f"{gm * PEAK:.2f}\nγ = {gm:.2g}" for gm in sorted(res["gamma"].unique())])
        ax.set_xlabel("drop rate at iteration 6000")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("ΔPSNR (dB), seed 0")
    n = int(res["n"].max()) if len(res) else 0
    fig.suptitle(f"Primary contrasts vs dropout dose ({n} scenes, 95% CI; ±δ = {DELTA['psnr']:g} dB shaded; "
                 "lines: scenes)", x=0.01, ha="left", fontsize=9)
    style.save(fig, a.figures, a.name)


if __name__ == "__main__":
    main()
