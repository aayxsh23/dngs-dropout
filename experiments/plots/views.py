#!/usr/bin/env python
"""F5: PSNR vs number of training views for B, C, D and E (Tier 2, PLAN §7.1).

Example:
    python experiments/plots/views.py --out_root <drive>/sparse3dgs/outputs/laptop

Reads every metrics.json under <out_root>/llff_<n>v/ (same layout as Tier 1) for one seed.
Panel (a): mean PSNR over scenes per variant and view count.
Panel (b): X − B per scene, paired, mean with 95% t-interval, for C, D and E: the effect of each
           regularizer relative to DNGaussian alone as the number of views grows.
At each view count only scenes that have all the plotted variants are used, so every point is paired.

Writes <figures>/fig_views.pdf and .png and <tables>/views_summary.csv.
Needs matplotlib >= 3.3 (Colab, or any Python >= 3.8 environment), not the laptop's Python 3.7 training env.
"""
import argparse, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import style  # noqa: E402
from aggregate import load_runs  # noqa: E402
from stats import paired  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=["B", "C", "D", "E"])
    ap.add_argument("--reference", default="B", help="panel (b) plots X − reference")
    ap.add_argument("--metric", default="psnr", choices=["psnr", "ssim", "ssim_sk", "lpips"])
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--name", default="fig_views")
    a = ap.parse_args()

    runs = load_runs(a.out_root)
    if runs.empty:
        sys.exit(f"No metrics.json under {a.out_root}")
    runs = runs[(runs["seed"] == a.seed) & runs["variant"].isin(a.variants)]
    if a.metric not in runs:
        sys.exit(f"No run has {a.metric}")

    rows = []
    for nv, g in runs.groupby("n_views"):
        t = g.pivot_table(index="scene", columns="variant", values=a.metric, aggfunc="mean")
        have = [v for v in a.variants if v in t]
        t = t[have].dropna()
        lacking = sorted(set(g["scene"]) - set(t.index))
        if lacking:
            print(f"NOTE {nv}-view: scenes {lacking} lack some variant, left out of this view count", file=sys.stderr)
        for v in have:
            r = dict(n_views=nv, variant=v, n_scenes=len(t), mean=t[v].mean() if len(t) else float("nan"))
            if v != a.reference and a.reference in t and len(t):
                p = paired(t[v] - t[a.reference], a.metric != "lpips")
                r.update(delta=p["mean"], ci_lo=p["ci_lo"], ci_hi=p["ci_hi"], wins=p["wins"])
            rows.append(r)
    s = pd.DataFrame(rows)
    if s.empty:
        sys.exit("Nothing to plot")
    os.makedirs(a.tables, exist_ok=True)
    s.round(4).to_csv(os.path.join(a.tables, "views_summary.csv"), index=False)
    print(s.round(3).to_string(index=False))

    style.apply()
    fig, (ax1, ax2) = style.plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    views = sorted(s["n_views"].unique())
    items1, items2 = [], []
    for v in a.variants:
        g = s[s["variant"] == v].sort_values("n_views")
        if g.empty:
            continue
        kw = dict(color=style.COLORS[v], marker=style.MARKERS[v], markersize=5, label=style.NAMES[v])
        ax1.plot(g["n_views"], g["mean"], **kw)
        items1.append((g["n_views"].iloc[-1], g["mean"].iloc[-1], v))
        if v != a.reference and "delta" in g:
            g = g.dropna(subset=["delta"])
            others = [x for x in a.variants if x != a.reference]
            dodge = 0.12 * (others.index(v) - (len(others) - 1) / 2)   # so the CIs at one view count don't overlap
            err = [g["delta"] - g["ci_lo"].fillna(g["delta"]), g["ci_hi"].fillna(g["delta"]) - g["delta"]]
            ax2.errorbar(g["n_views"] + dodge, g["delta"], yerr=err, capsize=2, elinewidth=0.8, **kw)
            if len(g):
                items2.append((g["n_views"].iloc[-1] + dodge, g["delta"].iloc[-1], v))
    ax2.axhline(0, color=style.INK_2, linewidth=0.8)
    label = a.metric.upper()
    n_txt = "/".join(str(int(x)) for x in s.groupby("n_views")["n_scenes"].max())
    ax1.set_title(f"(a) mean {label} over scenes (seed {a.seed})", loc="left")
    ax2.set_title(f"(b) X − {a.reference}, paired by scene, 95% CI", loc="left")
    for ax, ylab in ((ax1, label), (ax2, f"Δ{label} vs {a.reference}")):
        ax.set_xlabel(f"training views (scenes per point: {n_txt})")
        ax.set_ylabel(ylab)
        ax.set_xticks(views)
    style.end_labels(ax1, items1)
    style.end_labels(ax2, items2)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False)
    style.save(fig, a.figures, a.name)


if __name__ == "__main__":
    main()
