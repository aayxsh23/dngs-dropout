#!/usr/bin/env python
"""Summarize probes.py output over the T1 seed-0 runs (PLAN_V2 §5, novelty items 1 and 3).

Example:
    python experiments/probes_summary.py --out_root results/raw/main --tables results/tables --figures results/figures

Reads <out_root>/llff_3v/<variant>/<scene>/seed0/probes/{robustness.csv,depth_bias.json}.
Writes:
    <tables>/probes_robustness_3v.csv   per variant, compensated and fraction f: mean over scenes of the PSNR change
                                        from f = 0 (eval PSNR with a random fraction f of the Gaussians removed),
                                        with a 95% t-interval over scenes
    <tables>/probes_depth_bias_3v.csv   per variant and scene: median relative shift of the hard-depth pass under
                                        compensated dropout at r = 0.12 and 0.20 (opacity-cap mechanism, condition 3)
    <figures>/fig_robustness.pdf/.png   the robustness curves, compensated and not
Needs matplotlib >= 3.3 (Colab, or any Python >= 3.8 environment), not the laptop's Python 3.7 training env.
"""
import argparse, glob, json, os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "plots"))
sys.path.insert(0, HERE)
from stats import t975  # noqa: E402

PAT = re.compile(r"llff_3v/([^/]+)/([^/]+)/seed0/probes/")


def load(out_root):
    rob, bias = [], []
    for f in sorted(glob.glob(os.path.join(out_root, "llff_3v", "*", "*", "seed0", "probes", "robustness.csv"))):
        v, s = PAT.search(f.replace(os.sep, "/")).groups()
        r = pd.read_csv(f).assign(variant=v, scene=s)
        p0 = r.loc[r["fraction"] == 0, "psnr"].iloc[0]
        rob.append(r.assign(delta=r["psnr"] - p0))
        b = json.load(open(os.path.join(os.path.dirname(f), "depth_bias.json")))
        bias.append(dict(variant=v, scene=s, shift_r012=b["r=0.12"]["median_rel_shift"],
                         shift_r020=b["r=0.20"]["median_rel_shift"], n_gaussians=b["n_gaussians"]))
    return (pd.concat(rob) if rob else pd.DataFrame()), pd.DataFrame(bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="T1 folder holding llff_3v/")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    a = ap.parse_args()
    rob, bias = load(a.out_root)
    if rob.empty:
        sys.exit("no probes/robustness.csv under %s" % a.out_root)
    rows = []
    for (v, c, f), g in rob.groupby(["variant", "compensated", "fraction"]):
        n, m, sd = len(g), g["delta"].mean(), g["delta"].std(ddof=1)
        h = t975(n - 1) * sd / n ** 0.5 if n > 1 else float("nan")
        rows.append(dict(variant=v, compensated=c, fraction=f, n_scenes=n, delta_psnr=m, ci_lo=m - h, ci_hi=m + h))
    summ = pd.DataFrame(rows)
    os.makedirs(a.tables, exist_ok=True)
    summ.round(4).to_csv(os.path.join(a.tables, "probes_robustness_3v.csv"), index=False)
    bias.to_csv(os.path.join(a.tables, "probes_depth_bias_3v.csv"), index=False)
    print(summ[summ["fraction"] == 0.2].round(3).to_string(index=False))
    print("\nhard-depth shift (%), mean over scenes:")
    print((bias.groupby("variant")[["shift_r012", "shift_r020"]].mean() * 100).round(4).to_string())
    ok = bias.assign(ok=(bias["shift_r020"] > 0) & (bias["shift_r020"] > bias["shift_r012"])).groupby("variant")["ok"]
    print("scenes with a positive shift that grows from r = 0.12 to 0.20:",
          {v: "%d/%d" % (s.sum(), len(s)) for v, s in ok})

    import style
    import matplotlib.pyplot as plt
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    for ax, comp in zip(axes, (True, False)):
        for v in style.COLORS:
            g = summ[(summ["variant"] == v) & (summ["compensated"] == comp)].sort_values("fraction")
            if g.empty:
                continue
            ax.plot(g["fraction"], g["delta_psnr"], color=style.COLORS[v], marker=style.MARKERS[v], label=style.NAMES[v])
            ax.fill_between(g["fraction"], g["ci_lo"], g["ci_hi"], color=style.COLORS[v], alpha=0.12, linewidth=0)
        ax.set_title("(%s) %s" % ("a" if comp else "b", "opacity × 1/(1−f)" if comp else "no compensation"))
        ax.set_xlabel("fraction of Gaussians removed at test time")
    axes[0].set_ylabel("ΔPSNR vs f = 0 (dB)")
    axes[1].legend(loc="lower left", frameon=False)
    fig.suptitle("Test-time robustness, T1 seed 0, mean over 8 scenes (band: 95% CI)", y=1.06)
    os.makedirs(a.figures, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.figures, "fig_robustness." + ext))
    print("wrote %s/fig_robustness.pdf and .png" % a.figures)


if __name__ == "__main__":
    main()
