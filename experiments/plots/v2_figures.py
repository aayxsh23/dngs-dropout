#!/usr/bin/env python
"""Figures for the v1.1/v1.2 blocks (EXECUTION_PLAN D10), from the tables of the results-v2 analysis.

    python experiments/plots/v2_figures.py --tables <tables root> --figures results/figures
    python experiments/plots/v2_figures.py --self_test

<tables root> is the folder the results-v2 notebook writes (results_v2_tables.zip, `tables/`):
    error_budget/error_budget_<ds>_<n>v.csv   (bias_variance.py)
    pooled/pooled.csv                         (stats.py --pooled)
    extra/{comp,depthdose,power}.csv          (extra_blocks.py)
Each figure is skipped with a note when its input is missing, so a partial run still plots what it can.

F1 fig_budget       error budget: bias² + variance stacked per variant, one panel per dataset (pixel MSE)
F2 fig_budget_views bias² and variance of A, B, C, E against the number of LLFF views (BV4)
F3 fig_pooled       D − B, E − B, B − A, C − A, I_D, I_E, E − D per dataset and DerSimonian–Laird pooled (PSNR)
F5 fig_ceiling      BV3: mse(D) against bias²(B), one point per scene, every dataset, with the y = x line
F7 fig_power        minimum detectable effect against scenes, for 1, 3 and 5 runs per scene (B − A and I_D)
F8 fig_depthdose    I_D at depth_scale 0.5 / 1 / 2 with 95 % CIs and the ±δ band (N5)
F9 fig_comp         compensation contrasts C_nc − C, D_nc − D, C_ca − C, D_ca − D (PSNR, ±δ band, Holm class)
"""
import argparse, os, sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DATASETS = [("llff", 3, "LLFF 3-view"), ("dtu", 3, "DTU 3-view (masked)"), ("blender", 8, "Blender 8-view")]
DS_COLOR = dict(llff="#2a78d6", dtu="#eb6834", blender="#1baf7a", pooled="#1f1f1e")
DS_MARK = dict(llff="o", dtu="s", blender="^", pooled="D")
DELTA = 0.25


def read(path):
    if not os.path.exists(path):
        print("skip: no " + path)
        return None
    return pd.read_csv(path)


def f1_budget(T, F):
    panels = [(ds, nv, title, read(os.path.join(T, "error_budget", "error_budget_{}_{}v.csv".format(ds, nv))))
              for ds, nv, title in DATASETS]
    panels = [p for p in panels if p[3] is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels) + 0.6, 2.6), squeeze=False)
    for ax, (ds, nv, title, b) in zip(axes[0], panels):
        m = b.groupby("variant")[["bias2", "var", "mse"]].mean().reindex(list("ABCDE")).dropna(how="all")
        x = range(len(m))
        ax.bar(x, m["bias2"], color=[style.COLORS[v] for v in m.index], width=0.65, label="bias²")
        ax.bar(x, m["var"], bottom=m["bias2"], color="white", edgecolor=[style.COLORS[v] for v in m.index],
               hatch="////", width=0.65, linewidth=0.8, label="variance")
        for i, (v, r) in enumerate(m.iterrows()):
            ax.text(i, r["mse"], " {:.0f}%".format(100 * r["var"] / r["mse"]), ha="center", va="bottom", fontsize=6,
                    color=style.INK_2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(m.index)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("pixel MSE (mean over scenes)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend([h[0], h[1]], ["bias² (solid)", "variance (hatched)"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Error budget: solid = squared bias, hatched = run-to-run variance (label: variance share)",
                 fontsize=8, y=1.02)
    style.save(fig, F, "fig_budget")


def f2_views(T, F):
    rows = []
    for nv in (2, 3, 6, 9):
        b = read(os.path.join(T, "error_budget", "error_budget_llff_{}v.csv".format(nv)))
        if b is not None:
            g = b.groupby("variant")[["bias2", "var"]].mean()
            g["views"] = nv
            rows.append(g.reset_index())
    if len(rows) < 2:
        return
    d = pd.concat(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw=dict(wspace=0.45))
    for ax, q, title in zip(axes, ("bias2", "var"), ("squared bias", "run-to-run variance")):
        for v in "ABCE":
            g = d[d["variant"] == v].sort_values("views")
            if not g.empty:
                ax.plot(g["views"], g[q], marker=style.MARKERS[v], color=style.COLORS[v], markersize=4)
        style.end_labels(ax, [(d["views"].max(), d[(d["variant"] == v) & (d["views"] == d["views"].max())][q].iloc[0],
                               v) for v in "ABCE" if not d[(d["variant"] == v) & (d["views"] == d["views"].max())].empty])
        ax.set_xticks([2, 3, 6, 9])
        ax.set_xlabel("training views (LLFF)")
        ax.set_title(title)
    axes[0].set_ylabel("pixel MSE component")
    style.save(fig, F, "fig_budget_views")


def forest(ax, rows, ylabels, delta=DELTA, color=None, marker=None):
    for i, r in enumerate(rows):
        c = color(r) if color else style.INK
        ax.errorbar(r["mean"], -i, xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt=marker(r) if marker
                    else "o", color=c, markersize=4, capsize=2, linewidth=1.2)
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels(ylabels)
    style.margin_band(ax, delta, axis="x")
    ax.grid(axis="y", visible=False)


def f3_pooled(T, F):
    p = read(os.path.join(T, "pooled", "pooled.csv"))
    if p is None:
        return
    p = p[p["metric"] == "psnr"]
    order = ["D-B", "E-B", "B-A", "C-A", "I_D", "I_E", "E-D"]
    levels = [x for x in ("llff", "dtu", "blender", "pooled") if x in set(p["level"])]
    rows, labels = [], []
    for c in order:
        for lv in levels:
            g = p[(p["comparison"] == c) & (p["level"] == lv)]
            if not g.empty:
                rows.append(g.iloc[0])
                labels.append("{}  {}".format(c.replace("-", " − "), lv) if lv == levels[0] else lv)
    fig, ax = plt.subplots(figsize=(4.4, 0.18 * len(rows) + 0.8))
    forest(ax, rows, labels, color=lambda r: DS_COLOR[r["level"]], marker=lambda r: DS_MARK[r["level"]])
    ax.set_xlabel("PSNR difference (dB), 95 % CI; shaded: ±0.25 dB")
    ax.set_title("Each contrast per dataset and pooled (DerSimonian–Laird)")
    style.save(fig, F, "fig_pooled")


def f5_ceiling(T, F):
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    any_ = False
    lo, hi = 1e9, 0
    for ds, nv, title in DATASETS:
        b = read(os.path.join(T, "error_budget", "error_budget_{}_{}v.csv".format(ds, nv)))
        if b is None:
            continue
        t = b.pivot(index="scene", columns="variant", values="mse")
        tb = b.pivot(index="scene", columns="variant", values="bias2")
        if "D" not in t or "B" not in tb:
            continue
        x, y = tb["B"], t["D"]
        ax.scatter(x, y, s=14, color=DS_COLOR[ds], marker=DS_MARK[ds], label=title)
        lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
        any_ = True
    if not any_:
        plt.close(fig)
        return
    ax.plot([lo, hi], [lo, hi], color=style.INK_2, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bias²(B): DNGaussian's variance-free error")
    ax.set_ylabel("MSE(D): naive combination")
    ax.set_title("BV3: below the line = D beats B's bias floor")
    ax.legend(frameon=False, loc="upper left")
    style.save(fig, F, "fig_ceiling")


def f7_power(T, F):
    p = read(os.path.join(T, "extra", "power.csv"))
    if p is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    for ax, c in zip(axes, ("B-A", "I_D")):
        for R, ls in ((1, ":"), (3, "-"), (5, "--")):
            g = p[(p["contrast"] == c) & (p["runs"] == R)]
            ax.plot(g["scenes"], g["mde"], ls, marker="o", markersize=3, color=style.INK, label="{} run{}".format(
                R, "s" if R > 1 else ""))
        ax.axhline(DELTA, color=style.COLORS["B"], linewidth=0.8)
        ax.text(31, DELTA, " δ", va="center", fontsize=7, color=style.COLORS["B"])
        ax.set_xticks([8, 15, 23, 31])
        ax.set_xlabel("scenes")
        ax.set_title("{} (80 % power, α = 0.05)".format(c.replace("-", " − ")))
    axes[0].set_ylabel("minimum detectable effect (dB)")
    axes[0].legend(frameon=False)
    style.save(fig, F, "fig_power")


def f8_dose(T, F):
    d = read(os.path.join(T, "extra", "depthdose.csv"))
    if d is None:
        return
    i = d[d["comparison"] == "I_D"].sort_values("dose")
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.errorbar(i["dose"], i["mean"], yerr=[i["mean"] - i["ci_lo"], i["ci_hi"] - i["mean"]], fmt="o-",
                color=style.COLORS["D"], capsize=3, markersize=4)
    for _, r in i.iterrows():
        ax.text(r["dose"], r["ci_hi"], " " + str(r["class"]), fontsize=6, va="bottom", ha="center", color=style.INK_2)
    style.margin_band(ax, DELTA)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2])
    ax.set_xticklabels(["0.5", "1", "2"])
    ax.set_xlabel("depth_scale")
    ax.set_ylabel("I_D (dB), 95 % CI")
    ax.set_title("N5: interaction across depth weights")
    style.save(fig, F, "fig_depthdose")


def f9_comp(T, F):
    c = read(os.path.join(T, "extra", "comp.csv"))
    if c is None:
        return
    c = c[c["metric"] == "psnr"]
    rows = [r for _, r in c.iterrows()]
    fig, ax = plt.subplots(figsize=(4.0, 1.6))
    forest(ax, rows, [r["comparison"].replace("-", " − ") for r in rows],
           color=lambda r: style.COLORS[r["comparison"][0]], marker=lambda r: style.MARKERS[r["comparison"][0]])
    for i, r in enumerate(rows):
        ax.text(1.01, -i, "{}{}".format(r["class"], ", equiv." if r.get("equivalent") is True
                                                              or str(r.get("equivalent")) == "True" else ""),
                transform=ax.get_yaxis_transform(), fontsize=6, va="center", color=style.INK_2)
    ax.set_xlabel("PSNR difference (dB), 95 % CI; shaded: ±0.25 dB")
    ax.set_title("Compensation: no rescaling (nc) and clamp-aware (ca) vs uniform")
    style.save(fig, F, "fig_comp")


def run(T, F):
    style.apply()
    for f in (f1_budget, f2_views, f3_pooled, f5_ceiling, f7_power, f8_dose, f9_comp):
        f(T, F)


def self_test():
    import random
    import tempfile
    rnd = random.Random(1)
    T = tempfile.mkdtemp()
    os.makedirs(os.path.join(T, "error_budget"))
    os.makedirs(os.path.join(T, "pooled"))
    os.makedirs(os.path.join(T, "extra"))
    for ds, nv in [("llff", 2), ("llff", 3), ("llff", 6), ("llff", 9), ("dtu", 3), ("blender", 8)]:
        rows = []
        for s in range(8):
            for v in "ABCDE":
                b2, var = 0.004 + 0.001 * rnd.random(), 0.001 * rnd.random()
                rows.append(dict(scene="s%d" % s, variant=v, bias2=b2, var=var, mse=b2 + var))
        pd.DataFrame(rows).to_csv(os.path.join(T, "error_budget", "error_budget_{}_{}v.csv".format(ds, nv)),
                                  index=False)
    pr = []
    for c in ["D-B", "E-B", "B-A", "C-A", "I_D", "I_E", "E-D"]:
        for lv in ("llff", "dtu", "blender", "pooled"):
            m = rnd.uniform(-0.5, 0.8)
            pr.append(dict(level=lv, metric="psnr", comparison=c, mean=m, ci_lo=m - 0.3, ci_hi=m + 0.3))
    pd.DataFrame(pr).to_csv(os.path.join(T, "pooled", "pooled.csv"), index=False)
    pd.DataFrame([dict(contrast=c, scenes=s, runs=r, mde=1.0 / s ** 0.5 + 0.1 / r) for c in ("B-A", "I_D")
                  for s in (8, 15, 23, 31) for r in (1, 3, 5)]).to_csv(os.path.join(T, "extra", "power.csv"), index=False)
    pd.DataFrame([dict(dose=s, comparison="I_D", mean=0.05, ci_lo=-0.2, ci_hi=0.3, **{"class": "inconclusive"})
                  for s in (0.5, 1.0, 2.0)]).to_csv(os.path.join(T, "extra", "depthdose.csv"), index=False)
    pd.DataFrame([dict(metric="psnr", comparison=c, mean=0.1, ci_lo=-0.1, ci_hi=0.3, equivalent=True,
                       **{"class": "≈0"}) for c in ("C_nc-C", "D_nc-D", "C_ca-C", "D_ca-D")]).to_csv(
        os.path.join(T, "extra", "comp.csv"), index=False)
    F = os.path.join(T, "fig")
    run(T, F)
    names = sorted(os.listdir(F))
    want = ["fig_budget", "fig_budget_views", "fig_ceiling", "fig_comp", "fig_depthdose", "fig_pooled", "fig_power"]
    missing = [n for n in want if n + ".png" not in names]
    if missing:
        sys.exit("self-test: figures not written: {}".format(missing))
    print("V2-FIGURES SELF-TEST PASSED ({})".format(F))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", help="root of the results-v2 tables")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.tables:
        run(a.tables, a.figures)
    else:
        ap.error("--tables is required (or --self_test)")


if __name__ == "__main__":
    main()
