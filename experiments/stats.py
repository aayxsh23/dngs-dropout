#!/usr/bin/env python
"""Paired comparisons, interaction contrasts and the verdict of PLAN §6, from runs.csv.

Example:
    python experiments/aggregate.py --out_root <drive>/sparse3dgs/outputs --tables results/tables
    python experiments/plots/diag_curves.py --out_root <drive>/sparse3dgs/outputs     # optional, for the mechanism test
    python experiments/stats.py --runs results/tables/runs.csv --diag_summary results/tables/diag_summary_3v.csv

Method (PLAN §6, fixed before results):
    For each variant and scene, average over seeds, giving one value per scene. Every comparison is paired
    by scene: delta = X - Y per scene, mean delta, 95% t-interval over the scenes (df = n - 1), and wins.
    A "win" means the comparison favours its first term: delta > 0 for PSNR and SSIM, delta < 0 for LPIPS.
    best(B, C) is whichever of B and C has the higher mean PSNR; the same choice is used for every metric.
    Interaction contrasts: I_D = (D - C) - (B - A) and I_E = (E - C) - (B - A).

Writes to --tables:
    scene_means_<n>v.csv        variant x scene, mean over seeds, for every metric (what the tests use)
    paired_<n>v.csv/.md/.tex    one row per comparison and metric: n, mean, sd, 95% CI, wins
    paired_per_scene_<n>v.csv   the per-scene PSNR deltas (to pick failure cases, PLAN §7.2)
    verdict_<n>v.txt            the §6 verdict, rule by rule
"""
import argparse, math, os, sys
import pandas as pd

METRICS = ["psnr", "ssim", "lpips"]
HIGHER_IS_BETTER = dict(psnr=True, ssim=True, ssim_sk=True, lpips=False)
VARIANTS = ["A", "B", "C", "D", "E"]
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
DIGITS = dict(psnr=2, ssim=3, ssim_sk=3, lpips=3)

# two-sided 95% t quantiles, used when scipy is not installed (df 1..30)
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]


def t975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        return T975[df - 1] if df <= len(T975) else 1.96


def comparisons(best):
    """(key, label, question, function of the scene x variant table)."""
    return [
        ("B-A", "B − A", "does depth help?", lambda t: t["B"] - t["A"]),
        ("C-A", "C − A", "does dropout help?", lambda t: t["C"] - t["A"]),
        ("D-best", f"D − {best}", "naive combination vs the better single regularizer", lambda t: t["D"] - t[best]),
        ("E-best", f"E − {best}", "decoupled combination vs the better single regularizer", lambda t: t["E"] - t[best]),
        ("E-D", "E − D", "does decoupling help?", lambda t: t["E"] - t["D"]),
        ("I_D", "I_D = (D − C) − (B − A)", "does depth help more or less under dropout?",
         lambda t: (t["D"] - t["C"]) - (t["B"] - t["A"])),
        ("I_E", "I_E = (E − C) − (B − A)", "the same, decoupled", lambda t: (t["E"] - t["C"]) - (t["B"] - t["A"])),
    ]


def paired(d, higher_is_better=True):
    d = d.dropna()
    n = len(d)
    row = dict(n=n, mean=d.mean() if n else float("nan"), sd=float("nan"), ci_lo=float("nan"),
               ci_hi=float("nan"), wins=int((d > 0).sum() if higher_is_better else (d < 0).sum()))
    if n >= 2:
        row["sd"] = d.std(ddof=1)
        half = t975(n - 1) * row["sd"] / math.sqrt(n)
        row["ci_lo"], row["ci_hi"] = row["mean"] - half, row["mean"] + half
    return row


def sign(r):
    """'+' if the CI is entirely above 0, '-' if entirely below, '0' if it contains 0, '?' if undefined."""
    if pd.isna(r["ci_lo"]):
        return "?"
    return "+" if r["ci_lo"] > 0 else "-" if r["ci_hi"] < 0 else "0"


def scene_means(runs, n_views, metrics):
    g = runs[runs["n_views"] == n_views]
    seeds = g.groupby(["scene", "variant"])["seed"].nunique().unstack("variant")
    uneven = seeds[seeds.nunique(axis=1) > 1]
    if not uneven.empty:
        print("WARNING these scenes have different numbers of seeds per variant (means still taken):\n"
              + uneven.to_string(), file=sys.stderr)
    return g.groupby(["variant", "scene"])[metrics].mean(), seeds


def fmt_ci(r, digits):
    if pd.isna(r["mean"]):
        return "–"
    if pd.isna(r["ci_lo"]):
        return f"{r['mean']:+.{digits}f}"
    return f"{r['mean']:+.{digits}f} [{r['ci_lo']:+.{digits}f}, {r['ci_hi']:+.{digits}f}]"


def write_tables(res, comps, metrics, n_scenes, base):
    res.to_csv(base + ".csv", index=False)
    head = ["Comparison"] + [f"Δ{m.upper()} [95% CI]" for m in metrics] + [f"PSNR wins /{n_scenes}"]
    rows = []
    for key, label, _, _ in comps:
        cells = [label]
        for m in metrics:
            r = res[(res["comparison"] == key) & (res["metric"] == m)]
            cells.append(fmt_ci(r.iloc[0], DIGITS.get(m, 3)) if len(r) else "–")
        r = res[(res["comparison"] == key) & (res["metric"] == "psnr")]
        cells.append(str(int(r.iloc[0]["wins"])) if len(r) else "–")
        rows.append(cells)
    md = ["| " + " | ".join(head) + " |", "| --- " * len(head) + "|"] + ["| " + " | ".join(c) + " |" for c in rows]
    open(base + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{l" + "c" * (len(head) - 1) + "}", "\\toprule",
           " & ".join(["Comparison"] + [f"$\\Delta$ {m.upper()} [95\\% CI]" for m in metrics]
                      + [f"PSNR wins /{n_scenes}"]) + " \\\\", "\\midrule"]
    for c in rows:
        tex.append(" & ".join([c[0].replace("−", "$-$").replace("I_D", "$I_D$").replace("I_E", "$I_E$")]
                              + [x.replace("−", "$-$") for x in c[1:]]) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(base + ".tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")


def mechanism(diag_summary, n_views):
    """PLAN §6 mechanism test on diag_curves.py's per-scene summary. Returns (lines of text, whether it holds)."""
    s = pd.read_csv(diag_summary)
    if "n_views" in s:
        s = s[s["n_views"] == n_views]
    gap = s.pivot_table(index="scene", columns="variant", values="gap_mean", aggfunc="mean")
    out = []
    if "D" not in gap:
        return ["  no diag.csv summary for D yet"], False
    d = paired(gap["D"])
    out.append(f"  gap(D) = mean of loss_depth_used − loss_depth_clean over iterations ≥ 1000: "
               f"{fmt_ci(d, 4)}, CI sign {sign(d)}, n = {d['n']}")
    dc = None
    if "C" in gap:
        dc = paired(gap["D"] - gap["C"])
        out.append(f"  gap(D) − gap(C), paired by scene: {fmt_ci(dc, 4)}, CI sign {sign(dc)}")
    holds = sign(d) == "+" and dc is not None and sign(dc) == "+"
    out.append("  → dropout perturbs the depth signal in D, more than in C: " + ("YES" if holds else "not shown"))
    return out, holds


def verdict(res, best, mech):
    def r(key, m="psnr"):
        x = res[(res["comparison"] == key) & (res["metric"] == m)]
        return x.iloc[0] if len(x) else None

    def worse(key, m):
        x = r(key, m)
        if x is None or pd.isna(x["ci_lo"]):
            return False
        return x["ci_hi"] < 0 if HIGHER_IS_BETTER[m] else x["ci_lo"] > 0

    lines = [f"best(B, C) = {best} (higher mean PSNR)", ""]
    for key, label, _, _ in comparisons(best):
        x = r(key)
        lines.append(f"  PSNR {label:<26} {fmt_ci(x, 2):<26} CI sign {sign(x)}   wins {int(x['wins'])}/{x['n']}")
    lines.append("")

    comp, interf = {}, {}
    for X in ["D", "E"]:
        main, inter = r(f"{X}-best"), r(f"I_{X}")
        guards_ok = not worse(f"{X}-best", "ssim") and not worse(f"{X}-best", "lpips")
        comp[X] = sign(main) == "+" and sign(inter) != "-" and guards_ok
        interf[X] = sign(inter) == "-" and sign(main) == "-"
        lines.append(f"{X}: complementary rule {'MET' if comp[X] else 'not met'}"
                     f" ({X} − best CI {sign(main)}, I_{X} CI {sign(inter)}, SSIM/LPIPS "
                     f"{'not significantly worse' if guards_ok else 'significantly worse'});"
                     f" interfering rule {'MET' if interf[X] else 'not met'}")

    if comp["D"] or comp["E"]:
        v = "COMPLEMENTARY (via " + " and ".join(X for X in "DE" if comp[X]) + ")"
    elif interf["D"]:
        v = "INTERFERING (D)" + (" and E" if interf["E"] else "")
    elif sign(r("D-best")) == "0" and sign(r("E-best")) == "0":
        v = "REDUNDANT (neither combination differs detectably from best(B, C))"
    else:
        v = "NOT COVERED by the §6 rules: describe the pattern above instead of forcing a label"
    lines += ["", "VERDICT: " + v]

    if sign(r("C-A")) == "-" and sign(r("I_D")) == "0":
        lines.append("Note: dropout hurts on its own (C − A CI below 0) while I_D ≈ 0. By §6 that is NOT interference.")
    if sign(r("E-D")) == "+":
        lines.append("Note: decoupling helps (E − D CI above 0).")
    if mech is not None:
        mtext, holds = mech
        lines += ["", "Mechanism (diag.csv):"] + mtext
        if holds and sign(r("E-D")) == "+":
            lines.append("  → and E − D CI above 0: the interference runs through depth supervision.")
    lines += ["", "Power caveat: run-to-run noise is about 0.3–0.5 dB per run; differences much below "
                  "about 0.2 dB are unlikely to be resolvable with 8 scenes."]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/tables/runs.csv", help="written by aggregate.py")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--diag_summary", default=None, help="diag_summary_<n>v.csv from plots/diag_curves.py (optional)")
    a = ap.parse_args()

    runs = pd.read_csv(a.runs)
    metrics = [m for m in METRICS if m in runs]
    if "psnr" not in metrics:
        sys.exit(f"{a.runs} has no psnr column")
    means, seeds = scene_means(runs, a.n_views, metrics)
    if means.empty:
        sys.exit(f"No {a.n_views}-view runs in {a.runs}")
    os.makedirs(a.tables, exist_ok=True)
    wide = means.unstack("variant")
    wide.round(4).to_csv(os.path.join(a.tables, f"scene_means_{a.n_views}v.csv"))

    absent = [v for v in VARIANTS if v not in means.index.get_level_values("variant")]
    if absent:
        sys.exit(f"Variants {absent} have no {a.n_views}-view runs yet; every comparison needs all five")
    psnr = means["psnr"].unstack("variant")[VARIANTS]
    complete = psnr.dropna().index.tolist()
    if len(complete) < len(SCENES):
        print(f"NOTE paired over {len(complete)} scenes with all five variants: {complete}", file=sys.stderr)
    if len(complete) < 2:
        sys.exit("Need at least 2 scenes with all five variants for a confidence interval")
    print("seeds per scene and variant:\n" + seeds.loc[complete].to_string() + "\n")

    best = "B" if psnr.loc[complete, "B"].mean() >= psnr.loc[complete, "C"].mean() else "C"
    comps = comparisons(best)
    rows, per_scene = [], {}
    for m in metrics:
        t = means[m].unstack("variant").loc[complete, VARIANTS]
        for key, label, question, f in comps:
            d = f(t)
            if m == "psnr":
                per_scene[label] = d
            rows.append(dict(metric=m, comparison=key, label=label, question=question,
                             **paired(d, HIGHER_IS_BETTER[m])))
    res = pd.DataFrame(rows)
    base = os.path.join(a.tables, f"paired_{a.n_views}v")
    write_tables(res, comps, metrics, len(complete), base)
    pd.DataFrame(per_scene).round(3).to_csv(os.path.join(a.tables, f"paired_per_scene_{a.n_views}v.csv"))

    mech = mechanism(a.diag_summary, a.n_views) if a.diag_summary else None
    lines = verdict(res, best, mech)
    open(os.path.join(a.tables, f"verdict_{a.n_views}v.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(open(base + ".md", encoding="utf-8").read())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
