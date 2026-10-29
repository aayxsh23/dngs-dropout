#!/usr/bin/env python
"""Analysis rules v2 (PLAN_V2 §5, pre-registered): paired contrasts, Holm, the δ margin and the verdict.

Example (aggregate.py once per block, each into its own folder; only --runs, block T1, is required):
    python experiments/aggregate.py --out_root <drive>/sparse3dgs/outputs       --tables results/tables
    python experiments/aggregate.py --out_root <drive>/sparse3dgs/outputs_pass  --tables results/tables/pass
    python experiments/aggregate.py --out_root <drive>/sparse3dgs/outputs_dose  --tables results/tables/dose --seeds 0
    python experiments/aggregate.py --out_root <drive>/sparse3dgs/outputs_noise --tables results/tables/noise --seeds 0
    python experiments/plots/diag_curves.py --out_root <drive>/sparse3dgs/outputs     # for the mechanism test
    python experiments/geometry.py --out_root <drive>/sparse3dgs/outputs              # MVC@5% (optional)
    python experiments/stats.py --runs results/tables/runs.csv --diag_summary results/tables/diag_summary_3v.csv \\
        --pass_runs results/tables/pass/runs.csv --dose_runs results/tables/dose/runs.csv \\
        --noise_runs results/tables/noise/runs.csv --geometry results/tables/geometry_3v.csv

Method (PLAN_V2 §5, fixed before results):
    Unit: for each variant and scene, the mean over seeds. Every contrast is paired by scene: the mean of the
    per-scene differences, a 95% t-interval over the scenes (df = n - 1), wins (scenes where the contrast favours
    its first term: delta > 0 for PSNR, SSIM and MVC, delta < 0 for LPIPS) and a two-sided paired t-test p.
    Primary (T1): E − D, I_D = (D − C) − (B − A), I_E = (E − C) − (B − A); Holm across the three, per metric.
    Secondary (T1, uncorrected): B − A, C − A, D − B, D − C, E − B, E − C.
    Pass localization (block P with T1's E, 3 seeds): D_hard − E, D_soft − E; Holm across the two.
    Dose (block G with T1 seed 0 as anchors) and the noise floor (block N) are trends, not tests.
    Geometry: the primary contrasts on MVC@5% from geometry.py, Holm across the three.
    Each contrast gets one class, decided in this order (the order of the §5 table):
        ">0" / "<0"     the CI excludes 0 (Holm families: the Holm-adjusted p < 0.05)
        "≈0"            the 95% CI lies inside ±δ (δ = 0.25 dB for PSNR, 0.02 for MVC@5%; none for SSIM/LPIPS)
        "inconclusive"  anything else
    A ">0"/"<0" whose CI also lies inside ±δ is flagged "within ±δ".
    Verdict for X in {D, E}: the §5 table. Row, first match: X beats B and C (X − B and X − C both ">0"); X
    equivalent to B (X − B CI inside ±δ); X worse than B (X − B "<0"); otherwise. Column: the class of I_X.
    B is the fixed reference. SSIM and LPIPS must point the same way: if either has a significant class opposite
    to PSNR's on X − B, X − C or I_X, the verdict is "metric-dependent".

Writes to --tables:
    factors.csv                   the factor table: every planned variant ID, its flags, and runs found per input
    scene_means_<n>v.csv          T1 variant x scene, mean over seeds, for every metric
    paired_<n>v.csv/.md/.tex      one row per contrast and metric: family, n, mean, sd, 95% CI, wins, p, Holm p, class
    paired_per_scene_<n>v.csv     the per-scene PSNR deltas of every T1 and pass contrast
    dose_<n>v.csv                 the T1 contrasts at γ = 0.1, 0.2, 1/3, seed 0 (with --dose_runs)
    dose_per_scene_<n>v.csv       their per-scene PSNR deltas (plots/dose.py)
    noise_floor_<n>v.csv          same-seed repeat spread of B and D on fern vs the between-seed spread (--noise_runs)
    verdict_<n>v.txt              the verdict, the pass localization and the mechanism claims, rule by rule
"""
import argparse, math, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factors as F  # noqa: E402

METRICS = ["psnr", "ssim", "lpips"]
HIGHER_IS_BETTER = dict(psnr=True, ssim=True, ssim_sk=True, lpips=False, mvc5=True)
DELTA = dict(psnr=0.25, mvc5=0.02)
DIGITS = dict(psnr=2, ssim=3, ssim_sk=3, lpips=3, mvc5=3)
ALPHA = 0.05
T1_IDS, PASS_IDS = ["A", "B", "C", "D", "E"], ["D_hard", "D_soft"]
DOSE = [(0.1, "_g010"), (0.2, ""), (1 / 3, "_g033")]   # γ and the suffix of its C/D/E IDs (0.2: T1's own)
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]

# two-sided 95% t quantiles, used when scipy is not installed (df 1..30)
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]

# (key, label, question, function of a scene x variant table t and an ID map v). v maps a letter to the column
# that plays it, so the same contrast runs on T1 (v = identity) and on each dose level (v["D"] = "D_g010", ...).
PRIMARY = [
    ("E-D", "E − D", "does keeping the mask off the depth passes help?", lambda t, v: t[v["E"]] - t[v["D"]]),
    ("I_D", "I_D = (D − C) − (B − A)", "does depth help more or less under dropout?",
     lambda t, v: (t[v["D"]] - t[v["C"]]) - (t[v["B"]] - t[v["A"]])),
    ("I_E", "I_E = (E − C) − (B − A)", "the same, decoupled",
     lambda t, v: (t[v["E"]] - t[v["C"]]) - (t[v["B"]] - t[v["A"]])),
]
SECONDARY = [
    ("B-A", "B − A", "does depth help?", lambda t, v: t[v["B"]] - t[v["A"]]),
    ("C-A", "C − A", "does dropout help?", lambda t, v: t[v["C"]] - t[v["A"]]),
    ("D-B", "D − B", "naive combination vs published DNGaussian", lambda t, v: t[v["D"]] - t[v["B"]]),
    ("D-C", "D − C", "naive combination vs dropout alone", lambda t, v: t[v["D"]] - t[v["C"]]),
    ("E-B", "E − B", "decoupled combination vs published DNGaussian", lambda t, v: t[v["E"]] - t[v["B"]]),
    ("E-C", "E − C", "decoupled combination vs dropout alone", lambda t, v: t[v["E"]] - t[v["C"]]),
]
PASS = [
    ("D_hard-E", "D_hard − E", "does the mask cost anything when it reaches only the hard depth pass?",
     lambda t, v: t["D_hard"] - t["E"]),
    ("D_soft-E", "D_soft − E", "the same for the soft depth pass", lambda t, v: t["D_soft"] - t["E"]),
]
IDENTITY = {x: x for x in T1_IDS}


def t975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        return T975[df - 1] if df <= len(T975) else 1.96


def _betacf(a, b, x):
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return h


def t_pvalue(t_stat, df):
    """Two-sided p of a t statistic: P(|T| > |t|) = I_x(df/2, 1/2) with x = df / (df + t²)."""
    try:
        from scipy.stats import t
        return float(2 * t.sf(abs(t_stat), df))
    except ImportError:
        pass
    x, a, b = df / (df + t_stat * t_stat), df / 2.0, 0.5
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def holm(ps):
    """Holm step-down adjusted p-values, in the input order (NaN stays NaN)."""
    idx = sorted((i for i, p in enumerate(ps) if p == p), key=lambda i: ps[i])
    out, run = [float("nan")] * len(ps), 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (len(idx) - rank) * ps[i]))
        out[i] = run
    return out


def paired(d, higher_is_better=True):
    d = d.dropna()
    n = len(d)
    row = dict(n=n, mean=d.mean() if n else float("nan"), sd=float("nan"), ci_lo=float("nan"),
               ci_hi=float("nan"), wins=int((d > 0).sum() if higher_is_better else (d < 0).sum()), p=float("nan"))
    if n >= 2:
        row["sd"] = d.std(ddof=1)
        se = row["sd"] / math.sqrt(n)
        half = t975(n - 1) * se
        row["ci_lo"], row["ci_hi"] = row["mean"] - half, row["mean"] + half
        row["p"] = t_pvalue(row["mean"] / se, n - 1) if se > 0 else (1.0 if row["mean"] == 0 else 0.0)
    return row


def classify(r, delta, p_col="p"):
    """The §5 class of one contrast row: '>0', '<0', '≈0', 'inconclusive', or '?' without a CI."""
    if pd.isna(r["ci_lo"]):
        return "?"
    if r[p_col] < ALPHA:
        return ">0" if r["mean"] > 0 else "<0"
    if delta is not None and -delta < r["ci_lo"] and r["ci_hi"] < delta:
        return "≈0"
    return "inconclusive"


def inside(r, delta):
    return delta is not None and not pd.isna(r["ci_lo"]) and -delta < r["ci_lo"] and r["ci_hi"] < delta


def better(cls, metric):
    """+1 if the class says the first term is better, -1 if worse, 0 if neither."""
    s = 1 if cls == ">0" else -1 if cls == "<0" else 0
    return s if HIGHER_IS_BETTER[metric] else -s


def load(path, n_views, what):
    """runs.csv of one block, n_views only. Unknown IDs are dropped with a note; a run whose recorded flags contradict
    its ID (factors.mismatches) stops the analysis, because it would silently enter the wrong contrast."""
    runs = pd.read_csv(path)
    if "psnr" not in runs:
        sys.exit(f"{path} has no psnr column")
    runs = runs[runs["n_views"] == n_views].copy()
    unknown = sorted(set(runs["variant"]) - set(F.PLANNED))
    if unknown:
        print(f"NOTE {what}: ignoring IDs that are not in the factor table: {unknown}", file=sys.stderr)
        runs = runs[runs["variant"].isin(F.PLANNED)]
    bad = []
    for _, r in runs.iterrows():
        why = F.mismatches(r["variant"], r.to_dict())
        if why:
            bad.append(f"  {r['variant']}/{r['scene']}/seed{r['seed']}: " + "; ".join(why))
    if bad:
        sys.exit(f"{path}: runs whose recorded flags contradict their variant ID (fix or move them first):\n"
                 + "\n".join(bad))
    return runs


def scene_means(runs, metrics, what):
    """{metric: scene x variant table of means over seeds}; warns when a scene has uneven seed counts."""
    out = {}
    for m in metrics:
        g = runs[runs[m].notna()]
        sm = g.groupby(["scene", "variant"])["seed"].nunique().unstack("variant")
        uneven = sm[sm.nunique(axis=1) > 1]
        if not uneven.empty:
            print(f"WARNING {what} {m}: these scenes have different numbers of seeds per variant (means still "
                  "taken):\n" + uneven.to_string(), file=sys.stderr)
        out[m] = g.groupby(["scene", "variant"])[m].mean().unstack("variant")
    return out


def contrasts(tables, comps, family, holm_family, v=IDENTITY, scenes=None, extra=None):
    """Rows for every contrast x metric; Holm within (family, metric) when holm_family."""
    rows = []
    for m, t in tables.items():
        if scenes is not None:
            t = t.loc[t.index.isin(scenes)]
        block = []
        for key, label, question, f in comps:
            block.append(dict(family=family, metric=m, comparison=key, label=label, question=question,
                              **(extra or {}), **paired(f(t, v), HIGHER_IS_BETTER[m])))
        adj = holm([r["p"] for r in block]) if holm_family else [r["p"] for r in block]
        for r, pa in zip(block, adj):
            r["p_holm"] = pa if holm_family else float("nan")
            r["class"] = classify(dict(r, p_decide=pa), DELTA.get(m), "p_decide")
            r["within_delta"] = r["class"] in (">0", "<0") and inside(r, DELTA.get(m))
        rows += block
    return rows


def per_scene(tables, comps, v=IDENTITY, scenes=None):
    t = tables["psnr"] if scenes is None else tables["psnr"].loc[lambda x: x.index.isin(scenes)]
    return {label: f(t, v) for _, label, _, f in comps}


def complete_scenes(t, ids):
    have = [x for x in ids if x in t]
    return t[have].dropna().index.tolist() if len(have) == len(ids) else []


def fmt_ci(r, digits):
    if r is None or pd.isna(r["mean"]):
        return "–"
    if pd.isna(r["ci_lo"]):
        return f"{r['mean']:+.{digits}f}"
    return f"{r['mean']:+.{digits}f} [{r['ci_lo']:+.{digits}f}, {r['ci_hi']:+.{digits}f}]"


def fmt_p(p):
    return "–" if pd.isna(p) else "<0.001" if p < 0.001 else f"{p:.3f}"


def write_tables(res, metrics, base):
    res.to_csv(base + ".csv", index=False)
    head = (["Comparison", "Family"] + [f"Δ{m.upper()} [95% CI]" for m in metrics]
            + ["PSNR wins", "PSNR p", "PSNR class"])
    rows = []
    for (family, key), g in res[res["metric"].isin(metrics)].groupby(["family", "comparison"], sort=False):
        by = {r["metric"]: r for _, r in g.iterrows()}
        ps = by.get("psnr")
        cells = [g.iloc[0]["label"], family] + [fmt_ci(by.get(m), DIGITS.get(m, 3)) for m in metrics]
        if ps is not None:
            cells += [f"{int(ps['wins'])}/{int(ps['n'])}", fmt_p(ps["p_holm"] if family in ("primary", "pass")
                                                                  else ps["p"]),
                      ps["class"] + (" (within ±δ)" if ps["within_delta"] else "")]
        else:
            cells += ["–"] * 3
        rows.append(cells)
    md = ["| " + " | ".join(head) + " |", "| --- " * len(head) + "|"] + ["| " + " | ".join(c) + " |" for c in rows]
    md += ["", f"p is Holm-adjusted within the primary and pass families; classes use δ = {DELTA['psnr']} dB (§5)."]
    open(base + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")

    def tex_cell(x):
        return (x.replace("−", "$-$").replace("±", "$\\pm$").replace("≈", "$\\approx$").replace("δ", "$\\delta$")
                .replace("I_D", "$I_D$").replace("I_E", "$I_E$").replace("D_hard", "D\\textsubscript{hard}")
                .replace("D_soft", "D\\textsubscript{soft}").replace("<", "$<$").replace(">", "$>$")
                .replace("%", "\\%"))
    tex = ["\\begin{tabular}{ll" + "c" * (len(head) - 2) + "}", "\\toprule",
           " & ".join(tex_cell(h).replace("Δ", "$\\Delta$ ") for h in head) + " \\\\", "\\midrule"]
    tex += [" & ".join(tex_cell(x) for x in c) + " \\\\" for c in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(base + ".tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")


def get(res, key, metric="psnr", family=None):
    x = res[(res["comparison"] == key) & (res["metric"] == metric)]
    if family is not None:
        x = x[x["family"] == family]
    return x.iloc[0] if len(x) else None


VERDICT = {("beats", ">0"): "synergistic", ("beats", "≈0"): "complementary",
           ("beats", "<0"): "complementary but partly redundant", ("beats", "inconclusive"): "complementary",
           ("equivalent", "<0"): "redundant", ("equivalent", "inconclusive"): "redundant?",
           ("worse", "<0"): "interfering", ("worse", "inconclusive"): "worse (unclear why)",
           ("otherwise", "<0"): "partly redundant", ("otherwise", "inconclusive"): "inconclusive"}
ROWS = dict(beats="{X} beats B and C", equivalent="{X} equivalent to B", worse="{X} worse than B",
            otherwise="otherwise")


def verdict_for(res, X, metrics):
    """(label, explanation lines) for X in {D, E}, by the §5 table on PSNR plus the SSIM/LPIPS agreement rule."""
    xb, xc, ix = get(res, f"{X}-B"), get(res, f"{X}-C"), get(res, f"I_{X}")
    if xb["class"] == ">0" and xc["class"] == ">0":
        row = "beats"
    elif inside(xb, DELTA["psnr"]):
        row = "equivalent"
    elif xb["class"] == "<0":
        row = "worse"
    else:
        row = "otherwise"
    label = VERDICT.get((row, ix["class"]))
    lines = [f"  row: {ROWS[row].format(X=X)} ({X} − B {xb['class']}, {X} − C {xc['class']}); "
             f"column: I_{X} {ix['class']}"]
    if label is None:
        label = "not covered by the §5 table"
        lines.append("  this (row, column) cell is empty in §5: describe the pattern instead of forcing a label")
    clash = []
    for key in (f"{X}-B", f"{X}-C", f"I_{X}"):
        s = better(get(res, key)["class"], "psnr")
        for m in [m for m in metrics if m != "psnr"]:
            r = get(res, key, m)
            if s and r is not None and better(r["class"], m) == -s:
                clash.append(f"{m.upper()} on {key}")
    if clash:
        lines.append("  SSIM/LPIPS point the other way (" + ", ".join(clash) + ")")
        label = f"metric-dependent (PSNR alone: {label})"
    else:
        lines.append("  SSIM and LPIPS: no significant disagreement with PSNR")
    return label, lines


def mechanism_depth(res, diag_summary, n_views, scenes, dose_ps):
    """'Through the depth supervision': E − D > 0; gap(D) > gap(C); E − D larger at γ = 1/3 than 0.1 in ≥ 6/8."""
    out, ok = [], []
    ed = get(res, "E-D", family="primary")
    ok.append(ed["class"] == ">0")
    out.append(f"  1. E − D > 0 (Holm): {fmt_ci(ed, 2)}, class {ed['class']} → {'met' if ok[-1] else 'not met'}")
    if diag_summary is None:
        ok.append(None)
        out.append("  2. gap(D) > gap(C): not evaluated (give --diag_summary)")
    else:
        s = pd.read_csv(diag_summary)
        if "n_views" in s:
            s = s[s["n_views"] == n_views]
        gap = s.pivot_table(index="scene", columns="variant", values="gap_mean", aggfunc="mean")
        gap = gap[gap.index.isin(scenes)]
        failed = s[s["variant"].isin(["C", "D"]) & ~s["checks_ok"].astype(bool)] if "checks_ok" in s else s[:0]
        if len(failed):
            ok.append(False)
            out.append("  2. gap(D) > gap(C): diag.csv sharp checks FAILED for "
                       + ", ".join(failed["variant"] + "/" + failed["scene"]) + "; no conclusion until fixed")
        elif "C" not in gap or "D" not in gap:
            ok.append(None)
            out.append("  2. gap(D) > gap(C): no diag summary for C and D yet")
        else:
            d, dc = paired(gap["D"]), paired(gap["D"] - gap["C"])
            ok.append(dc["n"] >= 2 and dc["ci_lo"] > 0)
            out.append(f"  2. gap(D) − gap(C), paired by scene: {fmt_ci(dc, 4)} (gap(D) {fmt_ci(d, 4)}), n = {dc['n']}"
                       f" → {'met' if ok[-1] else 'not met'}")
    if dose_ps is None:
        ok.append(None)
        out.append("  3. E − D larger at γ = 1/3 than at γ = 0.1 in ≥ 6/8 scenes: not evaluated (give --dose_runs)")
    else:
        hi, lo = dose_ps.get((1 / 3, "E − D")), dose_ps.get((0.1, "E − D"))
        both = pd.concat([hi, lo], axis=1).dropna() if hi is not None and lo is not None else pd.DataFrame()
        k, n = int((both.iloc[:, 0] > both.iloc[:, 1]).sum()) if len(both) else 0, len(both)
        ok.append(True if k >= 6 else False if k + len(SCENES) - n < 6 else None)   # None: missing scenes decide
        out.append(f"  3. E − D larger at γ = 1/3 than at γ = 0.1: {k}/{n} scenes (needs ≥ 6/8) → "
                   + ("met" if ok[-1] else "not met" if ok[-1] is False else "pending, not all 8 scenes yet"))
    return out, ok


def mechanism_cap(res, why_not):
    """'Through the opacity cap': D_hard − E < 0; D_soft − E CI contains 0; the probe's far shift grows with r."""
    h, s = get(res, "D_hard-E", family="pass"), get(res, "D_soft-E", family="pass")
    if h is None or s is None:
        return ["  not evaluated (" + why_not + ")"], [None]
    ok = [h["class"] == "<0", not pd.isna(s["ci_lo"]) and s["ci_lo"] <= 0 <= s["ci_hi"], None]
    return [f"  1. D_hard − E < 0 (Holm): {fmt_ci(h, 2)}, class {h['class']} → {'met' if ok[0] else 'not met'}",
            f"  2. D_soft − E CI contains 0: {fmt_ci(s, 2)} → {'met' if ok[1] else 'not met'}",
            "  3. Monte-Carlo probe: positive hard-pass depth shift, larger at r = 0.2 than at 0.12: not evaluated "
            "here; read probes/depth_bias.json of the D and E seed-0 runs (probes.py)"], ok


def claim(ok):
    if all(x is True for x in ok):
        return "SUPPORTED (all conditions met)"
    if any(x is False for x in ok):
        return "not supported (a condition is not met)"
    return "open (conditions met so far, others pending)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/tables/runs.csv", help="block T1: aggregate.py on outputs/")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--diag_summary", default=None, help="diag_summary_<n>v.csv from plots/diag_curves.py on T1")
    ap.add_argument("--pass_runs", default=None, help="block P: aggregate.py's runs.csv for outputs_pass/")
    ap.add_argument("--dose_runs", default=None, help="block G: aggregate.py's runs.csv for outputs_dose/")
    ap.add_argument("--noise_runs", default=None, help="block N: aggregate.py's runs.csv for outputs_noise/")
    ap.add_argument("--geometry", default=None, help="geometry_<n>v.csv from geometry.py on T1 (MVC@5%%)")
    a = ap.parse_args()
    nv = a.n_views

    runs = load(a.runs, nv, "T1")
    metrics = [m for m in METRICS if m in runs]
    if runs.empty:
        sys.exit(f"No {nv}-view runs in {a.runs}")
    absent = [x for x in T1_IDS if x not in set(runs["variant"])]
    if absent:
        sys.exit(f"Variants {absent} have no {nv}-view runs yet; every T1 contrast needs all five")
    other = sorted(set(runs["variant"]) - set(T1_IDS))
    if other:
        print(f"NOTE --runs is block T1: ignoring {other} (give their block's runs.csv to its own option)",
              file=sys.stderr)
    t1 = runs[runs["variant"].isin(T1_IDS)]
    means = scene_means(t1, metrics, "T1")
    os.makedirs(a.tables, exist_ok=True)
    pd.concat(means, axis=1).round(4).to_csv(os.path.join(a.tables, f"scene_means_{nv}v.csv"))
    complete = complete_scenes(means["psnr"], T1_IDS)
    if len(complete) < len(SCENES):
        print(f"NOTE paired over {len(complete)} scenes with all five variants: {complete}", file=sys.stderr)
    if len(complete) < 2:
        sys.exit("Need at least 2 scenes with all five variants for a confidence interval")
    seeds = t1.groupby(["scene", "variant"])["seed"].nunique().unstack("variant")
    print("T1 seeds per scene and variant:\n" + seeds.loc[complete].to_string() + "\n")

    inputs = {"T1": t1}
    rows = (contrasts(means, PRIMARY, "primary", True, scenes=complete)
            + contrasts(means, SECONDARY, "secondary", False, scenes=complete))
    ps = per_scene(means, PRIMARY + SECONDARY, scenes=complete)

    pass_scenes, pass_why = [], "give --pass_runs"
    if a.pass_runs:
        pr = load(a.pass_runs, nv, "P")
        inputs["P"] = pr
        pr = pr[pr["variant"].isin(PASS_IDS)]
        pm = scene_means(pd.concat([t1[t1["variant"] == "E"], pr]), metrics, "P with T1's E")
        pass_scenes = complete_scenes(pm["psnr"], ["E"] + PASS_IDS)
        if len(pass_scenes) >= 2:
            rows += contrasts(pm, PASS, "pass", True, scenes=pass_scenes)
            ps.update(per_scene(pm, PASS, scenes=pass_scenes))
        else:
            pass_why = f"needs 2 scenes with E, D_hard and D_soft; have {pass_scenes}"
            print("NOTE pass localization " + pass_why, file=sys.stderr)

    if a.geometry:
        geo = pd.read_csv(a.geometry)
        geo = geo[(geo["n_views"] == nv) & geo["variant"].isin(T1_IDS)]
        gm = scene_means(geo, ["mvc5"], "geometry")
        geo_scenes = complete_scenes(gm["mvc5"], T1_IDS)
        if len(geo_scenes) >= 2:
            rows += contrasts(gm, PRIMARY, "geometry", True, scenes=geo_scenes)
        else:
            print(f"NOTE geometry needs 2 scenes with A–E in {a.geometry}; have {geo_scenes}", file=sys.stderr)

    res = pd.DataFrame(rows)
    base = os.path.join(a.tables, f"paired_{nv}v")
    write_tables(res, metrics, base)
    pd.DataFrame(ps).round(3).to_csv(os.path.join(a.tables, f"paired_per_scene_{nv}v.csv"))

    dose_ps = None
    if a.dose_runs:
        dr = load(a.dose_runs, nv, "G")
        inputs["G"] = dr
        dose_ids = [x + sfx for _, sfx in DOSE if sfx for x in "CDE"]
        s0 = pd.concat([t1[t1["seed"] == 0], dr[(dr["seed"] == 0) & dr["variant"].isin(dose_ids)]])
        dm = scene_means(s0, metrics, "dose (seed 0)")
        drows, dose_ps, dps_rows = [], {}, []
        for gamma, sfx in DOSE:
            v = dict(IDENTITY, C="C" + sfx, D="D" + sfx, E="E" + sfx)
            sc = complete_scenes(dm["psnr"], [v[x] for x in T1_IDS])
            if len(sc) < 2:
                print(f"NOTE dose γ = {gamma:.3g}: need 2 seed-0 scenes with {sorted(v.values())}; have {sc}",
                      file=sys.stderr)
                continue
            drows += contrasts(dm, PRIMARY + SECONDARY, "dose", False, v=v, scenes=sc, extra=dict(gamma=gamma))
            for label, d in per_scene(dm, PRIMARY + SECONDARY, v=v, scenes=sc).items():
                dose_ps[(gamma, label)] = d
                dps_rows += [dict(gamma=gamma, comparison=label, scene=s, psnr_delta=x) for s, x in d.items()]
        if drows:
            pd.DataFrame(drows).to_csv(os.path.join(a.tables, f"dose_{nv}v.csv"), index=False)
            pd.DataFrame(dps_rows).round(4).to_csv(os.path.join(a.tables, f"dose_per_scene_{nv}v.csv"), index=False)

    noise_lines = []
    if a.noise_runs:
        nr = load(a.noise_runs, nv, "N")
        inputs["N"] = nr
        out = []
        for X in ("B", "D"):
            reps = nr[nr["variant"].str.match(f"^{X}_r[1-9]$") & (nr["seed"] == 0)]
            cell = t1[t1["variant"] == X].groupby("scene")["psnr"].std(ddof=1).dropna()
            pooled = math.sqrt((cell ** 2).mean()) if len(cell) else float("nan")
            for scene, g in reps.groupby("scene"):
                anchor = t1[(t1["variant"] == X) & (t1["scene"] == scene) & (t1["seed"] == 0)]
                vals = pd.concat([anchor["psnr"], g["psnr"]]).dropna()
                out.append(dict(variant=X, scene=scene, n_repeats=len(vals), psnr_mean=vals.mean(),
                                psnr_sd_same_seed=vals.std(ddof=1), psnr_sd_between_seeds_T1=pooled))
                noise_lines.append(f"  {X} {scene}: {len(vals)} same-seed runs, SD {vals.std(ddof=1):.3f} dB "
                                   f"(T1 between-seed SD, pooled over scenes: {pooled:.3f} dB)")
        if out:
            pd.DataFrame(out).round(4).to_csv(os.path.join(a.tables, f"noise_floor_{nv}v.csv"), index=False)

    ftab = pd.DataFrame([F.factors(x) for x in F.PLANNED])
    for name, r in inputs.items():
        ftab[f"runs_{name}"] = ftab["variant"].map(r["variant"].value_counts()).fillna(0).astype(int)
    ftab.round(6).to_csv(os.path.join(a.tables, "factors.csv"), index=False)

    lines = [f"Reference: B (published DNGaussian), fixed. Margin δ = {DELTA['psnr']} dB. "
             f"T1 scenes paired: {len(complete)}.", "", "Primary (PSNR; Holm across the three):"]
    for key, label, _, _ in PRIMARY:
        r = get(res, key, family="primary")
        lines.append(f"  {label:<26} {fmt_ci(r, 2):<26} Holm p {fmt_p(r['p_holm']):<7} {r['class']:<12}"
                     f" wins {int(r['wins'])}/{int(r['n'])}" + ("  (within ±δ)" if r["within_delta"] else ""))
    lines += ["", "Secondary (PSNR; 95% CI, uncorrected):"]
    for key, label, _, _ in SECONDARY:
        r = get(res, key, family="secondary")
        lines.append(f"  {label:<26} {fmt_ci(r, 2):<26} p {fmt_p(r['p']):<12} {r['class']:<12}"
                     f" wins {int(r['wins'])}/{int(r['n'])}" + ("  (within ±δ)" if r["within_delta"] else ""))
    labels = {}
    for X in ("D", "E"):
        labels[X], why = verdict_for(res, X, metrics)
        lines += ["", f"Verdict for {X}: {labels[X]}"] + why
    lines += ["", f"VERDICT: D = {labels['D']}; E = {labels['E']}"]

    lines += ["", "Pass localization (block P, PSNR; Holm across the two):"]
    if get(res, "D_hard-E", family="pass") is None:
        lines.append("  not evaluated (" + pass_why + ")")
    else:
        for key, label, _, _ in PASS:
            r = get(res, key, family="pass")
            lines.append(f"  {label:<26} {fmt_ci(r, 2):<26} Holm p {fmt_p(r['p_holm']):<7} {r['class']:<12}"
                         f" wins {int(r['wins'])}/{int(r['n'])}")
    m1, ok1 = mechanism_depth(res, a.diag_summary, nv, complete, dose_ps)
    m2, ok2 = mechanism_cap(res, pass_why)
    lines += ["", "Mechanism 'through the depth supervision': " + claim(ok1)] + m1
    lines += ["", "Mechanism 'through the opacity cap': " + claim(ok2) + "; the probe condition is checked by hand"]
    lines += m2
    if a.geometry and get(res, "E-D", "mvc5") is not None:
        lines += ["", f"Geometry (MVC@5%, δ = {DELTA['mvc5']}; Holm across the three):"]
        for key, label, _, _ in PRIMARY:
            r = get(res, key, "mvc5")
            lines.append(f"  {label:<26} {fmt_ci(r, 3):<28} Holm p {fmt_p(r['p_holm']):<7} {r['class']}")
    if dose_ps is not None:
        lines += ["", "Dose (seed 0, trend; E − D and the interactions at each γ):"]
        for gamma, _ in DOSE:
            have = [lab for (g, lab) in dose_ps if g == gamma]
            if have:
                cells = []
                for _, label, _, _ in PRIMARY:
                    d = dose_ps[(gamma, label)].dropna()
                    cells.append(f"{label.split(' =')[0]} {d.mean():+.2f}")
                lines.append(f"  γ = {gamma:.3g}: " + ", ".join(cells))
    if noise_lines:
        lines += ["", "Noise floor (block N):"] + noise_lines
    lines += ["", "Power: with 8 scenes x 3 seeds, interactions smaller than about 0.35–0.5 dB are below the detection "
                  "limit (RESEARCH_ANALYSIS §5.3)."]
    open(os.path.join(a.tables, f"verdict_{nv}v.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(open(base + ".md", encoding="utf-8").read())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
