#!/usr/bin/env python
"""The LLFF blocks of v1.1-exp, analysed by the pre-registered rules of PLAN_V2 §5.2.6, §5.2.7 and §5.2.9.

    python experiments/extra_blocks.py --runs results/tables/runs.csv \\
        --comp_runs results/tables/comp/runs.csv --ddose_runs results/tables/depthdose/runs.csv \\
        --noise2_runs results/tables/noise2/runs.csv --noise_runs results/tables/noise/runs.csv \\
        --tables results/tables/extra
    python experiments/extra_blocks.py --self_test

Each input is aggregate.py's runs.csv for one block; only --runs (T1) is required, every other block is optional.

COMP (§5.2.9): C_nc − C, D_nc − D, C_ca − C, D_ca − D, paired by scene with T1's C and D (mean over seeds), Holm
    across the four per metric, the §5 class, and TOST equivalence on PSNR with δ = 0.25 dB (equivalent if both
    one-sided p < 0.05; reported beside the class, never instead of it).
DEPTH-DOSE (N5, §5.2.6): I_D(s) = (D_s − C) − (B_s − A) for s = 0.5, 1, 2 (s = 1 is T1's B and D; A and C from T1),
    Holm across the three. The prediction "additivity holds at every dose" holds only if all three classes are "≈0".
    B_s − A and D_s − C are reported as a trend.
NOISE2 (G6 / N2(a), §5.2.7), PSNR:
    - same-seed SD (5 runs at seed 0: T1 seed 0 + the 4 repeats) vs between-seed SD (T1 seeds 0–2), pooled over the
      scenes that have both, per variant; prediction: the ratio lies in [0.5, 2].
    - prior-free (A, C) vs depth (B, D) pooled run variance: one-sided F-test, holds if F > 1 and p < 0.05.
    - variance components: σ²_v pooled within (scene, variant) cells over T1 seeds and every repeat; τ²_c =
      var(scene means of c) − Σ coef² σ²_v / R (R = 3), truncated at 0, for the main effect B − A and for I_D.
    - power: SE_c = √(τ²_c/S + Σ coef² σ²_v/(S·R)), MDE = (t_.975 + t_.80)·SE_c for S ∈ {8, 15, 23, 31},
      R ∈ {1, 3, 5}; validation: at S = 8, R = 3 the predicted 95 % half-width of I_D must lie within ±25 % of the
      observed T1 half-width, else the MDE table is "indicative only".
    - noise band for the audit: ±1.96·√(2σ²_class/8) for "depth host" (B, D) and "prior-free" (A, C).

Writes to --tables: comp.csv, depthdose.csv, noise2_sd.csv, variance_components.csv, power.csv, verdict_extra.txt.
"""
import argparse, math, os, sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as S  # noqa: E402

METRICS = ["psnr", "ssim", "lpips"]
DELTA = 0.25
COMP = [("C_nc-C", "C_nc", "C"), ("D_nc-D", "D_nc", "D"), ("C_ca-C", "C_ca", "C"), ("D_ca-D", "D_ca", "D")]
DOSES = [(0.5, "_ds050"), (1.0, ""), (2.0, "_ds200")]
CONTRAST_COEF = {"B-A": {"B": 1, "A": -1}, "I_D": {"D": 1, "C": -1, "B": -1, "A": 1}}
GRID_S, GRID_R = [8, 15, 23, 31], [1, 3, 5]


def t_ppf(q, df):
    try:
        from scipy.stats import t
        return float(t.ppf(q, df))
    except ImportError:
        lo, hi = 0.0, 50.0                        # bisection on the two-sided p of stats.t_pvalue
        for _ in range(100):
            mid = (lo + hi) / 2
            if 1 - S.t_pvalue(mid, df) / 2 < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def t_upper(t_stat, df):
    """One-sided P(T > t)."""
    p2 = S.t_pvalue(t_stat, df)
    return p2 / 2 if t_stat > 0 else 1 - p2 / 2


def f_upper(f, d1, d2):
    """One-sided P(F > f) = I_x(d2/2, d1/2), x = d2 / (d2 + d1 f)."""
    try:
        from scipy.stats import f as F
        return float(F.sf(f, d1, d2))
    except ImportError:
        x, a, b = d2 / (d2 + d1 * f), d2 / 2.0, d1 / 2.0
        front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
        return front * S._betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - front * S._betacf(b, a, 1 - x) / b


def tost(d, delta):
    """Two one-sided tests of |mean| < delta; returns the larger one-sided p (equivalent if < 0.05)."""
    d = d.dropna()
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan")
    se = d.std(ddof=1) / math.sqrt(n)
    return max(t_upper((d.mean() + delta) / se, n - 1), 1 - t_upper((d.mean() - delta) / se, n - 1))


def means(runs, metric):
    return runs.groupby(["scene", "variant"])[metric].mean().unstack("variant")


def llff3(df):
    df = df.copy()
    if "dataset" in df:
        df = df[df["dataset"] == "llff"]
    return df[df["n_views"] == 3]


def comp_block(t1, comp):
    allr = pd.concat([t1[t1["variant"].isin(["C", "D"])], comp], ignore_index=True)
    rows = []
    for m in METRICS:
        t = means(allr, m)
        block = []
        for key, x, ref in COMP:
            if x not in t or ref not in t:
                continue
            d = (t[x] - t[ref]).dropna()
            r = S.paired(d, S.HIGHER_IS_BETTER[m])
            r.update(metric=m, comparison=key, tost_p=tost(d, DELTA) if m == "psnr" else float("nan"))
            block.append(r)
        for r, ph in zip(block, S.holm([r["p"] for r in block])):
            r["p_holm"] = ph
            r["class"] = S.classify(r, DELTA if m == "psnr" else None, "p_holm")
            r["equivalent"] = (r["tost_p"] < 0.05) if r["tost_p"] == r["tost_p"] else None
        rows += block
    return pd.DataFrame(rows)


def dose_block(t1, dd):
    allr = pd.concat([t1[t1["variant"].isin(list("ABCD"))], dd], ignore_index=True)
    t = means(allr, "psnr")
    rows = []
    for s, suf in DOSES:
        b, d = "B" + suf, "D" + suf
        if b not in t or d not in t:
            continue
        for key, f in [("I_D", lambda: (t[d] - t["C"]) - (t[b] - t["A"])),
                       ("B_s-A", lambda: t[b] - t["A"]), ("D_s-C", lambda: t[d] - t["C"])]:
            r = S.paired(f().dropna())
            r.update(dose=s, comparison=key)
            rows.append(r)
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    i = res["comparison"] == "I_D"
    res.loc[i, "p_holm"] = S.holm(list(res.loc[i, "p"]))
    res["class"] = [S.classify(r, DELTA, "p_holm" if r["comparison"] == "I_D" else "p") for _, r in res.iterrows()]
    return res


def pooled_var(groups):
    """Pooled within-group variance and its df from an iterable of value arrays."""
    ss, df = 0.0, 0
    for g in groups:
        g = pd.Series(g).dropna()
        if len(g) >= 2:
            ss += ((g - g.mean()) ** 2).sum()
            df += len(g) - 1
    return (ss / df if df else float("nan")), df


def noise_block(t1, reps):
    """reps: runs of the repeat blocks (noise, noise2), IDs <letter>_r<k>, seed 0."""
    r = reps.copy()
    r["letter"] = r["variant"].str[0]
    seed0 = t1[t1["seed"] == 0].copy()
    seed0["letter"] = seed0["variant"]
    same = pd.concat([r, seed0[seed0.set_index(["scene", "letter"]).index.isin(
        r.set_index(["scene", "letter"]).index)]], ignore_index=True)
    rows = []
    for v in "ABCD":
        cells = same[same["letter"] == v]
        if cells.empty:
            continue
        sc = sorted(cells["scene"].unique())
        vs, dfs = pooled_var(g["psnr"] for _, g in cells.groupby("scene"))
        tb = t1[(t1["variant"] == v) & t1["scene"].isin(sc)]
        vb, dfb = pooled_var(g["psnr"] for _, g in tb.groupby("scene"))
        ratio = math.sqrt(vs / vb) if vb and vb == vb and vb > 0 else float("nan")
        rows.append(dict(variant=v, scenes=" ".join(sc), same_seed_sd=math.sqrt(vs), df_same=dfs,
                         between_seed_sd=math.sqrt(vb) if vb == vb else float("nan"), df_between=dfb,
                         ratio=ratio, in_0_5_to_2=(0.5 <= ratio <= 2) if ratio == ratio else None))
    return pd.DataFrame(rows), same


def sigma2(t1, same):
    """σ²_v pooled within (scene, variant) cells over T1 seeds and the repeats (seed-0 run counted once)."""
    t = t1.copy()
    t["letter"] = t["variant"]
    rep = same[same["variant"].str.contains("_r")] if not same.empty else same
    allr = pd.concat([t, rep], ignore_index=True)
    out = {}
    for v in "ABCDE":
        g = allr[allr["letter"] == v]
        out[v] = pooled_var(x["psnr"] for _, x in g.groupby("scene"))
    return out


def components(t1, sig, R=3):
    t = means(t1, "psnr")
    rows, power = [], []
    for c, coef in CONTRAST_COEF.items():
        sc = sum(k * t[v] for v, k in coef.items()).dropna()
        noise = sum(k * k * sig[v][0] for v, k in coef.items())
        tau2 = max(0.0, sc.var(ddof=1) - noise / R)
        obs = S.paired(sc)
        pred_half = S.t975(len(sc) - 1) * math.sqrt(tau2 / len(sc) + noise / (len(sc) * R))
        obs_half = (obs["ci_hi"] - obs["ci_lo"]) / 2
        rows.append(dict(contrast=c, n_scenes=len(sc), tau2=tau2, run_noise_sum=noise, observed_half_width=obs_half,
                         predicted_half_width=pred_half, ratio=pred_half / obs_half if obs_half else float("nan")))
        for Sx in GRID_S:
            for Rx in GRID_R:
                se = math.sqrt(tau2 / Sx + noise / (Sx * Rx))
                power.append(dict(contrast=c, scenes=Sx, runs=Rx, se=se,
                                  mde=(t_ppf(0.975, Sx - 1) + t_ppf(0.80, Sx - 1)) * se))
    return pd.DataFrame(rows), pd.DataFrame(power)


def run(a):
    t1 = llff3(pd.read_csv(a.runs))
    t1 = t1[t1["variant"].isin(list("ABCDE"))]
    os.makedirs(a.tables, exist_ok=True)
    out = []
    if a.comp_runs:
        res = comp_block(t1, llff3(pd.read_csv(a.comp_runs)))
        res.to_csv(os.path.join(a.tables, "comp.csv"), index=False)
        out.append("COMP (§5.2.9), PSNR, Holm across 4, TOST δ = 0.25 dB:")
        for _, r in res[res["metric"] == "psnr"].iterrows():
            out.append("  {:<7} {} p_holm {:.3g}  class {}  TOST p {:.3g} -> {}".format(
                r["comparison"], S.fmt_ci(r, 2), r["p_holm"], r["class"], r["tost_p"],
                "equivalent" if r["equivalent"] else "not shown equivalent"))
    if a.ddose_runs:
        res = dose_block(t1, llff3(pd.read_csv(a.ddose_runs)))
        res.to_csv(os.path.join(a.tables, "depthdose.csv"), index=False)
        i = res[res["comparison"] == "I_D"]
        out.append("DEPTH-DOSE (N5, §5.2.6), I_D(s) on PSNR, Holm across the doses:")
        for _, r in i.iterrows():
            out.append("  s = {:<4} {} class {}".format(r["dose"], S.fmt_ci(r, 2), r["class"]))
        ok = len(i) == 3 and (i["class"] == "≈0").all()
        out.append("  prediction (all three ≈0): " + ("HOLDS" if ok else "does not hold" if len(i) == 3
                                                          else "incomplete"))
    reps = [llff3(pd.read_csv(p)) for p in (a.noise_runs, a.noise2_runs) if p]
    same = pd.DataFrame(columns=["variant", "letter"])
    if reps:
        sd, same = noise_block(t1, pd.concat(reps, ignore_index=True))
        sd.to_csv(os.path.join(a.tables, "noise2_sd.csv"), index=False)
        out.append("NOISE (G6), same-seed SD / between-seed SD per variant (prediction: in [0.5, 2]):")
        for _, r in sd.iterrows():
            out.append("  {}  {:.3f} / {:.3f} = {:.2f}  ({}; scenes {})".format(
                r["variant"], r["same_seed_sd"], r["between_seed_sd"], r["ratio"],
                "in range" if r["in_0_5_to_2"] else "OUT of range", r["scenes"]))
    sig = sigma2(t1, same)
    pf = [sig[v] for v in "AC" if sig[v][1]]
    dh = [sig[v] for v in "BD" if sig[v][1]]
    if pf and dh:
        v_pf = sum(s * d for s, d in pf) / sum(d for _, d in pf)
        v_dh = sum(s * d for s, d in dh) / sum(d for _, d in dh)
        d1, d2 = sum(d for _, d in pf), sum(d for _, d in dh)
        F = v_pf / v_dh
        p = f_upper(F, d1, d2)
        out.append("NOISE CLASS: σ²(A,C) / σ²(B,D) = {:.4f} / {:.4f} = F {:.2f} (df {}, {}), one-sided p {:.3g} -> {}"
                   .format(v_pf, v_dh, F, d1, d2, p, "HOLDS" if F > 1 and p < 0.05 else "does not hold"))
        out.append("NOISE BAND (audit, single-run 8-scene mean): depth host ±{:.2f} dB, prior-free ±{:.2f} dB".format(
            1.96 * math.sqrt(2 * v_dh / 8), 1.96 * math.sqrt(2 * v_pf / 8)))
    vc, pw = components(t1, sig)
    pd.DataFrame([dict(variant=v, sigma2=s, df=d) for v, (s, d) in sig.items()]).to_csv(
        os.path.join(a.tables, "sigma2.csv"), index=False)
    vc.to_csv(os.path.join(a.tables, "variance_components.csv"), index=False)
    pw.to_csv(os.path.join(a.tables, "power.csv"), index=False)
    out.append("VARIANCE COMPONENTS and power (§5.2.7):")
    for _, r in vc.iterrows():
        fit = abs(r["ratio"] - 1) <= 0.25
        out.append("  {:<4} τ² {:.4f}  predicted 95% half-width {:.3f} vs observed {:.3f} (ratio {:.2f}) -> {}".format(
            r["contrast"], r["tau2"], r["predicted_half_width"], r["observed_half_width"], r["ratio"],
            "model fits" if fit else "model does NOT fit: MDE table indicative only"))
    for c in CONTRAST_COEF:
        g = pw[(pw["contrast"] == c) & (pw["runs"] == 3)]
        out.append("  MDE {} at R = 3: ".format(c) + ", ".join("S={} {:.2f} dB".format(int(r["scenes"]), r["mde"])
                                                         for _, r in g.iterrows()))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(a.tables, "verdict_extra.txt"), "w") as f:
        f.write(txt)
    print(txt + "wrote " + a.tables)


def self_test():
    import random
    import tempfile
    rnd = random.Random(0)
    scenes = S.SCENES
    eff = dict(A=0, B=1.0, C=0.5, D=1.5, E=1.6, C_nc=0.5, D_nc=1.5, C_ca=1.5, D_ca=1.5,
               B_ds050=0.6, D_ds050=1.1, B_ds200=1.4, D_ds200=1.9)
    def rows(variants, seeds, sd=0.1, scn=scenes):
        out = []
        for sc in scn:
            s_i = scenes.index(sc)
            for v in variants:
                for k in seeds:
                    out.append(dict(dataset="llff", n_views=3, scene=sc, variant=v, seed=k,
                                    psnr=18 + 0.3 * s_i + eff[v.split("_r")[0]] + rnd.gauss(0, sd),
                                    ssim=0.6, lpips=0.3))
        return pd.DataFrame(out)
    tmp = tempfile.mkdtemp()
    paths = {}
    for name, df in [("t1", rows(list("ABCDE"), [0, 1, 2])), ("comp", rows(["C_nc", "D_nc", "C_ca", "D_ca"], [0, 1, 2])),
                     ("dd", rows(["B_ds050", "D_ds050", "B_ds200", "D_ds200"], [0, 1, 2])),
                     ("n2", rows(["A_r%d" % k for k in range(1, 5)] + ["C_r%d" % k for k in range(1, 5)]
                                 + ["B_r%d" % k for k in range(1, 5)] + ["D_r%d" % k for k in range(1, 5)], [0],
                                 scn=["room", "horns", "trex"]))]:
        paths[name] = os.path.join(tmp, name + ".csv")
        df.to_csv(paths[name], index=False)
    a = argparse.Namespace(runs=paths["t1"], comp_runs=paths["comp"], ddose_runs=paths["dd"],
                           noise2_runs=paths["n2"], noise_runs=None, tables=os.path.join(tmp, "out"))
    run(a)
    comp = pd.read_csv(os.path.join(a.tables, "comp.csv")).set_index(["metric", "comparison"])
    assert comp.loc[("psnr", "C_nc-C"), "class"] == "≈0", comp
    assert comp.loc[("psnr", "C_ca-C"), "class"] == ">0", comp           # +1 dB built in
    assert bool(comp.loc[("psnr", "D_nc-D"), "equivalent"]), comp
    dd = pd.read_csv(os.path.join(a.tables, "depthdose.csv"))
    assert (dd[dd["comparison"] == "I_D"]["class"] == "≈0").all(), dd     # effects are additive by construction
    vc = pd.read_csv(os.path.join(a.tables, "variance_components.csv"))
    assert (vc["tau2"] >= 0).all()
    sd = pd.read_csv(os.path.join(a.tables, "noise2_sd.csv"))
    assert sd["ratio"].between(0.3, 3).all(), sd                          # same noise everywhere by construction
    assert abs(t_ppf(0.975, 7) - 2.365) < 0.01 and abs(f_upper(1.0, 10, 10) - 0.5) < 1e-6
    print("EXTRA-BLOCKS SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="aggregate.py's runs.csv of T1 (outputs/)")
    ap.add_argument("--comp_runs")
    ap.add_argument("--ddose_runs")
    ap.add_argument("--noise2_runs")
    ap.add_argument("--noise_runs", help="block N (fern, B/D repeats), optional")
    ap.add_argument("--tables", default="results/tables/extra")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.runs:
        ap.error("--runs is required (or --self_test)")
    run(a)


if __name__ == "__main__":
    main()
