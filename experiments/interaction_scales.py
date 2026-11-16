#!/usr/bin/env python
"""N4 (PLAN_V2 §5.2.5): are the interaction and the combination contrasts the same on every error scale?

    python experiments/interaction_scales.py --runs results/tables/runs.csv \\
        --budget_runs results/tables/error_budget/error_budget_runs_llff_3v.csv --dataset llff --n_views 3 \\
        --tables results/tables
    python experiments/interaction_scales.py --self_test

Scales: PSNR and SSIM per run from results_eval.json (aggregate.py's runs.csv); MSE and RMSE per run from the error
budget's per-run file (bias_variance.py, exact pixel MSE; never converted from mean PSNR). For each scale: the mean over
seeds per (scene, variant), then I_D, I_E, E − D and D − B paired by scene with a 95 % t-interval.
MSE and RMSE are errors, so their contrasts are negated ("better" is positive on every scale).
Label (prereg rule): "robust" if the four point estimates have the same sign, otherwise "scale-dependent".
A non-crossover interaction can be created or removed by a monotone change of scale (Loftus 1978), so only robust
interactions are interpreted.

Writes <tables>/interaction_scales_<dataset>[_<n>v].csv (the _<n>v suffix only for non-default view counts).
"""
import argparse, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as S  # noqa: E402

CONTRASTS = [
    ("I_D", {"D": 1, "C": -1, "B": -1, "A": 1}),
    ("I_E", {"E": 1, "C": -1, "B": -1, "A": 1}),
    ("E-D", {"E": 1, "D": -1}),
    ("D-B", {"D": 1, "B": -1}),
]
SCALES = [("psnr", 1), ("mse", -1), ("rmse", -1), ("ssim", 1)]   # (scale, sign that makes "better" positive)
DEFAULT_VIEWS = dict(llff=3, dtu=3, blender=8)


def per_run_table(runs, budget_runs, dataset, n_views):
    r = runs.copy()
    if "dataset" in r:
        r = r[r["dataset"] == dataset]
    elif dataset != "llff":
        sys.exit("runs.csv has no dataset column, so it is LLFF only")
    r = r[(r["n_views"] == n_views) & r["variant"].isin(list("ABCDE"))][["scene", "variant", "seed", "psnr", "ssim"]]
    b = budget_runs[["scene", "variant", "seed", "mse", "rmse"]]
    m = r.merge(b, on=["scene", "variant", "seed"], how="inner")
    lost = len(r) - len(m)
    if lost:
        print("NOTE {} runs in runs.csv have no per-run MSE (fewer than two seeds with eval renders?)".format(lost),
              file=sys.stderr)
    return m


def analyse(table):
    rows = []
    for key, terms in CONTRASTS:
        signs = {}
        for scale, sign in SCALES:
            t = table.groupby(["scene", "variant"])[scale].mean().unstack("variant")
            if not all(v in t for v in terms):
                continue
            d = sign * sum(c * t[v] for v, c in terms.items())
            r = S.paired(d.dropna(), higher_is_better=True)
            signs[scale] = np.sign(r["mean"])
            rows.append(dict(contrast=key, scale=scale, oriented="better is +", n=r["n"], mean=r["mean"],
                             ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p=r["p"]))
        label = ("robust (same sign)" if len(signs) == len(SCALES) and len(set(signs.values())) == 1
                 else "scale-dependent" if len(signs) == len(SCALES) else "incomplete")
        for r in rows:
            if r["contrast"] == key:
                r["label"] = label
    return pd.DataFrame(rows)


def self_test():
    """A Loftus case: PSNR-additive effects (I = 0 in dB) give a non-zero interaction in MSE; the script must call it
    scale-dependent; the main effects E − D and D − B must stay robust."""
    rng = np.random.RandomState(0)
    rows = []
    for s in range(8):
        base = 18 + rng.rand()
        # dB effects: depth +1.0, dropout +0.8, nearly additive in dB (I_D = +0.02 dB > 0)
        db = dict(A=base, B=base + 1.0, C=base + 0.8, D=base + 1.82, E=base + 2.6)
        for v, p in db.items():
            for k in range(3):
                pk = p + 0.001 * rng.randn()
                mse = 10 ** (-pk / 10)
                rows.append(dict(scene="s{}".format(s), variant=v, seed=k, psnr=pk, mse=mse, rmse=np.sqrt(mse),
                                 ssim=0.5 + pk / 100))
    res = analyse(pd.DataFrame(rows))
    lab = res.groupby("contrast")["label"].first()
    if lab["I_D"] != "scale-dependent":
        sys.exit("self-test: I_D is additive in dB only, expected scale-dependent, got {}\n{}".format(lab["I_D"], res))
    if lab["E-D"] != "robust (same sign)" or lab["D-B"] != "robust (same sign)":
        sys.exit("self-test: main effects should be robust\n{}".format(res))
    print("INTERACTION-SCALES SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="aggregate.py's runs.csv for the block")
    ap.add_argument("--budget_runs", help="bias_variance.py's error_budget_runs_<dataset>_<n>v.csv")
    ap.add_argument("--dataset", choices=sorted(DEFAULT_VIEWS), default="llff")
    ap.add_argument("--n_views", type=int)
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not (a.runs and a.budget_runs):
        ap.error("--runs and --budget_runs are required (or --self_test)")
    nv = a.n_views or DEFAULT_VIEWS[a.dataset]
    table = per_run_table(pd.read_csv(a.runs), pd.read_csv(a.budget_runs), a.dataset, nv)
    if table.empty:
        sys.exit("no runs with both metrics and per-run MSE")
    res = analyse(table)
    os.makedirs(a.tables, exist_ok=True)
    name = "interaction_scales_{}{}.csv".format(a.dataset, "" if nv == DEFAULT_VIEWS[a.dataset] else "_{}v".format(nv))
    res.to_csv(os.path.join(a.tables, name), index=False)
    for key, _ in CONTRASTS:
        g = res[res["contrast"] == key]
        if g.empty:
            continue
        cells = ", ".join("{} {}".format(r["scale"], S.fmt_ci(r, 4 if r["scale"] in ("mse", "ssim") else 2))
                          for _, r in g.iterrows())
        print("{:<4} {}: {}".format(key, g["label"].iloc[0], cells))
    print("wrote " + os.path.join(a.tables, name))


if __name__ == "__main__":
    main()
