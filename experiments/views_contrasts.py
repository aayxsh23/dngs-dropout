#!/usr/bin/env python
"""Block V with all 3 seeds: paired PSNR contrasts per view count (writes results/v2/tables/views/contrasts_3seed.csv).

    python experiments/views_contrasts.py --views_runs results/v2/tables/views/runs.csv \\
        --t1_runs results/v2/tables/T1/runs.csv --out results/v2/tables/views/contrasts_3seed.csv

2/6/9 views come from block V, 3 views from T1 (seeds 0-2 each). Per view count: the mean over seeds per (scene, variant),
then each contrast as the mean of the per-scene differences with a 95 % t-interval over the 8 scenes (df 7) and wins,
using stats.paired (PLAN_V2 §5 unit and pairing). Unadjusted: no Holm across view counts. Report §5.4 and Table IX quote it.
With scipy installed (as when the committed file was made) the t-quantile is scipy's; without it, stats.t975's fallback can
move the 4th decimal of the CI bounds.
"""
import argparse, os, sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as S  # noqa: E402

CONTRASTS = [
    ("B-A", {"B": 1, "A": -1}), ("C-A", {"C": 1, "A": -1}), ("D-B", {"D": 1, "B": -1}), ("E-B", {"E": 1, "B": -1}),
    ("E-D", {"E": 1, "D": -1}), ("I_D", {"D": 1, "C": -1, "B": -1, "A": 1}), ("I_E", {"E": 1, "C": -1, "B": -1, "A": 1}),
]


def table(runs, metric="psnr"):
    rows = []
    for nv, g in sorted(runs.groupby("n_views")):
        sm = g.groupby(["scene", "variant"])[metric].mean().unstack("variant")
        seeds = sorted(g["seed"].unique())
        for name, coef in CONTRASTS:
            r = S.paired(sum(c * sm[v] for v, c in coef.items()))
            rows.append(dict(n_views=nv, contrast=name, metric=metric, n_scenes=r["n"],
                             seeds="%d-%d" % (seeds[0], seeds[-1]), mean=r["mean"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                             wins=r["wins"]))
    return pd.DataFrame(rows).round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views_runs", default="results/v2/tables/views/runs.csv")
    ap.add_argument("--t1_runs", default="results/v2/tables/T1/runs.csv")
    ap.add_argument("--out", default="results/v2/tables/views/contrasts_3seed.csv")
    a = ap.parse_args()
    keep = ["variant", "scene", "seed", "n_views", "psnr"]
    runs = pd.concat([pd.read_csv(a.views_runs)[keep], pd.read_csv(a.t1_runs)[keep]], ignore_index=True)
    table(runs[runs["variant"].isin(list("ABCDE"))]).to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
