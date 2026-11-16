#!/usr/bin/env python
"""Build result tables from every metrics.json under an outputs folder.

Example:
    python experiments/aggregate.py --out_root <drive>/sparse3dgs/outputs --tables results/tables

Expects outputs/<dataset>_<n>v/<variant>/<scene>/seed<k>/metrics.json, dataset one of llff, dtu, blender (the
layout agreed on Mon 12 Oct; the dataset prefix is the multi-dataset runner's, PLAN_V2 §5.2.1).
Missing runs and missing keys produce warnings, never a crash, so a half-finished matrix still aggregates.

Writes to --tables (<tag> is <n>v for LLFF, as before, and <dataset>_<n>v for DTU and Blender):
    runs.csv              one row per finished run (everything that was found); its last column is `dataset`
    per_scene_<tag>.csv   variant x scene, mean over seeds
    main_<tag>.csv/.md/.tex   variant x metric, mean ± std over seeds; LLFF 3-view shows published baselines below a rule
    efficiency_<tag>.csv  train time, fps, Gaussians, VRAM per GPU type and variant, from quiet-GPU runs only
    missing.txt           expected runs (A–E at each dataset's main view count) that have no metrics.json yet
"""
import argparse, glob, json, os, re, sys
import pandas as pd

METRICS = ["psnr", "ssim", "lpips"]                     # lpips: lower is better
OPTIONAL_METRICS = ["ssim_sk"]                          # sklearn SSIM; --ssim ssim_sk puts it in the main table
EFFICIENCY = ["train_time_s", "fps", "n_gaussians", "model_size_mb", "vram_peak_mb"]
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
VARIANTS = ["A", "B", "C", "D", "E"]
# Per dataset: the scenes a complete seed needs, and the view count of its main block (PLAN_V2 §5.2.1).
DATASETS = {
    "llff": dict(scenes=SCENES, n_views=3),
    "dtu": dict(scenes=["scan8", "scan21", "scan30", "scan31", "scan34", "scan38", "scan40", "scan41", "scan45",
                        "scan55", "scan63", "scan82", "scan103", "scan110", "scan114"], n_views=3),
    "blender": dict(scenes=["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], n_views=8),
}
PATH_RE = re.compile(r"(llff|dtu|blender)_(\d+)v/([^/]+)/([^/]+)/seed(\d+)/metrics\.json$")


def tag(dataset, n_views):
    """File-name tag: LLFF keeps the names it always had."""
    return f"{n_views}v" if dataset == "llff" else f"{dataset}_{n_views}v"


def load_runs(out_root):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_root, "*_*v", "*", "*", "seed*", "metrics.json"))):
        m = PATH_RE.search(f.replace(os.sep, "/"))
        if not m:
            continue
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING unreadable {f}: {e}", file=sys.stderr)
            continue
        dataset, n_views, variant, scene, seed = m[1], int(m[2]), m[3], m[4], int(m[5])
        for key, from_path in [("n_views", n_views), ("variant", variant), ("scene", scene), ("seed", seed),
                               ("dataset", dataset)]:
            if key in d and d[key] != from_path:
                print(f"WARNING {f}: {key}={d[key]!r} in file but {from_path!r} in path; using path",
                      file=sys.stderr)
            d[key] = from_path
        if "train_time_s" not in d and "wall_time_s" in d:
            d["train_time_s"] = d["wall_time_s"]       # runner writes wall_time_s; train_stats.json may not exist
        missing = [k for k in METRICS if k not in d]
        if missing:
            print(f"WARNING {f}: missing {missing}", file=sys.stderr)
        d.pop("dataset", None)
        d["dataset"] = dataset                          # always the last column of runs.csv
        d["path"] = f
        rows.append(d)
    return pd.DataFrame(rows)


def fmt(mean, std, digits):
    if pd.isna(mean):
        return "–"
    return f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def main_table(df, scenes=SCENES):
    """Mean ± std over seeds of the all-scene average. A seed counts only if all its scenes finished."""
    out = []
    for var, g in df.groupby("variant"):
        per_seed = g.groupby("seed")
        complete = [s for s, gs in per_seed if set(scenes) <= set(gs["scene"])]
        if len(complete) < per_seed.ngroups:
            print(f"NOTE variant {var}: seeds {sorted(set(per_seed.groups) - set(complete))} incomplete, "
                  "left out of mean ± std", file=sys.stderr)
        g = g[g["seed"].isin(complete)]
        row = {"variant": var, "n_seeds": len(complete)}
        if complete:
            seed_means = g.groupby("seed")[[c for c in METRICS + OPTIONAL_METRICS if c in g]].mean()
            for c in METRICS + OPTIONAL_METRICS:
                if c in seed_means:
                    row[c + "_mean"] = seed_means[c].mean()
                    row[c + "_std"] = seed_means[c].std(ddof=1) if len(complete) > 1 else float("nan")
        out.append(row)
    return pd.DataFrame(out).set_index("variant")


def write_main(t, reported, path_base, ssim="ssim", digits=dict(psnr=2, ssim=3, ssim_sk=3, lpips=3)):
    t.to_csv(path_base + ".csv")
    shown = ["psnr", ssim, "lpips"]                     # papers report one SSIM, so reported rows use their "ssim"
    cols = ["PSNR ↑", "SSIM ↑", "LPIPS ↓", "seeds"]
    lines = []
    for var, r in t.iterrows():
        lines.append([var] + [fmt(r.get(c + "_mean"), r.get(c + "_std"), digits[c]) for c in shown]
                     + [str(int(r["n_seeds"]))])
    rep = []
    if reported is not None:
        for _, r in reported.iterrows():
            rep.append([f"{r['method']} (reported)"] + [fmt(r.get(c), float("nan"), digits[c]) for c in METRICS] + [""])
    md = ["| Variant | " + " | ".join(cols) + " |", "| --- " * (len(cols) + 1) + "|"]
    md += ["| " + " | ".join(l) + " |" for l in lines]
    if rep:
        md += ["| *Reported* | | | | |"] + ["| " + " | ".join(l) + " |" for l in rep]
    open(path_base + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{lcccc}", "\\toprule",
           "Variant & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ & Seeds \\\\", "\\midrule"]
    tex += [" & ".join(l).replace("±", "$\\pm$") + " \\\\" for l in lines]
    if rep:
        tex += ["\\midrule", "\\multicolumn{5}{l}{\\textit{Reported in the original papers}} \\\\"]
        tex += [" & ".join(l).replace("±", "$\\pm$") + " \\\\" for l in rep]
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(path_base + ".tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")


def missing_runs(df, n_views, seeds, dataset="llff", scenes=SCENES):
    have = set(zip(df["variant"], df["scene"], df["seed"])) if len(df) else set()
    return [f"{dataset}_{n_views}v/{v}/{s}/seed{k}" for k in seeds for v in VARIANTS for s in scenes
            if (v, s, k) not in have]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="folder holding llff_3v/, llff_2v/, dtu_3v/, blender_8v/ ...")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--reported", default="results/reported_baselines.csv")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="seeds expected, for missing.txt")
    ap.add_argument("--ssim", choices=["ssim", "ssim_sk"], default="ssim",
                    help="SSIM in the main table: ssim (3DGS) or ssim_sk (sklearn); runs.csv keeps both")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="scenes excluded by a pre-registered rule (e.g. a DTU scan with < 49 registered images); "
                         "they are not expected in complete seeds or in missing.txt")
    a = ap.parse_args()

    os.makedirs(a.tables, exist_ok=True)
    df = load_runs(a.out_root)
    if df.empty:
        sys.exit(f"No metrics.json found under {a.out_root}")
    # `dataset` last: the other columns keep the order (and LLFF runs.csv the content) they had before it existed
    df = df[[c for c in df if c not in ("dataset", "path")] + ["dataset", "path"]]
    df.drop(columns=["path"]).to_csv(os.path.join(a.tables, "runs.csv"), index=False)
    print(f"{len(df)} finished runs found")

    reported = None
    if os.path.exists(a.reported):
        reported = pd.read_csv(a.reported)
        reported = reported.rename(columns={"psnr_3v": "psnr", "ssim_3v": "ssim", "lpips_3v": "lpips"})

    def expected(ds):
        return [s for s in DATASETS[ds]["scenes"] if s not in a.exclude]

    for (ds, nv), g in df.groupby(["dataset", "n_views"]):
        tg = tag(ds, nv)
        per_scene = g.pivot_table(index="variant", columns="scene", values="psnr", aggfunc="mean")
        per_scene.round(2).to_csv(os.path.join(a.tables, f"per_scene_{tg}.csv"))

        t = main_table(g, expected(ds))
        if a.ssim + "_mean" not in t:
            print(f"WARNING no run has {a.ssim}; the SSIM column of main_{tg} is empty", file=sys.stderr)
        write_main(t, reported if (ds, nv) == ("llff", 3) else None, os.path.join(a.tables, f"main_{tg}"),
                   ssim=a.ssim)
        print(f"\n== {dict(llff='LLFF', dtu='DTU', blender='Blender')[ds]} {nv}-view (SSIM = {a.ssim}) ==")
        print(open(os.path.join(a.tables, f"main_{tg}.md"), encoding="utf-8").read())

        # efficiency only from a GPU nobody else used, and never averaged across GPU types
        quiet = g[g["shared_gpu"].fillna(False) == False] if "shared_gpu" in g else g
        quiet = quiet.assign(gpu=quiet["gpu"].fillna("unknown") if "gpu" in quiet else "unknown")
        eff_cols = [c for c in EFFICIENCY if c in quiet]
        if eff_cols and not quiet.empty:
            eff = quiet.groupby(["gpu", "variant"])[eff_cols].mean()
            eff.round(2).to_csv(os.path.join(a.tables, f"efficiency_{tg}.csv"))
            absent = sorted(set(EFFICIENCY) - set(eff_cols))
            if absent:
                print(f"NOTE efficiency columns not produced by any run yet: {absent}", file=sys.stderr)

    miss = []
    for ds in [d for d in DATASETS if d in set(df["dataset"])]:
        nv = DATASETS[ds]["n_views"]
        miss += missing_runs(df[(df["dataset"] == ds) & (df["n_views"] == nv)], nv, a.seeds, ds, expected(ds))
    open(os.path.join(a.tables, "missing.txt"), "w").write("\n".join(miss) + ("\n" if miss else ""))
    print(f"{len(miss)} expected main-block runs still missing (see missing.txt)")


if __name__ == "__main__":
    main()
