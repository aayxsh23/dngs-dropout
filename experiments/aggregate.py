#!/usr/bin/env python
"""Build result tables from every metrics.json under an outputs folder.

Example:
    python experiments/aggregate.py --out_root <drive>/sparse3dgs/outputs --tables results/tables

Expects outputs/llff_<n>v/<variant>/<scene>/seed<k>/metrics.json (the layout agreed on Mon 12 Oct).
Missing runs and missing keys produce warnings, never a crash, so a half-finished matrix still aggregates.

Writes to --tables:
    runs.csv            one row per finished run (everything that was found)
    per_scene_<n>v.csv  variant x scene, mean over seeds
    main_<n>v.csv/.md/.tex   variant x metric, mean ± std over seeds, published baselines below a rule
    efficiency_<n>v.csv train time, fps, Gaussians, VRAM per variant, from quiet-GPU runs only
    missing.txt         expected runs that have no metrics.json yet
"""
import argparse, glob, json, os, re, sys
import pandas as pd

METRICS = ["psnr", "ssim", "lpips"]                     # lpips: lower is better
EFFICIENCY = ["train_time_s", "fps", "n_gaussians", "model_size_mb", "vram_peak_mb"]
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
VARIANTS = ["A", "B", "C", "D", "E"]
PATH_RE = re.compile(r"llff_(\d+)v/([^/]+)/([^/]+)/seed(\d+)/metrics\.json$")


def load_runs(out_root):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_root, "llff_*v", "*", "*", "seed*", "metrics.json"))):
        m = PATH_RE.search(f.replace(os.sep, "/"))
        if not m:
            continue
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING unreadable {f}: {e}", file=sys.stderr)
            continue
        n_views, variant, scene, seed = int(m[1]), m[2], m[3], int(m[4])
        for key, from_path in [("n_views", n_views), ("variant", variant), ("scene", scene), ("seed", seed)]:
            if key in d and d[key] != from_path:
                print(f"WARNING {f}: {key}={d[key]!r} in file but {from_path!r} in path; using path",
                      file=sys.stderr)
            d[key] = from_path
        if "train_time_s" not in d and "wall_time_s" in d:
            d["train_time_s"] = d["wall_time_s"]       # runner writes wall_time_s; train_stats.json may not exist
        missing = [k for k in METRICS if k not in d]
        if missing:
            print(f"WARNING {f}: missing {missing}", file=sys.stderr)
        d["path"] = f
        rows.append(d)
    return pd.DataFrame(rows)


def fmt(mean, std, digits):
    if pd.isna(mean):
        return "–"
    return f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def main_table(df):
    """Mean ± std over seeds of the 8-scene average. A seed counts only if all its scenes finished."""
    out = []
    for var, g in df.groupby("variant"):
        per_seed = g.groupby("seed")
        complete = [s for s, gs in per_seed if gs["scene"].nunique() == len(SCENES)]
        if len(complete) < per_seed.ngroups:
            print(f"NOTE variant {var}: seeds {sorted(set(per_seed.groups) - set(complete))} incomplete, "
                  "left out of mean ± std", file=sys.stderr)
        g = g[g["seed"].isin(complete)]
        row = {"variant": var, "n_seeds": len(complete)}
        if complete:
            seed_means = g.groupby("seed")[[c for c in METRICS if c in g]].mean()
            for c in METRICS:
                if c in seed_means:
                    row[c + "_mean"] = seed_means[c].mean()
                    row[c + "_std"] = seed_means[c].std(ddof=1) if len(complete) > 1 else float("nan")
        out.append(row)
    return pd.DataFrame(out).set_index("variant")


def write_main(t, reported, path_base, digits=dict(psnr=2, ssim=3, lpips=3)):
    t.to_csv(path_base + ".csv")
    cols = ["PSNR ↑", "SSIM ↑", "LPIPS ↓", "seeds"]
    lines = []
    for var, r in t.iterrows():
        lines.append([var] + [fmt(r.get(c + "_mean"), r.get(c + "_std"), digits[c]) for c in METRICS]
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


def missing_runs(df, n_views, seeds):
    have = set(zip(df["variant"], df["scene"], df["seed"])) if len(df) else set()
    return [f"llff_{n_views}v/{v}/{s}/seed{k}" for k in seeds for v in VARIANTS for s in SCENES
            if (v, s, k) not in have]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="folder holding llff_3v/, llff_2v/, ...")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--reported", default="results/reported_baselines.csv")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="seeds expected, for missing.txt")
    a = ap.parse_args()

    os.makedirs(a.tables, exist_ok=True)
    df = load_runs(a.out_root)
    if df.empty:
        sys.exit(f"No metrics.json found under {a.out_root}")
    df.drop(columns=["path"]).to_csv(os.path.join(a.tables, "runs.csv"), index=False)
    print(f"{len(df)} finished runs found")

    reported = None
    if os.path.exists(a.reported):
        reported = pd.read_csv(a.reported)
        reported = reported.rename(columns={"psnr_3v": "psnr", "ssim_3v": "ssim", "lpips_3v": "lpips"})

    for nv, g in df.groupby("n_views"):
        per_scene = g.pivot_table(index="variant", columns="scene", values="psnr", aggfunc="mean")
        per_scene.round(2).to_csv(os.path.join(a.tables, f"per_scene_{nv}v.csv"))

        t = main_table(g)
        write_main(t, reported if nv == 3 else None, os.path.join(a.tables, f"main_{nv}v"))
        print(f"\n== LLFF {nv}-view ==")
        print(open(os.path.join(a.tables, f"main_{nv}v.md"), encoding="utf-8").read())

        quiet = g[g.get("shared_gpu", False) == False]          # efficiency only from a GPU nobody else used
        eff_cols = [c for c in EFFICIENCY if c in quiet]
        if eff_cols and not quiet.empty:
            eff = quiet.groupby("variant")[eff_cols].mean()
            eff.round(2).to_csv(os.path.join(a.tables, f"efficiency_{nv}v.csv"))
            absent = sorted(set(EFFICIENCY) - set(eff_cols))
            if absent:
                print(f"NOTE efficiency columns not produced by any run yet: {absent}", file=sys.stderr)

    miss = missing_runs(df[df["n_views"] == 3], 3, a.seeds)
    open(os.path.join(a.tables, "missing.txt"), "w").write("\n".join(miss) + ("\n" if miss else ""))
    print(f"{len(miss)} expected 3-view runs still missing (see missing.txt)")


if __name__ == "__main__":
    main()
