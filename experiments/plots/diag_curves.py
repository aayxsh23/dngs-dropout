#!/usr/bin/env python
"""F2: diagnostic curves from every diag.csv (PLAN §5 W4, §7.1).

Example:
    python experiments/plots/diag_curves.py --out_root <drive>/sparse3dgs/outputs

Panel (a): gap = loss_depth_used − loss_depth_clean vs iteration for C, D and E.
Panel (b): depth_var_masks vs iteration for A–E.
Each line is the mean over scenes of the per-scene mean over seeds; the band is ± 1 std over scenes.

Writes:
    <figures>/fig_diag.pdf and .png
    <tables>/diag_summary_<n>v.csv   per variant and scene: mean gap and mean depth_var_masks over
                                     iterations >= --from_iter, averaged over seeds (stats.py reads it)
Only finished runs count: a diag.csv is skipped (with a warning) when its folder has no metrics.json or its
last row is below --final_iter, because a crashed run would bias the curves and the summary.
Also runs the W4 sharp checks on every run separately, names each failing run, and exits non-zero if
one fails (the figure and summary are still written; the summary's checks_ok column says so):
    every value is finite (a diverged run has NaN losses);
    the gap is exactly 0 in A, B and E, and not identically 0 in C and D;
    drop_rate_now is 0 in A and B and equals 0.2 * iter / 10000 in C, D and E (0.12 at iteration 6000);
    depth_var_masks is > 0 everywhere.
Needs matplotlib >= 3.3 (Colab, or any Python >= 3.8 environment), not the laptop's Python 3.7 training env.
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

COLUMNS = ["iter", "loss_rgb", "loss_depth_used", "loss_depth_clean", "depth_var_masks", "n_gaussians",
           "drop_rate_now", "iter_ms", "vram_peak_mb"]
PATH_RE = re.compile(r"llff_(\d+)v/([^/]+)/([^/]+)/seed(\d+)/diag\.csv$")


def load_diag(out_root, n_views, final_iter):
    frames = []
    for f in sorted(glob.glob(os.path.join(out_root, f"llff_{n_views}v", "*", "*", "seed*", "diag.csv"))):
        m = PATH_RE.search(f.replace(os.sep, "/"))
        if not m:
            continue
        try:
            d = pd.read_csv(f)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
            print(f"WARNING unreadable {f}: {e}", file=sys.stderr)
            continue
        missing = [c for c in COLUMNS if c not in d]
        if missing:
            print(f"WARNING {f}: missing columns {missing}; skipped", file=sys.stderr)
            continue
        d[COLUMNS] = d[COLUMNS].apply(pd.to_numeric, errors="coerce")   # a repeated header row becomes NaN
        if d["iter"].isna().any() or d["iter"].duplicated().any():
            print(f"WARNING {f}: non-numeric or repeated iterations (appended or restarted file?); skipped",
                  file=sys.stderr)
            continue
        if final_iter and not os.path.exists(os.path.join(os.path.dirname(f), "metrics.json")):
            print(f"WARNING {f}: run has no metrics.json (unfinished or failed); skipped", file=sys.stderr)
            continue
        if final_iter and (d.empty or d["iter"].max() < final_iter):
            print(f"WARNING {f}: last iteration {d['iter'].max() if len(d) else None} < {final_iter}; skipped",
                  file=sys.stderr)
            continue
        frames.append(d.assign(n_views=int(m[1]), variant=m[2], scene=m[3], seed=int(m[4])))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    return d.assign(gap=d["loss_depth_used"] - d["loss_depth_clean"])


def sharp_checks(d):
    """PLAN W4 sharp checks, per run (variant, scene, seed). Returns the set of failing (variant, scene, seed)."""
    failed = {}
    for (v, scene, seed), r in d.groupby(["variant", "scene", "seed"]):
        why = []
        if not np.isfinite(r[COLUMNS].to_numpy(dtype=float)).all():
            why.append("non-finite values")
        if v in ("A", "B", "E") and (r["gap"] != 0).any():
            why.append("gap should be exactly 0")
        if v in ("C", "D") and (r["gap"] == 0).all():
            why.append("gap identically 0: the mask never reaches the depth passes")
        if v in ("A", "B") and (r["drop_rate_now"] != 0).any():
            why.append("drop rate should be 0")
        if v in ("C", "D", "E") and not ((r["drop_rate_now"] - 0.2 * r["iter"] / 10000).abs() <= 1e-4).all():
            why.append("drop rate departs from 0.2 * iter / 10000")
        if not (r["depth_var_masks"] > 0).all():
            why.append("depth_var_masks must be > 0")
        if why:
            failed[(v, scene, seed)] = "; ".join(why)
    for v, g in d.groupby("variant"):
        last = g.sort_values("iter").groupby(["scene", "seed"])["drop_rate_now"].last()
        n_bad = sum(1 for k in failed if k[0] == v)
        print(f"  {v}: {g.groupby(['scene', 'seed']).ngroups} runs, max |gap| = {g['gap'].abs().max():.3g}, "
              f"final drop_rate_now {last.min():.4f}..{last.max():.4f}, failing runs: {n_bad}")
    for (v, scene, seed), why in sorted(failed.items()):
        print(f"  FAIL {v}/{scene}/seed{seed}: {why}")
    print("sharp checks " + ("FAILED: see the runs above" if failed else "passed"))
    return failed


def curves(d, col):
    """mean and std over scenes of the per-scene mean over seeds, per variant and iteration."""
    per_scene = d.groupby(["variant", "scene", "iter"])[col].mean()
    return per_scene.groupby(["variant", "iter"]).agg(["mean", "std", "count"]).reset_index()


def panel(ax, c, variants, title, ylabel, zero_line=False):
    items = []
    for v in variants:
        g = c[c["variant"] == v].sort_values("iter")
        if g.empty:
            continue
        ax.plot(g["iter"], g["mean"], color=style.COLORS[v], marker=style.MARKERS[v],
                markevery=max(1, len(g) // 6), markersize=4, label=style.NAMES[v])
        ax.fill_between(g["iter"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                        color=style.COLORS[v], alpha=0.15, linewidth=0)
        items.append((g["iter"].iloc[-1], g["mean"].iloc[-1], v))
    if zero_line:
        ax.axhline(0, color=style.INK_2, linewidth=0.8)
    ax.set_title(title, loc="left")
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    style.end_labels(ax, items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="folder holding llff_3v/ ...")
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--from_iter", type=int, default=1000, help="summary window start (PLAN §6: iterations >= 1000)")
    ap.add_argument("--name", default="fig_diag")
    ap.add_argument("--final_iter", type=int, default=6000,
                    help="skip runs whose diag.csv stops earlier (1100 for smoke runs; 0 keeps everything)")
    a = ap.parse_args()

    d = load_diag(a.out_root, a.n_views, a.final_iter)
    if d.empty:
        sys.exit(f"No diag.csv under {a.out_root}/llff_{a.n_views}v")
    n = d.groupby("variant")[["scene", "seed"]].nunique()
    print(f"{d.groupby(['variant', 'scene', 'seed']).ngroups} diag.csv files; scenes and seeds per variant:\n"
          + n.to_string())
    failed = sharp_checks(d)

    late = d[d["iter"] >= a.from_iter]
    summary = (late.groupby(["variant", "scene", "seed"])[["gap", "depth_var_masks"]].mean()
               .groupby(["variant", "scene"]).mean()
               .rename(columns={"gap": "gap_mean", "depth_var_masks": "var_mean"}).reset_index())
    summary.insert(0, "n_views", a.n_views)
    bad = {(v, sc) for v, sc, _ in failed}
    summary["checks_ok"] = [(v, sc) not in bad for v, sc in zip(summary["variant"], summary["scene"])]
    os.makedirs(a.tables, exist_ok=True)
    out = os.path.join(a.tables, f"diag_summary_{a.n_views}v.csv")
    summary.to_csv(out, index=False)
    print("wrote", out)

    style.apply()
    fig, (ax1, ax2) = style.plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    panel(ax1, curves(d, "gap"), ["C", "D", "E"], "(a) depth-loss gap under the dropout mask",
          "loss_depth_used − loss_depth_clean", zero_line=True)
    panel(ax2, curves(d, "depth_var_masks"), ["A", "B", "C", "D", "E"],
          "(b) depth variance across 4 probe masks", "depth_var_masks")
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False)
    style.save(fig, a.figures, a.name)
    if failed:
        sys.exit("sharp checks failed: the figure and summary were written, but fix the failing runs before using them")


if __name__ == "__main__":
    main()
