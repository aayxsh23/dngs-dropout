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
Also prints the W4 sharp checks: the gap must be exactly 0 in A, B and E, and drop_rate_now must end
at 0 in A and B and at 0.12 in C, D and E.
"""
import argparse, glob, os, re, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

COLUMNS = ["iter", "loss_rgb", "loss_depth_used", "loss_depth_clean", "depth_var_masks", "n_gaussians",
           "drop_rate_now", "iter_ms", "vram_peak_mb"]
PATH_RE = re.compile(r"llff_(\d+)v/([^/]+)/([^/]+)/seed(\d+)/diag\.csv$")


def load_diag(out_root, n_views):
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
        frames.append(d.assign(n_views=int(m[1]), variant=m[2], scene=m[3], seed=int(m[4])))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    return d.assign(gap=d["loss_depth_used"] - d["loss_depth_clean"])


def sharp_checks(d):
    ok = True
    for v, g in d.groupby("variant"):
        top = g["gap"].abs().max()
        last = g.sort_values("iter").groupby(["scene", "seed"])["drop_rate_now"].last()
        line = f"  {v}: max |gap| = {top:.3g}, final drop_rate_now {last.min():.4f}..{last.max():.4f}"
        if v in ("A", "B", "E") and top != 0:
            line += "   <-- expected exactly 0"
            ok = False
        if v in ("A", "B") and last.abs().max() != 0:
            line += "   <-- expected drop rate 0"
            ok = False
        print(line)
    print("sharp checks " + ("passed" if ok else "FAILED: see the marked lines"))


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
    a = ap.parse_args()

    d = load_diag(a.out_root, a.n_views)
    if d.empty:
        sys.exit(f"No diag.csv under {a.out_root}/llff_{a.n_views}v")
    n = d.groupby("variant")[["scene", "seed"]].nunique()
    print(f"{d.groupby(['variant', 'scene', 'seed']).ngroups} diag.csv files; scenes and seeds per variant:\n"
          + n.to_string())
    sharp_checks(d)

    late = d[d["iter"] >= a.from_iter]
    summary = (late.groupby(["variant", "scene", "seed"])[["gap", "depth_var_masks"]].mean()
               .groupby(["variant", "scene"]).mean()
               .rename(columns={"gap": "gap_mean", "depth_var_masks": "var_mean"}).reset_index())
    summary.insert(0, "n_views", a.n_views)
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
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    style.save(fig, a.figures, a.name)


if __name__ == "__main__":
    main()
