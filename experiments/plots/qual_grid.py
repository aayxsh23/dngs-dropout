#!/usr/bin/env python
"""F3 and F4: qualitative grid, scenes x [GT, variants], an RGB row and a depth row per scene (PLAN §7.1).

Examples:
    python experiments/plots/qual_grid.py --out_root <drive>/sparse3dgs/outputs                    # F3
    python experiments/plots/qual_grid.py --out_root <drive>/sparse3dgs/outputs \\
        --scenes horns orchids --variants B D E --name fig_fail                                    # F4

Reads <out_root>/llff_<n>v/<variant>/<scene>/seed<k>/eval/ours_<iter>/{gt,renders,depth}/<view>.png,
written by render.py. The depth row uses render.py's turbo colour map (depth/color_<view>.png) and falls
back to the grey depth map. Each render is labelled with its PSNR on this one view, computed from the
8-bit PNGs, so it is close to but not the same as the number in results_eval.json.
Writes <figures>/<name>.pdf and .png.
"""
import argparse, glob, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402


def eval_dir(out_root, n_views, variant, scene, seed):
    """The eval/ours_<iter> folder with the highest iteration, or None."""
    found = glob.glob(os.path.join(out_root, f"llff_{n_views}v", variant, scene, f"seed{seed}", "eval", "ours_*"))
    found = [f for f in found if re.search(r"ours_(\d+)$", f)]
    return max(found, key=lambda f: int(re.search(r"ours_(\d+)$", f).group(1))) if found else None


def read(path):
    if path is None or not os.path.exists(path):
        return None
    im = mpimg.imread(path)
    return im[..., :3] if im.ndim == 3 else np.stack([im] * 3, -1)


def psnr(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def blank(ax, text):
    ax.set_facecolor("#f1f0ec")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=7, color=style.INK_2, transform=ax.transAxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--scenes", nargs="+", default=["fern", "horns", "trex"])
    ap.add_argument("--variants", nargs="+", default=["B", "C", "D", "E"])
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", type=int, default=0, help="index of the eval image (render.py names them 00000.png ...)")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--name", default="fig_qual")
    ap.add_argument("--width", type=float, default=7.0, help="figure width in inches")
    a = ap.parse_args()

    name = "{0:05d}.png".format(a.view)
    cols = ["GT"] + a.variants
    cells, aspect, missing = [], None, []
    for scene in a.scenes:
        dirs = {v: eval_dir(a.out_root, a.n_views, v, scene, a.seed) for v in a.variants}
        gt_dir = next((d for d in dirs.values() if d), None)
        gt = read(os.path.join(gt_dir, "gt", name)) if gt_dir else None
        rgb, dep = [gt], [None]
        for v in a.variants:
            d = dirs[v]
            if d is None:
                missing.append(f"{v}/{scene}/seed{a.seed}")
            rgb.append(read(os.path.join(d, "renders", name)) if d else None)
            dmap = read(os.path.join(d, "depth", "color_" + name)) if d else None
            if d and dmap is None:
                dmap = read(os.path.join(d, "depth", name))
            dep.append(dmap)
        if aspect is None:
            first = next((im for im in rgb if im is not None), None)
            if first is not None:
                aspect = first.shape[0] / first.shape[1]
        cells.append((scene, rgb, dep, gt))
    if missing:
        print("WARNING no eval/ folder for: " + ", ".join(missing), file=sys.stderr)
    if aspect is None:
        sys.exit(f"No images found under {a.out_root}/llff_{a.n_views}v for seed {a.seed}")

    style.apply()
    cw = a.width / len(cols)
    fig, axes = style.plt.subplots(2 * len(a.scenes), len(cols), figsize=(a.width, cw * aspect * 2 * len(a.scenes)),
                                   gridspec_kw=dict(wspace=0.02, hspace=0.04), squeeze=False)
    for i, (scene, rgb, dep, gt) in enumerate(cells):
        for j, col in enumerate(cols):
            for k, (row, im) in enumerate((("RGB", rgb[j]), ("depth", dep[j]))):
                ax = axes[2 * i + k, j]
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.grid(False)
                if im is not None:
                    ax.imshow(im, interpolation="lanczos")
                else:
                    blank(ax, "no GT depth" if (j == 0 and row == "depth") else "missing")
                if i == 0 and k == 0:
                    ax.set_title(col, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{scene}\n{row}" if k == 0 else row, fontsize=7)
                if k == 0 and j > 0:
                    p = psnr(im, gt)
                    if p is not None:
                        ax.text(0.98, 0.03, f"{p:.2f} dB", ha="right", va="bottom", fontsize=6, color="white",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, lw=0))
    style.save(fig, a.figures, a.name)


if __name__ == "__main__":
    main()
