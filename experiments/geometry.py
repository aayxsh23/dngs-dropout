#!/usr/bin/env python
"""Multi-view depth consistency of the held-out views (PLAN_V2 §5, novelty item 2): MVC@5%.

    python experiments/geometry.py --out_root <outputs> [--tables results/tables]   # every run -> geometry_<n>v.csv
    python experiments/geometry.py --run <run folder> --self_test                    # i -> i must give 0 error

Reads eval/ours_<iter>/depth_raw/<view>.npz (alpha-weighted view-space depth and alpha, written by render.py) and
eval/ours_<iter>/cameras.json. For every ordered pair (i, j) of held-out views:
  1. Each pixel of view i with alpha > 0.5 is back-projected with depth / alpha.
  2. The point is projected into view j and compared with view j's depth at the nearest pixel.
  3. The pixel is valid if it lands inside view j and alpha_j > 0.5. It is consistent if the relative depth
     error is < 5%.
MVC@5% is the consistent fraction of valid pixels, averaged over pairs. No ground truth is needed, so the metric
doesn't favour the depth prior. Camera model (3DGS): [x, y, z, 1] @ world_view_transform gives camera
coordinates; fx = W / (2 tan(FoVx / 2)); the pixel centre of index u is at u, so cx = (W - 1) / 2.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

ALPHA_MIN, TOL = 0.5, 0.05


def load_views(eval_dir):
    cams = json.load(open(os.path.join(eval_dir, "cameras.json")))
    views = []
    for c in cams:
        d = np.load(os.path.join(eval_dir, "depth_raw", "%05d.npz" % c["index"]))
        alpha = d["alpha"].astype(np.float64)
        depth = np.where(alpha > ALPHA_MIN, d["depth"].astype(np.float64) / np.maximum(alpha, 1e-6), np.nan)
        W, H = c["width"], c["height"]
        views.append(dict(depth=depth, W=W, H=H, M=np.array(c["world_view_transform"], dtype=np.float64),
                          fx=W / (2 * math.tan(c["FoVx"] / 2)), fy=H / (2 * math.tan(c["FoVy"] / 2)),
                          cx=(W - 1) / 2.0, cy=(H - 1) / 2.0))
    return views


def pair_error(a, b):
    """Relative depth errors of view a's pixels reprojected into view b (valid pixels only)."""
    v, u = np.nonzero(np.isfinite(a["depth"]))
    z = a["depth"][v, u]
    cam = np.stack([(u - a["cx"]) / a["fx"] * z, (v - a["cy"]) / a["fy"] * z, z, np.ones_like(z)], axis=1)
    world = cam @ np.linalg.inv(a["M"])
    pb = world @ b["M"]
    zb = pb[:, 2]
    front = zb > 1e-6
    ub = np.round(b["fx"] * pb[:, 0] / np.where(front, zb, 1) + b["cx"]).astype(np.int64)
    vb = np.round(b["fy"] * pb[:, 1] / np.where(front, zb, 1) + b["cy"]).astype(np.int64)
    inside = front & (ub >= 0) & (ub < b["W"]) & (vb >= 0) & (vb < b["H"])
    target = np.full(zb.shape, np.nan)
    target[inside] = b["depth"][vb[inside], ub[inside]]
    ok = inside & np.isfinite(target)
    return np.abs(zb[ok] - target[ok]) / target[ok], ok.mean() if len(ok) else 0.0


def run_metrics(eval_dir, self_test=False):
    views = load_views(eval_dir)
    pairs = [(i, i) for i in range(len(views))] if self_test else \
        [(i, j) for i in range(len(views)) for j in range(len(views)) if i != j]
    mvc, med, valid = [], [], []
    for i, j in pairs:
        err, frac = pair_error(views[i], views[j])
        if len(err):
            mvc.append(float((err < TOL).mean()))
            med.append(float(np.median(err)))
            valid.append(float(frac))
    if not mvc:
        return None
    return dict(mvc5=float(np.mean(mvc)), median_rel_err=float(np.mean(med)), valid_frac=float(np.mean(valid)),
                n_pairs=len(mvc), n_views=len(views))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", help="folder holding llff_<n>v/<variant>/<scene>/seed<k>/")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--run", help="one run folder (for --self_test or a quick look)")
    ap.add_argument("--self_test", action="store_true", help="reproject each view into itself: error must be ~0")
    a = ap.parse_args()

    if a.run:
        found = sorted(glob.glob(os.path.join(a.run, "eval", "ours_*", "cameras.json")))
        if not found:
            sys.exit("no eval/ours_*/cameras.json under %s" % a.run)
        m = run_metrics(os.path.dirname(found[-1]), a.self_test)
        print(json.dumps(m, indent=2))
        if a.self_test and (m is None or m["median_rel_err"] > 1e-3 or m["mvc5"] < 0.999):
            sys.exit("self-test FAILED: a view reprojected into itself should agree exactly")
        return 0

    if not a.out_root:
        sys.exit("give --out_root or --run")
    rows = []
    pat = re.compile(r"llff_(\d+)v/([^/]+)/([^/]+)/seed(\d+)/eval/ours_\d+/cameras\.json$")
    for f in sorted(glob.glob(os.path.join(a.out_root, "llff_*v", "*", "*", "seed*", "eval", "ours_*", "cameras.json"))):
        m = pat.search(f.replace(os.sep, "/"))
        if not m:
            continue
        res = run_metrics(os.path.dirname(f))
        if res is None:
            print("WARNING no valid pixels: " + f, file=sys.stderr)
            continue
        res.update(n_views=int(m.group(1)), variant=m.group(2), scene=m.group(3), seed=int(m.group(4)))
        rows.append(res)
    if not rows:
        sys.exit("no runs with depth_raw under %s" % a.out_root)
    os.makedirs(a.tables, exist_ok=True)
    cols = ["variant", "scene", "seed", "n_views", "mvc5", "median_rel_err", "valid_frac", "n_pairs"]
    for nv in sorted(set(r["n_views"] for r in rows)):
        path = os.path.join(a.tables, "geometry_%dv.csv" % nv)
        with open(path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                if r["n_views"] == nv:
                    f.write(",".join(str(r[c]) for c in cols) + "\n")
        print("wrote " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
