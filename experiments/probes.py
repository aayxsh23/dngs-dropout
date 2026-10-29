#!/usr/bin/env python
"""Offline probes on a trained run (PLAN_V2 §5, novelty items 1 and 3). Never writes into eval/ (invariant 1).

    python experiments/probes.py -s data/llff/fern --model_path <run with chkpnt_latest.pth and point_cloud/> [--near 10]

Writes <model_path>/probes/:
  robustness.csv   test-time robustness: eval PSNR after removing a random fraction f of the Gaussians,
                   with and without the 1/(1-f) opacity compensation, 5 seeded masks per point
  depth_bias.json  Monte-Carlo bias of the hard-depth pass (opacity fixed at 0.95) under compensated dropout:
                   the median over pixels of (mean over 32 masks of masked depth - clean depth) / clean depth,
                   at r = 0.12 and 0.20. PLAN_V2 §2.4 predicts a positive ("far") shift that grows with r,
                   because 0.95 / (1 - r) exceeds the rasterizer's 0.99 alpha cap.
The model is loaded and near-pruned exactly as render.py does, so f = 0 reproduces the eval renders (up to the
8-bit PNG rounding that metrics.py sees). Masks come from a dedicated generator, so results are deterministic.
"""
import json
import os
import sys
from argparse import ArgumentParser

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render, render_for_depth  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import load_checkpoint, safe_state  # noqa: E402
from utils.image_utils import psnr  # noqa: E402

FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3]
ROBUST_MASKS, BIAS_MASKS, BIAS_RATES = 5, 32, [0.12, 0.2]


def keep_mask(n, f, compensate, gen):
    keep = (torch.rand(n, generator=gen, device="cuda") >= f).float()
    return keep / (1.0 - f) if compensate else keep


def load(dataset, near):
    gaussians = GaussianModel(dataset.sh_degree)
    (model_params, _) = load_checkpoint(os.path.join(dataset.model_path, "chkpnt_latest.pth"))
    gaussians.restore(model_params)
    gaussians.neural_renderer.keep_sigma = True
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    views = scene.getEvalCameras()
    if near > 0:   # as render.py: drop Gaussians closer than `near` to any eval camera
        mask_near = None
        for view in views:
            m = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True) < near
            mask_near = mask_near + m if mask_near is not None else m
        gaussians.prune_points_inference(mask_near)
    return gaussians, views


def robustness(gaussians, views, pipe, bg, gen):
    n = gaussians.get_xyz.shape[0]
    rows = []
    for compensate in (True, False):
        for f in FRACTIONS:
            vals = []
            for _ in range(1 if f == 0 else ROBUST_MASKS):
                comp = None if f == 0 else keep_mask(n, f, compensate, gen)
                per_view = [psnr(render(v, gaussians, pipe, bg, inference=True, drop_comp=comp)["render"].clamp(0, 1),
                                 v.original_image[0:3].cuda()).mean().item() for v in views]
                vals.append(sum(per_view) / len(per_view))
            rows.append(dict(compensated=compensate, fraction=f, psnr=sum(vals) / len(vals),
                             psnr_min=min(vals), psnr_max=max(vals), masks=len(vals)))
    return rows


def depth_bias(gaussians, views, pipe, bg, gen):
    n = gaussians.get_xyz.shape[0]
    out = {}
    for r in BIAS_RATES:
        shifts = []
        for v in views:
            clean = render_for_depth(v, gaussians, pipe, bg)
            valid = clean["alpha"] > 0.5
            z0 = clean["depth"] / clean["alpha"].clamp_min(1e-6)
            acc = torch.zeros_like(clean["depth"])
            for _ in range(BIAS_MASKS):
                pkg = render_for_depth(v, gaussians, pipe, bg, drop_comp=keep_mask(n, r, True, gen))
                acc += pkg["depth"] / pkg["alpha"].clamp_min(1e-6)
            rel = ((acc / BIAS_MASKS - z0) / z0)[valid]
            shifts.append(rel.median().item())
        out["r=%.2f" % r] = dict(median_rel_shift=sum(shifts) / len(shifts), per_view=shifts)
    return out


def main():
    parser = ArgumentParser(description="Offline probes on a trained run")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--near", default=0, type=int)
    parser.add_argument("--probe_seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    out_dir = os.path.join(dataset.model_path, "probes")
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        gaussians, views = load(dataset, args.near)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(args.probe_seed)
        rows = robustness(gaussians, views, pipe, bg, gen)
        bias = depth_bias(gaussians, views, pipe, bg, gen)
    with open(os.path.join(out_dir, "robustness.csv"), "w") as f:
        cols = ["compensated", "fraction", "psnr", "psnr_min", "psnr_max", "masks"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    with open(os.path.join(out_dir, "depth_bias.json"), "w") as f:
        json.dump(dict(bias, n_gaussians=int(gaussians.get_xyz.shape[0]), n_views=len(views)), f, indent=2)
    print("wrote %s/robustness.csv and depth_bias.json" % out_dir)


if __name__ == "__main__":
    main()
