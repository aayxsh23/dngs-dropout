#!/usr/bin/env python
"""Masked DTU metrics for one run, in the frozen results_eval.json format (EXECUTION_PLAN_15-21Nov A7).

    python experiments/eval_dtu.py --model_path <run dir> --mask_dir data/dtu/submission_data/idrmasks/<scan>

Reads <run>/eval/ours_<iter>/{renders,gt}/*.png (written by render.py; eval views = DTU's 25 test views) and the
RegNeRF object masks of that scan. Like metrics_dtu.py: masks are downsized by 4, renders and GT are resized to the
mask, the background is composited to white, PSNR is computed on mask pixels only, SSIM / SSIM_sk / LPIPS (VGG) on
the composited images. Writes
    <run>/results_eval.json   {"ours_<iter>": {"PSNR", "SSIM", "SSIM_sk", "LPIPS", "masked": true}}
    <run>/per_view_eval.json  per-image values
    <run>/mask/<render name>  the masks used, so bias_variance.py can read the same pixels
No dropout anywhere (AGENTS.md invariant 1). Python 3.7 compatible.

Mask files: either exactly one per eval render (sorted order = render order, as scripts/copy_mask_dtu.sh assumes; the
RegNeRF masks name them by test index, 001.png, 002.png, 009.png, ...), or one per DTU image named by its index (IDR
layout, scan110/scan114: mask/000.png ... 063.png), from which the 25 test indices of scene/dataset_readers.py are
taken. Anything else stops.
"""
import argparse
import json
import os
import re
import shutil
import sys

DTU_TRAIN = [25, 22, 28, 40, 44, 48, 0, 8, 13]
DTU_EXCLUDE = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
DTU_TEST = [i for i in range(49) if i not in DTU_TRAIN + DTU_EXCLUDE]   # 25 views, as scene/dataset_readers.py


def mask_files(mask_dir, n_renders):
    """The mask file for each eval render, in render order."""
    d = os.path.join(mask_dir, "mask") if os.path.isdir(os.path.join(mask_dir, "mask")) else mask_dir
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))
    if len(files) == n_renders:
        return [os.path.join(d, f) for f in files]
    # Otherwise one mask per DTU image, named by its 0-based index (IDR layout: <scan>/mask/000.png ..., 49 or 64 files,
    # e.g. scan110, scan114): take the test indices, as scripts/copy_mask_dtu.sh does via scan8's file names.
    by_idx = {}
    for f in files:
        digits = re.sub(r"\D", "", os.path.splitext(f)[0])
        if digits:
            by_idx[int(digits)] = f
    missing = [i for i in DTU_TEST if i not in by_idx]
    if missing:
        sys.exit("%s: %d mask files, not one per eval render, and no mask for test indices %s" % (d, len(files), missing))
    return [os.path.join(d, by_idx[i]) for i in DTU_TEST]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--mask_dir", required=True, help="idrmasks/<scan> (with or without a mask/ subfolder)")
    a = ap.parse_args()

    import torch
    import torchvision.transforms.functional as tf
    from PIL import Image
    from skimage.metrics import structural_similarity
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lpipsPyTorch import lpips
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    eval_dir = os.path.join(a.model_path, "eval")
    methods = [m for m in os.listdir(eval_dir) if m.startswith("ours_")]
    if len(methods) != 1:
        sys.exit("%s: expected exactly one ours_* folder, found %s" % (eval_dir, methods))
    method = methods[0]
    renders_dir = os.path.join(eval_dir, method, "renders")
    gt_dir = os.path.join(eval_dir, method, "gt")
    names = sorted(os.listdir(renders_dir))
    if len(names) != len(DTU_TEST):
        sys.exit("%s: %d renders; DTU has %d eval views" % (renders_dir, len(names), len(DTU_TEST)))
    masks = mask_files(a.mask_dir, len(names))
    out_mask = os.path.join(a.model_path, "mask")
    os.makedirs(out_mask, exist_ok=True)

    vals = {"PSNR": [], "SSIM": [], "SSIM_sk": [], "LPIPS": []}
    with torch.no_grad():
        for name, mf in zip(names, masks):
            shutil.copy(mf, os.path.join(out_mask, name))
            mask = Image.open(mf)
            mask = mask.resize((mask.size[0] // 4, mask.size[1] // 4))
            render = Image.open(os.path.join(renders_dir, name)).resize(mask.size)
            gt = Image.open(os.path.join(gt_dir, name)).resize(mask.size)
            m = tf.to_tensor(mask).unsqueeze(0)[:, :3, :, :].cuda()
            if m.shape[1] == 1:
                m = m.repeat(1, 3, 1, 1)
            r = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda() * m + (1 - m)
            g = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda() * m + (1 - m)
            sel = m == 1.0
            vals["PSNR"].append(psnr(r[sel][None, ...], g[sel][None, ...]).item())
            vals["SSIM"].append(ssim(r, g).item())
            vals["SSIM_sk"].append(float(structural_similarity(r[0].permute(1, 2, 0).cpu().numpy(),
                                                               g[0].permute(1, 2, 0).cpu().numpy(),
                                                               multichannel=True, channel_axis=2, data_range=1.0)))
            vals["LPIPS"].append(lpips(r, g, net_type="vgg").item())

    summary = {k: sum(v) / len(v) for k, v in vals.items()}
    summary["masked"] = True
    with open(os.path.join(a.model_path, "results_eval.json"), "w") as f:
        json.dump({method: summary}, f, indent=2)
    with open(os.path.join(a.model_path, "per_view_eval.json"), "w") as f:
        json.dump({method: {k: dict(zip(names, v)) for k, v in vals.items()}}, f, indent=2)
    print("%s %s: PSNR %.3f SSIM %.4f LPIPS %.4f (masked, %d views)"
          % (a.model_path, method, summary["PSNR"], summary["SSIM"], summary["LPIPS"], len(names)))


if __name__ == "__main__":
    main()
