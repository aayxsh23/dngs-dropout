#!/usr/bin/env python
"""Resumable experiment runner (PLAN.md W8, PLAN_V2 §3).

    python experiments/run_matrix.py --matrix experiments/matrix.yaml --data_root data/llff \\
        --out_root outputs --shard 0/2 --seeds 0 [--work_root /content/work] [--keep_ckpt_seeds 0] [--dry_run]

Each run is (n_views, seed, variant, scene) -> <out_root>/llff_<n>v/<variant>/<scene>/seed<k>/. A run is done when
its metrics.json exists; that file is written last, so a run killed or half-copied is redone next time. A STOP file
in the repo root lets the run in progress finish and starts no new one. The runner refuses to run unless HEAD is
exactly at a tag (AGENTS.md invariant 5); --allow_untagged is for smoke tests only. Python 3.7 compatible; it
needs no torch (GPU name, commit and torch version come from the run's own config.json / train_stats.json).
"""
import argparse
import glob
import itertools
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_KEYS = ["variant", "scene", "n_views", "seed", "psnr", "ssim", "ssim_sk", "lpips", "train_time_s", "fps",
                "n_gaussians", "model_size_mb", "vram_peak_mb", "wall_time_s", "git_commit", "code_tag", "gpu",
                "shared_gpu", "torch_version", "drop_rate", "depth_scale", "depth_render"]


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


def code_tag(allow_untagged):
    """The tag at HEAD. A tagged run must also have no changes to tracked files (AGENTS.md invariant 5)."""
    out = git("describe", "--tags", "--exact-match", "--match", "v*-exp", "HEAD")   # invariant 5: only v1.x-exp tags
    dirty = bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip())
    if out.returncode == 0 and not dirty:
        return out.stdout.strip()
    if allow_untagged:
        return "untagged-dirty" if dirty else "untagged"
    sys.exit("HEAD must be exactly at a tag with no changes to tracked files: experiments run only from a tag "
             "(AGENTS.md invariant 5). Check out the tag, or pass --allow_untagged for a smoke test.")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def runs_for(cfg, seeds):
    return sorted(itertools.product(cfg["n_views"], seeds, sorted(cfg["variants"]), cfg["scenes"]))


def cleanup(work, seed, keep_ckpt_seeds):
    """Delete what the analysis never reads (PLAN W8 step 9)."""
    for f in glob.glob(os.path.join(work, "events.out.tfevents*")):
        os.remove(f)
    for d in ("train", "test"):
        shutil.rmtree(os.path.join(work, d), ignore_errors=True)
    if seed not in keep_ckpt_seeds:
        for f in glob.glob(os.path.join(work, "*.pth")):
            os.remove(f)
        shutil.rmtree(os.path.join(work, "point_cloud"), ignore_errors=True)


def build_metrics(work, variant, scene, n_views, seed, wall, tag, shared_gpu):
    res = load_json(os.path.join(work, "results_eval.json"))
    if len(res) != 1:
        raise RuntimeError("results_eval.json must hold exactly one ours_* entry, found %d" % len(res))
    m = next(iter(res.values()))
    train = load_json(os.path.join(work, "train_stats.json"))
    rstats = load_json(os.path.join(work, "render_stats.json"))
    config = load_json(os.path.join(work, "config.json"))
    for name, d in (("train_stats.json", train), ("render_stats.json", rstats), ("config.json", config)):
        if not d:
            raise RuntimeError("%s is missing or unreadable" % name)
    for key in ("PSNR", "SSIM", "LPIPS"):
        v = m.get(key)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise RuntimeError("results_eval.json has no finite %s (%r)" % (key, v))
    rec = dict(train)   # all train_stats keys, incl. the C3 structure statistics
    rec.update(variant=variant, scene=scene, n_views=n_views, seed=seed,
               psnr=m.get("PSNR"), ssim=m.get("SSIM"), ssim_sk=m.get("SSIM_sk"), lpips=m.get("LPIPS"),
               fps=rstats.get("fps"), wall_time_s=round(wall, 1), git_commit=config.get("git_commit"),
               code_tag=tag, gpu=config.get("gpu"), shared_gpu=shared_gpu)
    for k in ("drop_rate", "depth_scale", "depth_render"):
        rec.setdefault(k, config.get(k))
    for k in METRICS_KEYS:
        rec.setdefault(k, None)
    return rec


def mark_failed(work, out, status):
    """W8 step 7: FAILED marker plus run.log next to the outputs (also on Drive); the run is redone next time."""
    for d in {work, out}:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "FAILED"), "w") as f:
            f.write(status + "\n")
    if work != out:
        shutil.copy(os.path.join(work, "run.log"), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="experiments/matrix.yaml")
    ap.add_argument("--data_root", required=True, help="folder holding the scene folders")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--work_root", default=None, help="train here, then copy to out_root (Colab: /content/work)")
    ap.add_argument("--shard", default="0/1", help="i/N: take runs i, i+N, i+2N, ...")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="override the matrix's seeds")
    ap.add_argument("--keep_ckpt_seeds", type=int, nargs="*", default=None,
                    help="seeds whose checkpoint and point cloud are kept (default: the matrix's keep_ckpt_seeds, "
                         "else none); give the flag with no values to keep none")
    ap.add_argument("--shared_gpu", action="store_true", help="several runs share this GPU (A100 slots)")
    ap.add_argument("--allow_untagged", action="store_true", help="smoke tests only")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    a.data_root, a.out_root = os.path.abspath(a.data_root), os.path.abspath(a.out_root)   # steps run with cwd=REPO
    a.work_root = os.path.abspath(a.work_root) if a.work_root else None

    with open(a.matrix) as f:
        cfg = yaml.safe_load(f)
    keep = a.keep_ckpt_seeds if a.keep_ckpt_seeds is not None else cfg.get("keep_ckpt_seeds", [])
    seeds = a.seeds if a.seeds else cfg["seeds"]
    i, n = (int(x) for x in a.shard.split("/"))
    runs = runs_for(cfg, seeds)
    mine = runs[i::n]
    tag = code_tag(a.allow_untagged or a.dry_run)   # a dry run only prints, so it may preview an untagged checkout
    work_root = a.work_root or a.out_root
    print("%s: shard %s, seeds %s: %d of %d runs, tag %s, keeping checkpoints for seeds %s"
          % (a.matrix, a.shard, seeds, len(mine), len(runs), tag, keep), flush=True)

    done, skipped, failed = 0, 0, []
    for nv, seed, variant, scene in mine:
        if os.path.exists(os.path.join(REPO, "STOP")):
            print("STOP file found: no new runs", flush=True)
            break
        rel = os.path.join("llff_%dv" % nv, variant, scene, "seed%d" % seed)
        out, work = os.path.join(a.out_root, rel), os.path.join(work_root, rel)
        if os.path.exists(os.path.join(out, "metrics.json")):
            skipped += 1
            continue
        flags = " ".join("--%s %s" % (k, v) for k, v in cfg["variants"][variant].items())
        q = shlex.quote   # paths may contain spaces ("/content/drive/My Drive"); the steps go through shlex.split
        fill = dict(python=q(sys.executable), scene_dir=q(os.path.join(a.data_root, scene)), work=q(work), n_views=nv,
                    seed=seed, variant_flags=flags, train_flags=" ".join(str(cfg.get("train_flags", "")).split()))
        cmds = [step.format(**fill) for step in cfg["steps"]]
        if a.dry_run:
            print("\n".join(">> " + c for c in cmds), flush=True)
            continue
        for d in {out, work}:   # leftovers of a killed run (no metrics.json)
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(work)
        t0 = time.time()
        status = "ok"
        with open(os.path.join(work, "run.log"), "a") as log:
            for k, cmd in enumerate(cmds):
                log.write(">> %s\n" % cmd)
                log.flush()
                rc = subprocess.run(shlex.split(cmd), stdout=log, stderr=subprocess.STDOUT, cwd=REPO).returncode
                if rc != 0:
                    status = "step %d exited %d" % (k, rc)
                    break
        wall = time.time() - t0
        if status != "ok":
            mark_failed(work, out, status)
            failed.append(rel)
            print("FAILED %s (%s, %.0f s)" % (rel, status, wall), flush=True)
            continue
        try:
            rec = build_metrics(work, variant, scene, nv, seed, wall, tag, a.shared_gpu)
        except Exception as e:  # a run without usable results is a failure, not a done run
            mark_failed(work, out, "results: %s" % e)
            failed.append(rel)
            print("FAILED %s (results: %s)" % (rel, e), flush=True)
            continue
        cleanup(work, seed, keep)
        if work != out:
            shutil.copytree(work, out)
        tmp = os.path.join(out, "metrics.json.tmp")
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=2)
        os.replace(tmp, os.path.join(out, "metrics.json"))   # written last and atomically: the run now counts as done
        done += 1
        print("done %s  psnr %.2f  %.0f s" % (rel, rec["psnr"], wall), flush=True)

    print("\nsummary: %d done, %d already done, %d failed" % (done, skipped, len(failed)), flush=True)
    for r in failed:
        print("  failed: " + r)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
