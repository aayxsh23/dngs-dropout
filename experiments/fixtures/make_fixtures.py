#!/usr/bin/env python
"""Write fake run outputs in the agreed layout, to test the analysis scripts before real runs exist.

python experiments/fixtures/make_fixtures.py fake_outputs
python experiments/aggregate.py --out_root fake_outputs --tables fake_tables
python experiments/plots/diag_curves.py --out_root fake_outputs --tables fake_tables --figures fake_figures
python experiments/stats.py --runs fake_tables/runs.csv --tables fake_tables --diag_summary fake_tables/diag_summary_3v.csv
python experiments/plots/qual_grid.py --out_root fake_outputs --figures fake_figures
python experiments/plots/views.py --out_root fake_outputs --tables fake_tables --figures fake_figures
Numbers are random and mean nothing. Seed 2 is deliberately incomplete and two files have quirks.
Seed 1 runs on a quiet laptop GPU, so the efficiency table must keep it apart from seed 0's T4.
Also written: a diag.csv and config.json for every 3-view run, Tier 2 metrics (2, 6 and 9 views, B-E, seed 0),
and tiny eval/ours_6000 images for seed 0 of fern, horns and trex.

The other run blocks (PLAN_V2 §3.1) go next to the T1 folder, as on Drive:
    fake_outputs_pass/   block P: D_hard, D_soft, 8 scenes x seeds 0-2
    fake_outputs_dose/   block G: C/D/E at γ = 0.1 (_g010) and 1/3 (_g033), 8 scenes x seed 0
    fake_outputs_noise/  block N: B_r1..B_r4 and D_r1..D_r4 on fern, seed 0
Test them with the same scripts, one aggregate.py per block, then stats.py with --pass_runs, --dose_runs and
--noise_runs (see stats.py), and diag_curves.py --out_root on each block folder.
"""
import json, os, random, struct, sys, zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factors as F  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "fake_outputs"


def flags(var):
    f = F.factors(var)
    return dict(drop_rate=f["drop_rate"], depth_scale=f["depth_scale"], depth_render=f["depth_render"])


def write_config(out, var, seed):
    json.dump(dict(flags(var), seed=seed, iterations=6000, soft_depth_start=1000, n_sparse=3),
              open(f"{out}/config.json", "w"), indent=2)
scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
base = dict(A=18.0, B=19.1, C=18.6, D=18.8, E=19.5)
rng = random.Random(0)
for seed in [0, 1, 2]:
    for var, p in base.items():
        for i, scene in enumerate(scenes):
            if seed == 2 and i >= 5:              # seed 2 unfinished: only 5 scenes
                continue
            out = f"{root}/llff_3v/{var}/{scene}/seed{seed}"
            os.makedirs(out, exist_ok=True)
            laptop = seed == 1
            m = dict(variant=var, scene=scene, n_views=3, seed=seed,
                     psnr=round(p + rng.gauss(0, 0.3), 3), ssim=round(0.6 + rng.gauss(0, 0.02), 4),
                     lpips=round(0.29 + rng.gauss(0, 0.02), 4),
                     train_time_s=(900 if laptop else 600) + 120 * (var == "E"),
                     n_gaussians=rng.randint(80000, 120000), vram_peak_mb=3000.0,
                     git_commit="abc1234", gpu="NVIDIA GeForce RTX 3050 Ti Laptop GPU" if laptop else "Tesla T4",
                     shared_gpu=seed == 2)
            m["ssim_sk"] = round(m["ssim"] - 0.006, 4)
            m.update(flags(var))
            if (var, scene, seed) == ("C", "room", 1):
                del m["lpips"]                    # missing key: should warn, not crash
            if (var, scene, seed) == ("D", "trex", 0):
                m["wall_time_s"] = m.pop("train_time_s")   # what run_matrix.py writes today
            json.dump(m, open(f"{out}/metrics.json", "w"), indent=2)
            write_config(out, var, seed)

# diag.csv (PLAN W4 columns), one row per 100 iterations. The gap is exactly 0 where no mask touches depth.
drng = random.Random(1)
gap_scale = dict(A=0.0, B=0.0, C=0.004, D=0.006, E=0.0)
for seed in [0, 1, 2]:
    for var in base:
        for i, scene in enumerate(scenes):
            out = f"{root}/llff_3v/{var}/{scene}/seed{seed}"
            if not os.path.exists(f"{out}/metrics.json"):
                continue
            with open(f"{out}/diag.csv", "w") as f:
                f.write("iter,loss_rgb,loss_depth_used,loss_depth_clean,depth_var_masks,n_gaussians,"
                        "drop_rate_now,iter_ms,vram_peak_mb\n")
                for it in range(100, 6001, 100):
                    clean = round(0.05 * (1 - it / 7000) + drng.gauss(0, 0.002), 6)
                    used = clean if gap_scale[var] == 0 else round(clean + gap_scale[var] * it / 6000
                                                                   + abs(drng.gauss(0, 0.001)), 6)
                    rate = 0.0 if var in ("A", "B") else round(0.2 * it / 10000, 4)
                    var_m = round((0.02 if var in ("A", "B") else 0.012) * (1 - it / 9000) + abs(drng.gauss(0, 0.001)), 6)
                    f.write(f"{it},{0.2 - it / 60000:.5f},{used},{clean},{var_m},{60000 + 8 * it},{rate},"
                            f"{40 + drng.random():.2f},1650.0\n")



def write_diag(out, var, rng):
    """diag.csv as train_llff.py writes it, from the run's flags: the gap is exactly 0 where no mask reaches a depth
    pass, and with soft_only it starts only after the soft pass does (iteration 1000)."""
    f = F.factors(var)
    reaches = F.mask_reaches_depth(f)
    with open(f"{out}/diag.csv", "w") as fh:
        fh.write("iter,loss_rgb,loss_depth_used,loss_depth_clean,depth_var_masks,n_gaussians,"
                 "drop_rate_now,iter_ms,vram_peak_mb,rgb_var_masks\n")
        for it in range(100, 6001, 100):
            clean = round(0.05 * (1 - it / 7000) + rng.gauss(0, 0.002), 6)
            on = reaches and not (f["depth_render"] == "soft_only" and it <= 1000)
            used = round(clean + 0.03 * f["drop_rate"] * it / 6000 + abs(rng.gauss(0, 0.001)), 6) if on else clean
            var_m = round((0.012 if f["drop_rate"] else 0.02) * (1 - it / 9000) + abs(rng.gauss(0, 0.001)), 6)
            fh.write(f"{it},{0.2 - it / 60000:.5f},{used},{clean},{var_m},{60000 + 8 * it},"
                     f"{round(f['drop_rate'] * it / 10000, 6)},{40 + rng.random():.2f},1650.0,{var_m / 10:.6f}\n")


def write_block(block_root, means, seeds, block_scenes, rng, sd=0.3):
    for seed in seeds:
        for var, p in means.items():
            for scene in block_scenes:
                out = f"{block_root}/llff_3v/{var}/{scene}/seed{seed}"
                os.makedirs(out, exist_ok=True)
                m = dict(variant=var, scene=scene, n_views=3, seed=seed, psnr=round(p + rng.gauss(0, sd), 3),
                         ssim=round(0.6 + (p - 19) / 50 + rng.gauss(0, 0.02), 4),
                         lpips=round(0.29 - (p - 19) / 50 + rng.gauss(0, 0.02), 4),
                         train_time_s=600, gpu="Tesla T4", shared_gpu=False, **flags(var))
                m["ssim_sk"] = round(m["ssim"] - 0.006, 4)
                json.dump(m, open(f"{out}/metrics.json", "w"), indent=2)
                write_config(out, var, seed)
                write_diag(out, var, rng)


# Blocks P, G and N (PLAN_V2 §3.1), from their own random stream so T1's numbers above stay as they were.
# P: the hard pass carries the cost (D_hard below E), the soft pass doesn't (D_soft ≈ E).
# G: E − D grows with γ. N: same-seed repeats scatter less than different seeds.
brng = random.Random(2)
write_block(root + "_pass", dict(D_hard=18.9, D_soft=19.45), [0, 1, 2], scenes, brng)
write_block(root + "_dose", dict(C_g010=18.8, D_g010=19.0, E_g010=19.4, C_g033=18.3, D_g033=18.4, E_g033=19.4),
            [0], scenes, brng)
write_block(root + "_noise", {f"{x}_r{k}": base[x] for x in "BD" for k in range(1, 5)}, [0], ["fern"],
            random.Random(3), sd=0.15)

# Tier 2: 2, 6 and 9 views (3-view seed 0 already exists above), variants B-E, seed 0
for nv, shift in [(2, -2.0), (6, 2.5), (9, 4.0)]:
    for var in ["B", "C", "D", "E"]:
        for scene in scenes:
            out = f"{root}/llff_{nv}v/{var}/{scene}/seed0"
            os.makedirs(out, exist_ok=True)
            m = dict(variant=var, scene=scene, n_views=nv, seed=0, psnr=round(base[var] + shift + drng.gauss(0, 0.3), 3),
                     ssim=round(0.6 + shift / 20 + drng.gauss(0, 0.02), 4), lpips=round(0.29 - shift / 40, 4),
                     gpu="NVIDIA GeForce RTX 3050 Ti Laptop GPU", shared_gpu=False)
            m["ssim_sk"] = round(m["ssim"] - 0.006, 4)
            json.dump(m, open(f"{out}/metrics.json", "w"), indent=2)


def write_png(path, w, h, pixel):
    raw = b"".join(b"\x00" + bytes(c for x in range(w) for c in pixel(x, y)) for y in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# eval images as render.py writes them: eval/ours_6000/{gt,renders,depth}/00000.png and depth/color_00000.png
W, H = 64, 48
for scene in ["fern", "horns", "trex"]:
    for k, var in enumerate(base):
        ev = f"{root}/llff_3v/{var}/{scene}/seed0/eval/ours_6000"
        for sub in ("gt", "renders", "depth"):
            os.makedirs(f"{ev}/{sub}", exist_ok=True)
        gt = lambda x, y: (4 * x % 256, 5 * y % 256, 128)
        write_png(f"{ev}/gt/00000.png", W, H, gt)
        write_png(f"{ev}/renders/00000.png", W, H, lambda x, y: tuple(min(255, c + 6 * k) for c in gt(x, y)))
        write_png(f"{ev}/depth/00000.png", W, H, lambda x, y: (4 * y,) * 3)
        write_png(f"{ev}/depth/color_00000.png", W, H, lambda x, y: (4 * y, 120, 255 - 4 * y))
print("fixtures written to", root, "and", ", ".join(root + x for x in ("_pass", "_dose", "_noise")))
