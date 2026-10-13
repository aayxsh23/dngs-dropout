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
Also written: a diag.csv for every 3-view run, Tier 2 metrics (2, 6 and 9 views, B-E, seed 0), and
tiny eval/ours_6000 images for seed 0 of fern, horns and trex.
"""
import json, os, random, struct, sys, zlib

root = sys.argv[1] if len(sys.argv) > 1 else "fake_outputs"
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
            if (var, scene, seed) == ("C", "room", 1):
                del m["lpips"]                    # missing key: should warn, not crash
            if (var, scene, seed) == ("D", "trex", 0):
                m["wall_time_s"] = m.pop("train_time_s")   # what run_matrix.py writes today
            json.dump(m, open(f"{out}/metrics.json", "w"), indent=2)

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
print("fixtures written to", root)
