#!/usr/bin/env python
"""Write fake metrics.json files in the agreed layout, to test aggregate.py before real runs exist.

python experiments/fixtures/make_fixtures.py fake_outputs
python experiments/aggregate.py --out_root fake_outputs --tables fake_tables
Numbers are random and mean nothing. Seed 2 is deliberately incomplete and two files have quirks.
"""
import json, os, random, sys

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
            m = dict(variant=var, scene=scene, n_views=3, seed=seed,
                     psnr=round(p + rng.gauss(0, 0.3), 3), ssim=round(0.6 + rng.gauss(0, 0.02), 4),
                     lpips=round(0.29 + rng.gauss(0, 0.02), 4), train_time_s=600 + 120 * (var == "E"),
                     n_gaussians=rng.randint(80000, 120000), vram_peak_mb=3000.0,
                     git_commit="abc1234", gpu="Tesla T4", shared_gpu=seed != 0)
            if (var, scene, seed) == ("C", "room", 1):
                del m["lpips"]                    # missing key: should warn, not crash
            if (var, scene, seed) == ("D", "trex", 0):
                m["wall_time_s"] = m.pop("train_time_s")   # what run_matrix.py writes today
            json.dump(m, open(f"{out}/metrics.json", "w"), indent=2)
print("fixtures written to", root)
