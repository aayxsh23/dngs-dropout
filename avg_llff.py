import json, os
scenes = ["fern","flower","fortress","horns","leaves","orchids","room","trex"]
keys = ["PSNR","SSIM","SSIM_sk","LPIPS"]
rows = []
for s in scenes:
    p = f"output/llff/{s}/results_eval.json"
    if not os.path.exists(p):
        print(f"{s:9s} MISSING"); continue
    r = json.load(open(p))["ours_6000"]
    rows.append(r)
    print(f"{s:9s} " + "  ".join(f"{k}={r[k]:.4f}" for k in keys))
if rows:
    print(f"{'MEAN':9s} " + "  ".join(f"{k}={sum(r[k] for r in rows)/len(rows):.4f}" for k in keys) + f"  (n={len(rows)})")
print("PAPER     PSNR=19.1200  SSIM=0.5910  LPIPS=0.2940  (DNGaussian Table 1, LLFF 3-view)")
