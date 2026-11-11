# Results v1 (tag v1.0-exp, lab RTX A4000, 344 runs)

Everything here is produced from `results/raw/` by the repo's scripts, with the pre-registered rules of PLAN_V2 §5
(as implemented in `stats.py` v2, PR #20). Nothing was tuned after seeing the data.

## How to regenerate

```bash
R=results/raw; T=results/tables; F=results/figures
python experiments/aggregate.py --out_root $R/main  --tables $T
python experiments/aggregate.py --out_root $R/pass  --tables $T/pass
python experiments/aggregate.py --out_root $R/dose  --tables $T/dose  --seeds 0
python experiments/aggregate.py --out_root $R/noise --tables $T/noise --seeds 0
python experiments/aggregate.py --out_root $R/views --tables $T/views --seeds 0
python experiments/plots/diag_curves.py --out_root $R/main  --tables $T       --figures $F
python experiments/plots/diag_curves.py --out_root $R/pass  --tables $T/pass  --figures $F/pass    # checks only
python experiments/plots/diag_curves.py --out_root $R/dose  --tables $T/dose  --figures $F/dose    # checks only
python experiments/plots/diag_curves.py --out_root $R/noise --tables $T/noise --figures $F/noise   # checks only
python experiments/stats.py --runs $T/runs.csv --tables $T --diag_summary $T/diag_summary_3v.csv \
    --pass_runs $T/pass/runs.csv --dose_runs $T/dose/runs.csv --noise_runs $T/noise/runs.csv
python experiments/plots/forest.py   --tables $T --figures $F            # also --metric ssim / lpips
python experiments/plots/pass_loc.py --tables $T --figures $F
python experiments/plots/dose.py     --tables $T --figures $F
# views.py needs llff_3v (T1 seed 0) next to llff_2v/6v/9v in one root; symlinks are enough:
V=$(mktemp -d); for n in 2 6 9; do ln -s $PWD/$R/views/llff_${n}v $V/; done; ln -s $PWD/$R/main/llff_3v $V/
python experiments/plots/views.py --out_root $V --tables $T/views --figures $F       # also --metric ssim / lpips
```

All diag sharp checks passed on every block (120 + 48 + 48 + 8 runs, all flags read from `config.json`).
MVC@5% (`geometry.py`) and the probes (`probes.py` at `9f8d55f`) need `depth_raw/` and the checkpoints, so they ran on
the lab PC on 11 Nov; their outputs are in `results/tables/geometry/<block>/` and `results/raw/main/.../seed0/probes/`.
Then, here:

```bash
python experiments/stats.py ... --geometry $T/geometry/main/geometry_3v.csv     # same call as above, plus --geometry
python experiments/probes_summary.py --out_root $R/main --tables $T --figures $F
```
The qualitative grids use eval view 0 of every T1 seed-0 run (renders, GT and depth PNGs, 160 files, copied from the
lab disk on 11 Nov as `qual_view0.tgz`; not committed, about 43 MB):

```bash
python experiments/plots/qual_grid.py --out_root <qual_view0>/outputs                                      # fig_qual
python experiments/plots/qual_grid.py --out_root <qual_view0>/outputs --scenes horns orchids --variants B D E --name fig_fail
```

## T1: LLFF 3 views, 8 scenes × 3 seeds (`tables/main_3v.md`)

Mean over seeds of the 8-scene average, ± SD over the 3 seeds.

| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- |
| A: none | 16.33 ± 0.23 | 0.460 ± 0.012 | 0.394 ± 0.010 |
| B: depth (DNGaussian) | 18.71 ± 0.04 | 0.592 ± 0.001 | 0.302 ± 0.001 |
| C: dropout | 16.80 ± 0.15 | 0.497 ± 0.003 | 0.378 ± 0.001 |
| D: depth + dropout, shared mask | 19.21 ± 0.06 | 0.611 ± 0.001 | 0.303 ± 0.002 |
| E: depth + dropout, clean depth | 19.31 ± 0.09 | 0.617 ± 0.003 | 0.303 ± 0.000 |

## Paired contrasts (`tables/paired_3v.md`, `figures/fig_forest*.png`)

Mean of the per-scene differences, 95% t-interval over 8 scenes. Classes use δ = 0.25 dB; Holm within the primary
(3) and pass (2) families.

| Contrast | ΔPSNR [95% CI] | ΔSSIM [95% CI] | ΔLPIPS [95% CI] | PSNR class |
| --- | --- | --- | --- | --- |
| E − D | +0.10 [−0.10, +0.30] | +0.006 [+0.001, +0.011] | −0.000 [−0.005, +0.005] | inconclusive |
| I_D = (D − C) − (B − A) | +0.02 [−0.34, +0.38] | −0.018 [−0.032, −0.003] | +0.017 [−0.001, +0.036] | inconclusive |
| I_E = (E − C) − (B − A) | +0.13 [−0.17, +0.42] | −0.012 [−0.022, −0.002] | +0.017 [−0.001, +0.035] | inconclusive |
| B − A | +2.38 [+1.80, +2.96] | +0.131 [+0.105, +0.157] | −0.092 [−0.115, −0.070] | > 0 |
| C − A | +0.48 [+0.18, +0.77] | +0.037 [+0.021, +0.053] | −0.016 [−0.022, −0.010] | > 0 |
| D − B | +0.50 [+0.08, +0.91] | +0.019 [+0.004, +0.034] | +0.001 [−0.015, +0.018] | > 0 |
| E − B | +0.60 [+0.30, +0.91] | +0.025 [+0.010, +0.040] | +0.001 [−0.016, +0.018] | > 0 |
| D − C | +2.40 [+1.69, +3.11] | +0.114 [+0.082, +0.146] | −0.075 [−0.111, −0.040] | > 0 |
| E − C | +2.51 [+1.86, +3.16] | +0.120 [+0.090, +0.149] | −0.075 [−0.111, −0.040] | > 0 |
| D_hard − E | −0.12 [−0.32, +0.08] | −0.002 [−0.007, +0.003] | +0.000 [−0.004, +0.004] | inconclusive |
| D_soft − E | −0.10 [−0.27, +0.07] | −0.006 [−0.012, +0.000] | −0.001 [−0.008, +0.006] | inconclusive |

**Verdict (`tables/verdict_3v.txt`): D = complementary; E = complementary.** Both beat B and C; the interaction
terms are inconclusive in PSNR. Note the SSIM interactions are small but negative with CIs that exclude 0
(SSIM has no δ), so the gains add up slightly less than additively in SSIM.

Mechanism claims: "through the depth supervision" not supported (E − D not > 0; the gap of D is no larger than C's);
condition 3 holds (E − D grows with γ in 7/8 scenes). "Through the opacity cap" not supported (D_hard − E not < 0).

## Dose, block G (seed 0; `tables/dose_3v.csv`, `figures/fig_dose.png`)

| γ (peak rate at 6000 it) | E − D | I_D | I_E |
| --- | --- | --- | --- |
| 0.1 (0.06) | −0.16 | +0.28 | +0.12 |
| 0.2 (0.12) | +0.12 | +0.45 | +0.57 |
| 1/3 (0.20) | +0.20 | −0.08 | +0.12 |

Trend only (one seed). E − D rises with the dose, in 7/8 scenes from γ 0.1 to 1/3.

## Noise floor, block N (fern, seed 0; `tables/noise_floor_3v.csv`)

| Variant | 5 same-seed runs, SD | T1 between-seed SD (pooled) |
| --- | --- | --- |
| B | 0.220 dB | 0.213 dB |
| D | 0.216 dB | 0.242 dB |

The same seed varies as much as different seeds: run-to-run variation is the rasterizer's nondeterminism, not the
seed. Any single-run difference below about 0.4–0.5 dB is noise.

## Views, block V (seed 0; 3 views = T1 seed 0; `tables/views/`, `figures/fig_views*.png`)

Mean PSNR over 8 scenes:

| Views | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| 2 | 14.32 | 15.90 | 14.90 | 16.34 | 16.48 |
| 3 | 16.59 | 18.74 | 16.67 | 19.27 | 19.39 |
| 6 | 20.89 | 22.06 | 21.77 | 21.97 | 22.03 |
| 9 | 22.94 | 22.59 | 22.84 | 22.58 | 22.61 |

Primary contrasts per view count (PSNR, unadjusted 95% CI over scenes):

| Views | E − D | I_D | I_E |
| --- | --- | --- | --- |
| 2 | +0.14 [−0.26, +0.53] | −0.14 [−0.50, +0.23] | +0.00 [−0.45, +0.46] |
| 3 | +0.12 [−0.10, +0.34] | +0.45 [−0.28, +1.19] | +0.57 [−0.06, +1.21] |
| 6 | +0.06 [−0.11, +0.23] | −0.96 [−1.68, −0.24] | −0.90 [−1.55, −0.25] |
| 9 | +0.03 [−0.10, +0.16] | +0.09 [−0.39, +0.57] | +0.13 [−0.27, +0.52] |

## Efficiency (T1, quiet A4000; `tables/efficiency_3v.csv`)

| Variant | Train time (s) | FPS | Gaussians | Peak VRAM (MB) |
| --- | --- | --- | --- | --- |
| A | 84 | 676 | 31.1k | 576 |
| B | 145 | 622 | 46.2k | 618 |
| C | 84 | 689 | 29.6k | 570 |
| D | 141 | 651 | 39.6k | 601 |
| E | 143 | 653 | 39.6k | 602 |

## Geometry, MVC@5% (T1; `tables/geometry/`, `tables/paired_3v.md`)

Multi-view depth consistency of the held-out views (fraction of pixels whose reprojected depth agrees within 5%;
higher is better; no ground truth needed). Lab self-test: mvc5 1.0, error 0.0.

| Variant | A | B | C | D | E | D_hard | D_soft |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MVC@5%, mean over 8 scenes × 3 seeds | 0.284 | 0.509 | 0.315 | 0.552 | 0.531 | 0.546 | 0.543 |

Primary contrasts on MVC@5% (δ = 0.02, Holm across 3):

| Contrast | ΔMVC@5% [95% CI] | Holm p | Class |
| --- | --- | --- | --- |
| E − D | −0.021 [−0.037, −0.005] | 0.049 | < 0 |
| I_D | +0.013 [−0.011, +0.036] | 0.471 | inconclusive |
| I_E | −0.008 [−0.026, +0.009] | 0.471 | inconclusive |

The shared mask (D) gives slightly *more* consistent geometry than the decoupled one (E), the opposite sign of E − D in
PSNR (+0.10, inconclusive). The effect is just past δ and just under the Holm threshold. Depth (B − A ≈ +0.23) is by
far the largest effect on geometry; dropout alone adds about +0.03. Views (seed 0): D ≥ E ≥ B > C > A at 2, 6 and 9 views.

## Probes (T1 seed 0, 40 runs; `tables/probes_*_3v.csv`, `figures/fig_robustness.png`)

`probes.py` at `9f8d55f` (md5 `1a2913d0b2fbf219c528cbd87d04b8e7`), run on the lab A4000 on 11 Nov. Gate: B/fern at
f = 0 gives 21.181 dB vs 21.169 dB eval (8-bit PNG rounding). 0 failures.

**Test-time robustness** (PSNR change when a random fraction f of the Gaussians is removed, mean over 8 scenes):

| f = 0.2 | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| with 1/(1−f) compensation | −0.79 | −1.03 | −0.32 | −0.48 | −0.55 |
| without | −0.89 | −1.05 | −0.49 | −0.51 | −0.51 |

Every dropout-trained model (C, D, E) loses about half as much as its no-dropout counterpart, so dropout makes the
representation more redundant, as DropGaussian argues. Depth (B) makes it less redundant than A.

**Hard-depth bias** (opacity-cap mechanism, condition 3): the median relative depth shift under compensated dropout is
positive and larger at r = 0.20 than at 0.12 in 8/8 scenes for D and 7/8 for E, as predicted, but it is tiny: +0.05%
(D) and +0.04% (E) of depth at r = 0.20, and about −0.01% at r = 0.12. Condition 3 holds in direction only. Condition 1
(D_hard − E < 0) is not met, so the opacity-cap mechanism stays not supported.
