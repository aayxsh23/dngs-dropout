# Results v2: three datasets, compensation, depth dose, error budget

These results extend [Results v1](../RESULTS.md) (LLFF, tag `v1.0-exp`). They add:
- the blocks of `v1.1-exp`: COMP, DDOSE and NOISE2;
- the blocks of `v1.2-exp`: DTU 3-view and Blender 8-view;
- the pre-registered analyses of PLAN_V2 §5.2 (`prereg-v3`).

All runs were on the lab RTX A4000, which was shared (`shared_gpu: true`), so no efficiency claims are made from them. The tables were made on Colab with [`notebooks/results_v2_colab.ipynb`](../../notebooks/results_v2_colab.ipynb), from the lab tarballs on Drive, each md5-checked.

| Block | Root | Runs | Tag |
| --- | --- | --- | --- |
| T1: LLFF 3v, A–E | `outputs` | 120 | v1.0-exp |
| Views: LLFF 2/6/9v, seeds 0–2 | `outputs_views` | 360 | v1.0-exp |
| COMP: C/D × nc/ca | `outputs_comp` | 96 | v1.1-exp |
| DDOSE: B/D × depth_scale 0.5/2 | `outputs_depthdose` | 96 | v1.1-exp |
| NOISE2: A–D same-seed repeats | `outputs_noise2` | 48 | v1.1-exp |
| DTU 3v, 15 scans, A–E | `outputs_dtu` | 225 | v1.2-exp |
| Blender 8v, 8 objects, A–E | `outputs_blender` | 120 | v1.2-exp |

Nothing is missing. (The `missing.txt` files of the non-T1 blocks list the A–E grid they don't contain, which is expected.)

## 1. Main tables

Mean over seeds of the scene average, ± SD over the 3 seeds. LLFF is in v1 (B 18.71, D 19.21, E 19.31).

| Variant | DTU PSNR | DTU SSIM | DTU LPIPS | Blender PSNR | Blender SSIM | Blender LPIPS |
| --- | --- | --- | --- | --- | --- | --- |
| A: none | 14.04 ± 0.19 | 0.690 | 0.260 | 23.47 ± 0.04 | 0.879 | 0.098 |
| B: depth | 18.37 ± 0.57 | 0.776 | 0.176 | 22.66 ± 0.03 | 0.868 | 0.112 |
| C: dropout | 14.51 ± 0.48 | 0.726 | 0.243 | **23.83 ± 0.04** | **0.886** | **0.094** |
| D: both, shared mask | 18.57 ± 0.26 | **0.810** | **0.160** | 22.77 ± 0.02 | 0.873 | 0.109 |
| E: both, clean depth | **18.67 ± 0.17** | 0.807 | 0.163 | 22.82 ± 0.03 | 0.874 | 0.108 |

DTU metrics are masked (`eval_dtu.py`). On Blender at 8 views, the monocular-depth prior **hurts**: B − A is −0.80 dB. That changes how the Blender interaction should be read (§2).

## 2. Factorial per dataset and pooled (X4, X5, H-PD)

Files: `tables/<ds>/verdict_*.txt`, `tables/pooled/`, [`figures/fig_pooled.png`](figures/fig_pooled.png). PSNR in dB with 95% CI; the primaries are Holm-adjusted per dataset.

| Contrast | LLFF (8) | DTU (15) | Blender (8) | Pooled (DL) |
| --- | --- | --- | --- | --- |
| B − A | +2.38 [+1.80, +2.96] | +4.34 [+3.25, +5.42] | −0.80 [−1.78, +0.18] | +1.96 [−0.56, +4.48], τ² 4.81 |
| C − A | +0.48 [+0.18, +0.77] | +0.48 [+0.01, +0.95] | +0.37 [+0.19, +0.55] | **+0.40 [+0.28, +0.53]**, τ² 0 |
| D − B | +0.50 [+0.08, +0.91] | +0.20 [−0.36, +0.77] | +0.11 [−0.05, +0.27] | +0.24 [−0.02, +0.50] |
| E − B | +0.60 [+0.30, +0.91] | +0.30 [−0.09, +0.68] | +0.15 [+0.02, +0.28] | **+0.34 [+0.04, +0.64]** |
| E − D | +0.10 [−0.10, +0.30] | +0.09 [−0.24, +0.42] | +0.04 [−0.05, +0.13] ≈0 | +0.05 [−0.01, +0.12], τ² 0 |
| I_D | +0.02 [−0.34, +0.38] | −0.28 [−0.85, +0.30] | **−0.26 [−0.37, −0.15]** | **−0.19 [−0.37, −0.01]** |
| I_E | +0.13 [−0.17, +0.42] | −0.18 [−0.65, +0.29] | **−0.21 [−0.36, −0.07]** | −0.09 [−0.34, +0.15] |
| **Verdict D / E** | complementary / complementary | inconclusive / inconclusive | partly redundant / partly redundant | |

Reading:
- **Dropout's own gain replicates exactly.** C − A is about +0.4 dB on every dataset, with τ² = 0.
- **Adding dropout to DNGaussian helps, by less outside LLFF.** E − B is positive pooled. D − B is positive on LLFF only.
- **Decoupling does nothing measurable.** E − D is about 0 on every dataset, and pooled it sits inside ±0.12 dB.
- **The interaction is ≤ 0.** I_D is about 0 on LLFF and negative on Blender, where depth itself hurts. Pooled, I_D is −0.19 dB (Q p 0.21, so the datasets don't disagree significantly).
- **X5** (D − B larger on DTU than on LLFF): not supported (p 0.82).
- **H-PD** (rank LLFF ≤ DTU ≤ Blender): does not hold; the order is reversed.

## 3. Error budget N1 (BV1–BV6) and scale robustness N4

Files: `tables/error_budget/`, `tables/n4/`, [`figures/fig_budget.png`](figures/fig_budget.png), [`fig_budget_views.png`](figures/fig_budget_views.png), [`fig_ceiling.png`](figures/fig_ceiling.png). Pixel MSE is split into squared bias plus run-to-run variance over the 3 seeds.

| | LLFF 2v | LLFF 3v | LLFF 6v | LLFF 9v | DTU 3v | Blender 8v |
| --- | --- | --- | --- | --- | --- | --- |
| BV1: depth acts on bias (Δbias² share of B − A) | 0.96 holds | 0.56 holds | 0.13 fails | −2.90 fails | 0.51 holds | 0.78 fails (p 0.058) |
| BV2: dropout acts on variance (Δvar share of C − A) | 0.74 holds | 0.79 holds | 0.84 holds | 1.39 fails | −0.76 fails | 0.78 holds |
| BV3: MSE(D) − bias²(B) | > 0 | > 0 | > 0 | > 0 | > 0 | > 0 |

Reading:
- **Depth reduces bias in the sparsest settings.** Its effect fades by 6 views.
- **Dropout reduces variance** on LLFF up to 6 views and on Blender, but not on DTU.
- **BV3 holds everywhere.** The naive combination never gets below DNGaussian's variance-free error. Run-to-run variance is 20–44% of the MSE, and a 3-seed ensemble would gain about 1 dB on LLFF.
- **N4 (scale robustness):**
  - LLFF: D − B and E − D keep their sign on every scale (PSNR, MSE, RMSE, SSIM). I_D and I_E are scale-dependent: about 0 on PSNR, below 0 on SSIM.
  - DTU: every contrast is scale-dependent.
  - Blender: I_D and I_E are negative on every scale.

## 4. Compensation (§5.2.9), depth dose (N5), noise and power (§5.2.7)

Files: `tables/extra/`, [`fig_comp.png`](figures/fig_comp.png), [`fig_depthdose.png`](figures/fig_depthdose.png), [`fig_power.png`](figures/fig_power.png).

**Compensation.** Holm across the 4 contrasts, TOST δ = 0.25 dB. No contrast is significant after Holm, and none is shown equivalent either.

| Contrast | PSNR | Holm p |
| --- | --- | --- |
| C_nc − C (no 1/(1−r) rescaling) | +0.14 [−0.10, +0.38] | 0.46 |
| D_nc − D | −0.18 [−0.48, +0.12] | 0.46 |
| C_ca − C (clamp-aware) | −0.27 [−0.48, −0.05] | 0.09 |
| D_ca − D | −0.23 [−0.56, +0.11] | 0.46 |

So the opacity-cap mechanism (H7) gets no support: clamp-aware dropout is, if anything, worse.

**Depth dose.** I_D(s) at depth_scale 0.5 / 1 / 2 is +0.15 / +0.02 / −0.00, all inconclusive. The pre-registered prediction "all three ≈ 0" does not hold, because the CIs are too wide to show equivalence.

**Noise.**
- Same-seed SD over between-seed SD is 1.0–1.7 for A–D. That is within the predicted [0.5, 2], so seeds add no structure beyond the rasterizer's nondeterminism.
- Runs without depth are noisier: σ²(A, C) / σ²(B, D) = F 2.90, one-sided p 2.5e-5. **Holds.**
- Noise band of a single-run 8-scene mean: ±0.26 dB with depth, ±0.43 dB without.

**Power.**

| Minimum detectable effect, R = 3 seeds | 8 scenes | 15 | 23 | 31 |
| --- | --- | --- | --- | --- |
| I_D | 0.49 dB | 0.33 | 0.26 | 0.22 |

So an interaction of about ±0.2 dB is below what any single dataset here can detect, and only pooling reaches it.

Caveat: the variance-component model "fits" with ratio 1.00 by construction. τ² is estimated from the same scene means it predicts, so this check is not an independent validation.

## 5. Summary for the report

1. Both regularizers help on their own on LLFF and DTU. On Blender at 8 views, the depth prior hurts and dropout alone is best.
2. The combination beats DNGaussian on LLFF (+0.5–0.6 dB), and E − B is positive pooled. The interaction itself is about 0 on LLFF and ≤ 0 elsewhere. The overall label is therefore **"complementary on LLFF, partly redundant elsewhere"**, not a synergy.
3. Decoupling the depth passes from the mask (E vs D) changes nothing measurable on any dataset.
4. Mechanism:
   - Depth removes bias, and dropout removes run-to-run variance, in the sparsest settings.
   - Neither removes the variance left above DNGaussian's bias floor (BV3).
   - The compensation and opacity-cap explanations are not supported.

## How to regenerate

Run `notebooks/results_v2_colab.ipynb` on a CPU runtime with the 5 tarballs in Drive `sparse3dgs/outputs/`. The error budget needs the eval renders, which are in the tarballs but not in the repo. Every other step can be rerun from `tables/*/runs.csv`:

```bash
T=results/v2/tables
python experiments/stats.py --runs $T/T1/runs.csv --tables $T/llff --noise_runs $T/noise/runs.csv
python experiments/stats.py --dataset dtu --n_views 3 --runs $T/dtu/runs.csv --tables $T/dtu
python experiments/stats.py --dataset blender --n_views 8 --runs $T/blender/runs.csv --tables $T/blender
python experiments/stats.py --pooled llff=$T/T1/runs.csv dtu=$T/dtu/runs.csv blender=$T/blender/runs.csv --blender_views 8 --tables $T/pooled
python experiments/extra_blocks.py --runs $T/T1/runs.csv --comp_runs $T/comp/runs.csv --ddose_runs $T/depthdose/runs.csv \
    --noise2_runs $T/noise2/runs.csv --noise_runs $T/noise/runs.csv --tables $T/extra
python experiments/plots/v2_figures.py --tables $T --figures results/v2/figures
```
