# 5 Results (draft for review)

Every number here comes from `results/RESULTS.md` (v1, LLFF) or `results/v2/RESULTS_v2.md` (v2, PR #37). PSNR differences are paired by scene, as the mean with a 95 % t-interval. "≈ 0" means the CI lies inside ±0.25 dB, and "inconclusive" means it crosses 0 and extends past ±0.25 dB (§4.6). Figures are numbered in order of first mention: Fig. 1 `fig_pooled`, Fig. 2 `fig_budget`, Fig. 3 `fig_budget_views`, Fig. 4 `fig_ceiling`, Fig. 5 `fig_comp`, Fig. 6 `fig_depthdose`, Fig. 7 `fig_power` (all in `results/v2/figures/`).

## 5.1 Main result on LLFF (RQ1)

**Table 1:** LLFF, 3 views, 8 scenes × 3 seeds. Mean ± SD over seeds.

| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- |
| A: none | 16.33 ± 0.23 | 0.460 | 0.394 |
| B: depth (DNGaussian) | 18.71 ± 0.04 | 0.592 | 0.302 |
| C: dropout | 16.80 ± 0.15 | 0.497 | 0.378 |
| D: depth + dropout, shared mask | 19.21 ± 0.06 | 0.611 | 0.303 |
| E: depth + dropout, clean depth passes | **19.31 ± 0.09** | **0.617** | 0.303 |

Our B (18.71 dB, the lab A4000 runs that every comparison uses) is 0.41 dB below DNGaussian's published 19.12 dB; as a code check, a single run of the upstream code on a Colab T4 gave 18.93 dB (RUNLOG, 13 Oct).

Both regularizers help on their own: depth by +2.38 dB [+1.80, +2.96], dropout by +0.48 dB [+0.18, +0.77]. Combining them adds to DNGaussian:
- D − B = +0.50 dB [+0.08, +0.91]
- E − B = +0.60 dB [+0.30, +0.91]

E − B is positive in 8 of 8 scenes. Both combinations also beat dropout alone by about 2.4 dB. Under the pre-registered rule, D and E are therefore **complementary** on LLFF (Fig. 1, LLFF rows).

The interaction terms are near zero:
- I_D = +0.02 [−0.34, +0.38]
- I_E = +0.13 [−0.17, +0.42]

Both are inconclusive after Holm. In SSIM the interactions are small but negative, with CIs that exclude 0 (I_D −0.018, I_E −0.012). The two gains therefore add up roughly additively in PSNR and slightly sub-additively in SSIM.

## 5.2 Does decoupling the depth passes matter? (RQ2)

E − D = +0.10 dB [−0.10, +0.30] on LLFF, inconclusive. It also stays small on the other datasets:
- DTU: +0.09 [−0.24, +0.42]
- Blender: +0.04 [−0.05, +0.13], ≈ 0
- pooled: +0.05 [−0.01, +0.12], τ² = 0

Pass localization (block P) finds no single pass responsible:
- D_hard − E = −0.12 [−0.32, +0.08]
- D_soft − E = −0.10 [−0.27, +0.07]

The two other probes of the dose and of geometry do not settle the question either:
- In the dose block (G, one seed), E − D grows with the drop rate in 7 of 8 scenes, from −0.16 dB at γ = 0.1 to +0.20 dB at γ = 1/3. That is a trend, not a test.
- On geometry, the shared mask is slightly *better*: MVC@5 % E − D = −0.021 [−0.037, −0.005]. That is just past δ = 0.02, with Holm p 0.049.

**Conclusion:** at the published drop rate, letting the mask reach the depth passes neither helps nor measurably hurts.

## 5.3 Does it generalize? DTU and Blender (RQ3)

**Table 2:** DTU (3 views, 15 scans, masked metrics) and Blender (8 views, 8 objects). Mean ± SD over 3 seeds.

| Variant | DTU PSNR | DTU SSIM | DTU LPIPS | Blender PSNR | Blender SSIM | Blender LPIPS |
| --- | --- | --- | --- | --- | --- | --- |
| A | 14.04 ± 0.19 | 0.690 | 0.260 | 23.47 ± 0.04 | 0.879 | 0.098 |
| B | 18.37 ± 0.57 | 0.776 | 0.176 | 22.66 ± 0.03 | 0.868 | 0.112 |
| C | 14.51 ± 0.48 | 0.726 | 0.243 | **23.83 ± 0.04** | **0.886** | **0.094** |
| D | 18.57 ± 0.26 | **0.810** | **0.160** | 22.77 ± 0.02 | 0.873 | 0.109 |
| E | **18.67 ± 0.17** | 0.807 | 0.163 | 22.82 ± 0.03 | 0.874 | 0.108 |

**Table 3:** Contrasts per dataset and pooled (DerSimonian–Laird, PSNR). Also Fig. 1.

| Contrast | LLFF | DTU | Blender | Pooled |
| --- | --- | --- | --- | --- |
| B − A | +2.38 | +4.34 | −0.80 [−1.78, +0.18] | heterogeneous (τ² 4.8) |
| C − A | +0.48 | +0.48 | +0.37 | **+0.40 [+0.28, +0.53]**, τ² 0 |
| D − B | **+0.50** | +0.20 | +0.11 | +0.24 [−0.02, +0.50] |
| E − B | **+0.60** | +0.30 | **+0.15** | **+0.34 [+0.04, +0.64]** |
| D − C | **+2.40** | **+4.06** | **−1.06 [−2.07, −0.05]** | – |
| E − C | **+2.51** | **+4.15** | **−1.02 [−2.01, −0.02]** | – |
| I_D | +0.02 | −0.28 | **−0.26 [−0.37, −0.15]** | **−0.19 [−0.37, −0.01]** |
| I_E | +0.13 | −0.18 | **−0.21 [−0.36, −0.07]** | −0.09 [−0.34, +0.15] |
| Verdict (D and E) | complementary | inconclusive | partly redundant | |

Four patterns stand out:
1. **Dropout's gain is consistent.** It is +0.37 to +0.48 dB on every dataset, with no detectable heterogeneity (τ² = 0).
2. **Depth's gain does not.** It is large on the forward-facing LLFF and on DTU. On Blender at 8 views it is not shown to help at all: B − A = −0.80 dB [−1.78, +0.18], inconclusive, with B ahead in only 1 of 8 objects. On Blender both combinations lose to dropout alone (D − C −1.06, E − C −1.02, CIs below 0; nominally significant, unadjusted), and I_D and I_E are negative after Holm; dropout alone is the best variant there.
3. **Adding dropout to DNGaussian still helps.** E − B is positive pooled and on Blender, though smaller than on LLFF. The pre-registered prediction that the gain would be larger on DTU than on LLFF (X5) is not supported (p 0.82). The predicted ordering LLFF ≤ DTU ≤ Blender (H-PD) is reversed.
4. **The interaction is never positive.** Pooled, I_D is −0.19 dB, with no significant heterogeneity (Q p 0.21). On Blender it is significantly negative for both D and E.

On DTU, all three primaries are inconclusive: the 15 scans have wide between-scan spread.

## 5.4 Number of views (RQ4)

**Table 4:** LLFF, 8 scenes, mean PSNR over 3 seeds (block V, seeds 0–2).

| Views | A | B | C | D | E | D − B | I_D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 14.30 | 15.76 | 14.83 | 16.21 | 16.44 | +0.45 [+0.24, +0.66] | −0.08 [−0.30, +0.14] |
| 3 | 16.33 | 18.71 | 16.80 | 19.21 | 19.31 | +0.50 [+0.08, +0.91] | +0.02 [−0.34, +0.38] |
| 6 | 20.96 | 21.97 | 21.50 | 22.05 | 22.08 | +0.08 [−0.12, +0.28] | −0.46 [−0.96, +0.04] |
| 9 | 22.85 | 22.64 | 22.91 | 22.60 | 22.55 | −0.04 [−0.18, +0.10] | −0.10 [−0.34, +0.14] |

These 3-seed numbers replace v1's seed-0 table; the contrasts and CIs are in `results/v2/tables/views/contrasts_3seed.csv`. In the seed-0 table, I_D at 6 views was −0.96 dB [−1.68, −0.24]. With three seeds it shrinks to −0.46 dB and its CI includes 0.

The pattern across view counts:
- **The combination's benefit is a very-sparse effect.** It lasts at 2 and 3 views and vanishes by 6.
- **At 9 views depth is not shown to help** (B − A −0.20 [−0.73, +0.32], B ahead in 3 of 8 scenes), like Blender at 8 views.
- **E − D's point estimate stays below 0.25 dB at every view count:** ≈ 0 at 6 and 9 views, inconclusive at 2 and 3 (unadjusted).

## 5.5 Why: the error budget (N1)

We split each variant's pixel MSE on the held-out views into squared bias (the error of the 3-seed mean image) and run-to-run variance (Figs. 2 and 3).

**Table 5:** Error budget per setting.

| | LLFF 2v | 3v | 6v | 9v | DTU 3v | Blender 8v |
| --- | --- | --- | --- | --- | --- | --- |
| Depth (B − A): share of the MSE drop that is bias (BV1) | 0.96 ✓ | 0.56 ✓ | 0.13 ✗ | −2.90 ✗ | 0.51 ✓ | 0.78 (p 0.058) ✗ |
| Dropout (C − A): share of the MSE drop that is variance (BV2) | 0.74 ✓ | 0.79 ✓ | 0.84 ✓ | 1.39 ✗ | −0.76 ✗ | 0.78 ✓ |
| MSE(D) above bias²(B) (BV3) | yes | yes | yes | yes | yes | yes |

**Depth removes bias, dropout removes variance.** Where the regularizers help, they act on different error components:
- Depth's gain is mostly a bias reduction in the sparse settings: LLFF 2/3 views and DTU.
- Dropout's gain is mostly a variance reduction: LLFF up to 6 views, and Blender.

This is the mechanistic basis for complementarity on LLFF. Two exceptions:
- On DTU, dropout's small PSNR gain is not a variance reduction.
- At 6 and 9 views, depth no longer reduces bias.

**Neither removes all the variance.** In every setting, the naive combination's MSE stays above DNGaussian's bias floor (BV3, Fig. 4). Run-to-run variance is still 18–34 % of the MSE for D and E (18–44 % over all variants; minimum D at 9 views, 18.4 %), and averaging the renders of 3 seeds would gain +0.6 to +1.1 dB. **The remaining headroom is variance, not bias.**

**Scale robustness (N4).** On LLFF, D − B and E − D keep their sign under PSNR, MSE, RMSE and SSIM. The interaction terms do not: I_D is near 0 in PSNR (inconclusive) but negative in SSIM. Interaction claims are therefore scale-dependent, and we make none beyond "not positive".

## 5.6 Mechanism probes

**Compensation (Fig. 5).** Holm across 4 contrasts, TOST δ = 0.25 dB:

| Contrast | ΔPSNR [95 % CI] |
| --- | --- |
| C_nc − C (no 1/(1−r) rescaling) | +0.14 [−0.10, +0.38] |
| D_nc − D | −0.18 [−0.48, +0.12] |
| C_ca − C (clamp-aware) | −0.27 [−0.48, −0.05], Holm p 0.09 |
| D_ca − D | −0.23 [−0.56, +0.11] |

None is significant after Holm, and none is shown to be equivalent. Removing the compensation does not matter detectably. Clamp-aware dropout tends to be *worse*, which is the opposite of what the opacity-cap explanation (H7) predicts.

**Depth dose (Fig. 6).** I_D at depth weight 0.5 / 1 / 2 is +0.15 / +0.02 / −0.00 dB, all inconclusive. There is no sign that a stronger or weaker prior changes the interaction.

**Test-time robustness.** Removing 20 % of the Gaussians costs dropout-trained models about half as much PSNR as their dropout-free counterparts:
- C: −0.32 vs A: −0.79
- D: −0.48 vs B: −1.03

Dropout makes the representation more redundant, as DropGaussian argues.

**Noise.** Repeating one seed varies as much as changing the seed (same-seed / between-seed SD 1.0–1.7). Variants without the depth prior have about 2.9× the run-to-run variance (σ² ratio F = 2.90, p 2.5 × 10⁻⁵), about 1.7× the SD. A single-run 8-scene mean is uncertain by ±0.26 dB with depth and ±0.43 dB without.

## 5.7 Statistical power (Fig. 7)

With 3 seeds, the smallest I_D we can detect at 80 % power is:

| Scenes | 8 | 15 | 23 | 31 |
| --- | --- | --- | --- | --- |
| Minimum detectable I_D | 0.49 dB | 0.33 dB | 0.26 dB | 0.22 dB |

The 31-scene case is all three datasets pooled. An interaction of ±0.2 dB is therefore invisible to any single dataset here, which is why the per-dataset verdicts are mostly "inconclusive" and only the pooled estimate excludes 0.

Two caveats:
- The variance-component model's pre-registered validation is circular: τ² is estimated from the same scene means it predicts. We report the minimum detectable effects as planning numbers, not as a validated model.
- **Efficiency:** the v2 runs shared the GPU, so the only timing numbers are from v1 (Table in §4).
