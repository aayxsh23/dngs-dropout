# 6 Discussion (draft for review)

## 6.1 What we found

We asked whether DNGaussian's monocular-depth regularization and DropGaussian's random dropout are complementary, redundant or interfering in sparse-view 3DGS. The answer depends on the regime:

| Regime | Answer |
| --- | --- |
| Very sparse forward-facing scenes (LLFF, 2–3 views) | **Complementary.** Adding dropout to DNGaussian gains +0.5 to +0.6 dB, and the two gains add up about additively in PSNR. |
| Across three datasets | **Additive at best.** The pooled interaction is slightly negative (I_D = −0.19 dB [−0.37, −0.01]). Where the depth prior itself stops helping (Blender at 8 views, LLFF at 9 views), the combination is partly redundant. |
| How dropout reaches the depth passes | **Doesn't matter measurably.** Shielding the depth passes from the mask (E vs D) changes PSNR by less than 0.25 dB on every dataset and view count. |

The practical recommendation is simple. On very sparse inputs, add dropout to a depth-regularized 3DGS model, as is. The decoupled variant is not needed.

## 6.2 Why they are complementary where they are

The error budget gives the mechanism.

**The two regularizers fix different errors.**
- **Depth fixes bias:** a systematic geometric error that every seed shares.
- **Dropout fixes variance:** the part of the error that changes from run to run.

When two methods act on different error components, their gains should add, and that is what we see on LLFF at 2–3 views. Both conditions hold there: depth's gain is ≥ 50 % bias, and dropout's is ≥ 70 % variance.

**When one condition fails, so does complementarity.**
- At 6–9 views, depth no longer reduces bias.
- On DTU, dropout no longer reduces variance.
- On Blender, the depth prior adds bias: monocular depth is a poor prior for synthetic objects seen from 8 surrounding views.

This turns a vague label ("complementary") into a testable prediction: two sparse-view regularizers combine well only when each acts on a different component of the error *in that regime*.

## 6.3 What did not explain the results

We pre-registered two other mechanisms. The data do not support either.

**Through the depth supervision.**
- **The hypothesis:** the dropout mask corrupts the depth passes, so shielding them helps.
- **What we found:**
  - E − D is ≈ 0 everywhere.
  - Neither depth pass alone carries an effect (block P).
  - Geometry is, if anything, slightly better with the shared mask.

**Through the opacity cap (H7).**
- **The hypothesis:** compensation pushes opacities into the 0.99 clamp.
- **What we found:**
  - Removing compensation (nc) or capping it per Gaussian (clamp-aware) changes nothing significantly.
  - Clamp-aware even trends worse.
  - The Monte-Carlo depth bias goes in the predicted direction, but it is tiny (0.05 % of depth).

We report these as negative results, because they rule out the most obvious "fixes" one might try when combining the two methods.

## 6.4 The remaining headroom is variance

In every setting, the combined model's error stays above DNGaussian's bias floor (BV3). Run-to-run variance is still 18–34 % of its MSE, and an ensemble of three seeds would gain +0.6 to +1.1 dB.

Dropout reduces variance, but far from completely. So the next gain in sparse-view 3DGS is more likely to come from variance reduction, such as seed ensembling, distillation of an ensemble, or stronger stochastic regularization, than from a better depth prior.

The same nondeterminism also limits evaluation. A single-run 8-scene mean is uncertain by ±0.26 dB with depth and ±0.43 dB without. That is the same size as many published improvements on LLFF 3-view.

## 6.5 Limitations (merge into the Limitations section if that is separate)

- **Power.** With 8 scenes and 3 seeds, interactions below about 0.5 dB cannot be detected on LLFF. Even all 31 scenes pooled detect only about 0.2 dB. Most per-dataset primaries are "inconclusive", which means "not shown", not "absent".
- **Scale dependence.** The interaction terms change sign between PSNR and SSIM (N4). Only the main contrasts (D − B, E − D) are robust across metrics.
- **One hyperparameter setting.** We kept DropGaussian's schedule verbatim (peak rate 0.12 at 6000 iterations) and DNGaussian's published LLFF settings on every dataset. The DTU and Blender numbers are therefore below those methods' tuned results. **[Disha: add reported DTU/Blender numbers if we cite them.]**
- **Hardware.**
  - The v2 runs shared a GPU, so they give no efficiency results.
  - All runs used one GPU type per seed, which keeps comparisons fair but does not test portability.
- **Blender** uses the non-SH path of `train_blender.py` and DPT depth that we generated ourselves; DNGaussian publishes none for Blender.
- **One pre-registered check was not testable.** The variance-component validation (§5.2.7) is circular by construction. We report the minimum detectable effects as planning numbers only.

## 6.6 Future work

- **Regime detector.** Predict per scene whether depth reduces bias: for example, from the agreement between DPT depth and the sparse views. Enable the depth loss only when it does.
- **Targeted variance reduction.** Combine depth with an explicit variance reducer, such as ensembling or EMA of the Gaussians, and test whether BV3's gap closes.
- **More seeds and scenes.** These are needed to settle the sign of the interaction. The power analysis says about 25 scenes per dataset are needed to detect 0.25 dB.
