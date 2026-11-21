# 7 Conclusion

We asked whether DNGaussian's monocular-depth regularization and DropGaussian's random dropout are complementary, redundant or interfering in sparse-view 3D Gaussian Splatting. We answered with a pre-registered factorial study: five variants of the same code, three datasets, three seeds and 1,169 training runs.

**Answers to the research questions:**

| | Question | Answer |
| --- | --- | --- |
| RQ1 | Complementary on LLFF 3-view? | **Yes.** D − B = +0.50 dB [+0.08, +0.91], E − B = +0.60 dB [+0.30, +0.91]; the interaction is near 0 (I_D +0.02 [−0.34, +0.38]), so the gains add roughly additively in PSNR. |
| RQ2 | Does it matter whether the mask reaches the depth passes? | **No measurable effect.** E − D is under 0.25 dB in every point estimate (pooled +0.05 [−0.01, +0.12]); no single depth pass is responsible. |
| RQ3 | Does it generalize? | **Partly.** Dropout's own gain is consistent (C − A pooled +0.40 dB, τ² = 0), and E − B is positive pooled (+0.34 dB [+0.04, +0.64]). But the pooled interaction is slightly negative (I_D −0.19 dB [−0.37, −0.01]), and on Blender at 8 views, where the depth prior is not shown to help, the combination is partly redundant. |
| RQ4 | How does sparsity change it? | **It is a very-sparse effect.** D − B is positive at 2 and 3 views and vanishes by 6; at 9 views depth no longer helps. |

**Why.** Splitting the error into bias and run-to-run variance explains the pattern. In the sparsest settings, depth mainly removes bias and dropout mainly removes variance; they act on different errors, so their gains add. Where either condition fails (more views, DTU for dropout, Blender for depth), complementarity fails too. The two most obvious alternative mechanisms, corruption of the depth passes by the mask and the opacity cap under compensated dropout, are not supported.

**What it means in practice.**
1. On very sparse inputs, add dropout to a depth-regularized 3DGS model as is. Shielding the depth passes is not needed.
2. Do not assume a monocular depth prior helps: check that it reduces bias in the target regime.
3. The headroom left above DNGaussian's bias floor is variance. Run-to-run variance is still 18–34 % of the combined model's error, and a 3-seed ensemble would gain +0.6 to +1.1 dB. Variance reduction is the more promising next step.
4. Evaluate with several seeds. A single-run 8-scene mean is uncertain by ±0.26 dB with depth and ±0.43 dB without, which is the size of many published improvements.

The limitations (Section 6.5 / Limitations) bound these claims: interactions below about 0.5 dB are below what one 8-scene dataset can detect, interaction signs depend on the metric scale, and every method ran with one published hyperparameter setting.
