# 1 Introduction

## 1.1 Background

Novel view synthesis renders a scene from viewpoints that were never photographed. Neural Radiance Fields (NeRF) [1] made this practical by fitting a volumetric scene function to many posed images. 3D Gaussian Splatting (3DGS) [2] replaced the slow ray-marched network with a set of explicit, anisotropic 3D Gaussians that are rasterized and alpha-composited, and reaches real-time rendering at comparable quality (§3.1).

Both methods assume dense input: dozens to hundreds of views. With only a few views (2–9), 3DGS overfits. It reproduces the training images with floating, view-specific Gaussians, and novel views degrade sharply. Sparse-view 3DGS is therefore an active problem, and two broad kinds of fix have appeared.

## 1.2 Two kinds of regularization

The failure has two faces, and the two most direct fixes each target one of them:

| Failure | What goes wrong | Fix | Representative method |
| --- | --- | --- | --- |
| **Geometric degeneracy** | Too few views to pin down depth; Gaussians land on wrong surfaces that still explain the training images | Supervise rendered depth with a monocular depth prior | **DNGaussian** [3] (CVPR 2024) |
| **Representational overfitting** | Individual Gaussians co-adapt to the few training views and memorize them | Randomly drop Gaussians during training, so the rest must explain each view on their own | **DropGaussian** [5] (CVPR 2025) |

Each method improves LLFF 3-view on its own. Because they target different failures, it is tempting to assume they stack. They are not independent in the code, though. DNGaussian's depth losses are computed from extra rasterizations of the same Gaussians, and a dropout mask applied to the model can reach those passes too (§3.4). Dropout could therefore corrupt the depth supervision. Alternatively, depth supervision could make the model redundant enough that dropout has nothing left to do.

## 1.3 Research questions

Recent papers do add dropout to depth-regularized baselines and report gains (§2.3). None of them asks *how* the two interact. We do, with a full factorial design:

- **RQ1 (main effect).** On LLFF 3-view, are depth regularization and Gaussian dropout complementary, redundant or interfering? We measure this with the interaction term of a 2 × 2 factorial (variants A–D).
- **RQ2 (placement).** Does it matter whether the dropout mask reaches the depth passes? Variant E shields them; blocks P and G localize and dose the effect.
- **RQ3 (generality).** Do the findings hold on DTU 3-view and Blender 8-view?
- **RQ4 (sparsity).** How does the interaction change from 2 to 9 training views?

We then ask *why* the answers come out as they do, by splitting each variant's error into a bias part (shared by every seed) and a variance part (run-to-run).

## 1.4 Contributions

1. **A controlled factorial study.** All five variants (A–E) are the same code with different flags, run on 3 datasets with 3 seeds each: 1,169 training runs in total. The hypotheses and decision rules were pre-registered in git tags before the data existed (§3.7, §4.6).
2. **An answer to RQ1–RQ4.** The two regularizers are complementary on very sparse forward-facing scenes: adding dropout to DNGaussian gains about +0.5 dB on LLFF at 2–3 views. Across three datasets the interaction is additive at best (pooled I_D = −0.19 dB [−0.37, −0.01]), and the combination is partly redundant on Blender. Where the mask acts (E vs D) has no measurable effect.
3. **A mechanism.** An error budget shows that depth mainly removes bias and dropout mainly removes run-to-run variance, in the sparsest settings. Complementarity appears exactly where both hold. Neither removes the variance left above DNGaussian's bias floor, so the remaining headroom is variance.
4. **Negative results.** Two plausible mechanisms are not supported: corruption of the depth passes by the mask, and the opacity cap under compensated dropout (H7).
5. **An honest error bar.** Same-seed reruns vary as much as different seeds. A single-run 8-scene mean is uncertain by ±0.26 dB with depth and ±0.43 dB without, and interactions below about 0.5 dB cannot be detected on one 8-scene dataset with 3 seeds.

Code, per-run results, the analysis notebook and a project page are public (README, `site/`).
