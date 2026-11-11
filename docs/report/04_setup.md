# 4 Experimental setup (draft, AY-20)

Draft of report section 4 (PLAN_V2 §9). Every setting below is read from `experiments/matrix*.yaml` and the code at tag `v1.0-exp`. Items marked **[fill]** are filled in when the runs are done.

## 4.1 Data and split

- **Dataset:** LLFF, 8 forward-facing scenes (fern, flower, fortress, horns, leaves, orchids, room, trex), at 1/8 resolution (`-r 8`, 504 × 378).
- **Split:** DNGaussian's split. Every 8th image is held out for evaluation; the `n_sparse` training views are spaced evenly over the rest. The main experiments use 3 training views; block V uses 2, 3, 6 and 9.
- **Depth prior:** DPT monocular depth for the training views, generated once and used from one archive on every machine (`llff_with_depth.tar`, md5 `0f6bd7bdae837ec58ad757c7c00b0561`).
- **Initialization:** a random point cloud (`--rand_pcd`), drawn from the run's seed. COLMAP points are never used in training; the per-scene analysis (§4.6) reads them only to score the depth prior.

## 4.2 Training

All variants use DNGaussian's published LLFF settings: 6000 iterations; L1 + 0.2 · D-SSIM; densification until iteration 6000 (gradient threshold 0.0013, prune threshold 0.01, percent dense 0.01, split-opacity threshold 0.1); position learning rate 0.016 → 0.00016 over 5500 iterations from iteration 500; scaling learning rate 0.003; shape penalty 0.002; opacity penalty 0.001; depth error tolerance 0.00025; near plane 10. The variants differ only in the three flags of §3.4. With γ = 0.2 the drop rate rises linearly to 0.12 at iteration 6000 (γ = 0.1: 0.06; γ = 1/3: 0.20).

## 4.3 Experiment blocks

| Block | Variants | Scenes × seeds | Views | Runs | Question |
| --- | --- | --- | --- | --- | --- |
| T1 main | A, B, C, D, E | 8 × 3 (seeds 0–2) | 3 | 120 | RQ1, RQ2 |
| P pass | D_hard, D_soft | 8 × 3 | 3 | 48 | RQ2: which depth pass |
| G dose | C, D, E at γ = 0.1 and 1/3 | 8 × seed 0 | 3 | 48 | RQ3 |
| N noise | B and D, 4 repeats each | fern × seed 0 | 3 | 8 | Run-to-run noise at a fixed seed |
| V views | A–E | 8 × seed 0 | 2, 3, 6, 9 | 160 | RQ4 |

Block P is paired with T1's D and E by scene and seed. Block G's γ = 0.2 point and its A and B anchors come from T1 seed 0. **[fill: the blocks actually completed, and any descoping per PLAN_V2 §8.]**

## 4.4 Hardware and reproducibility

- **GPUs:** T1, P, G and N run on NVIDIA T4s (Google Colab, Kaggle), one run per GPU. Block V runs on an A100 with several runs sharing the GPU. All variants of one seed run on the same GPU type. Timing and memory are reported only from runs that had a GPU to themselves (`shared_gpu: false`). **[fill: exact split per block.]**
- **Software:** Python 3.13, PyTorch 2.11, CUDA 13.0 on Colab **[fill: Kaggle and A100 stacks]**. The CUDA extensions (rasterizer, simple-knn, hash-grid and SH encoders) are built from the vendored sources for each GPU.
- **Code version:** every run is made from git tag `v1.0-exp` with a clean tree; the runner refuses anything else, and each run's `config.json` records the commit. The analysis rules of §4.6 were tagged `prereg-v1` before the first T1 metric existed.
- **Nondeterminism:** the 3DGS rasterizer is not deterministic (its backward pass uses atomic additions). The same seed on the same GPU gave PSNRs 0.46 dB apart on fern at 1100 iterations. Block N measures this spread at 6000 iterations, and all comparisons average over seeds and scenes.

## 4.5 Metrics

- **Image quality** on the held-out views: PSNR; SSIM (the 3DGS implementation; scikit-image SSIM in the appendix); LPIPS with the VGG backbone.
- **Geometry: multi-view depth consistency, MVC@5 %.** For every ordered pair (i, j) of held-out views, each pixel of view i with alpha > 0.5 is back-projected with its rendered depth and projected into view j. A pixel counts if it lands inside view j where view j's alpha is > 0.5. It is consistent if its depth agrees with view j's rendered depth within 5 % relative error. MVC@5 % is the consistent fraction, averaged over pairs; we also report the median relative error. It needs no ground truth, so it does not favour the depth prior. A synthetic two-camera test checks the camera model (a correct model gives 1.00; a transposed matrix gives 0.15).
- **Efficiency:** training time, rendering FPS, number of Gaussians, model size and peak GPU memory (unshared T4 runs only).
- **Training diagnostics,** logged every 100 iterations in `diag.csv`: the depth loss on the masked and on the clean render (their difference is the *gap*), the per-pixel variance of the photometric render and of the soft-depth render across 4 probe masks at r = 0.2 (drawn from a separate generator, so the diagnostics never change the training randomness), the Gaussian count and the current drop rate. At the end of training: opacity and scale quantiles and the median nearest-neighbour distance between Gaussian centres (root of the mean squared distance to the 3 nearest neighbours, as in 3DGS initialization).
- **Probes** (exploratory, on T1 seed-0 checkpoints):
  - *Test-time robustness:* remove a random fraction f ∈ {0, 0.05, 0.1, 0.2, 0.3} of Gaussians (5 masks, with and without compensation) and report the PSNR drop.
  - *Monte-Carlo depth bias:* over 32 masks at r = 0.12 and 0.2, the median relative shift of the masked hard-pass depth from the clean one (D and E checkpoints).

## 4.6 Analysis (pre-registered, `prereg-v1`)

- **Unit and pairing.** For each variant and scene we average over seeds. Every comparison is paired by scene: we report the mean of the 8 per-scene differences, a 95 % t-interval (df = 7) and wins out of 8.
- **Primary contrasts** (T1, PSNR, Holm-corrected across the three): E − D, $I_D$ and $I_E$.
- **Secondary contrasts:** B − A, C − A, D − B, D − C, E − B and E − C; D_hard − E and D_soft − E (Holm across the two); the same contrasts at the other doses and view counts, reported as trends with per-scene points; and the primary contrasts on MVC@5 %.
- **Margin.** δ = 0.25 dB (0.02 for MVC@5 %). A difference is "equivalent to 0" when its CI lies inside ±δ, and "inconclusive" when the CI crosses 0 but extends past ±δ.
- **Verdict.** D and E are each classified as synergistic, complementary, complementary but partly redundant, redundant, interfering or inconclusive. The classification uses two axes, whether X beats both B and C and the sign of its interaction, according to the table in PLAN_V2 §5. Published DNGaussian (B) is the fixed reference. SSIM and LPIPS must point the same way, otherwise the verdict is "metric-dependent".
- **Mechanism claims** need all their conditions:
  - *"through the depth supervision":* E − D > 0, gap(D) > gap(C), and E − D larger at γ = 1/3 than at γ = 0.1 in at least 6 of 8 scenes;
  - *"through the opacity cap":* D_hard − E < 0, D_soft − E consistent with 0, and a far-biased hard-pass depth that grows from r = 0.12 to r = 0.2.
- **Per-scene moderators** (exploratory): DPT reliability (Spearman between DPT disparity and the inverse depth of COLMAP points), texture (mean image-gradient magnitude) and depth range, each correlated with the per-scene primary contrasts (n = 8, so |ρ| > 0.74 for p < 0.05).
- **Power.** With 8 scenes × 3 seeds, interactions smaller than about 0.35–0.5 dB are below our detection limit (RESEARCH_ANALYSIS §5.3).
