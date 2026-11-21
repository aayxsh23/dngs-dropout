# 4 Experimental setup

Every setting below is read from `experiments/matrix*.yaml` and the code at the tag that ran it. Dates and checksums are in `RUNLOG.md`.

## 4.1 Datasets and splits

| | LLFF | DTU | Blender (NeRF synthetic) |
| --- | --- | --- | --- |
| Scenes | 8: fern, flower, fortress, horns, leaves, orchids, room, trex | 15 scans: 8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114 | 8: chair, drums, ficus, hotdog, lego, materials, mic, ship |
| Training views | 3 (also 2, 6, 9 in block V), spaced evenly over the non-held-out images | 3: image indices 25, 22, 28 | 8: indices 2, 16, 26, 55, 73, 76, 86, 93 of the training set |
| Test views | every 8th image | 25 (DNGaussian's DTU test list) | 25 (every 8th of the 200 test views) |
| Resolution | 1/8 (`-r 8`) | 1/4 (`-r 4`) | 1/2 (`-r 2`), white background |
| Metrics | full image | **masked** with RegNeRF's object masks | full image |
| Monocular depth | DPT (hybrid), `llff_with_depth.tar` | DPT (large), `dtu_with_depth.tar` | DPT, `blender_with_depth.tar` |
| Archive md5 | `0f6bd7bdae837ec58ad757c7c00b0561` | `dc8df470e9cf6e46cd024cb34d19c175` | `e7597085f8682b660699a39f63d5957f` |

The split code is in `scene/dataset_readers.py`. Each depth archive was generated once and every machine used it, checked by md5.

**DTU camera poses** come from COLMAP (4.2.1, CPU) via DNGaussian's `convert.py`. The DTU split selects views by index, so the pre-registered rule required all 49 images of a scan to register; a scan failing after one retry with a PINHOLE camera would have been excluded. All 15 scans registered 49/49. Four of them (scans 30, 63, 103 and 110) needed the PINHOLE retry (RUNLOG, 17 Nov).

**DTU go/no-go** (pre-registered, `prereg-v3` §5.2.1): upstream DNGaussian trained end to end on scan 8, with a masked PSNR of 19.11 dB (RUNLOG, 18 Nov). DTU therefore stayed in the study.

**Initialization:** every run starts from a random point cloud (`--rand_pcd`), drawn from the run's seed. For Blender the cloud is written to a temporary folder, as for LLFF, so parallel runs never share it.

## 4.2 Training

All variants use DNGaussian's published settings for each dataset, for 6000 iterations. The variants differ only in the flags of §3.4. With γ = 0.2 the drop rate rises linearly to 0.12 at iteration 6000 (γ = 0.1: 0.06; γ = 1/3: 0.20).

- **LLFF** (`scripts/run_llff.sh`):
  - loss: L1 + 0.2·D-SSIM;
  - densification until iteration 6000: gradient threshold 0.0013, prune threshold 0.01, percent dense 0.01, split-opacity threshold 0.1;
  - learning rates: position 0.016 → 0.00016 over 5500 iterations from iteration 500; scaling 0.003;
  - penalties: shape 0.002, opacity 0.001;
  - depth error tolerance 0.00025; near plane 10.
- **DTU** (`scripts/run_dtu.sh`):
  - loss: L1 + 0.6·D-SSIM;
  - densification until iteration 6000: gradient threshold 0.001, prune 0.01, percent dense 0.1;
  - learning rates: position 0.0016 → 0.000016 over 5500 iterations from iteration 500; opacity 0.05; scaling 0.003;
  - penalties: shape 0.005, opacity 0.001, scale 0.005;
  - depth error tolerance 0.01.
- **Blender** (`scripts/run_blender.sh`, setting 1, used for all 8 objects, with soft depth on):
  - loss: L1 + 0.2·D-SSIM;
  - densification from iteration 500 to 6000: gradient threshold 0.0005, prune 0.01, percent dense 0.01;
  - learning rates: position 0.00016 → 0.0000016, starting at iteration 5000; scaling 0.005;
  - penalties: none;
  - depth error tolerance 0.001; non-spherical-harmonics path.

  Upstream uses different settings per object, and a spherical-harmonics model for some of them. Our single protocol keeps the comparison controlled, but B on Blender is not tuned per object.

## 4.3 Experiment blocks

| Block | Variants | Data | Scenes × seeds | Runs | Tag | Question |
| --- | --- | --- | --- | --- | --- | --- |
| T1 main | A–E | LLFF 3v | 8 × 3 | 120 | `v1.0-exp` | Effects, interaction, placement |
| P pass | D_hard, D_soft | LLFF 3v | 8 × 3 | 48 | `v1.0-exp` | Which depth pass |
| G dose | C, D, E at γ = 0.1 and 1/3 | LLFF 3v | 8 × seed 0 | 48 | `v1.0-exp` | Dropout dose |
| N noise | B, D × 4 same-seed repeats | LLFF 3v, fern | 1 × seed 0 | 8 | `v1.0-exp` | Run-to-run noise |
| V views | A–E | LLFF 2, 6, 9v | 8 × 3 | 360 | `v1.0-exp` | Number of views (3 views = T1) |
| COMP | C_nc, D_nc, C_ca, D_ca | LLFF 3v | 8 × 3 | 96 | `v1.1-exp` | Compensation mode |
| DDOSE | B, D at λ_d = 0.5 and 2 | LLFF 3v | 8 × 3 | 96 | `v1.1-exp` | Depth dose |
| NOISE2 | A, B, C, D × 4 same-seed repeats | LLFF 3v: room, horns, trex | 3 × seed 0 | 48 | `v1.1-exp` | Noise floor beyond fern |
| DTU | A–E | DTU 3v | 15 × 3 | 225 | `v1.2-exp` | Replication |
| BLENDER | A–E | Blender 8v | 8 × 3 | 120 | `v1.2-exp` | Replication |
| | | | | **1,169** | | |

How blocks pair with each other:
- P, COMP and DDOSE are paired with T1 by scene and seed.
- G's γ = 0.2 point and its A and B anchors come from T1 seed 0.
- The NOISE2 scenes were drawn by a rule written into `prereg-v3` before the draw: `random.Random(15112026).sample` of the 7 non-fern scenes.

Every planned run finished; there were no failures and no reruns (`results/RESULTS.md`, `results/v2/RESULTS_v2.md`, RUNLOG 19 Nov).

## 4.4 Hardware, software and reproducibility

- **GPU:** every run trained on one NVIDIA RTX A4000 (16 GB) in a lab PC, as each run's `metrics.json` records.
  - The `v1.0-exp` blocks T1, P, G, N and V seed 0 ran one at a time (`shared_gpu: false`).
  - All later runs (V seeds 1–2, COMP, DDOSE, NOISE2, DTU, Blender) ran three at a time on the same GPU (`shared_gpu: true`), which a process from another lab user also occupied.
  - Timing and memory are therefore reported only from the T1 runs (`results/tables/efficiency_3v.csv`).
- **Software:** PyTorch 1.12.1 (recorded in every run's `train_stats.json`). The CUDA extensions (rasterizer, simple-knn, hash-grid and SH encoders) are built from the vendored sources. A 4 GB RTX 3050 Ti laptop with the same PyTorch was used only for smoke tests, data preparation (COLMAP, DPT) and debugging.
- **Code versions:**
  - Every run comes from a git tag (`v1.0-exp` at 1040559, `v1.1-exp` at c6c71f1, `v1.2-exp` at ca59f37) with a clean tree. The runner (`experiments/run_matrix.py`) refuses anything else, and each run's `config.json` records the commit.
  - Changes went through CI (`scripts/check.sh`) and a GPU smoke test before each tag.
- **Pre-registration:** the analysis rules were tagged before the data they govern existed:
  - `prereg-v1` (1040559): LLFF, before the first T1 metric;
  - `prereg-v2` (c6c71f1): compensation;
  - `prereg-v3` (283250d): DTU, Blender, error budget, noise and power, before any of those runs. It was written after the LLFF results were seen, and says so.
- **Nondeterminism:** the 3DGS rasterizer is not deterministic, because its backward pass uses atomic additions. The same seed on the same GPU gave 17.75 and 17.29 dB on fern at 1100 iterations (RUNLOG, 12 Oct). Blocks N and NOISE2 measure this spread at 6000 iterations. All comparisons average over seeds and pair by scene.

## 4.5 Metrics

- **Image quality** on the held-out views:
  - PSNR;
  - SSIM (the 3DGS implementation; scikit-image SSIM is also recorded);
  - LPIPS (VGG backbone).
- **DTU uses masked metrics** (`experiments/eval_dtu.py`, the same computation as DNGaussian's `metrics_dtu.py`). The masks are downsized by 4 and the background is composited to white. PSNR is computed over mask pixels only; SSIM and LPIPS are computed on the composited images.
- **Geometry, MVC@5 %** (LLFF, `experiments/geometry.py`): multi-view depth consistency.
  1. For every ordered pair (i, j) of held-out views, each pixel of view i with alpha > 0.5 is back-projected with its rendered depth and projected into view j.
  2. It counts if it lands inside view j where view j's alpha is > 0.5.
  3. It is consistent if its depth agrees with view j's rendered depth within 5 % relative error.

  MVC@5 % is the consistent fraction, averaged over pairs. It needs no ground truth, so it does not favour the depth prior.
- **Efficiency** (T1 only, unshared GPU): training time, rendering FPS, number of Gaussians, model size and peak GPU memory.
- **Training diagnostics,** logged every 100 iterations in `diag.csv`:
  - the depth loss on the masked and on the clean render (their difference is the *gap*);
  - the per-pixel variance of the photometric and soft-depth renders across 4 probe masks, drawn from a separate random generator so the diagnostics never change training;
  - the Gaussian count and the current drop rate.

  At the end of training, `train_stats.json` records opacity and scale quantiles and the median nearest-neighbour distance. On DTU and Blender the diagnostic depth loss uses the LLFF form of the loss, so it compares variants within a dataset only.
- **Probes** (exploratory; T1 seed-0 checkpoints; `experiments/probes.py` at 9f8d55f):
  - *Test-time robustness:* remove a random fraction f ∈ {0, 0.05, 0.1, 0.2, 0.3} of Gaussians, with and without compensation, and report the PSNR drop.
  - *Monte-Carlo depth bias:* over 32 masks at r = 0.12 and 0.2, the median relative shift of the masked hard-pass depth from the clean one.
- **Error budget:** bias² and run-to-run variance of the pixel MSE (§3.6), from the eval renders of the three seeds. DTU uses mask pixels only.

## 4.6 Analysis (pre-registered)

- **Unit and pairing.** For each variant and scene we average over seeds. Every comparison is paired by scene: we report the mean of the per-scene differences, a 95 % t-interval over scenes (df = scenes − 1) and wins.
- **Primary contrasts** (per dataset, PSNR, Holm-corrected across the three): E − D, $I_D$ and $I_E$.
- **Secondary contrasts:**
  - B − A, C − A, D − B, D − C, E − B and E − C;
  - D_hard − E and D_soft − E (Holm across the two);
  - the compensation contrasts C_nc − C, D_nc − D, C_ca − C and D_ca − D (Holm across the four, with TOST equivalence at ±0.25 dB);
  - the primary contrasts on MVC@5 %;
  - the dose and view sweeps, as trends.
- **Margin.**
  - δ = 0.25 dB (0.02 for MVC@5 %).
  - A difference is **"≈ 0" (equivalent to 0)** when its CI lies inside ±δ.
  - It is **"inconclusive"** when the CI crosses 0 but extends past ±δ.
- **Verdict.** D and E are each classified as synergistic, complementary, complementary but partly redundant, redundant, interfering or inconclusive. The classification uses two axes, following the table in PLAN_V2 §5:
  - does X beat both B and C;
  - the sign of its interaction.

  B (published DNGaussian) is the fixed reference. SSIM and LPIPS must point the same way, otherwise the verdict is "metric-dependent".
- **Mechanism claims** need all their conditions:
  - *"through the depth supervision":* E − D > 0, gap(D) > gap(C), and E − D larger at γ = 1/3 than at γ = 0.1 in at least 6 of 8 scenes;
  - *"through the opacity cap":* D_hard − E < 0, D_soft − E consistent with 0, and a far-biased hard-pass depth that grows from r = 0.12 to r = 0.2.
- **Across datasets** (`prereg-v3`):
  - A contrast **replicates LLFF** when it has the same sign and its CI overlaps the LLFF CI (X4).
  - Datasets are pooled with a two-stage random-effects model (DerSimonian–Laird), reporting the pooled estimate, τ² and Cochran's Q.
  - X5 is a one-sided Welch test on the per-scene D − B values.
- **Error budget:** BV1–BV6 (§3.7), with paired t-intervals over scenes and Holm within the BV family.
- **Scale robustness (N4):** each interaction is recomputed on PSNR, MSE, RMSE and SSIM. It is "robust" if it has the same sign on all four, otherwise "scale-dependent".
- **Noise and power:**
  - The run variance and the between-scene variance of each contrast come from the same-seed repeats (N, NOISE2) and the seeds.
  - The minimum detectable effect follows for 8, 15, 23 and 31 scenes.
  - The variance model's check against the observed T1 interval is not independent, because both use the same data. The results report this.
