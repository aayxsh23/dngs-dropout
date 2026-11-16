# Pre-registration v3 (PLAN_V2 §5.2): datasets, error budget, robustness, audit

**Tag:** `prereg-v3`, set on the commit that adds this file, **before** any DTU, Blender, compensation, depth-dose,
noise2 or error-budget result exists, and before any eval render is opened for the analyses below.
**Written:** Mon 16 Nov 2026 (IST), one day after the planned Sun 15 Nov 23:59. No new result had been computed by then:
the lab blocks of the 15–21 Nov plan (views seeds 1–2, compensation, DTU, Blender, depth-dose, noise2) had not produced
metrics, and `eval_T1.tgz` had not been opened.
**Relation to earlier rules:** PLAN_V2 §5 (tag `prereg-v1`) still holds for LLFF block T1 and its blocks P, G, N and V.
§5.1 (clamp-aware dropout, PR #27, intended tag `prereg-v2`) is untagged at the time of writing; item 9 below
pre-registers the compensation contrasts again, so they are covered whichever tag lands first.
Everything here *adds* rules; nothing in §5 or §5.1 is changed. Where this file is silent, §5 applies.

Notation as in §5: per-scene means over seeds, contrasts paired by scene, 95 % t-intervals over scenes, δ = 0.25 dB on
PSNR, the class order "≈0" → ">0"/"<0" → "inconclusive", Holm within the families named below.

---

## 1. New datasets and their frozen protocol

**DTU, 3 views.**
- Scans: the 15 of DNGaussian / RegNeRF: 8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114.
- Split: `scene/dataset_readers.py` (DTU branch). Train views 25, 22, 28; test = the 25 indices of 0–48 that are neither
  in `[25, 22, 28, 40, 44, 48, 0, 8, 13]` nor in the excluded list `[3–7, 16–21, 36–39]`.
- Metrics: **masked** PSNR, SSIM, SSIM_sk and LPIPS on the 25 test views, with the RegNeRF `idrmasks`, computed as
  `metrics_dtu.py` does (resize /4, background composited). Written by `experiments/eval_dtu.py` as
  `results_eval.json` with the added key `"masked": true`.
- Hyperparameters: the flags of `scripts/run_dtu.sh`, frozen in `experiments/matrix_dtu.yaml` at tag `v1.2-exp`.
- Data: COLMAP poses (`convert.py`), DPT depth run **once**; the archive `dtu_with_depth.tar` is frozen by md5,
  recorded in `docs/PLAN.md` and RUNLOG.
- **Exclusion rule (fixed now).** A scan whose COLMAP model registers fewer than 49 of 49 images is re-run once with
  `--camera PINHOLE`. If it still registers fewer than 49, it is excluded, because the split selects views by index.
  Excluded scans are listed with their counts in RUNLOG and in the report. No scan is excluded for any other reason
  (in particular, never for its metrics).
- Go/no-go (Mon 16 Nov 20:00): DTU is the second dataset if scan 8 trains end to end with upstream code and at least 13
  of 15 scans register 49/49. Otherwise Blender is the required second dataset and DTU is reported as attempted.

**Blender (NeRF synthetic), 8 views.**
- Objects: chair, drums, ficus, hotdog, lego, materials, mic, ship.
- Split: train views `[2, 16, 26, 55, 73, 76, 86, 93]` of `transforms_train.json`; eval = every 8th of the 200 test views
  (25 views), as in `readNerfSyntheticInfo`.
- Protocol: **one uniform protocol for all 8 objects**, setting 1 of `scripts/run_blender.sh` with soft depth left on,
  non-SH path only, frozen in `experiments/matrix_blender.yaml` at `v1.2-exp`:
  `-r 2 --eval --n_sparse 8 --rand_pcd --iterations 6000 --lambda_dssim 0.2 --white_background
  --densify_grad_threshold 0.0005 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01
  --densify_from_iter 500 --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000
  --position_lr_start 5000 --hard_depth_start 0 --split_opacity_thresh 0.1 --error_tolerance 0.001 --scaling_lr 0.005
  --shape_pena 0 --opa_pena 0 --scale_pena 0`.
  Our B on Blender is therefore not expected to match the published number; that is stated, not tuned away.
- Metrics: unmasked PSNR, SSIM, SSIM_sk, LPIPS from `metrics.py` on white background.
- Data: DPT depth run **once**; `blender_with_depth.tar` frozen by md5 and recorded.

**Both datasets:** variants A–E with the LLFF flags of the table in AGENTS.md (`drop_rate 0.2`, schedule
`r = drop_rate · iteration / 10000`, decision D1), seeds 0, 1, 2, one GPU type per seed. **No hyperparameter is tuned on
DTU or Blender results.** If a run fails, it is rerun once; a run that fails twice is reported as missing.
**Throughput rule:** if DTU has fewer than 150 of 225 runs by Wed 18 Nov 12:00, Blender runs seeds 0–1 only. Runs stop
Thu 19 Nov 12:00; missing runs are reported as missing.

## 2. Replication rules (X4)

- Per dataset, the same seven contrasts as LLFF: **B − A, C − A, D − B, E − B, I_D, I_E, E − D**, on PSNR (primary),
  SSIM and LPIPS (secondary), paired by scene/scan/object.
- Families: per dataset, Holm across the primary three (E − D, I_D, I_E) as in §5; the other four uncorrected.
- δ = 0.25 dB on PSNR for every dataset.
- **"Replicates LLFF"** for a contrast means both: (a) the dataset estimate has the same sign as the LLFF T1 estimate,
  and (b) its 95 % CI overlaps the LLFF 95 % CI. "Contradicts" means opposite signs with both CIs excluding 0.
  Anything else is "not informative".
- **Pooled estimate** (stats.py, D4): two-stage random effects. Stage 1: per dataset, the mean of the per-scene
  contrast and its SE (sd/√n). Stage 2: DerSimonian–Laird pooling across datasets; report the pooled estimate, its
  95 % CI, τ² and Cochran's Q with its p. With 2–3 datasets τ² is imprecise; Q and τ² are descriptive. A mixed model
  `metric ~ variant × dataset + (1 | scene)` is a robustness check only (if `statsmodels` is available).

## 3. Cross-dataset hypotheses

- **X5 (confirmatory, one-sided):** D − B on DTU > D − B on LLFF (PSNR). Test: Welch t on the per-scene D − B values
  (15 DTU scans vs 8 LLFF scenes), one-sided α = 0.05. Applies only if DTU passes go/no-go.
- **H-PD (exploratory, 3 points):** the rank order of D − B is LLFF ≤ DTU ≤ Blender (dropout helps more where the depth
  prior is weaker; DPT is out of domain on synthetic objects). Reported as holds / does not hold, no test.
- If Blender GT depth is available, the per-object DPT reliability (Spearman of DPT disparity vs GT inverse depth in the
  train views) is reported against per-object D − B. Exploratory.

## 4. N1, the error budget (BV1–BV6)

**Data.** For each (dataset, n views, variant, scene) the eval renders `eval/ours_<iter>/renders/*.png` of seeds
k = 1..n (n = 3) and the GT `gt/*.png`, as float32 in [0, 1]. GT must be identical across seeds (asserted). On DTU only
pixels with mask = 1 (same masks and resizing as `eval_dtu.py`) count.

**Estimators, per pixel and channel** (r_k the render of seed k, y the GT):
- m̄ = (1/n) Σ_k r_k, s² = (1/(n−1)) Σ_k (r_k − m̄)²
- mse = (1/n) Σ_k (r_k − y)² (expected single-run error)
- **var = s²** (run-to-run variance), **bias2 = (m̄ − y)² − s²/n** (unbiased for the squared bias)
- mse_ens = (m̄ − y)² (error of the seed ensemble)

Each is averaged over pixels, channels and views to one number per (scene, variant). The identity
**mse = bias2 + var** holds exactly, because mse = (m̄ − y)² + s²(n − 1)/n and bias2 + var = (m̄ − y)² − s²/n + s²;
it is asserted to 1e-6 relative. share_var = var / mse. Primary metric: pixel MSE (masked on DTU). Also reported:
psnr_single (mean over runs of the per-image PSNR averaged as `metrics.py` does) and psnr_ens.

**Contrasts** on mse, bias2 and var, paired by scene with 95 % t-intervals: B − A, C − A, D − B, E − B, I_D, I_E.
Holm across the BV family (the tests BV1–BV3 below) within each dataset.

**Predictions and decision rules** (Δ is "first term minus second", so a reduction is negative):
- **BV1, depth acts on bias.** Δbias2 share of (B − A) > 50 %, i.e. Δbias2(B − A) / Δmse(B − A) > 0.5, computed on the
  scene means, with Δmse(B − A) < 0. Holds if the ratio > 0.5 and the Δbias2 CI excludes 0.
- **BV2, dropout acts on variance.** Δvar share of (C − A) > 50 %, i.e. Δvar(C − A) / Δmse(C − A) > 0.5, with
  Δmse(C − A) < 0. Holds if the ratio > 0.5 and the Δvar CI excludes 0. **If BV2 fails, "dropout is a variance
  reducer" is reported as refuted**; that is a result, not a reason to change the estimator.
- **BV3, the ceiling.** The paired 95 % CI over scenes of mse(D) − bias2(B). If the CI lies below 0, dropout also
  reduces bias (D beats B's best possible variance-free error). If it contains 0, D is consistent with B's bias floor.
  If it lies above 0, D keeps variance above B's bias floor.
- **BV4, views.** On LLFF at 2, 3, 6 and 9 views (seeds 0–2, variants A, B, C, E): Δbias2(B − A) increases
  (becomes less negative) with views, and its sign at 9 views is > 0 (the prior adds bias once views are plentiful).
  Holds if the 9-view point estimate is > 0 and Spearman of Δbias2(B − A) against views is > 0 over the 4 points.
  The 9-view CI is reported; the trend has 4 points and is not tested.
- **BV5, the interaction.** I_mse = I_bias2 + I_var exactly; both terms are reported with CIs for I_D and I_E. No
  prediction about their signs.
- **BV6, generality.** The BV1 and BV2 conclusions (holds / fails) are the same on every dataset with data.

## 5. N4, the interaction on several scales

- I_D, I_E, E − D and D − B on **PSNR, MSE, RMSE and SSIM**, using the per-run mse from N1 (exact, not converted from
  mean PSNR); RMSE = √mse per run; PSNR and SSIM from `results_eval.json`.
- **"Robust"** = the point estimates have the same sign on all four scales (MSE and RMSE signs flipped so that
  "better" has one sign). Otherwise **"scale-dependent"**. Reported per dataset in
  `results/tables/interaction_scales_<dataset>.csv`; framed with Loftus (1978): a non-crossover interaction can be
  removed by a monotone rescaling, so only scale-robust interactions are interpreted.

## 6. N5, depth dose × dropout

- Block depth-dose (`matrix_depthdose.yaml`, LLFF, seeds 0–2): B and D at depth_scale 0.5 and 2; A and C from T1
  (they have no depth loss, so they serve every dose). I_D(s) = (D_s − C) − (B_s − A) for s ∈ {0.5, 1, 2}
  (s = 1 is T1's I_D).
- **Prediction:** |I_D(s)| < δ at every dose, i.e. each CI inside ±0.25 dB ("additivity is not an artefact of
  DNGaussian's depth weight"). Holm across the three doses. Each dose gets its §5 class; the prediction holds only if
  all three are "≈0". Also reported: B_s − A and D_s − C against s (trend).

## 7. G6 / N2(a): noise floor beyond fern, variance components, power

**Scene draw (rule fixed here, before the draw).**
`random.Random(15112026).sample(["flower", "fortress", "horns", "leaves", "orchids", "room", "trex"], 3)`
(the 7 non-fern LLFF scenes in alphabetical order, Python's `random` module). Evaluated after writing this rule, it
gives **room, horns, trex** (CPython 3.13; to be confirmed on the lab's Python 3.7 before `matrix_noise2.yaml` is
written; if it differs, the 3.7 result is used and the difference noted in RUNLOG).
Block noise2: variants A, B, C, D, 4 same-seed repeats each (`_r1.._r4`, seed 0) on the 3 drawn scenes, 48 runs; with
T1 seed 0 that gives 5 replicates per (scene, variant).

**Predictions.**
- Same-seed SD ≈ between-seed SD: for each variant, the ratio (same-seed SD / between-seed SD, pooled over the 3 scenes
  and fern) lies in [0.5, 2]. Then "seed" carries no information beyond run noise.
- Noise differs by variant class: pooled run SD of the prior-free variants (A, C) > that of the depth variants (B, D).
  Holds if the ratio of pooled variances > 1 with a one-sided F-test p < 0.05 on the pooled within-cell df.

**Variance components (per variant v and contrast c).**
- σ²_v: residual run variance, pooled within (scene, variant) cells across all replicate sources (T1 seeds, noise,
  noise2, and DTU/Blender seeds reported separately per dataset).
- τ²_c = var(scene means of c) − Σ_v coef²_v · σ²_v / R, truncated at 0, with R the runs per cell.

**Power.** For S scenes and R runs per scene: SE_c = √(τ²_c/S + Σ_v coef²_v σ²_v /(S·R)),
MDE = (t_{0.975, S−1} + t_{0.80, S−1}) · SE_c. Table and figure for S ∈ {8, 15, 23, 31}, R ∈ {1, 3, 5}, for a main
effect and for I_D.
**Validation (fixed now):** at S = 8, R = 3 the model's predicted 95 % CI half-width for I_D on LLFF PSNR,
t_{0.975,7} · SE, must lie within ±25 % of the observed half-width (0.36 dB, RESULTS.md: +0.02 [−0.34, +0.38]).
Otherwise the report says the variance model does not fit and the MDE table is shown as indicative only.

**Noise band for the audit.** The 95 % band for a single-run, 8-scene-mean difference between two variants of the same
noise class: ±1.96 · √(2 σ²_class / 8), for class "depth host" (pooled B, D) and "prior-free" (pooled A, C).

## 8. N2(b), the evidence audit (protocol)

- **Inclusion:** sparse-view 3DGS papers, 2024–2026, with an ablation table on **LLFF 3-view** that adds or removes a
  component and reports PSNR. Candidate list, taken in this order until 8–10 qualify: DNGaussian, FSGS, CoR-GS,
  DropGaussian, DropoutGS, D²GS, DOC-GS, DropAnSH-GS, Self-Ensembling GS, the co-adaptation paper (arXiv 2508.12720).
  A paper that fails inclusion is listed with the reason.
- **Coding sheet** `results/audit/ablation_audit.csv`, one row per component row of an ablation table:
  `paper, venue, year, url, table_id, host, component, baseline_psnr, with_psnr, delta_psnr, n_runs_reported
  (1 if unstated), spread_reported (y/n), n_scenes, init (random/sfm), coder`.
- **Classification:** |delta_psnr| < band ⇒ "within single-run noise", otherwise "above". The band is the §7 noise band
  of the matching host class (depth host for hosts with a depth prior, prior-free otherwise). The report uses
  conditional wording: "if the host's run-to-run noise resembles DNGaussian's".
- **Split and agreement:** papers are split half and half between Aayush and Disha; 2 papers are coded by both,
  independently, without discussion until both are done. Cohen's κ on (within/above) and on (runs reported y/n) over the
  shared rows. If either κ < 0.7, the coding is reported as unreliable.
- **Outputs:** the share of coded component gains within noise, the share of papers reporting more than one run, and
  the scatter of |Δ| against the band (F6).

## 9. Compensation study (LLFF, block comp)

- Variants `C_nc`, `D_nc` (no compensation: survivors × 1, not 1/(1 − r)) and `C_ca`, `D_ca` (clamp-aware, §5.1);
  8 scenes × seeds 0–2 = 96 runs, from the tag that merges #27 plus the `nc` mode.
- Contrasts: **C_nc − C, D_nc − D, C_ca − C, D_ca − D** (PSNR primary), paired by scene with T1's C and D. Holm across
  the four. Each also gets TOST equivalence with δ = 0.25 dB (equivalent if both one-sided p < 0.05; reported
  alongside the §5 class, which it does not replace).
- §5.1's readings of H7a/H7b (dose placement) remain as written there, exploratory.

## 10. N3 spectral budget (optional, only if time)

Each error image r − y is split into 5 radial FFT bands of equal width in normalised frequency; by Parseval the band
MSEs sum to the total MSE (asserted). Contrasts and the N1 bias/variance split are computed per band.
**Prediction:** depth and dropout effects overlap in the mid and high bands and separate in the low band. Exploratory.

## 11. Exploratory disclosure

- The MSE-scale and dataset-heterogeneity checks discussed on 15 Nov (`research_strategy_15nov.md`) were computed or
  reasoned about **after** seeing T1 results. Anything from them in the report is labelled exploratory.
- The 6-view interaction (≈ −0.9 dB, seed 0) motivated the views seeds 1–2 block; its seeds 1–2 result is
  confirmatory only for the sign at 6 views, under the §5 rules.
- Every analysis not listed in this file or in §5/§5.1 is exploratory and labelled as such.
