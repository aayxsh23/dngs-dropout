# Plan v2: restructured study, tasks and schedule to Tue 10 Nov

**Updated Thu 29 Oct 2026** · Maintainer: Aayush · Hard deadline **Sun 15 Nov**; we submit **Tue 10 Nov** and keep 11–15 Nov as buffer.

This plan replaces the scope, schedule, risks, budget and analysis rules in [PLAN.md](PLAN.md) (§2–§4, §6, §8, §9), and settles the decision sheet in [RESEARCH_ANALYSIS.md](RESEARCH_ANALYSIS.md) §10. Three parts of PLAN.md still apply:
- **§5, the specs W2–W13**: how to build each piece of code. This plan changes only *what* gets built and *when*.
- **§7**: figures and writing, as amended in §9 below.
- **The invariants in AGENTS.md.**

**Progress is tracked by task ID** (§7) in the two team context files on the `team-context` branch: `aayush_context.md` and `disha_context.md` (protocol in AGENTS.md, "Team context files"). This file is the plan; those files are the live status.

---

## 0. Answers first

**Can it be done by 10 Nov?** Yes, as restructured here, if four things hold:
1. The training code is merged and tagged by **Sat 31 Oct, 16:00**.
2. The A100 (2 h per weekday) and the two Colab accounts are available from 31 Oct to 4 Nov.
3. Disha reviews PRs the same day they open.
4. Experiments freeze on **Wed 4 Nov**.

**What changed on 29 Oct.** We added five novelty items (§2.6). All stay inside the original question and design:
1. **Pass localization**: which depth pass carries any interference.
2. **Geometry**: the interaction measured on multi-view depth consistency, not just image quality.
3. **A test-time redundancy probe.**
4. **Per-scene moderators.**
5. **Sharper framing.**

Items 1 and 2 need about 2 h of code **before** the freeze. Items 3–4 are offline analysis after it. Item 5 is free.

**What decides success:** the dropout code was due on 13 Oct and still doesn't exist. Thu 29 – Sat 31 Oct is almost entirely Aayush writing and testing it. Everything else waits on that.

---

## 1. Where we are on 29 Oct

| Done | Not done |
| --- | --- |
| DNGaussian reproduced: laptop 18.73, T4 **18.93** vs paper 19.12 | **Dropout port, variant E, `diag.csv`, stats files, runner (W2–W9).** Only `--seed`/`--depth_scale` exist, on branch `feat/dropout` |
| Canonical data archive (md5-locked), Colab notebook, cached wheels | Any experiment run |
| CI (11 checks), merge gate, GitHub ruleset | Report text |
| Analysis scripts tested on fixtures | Kaggle test; A100 setup |
| Verified published baselines | Open PRs: **#12** (gate, needs Disha's approval), **#11** and **#14** (Disha's, need a review) |
| Research analysis, literature update, this plan | — |

There are **12 working days** to submission: Thu 29 Oct – Mon 9 Nov, with submission on Tue 10 Nov.

---

## 2. The restructured study

### 2.1 Title and question
**Working title:** *Where Should Dropout Act? Depth Priors and Gaussian Dropout in Sparse-View 3D Gaussian Splatting*. The fallback title is *Do Depth Priors and Gaussian Dropout Help Each Other? A Controlled Study…*.

**Research question (unchanged):** in sparse-view 3DGS, are DNGaussian's monocular-depth regularization and DropGaussian's Gaussian dropout complementary, redundant or interfering, and does it matter whether the dropout mask reaches the depth supervision?

### 2.2 Sub-questions
| ID | Sub-question | Evidence |
| --- | --- | --- |
| **RQ1** Effect | Does each regularizer help alone, and do their effects add up? | T1: B − A, C − A, I_D, I_E |
| **RQ2** Placement | Does keeping the mask away from the depth passes (E) beat the naive combination (D)? **Which depth pass, hard or soft, carries any difference?** | T1: E − D. Block P: D_hard, D_soft |
| **RQ3** Dose | Does the answer depend on how much dropout? This includes the published end rate of 0.2 | Block G: γ = 0.1, 0.2, 1/3 |
| **RQ4** Views | Does it depend on how few views we have? | Block V: 2, 3, 6, 9 views |
| **RQ5** Geometry | Is the interaction the same for **geometry** (multi-view depth consistency) as for image quality? | Raw depth from every run; `geometry.py` |
| Mechanism | *Why* does any interference happen? | Diagnostics, Monte-Carlo depth-bias probe, robustness curve, theory (§2.4) |

### 2.3 Hypotheses (pre-registered with the analysis rules)
- **H1, depth-signal corruption.** In D the mask perturbs the depth loss: gap(D) > gap(C). If that harms training, E > D, and **E − D grows with dose**.
- **H2, the opacity cap (now tested).** In the hard-depth pass every Gaussian has opacity 0.95, so compensated dropout hits the rasterizer's 0.99 cap and the expected opacity drops (−8 % at r = 0.12, −17 % at r = 0.2). Predictions:
  - the Monte-Carlo probe shows masked hard-pass depth biased "far", by more at r = 0.2;
  - if D < E, the deficit sits mainly in **D_hard**, so D_hard − E ≈ D − E and D_soft − E ≈ 0.
- **H3, neighbour compensation (partly tested).** Depth supervision packs Gaussians closer (smaller kNN distance in B than A). The model then becomes more redundant: its PSNR drops less when Gaussians are removed at test time. That weakens dropout, so I_E < 0 even if E − D ≈ 0.
- **H4, variance penalty.** `rgb_var_masks` is lower in C, D and E than in A and B; `depth_var_masks` is lower in D than in E.
- **H6, diminishing returns.** The effects and the interaction shrink from 2 to 9 views.

H5 (region specificity) is future work.

### 2.4 Theory (half a page in the Method section)
- **Proposition.** Dropping each Gaussian with probability r and scaling survivors' opacity by 1/(1−r) leaves the *expected* rendered image unchanged. The expected value of a product of independent terms is the product of their expected values.
- **The exception.** The rasterizer caps per-pixel opacity at 0.99.
  - In DNGaussian's hard-depth pass every Gaussian has opacity 0.95, so the compensated value exceeds the cap once r > 0.04.
  - Its expected opacity then falls by 8 % at r = 0.12 and by 17 % at r = 0.2.
  - In a fixed-opacity pass, compensated dropout **cannot** be made unbiased above r ≈ 0.04. As far as we know this hasn't been pointed out before.
- **What follows.** Only the passes that receive the mask are affected: D's hard pass, D_hard, and D's soft pass only for very opaque Gaussians. E is not affected.

Derivation: RESEARCH_ANALYSIS.md §6.

### 2.5 Contributions (as the report will state them)
1. **The first controlled measurement of the depth × dropout interaction**: a 2×2 factorial plus decoupling, 8 LLFF scenes × 3 seeds, with analysis rules fixed and tagged before any result. It includes **the first dose–response curve of a regularizer interaction**, plus a view-count sweep.
2. **Where dropout should act.** The decoupled vs shared mask, and the hard vs soft depth pass. A short theory shows why compensated dropout is biased in fixed-opacity depth rendering. Result: a zero-cost recipe, **"decoupled dropout"**, if E ≥ D.
3. **Beyond image metrics.** The interaction measured on geometry (multi-view depth consistency), a test-time redundancy probe, and per-scene moderators.
4. **A reproducible harness:** tags, md5-locked data, CI-tested analysis, released run outputs.

### 2.6 Scope decisions (the RESEARCH_ANALYSIS decision sheet, settled)
**IN** = in this plan · **STRETCH** = only if a buffer slot frees up · **FUTURE** = after 15 Nov.

| Item | Decision | Note |
| --- | --- | --- |
| A1 two-axis verdict, A2 margin δ, A3 fixed reference B, A4 primary contrasts | **IN** (§5) | Simplified for the viva |
| A5 bootstrap, Wilcoxon, MSE scale | FUTURE | — |
| A6 adaptive seeds | CUT | No decision point left |
| A7 pre-registration tag | **IN** | `prereg-v1` |
| A8 forest plot | **IN** | — |
| B1 dose | **IN**: γ ∈ {0.1, 1/3} for C/D/E, seed 0 | Block G |
| B2 initialization (dense clouds) | **STRETCH** | Block S |
| B3 views with A | **IN** | Block V (A100) |
| B4 DTU, B5 prior quality, B6 surface, B7 DropoutGS, B8 second host | FUTURE | — |
| B9 noise floor | **IN**, reduced | Block N |
| C1 depth-gradient cosine | STRETCH | Never before the tag |
| C2 `rgb_var_masks`, C3 structure stats | **IN** | In W4/W5 |
| **C4 clamp decomposition → Monte-Carlo depth-bias probe** | **IN** (novelty item 1) | `probes.py` |
| **C5a test-time robustness curve** | **IN** (novelty item 3) | `probes.py` |
| C5b region-stratified error | FUTURE | — |
| **C6 single-pass masking (`D_hard`, `D_soft`)** | **IN** (novelty item 1) | Block P |
| **E5 geometry metric** | **IN** (novelty item 2): multi-view depth consistency | `geometry.py` |
| **Per-scene moderators** | **IN** (novelty item 4) | `scene_factors.py` |
| **Framing** | **IN** (novelty item 5) | §2.1, §2.5 |
| D-a…D-f implementation decisions | **IN** | §3–§4 |
| E1–E3 reporting | **IN** | E4 archive: STRETCH |
| F1 framing, F2 citations, F5 write early | **IN** | CVPR: no. Use the course's template |

---

## 3. Experiment matrix

### 3.1 Blocks
One LLFF run takes about 4.9 min on a T4. All blocks run from tag `v1.0-exp`. Each block has its own output root, so Tier 1's tables stay clean.

| Block | Variants | Scenes × seeds | Runs | Where | Hours | Output root | Matrix file | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **T1 main** | A, B, C, D, E (γ = 0.2) | 8 × 3 (seeds 0–2) | 120 | Colab T4 ×2 | 9.8 T4-h | `outputs/` | `matrix.yaml` | 1 |
| **P pass** | `D_hard`, `D_soft` (γ = 0.2) | 8 × 3 | 48 | Colab T4 ×2 (paired with T1) | 3.9 T4-h | `outputs_pass/` | `matrix_pass.yaml` | 2 |
| **G dose** | `C_g010 D_g010 E_g010 C_g033 D_g033 E_g033` | 8 × seed 0 | 48 | Colab T4 ×2 | 3.9 T4-h | `outputs_dose/` | `matrix_dose.yaml` | 3 |
| **N noise** | `B_r1..B_r4`, `D_r1..D_r4` | fern × seed 0 | 8 | Colab T4 | 0.7 T4-h | `outputs_noise/` | `matrix_noise.yaml` | 4 |
| **V views** | A–E, self-contained | 8 × seed 0 at 2, 3, 6, 9 views | 160 | **A100** (`--shared_gpu`) | 2–3 visits (middle case) | `outputs_views/` | `matrix_views_a100.yaml` | 5 |
| S init (stretch) | `A_mvs … E_mvs` | 8 × seed 0 | 40 | A100 or Colab | ~4 T4-h | `outputs_mvs/` | `matrix_mvs.yaml` | 6 |

**Core runs:** 224 on T4 (18.3 T4-h) + 160 on the A100.

**Block V fallbacks:**
1. Kaggle T4: `matrix_views.yaml`, 2/6/9 views, 120 runs, with 3-view taken from T1.
2. Laptop nights: `matrix_views_a100.yaml`, 160 runs × 7.5 min ≈ 20 h over 3 nights.

**Pairing:**
- Block P and T1's D and E are paired by scene and seed on one GPU type.
- Block G's anchors (A, B and the γ = 0.2 point) come from T1 seed 0.
- Peak drop rates at iteration 6000 are 0.06 / 0.12 / 0.20 for γ = 0.1 / 0.2 / 1/3.

### 3.2 Matrix files
Every variant uses the same three keys. The new `depth_render` values `hard_only` and `soft_only` need `check.sh` step 9 to accept them; that change goes in PR-A. Each file differs from `matrix.yaml` only in the lines shown:

```yaml
# experiments/matrix_pass.yaml
seeds: [0, 1, 2]
n_views: [3]
variants:
  D_hard: {depth_scale: 1, drop_rate: 0.2, depth_render: hard_only}   # mask reaches only the hard-depth pass
  D_soft: {depth_scale: 1, drop_rate: 0.2, depth_render: soft_only}   # mask reaches only the soft-depth pass

# experiments/matrix_dose.yaml
seeds: [0]
n_views: [3]
variants:
  C_g010: {depth_scale: 0, drop_rate: 0.1,      depth_render: shared}
  D_g010: {depth_scale: 1, drop_rate: 0.1,      depth_render: shared}
  E_g010: {depth_scale: 1, drop_rate: 0.1,      depth_render: clean}
  C_g033: {depth_scale: 0, drop_rate: 0.333333, depth_render: shared}
  D_g033: {depth_scale: 1, drop_rate: 0.333333, depth_render: shared}
  E_g033: {depth_scale: 1, drop_rate: 0.333333, depth_render: clean}

# experiments/matrix_noise.yaml
scenes: [fern]
seeds: [0]
variants:
  B_r1: {depth_scale: 1, drop_rate: 0.0, depth_render: shared}   # B_r2..B_r4 identical
  D_r1: {depth_scale: 1, drop_rate: 0.2, depth_render: shared}   # D_r2..D_r4 identical

# experiments/matrix_views_a100.yaml   (fallback: matrix_views.yaml with n_views [2, 6, 9])
seeds: [0]
n_views: [2, 3, 6, 9]
variants: {A, B, C, D, E as in matrix.yaml}
```

**Checkpoints and storage.** Keep checkpoints **only for T1 seed 0** (40 runs ≈ 6.6 GB, split across the two Colab accounts). The probes need them. Every other block runs with `--keep_ckpt_seeds` given no values, so it keeps none. Raw depth adds about 2 MB per run.

### 3.3 Compute

| When | Colab A (Aayush) | Colab D (Disha) | A100 (2 h/weekday) | Kaggle (if it works) |
| --- | --- | --- | --- | --- |
| Fri 30 Oct | — | — | Setup and timing test at `upstream-baseline` | Disha's smoke test (Thu) |
| Sat 31 Oct, eve | T1 seed 0, shard 0/2 | T1 seed 0, shard 1/2 | — | — |
| Sun 1 Nov | T1 seeds 1–2, then P, shard 0/2 | Same, shard 1/2 | — | N |
| Mon 2 Nov | G, shard 0/2 | G, shard 1/2 | V (2-view first) | — |
| Tue 3 Nov | Reruns; N if not done | Reruns | V | — |
| Wed 4 Nov | Probes (T1 seed-0 checkpoints) | Last reruns | V finishes; **freeze 23:59** | — |
| Thu 5 Nov | — | — | Reruns, or stretch S only | — |

T4 need is 18.3 h against roughly 28 h of capacity (two accounts, Sat–Tue). If one Colab account fails, the A100's Monday visit runs T1 seed 2 (40 runs) first.

### 3.4 A100 throughput (not measured yet; the first visit measures it)

Assumptions: about 95 usable minutes per 2-hour window. Parallel slots make runs `shared_gpu: true`, so efficiency numbers come from the T4 only.

| Scenario | Slots × min per run | Runs per visit | By Wed 4 Nov (3 visits) | By Fri 6 Nov (5 visits) | Visits for V (160) |
| --- | --- | --- | --- | --- | --- |
| Pessimistic | 1 × 3.5 | 27 | 81 | 135 | 6. Cut 9-view (120 runs, 5 visits), or use the laptop fallback |
| Middle | 3 × 4.5 | 63 | 189 | 315 | 3 |
| Optimistic | 4 × 3.5 | 108 | 324 | 540 | 2 |

**Ask the lab:** can jobs keep running after the 2-hour window? If yes, the A100 could take T1 too.

---

## 4. Code work (Aayush; frozen at tag `v1.0-exp`, Sat 31 Oct 16:00)

Specs are in PLAN.md §5; definitions of the new pieces are in RESEARCH_ANALYSIS.md Appendix B.

| # | Item | PR | Acceptance test (§6) | Est. |
| --- | --- | --- | --- | --- |
| 1 | Rebase `feat/dropout` onto `main` | A | `check.sh` green | 0.5 h |
| 2 | **W2** dropout port: `drop_comp` in `render`, `render_for_depth`, `render_for_opa`; schedule `γ·t/10000`; one mask per iteration; `"opacity"` returns pre-dropout opacity | A | T1–T4 | 4 h |
| 3 | **W3** `--depth_render clean`, plus **`hard_only` / `soft_only`** (novelty 1): the hard-pass mask is applied for `shared` and `hard_only`, the soft-pass mask for `shared` and `soft_only` | A | T4–T6 | 1 h |
| 4 | **W4** `diag.csv` + **C2** `rgb_var_masks`. `loss_depth_used` follows the per-pass masks | A | Sharp checks on smoke runs | 5 h |
| 5 | **W5** `train_stats.json` + **C3** keys (opacity quantiles, `frac_opacity_gt_0_9`, `knn_dist_q50`) | A | Keys finite | 1.5 h |
| 6 | **W7** random point-cloud race fix | A | Two runs don't share `input.ply` | 0.5 h |
| 7 | **Raw depth** (novelty 2): `render.py` writes `eval/ours_<it>/depth_raw/<view>.npz` (float16 depth and alpha) and `eval/ours_<it>/cameras.json` (per-view intrinsics and world-to-camera). Additive only; no dropout | A | T7 | 1 h |
| 8 | `check.sh`: step 9 accepts `hard_only`/`soft_only`; step 4 still forbids dropout in eval paths | A | `check.sh` green | 0.3 h |
| 9 | **W8** runner + `matrix.yaml`, `matrix_pass.yaml`, `matrix_dose.yaml`, `matrix_noise.yaml`, `matrix_views_a100.yaml`, `matrix_views.yaml`; `--keep_ckpt_seeds` with no values keeps none | B | V1 dry run: 120 / 48 / 48 / 8 / 160 / 120 | 5 h |
| 10 | **W6** FPS in `render.py` | B | `render_stats.json`; renders unchanged (V4) | 1.5 h |
| 11 | Validation: `gpu_smoke.sh`, V1, V4, plus a manual smoke of `D_hard`/`D_soft` | A/B | L2 | 3 h |

That is about 23 h. **Cut order if Fri 21:30 shows a slip:**
1. W6 (FPS), which becomes `v1.1-exp` later without affecting training.
2. Raw depth (item 7). Re-render T1 seed-0 checkpoints later for geometry, seed 0 only.
3. C3.
4. `hard_only`/`soft_only` (item 3; block P is dropped).

**Never cut:** W2, W3 `clean`, W4, W8.

**Analysis scripts after the freeze** (routine class; they don't change training):

| Script | What it does | Owner |
| --- | --- | --- |
| `experiments/probes.py` | Novelty 1 and 3: the robustness curve and the Monte-Carlo depth bias, on T1 seed-0 checkpoints. Writes only to `<run>/probes/`, never to `eval/` | Aayush |
| `experiments/geometry.py` | Novelty 2: multi-view depth consistency from `depth_raw` and `cameras.json` | Aayush |
| `experiments/scene_factors.py` | Novelty 4 | Disha |
| `stats.py` v2, `plots/dose.py`, `views.py` with A, forest plot, pass-localization bars | Analysis and figures | Disha |

---

## 5. Analysis rules v2 (pre-registered; tagged `prereg-v1` before the first Tier 1 metric)

**Unit.** For each variant and scene, average over seeds. Every comparison is paired by scene: report the mean of the per-scene differences, a 95 % t-interval over the 8 scenes (df = 7), and wins out of 8.

**Primary** (T1, γ = 0.2, PSNR; Holm correction across the three):
1. **E − D**
2. **I_D = (D − C) − (B − A)**
3. **I_E = (E − C) − (B − A)**

**Secondary:**
- B − A, C − A, D − B, D − C, E − B, E − C.
- **Pass localization:** D_hard − E and D_soft − E (3 seeds; Holm across the two).
- The same contrasts at γ = 0.1 and 1/3, and at 2/3/6/9 views (seed 0: reported as trends with per-scene points).
- **Geometry:** the primary contrasts recomputed on MVC@5 % (below).

**Margin.** δ = 0.25 dB (for geometry, δ = 0.02 in MVC@5 %).
- "Equivalent to 0" means the 95 % CI lies inside ±δ.
- A CI that crosses 0 but extends beyond ±δ is "inconclusive".

**Verdict for X ∈ {D, E}:**

| | Interaction CI > 0 | Interaction equivalent to 0 | Interaction CI < 0 | Inconclusive |
| --- | --- | --- | --- | --- |
| **X beats B and C** (both CIs > 0) | synergistic | **complementary** | complementary but partly redundant | complementary |
| **X equivalent to B** | — | — | **redundant** | redundant? |
| **X worse than B** (CI < 0) | — | — | **interfering** | worse (unclear why) |
| **Otherwise** | — | — | partly redundant | **inconclusive** |

B (published DNGaussian) is the fixed reference; we never pick "the better of B and C" from the same data. SSIM and LPIPS must point the same way, otherwise the verdict is "metric-dependent".

**Mechanism claims.** Each needs all of its conditions:
- **"Through the depth supervision"**: E − D > 0 (CI); gap(D) > gap(C) (CI); and E − D at γ = 1/3 larger than at γ = 0.1 in ≥ 6/8 scenes.
- **"Through the opacity cap"**: D_hard − E < 0 (CI); D_soft − E has a CI containing 0; and the Monte-Carlo probe shows a positive ("far") hard-pass depth shift that is larger at r = 0.2 than at r = 0.12.

**New measures (definitions):**
- **MVC@5 %, multi-view depth consistency.**
  1. For every ordered pair (i, j) of held-out views, back-project each pixel of view i with its rendered depth.
  2. Project it into view j and compare with view j's rendered depth there.
  3. A pixel counts as valid if both alphas are > 0.5 and it lands inside the image. It is consistent if the relative error is < 5 %.
  4. MVC@5 % is the fraction of valid pixels that are consistent, averaged over pairs. Also report the median relative error.

  It needs no ground truth, so it isn't biased towards the depth prior.
- **Robustness curve.** On T1 seed-0 checkpoints, remove a random fraction f ∈ {0, 0.05, 0.1, 0.2, 0.3} of Gaussians at test time (5 seeded masks; with and without compensation). Report eval PSNR(f) − PSNR(0); the summary is the area under the drop curve. Exploratory.
- **Monte-Carlo depth bias.** Over 32 masks at r = 0.12 and 0.2, take the mean of (masked hard-pass depth − clean hard-pass depth), as the median relative shift over pixels. Run on D and E seed-0 checkpoints.
- **Per-scene moderators.** For each scene:
  - DPT reliability ρ: Spearman between DPT disparity and the inverse depth of COLMAP points (`sparse/0/points3D.bin`, analysis only) in the training views;
  - texture: mean image-gradient magnitude;
  - depth range.

  Correlate each with per-scene E − D, I_D and I_E (Spearman, n = 8; |ρ| > 0.74 for p < 0.05). Exploratory, labelled as such.

**Power (state it in the report).** With 8 scenes × 3 seeds, interactions smaller than about 0.35–0.5 dB are below our detection limit (RESEARCH_ANALYSIS §5.3).

---

## 6. Testing and validation

| Level | What | Who | When | Pass criterion |
| --- | --- | --- | --- | --- |
| **L0 static** | `bash scripts/check.sh` locally and in CI on every PR | Both | Every PR | ALL CHECKS PASSED |
| **L1 unit smoke** (laptop, fern, 1100 iterations) | T1: `--drop_rate 0` creates no mask, and B's PSNR is in 17.3–17.8. T2: no `drop_comp`/`drop_rate` in `render.py`, `metrics.py`, `spiral.py`. T3: mask zero-fraction tracks `0.2·t/10000` ±0.01. T4: E's depth passes get `None`. **T5:** `hard_only` masks only `render_for_depth`. **T6:** `soft_only` masks only `render_for_opa`. **T7:** `depth_raw` is finite and positive where alpha > 0.5; `cameras.json` reprojects view i into itself with zero error. New `train_stats.json` keys finite. Sharp checks pass on every smoke `diag.csv` | Aayush | Thu–Fri | All true |
| **L2 integration** | `gpu_smoke.sh` (A–E) with its PASS line on the PR; V1 dry-run counts; V4 render determinism | Aayush | Fri | All pass |
| **L3 regression** | V3: B at the PR-A head, fern and horns, 6000 iterations, on Colab. Mean within ±0.5 dB of **19.85**; rerun once before debugging | Disha | Sat 11:00 | Pass |
| **L4 pilot** | First 10 real T1 runs: losses finite; Gaussian counts plausible; `drop_rate_now` reaches 0.12 in C/D/E; sharp checks pass; `stats.py` runs on the partial data | Both | Sat night | Pass; otherwise stop, fix, tag `v1.1-exp` |
| **L5 production** | After each shard: count `FAILED` runs, rerun once, note it in the context file and RUNLOG; sharp checks on each finished block | Shard owner | 31 Oct – 4 Nov | Every planned run has `metrics.json` or a logged reason |
| **L6 analysis check** | Recompute one interaction by hand from `runs.csv`. `geometry.py` gives 0 error for i → i and symmetric pairs agree. `probes.py` at f = 0 matches the eval PSNR. Regenerate every figure from frozen outputs | Cross-check: Disha checks Aayush's numbers and vice versa | Thu 5 and Mon 9 Nov | Identical numbers |
| **L7 report check** | Every number traces to a `metrics.json` or a cited table; seeds, GPUs, SSIM implementation and LPIPS backbone stated; citations verified | Both | Mon 9 Nov | Checklist done |

---

## 7. Who does what, and by when

**How to use this section**
- Task IDs: **AY-nn** for Aayush, **DS-nn** for Disha.
- When a task is done, record its ID in your context file. Your agent does that, plus the "Message to Disha" on Aayush's side.
- The 21:30 call (15 min) checks the gate for that day. If a gate fails, apply §8.

**How PRs flow**
- **Aayush's PRs.** His agent runs `pr-gate`. Experiment and gate PRs need Disha's approval; docs-only PRs merge without a reviewer.
- **Disha's PRs.** Her agent posts the review report. Her environment can't set statuses, so Aayush's agent checks the report, sets `agent-review` on the reviewed SHA, and merges through `pr-gate`.

### 7.1 Thu 29 Oct: start the code, clear the queue

| ID | Task | Depends on | Done when | Due |
| --- | --- | --- | --- | --- |
| AY-01 | Merge the plan PR (this file) and the team-context PR once Disha has approved | DS-01 | Both merged; `team-context` branch exists | 21:30 |
| AY-02 | Rebase `feat/dropout` onto `main` | — | `check.sh` green | 12:00 |
| AY-03 | W2 dropout port + W3 (`clean`, `hard_only`, `soft_only`) | AY-02 | L1 T1–T6 pass on the laptop | 23:59 |
| AY-04 | Set `agent-review` on and merge #11 and #14 once Disha's reviewer has approved them; merge #12 after her approval | DS-01, DS-02 | Merged | 23:59 |
| AY-05 | A100 go-bag: git bundle, `conda pack`, data tar with md5. Ask the lab: OS, can jobs outlive the 2-hour window, home-folder persistence | — | Packed; questions sent | 23:59 |
| DS-01 | Read this plan; approve #12 and the team-context PR (gate); confirm your task list at the call | — | Approvals on GitHub | 21:30 |
| DS-02 | Run your reviewer on #11 and #14 and post the reports | — | Reports posted | 18:00 |
| DS-03 | `stats.py` v2: δ rule, fixed reference B, verdict table (§5), and a **factor table** mapping variant IDs (`D_hard`, `C_g033`, `B_r1`…) to depth, γ and `depth_render`. `diag_curves.py` sharp checks keyed by config, not by letter (`hard_only`/`soft_only` must show a gap > 0). Extend fixtures with the new IDs | — | PR open, CI green | Fri 18:00 |
| DS-04 | Kaggle smoke test: one fern run at `upstream-baseline` on a Kaggle T4; record minutes per run | — | Yes/no in your context file | 23:59 |

**Gate at 21:30:** plan agreed; #12, #11 and #14 merged or scheduled; W2 runs on the laptop.

### 7.2 Fri 30 Oct: finish and test the code; set up the A100

| ID | Task | Depends on | Done when | Due |
| --- | --- | --- | --- | --- |
| AY-06 | W4 `diag.csv` + C2; W5 + C3; W7 | AY-03 | Sharp checks pass on smoke runs | 15:00 |
| AY-07 | Raw depth + `cameras.json` in `render.py`; `check.sh` step 9 | AY-03 | T7 passes | 17:00 |
| AY-08 | Upload a **sample run folder** (all 5 variants plus `D_hard`, at 1100 iterations: `diag.csv`, `train_stats.json`, `depth_raw/`, `cameras.json`) to Drive `sparse3dgs/samples/` | AY-06, AY-07 | Folder on Drive; path in context file | 18:00 |
| AY-09 | Open **PR-A** (items 1–8), post the `gpu_smoke.sh` PASS line, run the opus reviewer | AY-06–AY-08 | PR-A open with report | 19:00 |
| AY-10 | **A100 visit 1:** environment, extensions built for sm_80, data md5, timing test (1, 2 and 4 parallel runs of B on fern at `upstream-baseline`) | AY-05 | Minutes per run and slot scaling recorded; §3.4 scenario chosen | Lab window |
| DS-05 | Review PR-A: read the diff, ask questions on the PR, check the sample `diag.csv` with `diag_curves.py` | AY-08, AY-09 | Comments posted | 23:59 |
| DS-06 | `plots/dose.py`, `views.py` with A and per-view interaction, forest plot, pass-localization bar chart, all on fixtures | DS-03 | PR open | Sat 12:00 |
| DS-07 | Related Work skeleton and source list (§9) | — | Doc started | 23:59 |

**Gate at 21:30:** PR-A open; L1 and L2 pass; A100 scenario known.

### 7.3 Sat 31 Oct: code freeze 16:00, Tier 1 starts

| ID | Task | Depends on | Done when | Due |
| --- | --- | --- | --- | --- |
| AY-11 | **PR-B**: W8 runner, all matrix files, W6; V1 dry-run counts | AY-09 | PR-B open, CI green | 12:00 |
| AY-12 | Fix review findings; merge PR-A and PR-B; **tag `v1.0-exp`**, and **`prereg-v1`** on the commit containing §5 | DS-08, DS-09 | Both tags pushed | **16:00** |
| AY-13 | Colab A: T1 seed 0, shard 0/2 (Cell 7b, `TAG = 'v1.0-exp'`) | AY-12 | Running | 16:30 |
| AY-14 | L4 pilot on the first 10 runs | AY-13, DS-03 | Pass, noted in context file | 23:00 |
| DS-08 | **V3 regression** on Colab at the PR-A head; post the result on PR-A | AY-09 | Mean within ±0.5 dB of 19.85 | 11:00 |
| DS-09 | Approve PR-A and PR-B (after an Approve report, the V3 pass and the GPU-smoke line) | DS-08, AY-11 | Approved | 14:00 |
| DS-10 | Colab D: T1 seed 0, shard 1/2 | AY-12 | Running | 16:30 |
| DS-11 | Get DS-03 and DS-06 reviewed (Aayush merges); `aggregate.py` on the first runs | DS-03, DS-06 | Merged; first tables | 23:59 |

**Gate at 21:30: code freeze.** Tags exist; L3 and L4 pass.

### 7.4 Sun 1 Nov: Tier 1 and block P

| ID | Task | Depends on | Done when | Due |
| --- | --- | --- | --- | --- |
| AY-15 | Colab A: T1 seeds 1–2, then block P, shard 0/2 | AY-13 | Shards done; failures listed | 23:59 |
| AY-16 | Write **Method**: 3DGS in a paragraph, three passes, the port, A–E, `hard_only`/`soft_only`, theory (§2.4) | — | Draft in the report doc | 23:59 |
| DS-12 | Colab D: T1 seeds 1–2, then block P, shard 1/2; block N on Kaggle (or Colab after P) | DS-10 | Shards done | 23:59 |
| DS-13 | Related Work draft (1.5 pages; must cite DropoutGS Tab. 3, StereoGS, D²GS, DropAnSH-GS) | DS-07 | Draft | 23:59 |
| DS-14 | `scene_factors.py`: DPT reliability ρ, texture, depth range for the 8 scenes (needs only the data) | — | CSV of 8 rows | Mon 18:00 |

**Gate at 21:30:** at least 80 T1 runs done.

### 7.5 Mon 2 – Wed 4 Nov: remaining blocks, first analysis, experiments freeze

| ID | Task | Depends on | Done when | Due |
| --- | --- | --- | --- | --- |
| AY-17 | A100 visits on Mon, Tue and Wed: block V (`matrix_views_a100.yaml`, `--shared_gpu`, 2-view first). On Mon, T1 seed 2 first if Colab fell short | AY-10, AY-12 | 160 runs (or 120 without 9-view) | Wed 23:59 |
| AY-18 | Colab A: block G, shard 0/2 (Mon); reruns (Tue–Wed) | AY-15 | Done | Tue 23:59 |
| AY-19 | **First T1 analysis** with Disha: `stats.py` on 3 seeds; preliminary verdict | T1 complete | Verdict note in context file | Mon 23:59 |
| AY-20 | Write **Experimental setup** | — | Draft | Mon 23:59 |
| AY-21 | `geometry.py` (MVC@5 %) on the sample folder; L6 self-consistency test | AY-08 | PR open | Tue 23:59 |
| AY-22 | `probes.py` (robustness + Monte-Carlo bias) on one checkpoint; f = 0 matches the eval PSNR | AY-12 | PR open | Tue 23:59 |
| AY-23 | Run `probes.py` on all T1 seed-0 checkpoints (Colab); `geometry.py` on all runs | AY-21, AY-22 | `probes/` and geometry CSVs | Wed 23:59 |
| AY-24 | Figure F1 (D vs E vs D_hard/D_soft iteration diagram) | — | PDF | Wed 23:59 |
| DS-15 | Colab D: block G, shard 1/2 (Mon); reruns (Tue–Wed) | DS-12 | Done | Tue 23:59 |
| DS-16 | Tables T1–T3 + forest plot from real T1 data | T1 complete | Generated by script | Tue 23:59 |
| DS-17 | Dose plot (G), pass-localization chart (P), views plot (V) | Blocks done | Figures by script | Wed 23:59 |
| DS-18 | Per-scene correlations (`scene_factors.py` × per-scene contrasts) | DS-14, DS-16 | Table T5 | Wed 23:59 |

**Gates:**
- Mon 21:30: T1 complete (120).
- Tue 21:30: G, P and N complete.
- **Wed 23:59: experiments frozen**; everything on Drive.

### 7.6 Thu 5 – Mon 9 Nov: analysis and writing

| ID | Task | Due |
| --- | --- | --- |
| AY-25 | Mechanism analysis: gap, variance curves, structure stats, pass localization, Monte-Carlo bias, robustness, geometry; final verdict. L6 cross-check of Disha's numbers. A100: reruns or stretch S only | Thu 5 |
| DS-19 | Final tables and every figure regenerated by script from frozen outputs; L6 cross-check of Aayush's numbers | Thu 5 |
| AY-26 | Write Mechanism + Verdict + Discussion | Fri 6 |
| DS-20 | Write Introduction + Results | Fri 6 |
| AY-27 | Limitations, failure cases (2–3 scenes, RGB + depth) | Sat 7 |
| DS-21 | Results text, captions, efficiency table | Sat 7 |
| AY-28 + DS-22 | Swap sections and review; abstract and conclusion together | Sun 8 |
| AY-29 | L6 regeneration from scratch; L7 checklist; README "how to reproduce"; tag `v1.0-report` | Mon 9 |
| DS-23 | Citations check; L7 checklist | Mon 9 |
| AY-30 + DS-24 | Viva rehearsal (§10.3) | Mon 9 |

**Gates:** Sat 21:30 full draft; Mon 21:30 submission-ready.

### 7.7 Tue 10 – Sun 15 Nov
- **Tue 10:** submit (both).
- **11–15 Nov:** buffer: fixes only, plus viva slides and rehearsal. No new experiments unless a result is broken.

### 7.8 Handoffs at a glance

| From → to | What | By |
| --- | --- | --- |
| Aayush → Disha | Sample run folder (AY-08) | Fri 30, 18:00 |
| Aayush → Disha | PR-A for review (AY-09) | Fri 30, 19:00 |
| Disha → Aayush | V3 result (DS-08) and approvals (DS-09) | Sat 31, 11:00 / 14:00 |
| Aayush → Disha | Tags `v1.0-exp` and `prereg-v1` (AY-12) | Sat 31, 16:00 |
| Disha → Aayush | `stats.py` v2 + plots for review/merge (DS-03, DS-06) | Fri 30 / Sat 31 |
| Both | Run status per shard, in the context files | Every evening |
| Aayush → Disha | Probe and geometry outputs (AY-23) | Wed 4 |
| Disha → Aayush | Tables and figures (DS-16, DS-17) | Tue 3 / Wed 4 |

---

## 8. Risks, triggers and the descoping ladder

| ID | Risk | Detect by | Response |
| --- | --- | --- | --- |
| R1 | Code not merged by Sat 16:00 | Fri 21:30 | Cut order in §4. Freeze may slip to **Sun 1 Nov 12:00**, then descope to Level 1 |
| R2 | V3 regression fails | Sat noon | Diff against `upstream-baseline` (flag defaults first); at most 3 h |
| R3 | Pilot finds a bug after the tag | Sat night | Stop, fix, tag `v1.1-exp`, rerun only affected runs, log it |
| R4 | A100 setup fails or is pessimistic | Fri visit | V → Kaggle T4 (`matrix_views.yaml`), or laptop nights (160 runs ≈ 20 h, nights of 1–3 Nov) |
| R5 | Colab limits | Any session | Switch account. Priority: T1 > P > G > N. A100 takes T1 seed 2 |
| R6 | T1 not complete by Mon 23:59 | Mon call | Descope to Level 2 |
| R7 | Drive quota | Storage page | Checkpoints only for T1 seed 0 (§3.2); delete probe checkpoints after AY-23 |
| R8 | Writing slips | Sat 7: full draft? | Views → one figure; per-scene moderators → appendix; Discussion → half a page |
| R9 | Illness or exams | Any | Buffer is for writing only, never new experiments |

**Descoping ladder.** Each level is a complete, defensible report.

| Level | Contents | Used when |
| --- | --- | --- |
| **0 (plan)** | T1 + P + G + N + V + geometry + probes + per-scene + theory | Default |
| **1** | T1 + P + G + N; V at 2/3/6 views; geometry seed 0 only | Freeze slips to Sunday |
| **2** | T1 + G at γ = 1/3 only; P seed 0; no V | T1 late by Monday |
| **3 (minimum)** | T1 with 2 seeds + G at γ = 1/3 | Severe trouble; still answers RQ1–RQ3 |

---

## 9. Report plan

**Format:** the course's template and page limit (still to confirm with the instructor).

| Section | Owner | Content | Draft by |
| --- | --- | --- | --- |
| Abstract, Conclusion | Both | The verdict and the placement recipe in two sentences, with numbers | Sun 8 |
| 1 Introduction | Disha | Sparse views → overfitting; two families of fixes; prior work stacked them (DropoutGS Tab. 3, StereoGS) but never measured the interaction or where the mask should act; RQ1–RQ5; contributions (§2.5) | Fri 6 |
| 2 Related work | Disha | Sparse-view NeRF and 3DGS; depth priors (DNGaussian, FSGS, DNGaussian++); dropout family (DropGaussian, DropoutGS, D²GS, DropAnSH-GS, UGOD); dropout theory (Srivastava 2014, Wager 2013) | Sun 1 |
| 3 Method | Aayush | 3DGS; DNGaussian's passes; the port (D1); A–E, D_hard, D_soft; F1; the proposition and the cap table | Sun 1 |
| 4 Experimental setup | Aayush | Split, resolution, iterations, seeds, GPUs, metrics, MVC@5 %, probes, analysis rules with `prereg-v1` | Mon 2 |
| 5 Results | Disha | T1 main table; T3 contrasts and verdict; forest plot (F6); dose (F7); pass localization (F8); views (F5); efficiency (T4) | Sat 7 |
| 6 Mechanism and geometry | Aayush | Diagnostics (F2); Monte-Carlo bias; robustness curves (F9); geometry vs image interaction (F10); per-scene moderators (T5, can go to the appendix) | Fri 6 |
| 7 Discussion, limitations | Both | Failure cases (F4); limitations: one host, one dataset, random init, DPT, seed-0 sweeps, nondeterminism; future work (§11) | Sat 7 |

**Figures and tables:**
- F1 iteration diagram (incl. D_hard/D_soft)
- F2 diagnostics
- F3 qualitative grid
- F4 failure cases
- F5 views
- F6 forest plot
- F7 dose curves
- **F8 pass localization**
- **F9 robustness curves**
- **F10 geometry vs PSNR interaction**
- T1 main table
- T2 per-scene table
- T3 contrasts and verdict
- T4 efficiency
- **T5 per-scene moderators**

Every figure comes from a script run on the frozen outputs.

---

## 10. Viva: assessment and preparation

### 10.1 Assessment
These are my estimates against a typical introductory ML course project (one model applied to one dataset, plus an accuracy table).

| Dimension | Typical intro-ML project | This project (design) | **Today** | After this plan |
| --- | --- | --- | --- | --- |
| Problem and motivation | 4 | 8.5 | 8.5 | 8.5 |
| Technical depth | 4 | 8.5 | 5 | 8.5 |
| Experimental design | 3 | 9 | 9 (on paper) | 9 |
| Novelty | 3 | 8 | — | 8 |
| Results and evidence | 5 | — | **2** | 7.5–8.5 |
| Statistical analysis | 2 | 8 | 3 | 7.5–8 |
| Reproducibility | 3 | 9.5 | 9 | 9.5 |
| Communication | 5 | — | **1** | 7.5–8 |

**Ambition: about 8.5–9/10. Completion today: about 3/10. Projected: about 8–8.5/10.** The viva mark depends on whether **both** of you can explain every part in your own words, including AI-assisted code; disclose AI use as the course requires. Cut anything you can't defend.

### 10.2 The project in the course's terms
| Course concept | Where it appears |
| --- | --- |
| Overfitting, generalization | 3 training views: high training PSNR, poor novel-view PSNR |
| Regularization | Depth prior (external knowledge) and dropout (structural) |
| Dropout | DropGaussian on Gaussians; inverted-dropout scaling 1/(1−r); where to apply it (D vs E, hard vs soft) |
| Bias–variance | Dropout as a variance penalty; the opacity cap as a bias |
| Train/test split | Every 8th LLFF image held out; dropout never active at evaluation |
| Hyperparameters | γ (dose), number of views |
| Significance | Paired differences, CIs, seeds, "inconclusive" |

### 10.3 Likely questions
1. **What is 3D Gaussian Splatting?** Many 3-D Gaussians (position, shape, opacity, colour), projected and blended front to back, and fitted to photos by gradient descent.
2. **Why do few views fail?** Many 3-D scenes explain 3 photos equally well, so the optimizer overfits and produces floaters and wrong geometry.
3. **What does DNGaussian add?** Monocular depth (DPT) as extra supervision, via two extra depth passes per iteration. Patches are normalized because monocular depth has unknown scale and shift.
4. **What does DropGaussian add?** It removes each Gaussian with probability r each iteration and scales survivors' opacity by 1/(1−r). At test time it uses all Gaussians.
5. **Why scale by 1/(1−r)?** To keep the expected image unchanged. The exception is the 0.99 opacity cap, which bites in the fixed-opacity (0.95) hard-depth pass.
6. **Why five variants?** A 2×2 design isolates each effect and their interaction; E tests where the mask should act.
7. **D vs E? D_hard vs D_soft?** D: the mask also hits both depth passes. E: neither. D_hard and D_soft: only one of them, so we can see which pass carries any harm. The cap predicts the hard pass.
8. **Interaction contrast?** I = (D − C) − (B − A): depth's benefit with dropout minus without. 0 means additive; negative means overlap or conflict.
9. **Why seeds and scene pairing?** Runs vary by 0.2–0.4 dB; pairing within scenes removes scene difficulty.
10. **Why a t-interval with df = 7?** 8 paired differences: a small sample with unknown spread.
11. **What does "inconclusive" mean?** The CI includes 0 but is too wide to show the effect is negligible.
12. **What is MVC@5 %?** Project each held-out view's depth into another held-out view and check agreement within 5 %. It measures geometry without ground truth.
13. **What does the robustness curve show?** How much PSNR drops when Gaussians are removed at test time, i.e. how redundant the learned scene is.
14. **PSNR, SSIM, LPIPS?** Pixel error on a log scale; local structure; deep-feature distance (lower is better).
15. **Why not compare with the published numbers?** Different initialization, metric code and hardware; only our controlled variants are compared statistically.
16. **Same seed, different result?** GPU atomic additions in the backward pass sum in a varying order.
17. **With more time?** DTU, dense initialization, a stronger depth model, a mask-aware depth loss.

---

## 11. After 15 Nov: the publication path

If the results are clear, extend toward TMLR (rolling review) or a CVPR 2027 workshop (typically due Feb–Mar 2027). In order:
1. Initialization moderator (block S).
2. DTU 3-view.
3. Bootstrap, adaptive seeds and full equivalence tests.
4. Gradient-cosine diagnostic and region-stratified error.
5. A second host.
6. A mask-aware depth loss as a method.

RESEARCH_ANALYSIS.md §7–§9 has the details.
