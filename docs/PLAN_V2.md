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
One LLFF run takes about 4.9 min on a T4. All blocks run from tag `v1.0-exp`, except block CA (`v1.1-exp`, §5.1). Each block has its own output root, so Tier 1's tables stay clean.

| Block | Variants | Scenes × seeds | Runs | Where | Hours | Output root | Matrix file | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **T1 main** | A, B, C, D, E (γ = 0.2) | 8 × 3 (seeds 0–2) | 120 | Colab T4 ×2 | 9.8 T4-h | `outputs/` | `matrix.yaml` | 1 |
| **P pass** | `D_hard`, `D_soft` (γ = 0.2) | 8 × 3 | 48 | Colab T4 ×2 (paired with T1) | 3.9 T4-h | `outputs_pass/` | `matrix_pass.yaml` | 2 |
| **G dose** | `C_g010 D_g010 E_g010 C_g033 D_g033 E_g033` | 8 × seed 0 | 48 | Colab T4 ×2 | 3.9 T4-h | `outputs_dose/` | `matrix_dose.yaml` | 3 |
| **N noise** | `B_r1..B_r4`, `D_r1..D_r4` | fern × seed 0 | 8 | Colab T4 | 0.7 T4-h | `outputs_noise/` | `matrix_noise.yaml` | 4 |
| **V views** | A–E, self-contained | 8 × seed 0 at 2, 3, 6, 9 views | 160 | **A100** (`--shared_gpu`) | 2–3 visits (middle case) | `outputs_views/` | `matrix_views_a100.yaml` | 5 |
| **COMP compensation** (§5.1 + §5.2) | `C_ca`, `D_ca` (`drop_mode: clamp_aware`), `C_nc`, `D_nc` (`drop_mode: nc`), γ = 0.2 | 8 × 3 | 96 | lab A4000 | ~2 h (3 slots) | `outputs_comp/` | `matrix_comp.yaml` | from `v1.1-exp` (16 Nov: replaces block CA) |
| **DDOSE depth dose** (§5.2) | `B_ds050`, `D_ds050`, `B_ds200`, `D_ds200` (depth_scale 0.5 / 2) | 8 × 3 | 96 | lab A4000 | ~2 h (3 slots) | `outputs_depthdose/` | `matrix_depthdose.yaml` | from `v1.1-exp` |
| **NOISE2** (§5.2.7) | `A_r1..4`, `B_r1..4`, `C_r1..4`, `D_r1..4`, seed 0 | room, horns, trex (drawn) × 1 | 48 | lab A4000 | ~1 h (3 slots) | `outputs_noise2/` | `matrix_noise2.yaml` | from `v1.1-exp` |
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
Every variant uses the same three keys, plus an optional fourth, `drop_mode`, in block CA (§5.1). The new `depth_render` values `hard_only` and `soft_only` need `check.sh` step 9 to accept them; that change goes in PR-A. Each file differs from `matrix.yaml` only in the lines shown:

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

### 5.1 Addendum (3 Nov, tag `prereg-v2`, before any block-CA result): clamp-aware dropout

**Method.** `--drop_mode clamp_aware` (`utils/dropout.py`): each Gaussian's drop rate is lowered to r_i = min(r, 1 − α_i/0.99), or 0 when α_i ≥ 0.99, where α_i is its opacity in that pass (0.95 in the hard-depth pass). Survivors are scaled by 1/(1 − r_i), so the compensated opacity never exceeds the rasterizer's alpha cap and Proposition 1 holds in every pass. All passes of an iteration threshold one uniform draw per Gaussian; the photometric mask is re-capped after the soft pass's optimizer step. The default `--drop_mode uniform` stays DropGaussian verbatim. (A fixed base opacity of 0.99(1 − r) in the hard pass is *not* a fix: its survivors land on exactly the clamped value, so it changes nothing.)

**Unbiasedness costs dose.** For a Gaussian with α ≥ 0.99(1 − r), no rescaling of the survivors' opacity can make dropout at rate r unbiased, since dropping it alone lets through light with probability r. The largest unbiased rate is 1 − α/0.99, and clamp-aware dropout uses exactly that. With the opacities we observe (median ≈ 0.99 in the smoke runs), clamp-aware is therefore a much **smaller** dose than DropGaussian's: about 0 for saturated Gaussians and at most 0.0404 in the hard pass. Uniform dropout keeps the dose and accepts the bias; clamp-aware removes the bias and gives up most of the dose. The two can't both be kept.

**Recorded dose.**
- `diag.csv`: `drop_rate_eff` (mean per-Gaussian rate of the soft/photometric mask) and `drop_rate_eff_hard`. Both equal `drop_rate_now` in uniform mode.
- `train_stats.json`: `drop_rate_mean`, `drop_rate_eff_mean`, `drop_rate_eff_hard_mean` over all iterations.
- `frac_over_cap`: the share of Gaussians that uniform compensation would push over the cap in the soft and photometric passes. The hard pass is fully exposed whenever r > 0.0404.
- Runs from `v1.0-exp` (T1, G: the uniform reference) don't have these fields. For them, `drop_rate_mean` is the schedule mean, 0.30005·γ over 6000 iterations, and the uniform `frac_over_cap` comes from the `v1.1-exp` smoke and probe runs. The `_hard` fields mean something only when `depth_scale` > 0 (no hard pass runs in C or C_ca).

**Block CA:** `C_ca`, `D_ca` (γ = 0.2), 8 scenes × seeds 0–2 = 48 runs on Colab T4, from tag `v1.1-exp` (`experiments/matrix_ca.yaml`, output root `outputs_ca/`). *(16 Nov: block CA is folded into block COMP, `experiments/matrix_comp.yaml`, output root `outputs_comp/`, which adds `C_nc`/`D_nc`; the gate below is superseded by EXECUTION_PLAN_15-21Nov W5, which runs it unconditionally.)* Paired with T1's C and D, with E and A as references. A–E, D_hard and D_soft are unchanged in `v1.1-exp`. Runs from `v1.0-exp` have no `drop_mode` key; it counts as `uniform`.

**Gate (decided Thu 5 Nov 12:00, recorded in RUNLOG).** Run block CA only if both hold:
1. T1 seeds 0 and 1 are complete (80 runs);
2. the bias is not negligible: in T1 seed 0, the median over scenes of `frac_opacity_gt_0_9` in C or D is ≥ 0.05, **or** the Monte-Carlo probe on the seed-0 D checkpoints gives a median relative hard-pass depth shift ≥ 0.5 % at r = 0.12.

If the gate fails, block CA is not run, and the report states the measured exposure and probe values.

**H7: does DropGaussian's benefit need the biased, full-dose dropout of opaque Gaussians?**
- **Dose reference.** C at nominal mean rates 0 (A), and γ = 0.1, 0.2, 1/3 (blocks G and T1, seed 0), is a uniform dose–response curve: PSNR against `drop_rate_mean`. C_ca is placed on it at its `drop_rate_eff_mean`, and D_ca is placed on the D curve the same way.
- **H7a, C_ca vs C.** Report C_ca − C and the dose ratio (`drop_rate_eff_mean` of C_ca over that of C). Reading:
  - C_ca − C ≥ 0 (equivalent or better): an unbiased dose a fraction of the size does at least as well, so the biased full-dose dropout of opaque Gaussians adds nothing.
  - C_ca − C < 0, and C_ca at or below the uniform curve at its effective dose: the loss is explained by the dose; there is no evidence that the bias matters.
  - C_ca − C < 0, but C_ca above the curve at its effective dose: unbiasedness helps at equal dose, but the dose cut costs more. Exploratory, since the curve is seed 0 only.
- **H7b, D_ca vs D.** The same three readings on the D curve. The placement is one-dimensional and approximate: D_ca's hard pass also gets a lower dose (≤ 0.0404), which no point on the uniform D curve reproduces.
- **D_ca − E** is reported but not read as evidence about the clamp bias: D_ca's depth passes get a much lower dose, so D_ca moves toward E by construction.
- **Falsified if** C_ca − C and D_ca − D are both equivalent to 0. In that case DropGaussian's benefit does not depend on the biased dropout of opaque Gaussians.

**Analysis.** §5 rules, paired by scene over seeds 0–2: 95 % t-intervals, δ = 0.25 dB, Holm across the two contrasts C_ca − C and D_ca − D. The dose-curve placement is exploratory (seed 0, interpolation between the measured γ). Secondary: `frac_over_cap` and `drop_rate_eff` over training, uniform vs clamp-aware.

### 5.2 Addendum (16 Nov, tag `prereg-v3`): DTU and Blender, error budget, robustness, compensation, audit

**Tag:** `prereg-v3`, on the commit that adds this section, **before** any DTU, Blender, compensation, depth-dose, noise2,
views seeds 1–2 or error-budget result exists.
**Written:** Mon 16 Nov 2026 (IST), one day after the planned Sun 15 Nov 23:59. §5 (tag `prereg-v1`) still holds for
LLFF T1 and blocks P, G, N and V. §5.1 (clamp-aware, #27, tag `prereg-v2` when #27 merges) comes before this section
once #27 is merged; 5.2.9 states the compensation contrasts itself, so they are covered whichever tag lands first.
This addendum only adds rules; where it is silent, §5 applies. Notation as in §5: per-scene means over seeds, contrasts
paired by scene, 95 % t-intervals over scenes, δ = 0.25 dB on PSNR, the class order "≈0" → ">0"/"<0" →
"inconclusive", Holm within the families named below.

#### 5.2.1 New datasets and their frozen protocol

**DTU, 3 views.**
- Scans: the 15 of DNGaussian / RegNeRF: 8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114.
- Split: `scene/dataset_readers.py` (DTU branch). Train views 25, 22, 28; test = the 25 indices of 0–48 that are neither
  in `[25, 22, 28, 40, 44, 48, 0, 8, 13]` nor in the excluded list `[3–7, 16–21, 36–39]`.
- Metrics: **masked** PSNR, SSIM, SSIM_sk and LPIPS on the 25 test views, with the RegNeRF `idrmasks`, computed as
  `metrics_dtu.py` does (resize /4, background composited). Written by `experiments/eval_dtu.py` as
  `results_eval.json` with the added key `"masked": true`.
- Hyperparameters: the flags of `scripts/run_dtu.sh`, frozen in `experiments/matrix_dtu.yaml` at tag `v1.2-exp`.
- Data: poses from COLMAP run on the CPU (`convert.py --no_gpu`, conda env `colmap`), DPT depth run **once**; the archive `dtu_with_depth.tar` is frozen by md5,
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
**Throughput rule** (schedule of the 16 Nov catch-up plan: DTU and Blender start Wed 18 Nov from `v1.2-exp`): if DTU has
fewer than 150 of 225 runs by Thu 19 Nov 12:00, Blender runs seeds 0–1 only. Runs stop Fri 20 Nov 12:00; missing runs
are reported as missing.

#### 5.2.2 Replication rules (X4)

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

#### 5.2.3 Cross-dataset hypotheses

- **X5 (confirmatory, one-sided):** D − B on DTU > D − B on LLFF (PSNR). Test: Welch t on the per-scene D − B values
  (15 DTU scans vs 8 LLFF scenes), one-sided α = 0.05. Applies only if DTU passes go/no-go.
- **H-PD (exploratory, 3 points):** the rank order of D − B is LLFF ≤ DTU ≤ Blender (dropout helps more where the depth
  prior is weaker; DPT is out of domain on synthetic objects). Reported as holds / does not hold, no test.
- If Blender GT depth is available, the per-object DPT reliability (Spearman of DPT disparity vs GT inverse depth in the
  train views) is reported against per-object D − B. Exploratory.

#### 5.2.4 N1, the error budget (BV1–BV6)

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

#### 5.2.5 N4, the interaction on several scales

- I_D, I_E, E − D and D − B on **PSNR, MSE, RMSE and SSIM**, using the per-run mse from N1 (exact, not converted from
  mean PSNR); RMSE = √mse per run; PSNR and SSIM from `results_eval.json`.
- **"Robust"** = the point estimates have the same sign on all four scales (MSE and RMSE signs flipped so that
  "better" has one sign). Otherwise **"scale-dependent"**. Reported per dataset in
  `results/tables/interaction_scales_<dataset>.csv`; framed with Loftus (1978): a non-crossover interaction can be
  removed by a monotone rescaling, so only scale-robust interactions are interpreted.

#### 5.2.6 N5, depth dose × dropout

- Block depth-dose (`matrix_depthdose.yaml`, LLFF, seeds 0–2, from `v1.1-exp`): B and D at depth_scale 0.5 and 2; A and C from T1
  (they have no depth loss, so they serve every dose). I_D(s) = (D_s − C) − (B_s − A) for s ∈ {0.5, 1, 2}
  (s = 1 is T1's I_D).
- **Prediction:** |I_D(s)| < δ at every dose, i.e. each CI inside ±0.25 dB ("additivity is not an artefact of
  DNGaussian's depth weight"). Holm across the three doses. Each dose gets its §5 class; the prediction holds only if
  all three are "≈0". Also reported: B_s − A and D_s − C against s (trend).

#### 5.2.7 G6 / N2(a): noise floor beyond fern, variance components, power

**Scene draw (rule fixed here, before the draw).**
`random.Random(15112026).sample(["flower", "fortress", "horns", "leaves", "orchids", "room", "trex"], 3)`
(the 7 non-fern LLFF scenes in alphabetical order, Python's `random` module). Evaluated after writing this rule, it
gives **room, horns, trex** (CPython 3.13; to be confirmed on the lab's Python 3.7 before `matrix_noise2.yaml` is
written; if it differs, the 3.7 result is used and the difference noted in RUNLOG).
Block noise2 (`matrix_noise2.yaml`, from `v1.1-exp` or `v1.2-exp`): variants A, B, C, D, 4 same-seed repeats each (`_r1.._r4`, seed 0) on the 3 drawn scenes, 48 runs; with
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

#### 5.2.8 N2(b), the evidence audit (protocol)

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

#### 5.2.9 Compensation study (LLFF, block comp)

- **`nc` mode (`--drop_mode nc`), defined here:** the same schedule r = drop_rate · iteration / 10000 and the same random
  draw as `uniform`, but the surviving Gaussians keep their opacity × 1 instead of × 1/(1 − r), in **every pass the mask
  reaches** (the photometric pass, and in D the hard- and soft-depth passes). At drop_rate 0 no mask is created.
- Variants `C_nc`, `D_nc` and `C_ca`, `D_ca` (clamp-aware, §5.1), with the depth_scale and drop_rate of C and D;
  8 scenes × seeds 0–2 = 96 runs (`matrix_comp.yaml`), from tag `v1.1-exp` (#27 + `nc`).
- Contrasts: **C_nc − C, D_nc − D, C_ca − C, D_ca − D** (PSNR primary), paired by scene with T1's C and D. Holm across
  the four. Each also gets TOST equivalence with δ = 0.25 dB (equivalent if both one-sided p < 0.05; reported
  alongside the §5 class, which it does not replace).
- §5.1's readings of H7a/H7b (dose placement) remain as written there, exploratory.
- **§5.1's Thu 5 Nov gate is superseded:** the block runs regardless of it, and the gate quantities are reported only
  descriptively.

#### 5.2.10 N3 spectral budget (optional, only if time)

Each error image r − y is split into 5 radial FFT bands of equal width in normalised frequency; by Parseval the band
MSEs sum to the total MSE (asserted). Contrasts and the N1 bias/variance split are computed per band.
**Prediction:** depth and dropout effects overlap in the mid and high bands and separate in the low band. Exploratory.

#### 5.2.11 Exploratory disclosure

- **The LLFF results (T1 and blocks P, G, N, V seed 0; `results/RESULTS.md`) were seen before this addendum was written.**
  Nothing new had been run: no compensation, depth-dose, noise2, DTU, Blender or views seeds 1–2 result existed, and no
  eval render had been opened for the error budget.
- The MSE-scale and dataset-heterogeneity checks discussed on 15 Nov (`research_strategy_15nov.md`) were computed or
  reasoned about **after** seeing T1 results. Anything from them in the report is labelled exploratory.
- The 6-view interaction (≈ −0.9 dB, seed 0) motivated the views seeds 1–2 block; its seeds 1–2 result is
  confirmatory only for the sign at 6 views, under the §5 rules.
- Every analysis not listed in this addendum or in §5/§5.1 is exploratory and labelled as such.

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
