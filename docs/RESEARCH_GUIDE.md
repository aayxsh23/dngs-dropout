# Research guide: depth priors × Gaussian dropout in sparse-view 3DGS

**Living document.** Last updated Mon 16 Nov 2026, 21:45 IST · Maintainers: Aayush and Disha · Submission **Sat 21 Nov, 20:00** (hard deadline 23:59, per [EXECUTION_PLAN_15-21Nov.md](EXECUTION_PLAN_15-21Nov.md))

**Who this is for.** The two of us, this week, for three jobs:
- running the experiments;
- writing the report;
- preparing for the viva.

It should make sense with no other file open. Where it summarizes a plan, the plan wins; the order of authority is in §0.

**Companion snapshot.** [STATUS_AUDIT_16Nov.md](STATUS_AUDIT_16Nov.md) is the evidence-backed audit of what was done up to tonight. This guide links to it; it does not repeat it.

**Conventions.**
- Every result number cites the file it comes from.
- "(inference)" marks a reading that is ours and not recorded anywhere.
- "unknown" means nobody has written it down.
- Task labels: **task A1** = Aayush's task 1 in the execution plan; **task D1** = Disha's task 1; **decision D1** = the dropout-schedule decision of 13 Oct. Variant letters (A–E) are always written plainly.

---

## Contents
0. [Sources and their order of authority](#0-sources-and-their-order-of-authority)
1. [The research question and how it evolved](#1-the-research-question-and-how-it-evolved)
2. [Background: DNGaussian, DropGaussian and our variants](#2-background-dngaussian-dropgaussian-and-our-variants)
3. [Hypotheses and predictions](#3-hypotheses-and-predictions)
4. [Experimental design](#4-experimental-design)
5. [What has been done: results so far](#5-what-has-been-done-results-so-far)
6. [What remains: the plan to Sat 21 Nov](#6-what-remains-the-plan-to-sat-21-nov)
7. [Analysis and statistics plan](#7-analysis-and-statistics-plan)
8. [Report, figures and definition of done](#8-report-figures-and-definition-of-done)
9. [Runbooks](#9-runbooks)
10. [Glossary, decision log, file index](#10-glossary-decision-log-file-index)

---

## 0. Sources and their order of authority

When two plans disagree, the one higher in this list wins:

| Rank | File | Date | What it still governs |
| --- | --- | --- | --- |
| 1 | [EXECUTION_PLAN_15-21Nov.md](EXECUTION_PLAN_15-21Nov.md) | 15 Nov | Everything this week: workstreams W1–W12, tasks A1–A13 and D1–D13, gates, handoffs, §9 definition of done |
| 2 | [PLAN_V2.md](PLAN_V2.md) | 29 Oct | The LLFF blocks (T1/P/G/N/V), the pre-registered rules of §5 (`prereg-v1`), hypotheses H1–H6. New pre-registration goes in its §5.2 (task D1) |
| 3 | [RESEARCH_ANALYSIS.md](RESEARCH_ANALYSIS.md) | 14 Oct | Theory (§6), power (§5.3), literature (§4), future work (§9) |
| 4 | [PLAN.md](PLAN.md) | 12–14 Oct | The code specs W2–W13 (§5), decisions D1–D11 (§3), reference numbers (§10) |
| 5 | [research_proposal_sparse_view_3dgs_2.md](research_proposal_sparse_view_3dgs_2.md) | 12 Oct | Background, the Colab and lab runbooks |

The invariants in [AGENTS.md](../AGENTS.md) bind every plan.

**Where these rules conflict (audit I-3, I-6, I-8):**
- AGENTS.md and CLAUDE.md call the PLAN.md status table "the single place" for status. PLAN_V2 points to the team context files instead, and the new plan to RUNLOG plus the 21:30 sync. All three stopped being updated between 14 Oct and 4 Nov.
- **This guide proposes:** events go in RUNLOG, and the live status of this week goes in the table in §6.1 here. Agree it at the next sync.

---

## 1. The research question and how it evolved

### 1.1 The question now (15 Nov)

**Working title:** *Where does sparse-view 3DGS error come from? A bias–variance account of depth priors and Gaussian dropout across three datasets.*

**Core question (unchanged since 12 Oct).** In sparse-view 3D Gaussian Splatting, are two regularizers complementary, redundant or interfering?
- DNGaussian's monocular-depth regularization;
- DropGaussian's random Gaussian dropout.

A second part asks whether it matters that the dropout mask also reaches the depth supervision.

**What the 15 Nov plan adds:**
1. **An explanation: the error budget (N1).** We split each variant's test error into a systematic part (bias²) and a run-to-run part (variance). The hypothesis: depth mainly cuts bias and dropout mainly cuts variance. If so, their effects add up, and this explains our additive result. It predicts a "ceiling" for what dropout can add, and a crossover at 9 views where depth starts to *add* bias.
2. **Generality.** DTU 3-view (required) and Blender 8-view, both under the protocol frozen from LLFF.
3. **Rigor (N2).**
   - a measured noise floor;
   - a power model;
   - an audit asking how many published ablation gains lie within single-run noise.
4. **Defensibility:**
   - N4: the interaction on several metric scales;
   - N5: a depth-dose surface;
   - G6: the noise floor beyond fern.
5. **Kept from the earlier memo:** the compensation study (uniform vs clamp-aware vs no compensation).

### 1.2 How the framing changed

| Date | Framing | Why it changed |
| --- | --- | --- |
| 12 Oct (playbook) | "Are depth priors and dropout complementary, redundant or interfering?" Five variants A–E on LLFF; deadline 20 Oct | The starting design |
| 14 Oct ([RESEARCH_ANALYSIS](RESEARCH_ANALYSIS.md) §1) | Not "is the combination new" but "is the interaction additive, sub-additive or interfering, measured in a factorial, and does the mask's placement matter?" | DropoutGS (Tab. 3), StereoGS and D²GS had already stacked the two, so "is the combination open?" would not survive review. The deadline moved to 15 Nov |
| 29 Oct ([PLAN_V2](PLAN_V2.md) §2) | *Where should dropout act?* Added pass localization (D_hard/D_soft), geometry (MVC@5%), probes, per-scene moderators and a pre-registration | Make the null-or-not result informative through mechanism tests |
| 3 Nov (#27, unmerged) | Clamp-aware dropout (H7): an unbiased compensated dropout under the 0.99 alpha cap | The opacity-cap theory (§2.4) suggested a fix |
| 6–11 Nov (results) | Both D and E are **complementary**; every interaction is **inconclusive** in PSNR (≈ additive); both mechanism claims are **not supported** | The data: [results/RESULTS.md](../results/RESULTS.md) |
| 15 Nov ([EXECUTION_PLAN](EXECUTION_PLAN_15-21Nov.md)) | The bias–variance account plus a multi-dataset replication and a rigor contribution | A null interaction with no explanation is a weak paper. N1 explains it at zero GPU cost. Two or more datasets are described as mandatory (the source of that requirement is not written down: confirm with the course) |

### 1.3 Contributions, as the report should state them (draft)
1. A pre-registered **factorial** measurement of the depth × dropout interaction: LLFF, plus DTU and/or Blender, three seeds. It comes with dose, view-count, pass-localization and depth-weight sweeps.
2. **An error-budget explanation** (bias² vs variance) of why the two regularizers stack, with predictions BV1–BV6 tested.
3. **Evidence standards for the field:**
   - a measured noise floor;
   - a power calculator;
   - an audit of published ablation gains against that noise.
4. **Mechanism and theory of compensated dropout in alpha compositing.** Proposition 1 shows the clamp bias. Its tests came out as not supported (§5). The clamp-aware and no-compensation variants are pending.
5. A reproducible harness:
   - tags and md5-locked data;
   - CI-tested analysis;
   - released run outputs.

---

## 2. Background: DNGaussian, DropGaussian and our variants

### 2.1 3DGS in one paragraph
A scene is a set of 3-D Gaussians. Each has a position, a shape (scale and rotation), an opacity and a colour. To render, they are projected to the image, sorted front to back and alpha-blended. Training fits all parameters to the training photos by gradient descent. With only 3 photos, many wrong scenes explain the photos equally well, so the model overfits: floaters, wrong geometry and poor novel views. Regularizers try to prevent this.

### 2.2 DNGaussian (CVPR 2024), our host

Each training iteration in [train_llff.py](../train_llff.py) runs **three separate rasterizations**. Each has its own backward pass and optimizer step:

| Pass | Function | What is free | Loss |
| --- | --- | --- | --- |
| 1. Hard depth | `render_for_depth` | Opacity fixed at 0.95; gradients reach **positions only** | Patch-normalized depth loss against DPT monocular depth |
| 2. Soft depth (from iteration 1000) | `render_for_opa` | Positions, scales and rotations detached; gradients reach **opacity only** | The same depth loss |
| 3. Photometric | `render` | Opacity and colour from the neural field (`gridencoder`/`shencoder`) | L1 + D-SSIM + shape/scale/opacity penalties |

- The depth losses never use the photometric render.
- `--depth_scale 0` skips passes 1 and 2 entirely.
- The depth loss normalizes each patch by its own standard deviation, because monocular depth has an unknown scale and shift.
- The canonical DPT depth is `llff_with_depth.tar`, md5 `0f6bd7bdae837ec58ad757c7c00b0561`. **Never regenerate it.**

### 2.3 DropGaussian (CVPR 2025) and our exact port

The official code, kept verbatim (AGENTS invariant 7):
```python
compensation = torch.nn.Dropout(p=r)(torch.ones(N))   # zeros w.p. r, survivors scaled by 1/(1-r)
opacity = opacity * compensation[:, None]
r = drop_rate * iteration / 10000                      # 10000 = DropGaussian's total iterations
```
- **Decision D1** (13 Oct, confirmed 14 Oct): keep `/10000` even though we train for 6000 iterations. With `--drop_rate 0.2` the rate rises linearly and **peaks at 0.12**. Block G covered γ = 1/3, which peaks at 0.20, DropGaussian's published end rate.
- One mask per iteration. At `drop_rate 0` no mask is created (invariant 2).
- Dropout acts **only in training**: never in `render.py`, `metrics.py`, `spiral.py` or evaluation (invariant 1).
- `opa_pena` uses the pre-dropout opacity (decision D7).
- DropGaussian itself starts from a dense COLMAP cloud. We start from random points, so variant C is **not** expected to reach their 20.76 dB.

### 2.4 The theory (RESEARCH_ANALYSIS §6)

**Proposition 1.** Compensated dropout leaves the expected render unchanged: E[α̃] = α, and the render is a product of independent terms. **Unless the rasterizer's 0.99 alpha cap binds.**

**The clamp bias.** In the hard-depth pass every Gaussian has opacity 0.95, so α/(1−r) > 0.99 once r > 0.0404. The expected alpha then falls to (1−r)·0.99:

| r | 0.06 | 0.12 | 0.20 | 0.30 |
| --- | --- | --- | --- | --- |
| Expected centre alpha (clean 0.95) | 0.931 | 0.871 (−8 %) | 0.792 (−17 %) | 0.693 (−27 %) |

Only passes that receive the mask are affected: D's hard pass, and D_hard.

**Proposition 2.** For squared error, E‖y − R̃‖² = ‖y − E R̃‖² + Σ Var(R̃). Dropout therefore trains the clean loss plus a penalty on variance across masks (Wager et al., 2013). This is where the "dropout reduces variance" idea of N1 comes from.

**Clamp-aware dropout (#27).**
- Each Gaussian gets its own rate min(r, 1 − opacity/0.99), so compensation never exceeds the cap.
- Its smoke test found the median opacity in this host is about 0.98–0.998. So clamp-aware dropout cuts the effective dose to about 43–50 % of nominal (context log, 4 Nov).
- H7 therefore asks whether DropGaussian's benefit *needs* the biased full-dose dropout of opaque Gaussians.

### 2.5 Variants

All variants are the same code with different flags (AGENTS.md).

| ID | `--depth_scale` | `--drop_rate` (γ) | `--depth_render` | Meaning |
| --- | --- | --- | --- | --- |
| A | 0 | 0 | shared | No depth, no dropout |
| B | 1 | 0 | shared | DNGaussian as published (the fixed reference) |
| C | 0 | 0.2 | shared | Dropout only |
| D | 1 | 0.2 | shared | Naive combination: the mask also hits both depth passes |
| E | 1 | 0.2 | clean | Decoupled: the depth passes never see the mask |
| D_hard | 1 | 0.2 | hard_only | Only the hard-depth pass sees the mask (block P) |
| D_soft | 1 | 0.2 | soft_only | Only the soft-depth pass sees the mask (block P) |
| C/D/E_g010, _g033 | as C/D/E | 0.1 / 1/3 | as C/D/E | Dose (block G) |
| B_r1–4, D_r1–4 | as B/D | as B/D | shared | Same-seed repeats (block N) |
| *planned:* C_ca, D_ca | as C/D | 0.2 | shared | `--drop_mode clamp_aware` (#27) |
| *planned:* C_nc, D_nc | as C/D | 0.2 | shared | No compensation. **Open: in all passes (task A1) or only the depth passes (RESEARCH_ANALYSIS C4)?** Fix it in prereg-v3 |
| *planned:* B/D_ds050, B/D_ds200 | 0.5 / 2 | 0 / 0.2 | shared | Depth-dose surface (N5) |
| *planned:* A/B/C/D_r1–4 on 3 drawn scenes | — | — | — | noise2 (G6) |

**Contrasts.** With X − Y meaning the per-scene difference:
- **I_D** = (D − C) − (B − A): does depth help more or less when dropout is on? 0 = additive, < 0 = sub-additive.
- **I_E** = (E − C) − (B − A): the same for the decoupled variant.
- **E − D**: the effect of keeping the mask away from the depth passes. This is the cleanest causal contrast: one switch, the same number of renders.

---

## 3. Hypotheses and predictions

| ID | Source | Prediction | Decision rule | Status |
| --- | --- | --- | --- | --- |
| Verdict | PLAN_V2 §5 | Complementary, redundant or interfering | Two-axis table: does X beat B and C (CIs > 0) × interaction class (δ = 0.25 dB) | **D = complementary, E = complementary** ([verdict_3v.txt](../results/tables/verdict_3v.txt)) |
| H1 depth-signal corruption | PLAN_V2 §2.3 | gap(D) > gap(C); E − D > 0; E − D grows with dose | All three: E − D CI > 0 (Holm); gap(D) − gap(C) CI > 0; E − D larger at γ 1/3 than 0.1 in ≥ 6/8 scenes | **Not supported**: conditions 1–2 fail, 3 holds (7/8) |
| H2 opacity cap | PLAN_V2 §2.3 | D_hard − E < 0; D_soft − E ≈ 0; probe shows a far shift that grows with r | All three | **Not supported**: D_hard − E inconclusive; the probe direction is right but the shift is tiny (+0.05 %) |
| H3 neighbour compensation | PLAN_V2 §2.3 | B is more redundant than A; I_E < 0 | Exploratory | **Premise contradicted**: B loses *more* PSNR than A when Gaussians are removed; kNN distances not analysed |
| H4 variance penalty | PLAN_V2 §2.3 | `rgb_var_masks` lower in C/D/E than A/B | Exploratory | **Not analysed** (the data is logged). N1 now tests the related claim at the level of test error |
| H6 diminishing returns | PLAN_V2 §2.3 | Effects shrink from 2 to 9 views | Trend (seed 0) | **Mixed**: B falls below A at 9 views; I_D at 6 views is negative (unadjusted CI excludes 0). Seeds 1–2 (W4) decide |
| H7 clamp-aware | #27, PLAN_V2 §5.1 (prereg-v2, unmerged) | C_ca vs C, D_ca vs D, read against the dose curve | §5.1 of #27 | **Not run** |
| BV1 | prereg-v3 (draft in task D1) | Depth's gain (B − A) is mostly bias: Δbias² share > 50 % | Point estimate + CI | Pending |
| BV2 | prereg-v3 | Dropout's gain (C − A) is mostly variance: Δvar share > 50 % | same | Pending |
| BV3 ("ceiling") | prereg-v3 | MSE(D) − b̂²(B): CI below 0 ⇒ dropout also reduces bias; otherwise dropout's possible gain on top of depth is bounded by B's variance | Paired CI over scenes | Pending |
| BV4 | prereg-v3 | b̂²(B) − b̂²(A) rises with views, > 0 at 9 views (depth adds bias at 9 views: the crossover) | Sign at 9 views + trend | Pending; needs W4 |
| BV5 | prereg-v3 | I_MSE = I_bias + I_var | Report both | Pending |
| BV6 | prereg-v3 | BV1/BV2 hold on every dataset | Agreement | Pending; needs DTU/Blender |
| X4 replication | prereg-v3 | DTU/Blender reproduce the LLFF contrasts | Same sign and the CI overlaps the LLFF estimate; δ and Holm as on LLFF | Pending |
| X5 | prereg-v3 | D − B (DTU) > D − B (LLFF) | One-sided | Pending |
| H-PD prior domain shift | prereg-v3 | Rank of D − B: LLFF ≤ DTU ≤ Blender | Exploratory, 3 points | Pending |
| N4 | prereg-v3 | The interaction keeps its sign on PSNR, MSE, RMSE and SSIM | Same sign on all = "robust", else "scale-dependent" | Pending. Already visible on T1: SSIM interactions are small and negative with CIs excluding 0 ([paired_3v.md](../results/tables/paired_3v.md)) |
| N5 | prereg-v3 | \|I_D\| < δ at depth_scale 0.5, 1 and 2 | Each | Pending |
| G6 / N2(a) | prereg-v3 | Same-seed SD ≈ between-seed SD on 3 drawn scenes; noise differs by class (A, C > B, D) | — | Pending; fern already shows it ([noise_floor_3v.csv](../results/tables/noise_floor_3v.csv)) |
| Compensation | prereg-v3 | C_nc − C, D_nc − D, C_ca − C, D_ca − D | Holm across 4; TOST δ = 0.25 | Pending |

**Integrity rule (new plan, rule 7).** Nobody opens a render or computes a new statistic for N1/N3/N5/G6/DTU/Blender before `prereg-v3` is tagged. The MSE-scale and heterogeneity checks of 15 Nov are exploratory and must be labelled so. Those 15 Nov numbers are not in the repo (audit I-10).

---

## 4. Experimental design

### 4.1 Datasets

| Dataset | Scenes | Train views | Eval views | Resolution | Metrics | State |
| --- | --- | --- | --- | --- | --- | --- |
| LLFF | fern, flower, fortress, horns, leaves, orchids, room, trex | 3 (also 2/6/9), evenly spaced over the non-held-out images | every 8th image | `-r 8` | PSNR, SSIM (3DGS; `SSIM_sk` in the appendix), LPIPS (VGG) | Done: data in `llff_with_depth.tar` |
| DTU | 15 scans | indices 25, 22, 28 ([dataset_readers.py:199-205](../scene/dataset_readers.py)) | 25 (49 − 9 train candidates − 15 excluded) | `-r 4` | **masked** PSNR/SSIM/LPIPS (RegNeRF idrmasks) | Not started. Index-based split: **all 49 images must register in COLMAP**, else the split shifts silently |
| Blender (nerf_synthetic) | chair, drums, ficus, hotdog, lego, materials, mic, ship | 8: `[2, 16, 26, 55, 73, 76, 86, 93]` ([dataset_readers.py:337](../scene/dataset_readers.py)) | 25 (every 8th of 200 test views) | `-r 2`, white background | PSNR/SSIM/LPIPS | Not started. One uniform protocol (run_blender.sh setting 1 with soft depth on); non-SH path only |

**Depth.** DPT runs **once** per dataset. The output is frozen as `<dataset>_with_depth.tar` with an md5, and nobody reruns it (new plan, rule 3).

### 4.2 Protocol and invariants (AGENTS.md)
1. Dropout only in the training loop.
2. Flag defaults reproduce upstream (B). At `drop_rate 0` no mask is created.
3. **Frozen output layout:** `<out_root>/<dataset>_<n>v/<variant>/<scene>/seed<k>/` holding `config.json, diag.csv, metrics.json, train_stats.json, render_stats.json, results_eval.json, eval/`. Keys and paths may be added, never renamed. `dtu_3v` and `blender_8v` are additions.
4. One GPU type per seed. Every result so far comes from the lab RTX A4000. Parallel slots set `shared_gpu: true`, and efficiency numbers use `shared_gpu: false` runs only.
5. Experiments run only from a tag:
   - `v1.0-exp`: LLFF blocks and views seeds 1–2;
   - `v1.1-exp`: compensation, after #27 and `nc` merge;
   - `v1.2-exp`: DTU, Blender, depth-dose and noise2, after the multi-dataset port.
6. Same data everywhere, checked by md5.
7. The DropGaussian port stays verbatim (decision D1).

**Python 3.7 syntax only** in anything training imports. Never call `torch.load` directly; use `utils.general_utils.load_checkpoint()`.

### 4.3 Matrix files

| Block | Matrix | Runs | Tag | Out root | State |
| --- | --- | --- | --- | --- | --- |
| T1 | `matrix.yaml` | 120 | v1.0-exp | `outputs/` | done |
| P | `matrix_pass.yaml` | 48 | v1.0-exp | `outputs_pass/` | done |
| G | `matrix_dose.yaml` | 48 | v1.0-exp | `outputs_dose/` | done |
| N | `matrix_noise.yaml` | 8 | v1.0-exp | `outputs_noise/` | done |
| V seed 0 | `matrix_views.yaml` | 120 | v1.0-exp | `outputs_views/` | done |
| V seeds 1–2 | `matrix_views.yaml --seeds 1 2` | 240 | v1.0-exp | `outputs_views` (new worktree) | unknown (lab) |
| Compensation | `matrix_comp.yaml` (to write; #27 has `matrix_ca.yaml` with 48) | 96 | v1.1-exp | `outputs_comp` | to build |
| DTU | `matrix_dtu.yaml` | 225 | v1.2-exp | `outputs_dtu` | to build |
| Blender | `matrix_blender.yaml` | 120 | v1.2-exp | `outputs_blender` | to build |
| Depth-dose | `matrix_depthdose.yaml` | 96 | v1.2-exp | `outputs_depthdose` | to build |
| noise2 | `matrix_noise2.yaml` (3 scenes from the pre-registered draw) | 48 | v1.2-exp | `outputs_noise2` | to build, after prereg-v3 |

Checkpoints: keep them only for seed 0 of a block that needs probes (`keep_ckpt_seeds`). The runner deletes TensorBoard events, `train/` and `test/`, and keeps `eval/` ([run_matrix.py:62-71](../experiments/run_matrix.py)).

### 4.4 Tags

| Tag | Commit | What it marks | State |
| --- | --- | --- | --- |
| `upstream-baseline` | 81c3d83 | Imported DNGaussian | exists |
| `v1.0-exp` | 1040559 | LLFF experiment freeze (#19) | exists (3 Nov) |
| `prereg-v1` | 1040559 | PLAN_V2 §5 rules + stats.py v2 | exists (3 Nov) |
| `prereg-v2` | — | #27's §5.1 addendum (H7) | **missing** |
| `v1.1-exp` | — | #27 + `nc` | missing |
| `prereg-v3` | — | PLAN_V2 §5.2: datasets, N1–N5, G6, audit, compensation | **missing; was due Sun 23:59** |
| `v1.2-exp` | — | Multi-dataset port | missing; due Tue 23:59 |
| `v1.0-report` | — | The submitted report | missing |

---

## 5. What has been done: results so far

All numbers are LLFF, from 344 lab runs at `v1.0-exp`. Source: [results/RESULTS.md](../results/RESULTS.md) and the tables it names.

### 5.1 Main table, 3 views (8 scenes × 3 seeds; [tables/main_3v.md](../results/tables/main_3v.md))

| Variant | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| A | 16.33 ± 0.23 | 0.460 ± 0.012 | 0.394 ± 0.010 |
| B | 18.71 ± 0.04 | 0.592 ± 0.001 | 0.302 ± 0.001 |
| C | 16.80 ± 0.15 | 0.497 ± 0.003 | 0.378 ± 0.001 |
| D | 19.21 ± 0.06 | 0.611 ± 0.001 | 0.303 ± 0.002 |
| E | 19.31 ± 0.09 | 0.617 ± 0.003 | 0.303 ± 0.000 |

B (18.71) is 0.41 dB below the paper's 19.12. That sits within the reproduction tolerance: the T4 reproduction got 18.93 and the V3 regression check passed at 19.66 vs 19.85 (PLAN.md §10; #19).

### 5.2 Contrasts and verdict ([tables/paired_3v.md](../results/tables/paired_3v.md), [verdict_3v.txt](../results/tables/verdict_3v.txt))

| Contrast | ΔPSNR [95 % CI] | Class |
| --- | --- | --- |
| E − D | +0.10 [−0.10, +0.30] | inconclusive |
| I_D | +0.02 [−0.34, +0.38] | inconclusive |
| I_E | +0.13 [−0.17, +0.42] | inconclusive |
| B − A | +2.38 [+1.80, +2.96] | > 0 |
| C − A | +0.48 [+0.18, +0.77] | > 0 |
| D − B | +0.50 [+0.08, +0.91] | > 0 |
| E − B | +0.60 [+0.30, +0.91] | > 0 |
| D_hard − E | −0.12 [−0.32, +0.08] | inconclusive |
| D_soft − E | −0.10 [−0.27, +0.07] | inconclusive |

**Verdict: D = complementary, E = complementary.** Both beat B and C. No interaction is detectable in PSNR.

In SSIM the interactions are small but negative, with CIs that exclude 0: I_D −0.018 [−0.032, −0.003] and I_E −0.012 [−0.022, −0.002]. So the effects add up slightly less than additively in SSIM. This is what N4 and N3 will examine.

**What it means.**
- Dropout adds about +0.5 dB on top of depth. That is about the same as it adds without depth (+0.48), hence "≈ additive".
- Decoupling the mask (E vs D) makes no detectable difference.
- Neither proposed mechanism of interference is supported.

### 5.3 Dose, noise, views, efficiency, geometry, probes

**Dose (block G, seed 0; [dose_3v.csv](../results/tables/dose_3v.csv)).**

| γ | E − D | I_D | I_E |
| --- | --- | --- | --- |
| 0.1 | −0.16 | +0.28 | +0.12 |
| 0.2 | +0.12 | +0.45 | +0.57 |
| 1/3 | +0.20 | −0.08 | +0.12 |

E − D rises with dose in 7/8 scenes. This is a trend only (one seed).

**Noise floor (fern; [noise_floor_3v.csv](../results/tables/noise_floor_3v.csv)).**
- SD over 5 same-seed runs: B 0.220 dB, D 0.216 dB.
- Pooled between-seed SD in T1: 0.213 and 0.242 dB.
- So run-to-run variation is rasterizer nondeterminism, not the seed. Any single-run difference under about 0.4–0.5 dB is noise.

**Views (seed 0; [tables/views/](../results/tables/views/)).** Mean PSNR over 8 scenes:

| Views | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| 2 | 14.32 | 15.90 | 14.90 | 16.34 | 16.48 |
| 3 | 16.59 | 18.74 | 16.67 | 19.27 | 19.39 |
| 6 | 20.89 | 22.06 | 21.77 | 21.97 | 22.03 |
| 9 | 22.94 | 22.59 | 22.84 | 22.58 | 22.61 |

- I_D at 6 views: −0.96 [−1.68, −0.24], unadjusted.
- At 9 views depth hurts: B is 0.35 dB below A. This is the "crossover" BV4 predicts.

**Efficiency (quiet A4000; [efficiency_3v.csv](../results/tables/efficiency_3v.csv)).**
- Training time: A/C 84 s, B 145 s, D/E about 142 s.
- FPS: 622–689.
- Peak VRAM: 570–618 MB.
- Depth roughly doubles training time. Dropout costs nothing.

**Geometry, MVC@5% (δ = 0.02; [paired_3v.md](../results/tables/paired_3v.md), [tables/geometry/](../results/tables/geometry/)).**
- Means: A 0.284, B 0.509, C 0.315, D 0.552, E 0.531.
- E − D = −0.021 [−0.037, −0.005], Holm p 0.049, class "< 0": the shared mask gives slightly *more* consistent geometry. This is the opposite sign to PSNR.

**Probes (T1 seed 0; [probes_robustness_3v.csv](../results/tables/probes_robustness_3v.csv), [probes_depth_bias_3v.csv](../results/tables/probes_depth_bias_3v.csv)).**
- Robustness: models trained with dropout lose about half as much PSNR when 20 % of Gaussians are removed (C −0.32 vs A −0.79 dB, compensated), so dropout makes the scene representation more redundant. B loses more than A (−1.03).
- Hard-depth bias: in the predicted direction, but tiny (+0.05 % of depth at r = 0.2).

### 5.4 Threats to validity (keep these in the report)
- **Power.** With 8 scenes × 3 seeds the minimum detectable interaction is about 0.35–0.5 dB (RESEARCH_ANALYSIS §5.3). "Inconclusive" must not be written as "no interaction". DTU's 15 scans roughly halve the MDE. This is the statistical case for W1.
- **One host, random initialization.** Published dropout gains used dense (MVS) initialization. StereoGS shows an interaction can flip sign with initialization.
- **Only one depth prior (DPT)**, and only one dropout family.
- **Single-seed sweeps.** Dose, views (until W4) and probes rest on seed 0 only.
- **Nondeterminism.** The same seed gives a different result, so compare means over seeds and scenes only.
- **Post hoc elements.** The SSIM interaction note, the 15 Nov MSE checks and every new analysis come after the LLFF data were seen. prereg-v3 must say so.
- **Metric implementation.** The main table uses 3DGS SSIM; DNGaussian's paper used scikit-image SSIM.
- **Optimization variance is not data variance.** N1's "variance" is variance over training runs on fixed views, not over different view sets.

---

## 6. What remains: the plan to Sat 21 Nov

### 6.1 Live status (edit this table at every 21:30 sync)

> **Update, Fri 20 Nov.** Every item below is done:
> - `prereg-v3` tagged (17 Nov);
> - `v1.1-exp` (#27) and `v1.2-exp` (#36) tagged;
> - all 825 new lab runs complete with 0 failures (19 Nov);
> - DTU go/no-go: GO;
> - Results v2 merged (#37).
>
> [RUNLOG.md](../RUNLOG.md) has the dated record and [results/v2/RESULTS_v2.md](../results/v2/RESULTS_v2.md) the numbers. What remains is the Fri 20:00 analysis freeze, the report, and submission on Sat 20:00.

| Item | Owner | Due | Status (16 Nov 21:45) |
| --- | --- | --- | --- |
| prereg-v3 PR + tag (task D1) | Disha | ~~Sun 23:59~~ **tonight** | missed; not started on GitHub |
| #27 rebase + `nc` (task A1) | Aayush | ~~Sun~~ **tonight / Tue am** | not done |
| DTU go/no-go (task A6) | Aayush | Mon 20:00 | **unrecorded: write the outcome in RUNLOG tonight** |
| Views seeds 1–2 (task A3) | Aayush | Mon | unknown |
| `eval_T1.tgz` (task A4) | Aayush | Mon eve | unknown |
| Blender data + tar (task D2) | Disha | Tue am | unknown |
| `bias_variance.py` on fixtures (task D3) | Disha | Tue | not started |
| Multi-dataset port, `v1.2-exp` (task A7) | Aayush | Tue 23:59 | not started |
| #27 → `v1.1-exp` (task A8) | Aayush | Tue | not started |

### 6.2 Day by day, adjusted for tonight's slips

This is the new plan's timeline (§3 there) with the audit's findings folded in. Changes are marked **[adj]** and are proposals for the next sync, not decisions.

**Mon 16 (tonight)**
- **[adj] Disha:** finish and tag `prereg-v3`; it gates everything else.
  - Its exploratory disclosure must state that the LLFF data have been seen.
  - It must fix the `nc` definition (audit I-9).
  - It must write the noise2 scene-draw rule *before* the draw.
  - It needs a merge-order decision with #27 (I-6): either merge #27 with `prereg-v2` first, or number the addendum independently.
- **[adj] Aayush:** write the DTU go/no-go result in RUNLOG. If it didn't run, run task A6's test first thing Tue, and treat Blender as required until DTU passes.
- Aayush: start task A1 (rebase #27, `nc` mode, `matrix_comp.yaml`).

**Tue 17**
- Aayush, task A7 multi-dataset port. **[adj]** Add to its scope:
  - gap G-a: make the Blender reader write its random cloud to a temp dir, as W7 did for LLFF/DTU, plus a check.sh test;
  - AGENTS invariant 3 and the environment table (I-2, I-5); both are gate class, so Disha approves.
- Aayush, task A8: GPU smoke → review → Disha → merge #27 → `v1.1-exp` (+ `prereg-v2`). Start the compensation block.
- Disha:
  - task D3 on fixtures, then on T1 renders after the tag;
  - task D4 (dataset factor). **[adj]** Include gap G-b: extend `factors.py` for `_nc`, `_ca`, `_ds050`, `_ds200` and `A/C_r*` (#28 covers `_ca`).
  - task D5 (N4). Same-day reviews of A7/A8.
- Lab: views s1–2 finish; compensation (96); DTU (225) from the evening.
- **Gate: `v1.2-exp` by 23:59.** If it slips, the plan's decision table applies: DTU-only PR, Blender follows as `v1.3-exp` on Wed.

**Wed 18**
- Aayush: task A9 run queue, in this order:
  1. DTU;
  2. Blender (seeds 0–1 only if DTU has < 150/225 by 12:00);
  3. depth-dose;
  4. noise2.

  Also task A10 (his audit half) and task A12 (Method/Setup). **[adj]** A12 must rewrite #23's Setup: everything ran on the A4000, and V has 120 runs + T1.
- Disha: task D6 (N1 on views, BV4), task D7 (power, noise bands), task D8 (her audit half), DTU tables as runs land.

**Thu 19 (lab buffer day; changed 16 Nov)**
- Lab visit: start any block that missed Wednesday (for example, Blender if `v1.2-exp` was late) plus reruns, and let it run overnight.
- 12:00 DTU throughput check: below about 150/225 means Blender runs seeds 0–1 only.
- Aayush: task A10 κ subset; task A12 Method/Setup drafts; **[adj]** the F1 iteration diagram (from PLAN_V2 AY-24, still undone).
- Disha: task D6 on views and on DTU as it lands; tasks D7/D8 finish; figure scripts run on the data landed so far.

**Fri 20**
- Lab visit: final counts, last reruns. **12:00 all runs stop.**
- Aayush: task A11 sync to Drive with md5s; RUNLOG counts per block; task A12 Limitations and reproducibility appendix.
- Disha: task D6 (BV6), task D9 (pooled), task D10 (figures), task D11 (N3 only if time).
- **20:00 analysis freeze.** Report writing starts Fri night: task D12 Results.

**Sat 21**
- Aayush: task A12 final; task A13 number cross-check.
- Disha: task D12 final; task D13 README + site. **[adj]** Gap G-d: the README is still upstream's, so it needs a project README.
- Both: final read; submit 20:00; tag `v1.0-report`; RUNLOG final.

Moving the run stop to Friday leaves only about 24 hours for the report. So write everything that doesn't depend on new numbers Tue–Thu, before the freeze:
- Method, Setup, Related Work, Introduction;
- the LLFF results;
- figure scripts tested on fixtures.

### 6.3 Critical path and fallbacks
`prereg-v3` → (task A7 + task A8) → `v1.2-exp` / `v1.1-exp` → lab queue (about 18 GPU-hours with 3 slots) → Fri 12:00 stop → D6/D9/D10 → Fri 20:00 freeze → writing.

**Lab visits (one a day; the PC runs unattended overnight in `tmux`, confirmed by Aayush 16 Nov):**

| Visit | Start (chained, so the GPU never idles) | Needs before the visit |
| --- | --- | --- |
| Tue | Views seeds 1–2, then comp → depth-dose → noise2 | `v1.1-exp` tagged (otherwise views only) |
| Wed | DTU → Blender, plus Tuesday's leftovers | `v1.2-exp` tagged; DTU and Blender tars on the lab PC |
| Thu | Late blocks and reruns (buffer) | — |
| Fri (before 12:00) | Last reruns, then stop; copy outputs to Drive | — |

**Lab checklist:**
- Before leaving: dry-run every block, then watch the first 2–3 runs finish.
- Keep the PC from sleeping and rebooting; lock the screen, but don't log out.
- New worktrees have no `data/`. Pass `--data_root /home/snu/dngs-dropout/data/<dataset>`, or link `data` into each worktree.

**Decision table (new plan §7; its run-stop deadline is now Fri 12:00):**
- DTU no-go → Blender is required.
- A Blender port problem → DTU-only PR, Blender follows in `v1.3-exp`.
- Both fail → GT-camera fallback for DTU. It is a documented protocol deviation, decided at the call.
- #27 drags on → compensation becomes a probe-only analysis.
- Lab lost for a day → cut in this order: N3, Blender seed 2, depth-dose seeds 1–2, noise2.
- N1 predictions fail → report it. A failed BV2 is itself a headline.

**Never cut:** prereg-v3, the second dataset, N1 on LLFF, the report cross-check.

### 6.4 Smaller open items (cheap; do them when idle)
- Open a routine PR for `experiments/probes.py` from `feat/probes` (gap G-c), so committed results trace to merged code.
- Commit `docs/EXECUTION_PLAN_15-21Nov.md` and `LAB_RUNBOOK_11nov.md` (docs-only).
- Back-fill RUNLOG for 3–16 Nov (audit I-4).
- Decide #17 (team-context protocol) and #24 (tracker): merge or close. Both are gate class, and both have waited since 3 Nov.
- Per-scene moderators (PLAN_V2 DS-14/DS-18) were never built and the new plan doesn't mention them. Declare them dropped, or fold them into the Discussion.
- H3/H4 data are logged but not analysed. This could be a short appendix paragraph, or be folded into N1's variance story.

---

## 7. Analysis and statistics plan

### 7.1 Unit and contrasts (PLAN_V2 §5, `prereg-v1`)
- **Unit:** per scene, average over seeds. Every comparison is paired by scene and reports:
  - the mean of the per-scene differences;
  - a 95 % t-interval (df = scenes − 1);
  - wins out of N.
- **Primary family (PSNR, Holm):** E − D, I_D, I_E.
- **Margin:** δ = 0.25 dB (0.02 for MVC@5%). "Equivalent" means the CI lies inside ±δ. A CI that crosses 0 and extends beyond ±δ is "inconclusive".
- **Reference:** B is fixed; best(B, C) is never chosen from the data.
- **Implementation:** [experiments/stats.py](../experiments/stats.py) v2, with the variant factor table in [experiments/factors.py](../experiments/factors.py).

### 7.2 N1, the error budget (task D3, `experiments/bias_variance.py`)

For each scene and variant, use n seeds of renders r_k and the ground truth y, for every pixel and channel (DTU: masked pixels only):
- m̄ = mean_k r_k
- s² = var_k(r_k, ddof = 1)
- mse = mean_k (r_k − y)²
- b̂² = (m̄ − y)² − s²/n
- var = s²
- mse_ens = (m̄ − y)²

These are averaged over pixels, channels and views. The identity **mse = b̂² + var** holds exactly, because (m̄ − y)² + (n−1)/n·s² = b̂² + s², and the script asserts it to 1e-6.

- Contrasts (Δmse, Δb̂², Δvar) for B−A, C−A, D−B, E−B, I_D, I_E, plus BV3's MSE(D) − b̂²(B), each with a paired t-interval over scenes, Holm within the BV family.
- **Self-test:** synthetic bias and noise must be recovered within 5 %.

**Caveats to state:**
- "Bias" here is the error that remains after averaging training runs, measured against GT. It includes error that no regularizer can remove: unseen regions and view-dependent effects.
- With n = 3, s² is noisy per pixel. The averaged estimate is unbiased, but the per-scene shares are noisy.
- Share ratios (BV1/BV2) are unstable when the denominator is small. C − A is only about 0.5 dB. Consider reporting a scene-bootstrap interval for the share (a suggestion, not in the plan).
- PSNR and MSE are not linear in each other. Use per-run MSE for N4 (task D5), and never convert mean PSNR into MSE.

### 7.3 N4, scale robustness (task D5)
- Compute I_D, I_E, E − D and D − B on PSNR, MSE, RMSE and SSIM.
- If the sign is the same on all four, the result is "robust". Otherwise it is "scale-dependent" (Loftus, 1978: an interaction that flips under a monotone transform is removable).

### 7.4 N2, noise and power (task D7)
- **Run noise:** residual variance σ²_v per variant, pooled within scenes, from T1 + N + noise2.
- **Contrast heterogeneity:** τ²_c = var(scene means of c) − Σ coef²·σ²_v / R.
- **Power:** SE_c = √(τ²_c/S + Σ coef²σ²_v/(S·R)); MDE = (t₀.₉₇₅,S−1 + t₀.₈,S−1)·SE_c. Tabulate for S ∈ {8, 15, 23, 31} and R ∈ {1, 3, 5}.
- **Validation:** at S = 8, R = 3 the predicted CI half-width for I_D must come out at about ±0.36 dB. The observed CI is [−0.34, +0.38]. If it doesn't match, say the model is wrong.
- **Audit band:** a 95 % band for a single-run, 8-scene-mean difference, for each noise class.

### 7.5 N2(b), the evidence audit (task D8 + task A10)
- **Inclusion:** sparse-view 3DGS papers from 2024–26 with an LLFF 3-view ablation table; take 8–10.
- **Coding sheet:** `results/audit/ablation_audit.csv`.
- **Classification:** |Δ| < band → "within single-run noise".
- **Agreement:** Cohen's κ on 2 papers coded by both. If κ < 0.7, report the coding as unreliable.
- **Wording:** conditional, i.e. "if the host's noise resembles DNGaussian's".

### 7.6 Cross-dataset pooling (task D4 + task D9)
- Two-stage random effects: per-scene contrast → per-dataset mean and SE → DerSimonian–Laird pooling. Report the pooled estimate, τ² and Q.
- Robustness check: a mixed model `metric ~ variant × dataset + (1|scene)`.
- Test X4, X5 and H-PD (§3).

### 7.7 N5, N3, compensation
- **N5:** I_D at depth_scale 0.5 / 1 / 2 (A and C come from T1).
- **N3 (optional):** radial-FFT bands. By Parseval the band MSEs sum to the total, and the script asserts it. Prediction: the two regularizers overlap in the mid/high bands and separate in the low band.
- **Compensation:** four contrasts against C and D, Holm across 4, TOST at δ = 0.25.

---

## 8. Report, figures and definition of done

### 8.1 Outline and owners (new plan §8; the template is still to be confirmed, audit I-12)

| § | Section | Owner | Inputs ready? |
| --- | --- | --- | --- |
| 1 | Introduction: no factorial, no error budget, no noise-aware evidence | Both (Aayush drafts) | Positioning text in RESEARCH_ANALYSIS §4 |
| 2 | Related work | Disha | Draft is a Claude Doc (DS-07/13); status unknown |
| 3 | Method: passes, dropout and compensation modes, factorial, error-budget estimator | Aayush (+ Disha's estimator paragraph) | #23 draft (3 Nov) |
| 4 | Setup: 3 datasets, frozen protocol, prereg, hardware, tags | Aayush | #23 draft is **stale** (I-13) |
| 5 | Results | Disha | LLFF parts ready |
| 6 | Discussion | Both | — |
| 7 | Limitations and future work | Aayush | §5.4 here; RESEARCH_ANALYSIS §9 |
| A | Reproducibility appendix, audit sheet, power tables | Aayush / Disha | raw README has the md5s |

### 8.2 Figures (new plan numbering; refer to them by file name in the text, audit I-11)

| Fig | Content | State |
| --- | --- | --- |
| F1 | Error budget, stacked bias²/var per variant, one panel per dataset | to do |
| F2 | Error budget vs views (2/3/6/9) | to do; needs W4 |
| F3 | Forest per dataset + pooled | LLFF part exists (`fig_forest*.png`) |
| F4 | D − B by dataset | to do |
| F5 | BV3 ceiling scatter | to do |
| F6 | Audit: reported Δ vs noise band | to do |
| F7 | Power curves | to do |
| F8 | Depth-dose × dropout | to do |
| F9 | Compensation forest | to do |
| F10 | Qualitative grids per dataset | LLFF exists (`fig_qual.png`, `fig_fail.png`) |
| F11 | Per-band interaction (N3) | optional |
| (PLAN_V2 F1) | D vs E vs D_hard/D_soft iteration diagram | **never made**; useful for Method |
| Existing, keep for appendix | `fig_diag`, `fig_dose`, `fig_pass`, `fig_views*`, `fig_robustness` | done |

### 8.3 Definition of done (new plan §9, Sat 20:00)
- [ ] LLFF + DTU (or Blender) complete with 3 seeds; md5s recorded; `v1.1-exp`/`v1.2-exp` in RUNLOG
- [ ] `prereg-v3` tag predates every new result file
- [ ] N1 tables for every dataset and for views; BV1–BV6 verdicts written
- [ ] N2 power table validated against ±0.36; audit with κ
- [ ] N4 and N5 tables; compensation verdict
- [ ] Figures in `results/figures/`, regenerated from scripts
- [ ] Every report number traced to `results/tables/` (task A13)
- [ ] README + website live; report PDF submitted
- [ ] Viva: both of us can explain every part, including AI-assisted code; AI use disclosed as the course requires

### 8.4 Viva preparation
- PLAN_V2 §10.3 has 17 likely questions with short answers.
- New questions to prepare:
  - "What is bias and what is variance here?"
  - "Why does additivity follow if depth cuts bias and dropout cuts variance?"
  - "Why is the same-seed SD as large as the between-seed SD?"
  - "What does 'inconclusive' allow you to claim?"
  - "Why masked metrics on DTU?"
  - "Why might a depth prior hurt at 9 views?"
- Replace the §10.2 line "dropout as a variance penalty" with what N1 actually finds (task D12).

---

## 9. Runbooks

**Environment (laptop).**
- Python: `~/miniconda3/envs/dngaussian/bin/python`.
- The rebuild variables are in [CLAUDE.md](../CLAUDE.md).
- Before any GPU job: `pgrep -af "train_|render.py|metrics.py"`. Run one job at a time; the GPU has 4 GB.
- Long jobs go to the lab, in `tmux`.

**Training and evaluation (single run).**
```bash
bash scripts/run_llff.sh data/llff/fern output/llff/fern 0             # upstream full run
python render.py -s data/llff/fern --model_path <out> -r 8 --near 10 --skip_train --skip_test
python metrics.py --model_paths <out>                                  # writes results_eval.json
bash scripts/gpu_smoke.sh                                              # gate smoke, 1.5–7 min
bash scripts/check.sh                                                  # static checks, seconds
```

**Matrix (lab), always dry-run first.**
```bash
git worktree add ~/dngs-v1.0 v1.0-exp      # one worktree per tag (task A2)
python experiments/run_matrix.py --matrix experiments/matrix_views.yaml --data_root data/llff \
  --out_root outputs_views --seeds 1 2 --dry_run                                   # expect 240
for i in 0 1 2; do tmux new -d -s views$i "cd ~/dngs-v1.0 && ~/miniconda3/envs/dngaussian/bin/python \
  experiments/run_matrix.py --matrix experiments/matrix_views.yaml --data_root data/llff \
  --out_root outputs_views --seeds 1 2 --shard $i/3 --shared_gpu > output/views_s12_shard$i.log 2>&1"; done
ls outputs_views/llff_*v/*/*/seed[12]/metrics.json | wc -l                          # progress
grep -l Traceback output/*shard*.log                                                 # failures; rerun skips done runs
```
Copy commands as plain text: Markdown renderers have eaten `$` signs and underscores before ([LAB_RUNBOOK_11nov.md](../LAB_RUNBOOK_11nov.md)).

**Aggregate, stats and figures (LLFF, from [RESULTS.md](../results/RESULTS.md)).**
```bash
R=results/raw; T=results/tables; F=results/figures
python experiments/aggregate.py --out_root $R/main --tables $T
python experiments/stats.py --runs $T/runs.csv --tables $T --diag_summary $T/diag_summary_3v.csv \
  --pass_runs $T/pass/runs.csv --dose_runs $T/dose/runs.csv --noise_runs $T/noise/runs.csv \
  --geometry $T/geometry/main/geometry_3v.csv
python experiments/plots/forest.py --tables $T --figures $F
```
The full list, including views, dose, pass, probes and the qualitative grids, is in RESULTS.md "How to regenerate".

**DTU data (task A5), summarized.**
1. `organize_dtu_dataset.sh <rectified>`.
2. COLMAP from a conda env: `convert.py --no_gpu --colmap_executable ...`.
3. Check **49/49 registered** with `colmap model_analyzer`. Retry once with `--camera PINHOLE`. If it still fails, exclude the scan by the pre-registered rule.
4. DPT once: `get_depth_map_for_llff_dtu.py -b DTU`.
5. idrmasks.
6. `tar` + `md5sum`.

**Merging (pr-gate).**
1. Branch.
2. `bash scripts/check.sh`. For the experiment class, also `bash scripts/gpu_smoke.sh` on the clean head.
3. `python scripts/review_bundle.py <N>`, then the `pr-reviewer` agent: sonnet for routine, opus for experiment and gate. Docs-only PRs need no reviewer.
4. Fix blocking findings, with a failing test first.
5. Set the `agent-review` status on the reviewed SHA.
6. `gh pr checks <N> --repo aayxsh23/dngs-dropout --watch`.
7. `python3 scripts/pr_ready.py <N>`.
8. `gh pr merge <N> --repo aayxsh23/dngs-dropout --squash --delete-branch --match-head-commit <sha>`. Write the arguments literally: the hook refuses `$vars`.

Rules:
- The experiment class needs the other teammate's approval, a new tag and a RUNLOG line.
- Agents never approve.
- No "Generated with Claude Code" footer anywhere.

**Tagging.**
```bash
git tag -a v1.2-exp -m "multi-dataset port: DTU, Blender non-SH, runner dataset key"
git push origin v1.2-exp
```
Add a RUNLOG line in the same hour.

---

## 10. Glossary, decision log, file index

### 10.1 Glossary
| Term | Meaning |
| --- | --- |
| Hard / soft depth pass | DNGaussian's position-only depth render (opacity 0.95) / opacity-only depth render |
| γ (`--drop_rate`) | Dose; the rate at iteration t is γ·t/10000 (peak 0.12 at γ = 0.2) |
| Compensation | Scaling surviving opacities by 1/(1−r) (uniform); per-Gaussian capped (clamp-aware, `ca`); none (`nc`) |
| Clamp / alpha cap | The rasterizer limits per-pixel alpha to 0.99; it biases compensated dropout when α/(1−r) > 0.99 |
| I_D, I_E | Interaction contrasts (D − C) − (B − A) and (E − C) − (B − A) |
| δ | Smallest effect of interest: 0.25 dB PSNR, 0.02 MVC |
| Inconclusive | CI contains 0 but extends beyond ±δ: neither an effect nor equivalence shown |
| Holm | Step-down multiplicity correction within a pre-declared family |
| TOST | Two one-sided tests: shows equivalence within ±δ |
| MDE | Minimum detectable effect at α = 0.05, 80 % power |
| MVC@5% | Multi-view depth consistency: fraction of valid reprojected pixels with relative depth error < 5 % |
| Error budget, b̂², var | §7.2 |
| Noise floor | SD of PSNR across repeated runs; same-seed ≈ between-seed here |
| `shared_gpu` | Several runs on one GPU; such runs are excluded from timing numbers |
| prereg | A git tag that freezes analysis rules before the data they apply to exist |
| X4, X5, H-PD, BV1–6, N1–N5, G6 | §3 |

### 10.2 Decision log
| Date | Decision | Where recorded |
| --- | --- | --- |
| 12 Oct | Repo public (D10); fork of DNGaussian; `upstream-baseline` tag | PLAN §1 |
| 13 Oct | D1–D11 recorded; ruleset on `main` | RUNLOG |
| 13–14 Oct | **D1:** schedule γ·t/10000 verbatim, peak 0.12 (Disha confirmed 14 Oct) | RUNLOG, AGENTS invariant 7 |
| 14 Oct | D11 → γ = 1/3 plus C at γ = 1/3; deadline moved to 15 Nov | RUNLOG |
| 29 Oct | PLAN_V2: novelty items 1–5; decision sheet settled (§2.6) | PLAN_V2 |
| 30 Oct | Kaggle usable (DS-04) | Disha's context log |
| 3 Nov | `v1.0-exp` + `prereg-v1` on 1040559; #20's five analysis choices confirmed; tooling freeze; clamp-aware coded but gated | tag messages; context logs |
| ~4–6 Nov | All blocks run on the lab RTX A4000 instead of Colab/A100; V in its 120-run fallback form (the date is inference; no record) | `results/raw/README.md` only |
| 11 Nov | Probes and geometry run at the lab; results v1 merged (#29–#31) | RESULTS.md, raw README |
| 15 Nov | New plan: error-budget framing; DTU required, Blender added; dropped: variant F, the separate co-adaptation score, DTU compensation arms, DAv2, the init moderator, region-stratified error, a second host; submission Sat 21 Nov | EXECUTION_PLAN §0 (the deadline change is not in RUNLOG) |
| 16 Nov | Run stop moved Thu 12:00 → **Fri 12:00**; analysis freeze Thu 23:59 → **Fri 20:00**; DTU throughput check Wed 12:00 → Thu 12:00; report written Fri night + Sat (Aayush) | EXECUTION_PLAN §0 note; this guide §6.2 |
| 16 Nov | Lab PC may run unattended overnight in `tmux`; one visit per day (Aayush) | this guide §6.3 |
| 16 Nov | (pending) DTU go/no-go | — |

### 10.3 File index
| Path | What |
| --- | --- |
| [train_llff.py](../train_llff.py), [train_dtu.py](../train_dtu.py), [train_blender.py](../train_blender.py) | Training loops (only LLFF has our flags so far) |
| [gaussian_renderer/__init__.py](../gaussian_renderer/__init__.py) | `render`, `render_for_depth`, `render_for_opa` with the optional compensation vector |
| [arguments/__init__.py](../arguments/__init__.py) | New flags go in `OptimizationParams` |
| [scene/dataset_readers.py](../scene/dataset_readers.py) | Splits for LLFF, DTU and Blender; random point clouds |
| [render.py](../render.py), [metrics.py](../metrics.py), `metrics_dtu.py` | Evaluation; `results_eval.json` |
| [experiments/run_matrix.py](../experiments/run_matrix.py) + `matrix_*.yaml` | Runner and blocks |
| [experiments/aggregate.py](../experiments/aggregate.py), [stats.py](../experiments/stats.py), [factors.py](../experiments/factors.py) | Tables, paired statistics and verdict, variant factor table |
| [experiments/geometry.py](../experiments/geometry.py), [probes_summary.py](../experiments/probes_summary.py) | MVC@5%; probe tables. `probes.py` is on branch `feat/probes` |
| [experiments/plots/](../experiments/plots/) | Forest, dose, pass, views, diag, qualitative; `style.py` |
| [scripts/check.sh](../scripts/check.sh), [gpu_smoke.sh](../scripts/gpu_smoke.sh), [pr_ready.py](../scripts/pr_ready.py), [review_bundle.py](../scripts/review_bundle.py) | Gate |
| [scripts/organize_dtu_dataset.sh](../scripts/organize_dtu_dataset.sh), [run_dtu.sh](../scripts/run_dtu.sh), [run_blender.sh](../scripts/run_blender.sh) | Upstream data and run scripts |
| [results/RESULTS.md](../results/RESULTS.md), [results/tables/](../results/tables/), [results/figures/](../results/figures/), [results/raw/](../results/raw/) | Results v1 |
| [results/reported_baselines.csv](../results/reported_baselines.csv) | Published numbers, each verified against its own paper |
| [RUNLOG.md](../RUNLOG.md) | Event log (needs back-filling) |
| [LAB_RUNBOOK_11nov.md](../LAB_RUNBOOK_11nov.md) | 11 Nov lab procedure (untracked) |
| `docs/report/03_method.md`, `04_setup.md` | Drafts on branch `docs/report-method-setup` (#23) |
| `team-context` branch | `aayush_context.md`, `disha_context.md` (last entries 3–4 Nov) |
| [STATUS_AUDIT_16Nov.md](STATUS_AUDIT_16Nov.md) | Tonight's evidence audit |
