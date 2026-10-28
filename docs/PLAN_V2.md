# Plan v2: restructured study and schedule to Tue 10 Nov

**Wed 28 Oct 2026** · Maintainer: Aayush · Hard deadline **Sun 15 Nov**; we submit **Tue 10 Nov** and keep 11–15 Nov as buffer.

This plan replaces the scope, schedule, risks and budget in [PLAN.md](PLAN.md) (§2–§4, §8, §9). It also settles the decision sheet in [RESEARCH_ANALYSIS.md](RESEARCH_ANALYSIS.md) §10. Three parts of PLAN.md still apply:
- **§5, the specs W2–W13.** These say how to build each piece of code; this plan only changes *what* gets built and *when*.
- **§7,** figures and writing, as amended in §9 below.
- **The invariants in AGENTS.md.**

The analysis rules in §5 below replace PLAN.md §6.

---

## 0. Answers first

**Can it be done by 10 Nov?**
- **The study as restructured below: yes, but only just.** Four conditions must hold:
  1. The training code (W2–W9) is merged and tagged by **Sat 31 Oct**.
  2. At least two T4 sources (Colab and Kaggle accounts) are available 31 Oct – 4 Nov.
  3. Disha reviews PRs the same day they open.
  4. Experiments freeze on **Wed 4 Nov**.
- **The "publication-grade" versions in RESEARCH_ANALYSIS.md (packages P1 and P2): no.** They assumed 27 days from 14 Oct with the code done that week. Nothing has landed since 14 Oct: the dropout code doesn't exist and no experiment has run. The 13 days that remain cover the core study plus two extensions, not P1 or P2.
- **CVPR 2027 (16 Nov): no.** That attempt needed Tier 1 results by 27 Oct. Publication becomes a post-course step (§11).

**What changes.** Keep the question, the A–E design and LLFF. Add the two cheapest extensions that most improve the conclusion:
- **dropout strength**, which settles the 0.12 vs 0.2 dose question;
- **number of views**.

Simplify the statistics to what both of you can defend in a viva. Move DTU, initialization, depth-prior quality and the extra probes to future work, or to a stretch slot if you get ahead.

**The one thing that decides success:** the dropout code was due on 13 Oct and still doesn't exist. Days 1–4 (28–31 Oct) are almost entirely Aayush writing and testing it. Everything else waits on that.

---

## 1. Where we are on 28 Oct

| Done | Not done |
| --- | --- |
| DNGaussian reproduced: laptop 18.73, T4 **18.93** vs paper 19.12 | **Dropout port, variant E, `diag.csv`, stats files, runner (W2–W9)**: only `--seed`/`--depth_scale` exist, on branch `feat/dropout` |
| Canonical data archive (md5-locked), Colab notebook, cached wheels | Any experiment run |
| CI with 11 checks, merge gate, GitHub ruleset | Report text (zero sections) |
| Analysis scripts on fixtures: paired tests, interaction contrasts, verdict, 4 figure scripts | Kaggle test (status unknown) |
| Verified published baselines | Open PRs since 14 Oct: **#12** (gate, waits on Disha's approval), **#11** and **#14** (Disha's, wait on review) |
| Research analysis and literature update | — |

Working days to submission: **13** (Wed 28 Oct – Mon 9 Nov), with submission on Tue 10 Nov.

---

## 2. The restructured study

### 2.1 Title and question
**Working title:** *Do Depth Priors and Gaussian Dropout Help Each Other? A Controlled Study in Sparse-View 3D Gaussian Splatting*

**Research question (unchanged):** in sparse-view 3DGS, are DNGaussian's monocular-depth regularization and DropGaussian's Gaussian dropout complementary, redundant or interfering, and does it matter whether the dropout mask reaches the depth supervision?

### 2.2 Sub-questions (what the report answers)
| ID | Sub-question | Evidence |
| --- | --- | --- |
| **RQ1** | Does each regularizer help alone, and does their effect add up when combined? | Tier 1: B − A, C − A, interaction I_D and I_E |
| **RQ2** | Does keeping the mask away from the depth passes (E) beat the naive combination (D)? | Tier 1: E − D (primary contrast) |
| **RQ3** | Does the answer depend on how much dropout we apply? This includes the published end rate of 0.2 | Dose block: γ = 0.1, 0.2, 1/3 |
| **RQ4** | Does it depend on how few views we have? | Views block: 2, 3, 6, 9 views |
| Mechanism | *Why* does any interference happen? | Diagnostics in every run, plus a half-page derivation (§2.4) |

### 2.3 Hypotheses (written before results; part of the pre-registration)
- **H1, depth-signal corruption.** In D the mask perturbs the depth loss: gap(D) > gap(C). If that harms training, E > D, and **E − D grows with dose**.
- **H4, variance penalty.** Dropout trains the model to give similar renders under different masks. So `rgb_var_masks` is lower in C, D and E than in A and B, and `depth_var_masks` is lower in D than in E.
- **H6, diminishing returns.** Both regularizers matter less with more views; the effects and the interaction shrink from 2 to 9 views.

H2 (alpha-clamp bias), H3 (neighbour compensation) and H5 (region specificity) from the analysis stay in the Discussion as explanations to test later. They are not claims of this report.

### 2.4 Theory (half a page in the Method section)
- **Proposition (no compute needed).** Dropping each Gaussian with probability r and scaling survivors' opacity by 1/(1−r) leaves the *expected* rendered image unchanged. The reason: the expected value of a product of independent terms is the product of their expected values.
- **The exception.** The rasterizer caps per-pixel opacity at 0.99. In DNGaussian's hard-depth pass every Gaussian has opacity 0.95, so with compensation it exceeds the cap once r > 0.04. Its expected opacity then falls by 8 % at r = 0.12 and by 17 % at r = 0.2.
- **What follows.** Only D's depth passes are affected; E's are not. This gives the dose block a concrete prediction. Full derivation: RESEARCH_ANALYSIS.md §6.

### 2.5 Contributions (as the report will state them)
1. A controlled, single-codebase comparison of five variants (2×2 factorial plus decoupling) on 8 LLFF scenes × 3 seeds, with analysis rules fixed before any result.
2. A decoupling test (D vs E) and dose and view-count sweeps showing *when* the two regularizers help each other.
3. Mechanism evidence: diagnostics logged during training plus a short derivation of what compensated dropout does to the render.
4. A reproducible harness: tags, md5-locked data, CI-tested analysis.

### 2.6 Scope decisions: the RESEARCH_ANALYSIS.md decision sheet, settled
**IN** = done in this plan; **STRETCH** = only if a buffer slot frees up; **FUTURE** = after 15 Nov.

| Item | Decision | Why |
| --- | --- | --- |
| A1 two-axis verdict | **IN** (simplified, §5) | Every outcome gets a label; easy to explain |
| A2 equivalence margin | **IN**, as the "inconclusive" rule only | One number, δ = 0.25 dB |
| A3 no data-driven best(B,C) | **IN** | Removes a bias at no cost |
| A4 primary contrasts (E − D, I_D, I_E) | **IN** | Focus |
| A5 bootstrap, Wilcoxon, MSE scale | FUTURE | Hard to defend in a viva; little gain |
| A6 adaptive seeds | CUT | No time for a decision point |
| A7 pre-registration tag | **IN** | One command; strong credibility |
| A8 forest plot | **IN** | One figure, reuses `paired_per_scene_3v.csv` |
| B1 dose–response | **IN**: γ ∈ {0.1, 1/3} plus Tier 1's 0.2, C/D/E, 1 seed | Settles 0.12 vs 0.2; tests H1 |
| B2 initialization (dense clouds) | **STRETCH** (40 runs) | Valuable but needs data prep |
| B3 views with A on T4 | **IN**: A–E × {2, 6, 9}, 1 seed | Tests H6; interaction per view count |
| B4 DTU | FUTURE | ~2.5 person-days we don't have |
| B5 prior quality, B6 surface, B7 DropoutGS, B8 second host | FUTURE | — |
| B9 noise floor | **IN**, reduced: B and D on fern, 4 extra same-seed repeats | Measures our noise in ~40 min |
| C1 depth-gradient cosine | **STRETCH** | Most complex diagnostic; don't let it delay W4 |
| C2 `rgb_var_masks` | **IN** | ~10 lines next to `depth_var_masks` |
| C3 structure stats in `train_stats.json` | **IN** | ~15 lines; Gaussian count, opacity and kNN as mediators |
| C4–C6 probes and variants | FUTURE | — |
| D-a diagnostics inside Tier 1 | **IN** | — |
| D-c new arms as new variant IDs and output roots | **IN** (§3.2), needs no gate change | — |
| D-d all inferential runs on T4 | **IN**; laptop only as the views fallback (§8) | — |
| D-f main table at γ = 0.2 and γ = 1/3 | **IN** | — |
| E1–E3 metrics, context table, efficiency | **IN** | E4 artefact archive: **STRETCH**; E5: FUTURE |
| F1 re-framing, F2 citations, F5 write early | **IN** | F3 CVPR: no (§0); F4 template: use the course's |

---

## 3. Experiment matrix

### 3.1 Blocks
One LLFF run is about 4.9 min on a T4. All blocks run from tag `v1.0-exp`. Each block has its own output root, so Tier 1's tables stay clean.

| Block | Variants | Scenes × seeds | Runs | T4-hours | Output root | Matrix file | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **T1, main** | A, B, C, D, E (γ = 0.2) | 8 × 3 (seeds 0, 1, 2) | 120 | 9.8 | `outputs/` | `matrix.yaml` (PLAN W8) | 1 |
| **N, noise floor** | `B_r1..B_r4`, `D_r1..D_r4` (identical flags, repeats) | fern × seed 0 | 8 | 0.7 | `outputs_noise/` | `matrix_noise.yaml` | 2 |
| **G, dose** | `C_g010 D_g010 E_g010` (γ = 0.1), `C_g033 D_g033 E_g033` (γ = 1/3) | 8 × seed 0 | 48 | 3.9 | `outputs_dose/` | `matrix_dose.yaml` | 3 |
| **V, views** | A, B, C, D, E | 8 × seed 0, at 2, 6 and 9 views | 120 | 9.8 | `outputs_views/` | `matrix_views.yaml` | 4 |
| S, initialization (stretch) | `A_mvs … E_mvs` with DNGaussian's MVS configuration | 8 × seed 0 | 40 | ~4 | `outputs_mvs/` | `matrix_mvs.yaml` | 5 |
| **Total core** | | | **296** | **~24** | | | |

Notes:
- The dose block's anchors (A, B) and the γ = 0.2 point come from Tier 1 seed 0, so every dose comparison is paired within seed 0.
- The views block's 3-view row comes from Tier 1 seed 0, on the same GPU type.
- Peak drop rates at iteration 6000 are 0.06 (γ = 0.1), 0.12 (γ = 0.2, D1) and 0.20 (γ = 1/3, the published end rate).

### 3.2 Matrix files (no gate change needed)
Every new variant uses only the three keys that `check.sh` step 9 accepts. The YAML files differ from `matrix.yaml` only in the lines shown:

```yaml
# experiments/matrix_dose.yaml   (scenes, train_flags, steps: as in matrix.yaml)
seeds: [0]
n_views: [3]
variants:
  C_g010: {depth_scale: 0, drop_rate: 0.1,      depth_render: shared}
  D_g010: {depth_scale: 1, drop_rate: 0.1,      depth_render: shared}
  E_g010: {depth_scale: 1, drop_rate: 0.1,      depth_render: clean}
  C_g033: {depth_scale: 0, drop_rate: 0.333333, depth_render: shared}
  D_g033: {depth_scale: 1, drop_rate: 0.333333, depth_render: shared}
  E_g033: {depth_scale: 1, drop_rate: 0.333333, depth_render: clean}

# experiments/matrix_views.yaml
seeds: [0]
n_views: [2, 6, 9]
variants: {A, B, C, D, E as in matrix.yaml}

# experiments/matrix_noise.yaml
scenes: [fern]
seeds: [0]
n_views: [3]
variants:
  B_r1: {depth_scale: 1, drop_rate: 0.0, depth_render: shared}   # ... B_r2, B_r3, B_r4 identical
  D_r1: {depth_scale: 1, drop_rate: 0.2, depth_render: shared}   # ... D_r2, D_r3, D_r4 identical
```

With Tier 1's seed-0 runs, the noise block gives 5 same-seed runs each of B and D on fern. Their spread is the nondeterminism floor.

### 3.3 Compute plan
- **Colab:** Aayush's account takes shard 0/2 and Disha's takes shard 1/2.
- **Kaggle** (if Disha's test works): takes the views block.

| When | Colab A (Aayush) | Colab D (Disha) | Kaggle (Disha) |
| --- | --- | --- | --- |
| Sat 31 Oct, evening | T1 seed 0, shard 0/2 (~1.7 h) | T1 seed 0, shard 1/2 | — |
| Sun 1 Nov | T1 seed 1, then seed 2, shard 0/2 (~3.3 h) | T1 seeds 1 and 2, shard 1/2 | Block N (0.7 h) |
| Mon 2 Nov | Block G, shard 0/2 (~2 h) | Block G, shard 1/2 | Block V, shard 0/2 (~5 h) |
| Tue 3 Nov | Reruns; stretch S | Reruns | Block V, shard 1/2 |
| Wed 4 Nov | Last reruns; **freeze 23:59** | Same | Same |

That is about 24 T4-hours over four days, so it fits even with Colab's daily limits. If Kaggle doesn't work, see trigger R4 in §8.

---

## 4. Code work (Aayush, Wed 28 – Sat 31 Oct)

Specs are in PLAN.md §5. This list adds only the order, the cuts and the acceptance test for each item.

| # | Item | Spec | Acceptance test (§6 levels) | Est. |
| --- | --- | --- | --- | --- |
| 1 | Rebase `feat/dropout` onto `main` (needed for the hook and CI) | PLAN W1 | `check.sh` passes | 0.5 h |
| 2 | **W2** dropout port: `drop_comp` in the three render functions, schedule `γ·t/10000`, one mask per iteration | PLAN W2 | T1–T4 (L1) | 4 h |
| 3 | **W3** variant E: `--depth_render clean` | PLAN W3 | T4 (L1) | 0.5 h |
| 4 | **W4** `diag.csv`, **plus C2** `rgb_var_masks` | PLAN W4; RESEARCH_ANALYSIS App. B | Sharp checks pass on the smoke run's `diag.csv` (L1) | 5 h |
| 5 | **W5** `train_stats.json`, **plus C3** keys (opacity quantiles, `knn_dist_q50`, `frac_opacity_gt_0_9`) | PLAN W5; App. B | Keys present and finite (L1) | 1.5 h |
| 6 | **W7** random-point-cloud race fix | PLAN W7 | Two runs on one scene don't share `input.ply` (L1) | 0.5 h |
| 7 | **W8** runner plus the four matrix files | PLAN W8; §3.2 | V1 dry run lists 120 / 8 / 48 / 120 runs (L2) | 5 h |
| 8 | **W6** FPS in `render.py` | PLAN W6 | `render_stats.json` written; renders unchanged (V4) | 1.5 h |
| 9 | Validation | PLAN W9 | L2 and L3 (§6) | 3 h |
| — | C1 gradient cosine | App. B | — | **STRETCH**, never before the tag |

About 22 working hours. **Cut order if Friday night shows a slip:**
1. Drop W6 (FPS); add it later as `v1.1-exp`, which doesn't change training.
2. Drop C3, computing it later from seed-0 checkpoints.

Never cut: W2, W3, W4, W8.

**PRs.** Experiment class, which needs a GPU-smoke comment, an opus review and Disha's approval.
- **PR-A** (items 1–6), opened by Thu night.
- **PR-B** (items 7–8), opened Fri.

Both are reviewed and merged by Sat afternoon, then:
- tag `v1.0-exp` on the merge commit;
- tag `prereg-v1` on the commit that contains §5 of this file.

---

## 5. Analysis rules v2 (pre-registered; replaces PLAN.md §6)

These rules are fixed **before the first Tier 1 metric exists** and tagged `prereg-v1`. Change them only by PR, with the reason stated.

**Unit.** For each variant and scene, average PSNR, SSIM and LPIPS over seeds. Each comparison is paired by scene: compute the per-scene difference, then report its mean, a 95 % t-interval over the 8 scenes (df = 7), and wins out of 8.

**Primary contrasts** (Tier 1, γ = 0.2, PSNR; Holm correction across the three):
1. **E − D**: does decoupling help?
2. **I_D = (D − C) − (B − A)**: does depth help less (or more) when dropout is on, in the naive combination?
3. **I_E = (E − C) − (B − A)**: the same for the decoupled combination.

**Secondary:** B − A, C − A, D − B, D − C, E − B, E − C; the same contrasts at γ = 0.1 and 1/3 (seed 0) and at 2, 6 and 9 views (seed 0).

**Margin.** δ = 0.25 dB, roughly one run's noise. A difference counts as **"equivalent to 0"** only if its 95 % CI lies inside ±δ. A CI that crosses 0 but sticks out of ±δ is **"inconclusive"**: we can't tell.

**Verdict for X ∈ {D, E}:**

| | Interaction CI > 0 | Interaction equivalent to 0 | Interaction CI < 0 | Inconclusive |
| --- | --- | --- | --- | --- |
| **X beats B and C** (both CIs > 0) | synergistic | **complementary** | complementary but partly redundant | complementary |
| **X equivalent to B** | — | — | **redundant** | redundant? (interaction unclear) |
| **X worse than B** (CI < 0) | — | — | **interfering** | worse (unclear why) |
| **Otherwise** | — | — | partly redundant | **inconclusive** |

Notes on the verdict:
- B, the published DNGaussian, is the fixed reference single. We never choose "the better of B and C" from the same data.
- SSIM and LPIPS must point the same way. If they don't, the verdict is reported as "metric-dependent".

**Mechanism claim** ("the interference runs through the depth supervision"). Make it only if all three hold:
- E − D > 0 (CI);
- gap(D) > gap(C) (CI);
- E − D at γ = 1/3 is larger than at γ = 0.1 in at least 6 of 8 scenes.

**Power, stated in the report.** With 8 scenes and 3 seeds, interactions smaller than about 0.35–0.5 dB are below our detection limit (RESEARCH_ANALYSIS §5.3). Seed-0-only blocks (dose, views) are reported as **trends with per-scene points**, not significance tests.

**Implementation (Disha, by Sun 1 Nov):**
- `stats.py` gets the δ rule, the fixed reference B and the table above. The existing paired tests and contrasts stay.
- `plots/dose.py` (new): E − D, I_D and I_E against peak drop rate (0.06 / 0.12 / 0.20), per scene and mean.
- `views.py` adds A and plots I_D and I_E against views.
- A forest plot of per-scene E − D, I_D and I_E.

---

## 6. Testing and validation

Each level must pass before the next starts. Most failures are cheapest to catch at L0–L2.

| Level | What | Who | When | Pass criterion |
| --- | --- | --- | --- | --- |
| **L0 static** | `bash scripts/check.sh` locally and in CI on every PR: ruff, Python 3.7 syntax, "no dropout in eval paths", data hygiene, notebooks, analysis scripts on fixtures | Both | Every PR | ALL CHECKS PASSED |
| **L1 unit smoke** (laptop, fern, 1100 iterations) | T1: `--drop_rate 0` creates no mask, and B's PSNR lands in 17.3–17.8. T2: `grep drop_comp\|drop_rate render.py metrics.py spiral.py` is empty. T3: fraction of zeros in the mask tracks `0.2·t/10000` ±0.01. T4: in E the depth passes get `None`. `diag_curves.py` sharp checks pass on the smoke `diag.csv`. New `train_stats.json` keys are finite | Aayush | Thu–Fri | All true |
| **L2 integration** | `bash scripts/gpu_smoke.sh` (A–E, ~7 min), with its `GPU-SMOKE PASS` line posted on the PR. V1: the runner's `--dry_run` lists 120 / 8 / 48 / 120 runs with the right flags. V4: `render.py` twice on a D model gives identical render md5s (evaluation never drops) | Aayush | Fri | All pass |
| **L3 regression vs upstream** | V3: variant B at the PR head, fern and horns, full 6000 iterations on Colab. Their mean must be within ±0.5 dB of **19.85**; if not, rerun once before debugging | Disha | Sat morning | Pass |
| **L4 pilot** | V5: the first 10 real Tier 1 runs. Losses finite; Gaussian counts in the tens of thousands; `drop_rate_now` reaches 0.12 at iteration 6000 in C, D and E; `diag_curves.py` sharp checks pass; `stats.py` runs on the partial data | Aayush and Disha | Sat night | All pass; otherwise stop the matrix, fix, tag `v1.1-exp` |
| **L5 production** | After each shard: count `FAILED` markers, rerun once, log in RUNLOG. Run `diag_curves.py` sharp checks on every finished block before analysing it | Owner of the shard | 1–4 Nov | Every planned run has `metrics.json`, or a logged reason why not |
| **L6 analysis check** | Recompute one interaction contrast by hand in a spreadsheet from `runs.csv` and compare with `stats.py`. Regenerate **every** figure and table from scratch from the frozen outputs | Disha checks Aayush's numbers, Aayush checks Disha's | Thu 5 Nov; Mon 9 Nov | Identical numbers |
| **L7 report check** | Every number in the report traces to a `metrics.json` or a cited table. Seeds, GPU, SSIM implementation and LPIPS backbone are stated. Citations verified | Both | Mon 9 Nov | Checklist complete |

---

## 7. Day-by-day schedule

Times are IST. The 21:30 daily call (15 min) continues.

| Day | Aayush | Disha | Compute | Gate checked at the call |
| --- | --- | --- | --- | --- |
| **Wed 28 Oct** | Read this plan. Rebase `feat/dropout`. Start W2 (renderer) | Approve #12. Read this plan. Start `stats.py` v2 (§5) | — | Plan agreed; #12, #11, #14 merged or reviewed |
| **Thu 29** | W2 and W3 done; T1–T4 on the laptop. W4 + C2. **Open PR-A by night** | Finish `stats.py` v2, `dose.py`, views with A. Run the Kaggle smoke test | Laptop smoke runs (daytime OK, < 7 min each) | PR-A open; L1 passes |
| **Fri 30** | W5 + C3, W7. W8 runner and matrix files; V1. `gpu_smoke.sh`, V4. **Open PR-B.** Address PR-A review | Review PR-A (same day). Related Work draft | Laptop: gpu_smoke | L2 passes |
| **Sat 31** | Fix review comments. Merge PR-A and PR-B. **Tag `v1.0-exp` and `prereg-v1` by 16:00.** Start Colab T1 seed 0 shard 0/2. Pilot checks (L4) at night | Morning: **V3 regression** (L3). Approve PRs. Start T1 seed 0 shard 1/2 | T1 seed 0 | **Code freeze**; L3 and L4 pass |
| **Sun 1 Nov** | T1 seeds 1 and 2 (shard 0/2). Write **Method** (incl. §2.4 theory) | T1 seeds 1 and 2 (shard 1/2). Block N on Kaggle. Related Work | T1 seeds 1–2, N | T1 ≥ 80 runs done |
| **Mon 2** | Block G (shard 0/2). **First T1 analysis** (all seeds). Write **Experimental setup** | Block G (shard 1/2); block V shard 0/2 on Kaggle. Tables T1–T3 from real data | G, V | T1 complete (120); preliminary verdict |
| **Tue 3** | Reruns of failures. Stretch S only if T1, N and G are complete. Figure F1 (D vs E diagram) | Block V shard 1/2. Forest plot, dose plot | V, reruns | G complete |
| **Wed 4** | Last reruns. **Experiments freeze 23:59.** Failure-case scene picks | Views plot, efficiency table | Reruns only | **Experiments frozen** |
| **Thu 5** | Final analysis: verdict, mechanism (`diag.csv`), dose trend. L6 cross-check | Final tables and figures from the frozen outputs. L6 cross-check | — | All figures generated by script |
| **Fri 6** | Write Diagnostics + Verdict, Discussion | Write Introduction, Results | — | All sections drafted |
| **Sat 7** | Limitations, failure cases | Results text, captions | — | Full draft |
| **Sun 8** | Swap sections and review each other's. Abstract and conclusion together | Same | — | Reviewed draft |
| **Mon 9** | L6 regeneration from scratch, L7 checklist, README "how to reproduce". Tag `v1.0-report`. Viva Q&A rehearsal (§10) | Same; citations check | — | **Submission-ready** |
| **Tue 10** | **Submit.** | **Submit.** | — | Submitted |
| **11–15 Nov** | Buffer: fixes if anything broke; viva preparation and slides | Same | Stretch runs only if useful for the viva | — |

---

## 8. Risks, triggers and the descoping ladder

| ID | Risk | Detect by | Response |
| --- | --- | --- | --- |
| R1 | Code not merged by Sat 31 Oct 16:00 | Fri 21:30 call | Cut W6 then C3 (§4). Freeze may slip to **Sun 1 Nov 12:00**, then descope to **Level 1** |
| R2 | V3 regression fails (B mean off 19.85 by > 0.5 dB after one rerun) | Sat noon | Diff `train_llff.py` against `upstream-baseline` (flag defaults first). At most 3 h, then freeze with the fix |
| R3 | Pilot (L4) finds a bug after the tag | Sat night | Stop the matrix, fix, tag `v1.1-exp`, rerun only the affected runs, log in RUNLOG |
| R4 | Kaggle unavailable | Thu 29 Kaggle test | Block V moves to the **laptop at night** as a self-contained block, including its own 3-view row so it is on one GPU type: A–E × {2, 3, 6, 9} × 8 = 160 runs × 7.5 min ≈ 20 h = nights of 1–3 Nov. Report it as the laptop's GPU |
| R5 | Colab refuses GPUs or hits limits | Any session | Switch account. Order of priority: T1 > G > N > V. Cut V's 9-view first |
| R6 | T1 not complete by Mon 2 Nov 23:59 | Mon call | Descope to **Level 2** |
| R7 | Writing slips | Sat 7 Nov: full draft? | Cut the views section to one figure; Discussion to half a page |
| R8 | Illness or exams | Any | Use the buffer (11–15 Nov) for writing only; never for new experiments |

**Descoping ladder.** Each level is a complete, defensible report.

| Level | Contents | When used |
| --- | --- | --- |
| **0 (plan)** | T1 (3 seeds) + N + G + V + mechanism + theory | Default |
| **1** | T1 + N + G; V at 2 and 6 views only | Code freeze slips to Sunday |
| **2** | T1 with 3 seeds + G at γ = 1/3 only (24 runs); no V | T1 late by Monday |
| **3 (minimum)** | T1 with 2 seeds (80 runs) + G at γ = 1/3 | Severe compute or code trouble. Still answers RQ1–RQ3 |

---

## 9. Report plan

**Format:** the course's template and page limit (ask the instructor if unknown; open question).

| Section | Owner | Content | Draft by |
| --- | --- | --- | --- |
| Abstract, Conclusion | Both | The verdict in one sentence, with its number | Sun 8 |
| 1 Introduction | Disha | Sparse views → overfitting. Two families of fixes. Prior work stacked them (DropoutGS Tab. 3, StereoGS) but never measured the interaction or D vs E. Our RQs and contributions (§2) | Fri 6 |
| 2 Related work | Disha | Sparse-view NeRF and 3DGS; depth priors (DNGaussian, FSGS, DNGaussian++); dropout family (DropGaussian, DropoutGS, D²GS, DropAnSH-GS, UGOD); dropout theory (Srivastava 2014, Wager 2013) | Sun 1 |
| 3 Method | Aayush | 3DGS in one paragraph; DNGaussian's three passes; the dropout port (D1 schedule); variants A–E; Figure F1; the proposition and clamp table (§2.4) | Sun 1 |
| 4 Experimental setup | Aayush | LLFF 3-view split, `-r 8`, 6000 iterations, seeds, T4, metrics (3DGS SSIM, LPIPS VGG), analysis rules (§5) with the pre-registration tag | Mon 2 |
| 5 Results | Disha | T1 main table (mean ± std; published numbers below a rule labelled "reported, not comparable"); T3 primary contrasts and verdict; forest plot; dose figure; views figure; efficiency | Sat 7 |
| 6 Mechanism | Aayush | Diagnostic curves (gap, `depth_var_masks`, `rgb_var_masks`), structure stats, the dose trend vs H1, H4 and H6 | Fri 6 |
| 7 Discussion, limitations | Both | Failure cases (2–3 scenes, with RGB and depth); limitations: one host, one dataset, random init, DPT, seed-0-only sweeps, nondeterminism; future work (§11) | Sat 7 |

**Figures and tables.**
- F1: D vs E diagram.
- T1: main table.
- T2: per-scene table (appendix).
- T3: contrasts and verdict.
- F2: diagnostics.
- F3: qualitative grid.
- F4: failure cases.
- **F6: forest plot** (new).
- **F7: dose curves** (new).
- F5: views.
- T4: efficiency.

Every figure comes from a script in `experiments/plots/`, run on the frozen outputs.

---

## 10. Viva: how advanced the work is, and preparation

### 10.1 Assessment

These are my estimates, against what is typical for an introductory ML course project. A typical project there applies an existing model to a dataset and reports an accuracy table.

| Dimension | Typical intro-ML project | This project (design) | This project **today** | After this plan |
| --- | --- | --- | --- | --- |
| Problem and motivation | 4 | 8 | 8 | 8 |
| Technical depth (code, methods) | 4 | 8 | 5: reproduction and infrastructure only | 8 |
| Experimental design | 3 | 9 | 9, on paper | 9 |
| Results and evidence | 5 | — | **2**: reproduction only | 7–8 |
| Statistical analysis | 2 | 8 | 3: scripts tested on fake data | 7–8 |
| Reproducibility | 3 | 9.5 | 9 | 9.5 |
| Communication (report) | 5 | — | **1**: not started | 7–8 |

**Headline:**
- **Ambition and sophistication: about 8.5/10.** The design (a controlled factorial comparison of two recent CVPR methods, pre-registered statistics, a mechanism analysis) is closer to graduate or workshop-paper level than to an introductory course project.
- **Completion today: about 3/10, roughly 30 % of the total effort.** The infrastructure is done; the results and the report are not.
- **Projected if this plan is executed: about 8/10.** The remaining gap to a publishable paper is generality (one dataset, one host), not quality.

**What will decide the viva mark** is whether *both of you* can explain every part in your own words. That covers the code you submit, including any written with AI help: be ready to walk through `train_llff.py`'s three passes and your dropout lines. Disclose AI assistance as the course's policy requires. If a part of the report can't be defended, cut it. Equivalence tests, Holm correction and the theory are optional; the core contrasts are not.

### 10.2 Map the project to the course syllabus
Examiners in an introductory ML course will probe fundamentals. Frame the project in their terms:

| Course concept | Where it appears in the project |
| --- | --- |
| Overfitting, generalization | 3 training views → good training PSNR, poor novel-view PSNR (floaters, wrong depth) |
| Regularization | Depth prior = an external prior; dropout = a structural regularizer |
| Dropout (neural nets) | DropGaussian applies it to Gaussians; inverted-dropout scaling 1/(1−r) |
| Bias–variance | Dropout as a variance penalty (§2.4); depth prior reduces variance at the risk of bias if the prior is wrong |
| Train/test split, held-out evaluation | LLFF: every 8th image held out; dropout never active at evaluation |
| Hyperparameters | Dropout rate γ (the dose block), number of views |
| Statistical significance | Paired differences, confidence intervals, multiple seeds, "inconclusive" |

### 10.3 Likely questions, with short answers to rehearse
1. **What is 3D Gaussian Splatting?** A scene stored as many 3-D Gaussians, each with position, shape, opacity and colour. They are projected to the image and blended front to back (alpha compositing), and fitted by gradient descent to the training photos.
2. **Why do few views fail?** Many 3-D scenes explain 3 photos equally well, so the optimizer overfits: floaters and wrong geometry that look fine from training views but break from new ones.
3. **What does DNGaussian add?** A monocular depth map (from DPT) as extra supervision. It renders depth in two extra passes per iteration and compares patch-normalized depth, because monocular depth has unknown scale and shift.
4. **What does DropGaussian add?** Each iteration it removes each Gaussian with probability r (rising during training) and scales the survivors' opacity by 1/(1−r). It is dropout for Gaussians; at test time all Gaussians are used.
5. **Why scale by 1/(1−r)?** So the expected opacity, and therefore the expected image, is unchanged. Exception: the 0.99 opacity cap (§2.4).
6. **What are variants A–E, and why five?** A 2×2 design (depth on/off × dropout on/off) isolates each effect and their interaction. E tests one design choice of the combination.
7. **What is the difference between D and E?** In D the same dropout mask also hits the two depth passes. In E the depth passes see the full model, so we can tell whether dropout spoils the depth supervision.
8. **What is the interaction contrast?** I = (D − C) − (B − A): how much depth helps with dropout on, minus how much it helps without. 0 means the effects add up; negative means they overlap or conflict.
9. **Why seeds, and why pair by scene?** Runs vary by about 0.2–0.4 dB even with the same settings. Scenes differ in difficulty, so we compare variants within each scene and average over seeds.
10. **Why a t-interval with 7 degrees of freedom?** 8 scenes give 8 paired differences; for a small sample, a t-interval accounts for not knowing the true spread.
11. **What does "inconclusive" mean?** The confidence interval includes 0 but is too wide to show the effect is negligible, so we can't claim either.
12. **What do PSNR, SSIM and LPIPS measure?** PSNR is pixel error on a log scale. SSIM compares local structure (means, variances, covariance). LPIPS is a distance between deep-network features, closer to human judgement; lower is better.
13. **Why not compare with the published numbers?** Different initialization, metric implementations and hardware. Only our own controlled variants are compared statistically.
14. **Same seed, different result?** The rasterizer's backward pass adds numbers in parallel (atomic additions on the GPU), so their order varies, and floating-point rounding differs.
15. **What would you do with more time?** A second dataset (DTU), dense initialization, a stronger depth model, and a depth loss that knows about the mask.

---

## 11. After 15 Nov: the publication path

If the results are clear, extend to a paper (TMLR, which reviews on a rolling basis, or a CVPR 2027 workshop, typically due Feb–Mar 2027). In order of value per effort:
1. Initialization moderator (stretch S, 40–120 runs).
2. DTU 3-view (B4).
3. Adaptive seeds, bootstrap and equivalence tests (A5, A6).
4. Gradient-cosine diagnostic and clamp probes (C1, C4).
5. A second host (B8).

RESEARCH_ANALYSIS.md §7–§9 has the details.
