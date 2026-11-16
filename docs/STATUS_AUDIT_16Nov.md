# Status audit, Mon 16 Nov 2026 (snapshot at 21:41 IST)

This is a dated snapshot. It answers three questions before we start [EXECUTION_PLAN_15-21Nov.md](EXECUTION_PLAN_15-21Nov.md):
1. What is actually done, with evidence?
2. What is missing, stale or inconsistent?
3. What has to happen before each workstream of the new plan can start?

The living document is [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md). Update it, not this file.

**Method.**
- Ran `git fetch --all --tags`, then read tags, branches, `gh pr list --state all` and `gh issue list --state all` on `aayxsh23/dngs-dropout`.
- Read all five plan documents, [RUNLOG.md](../RUNLOG.md), [results/RESULTS.md](../results/RESULTS.md), [results/raw/README.md](../results/raw/README.md), [LAB_RUNBOOK_11nov.md](../LAB_RUNBOOK_11nov.md), the matrix files, the team context files on the `team-context` branch, and the code the new plan touches.
- Counted the files under `results/`.
- Nothing ran on a GPU, and no code or plan was changed.

**Evidence rule.** A row is DONE only if one of these backs it: a merged PR, a tag, a commit, a file path, a RUNLOG line or a row in `results/tables/`. A plan's own status table counts as a claim, not as evidence. The team context files are the authors' own logs, so they are cited as "context log". Work that only the lab PC, Drive or Disha's laptop could show is marked CANNOT VERIFY.

**Status values.** DONE · PARTIAL · NOT STARTED · DROPPED (where it was dropped is cited) · SUPERSEDED (a later plan replaced it) · CANNOT VERIFY.

---

## 1. Summary

**The LLFF study is complete and committed.** Five blocks, 344 runs on the lab RTX A4000 from tag `v1.0-exp`, with probes, geometry, tables, figures and a pre-registered verdict, merged in #29–#31 on 11 Nov.

**The new plan has not started anywhere we can see.**
- Its first gate has been missed: no `prereg-v3` tag exists on origin, and it was due Sun 23:59.
- None of the files, branches or tags it creates exist yet: `feat/multi-dataset`, `bias_variance.py`, the new matrix files and `results/audit/` are all missing.
- #27 has not been rebased, and it has no `nc` mode.
- The DTU go/no-go was due Mon 20:00. Its outcome is recorded nowhere.
- Today's lab work (views seeds 1–2, DTU download, the eval tarball) may have happened, but the repo can't show it.

**Record-keeping has fallen behind the work.**
- RUNLOG's last entry is 14 Oct. It has nothing on the tags, the 344 runs, the move to the lab or the deadline change.
- The PLAN.md status table stops at 14 Oct. Both team context files stop at 3–4 Nov.
- AGENTS.md still gives the deadline as 20 Oct.
- The execution plan itself and `LAB_RUNBOOK_11nov.md` are untracked files.

**This audit found four gaps the new plan doesn't mention:**
- **G-a.** With `--rand_pcd`, Blender still writes its random point cloud into the dataset folder, as `<scene>/points3d.ply`. The W7 fix covers only the COLMAP (LLFF/DTU) reader, so parallel Blender runs would race and change the data's md5.
- **G-b.** `experiments/factors.py` rejects every new variant ID the plan introduces (`C_nc`, `D_nc`, `C_ca`, `D_ca`, `B_ds050`, `D_ds200`, `A_r1`, `C_r1`, …). Under its fail-closed rule, `stats.py` stops on them.
- **G-c.** `experiments/probes.py` produced committed results but exists only on branch `feat/probes`, with no PR.
- **G-d.** The README is still upstream DNGaussian's README. D13 has to write a project README from scratch, not update one.

## 2. Repository state

| Item | State | Evidence |
| --- | --- | --- |
| `main` | `f368680` (11 Nov, #31) | `git log` |
| Tags on origin | `upstream-baseline` (81c3d83), `v1.0-exp` and `prereg-v1` (both on 1040559, 3 Nov) | `git ls-remote --tags origin` |
| Tags the plans expect but which **don't exist** | `prereg-v2` (planned with #27, 3 Nov), `v1.1-exp`, `prereg-v3`, `v1.2-exp`, `v1.0-report` | same |
| Open PRs | #27 clamp-aware dropout (head `9d0cfa5`, 3 Nov, `checks` and `agent-review` green, 3 commits behind main, merges cleanly, not approved). #24 tracker (gate class, not approved). #23 Method and Setup drafts (docs-only). #17 team-context protocol (gate class, not approved) | `gh pr view` |
| Open issues | #15, #18, #21, #25, #26 (gate tooling follow-ups); #28 (analysis: `drop_mode`, `C_ca`/`D_ca` in the factor table) | `gh issue list` |
| Branches with work not on main | `feat/probes` (probes.py, no PR), `feat/clamp-aware` (#27), `docs/report-method-setup` (#23), `feat/tracker` (#24), `chore/team-context-protocol` (#17), `team-context` (context logs) | `git for-each-ref` |
| Stale local refs | `origin/exp/results-v1-*` and `origin/claude/project-thread-d8fi8g` were deleted on GitHub after #29–#31 merged; local remote-tracking refs remain (`git fetch --prune` clears them) | GitHub branch list |
| Untracked files | `docs/EXECUTION_PLAN_15-21Nov.md`, `LAB_RUNBOOK_11nov.md` | `git status` |
| CI | `checks` green on #27 head; ruleset active since 13 Oct | `gh pr view 27`, RUNLOG 13 Oct |

## 3. Runs: planned vs present

Every committed run was trained on the lab RTX A4000 from `v1.0-exp` (commit `1040559`) with `shared_gpu: false`. The software was PyTorch 1.12.1 at `/home/snu/dngs-dropout`, and all runs finished by Fri 6 Nov ([results/raw/README.md](../results/raw/README.md); the `gpu` and `code_tag` keys in `metrics.json`; `torch_version` in `train_stats.json`).

| Block | Plan says | Present in `results/raw/` | Status |
| --- | --- | --- | --- |
| T1 main (A–E, 8 × 3 seeds, 3 views) | 120 (PLAN_V2 §3.1) | 120 `metrics.json`, 24 per variant | DONE |
| P pass (D_hard, D_soft, 8 × 3) | 48 | 48 | DONE |
| G dose (C/D/E at γ 0.1 and 1/3, seed 0) | 48 | 48 | DONE |
| N noise (B_r1–4, D_r1–4, fern, seed 0) | 8 | 8 | DONE |
| V views | 160 on an A100 (`matrix_views_a100.yaml`), or the fallback of 120 at 2/6/9 views with 3 views taken from T1 | 120 (`matrix_views.yaml`, 40 each at 2/6/9 views), on the A4000 | DONE (fallback form) |
| Probes (T1 seed 0) | 40 | 40 `probes/` folders; `tables/probes_*_3v.csv` | DONE |
| Geometry MVC@5% | all runs with `depth_raw` (344) | 120 + 48 + 48 + 8 + 120 = 344 rows in `tables/geometry/*/` | DONE |
| GPU smoke of 1040559 | 2 | `results/raw/smoke/1040559/{B,D}` | DONE |
| S init (stretch) | 40 | 0 | DROPPED (stretch, PLAN_V2 §2.6; future work in the new plan §0) |
| CA clamp-aware (#27) | 48 | 0 | NOT STARTED (#27 unmerged) |
| **New plan blocks:** views seeds 1–2 (240), comp (96), DTU (225), Blender (120), depth-dose (96), noise2 (48) | 825 | 0 in the repo | CANNOT VERIFY for views s1–2 (lab, started Mon per A3?). The other matrix files don't exist yet, so those runs can't have started |

The views fallback appears in only two places: `results/raw/README.md` and RESULTS.md. Neither PLAN_V2 nor RUNLOG records it.

## 4. Traceability

### 4.1 PLAN.md (12–14 Oct): workstreams, decisions, figures

| ID | Description | Status | Evidence | Notes / gaps |
| --- | --- | --- | --- | --- |
| W1 | Merge #1–#4 | DONE | #1–#4 merged 12 Oct | — |
| W2 | DropGaussian port, D1 schedule | DONE | #19 → 1040559; `check.sh` D1 grep; AGENTS invariant 7 (#12) | — |
| W3 | `--depth_render clean` (E) | DONE | #19 | Also `hard_only` / `soft_only` (PLAN_V2) |
| W4 | `diag.csv` | DONE | columns `loss_depth_used, loss_depth_clean, depth_var_masks, drop_rate_now, rgb_var_masks, …` in every raw run | `depth_grad_cos` (C1) was never added |
| W5 | `train_stats.json` | DONE | keys incl. `knn_dist_q50`, `frac_opacity_gt_0_9` | — |
| W6 | FPS, `render_stats.json` | DONE | file in every raw run; `tables/efficiency_3v.csv` | — |
| W7 | Random point-cloud race fix | PARTIAL | LLFF/DTU reader uses `tempfile.mkdtemp` (`scene/dataset_readers.py:226`) | **Gap G-a:** the Blender reader still writes `<scene>/points3d.ply` (`dataset_readers.py:353-362`) |
| W8 | Runner + matrix files | DONE | `experiments/run_matrix.py`, 7 matrix yaml files | Has no `dataset` key yet (new plan A7) |
| W9 | Validation (V1–V4, GPU smoke) | DONE | context log 29 Oct/3 Nov: GPU-SMOKE PASS at 4f03a37, 2957499 and 7633acc; V3 19.66 vs 19.85 (Disha, comment on #19) | — |
| W10 | Freeze and tag | DONE | `v1.0-exp` 3 Nov (planned for 14 Oct) | — |
| W11 / W12 | Colab and laptop operations | SUPERSEDED | Every run used the lab A4000 (`metrics.json` `gpu`) | Not recorded in any plan or RUNLOG |
| W13 | A100 lab | SUPERSEDED | The lab GPU is an RTX A4000 ([LAB_RUNBOOK_11nov.md](../LAB_RUNBOOK_11nov.md)) | AGENTS.md still lists an "A100 lab machine" |
| W16 | Failure cases | PARTIAL | `results/figures/fig_fail.png` (horns, orchids; B, D, E) | No text |
| D1 | Schedule `γ·t/10000`, peak 0.12 | DONE | RUNLOG 13–14 Oct; #12 | — |
| D2 | 3DGS SSIM in the main table | DONE | `tables/main_3v.md` | E1's rationale correction (DNGaussian used sklearn SSIM) belongs in the T1 caption |
| D3–D10 | FPS owner, Tier 2 variants, machine per seed, D definition, pre-dropout `opa_pena`, F1 owner, daily call, repo public | DONE / SUPERSEDED | RUNLOG 13 Oct | D4 and D5 were superseded by PLAN_V2 and then by the lab |
| D11 | Tier 2b doses | SUPERSEDED | became block G | — |
| T1–T4 | Main, per-scene, paired, efficiency tables | DONE | `tables/main_3v.{md,tex}`, `per_scene_3v.csv`, `paired_3v.{md,tex}`, `efficiency_3v.csv` | T1's footnote on the reported rows is unchecked |
| F1 | D vs E iteration diagram | NOT STARTED | no `fig1_*` in `results/figures/` | Still AY-24 |
| F2–F5 | Diag curves, qualitative, failure, views | DONE | `fig_diag`, `fig_qual`, `fig_fail`, `fig_views*` | — |
| §7.4 | Pre-submission checklist (`v1.0-report`, status table, RUNLOG final) | NOT STARTED | no tag; RUNLOG ends 14 Oct | — |

### 4.2 RESEARCH_ANALYSIS.md (14 Oct): recommendations

The decision sheet (§10) was never ticked in this file. PLAN_V2 §2.6 settled it, so the status below follows PLAN_V2 §2.6.

| ID | Status | Evidence / where decided |
| --- | --- | --- |
| A1 two-axis verdict | DONE | PLAN_V2 §5 verdict table; `stats.py` v2 (#20); `verdict_3v.txt` |
| A2 equivalence margin δ | DONE (PSNR only) | δ = 0.25 dB; SSIM/LPIPS have no δ (RESULTS.md notes it) |
| A3 no data-driven best(B, C) | DONE | B is the fixed reference (`verdict_3v.txt`, line 1) |
| A4 primary family + Holm | DONE | `verdict_3v.txt` |
| A5 bootstrap, Wilcoxon, MSE scale | FUTURE in PLAN_V2 → **revived** as N4 (MSE/RMSE/SSIM scales) in the new plan | not started |
| A6 adaptive seeds | DROPPED | PLAN_V2 §2.6 "CUT" |
| A7 pre-registration tag | DONE | `prereg-v1` on 1040559 (the tag message says "before any Tier 1 metric exists") |
| A8 forest plot | DONE | `fig_forest*.png` |
| B1 dose | DONE (reduced: γ 0.1 and 1/3, 1 seed) | block G |
| B2 initialization | DROPPED | PLAN_V2 stretch; new plan §0 "init moderator → Future Work" |
| B3 views with A | DONE (1 seed) → seeds 1–2 are new-plan W4 | block V |
| B4 DTU | FUTURE in PLAN_V2 → **revived** as new-plan W1 | not started |
| B5 prior quality / DAv2 | DROPPED | new plan §0 |
| B6 interaction surface | **revived in part** as N5 (depth_scale 0.5/1/2 at γ 0.2) | not started |
| B7 DropoutGS-style dropout | FUTURE | — (close to the new `nc` mode, which has no consistency loss) |
| B8 second host | DROPPED | new plan §0 |
| B9 noise floor | DONE (reduced: fern only) → extended as G6/noise2 | block N |
| C1 depth-gradient cosine | DROPPED | PLAN_V2 stretch; never logged |
| C2 `rgb_var_masks` | PARTIAL | logged in every `diag.csv`; **never analysed** (H4) |
| C3 structure stats | PARTIAL | `train_stats.json` keys present; **never analysed** (H3) |
| C4 clamp decomposition | DONE as the probe | `probes_depth_bias_3v.csv`; `D_nc` arm → new plan W5 |
| C5a robustness curve | DONE | `probes_robustness_3v.csv`, `fig_robustness.png` |
| C5b region-stratified error | DROPPED | new plan §0 |
| C6 single-pass masking | DONE | block P |
| D-a…D-f | DONE / SUPERSEDED | D-d (T4 only) superseded by "lab A4000 only" |
| E1–E3 | DONE | tables; efficiency from quiet A4000 runs |
| E4 artefact release | PARTIAL | `results/raw/`, and `results_full_nockpt.tgz` on Drive (md5 in raw README) |
| E5 geometry | DONE | MVC@5% |
| F1 framing | SUPERSEDED | new title in the new plan (bias–variance account) |
| F2 citations | CANNOT VERIFY | Related Work is in Disha's Claude Doc (context log 30 Oct) |
| F3 venue / F4 CVPR template | PARTIAL | the new plan §8 says "CVPR template"; PLAN_V2 §9 says "course template, to confirm" |
| F5 write early | PARTIAL | #23 drafts (3 Nov), unmerged and stale (§6) |

### 4.3 PLAN_V2.md (29 Oct): tasks, hypotheses, gates

| ID | Status | Evidence / gap |
| --- | --- | --- |
| AY-01 | PARTIAL | #16 merged; the team-context PR **#17 is still open** |
| AY-02, 03, 06, 07, 09, 11 | DONE | #19 |
| AY-04 | DONE | #11, #12, #14, #20, #22 merged |
| AY-05 go-bag | DONE | `~/dngs-gobag/` exists on the laptop |
| AY-08 sample folder | DONE (context log) | Drive `sparse3dgs/samples/sample_runs_text_29oct.tgz`, CANNOT VERIFY on Drive |
| AY-10, 13–15, 17, 18 (Colab/A100 runs) | SUPERSEDED | all blocks ran on the lab A4000 |
| AY-12 tags | DONE | `v1.0-exp`, `prereg-v1` |
| AY-16, AY-20 Method / Setup | PARTIAL | #23 open; Setup is stale (says T4/A100, Python 3.13; actual A4000, PyTorch 1.12.1); has `[fill]` markers |
| AY-19 first T1 analysis | DONE | `verdict_3v.txt` |
| AY-21 `geometry.py` | DONE | on main; 344 geometry rows |
| AY-22 `probes.py` | PARTIAL | **only on `feat/probes` (9f8d55f), no PR (gap G-c)**; ran from the pinned commit (md5 in RESULTS.md) |
| AY-23 probes + geometry on all runs | DONE | §3 |
| AY-24 F1 diagram | NOT STARTED | — |
| AY-25 mechanism analysis | PARTIAL | verdict mechanism lines only; H3/H4 not analysed |
| AY-26–AY-30 | NOT STARTED (no evidence) | writing, L6/L7, README, `v1.0-report`, viva |
| DS-01–DS-07 | DONE (context log + merged PRs) | #20, #22; reports on #11/#14/#19; Kaggle YES |
| DS-08, DS-09 | DONE | V3 19.66 (−0.19 dB, PASS) on #19 |
| DS-10, 12, 15 | SUPERSEDED | lab |
| DS-11, DS-16, DS-17, DS-19 | DONE | tables and figures; RESULTS.md "How to regenerate" |
| DS-13 Related Work | CANNOT VERIFY | Claude Doc (artifact link in Disha's context log) |
| **DS-14 `scene_factors.py`, DS-18 per-scene moderators** | NOT STARTED | no such file on any branch. Not explicitly dropped by the new plan |
| DS-20–DS-24 | NOT STARTED (no evidence) | — |
| §6 L4 pilot / L6 cross-check / L7 report check | CANNOT VERIFY / NOT STARTED | no record |
| Gates of 31 Oct – 10 Nov | All superseded | the freeze was 3 Nov, runs ran 6 Nov, results merged 11 Nov; the planned 10 Nov submission was replaced |

**Hypotheses (PLAN_V2 §2.3, plus H7 from #27):**

| H | Prediction | Status | Evidence |
| --- | --- | --- | --- |
| H1 depth-signal corruption | gap(D) > gap(C); E > D; E − D grows with dose | **Not supported** (2 of 3 conditions fail) | `verdict_3v.txt`: E − D +0.10 [−0.10, +0.30]; gap(D) − gap(C) +0.0013 [−0.0060, +0.0087]; dose condition met 7/8 |
| H2 opacity cap | D_hard − E < 0; probe shows a far shift that grows with r | **Not supported** (direction right, size tiny) | D_hard − E −0.12 [−0.32, +0.08]; median shift +0.05% (D) at r = 0.2 |
| H3 neighbour compensation | B more redundant than A; I_E < 0 | **Premise contradicted (exploratory)** | probes: B loses *more* than A at f = 0.2 (−1.03 vs −0.79 dB, compensated); I_E inconclusive; kNN not analysed |
| H4 variance penalty | `rgb_var_masks` lower in C/D/E | **Not analysed** | data in every `diag.csv` |
| H6 diminishing returns | effects shrink 2 → 9 views | **Mixed, seed 0 only** | I_D at 6 views −0.96 [−1.68, −0.24] (unadjusted); B < A at 9 views |
| H7 clamp-aware (#27) | C_ca vs C, D_ca vs D | NOT STARTED | #27 unmerged; no `prereg-v2` |

### 4.4 EXECUTION_PLAN_15-21Nov.md: workstreams and tasks

| ID | Owner | Due | Status (Mon 21:41) | Evidence / gap |
| --- | --- | --- | --- | --- |
| W1 DTU | A | Mon | NOT STARTED in the repo; lab CANNOT VERIFY | `data/dtu/` on the laptop holds only `.gitkeep`; `train_dtu.py` has none of our flags (`safe_state(args.quiet)` without a seed, line 365) |
| W2 Blender | A+D | Mon–Tue | NOT STARTED in the repo; Disha's laptop CANNOT VERIFY | `data/nerf_synthetic/` empty here; gap G-a |
| W3 factorial on DTU/Blender | A | Tue–Thu | NOT STARTED | needs `v1.2-exp` |
| W4 views seeds 1–2 | A | Mon start | CANNOT VERIFY | ready to run: `matrix_views.yaml` + `--seeds 1 2` gives 240 |
| W5 compensation | A | Tue | NOT STARTED | #27 has `clamp_aware` only; no `nc`; `matrix_ca.yaml` has 48 runs, not 96 |
| W6 N1 | D | Mon–Tue | NOT STARTED | no `bias_variance.py`; eval renders not in the repo |
| W7 N2 + G6 | D/both/A | Wed | NOT STARTED | no `results/audit/`, no `matrix_noise2.yaml` |
| W8 N4 | D | Tue | NOT STARTED | needs D3 |
| W9 N5 | A/D | Wed | NOT STARTED | no `matrix_depthdose.yaml`; gap G-b |
| W10 N3 | D | Thu, optional | NOT STARTED | — |
| W11 `prereg-v3` | D | **Sun 23:59** | **MISSED** | no tag on origin; no PR open |
| W12 report, site, README, viva | both | Fri–Sat | PARTIAL | #23 drafts only; gap G-d |
| A1 rebase #27 + `nc` | A | Sun | **NOT DONE** | #27 head `9d0cfa5` is from 3 Nov, 3 commits behind main, no `nc` |
| A2 lab checklist | A | Sun | CANNOT VERIFY | — |
| A3 views s1–2 start | A | Mon | CANNOT VERIFY | no RUNLOG line |
| A4 `eval_T1.tgz` | A | Mon eve | CANNOT VERIFY | possible fallback: Drive `results_full_nockpt.tgz` (2.84 GB). Whether it holds `eval/*/renders` is not stated (inference: it should, since cleanup keeps `eval/`) |
| A5 DTU acquisition | A | Mon | CANNOT VERIFY | — |
| A6 DTU go/no-go | A | Mon 20:00 | **NO RECORD** | the time has passed; no decision written anywhere |
| A7 multi-dataset PR | A | Tue | NOT STARTED | no branch |
| A8 #27 → `v1.1-exp` | A | Tue | NOT STARTED | needs A1 + GPU smoke + Disha |
| A9–A13 | A | Tue–Sat | NOT STARTED | future |
| D1 `prereg-v3` | D | Sun | **MISSED** | see W11 |
| D2 Blender data | D | Sun–Mon | CANNOT VERIFY | — |
| D3 `bias_variance.py` | D | Mon–Tue | NOT STARTED (no branch) | can be built on fixtures now; may not run on real renders before `prereg-v3` (rule 7) |
| D4 dataset factor | D | Mon–Tue | NOT STARTED | `aggregate.py` hard-codes `llff_` (`PATH_RE`, line 25) and LLFF `SCENES` |
| D5–D13 | D | Tue–Sat | NOT STARTED | future |

**Definition of done (§9):** 0 of 9 boxes can be ticked today.

## 5. Gates and handoffs as of today

| When | Gate / handoff | State | Consequence |
| --- | --- | --- | --- |
| Sun 23:59 | `prereg-v3` tagged + noise2 draw | **Missed** | Rule 7 blocks every new statistic and `matrix_noise2.yaml`. Each extra hour widens the gap between data access and the tag |
| Sun | A1 #27 rebased + `nc` | **Missed** | A8 and the compensation block (first in the A9 queue) slip |
| Sun–Mon | D2 Blender data | Unknown | A7's Blender part needs `blender_with_depth.tar` by Tue morning |
| Mon | A3 views s1–2 started | Unknown | BV4 (D6) needs it |
| Mon eve | A4 `eval_T1.tgz` → Disha | Unknown | D3 has no real data without it |
| **Mon 20:00** | **A6 DTU go/no-go** | **Unrecorded** | If no-go or not run, the plan's own rule makes Blender the required second dataset; the decision must be written down tonight |
| Tue 23:59 | `v1.2-exp` | At risk | A7 is 6–8 h of work plus review, smoke and Disha's approval; nothing exists yet |
| Wed 12:00 | DTU throughput | At risk | depends on A6 and A7 |

## 6. Inconsistencies

| # | Where | What | Suggested fix |
| --- | --- | --- | --- |
| I-1 | AGENTS.md line 7; PLAN.md header; PLAN_V2 header | Deadline: "Tue 20 Oct" / "Sun 15 Nov" / "submit 10 Nov". The new plan says submit **Sat 21 Nov 20:00, deadline 23:59**. Nothing records the move | RUNLOG line + AGENTS.md fix (gate-class PR, Disha approves) |
| I-2 | AGENTS.md environment table and runbook text | Lists an "A100 lab machine ?"; the lab is an RTX A4000 with PyTorch 1.12.1 at `/home/snu/dngs-dropout` | Same PR |
| I-3 | AGENTS.md / CLAUDE.md vs PLAN_V2 vs the new plan | Where status lives: "PLAN.md status table is the single place" (AGENTS, CLAUDE.md) vs "the team context files" (PLAN_V2 header) vs "daily sync + RUNLOG" (new plan rule 8). All three are stale | Pick one (proposal: RUNLOG for events + the status table in RESEARCH_GUIDE §6) |
| I-4 | RUNLOG.md | Last entry 14 Oct. Missing: `v1.0-exp`/`prereg-v1` (3 Nov), the lab move, the 344 runs (6 Nov), the probes/geometry lab day (11 Nov), #29–#31, the deadline change. Rule 5 of the new plan requires a RUNLOG entry per tag | Back-fill from `results/raw/README.md` and the context logs |
| I-5 | AGENTS.md invariant 3 | Layout given as `outputs/llff_<n>v/...`; the new plan adds `dtu_3v` and `blender_8v` (allowed as an addition by RESEARCH_ANALYSIS D-c, but not written into the invariant) | Note it in the A7 PR |
| I-6 | New plan D1 vs `main` | prereg-v3 goes into "PLAN_V2 §5.2, mirroring #27's §5.1". §5.1 and the `prereg-v2` tag exist only on the unmerged #27, so on main §5.2 would follow §5 directly | Merge order: #27 (with prereg-v2) first, or number the new addendum independently |
| I-7 | New plan A12 | Lists `prereg-v1/v2/v3` and `v1.0/1.1/1.2-exp` as existing. Only `prereg-v1` and `v1.0-exp` exist | — |
| I-8 | New plan rule 3 | Says to record the DPT decision in `docs/PLAN.md`, the oldest plan | Record it in RUNLOG and the guide |
| I-9 | New plan, `nc` definition | A1: `nc` = no compensation in **every** pass. RESEARCH_ANALYSIS C4 / H2: `D_nc` = no compensation **in the depth passes only**. These test different things | Fix the definition in prereg-v3 before A1's code |
| I-10 | New plan §0 / rule 7 | Cites a "15 Nov memo" and "15 Nov MSE-scale and heterogeneity numbers"; neither is in the repo | Commit the memo or quote it in prereg-v3's exploratory disclosure |
| I-11 | New plan figures F1–F11 vs PLAN_V2 F1–F10 | Same labels, different figures (e.g. F1 = iteration diagram vs error budget) | Use file names, not F-numbers, in the report |
| I-12 | Report template | New plan §8 says CVPR; PLAN_V2 §9 says the course template, "still to confirm with the instructor" | Confirm with the course |
| I-13 | #23 Setup draft | Says T1/P/G/N on T4 and V on an A100 with 160 runs; actual: A4000 for everything, V = 120 + T1 seed 0 | Rewrite in A12 |
| I-14 | RESULTS.md vs PLAN_V2 §1 / §3.3 | Results came from the lab; PLAN_V2 still describes Colab shards | PLAN_V2 is historical; say so at its top |
| I-15 | Labels | "D1" is both a decision (dropout schedule) and Disha's task 1; "A1…" is a recommendation, an Aayush task and variant A | Write "decision D1" and "task D1" |
| I-16 | Readers of `--rand_pcd` | Gap G-a (Blender writes into the dataset) | Fix in A7, add to `check.sh` |
| I-17 | `factors.py` | Gap G-b (new IDs rejected, including `A_r*`/`C_r*` for noise2) | Extend in D4, plus #28 |
| I-18 | `probes.py` | Gap G-c (committed results depend on unmerged code) | Routine PR from `feat/probes` |
| I-19 | README.md | Gap G-d (upstream README; nothing on this project) | D13 |
| I-20 | PLAN_V2 §10.2 viva table | "Dropout as a variance penalty" is unmeasured (H4 not analysed) | D12 already plans to replace it |

## 7. Readiness for each new workstream

| WS | Needs before it can start | Exists | Missing |
| --- | --- | --- | --- |
| W4 views s1–2 | lab, `v1.0-exp` worktree, LLFF data | matrix, runner `--seeds` | Seed 0 lives in `/home/snu/dngs-dropout/outputs_views`; A3 writes to `~/dngs-v1.0/outputs_views`. The two need joining before D6 |
| W5 compensation | #27 rebased, `nc` mode, `matrix_comp.yaml` (96), check.sh grep, GPU smoke, Disha's approval, `v1.1-exp` + `prereg-v2` tags, factor table (#28) | #27 (clamp_aware, 48-run matrix) | everything else |
| W1/W3 DTU | ≥ 150 GB or range access, COLMAP env, 49/49 registration, DPT, idrmasks, `dtu_with_depth.tar` + md5, `train_dtu.py` port, `eval_dtu.py`, `matrix_dtu.yaml`, runner `dataset` key, CODEOWNERS, check.sh, smokes, `v1.2-exp` | `organize_dtu_dataset.sh`, `copy_mask_dtu.sh`, `run_dtu.sh`, `metrics_dtu.py`, DTU split in `dataset_readers.py:199-205` (25 test views) | all data and port work |
| W2/W3 Blender | data, DPT, tar + md5, non-SH port + SH guard, **G-a fix**, `matrix_blender.yaml` | 8-view split `[2,16,26,55,73,76,86,93]` and 25 eval views (`idx % 8`) in `dataset_readers.py:337-338`; `get_depth_map_for_blender.py` | the rest |
| W6 N1 | `prereg-v3`; eval renders for T1 (and views, DTU, Blender); `bias_variance.py` + self-test + fixtures | renders on the lab disk (and maybe in the Drive backup) | script, tarballs |
| W7 N2 | `prereg-v3` with the draw rule; `matrix_noise2.yaml`; factor table for `A_r*`/`C_r*`; audit protocol and sheet | block N data | the rest |
| W8 N4 | D3's per-run MSE | — | D3 |
| W9 N5 | `matrix_depthdose.yaml`; factor-table IDs `B_ds050`… | T1 A/C anchors | the rest |
| W11 prereg-v3 | the D1 checklist (10 items) | PLAN_V2 §5 style; #27's §5.1 as a model | the text and the tag |
| W12 writing | results; #23; Related Work doc | RESULTS.md, figures | F1 diagram, README, site |

## 8. To check outside the repo

| Where | Check | Why |
| --- | --- | --- |
| Lab PC | `tmux ls`; `ls ~/dngs-v1.0/outputs_views/llff_*v/*/*/seed[12]/metrics.json \| wc -l` | A3 progress |
| Lab PC | `df -h ~`; DTU download or `remotezip` state; `ls data/dtu/scan*/input \| head`; COLMAP registered counts | A5/A6 |
| Lab PC | `ls outputs/llff_3v/*/*/seed*/eval/ours_6000/renders \| head`; `eval_T1.tgz` + md5 | A4 |
| Lab PC | Can it stay on overnight Mon–Thu? Who opens the room? | A2; the run queue assumes overnight runs |
| Drive | `tar tzf results_full_nockpt.tgz \| grep -m3 renders/` | fallback for A4 |
| Disha | Blender download/DPT state; `prereg-v3` draft; any `bias_variance.py` work | D1–D3 |
| Disha | Related Work doc status (DS-13) | report |
| Course | Template, page limit, deadline time, AI-use disclosure | I-1, I-12 |
