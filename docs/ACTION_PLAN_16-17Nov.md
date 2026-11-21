# Action plan: Mon 16 Nov night → Wed 18 Nov lab visit

**For:** Aayush and Disha. **Goal:** use tonight and Tuesday morning, when nobody has the lab GPU, so that:
- the **Tuesday** lab visit can start every LLFF block: views seeds 1–2, compensation, depth-dose, noise2;
- the **Wednesday** visit can start DTU and Blender.

Background: [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) and [STATUS_AUDIT_16Nov.md](STATUS_AUDIT_16Nov.md). The authority is [EXECUTION_PLAN_15-21Nov.md](EXECUTION_PLAN_15-21Nov.md) (runs now stop Fri 12:00).

---

## 1. The one change to the plan: two code releases instead of one

The plan puts every new block behind the multi-dataset port (`v1.2-exp`). But compensation, depth-dose and noise2 are LLFF runs that only need matrix files. So:

| Release | Contents | Unlocks at the lab | Needed by |
| --- | --- | --- | --- |
| **`v1.1-exp`** | #27 (clamp-aware) rebased + `nc` mode + `matrix_comp.yaml` (96) + `matrix_depthdose.yaml` (96) + `matrix_noise2.yaml` (48, only if `prereg-v3` is tagged first) | 240 LLFF runs, about 5 h | **Tue, before Aayush leaves for the lab** |
| **`v1.2-exp`** | DTU + Blender port, `eval_dtu.py`, runner `dataset` key, matrix files, Blender point-cloud fix | DTU 225 + Blender 120, about 8–9 h | **Wed, before the lab visit** |

If noise2 isn't ready for `v1.1-exp`, it goes into `v1.2-exp`. Nothing else changes.

## 2. Decisions to settle on the phone tonight (10 min)

| # | Decision | Recommendation |
| --- | --- | --- |
| 1 | The two-release split above | Yes |
| 2 | `nc` = no compensation in **every pass the mask reaches** (plan task A1), or in the depth passes only (RESEARCH_ANALYSIS C4) | Every pass, as the plan says. It is coded that way tonight; prereg-v3 states it |
| 3 | prereg numbering. #27 carries PLAN_V2 §5.1 (tag `prereg-v2`), and prereg-v3 is §5.2 | Disha writes §5.2 on `main` now and tags `prereg-v3`. #27 merges Tuesday and is tagged `prereg-v2`. The tag dates still precede every result they govern, which is all that matters |
| 4 | noise2 draw: the exact list order | `random.Random(15112026).sample(["flower", "fortress", "horns", "leaves", "orchids", "room", "trex"], 3)`, alphabetical. The list and the code are written into prereg-v3 **before** anyone runs it |
| 5 | Same-day approvals Tuesday morning | Disha approves the `v1.1-exp` PR by about 11:00 |

---

## 3. Aayush

### Tonight (Mon), in order

**A-1. Record the DTU status in RUNLOG (5 min).**
- The go/no-go test (`run_dtu.sh` on scan8) hasn't run, because there was no data.
- Write: "16 Nov · Aayush · DTU go/no-go moved to Tue/Wed: data prep moved to the laptop (range download of 735 images, COLMAP on CPU). Blender is the required second dataset until DTU passes."

**A-2. DTU data on the laptop.** *Claude started this tonight:*
- The DTU server supports HTTP ranges, so only the 735 needed images are downloading: 15 scans × 49 views, light condition 3, out of the 129 GB zip. They go to `~/dtu_raw/Rectified/`; the log is `~/dtu_raw/extract.log`.
- COLMAP is installing as conda env `colmap`; the log is `~/colmap_env_install.log`.

When both finish (tonight or early Tue), run this on the CPU, no GPU:
```bash
cd ~/projects/DNGaussian
bash scripts/organize_dtu_dataset.sh ~/dtu_raw/Rectified        # -> data/dtu/scanN/input/ (49 PNGs each)
COLMAP=$(~/miniconda3/bin/conda run -n colmap which colmap)
for s in data/dtu/scan*; do
  ~/miniconda3/envs/dngaussian/bin/python convert.py -s $s --no_gpu --colmap_executable "$COLMAP" > $s/colmap.log 2>&1
  ~/miniconda3/bin/conda run -n colmap colmap model_analyzer --path $s/sparse/0 2>&1 | grep -i "registered images"
done
```
- **Every scan must show 49 registered images.** The DTU split picks views by index, so one missing image silently shifts the split.
- Rerun any failed scan once with `--camera PINHOLE`. If it still fails, it is **excluded** by the pre-registered rule; write that in RUNLOG.

**A-3. `v1.1-exp` code (branch `feat/clamp-aware`).** *Claude does this tonight; Aayush reviews in the morning:*
- rebase onto `main`;
- `--drop_mode nc`;
- `matrix_comp.yaml` (C_nc, D_nc, C_ca, D_ca × 8 scenes × seeds 0–2 = 96);
- `matrix_depthdose.yaml` (B/D at depth_scale 0.5 and 2 × 8 × 3 = 96);
- `check.sh` tests: `nc` never divides by (1 − r), and the dry-run counts.

**A-4. `v1.2-exp` code (branch `feat/multi-dataset`, stacked on A-3).** *Claude starts it tonight:*
- the `train_dtu.py` port;
- the `train_blender.py` port (non-SH path only, plus the SH guard);
- the Blender random point cloud written to a temp dir (audit gap G-a);
- the runner's `dataset` key;
- `experiments/eval_dtu.py`;
- `matrix_dtu.yaml` and `matrix_blender.yaml`;
- CODEOWNERS and `check.sh` additions;
- the stale AGENTS.md facts (deadline, lab GPU, layout).

**A-5. Lab kit for Tuesday.** *Claude writes `~/lab_kit/` tonight:*
- `lab_queue.sh` chains the blocks per shard with absolute paths;
- `lab_check.sh` counts finished runs and lists failures;
- `README.txt` has the exact commands.

Copy `~/lab_kit/` to a USB stick or Drive. **Never paste the commands from a rendered Markdown page**: it ate the `$` signs on 11 Nov.

**A-6. GPU smoke on the laptop (optional tonight, otherwise first thing Tue; 2–7 min).** The laptop may run GPU jobs at night. Check first that nothing else is on the GPU:
```bash
pgrep -af "train_|render.py|metrics.py"      # must print nothing
cd ~/projects/DNGaussian && git checkout feat/clamp-aware && bash scripts/gpu_smoke.sh
```
Post its last line (`GPU-SMOKE PASS <sha>`) on #27. It must run on the exact head that will be merged.

### Tuesday morning, before the lab

| Time | Step |
| --- | --- |
| 08:00 | Read Claude's diffs on `feat/clamp-aware`. If `prereg-v3` is tagged, add `matrix_noise2.yaml` with Disha's 3 scenes (`A_r1..r4, B_r1..r4, C_r1..r4, D_r1..r4`, seed 0 = 48 runs) |
| 08:15 | `bash scripts/check.sh`; `bash scripts/gpu_smoke.sh` (if not done overnight); push; run pr-gate (opus reviewer, experiment + gate class) |
| ~10:00 | Fix any findings. Message Disha: "#27 ready at `<sha>`, please approve" |
| ~11:00 | After her approval: `pr_ready.py`, merge with `--match-head-commit`, then `git tag -a v1.1-exp` and `git tag -a prereg-v2` on the merge commit, push both, add a RUNLOG line |
| Before leaving | `~/lab_kit/` on USB; the LLFF data is already on the lab PC |
| **Lab** | See §5 |
| Evening | Continue the `v1.2-exp` PR: DTU smoke (scan8) and Blender smoke (lego) on the laptop at 1100 iterations; DPT for DTU (laptop GPU, minutes); idrmasks; `dtu_with_depth.tar` + md5 to Drive. Ask Disha for same-evening approval |

### Wednesday morning
- Merge `v1.2-exp`, tag it, add a RUNLOG line.
- Copy `dtu_with_depth.tar` and `blender_with_depth.tar` to the lab (USB or Drive) and check the md5s there.
- Lab: DTU → Blender (§5).

---

## 4. Disha

### Tonight (Mon), in order

**D-1. `prereg-v3` (3–4 h, docs-only PR, no reviewer). The most important thing tonight: until it is tagged, no new statistic may be computed and the noise2 runs can't be written.**

1. Branch from `main`: `git checkout main && git pull && git checkout -b docs/prereg-v3`.
2. In `docs/PLAN_V2.md`, after §5 (before §6), add **"### 5.2 Addendum (16 Nov, tag `prereg-v3`, before any result it governs)"** with these items. Plan task D1 has the full wording.
   1. **Datasets and protocol.**
      - DTU: 15 scans, train views 25/22/28, 25 test views, masked metrics, `-r 4`, the `run_dtu.sh` flags.
      - Blender: 8 objects, views `[2,16,26,55,73,76,86,93]`, 25 eval views, `-r 2`, run_blender.sh setting 1 with soft depth on, non-SH path.
      - Frozen hyperparameters.
      - **Exclusion rule:** a DTU scan with fewer than 49 registered images after one PINHOLE retry is excluded.
      - **New tonight:** DTU poses come from COLMAP on CPU (`--no_gpu`). DPT runs once per dataset, and its tar md5 is recorded.
   2. **Replication (X4):** same contrasts, δ = 0.25 dB and Holm families as LLFF. "Replicates" = same sign and the CI overlaps the LLFF estimate.
   3. **X5:** D − B (DTU) > D − B (LLFF), one-sided. **H-PD:** LLFF ≤ DTU ≤ Blender in D − B (exploratory).
   4. **N1 / BV1–BV6:** the estimators (b̂² = (m̄ − y)² − s²/n, σ̂² = s², n = 3), pixel MSE as the primary metric (masked on DTU), and each decision rule.
   5. **N4:** I on PSNR, MSE, RMSE and SSIM; "robust" = same sign on all.
   6. **N5:** I_D at depth_scale 0.5 / 1 / 2; prediction |I_D| < δ.
   7. **G6 / N2(a):** same-seed SD ≈ between-seed SD; noise class A, C > B, D.
      - **Write the draw rule here, before you run it:** `random.Random(15112026).sample(["flower","fortress","horns","leaves","orchids","room","trex"], 3)`.
   8. **N2(b) audit protocol:** inclusion criteria, coding sheet columns, noise-band rule, κ on 2 shared papers, κ < 0.7 ⇒ "unreliable".
   9. **Compensation:**
      - **`nc` = survivors keep opacity × 1 (no 1/(1 − r)) in every pass the mask reaches; same schedule and the same random draw as uniform.**
      - Contrasts C_nc − C, D_nc − D, C_ca − C, D_ca − D; Holm across 4; TOST δ = 0.25.
   10. **Exploratory disclosure:**
       - the LLFF results (T1/P/G/N/V seed 0) were seen before this addendum;
       - the 15 Nov MSE-scale and heterogeneity checks are post hoc;
       - nothing in DTU, Blender, comp, depth-dose, noise2 or views seeds 1–2 had been run.
3. Check, commit and open the PR:
   ```bash
   bash scripts/check.sh
   git add docs/PLAN_V2.md && git commit -m "PLAN_V2 §5.2: prereg-v3"
   git push -u origin docs/prereg-v3
   gh pr create --repo aayxsh23/dngs-dropout --base main --title "prereg-v3: PLAN_V2 §5.2 addendum" --body-file <file>
   ```
   Then go through pr-gate: the docs-only class needs no reviewer.
4. After the merge, tag the merge commit and push:
   ```bash
   git tag -a prereg-v3 -m "PLAN_V2 §5.2" && git push origin prereg-v3
   ```
5. **Only now** run the draw and send Aayush the 3 scenes:
   ```bash
   python3 -c 'import random; print(random.Random(15112026).sample(["flower","fortress","horns","leaves","orchids","room","trex"], 3))'
   ```

**D-2. Blender data (start the download first, then do D-1 while it runs).**
1. Download `nerf_synthetic.zip` from the NeRF data link (https://www.matthewtancik.com/nerf → "Data") and extract it to `data/nerf_synthetic/<object>/`.
2. Check: each of the 8 objects has `transforms_train.json` (100 frames) and `transforms_test.json` (200 frames).
3. DPT once. It needs a GPU for a few minutes; Colab or Kaggle is fine because this is data prep, not an experiment:
   ```bash
   cd dpt && python get_depth_map_for_blender.py -r ../data/nerf_synthetic
   ```
   Check that the output folder name matches what `train_blender.py` reads.
4. Note whether the test folders hold GT depth (`r_*_depth_*.png`). It is optional, for H-PD.
5. Freeze it and share it:
   ```bash
   tar cf blender_with_depth.tar -C data nerf_synthetic && md5sum blender_with_depth.tar
   ```
   Upload to Drive `sparse3dgs/data/`, and send Aayush the md5 **by Tue evening**.

**D-3. Check whether the T1 renders are already on Drive (10 min).**
```bash
tar tzf results_full_nockpt.tgz | grep -m3 "eval/ours_6000/renders/"
```
`results_full_nockpt.tgz` is in Drive `sparse3dgs` (md5 `2ea56c510519ff48a8a66fc3bae548ae`). If the renders are there, N1 on T1 can start Tuesday without waiting for the lab tarball. Tell Aayush either way.

### Tuesday (no GPU needed at all)

| Time | Step |
| --- | --- |
| Morning | **Approve Aayush's `v1.1-exp` PR the same morning.** Read the diff (the `nc` branch in `train_llff.py`, the matrix files, the `check.sh` tests), check the `GPU-SMOKE PASS <head sha>` comment and the reviewer's Approve, then approve on GitHub. He can't go to the lab with the full queue until then |
| Day | **D3 `experiments/bias_variance.py`**, per plan task D3: self-test with synthetic bias and noise, fixtures, the `check.sh` line (gate class: Aayush approves). Run it on real T1 renders only after `prereg-v3` is tagged |
| Day | **D4, including audit gap G-b:** `factors.py` must accept the new IDs, or `stats.py` stops on them. That means `C_nc`, `D_nc`, `C_ca`, `D_ca` (issue #28), `B_ds050`, `D_ds050`, `B_ds200`, `D_ds200`, `A_r1..4`, `C_r1..4`. Also the `aggregate.py` dataset column (`(llff|dtu|blender)_(\d+)v`, per-dataset scene lists), with the LLFF outputs kept byte-identical |
| Day | **D5 N4** on T1 once D3 runs |
| Any gap | Audit (plan task D8): finish the protocol (it goes into prereg-v3) and start coding your half of the papers. Reading published tables needs no data and no GPU |
| Evening | Approve the `v1.2-exp` PR if it is ready. Send the Blender tar md5 |

### Writing that can start now (any free hour this week)
- Related Work: finish it from the existing Claude Doc (DS-07/13).
- Results, LLFF part: the T1 table, contrasts, verdict, dose, noise, views (seed 0), efficiency, geometry, probes. All numbers are in [results/RESULTS.md](../results/RESULTS.md) and are final.
- Figure scripts F1–F9 on fixtures, so Friday is only a rerun on real data.

---

## 5. Lab visits

All paths are absolute. Every output goes under `/home/snu/dngs-dropout/outputs_*`, so views seeds 1–2 land next to seed 0. A new git worktree has **no `data/` folder**, so always pass `--data_root /home/snu/dngs-dropout/data/<dataset>`.

**One-time setup (Tuesday):**
```bash
cd /home/snu/dngs-dropout && git fetch --tags origin
git worktree add ~/dngs-v1.0 v1.0-exp
git worktree add ~/dngs-v1.1 v1.1-exp          # after the tag exists
mkdir -p /home/snu/dngs-dropout/output
```

**Every visit:**
1. `pgrep -af "train_|render.py|metrics.py|run_matrix"`: should be empty, or show only your own queue.
2. `bash lab_check.sh`: counts per block and failed runs.
3. Dry run, then start the queue:
   ```bash
   bash lab_queue.sh <day> --dry_run
   for i in 0 1 2; do tmux new -d -s q$i "bash lab_queue.sh <day> $i"; done
   ```
4. **Watch the first 2–3 runs finish** (about 10 min) before leaving.
5. Lock the screen; **don't log out**. Check that the PC won't sleep or reboot overnight.
6. Add a RUNLOG line: block, tag, start time, `shared_gpu: true`.

| Visit | `<day>` queue (each shard runs these blocks one after another) | Tags needed | Expected |
| --- | --- | --- | --- |
| Tue | `tue`: views s1–2 (240) → comp (96) → depth-dose (96) → noise2 (48, if present) | `v1.0-exp`, `v1.1-exp` | ~9.5 h, done Tue night |
| Wed | `wed`: DTU (225) → Blender (120), then anything Tuesday left (the runner skips finished runs) | `v1.2-exp` + data tars | ~8–9 h |
| Thu | `thu`: everything again (only failed or missing runs actually run) | — | buffer |
| Fri ≤ 12:00 | Last reruns, then stop. Copy all `outputs_*` (no checkpoints) and eval tarballs to Drive with md5s | — | runs stop 12:00 |

---

## 6. Handoffs

| From → to | What | By |
| --- | --- | --- |
| Disha → Aayush | `prereg-v3` tag + the 3 noise2 scenes | Mon night (at the latest Tue 08:00) |
| Disha → Aayush | Are T1 renders in the Drive backup? (D-3) | Mon night |
| Aayush → Disha | `v1.1-exp` PR for approval | Tue ~10:00 |
| Disha → Aayush | Approval | Tue ~11:00 |
| Disha → Aayush | `blender_with_depth.tar` md5 | Tue evening |
| Aayush → Disha | `v1.2-exp` PR for approval | Tue evening |
| Disha → Aayush | Approval | Wed morning, before the lab |
| Aayush → Disha | `eval_T1.tgz` / `eval_views.tgz` (+ md5), only if D-3 says the renders aren't on Drive | Wed |
| Both | 21:30 sync: done / blocked / tomorrow, plus RUNLOG lines | daily |

## 7. Status tonight (Claude, Mon 16 Nov, ~00:15 IST)

Everything below is **local only**: nothing has been pushed, tagged or opened as a PR.

| Item | Where | State |
| --- | --- | --- |
| A-3 `v1.1-exp` code | branch `feat/clamp-aware` (worktree `~/projects/dngs-ca`), commit `ffe8e65` on top of #27 rebased onto `main` | Done. `--drop_mode nc` (same draw as uniform, `(mask > 0).float()`); `matrix_comp.yaml` (96: C_ca, D_ca, C_nc, D_nc); `matrix_depthdose.yaml` (96); check.sh step 9 accepts `nc`; new step 19 (no rescaling in the `nc` branch, mutation-tested; dry-run counts); AGENTS variant row; PLAN_V2 §3.1 rows. `check.sh`: ALL PASSED. **Needs:** GPU smoke on this head, `matrix_noise2.yaml` once the draw exists, pr-gate, Disha |
| A-4 `v1.2-exp` code | branch `feat/multi-dataset` (worktree `~/projects/dngs-md`), commit `71e3d39`, stacked on A-3 | Done; not yet run on a GPU. Ports `train_dtu.py` and `train_blender.py` (non-SH): seed, depth_scale, D1 mask routing, diag.csv/train_stats.json via the train_llff helpers, config.json; uniform mode only; the DTU background-alpha pass never sees the mask; the Blender `--use_SH` guard; Blender cloud in a temp dir; runner `dataset` key + `{scene}`, `{dataset}`, `{data_root}`; `experiments/eval_dtu.py`; `matrix_dtu.yaml` (225), `matrix_blender.yaml` (120); CODEOWNERS; check.sh steps 4, 15, 20 (mutation-tested); AGENTS.md deadline, lab PC, layout. **Needs:** DTU + Blender smokes, GPU smoke, pr-gate, Disha |
| A-5 lab kit | `~/lab_kit/` (`lab_queue.sh`, `lab_check.sh`, `README.txt`) | Done; dry-run tested against local copies of the tags |
| A-2 DTU data | `~/dtu_raw/Rectified/` (5 parallel downloads, logs `~/dtu_raw/extract_*.log`); COLMAP env `colmap` (4.2.1) + wrapper `~/dtu_prep/colmap`; prep script `~/dtu_prep/prep_dtu.sh` | Downloading. When it finishes, run `bash ~/dtu_prep/prep_dtu.sh ~/dtu_raw/Rectified` from the repo root (CPU only). It organizes, runs COLMAP, retries PINHOLE once, and writes `data/dtu/registration.txt` |
| probes.py on main | branch `chore/probes-script` (worktree `~/projects/dngs-probes`), commit `a81f943` | Done: the exact file from 9f8d55f (md5 matches RESULTS.md). Routine PR |
| RUNLOG back-fill | this branch (`docs/research-guide`) | Done: 3–16 Nov |

**Two design choices in A-4 for review:**
- The DTU background-alpha pass (`alpha[bg_mask]²`) gets no mask, in every variant.
- The Blender matrix keeps setting 1's `--test_iterations`/`--save_iterations`. The host skips densification on those iterations, so leaving them out would change the protocol.
