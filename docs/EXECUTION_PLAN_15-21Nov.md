# Execution plan, Sun 15 – Sat 21 Nov 2026
## "Where does sparse-view 3DGS error come from? A bias–variance account of depth priors and Gaussian dropout across three datasets"

---

## 0. TL;DR

**The approach.** We combine the 15 Nov memo with the new findings:

1. **Two or more datasets (mandatory):**
   - LLFF (done) + **DTU 3-view (required)** + **Blender 8-view** (third dataset, and the fallback if DTU fails).
   - Same five variants A–E, 3 seeds, and a **protocol frozen from LLFF**.
2. **The central new contribution: an error budget (N1).** Decompose each variant's error into bias² and run-to-run variance, on every dataset and view count. It explains *why* depth and dropout stack: the predictions BV1–BV6, the "ceiling", and the 9-view crossover.
3. **Rigor contribution (N2).** A measured noise floor, a power calculator, and an audit of published ablation gains against that noise.
4. **Defensibility fixes:**
   - N4: the interaction reported on several scales.
   - N5: a depth-dose surface.
   - G6: the noise floor extended beyond fern.
5. **Kept from the memo:**
   - the compensation study on LLFF (#27 + a no-compensation mode);
   - views seeds 1–2, which N1 also needs.
6. **Dropped:**
   - variant F; a separate co-adaptation score;
   - DTU compensation arms; DAv2;
   - the init moderator; region-stratified error; a second host.
   - (These go to Future Work.)

**The split**
- **Aayush: pipeline and lab owner.** Data prep, training-code port, runner, lab runs, compensation, half the audit, Method and Setup.
- **Disha: analysis and presentation owner.** Preregistration, all analysis scripts, statistics, figures, half the audit, Results, website and README.

**Gates that cannot slip**
| When | Gate | If it fails |
|---|---|---|
| Sun 23:59 | `prereg-v3` tagged | No new analysis is opened until it exists |
| Mon 20:00 | DTU go/no-go (scan8 trains, 49/49 images registered by COLMAP) | Blender is promoted to the required second dataset |
| Tue 23:59 | `v1.2-exp` (multi-dataset port) tagged | No DTU/Blender runs; escalate the same night |
| Thu 12:00 | DTU throughput on plan | Blender drops to 2 seeds |
| Fri 12:00 | All runs stop | Missing runs are reported as missing |
| Fri 20:00 | Analysis freeze | Only text changes afterwards (report: Fri night + Sat) |

*Changed Mon 16 Nov (Aayush): the run stop moved from Thu 12:00 to Fri 12:00, the analysis freeze from Thu 23:59 to Fri 20:00, and the DTU throughput check from Wed 12:00 to Thu 12:00. Thursday is now a lab buffer day. The report is written Fri night and Sat.*
| Sat 20:00 | Submission | 4 h buffer to the 23:59 deadline |

---

## 1. What we are building, and why this mix

| ID | Workstream | Source | Why it is in | Owner |
|---|---|---|---|---|
| W1 | DTU 3-view data + port | memo §4, new protocol | Mandatory second dataset; DNGaussian's own benchmark (published B = 18.91); same non-SH renderer | Aayush |
| W2 | Blender 8-view data + port | new | Third dataset or fallback; tests prior domain shift (DPT out of domain on synthetic scenes) | Aayush (code), Disha (data) |
| W3 | Factorial A–E × 3 seeds on DTU and Blender | memo + new | External validation under a frozen protocol | Aayush (runs) |
| W4 | Views seeds 1–2 (2/6/9 views) | memo | Confirms or refutes the 6-view interference; **feeds N1-BV4** | Aayush (runs) |
| W5 | Compensation study on LLFF (#27 + `nc`) | memo | The most novel memo idea; code is about 80 % done | Aayush |
| W6 | **N1 error budget** | **new (core)** | The explanatory model; turns the nulls into a predictive account | Disha |
| W7 | **N2 noise floor + power + audit**, with G6 runs | **new (core)** | A distinctive rigor contribution relevant to the whole field | Disha (model), both (audit), Aayush (G6 runs) |
| W8 | N4 scale-robust interaction | new (must) | "Additive" holds only in dB today | Disha |
| W9 | N5 depth-dose surface | new | Is additivity an artefact of DNGaussian's depth weight? | Aayush (runs), Disha (analysis) |
| W10 | N3 spectral budget | new (optional) | Explains PSNR-additive vs SSIM-sub-additive | Disha, if time |
| W11 | `prereg-v3` | memo + new | Keeps every new result credible | Disha |
| W12 | Report, website, README, viva prep | — | Marks | Both |

**Why this mix and not something else**
- The memo's direction A gives novelty (compensation) and a dataset (DTU), but no *explanation* of our null interaction results.
- N1 supplies that explanation, at zero GPU cost and on data we already have.
- N2 turns our unusual noise measurements into a contribution nobody else in the class, or in this subfield, has.
- Blender protects the mandatory dataset requirement against a DTU failure and adds a prior-domain-shift test.
- Everything else was cut on a scored basis. The tables are in the analysis section of the plan history.

---

## 2. Ground rules (from AGENTS.md / CLAUDE.md, apply to both)

1. **Python 3.7 syntax** in anything the training pipeline imports: no walrus, no `list[int]`, no `dict | dict`.
2. **Invariants.**
   - Dropout only in the training loop.
   - Defaults reproduce upstream; no mask is created at `drop_rate 0`.
   - The output layout is frozen: we may *add* paths and keys, never rename them.
   - One GPU type per seed, with `shared_gpu: true` on parallel runs.
   - Experiments run **only from a tag**.
   - Same data everywhere, verified by md5.
3. **New-dataset depth.** DPT runs **once** per new dataset. The archive is frozen with md5 (`dtu_with_depth.tar`, `blender_with_depth.tar`) and the decision is recorded in `docs/PLAN.md`. Nobody reruns it after that.
4. **Never commit** `data/`, `output*/`, checkpoints, `.ply` files, archives or tokens. Stage files by name.
5. **Every merge goes through pr-gate.**
   - Experiment class (training, eval, runner, yaml): GPU smoke + the other teammate's approval + a new tag + a RUNLOG entry.
   - Gate class (`.github/`, `scripts/check.sh`, …): the other teammate's approval.
   - Docs-only: no reviewer.
   - **Same-day approvals are part of this plan.**
6. **The laptop** has a 4 GB GPU. Short jobs (smoke, DPT, about 1–7 min) are fine during the day. Everything long runs at the lab in `tmux`. Check `pgrep -af "train_|render.py|metrics.py"` before starting any GPU job.
7. **Integrity.**
   - Nobody opens a render or computes a new statistic for N1/N3/N5/G6/DTU/Blender before `prereg-v3` is tagged.
   - The MSE-scale and heterogeneity numbers computed on 15 Nov are **exploratory** and labelled as such in the report.
8. **Daily sync at 21:30.** Each person posts: done / blocked / tomorrow, plus one RUNLOG line per block started or finished.

---

## 3. Master timeline

| Day | Aayush | Disha | Lab GPU (A4000, 2–3 parallel slots) |
|---|---|---|---|
| **Sun 15** | A1 rebase #27 + `nc` mode; A2 lab prep checklist; review D1 | **D1 `prereg-v3`** (tag tonight); D2 start Blender download | — |
| **Mon 16** | Lab: **A3** views seeds 1–2 start; **A4** eval tarball; **A5 DTU acquisition**; **A6 go/no-go 20:00** | **D2** Blender DPT + tar; **D3** `bias_variance.py` + fixture; D4 dataset factor (start) | W4 views (240 runs) |
| **Tue 17** | **A7 multi-dataset port PR** → smoke → review → **tag `v1.2-exp`**; A8 #27 → `v1.1-exp`; start the queue | Same-day **review/approval** of A7 and A8; **D3** N1 on T1; **D5** N4; D4 finish | W5 compensation (96), then W3 DTU (225) in the evening |
| **Wed 18** | A9 queue monitoring + nightly sync; **A10 audit half**; A12 Method/Setup draft | **D6** N1 on views (BV4); **D7** variance components + power; DTU tables as runs land; **D8 audit half** | DTU → Blender (120) → depth-dose (96) → noise2 (48) |
| **Thu 19** | A9 lab visit (buffer: late blocks, reruns); A10 κ subset; A12 Method/Setup drafts | **D6** N1 on views and DTU as it lands; D7/D8 finish; figure scripts on the data landed so far | Late blocks and reruns (overnight) |
| **Fri 20** | Lab: final check, last reruns, **runs stop 12:00**; A11 final sync; A12 Method / Setup / Limitations / appendix | **D6** BV6; D9 pooled analysis; **D10 figures**; D11 N3 if time; **freeze 20:00**; D12 Results from Fri night | Reruns only, until 12:00 |
| **Sat 21** | A12 final; A13 number cross-check; submit 20:00 | D12 final; **D13 website + README**; site live | — |

---

## 4. Aayush: detailed tasks

### A1 · Sun · Rebase #27 and add the no-compensation mode (2–3 h)
1. Rebase: `git fetch origin && git checkout feat/clamp-aware && git rebase origin/main`. Resolve conflicts, then run `bash scripts/check.sh`.
2. Add the `nc` mode next to #27's existing modes, following #27's own flag pattern (it already adds a mode flag for `clamp_aware`):
   - `nc` drops Gaussians with the same schedule `r = drop_rate * iteration / 10000`;
   - but multiplies survivors' opacity by **1**, not `1/(1−r)`.
   - It lives only in the training loop (invariant 1). At `drop_rate 0` no mask is created.
3. Matrix (`experiments/matrix_comp.yaml`, or extend #27's block-CA file):
   - Variants `C_nc`, `D_nc`, `C_ca`, `D_ca` (depth_scale/drop_rate as in C/D; mode via #27's key).
   - 8 LLFF scenes, seeds 0–2, `keep_ckpt_seeds: []`.
   - Make sure the variant-key assert in `scripts/check.sh` (step at line ≈120) accepts #27's extra key, as #27 already arranged.
4. Add a deterministic check in `scripts/check.sh`: grep that the `nc` branch never divides by `(1 - r)`. This touches a gate file, so Disha approves.
5. Push. Do **not** merge yet (A8).

**Done when:** check.sh passes and `run_matrix.py --matrix experiments/matrix_comp.yaml ... --dry_run` lists 96 runs.

### A2 · Sun · Lab preparation checklist (30 min)
- Confirm lab access for Mon–Fri: who opens the room, hours, and whether the PC can stay on overnight.
- Pack: the Drive link to `llff_with_depth.tar`, the repo URL, the DNGaussian README DTU section, and the masks Drive link.
- Decide the lab repo layout. One **git worktree per tag** keeps runs tied to tags:
  `git worktree add ~/dngs-v1.0 v1.0-exp` (later `~/dngs-v1.1`, `~/dngs-v1.2`).

### A3 · Mon (first thing) · Start views seeds 1–2 (20 min hands-on, about 4.5 h wall)
1. In `~/dngs-v1.0` (tag `v1.0-exp`), check which matrix produced seed 0. Read `matrix`/`code_tag` in any `results/raw/views/llff_6v/*/*/seed0/config.json`, or use `matrix_views.yaml` if it isn't recorded.
2. Dry run first:
   `python experiments/run_matrix.py --matrix experiments/matrix_views.yaml --data_root data/llff --out_root outputs_views --seeds 1 2 --dry_run` → expect 240 runs.
3. Launch 3 slots (`tmux`):
   ```bash
   for i in 0 1 2; do tmux new -d -s views$i "cd ~/dngs-v1.0 && ~/miniconda3/envs/dngaussian/bin/python experiments/run_matrix.py --matrix experiments/matrix_views.yaml --data_root data/llff --out_root outputs_views --seeds 1 2 --shard $i/3 --shared_gpu > output/views_s12_shard$i.log 2>&1"; done
   ```
4. RUNLOG line: block V seeds 1–2, tag, matrix, start time, `shared_gpu: true`.

**Done when** 240 `metrics.json` exist (`ls outputs_views/llff_*v/*/*/seed[12]/metrics.json | wc -l`).

### A4 · Mon · Eval-render tarball for Disha (30 min)
N1 needs every T1 seed's renders and GT. `cleanup()` keeps `eval/`.
```bash
cd <lab outputs parent>
find outputs -path '*/eval/*' -type d -name renders | sed 's#/renders$##' > eval_dirs.txt   # each eval/ours_<iter>
tar czf eval_T1.tgz -T eval_dirs.txt && md5sum eval_T1.tgz > eval_T1.tgz.md5
```
- Repeat for `outputs_views` once A3 finishes (Tue morning) → `eval_views.tgz`.
- Upload both to the team Drive. Expected size is about 0.5–2 GB; if it is bigger, drop the `depth/` subfolders.

**Done when** Disha confirms the md5 matches on her side.

### A5 · Mon · DTU acquisition and preprocessing (4–6 h, mostly waiting)
1. **Disk:** `df -h ~` needs ≥ 150 GB free for the full-zip route, or ≥ 10 GB for the range route.
2. **Range test** (copy the "Rectified" link from https://roboimagedata.compute.dtu.dk/?page_id=36):
   - Run `curl -sI <zip-url>` and look for `Accept-Ranges: bytes`.
   - If present: in a throwaway venv, `pip install remotezip`, then list the members (`remotezip -l <url> | head`). Extract only the 15 scans' `*_3_r5000.png` (light condition 3). **List first, then adapt the path pattern:** `scripts/organize_dtu_dataset.sh` expects `<rectified>/<scan>/*_3_r5000.png`.
   - If not present: start the full 123 GB download immediately (`wget -c` in `tmux`) and continue with Blender support work meanwhile.
3. **Organize:** `bash scripts/organize_dtu_dataset.sh <rectified_path>` → `data/dtu/scanN/input/` (49 PNGs each, 15 scans).
4. **COLMAP without sudo:**
   ```bash
   conda create -n colmap -c conda-forge colmap -y
   COLMAP=$(conda run -n colmap which colmap)
   ~/miniconda3/envs/dngaussian/bin/python convert.py -s data/dtu/scan8 --no_gpu --colmap_executable "$COLMAP"
   ```
   - **Critical check:** **all 49 images must be registered.** The DTU split in `scene/dataset_readers.py:199-205` selects views by *index* (train 25, 22, 28). A dropped image shifts every index and silently corrupts the split.
   - Check with `conda run -n colmap colmap model_analyzer --path data/dtu/scan8/sparse/0` (registered images = 49).
   - A scan with fewer than 49 is re-run once with `--camera PINHOLE`. If it still fails, it is **excluded by the pre-registered rule** and recorded.
5. Run the other 14 scans in a loop. COLMAP is CPU work, so it can run beside the GPU jobs.
6. **DPT, once:** `cd dpt && ~/miniconda3/envs/dngaussian/bin/python get_depth_map_for_llff_dtu.py -r ../data/dtu -b DTU`. It needs internet the first time (torch.hub MiDaS).
7. **Masks:** download the RegNeRF `idrmasks` (Drive link in the README's DTU section) to `data/dtu/submission_data/idrmasks/`.
8. **Freeze:** `tar cf dtu_with_depth.tar -C data dtu && md5sum dtu_with_depth.tar`. Upload to Drive. Send the md5 to Disha for PLAN and prereg.

### A6 · Mon 20:00 · DTU go/no-go (20 min)
- **Test:** upstream `bash scripts/run_dtu.sh data/dtu/scan8 output/dtu_go/scan8 0`. This is unported code (variant B behaviour), so it tests the *data*, not our port.
- **Go:** it trains and renders end to end, masked PSNR is plausible (about 16–20), and ≥ 13 of 15 scans are registered at 49/49.
- **No-go:** Blender becomes the required second dataset. Tell Disha at once; DTU becomes "attempted, documented" in Limitations.
- **Last-resort fallback** (only if COLMAP itself won't run): write the GT DTU cameras into COLMAP text format plus a bounds-only `points3D.txt`. It must be documented as a protocol deviation. Decide at the call; don't improvise at the lab.

### A7 · Tue · Multi-dataset port PR, `feat/multi-dataset` (6–8 h)
**Approach.** Make the reference diff with `git diff upstream/main -- train_llff.py > $SCRATCH/llff.diff`, then mirror each hunk by hand.

**`train_dtu.py`** (currently 0 of our flags):
- seed flag → `safe_state(silent, seed)`;
- `depth_scale` multiplier, and skipping both depth passes when it is 0;
- per-iteration mask with schedule `drop_rate_now = opt.drop_rate * iteration / 10000`, created **only if drop_rate > 0**;
- `depth_render` routing (shared / clean / hard_only / soft_only), exactly as in `train_llff.py`;
- passing `drop_comp` / `depth_comp` into `render`, `render_for_depth`, `render_for_opa`;
- `diag.csv`, `train_stats.json`, `config.json` writing; `load_checkpoint()`; the temp-dir random point cloud (already in `dataset_readers.py`).

**`train_blender.py`:**
- The same port, but **only in the non-SH training function** (around line 44, `GaussianModel`).
- Add a guard: if `--use_SH` and (`drop_rate > 0` or `depth_render != shared`), raise `SystemExit("dropout is only ported to the non-SH path")`.

**`experiments/run_matrix.py`:**
- Optional top-level yaml key `dataset` (default `"llff"`) → run dir `<out_root>/<dataset>_<n>v/<variant>/<scene>/seed<k>/` (`run_matrix.py:146`).
- Add `{scene}` and `{dataset}` template variables.
- Existing yaml files must give byte-identical paths (check below).

**`experiments/eval_dtu.py`** (new):
- Per run: copy masks from `idrmasks/<scan>/mask/` into `<work>/mask/` with the same renaming as `scripts/copy_mask_dtu.sh`.
- Assert the mask count equals the number of eval renders (25).
- Compute **masked** PSNR / SSIM / SSIM_sk / LPIPS as `metrics_dtu.py` does (resize /4, composite).
- Write **`results_eval.json`** as `{"ours_<iter>": {"PSNR","SSIM","SSIM_sk","LPIPS","masked": true}}`, the frozen keys plus one added key.

**Matrix files** (all `keep_ckpt_seeds: [0]` except noise2):
| File | Content |
|---|---|
| `matrix_dtu.yaml` | `dataset: dtu`; 15 scans; seeds 0–2; A–E; `train_flags` = the `scripts/run_dtu.sh` flags; steps: `train_dtu.py --dataset DTU …` → `render.py -s {scene_dir} --model_path {work} -r 4 --skip_train --skip_test` → `experiments/eval_dtu.py …`. Check that `render.py` picks up `--dataset DTU` from `cfg_args`; if not, pass it |
| `matrix_blender.yaml` | `dataset: blender`; chair, drums, ficus, hotdog, lego, materials, mic, ship; seeds 0–2; A–E; **one uniform protocol**: `-r 2 --eval --n_sparse 8 --rand_pcd --iterations 6000 --lambda_dssim 0.2 --white_background --densify_grad_threshold 0.0005 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 --densify_from_iter 500 --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 --hard_depth_start 0 --split_opacity_thresh 0.1 --error_tolerance 0.001 --scaling_lr 0.005 --shape_pena 0 --opa_pena 0 --scale_pena 0` (setting 1 of `run_blender.sh` **with soft depth left on**); steps: `train_blender.py` → `render.py -r 2 --skip_train --skip_test` → `metrics.py --model_paths {work}` |
| `matrix_depthdose.yaml` | LLFF; `B_ds050 {depth_scale: 0.5, drop_rate: 0.0}`, `D_ds050 {0.5, 0.2}`, `B_ds200 {2, 0.0}`, `D_ds200 {2, 0.2}` (all `depth_render: shared`); seeds 0–2; T1 flags (96 runs) |
| `matrix_noise2.yaml` | LLFF; **3 scenes drawn by the pre-registered rule** (D1: `random.Random(15112026).sample` of the 7 non-fern scenes); `A_r1..r4, B_r1..r4, C_r1..r4, D_r1..r4`; seed 0 (48 runs; with T1 seed 0 that makes 5 replicates each) |

**`.github/CODEOWNERS`:** add `/train_dtu.py`, `/train_blender.py`, `/metrics_dtu.py` and `/experiments/eval_dtu.py` to the `[experiment]` section, so the gate treats them as training/eval code. This is a gate-class file; Disha approves.

**`scripts/check.sh` additions:**
- (a) The D1 schedule grep for `train_dtu.py` and `train_blender.py`, mirroring line ≈232.
- (b) `eval_dtu.py` added to the eval paths that must never contain `drop_comp|drop_rate|Dropout` (line ≈59).
- (c) Runner dry runs: `matrix.yaml` still gives `llff_3v/...`; `matrix_dtu.yaml` gives `dtu_3v/...`; `matrix_blender.yaml` gives `blender_8v/...`.
- (d) The SH guard exists in `train_blender.py`.

**Smoke:**
- `bash scripts/gpu_smoke.sh` (the gate requires it; LLFF fern) on the clean head.
- **Plus** runner smokes `matrix_smoke_dtu.yaml` (scan8) and `matrix_smoke_blender.yaml` (lego): A–E, 1100 iterations, `--allow_untagged`, on the lab PC or the laptop.
- Post their PSNR table as a PR comment.
- Sanity: B > A, the run finishes, and `results_eval.json` keys are present.

**Gate:** pr-gate → `pr-reviewer` (opus; experiment + gate class) → fix → Disha approves → merge with the pinned SHA → `git tag -a v1.2-exp -m "multi-dataset port: DTU, Blender non-SH, runner dataset key"` and `git push origin v1.2-exp` → RUNLOG entry.

### A8 · Tue · Merge #27 → `v1.1-exp` (2 h, interleaved with A7)
- Run `bash scripts/gpu_smoke.sh` on the #27 head, then pr-gate (opus) and Disha's approval. Merge.
- Tag `v1.1-exp` and start the compensation block (A9).
- If A7 merges first, rebase #27 on it and tag only `v1.2-exp`. One tag carrying both is fine; record it in RUNLOG.

### A9 · Tue evening → Fri 12:00 · Run queue at the lab
- Work from `~/dngs-v1.2` (or `~/dngs-v1.1` for compensation).
- Always `--dry_run` first, then 3 `tmux` slots per block with `--shard i/3 --shared_gpu`.

| Order | Block | Matrix | Out root | Runs | Wall (3 slots) |
|---|---|---|---|---|---|
| 1 | Compensation | `matrix_comp.yaml` | `outputs_comp` | 96 | ~2 h |
| 2 | DTU | `matrix_dtu.yaml` | `outputs_dtu` | 225 | ~5–6 h |
| 3 | Blender | `matrix_blender.yaml` | `outputs_blender` | 120 | ~2.5–3 h |
| 4 | Depth-dose | `matrix_depthdose.yaml` | `outputs_depthdose` | 96 | ~2 h |
| 5 | Noise2 | `matrix_noise2.yaml` | `outputs_noise2` | 48 | ~1 h |

Template, one block:
```bash
for i in 0 1 2; do tmux new -d -s dtu$i "cd ~/dngs-v1.2 && ~/miniconda3/envs/dngaussian/bin/python experiments/run_matrix.py --matrix experiments/matrix_dtu.yaml --data_root data/dtu --out_root outputs_dtu --shard $i/3 --shared_gpu > output/dtu_shard$i.log 2>&1"; done
```
- **Twice daily:** count `metrics.json` files, grep the logs for `Traceback`, and rerun any failed runs (the runner skips completed ones).
- **Nightly:** `aggregate.py` per block on the lab PC (or send raw `metrics.json` + `config.json` + `diag.csv` to Disha), then copy `eval/` dirs for N1 (`eval_dtu.tgz`, `eval_blender.tgz`) with md5.
- **Thu 12:00:** if DTU has fewer than about 150/225 done, Blender drops to seeds 0–1 (`--seeds 0 1`).

### A10 · Wed–Thu · Audit, Aayush's half (4–5 h)
- Follow Disha's protocol (D8): code the assigned papers into `results/audit/ablation_audit.csv`.
- Both of you code the 2 shared papers independently (for κ). Don't discuss them until both are done.

### A11 · Fri · Final lab sync (1 h)
- Copy all `outputs_*` (minus checkpoints) and the eval tarballs to Drive with md5.
- Leave the seed-0 checkpoints on the lab disk.
- Final RUNLOG entries with run counts per block.

### A12 · Wed–Sat · Writing (about 1.5 days)
- **Method:**
  - DNGaussian's three passes;
  - the dropout port and the compensation modes (uniform / nc / clamp-aware, with Proposition 1 from #23 / #27);
  - the factorial and contrasts;
  - the **error-budget estimator** (with Disha's text from D3).
- **Setup:**
  - a dataset table (LLFF, DTU, Blender with splits);
  - the frozen-protocol rule;
  - the uniform Blender protocol and why;
  - COLMAP/DPT provenance with md5s;
  - hardware, tags, run counts.
- **Limitations:** one host; random init; one depth prior; Blender's B is below the published number; DTU poses from COLMAP; optimization variance ≠ data variance.
- **Reproducibility appendix:** tags (`v1.0/1.1/1.2-exp`, `prereg-v1/v2/v3`), md5s, exact commands.

### A13 · Sat · Number cross-check (2–3 h)
- Every number in the report must trace to a file in `results/tables/`. Use a checklist with one row per number: table, value, file and row.

---

## 5. Disha: detailed tasks

### D1 · Sun (by 23:59) · `prereg-v3` (3–4 h, docs-only PR, no reviewer)
- **File:** `docs/PLAN_V2.md` §5.2 (addendum), mirroring #27's §5.1 style.
- **Content checklist:**
  1. **Datasets and protocol.** DTU (15 scans, train 25/22/28, 25 test views, masked metrics) and Blender (8 views [2, 16, 26, 55, 73, 76, 86, 93], 25 eval views, uniform protocol). Hyperparameters frozen as in `matrix_dtu.yaml` / `matrix_blender.yaml`. **Exclusion rule:** a DTU scan with < 49 registered images after one retry is excluded.
  2. **Replication rules (X4).** Per dataset: the same five contrasts (B−A, C−A, D−B, E−B, I_D, I_E, E−D), the same δ = 0.25 dB and Holm families as LLFF. "Replicates" = same sign and the CI overlaps the LLFF estimate.
  3. **X5:** D−B(DTU) > D−B(LLFF), one-sided. **H-PD:** the rank order of D−B is LLFF ≤ DTU ≤ Blender (exploratory, 3 points).
  4. **N1, BV1–BV6.** The estimators (b̂² = (m̄−y)² − s²/n, σ̂² = s², n = 3), the primary metric (pixel MSE; masked on DTU), and each prediction with its decision rule:
     - **BV1:** Δbias² share of (B−A) > 50 %.
     - **BV2:** Δvar share of (C−A) > 50 %.
     - **BV3:** the paired CI of MSE(D) − b̂²(B) over scenes. Below 0 ⇒ dropout also reduces bias.
     - **BV4:** b̂²(B) − b̂²(A) increases with views; sign at 9 views > 0.
     - **BV5:** I_MSE = I_bias + I_var, both reported.
     - **BV6:** the BV1/BV2 conclusions agree on all datasets.
  5. **N4:** I on PSNR, MSE, RMSE and SSIM. "Robust" = same sign on all; otherwise "scale-dependent".
  6. **N5:** I_D at depth_scale 0.5 / 1 / 2 (A and C from T1). Prediction: every |I_D| < δ.
  7. **G6/N2(a):** prediction: same-seed SD ≈ between-seed SD on the 3 drawn scenes; noise differs by variant (A, C > B, D). **The scene-draw rule is written here, before the draw.**
  8. **N2(b) audit protocol:** inclusion criteria, coding sheet, noise-band rule, κ subset (see D8).
  9. **Compensation (from #27 / memo):** C_nc − C, D_nc − D, C_ca − C, D_ca − D, Holm across 4, TOST δ = 0.25.
  10. **Exploratory disclosure:** the 15 Nov MSE-scale and heterogeneity checks were computed post hoc.
- **Then:** PR (docs-only), merge, `git tag -a prereg-v3 -m "..." && git push origin prereg-v3`. Tell Aayush the scene draw for noise2.

### D2 · Sun–Mon · Blender data on the laptop (2 h)
1. Download `nerf_synthetic.zip` from the NeRF project page (https://www.matthewtancik.com/nerf → data link) and extract to `data/nerf_synthetic/<obj>/`.
2. Check: each object has `transforms_{train,test}.json`, 100 train, 200 test.
3. DPT once: `cd dpt && ~/miniconda3/envs/dngaussian/bin/python get_depth_map_for_blender.py -r ../data/nerf_synthetic`. Check the output folder against what `train_blender.py` reads.
4. Look for GT depth in the test folders (`r_*_depth_*.png`). If it is there, note it: it allows a prior-reliability measurement for H-PD (optional).
5. Freeze: `tar cf blender_with_depth.tar -C data nerf_synthetic && md5sum …` → Drive; send the md5 to Aayush.
6. Optional 2-minute data sanity check on the laptop: upstream `train_blender.py` with the setting-1 flags on lego, `--iterations 1100`. Run `pgrep` first.

### D3 · Mon–Tue · `experiments/bias_variance.py`, N1 (1 day)
**CLI:**
`python experiments/bias_variance.py --out_root <block root> --dataset {llff,dtu,blender} --n_views N --variants A B C D E --seeds 0 1 2 [--mask_root data/dtu/submission_data/idrmasks] --tables results/tables/error_budget [--self_test]`

**Algorithm, per (scene, variant):**
1. For each seed k, load `<dataset>_<n>v/<variant>/<scene>/seed<k>/eval/ours_<iter>/renders/*.png` as float32 in [0, 1], plus `gt/*.png`. Assert GT is identical across seeds and the image counts match.
2. DTU only: use the same per-view masks as `eval_dtu.py` (from `<work>/mask/` or the mask root, resized /4). Keep only pixels where mask == 1.
3. Per pixel and channel, with n seeds:
   - m̄ = mean_k r_k; s² = var_k(r_k, ddof = 1);
   - mse_single = mean_k (r_k − y)²; b2 = (m̄ − y)² − s²/n; var = s²; mse_ens = (m̄ − y)².
4. Average over pixels, channels and views → `mse, bias2, var, mse_ens, share_var = var / mse`.
   - **Identity check, asserted:** `mse == bias2 + var` to 1e-6 relative.
   - Also report `psnr_single` = the mean over runs of the per-image-averaged PSNR, as `metrics.py` does, and `psnr_ens`.

**Outputs:**
- `error_budget_<dataset>_<n>v.csv` (scene, variant, mse, bias2, var, mse_ens, share_var).
- `error_budget_contrasts_<dataset>_<n>v.csv`: Δmse, Δbias2 and Δvar for B−A, C−A, D−B, E−B, I_D, I_E, plus the BV3 quantity MSE(D) − b̂²(B). Each gets a paired t-interval over scenes and Holm within the BV family. Reuse the helpers in `stats.py`.

**Self-test and fixture:**
- `--self_test` builds synthetic GT, a known bias map and Gaussian noise with known σ for 3 "seeds" × 2 variants, and asserts recovery of bias² and σ² within 5 % and the identity.
- Add `experiments/fixtures/bv_*` via `make_fixtures.py`.
- Add the line `python experiments/bias_variance.py --self_test` to `scripts/check.sh`. That is gate class; Aayush approves.

**Run it** on T1 (3-view) once `prereg-v3` is tagged and `eval_T1.tgz` has arrived; on views (2/6/9) once `eval_views.tgz` arrives; on DTU and Blender as their tarballs arrive.

**Ownership:** analysis code. The PR is routine class except for the check.sh line.

### D4 · Mon–Tue · Dataset factor in `aggregate.py` / `stats.py` (0.5 day)
- **`aggregate.py`:**
  - `PATH_RE` → `r"(llff|dtu|blender)_(\d+)v/([^/]+)/([^/]+)/seed(\d+)/metrics\.json$"`, glob `"*_*v"`, and a new `dataset` column in `runs.csv`.
  - Per-dataset `SCENES` / `VARIANTS` dicts for `missing.txt`.
  - LLFF outputs must stay **identical** (fixture check).
- **`stats.py`:**
  - A `--dataset` filter (default llff, existing behaviour unchanged), per-dataset verdict files (`verdict_<dataset>_<n>v.txt`).
  - Pooled **two-stage random-effects** analysis: per-scene contrast → per-dataset mean and SE → DerSimonian–Laird pooling across datasets. Report the pooled estimate, τ² and Q (dataset heterogeneity).
  - A mixed model (`metric ~ variant × dataset + (1 | scene)`) as a robustness check if statsmodels is available.
- **Fixtures:** extend `experiments/fixtures/make_fixtures.py` with fake DTU and Blender blocks; check.sh step 12 must still pass.

### D5 · Tue · N4, the interaction on several scales (2 h)
- Use the **per-run MSE from D3**, which is exact; don't convert mean PSNR to MSE.
- Compute I_D, I_E, E−D, D−B on PSNR, MSE, RMSE and SSIM. Output: `results/tables/interaction_scales_<dataset>.csv`.
- Label each contrast "robust (same sign)" or "scale-dependent" by the prereg rule.
- Write 3 sentences for the report, framed with Loftus (1978).

### D6 · Wed–Fri · N1 across views and datasets (BV4, BV6) (0.5 day)
- **Views:** error budget for A, B, C, E at 2/3/6/9 views (seeds 0–2). Test BV4: the sign of b̂²(B) − b̂²(A) at 9 views, and the trend.
- **Datasets:** the same table for DTU and Blender, then BV6.
- Write up exactly which predictions held and which failed, with the CIs.

### D7 · Wed · N2(a), variance components and power (0.5 day)
- **Data:** T1 `runs.csv` + noise (fern) + noise2 (3 scenes) + DTU/Blender seeds.
- **Estimate per variant:**
  - residual σ²_v (run noise; pooled within scene);
  - τ²_c for each contrast (between-scene variance of the true contrast) = var(scene means of c) − Σ coef²·σ²_v / R.
- **Power:** for S scenes and R runs per scene, SE_c = sqrt(τ²_c/S + Σ coef²σ²_v/(S·R)) and MDE = (t₀.₉₇₅,S−1 + t₀.₈,S−1)·SE_c.
  - Table and figure: MDE vs S ∈ {8, 15, 23, 31} for R ∈ {1, 3, 5}, for a main effect and for I.
- **Validation:** with S = 8 and R = 3, the predicted CI half-width for I_D must be about ±0.36 dB, as observed. If it isn't, the model is wrong; say so.
- **Noise band** for the audit: a 95 % band for a *single-run, 8-scene-mean* difference, per noise class (depth host ≈ B/D, prior-free ≈ A/C).

### D8 · Sun–Wed · N2(b) evidence audit, protocol + Disha's half (5–6 h)
1. **Protocol (goes into D1):**
   - Inclusion: sparse-view 3DGS papers 2024–26 with an **LLFF 3-view ablation table**.
   - Candidates: DNGaussian, FSGS, CoR-GS, DropGaussian, DropoutGS, D²GS, DOC-GS, DropAnSH-GS, Self-Ensembling GS, the co-adaptation paper. Take the first 8–10 that meet the inclusion rule.
   - Split them half and half; 2 papers coded by both.
2. **Coding sheet** `results/audit/ablation_audit.csv`: `paper, venue, year, url, table_id, host, component, baseline_psnr, with_psnr, delta_psnr, n_runs_reported (1 if unstated), spread_reported (y/n), n_scenes, init (random/sfm), coder`.
3. **Classification:**
   - `|delta| < band` ⇒ "within single-run noise"; else "above".
   - The band comes from D7, for the matching host class.
   - **Conditional wording** in the report: "if the host's run-to-run noise resembles DNGaussian's".
4. **κ:** Cohen's κ on (within/above, runs reported y/n) for the shared papers. If κ < 0.7, report the coding as unreliable.
5. **Output:** the share of coded component gains within noise, the share of papers reporting more than one run, and a scatter (F6).

### D9 · Fri · Pooled cross-dataset analysis (3 h)
- Run `stats.py` per dataset and pooled (D4). Main table: per dataset × contrast, plus the pooled row with τ² and Q.
- Test X5 and H-PD. Write the verdict per dataset and overall.

### D10 · Wed–Fri · Figures (1 day; `experiments/plots/`, consistent style via `style.py`)
| Fig | Content |
|---|---|
| F1 | **Error budget**: stacked bias²/variance bars per variant, one panel per dataset (the headline figure) |
| F2 | Error budget vs views (2/3/6/9) for A, B, C, E: the 9-view crossover |
| F3 | Forest plot per dataset: I_D, I_E, E−D, D−B, plus the pooled estimate |
| F4 | D−B by dataset (X5, H-PD) |
| F5 | BV3 ceiling: MSE(D) vs b̂²(B), one point per scene, all datasets |
| F6 | Audit: reported Δ vs noise band, coloured by whether more than one run was reported |
| F7 | Power curves: MDE vs scenes × runs |
| F8 | Depth-dose × dropout interaction (I_D at depth_scale 0.5 / 1 / 2) |
| F9 | Compensation forest (C_nc, D_nc, C_ca, D_ca vs C/D) |
| F10 | Qualitative grids per dataset (one view, A–E) |
| F11 | (Optional) per-band interaction (N3) |

### D11 · Fri, if time · N3 spectral error budget (3–4 h)
- Split each error image into **radial FFT bands** (4–5 bands). By Parseval, the band MSEs sum exactly to the total; assert this.
- Compute every contrast and N1's bias and variance per band.
- Test the prereg prediction: overlap in the mid/high bands, separation in the low band.

### D12 · Fri night–Sat · Results and Discussion (1 day)
- **Results order:**
  1. Datasets and sanity (B vs published).
  2. The factorial per dataset + pooled.
  3. Error budget (BV1–BV6).
  4. Robustness (N4, N5, G6).
  5. Compensation.
  6. Audit and power.
  7. Views.
- **Discussion:** what stacks and why, the ceiling, prior domain shift, evidence standards.
- **Replace** the PLAN_V2 §10.2 viva line "dropout as a variance penalty" with what N1 actually found.

### D13 · Fri–Sat · Website and README (0.5–1 day)
- **README:** a 3-dataset results table, the headline figures F1/F3/F5, how to reproduce (tags, md5s, matrix commands), links to the report and site.
- **Site:** a static GitHub Pages site from a `gh-pages` branch (no push to `main`).
  - Pages: Overview → Datasets → Error budget → Factorial → Rigor (noise/audit) → Reproduce.
  - Figures from `results/figures/`.

---

## 6. Handoffs and dependencies
| From | To | What | Needed by |
|---|---|---|---|
| Disha D1 | Both | `prereg-v3` tag + noise2 scene draw | Sun 23:59 (before A7's `matrix_noise2.yaml`) |
| Aayush A4 | Disha D3 | `eval_T1.tgz` + md5 | Mon evening |
| Aayush A5 | Disha D1/D4 | `dtu_with_depth.tar` md5, the excluded scans | Mon 20:00 |
| Disha D2 | Aayush A7 | `blender_with_depth.tar` + md5 on the lab PC | Tue morning |
| Aayush A7/A8 | Disha | PRs ready for review | Tue afternoon |
| Disha | Aayush | Same-day approval of A7/A8 | Tue evening |
| Aayush A3/A9 | Disha D6/D9 | `eval_views.tgz`, `eval_dtu.tgz`, `eval_blender.tgz`, raw metrics | Wed–Fri 12:00 |
| Disha D7 | Both D8/A10 | Noise bands for the audit | Wed evening |
| Disha D3 | Aayush A12 | Estimator paragraph for Method | Fri |

## 7. Decision table for failures
| Situation | Action |
|---|---|
| DTU no-go on Mon | Blender is the required second dataset (3 seeds, priority 2 in the queue). DTU goes in Limitations with what failed |
| Blender port problem on Tue | Ship the PR with DTU only; Blender in a follow-up PR on Wed (tag `v1.3-exp`) |
| Both DTU and Blender fail | Escalate at the Tue call: fall back to the documented GT-camera route for DTU. Do not drop the 2-dataset requirement silently |
| #27 review drags past Tue | Compensation runs from `v1.1-exp` on Wed; if still blocked, compensation becomes a probe-only analysis |
| Lab unavailable for a day | Shift the queue; cut in this order: N3, Blender seed 2, depth-dose seeds 1–2, noise2 |
| N1 predictions fail | Report it honestly. A failed BV2 ("dropout is not a variance reducer") is itself the headline correction |

## 8. Report outline and owners (CVPR template)
| § | Section | Owner |
|---|---|---|
| 1 | Introduction (gap statement: no factorial, no error budget, no noise-aware evidence) | Both (Aayush draft) |
| 2 | Related work (16-paper table condensed) | Disha |
| 3 | Method (passes, dropout and compensation modes, factorial, error budget) | Aayush (+ Disha §3.x estimator) |
| 4 | Setup (3 datasets, frozen protocol, prereg, hardware, tags) | Aayush |
| 5 | Results | Disha |
| 6 | Discussion | Both |
| 7 | Limitations and future work | Aayush |
| A | Reproducibility appendix, audit sheet, power tables | Aayush / Disha |

## 9. Definition of done (Sat 20:00)
- [ ] Datasets: LLFF + DTU (or Blender) complete with 3 seeds; md5s recorded; tags `v1.1-exp` / `v1.2-exp` in RUNLOG.
- [ ] `prereg-v3` tag predates every new result file.
- [ ] N1 tables for every dataset and for views; BV1–BV6 verdicts written.
- [ ] N2: power table validated against ±0.36; audit with κ.
- [ ] N4 and N5 tables; compensation verdict.
- [ ] Figures F1–F10 in `results/figures/`, regenerated from scripts.
- [ ] Every report number traced to `results/tables/` (A13 checklist).
- [ ] README + website live; the report PDF submitted.
- [ ] Viva: each person can explain every part, including AI-assisted code, and AI use is disclosed as the course requires.
