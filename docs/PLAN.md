# Execution plan: depth × dropout in sparse-view 3DGS (12–20 Oct 2026)

**Last updated:** Mon 12 Oct 2026, evening (IST) · **Maintainer:** Aayush (changes go through PRs, like code) · **Deadline:** Tue 20 Oct 2026

This plan replaces the day-by-day sections of the playbook ([research_proposal_sparse_view_3dgs_2.md](research_proposal_sparse_view_3dgs_2.md)) wherever the two differ. The playbook stays the reference for background, the GitHub workflow and the A100 runbook. All times are IST.

**Contents**
1. [Status at a glance](#1-status-at-a-glance)
2. [What changed versus the playbook, and why](#2-what-changed-versus-the-playbook-and-why)
3. [Decisions for tonight's call](#3-decisions-for-tonights-call)
4. [Schedule](#4-schedule)
5. [Workstreams: what to build and how](#5-workstreams-what-to-build-and-how)
6. [Analysis plan and verdict rules (fixed before results)](#6-analysis-plan-and-verdict-rules-fixed-before-results)
7. [Figures, tables and writing](#7-figures-tables-and-writing)
8. [Risks and fallbacks](#8-risks-and-fallbacks)
9. [Compute and storage budget](#9-compute-and-storage-budget)
10. [Reference numbers we already have](#10-reference-numbers-we-already-have)

---

## 1. Status at a glance

Update this table whenever something changes. It is the single place where status is recorded.

| # | Item | Owner | State | Evidence or link |
| --- | --- | --- | --- | --- |
| 1 | Team repo, Disha added as collaborator | Aayush | ✅ Done | `github.com/aayxsh23/dngs-dropout` (currently **public**). Tag `upstream-baseline` = `81c3d83` |
| 2 | Drive folder `sparse3dgs` plus shortcuts | Disha, Aayush | ✅ Done | Both of us can reach `/content/drive/MyDrive/sparse3dgs` |
| 3 | LLFF + DPT depth archive | Disha | ✅ Done | `sparse3dgs/data/llff_with_depth.tar`, 2.4 GB, md5 `0f6bd7bdae837ec58ad757c7c00b0561`. Per-scene counts match the laptop's data |
| 4 | Colab notebook, cached extension wheels, `aggregate.py` | Disha | ✅ Merged (#1) | Wheels in `sparse3dgs/wheels/torch-2.11.0+cu130/` |
| 5 | Colab T4 timing, fern | Disha | ✅ Done | **4.9 min/run** (train 4.4 + render/metrics 0.5). PSNR 20.80, SSIM 0.674, SSIM_sk 0.669, LPIPS 0.257 |
| 6 | Laptop environment, `setup_env.sh` | Aayush | ✅ Merged (#2) | Python 3.7.13, PyTorch 1.12.1, CUDA 11.3 |
| 7 | Laptop reproduction, 8 scenes | Aayush | ✅ Done | Mean **18.73 PSNR / 0.594 SSIM / 0.297 LPIPS** vs paper 19.12 / 0.591 / 0.294 (−0.39 dB, inside the 0.5 dB gate). See §10 |
| 8 | Laptop timing and VRAM | Aayush | ✅ Done | 7.5 min/run. Training footprint **1663 MB** (2509 MB whole-GPU peak of 4096) |
| 9 | Build and load fixes (`float.h`, C++17, `load_checkpoint`) | Aayush | ✅ Merged (#3) | Tested on the laptop; Colab check pending (next session) |
| 10 | `--seed`, `--depth_scale`, `config.json`, `scripts/peak_vram.py` | Aayush | ✅ On branch `feat/dropout` (pushed, no PR yet) | 1100-iteration smoke tests pass |
| 11 | Review of PR #1 | Aayush | ✅ Done | Disha fixed all 5 points in `222b818` (per-GPU efficiency table, no `KeyError`, optional `ssim_sk`, wheel cache keyed by source hash, tag assert in 7b); re-tested and merged |
| 12 | Dropout port, `--depth_render`, diagnostics, stats | Aayush | ⏳ Tue | W2–W7 |
| 13 | `run_matrix.py`, `matrix.yaml`, `matrix_tier2.yaml` | Aayush | ⏳ Tue | W8 |
| 14 | Questions to the lab admin | Aayush | ⏳ Not sent | §5 W13 |
| 15 | Go-bag disk | Aayush | ⏸ Reduced to a light version (R3) | W13 |
| 16 | Formats and decisions agreed | Both | ⏳ Tonight's call | §3 |
| 17 | T4 reproduction, 8 scenes (Cell 7a) | Disha | ⏳ Tue | W11 |
| 18 | Related work, verified baseline numbers | Disha | 🔄 Started | `results/reported_baselines.csv` (not yet verified) |

Pull requests:

| PR | What it does | State |
| --- | --- | --- |
| #3 | `float.h`, C++17, version-safe `load_checkpoint()` | ✅ Merged Mon night, before Disha's review: **Disha to read it post-merge** |
| #2 | `setup_env.sh`, `pip_freeze_laptop.txt` | ✅ Merged Mon night, before Disha's review: Disha to skim post-merge |
| #1 | Colab notebook, `aggregate.py`, results starters | ✅ Reviewed by Aayush, fixes in `222b818`, approved and merged |
| #4 | This plan, the revised playbook, AGENTS/CLAUDE/GEMINI/Copilot guides | ⏳ Open: Disha reviews (§2, §3, §4.2, §6), then squash-merge |

Next PR: `feat/dropout` → `main` (Tue 22:00).

---

## 2. What changed versus the playbook, and why

Monday's measurements change several assumptions in the playbook.

| ID | Change | Reason |
| --- | --- | --- |
| **R1** | **All of Tier 1 (120 runs, seeds 0–2) runs on Colab T4s** with both accounts, not split between Colab and the A100. | A T4 run takes 4.9 min, so Tier 1 is about 9.8 GPU-hours: roughly 5 h per account over Wed–Thu. One GPU type for every Tier 1 run also means the seed-to-seed std is not mixed up with GPU differences. That is cleaner than the playbook's "seed 0 on T4, seeds 1–2 on A100". |
| **R2** | **Tier 2 runs on the laptop, overnight only:** `n_views` 2, 3, 6, 9; variants B, C, D, E; seed 0. That is 128 runs × 7.5 min ≈ 16 h, split over **two nights** (Wed→Thu: 2- and 3-view; Thu→Fri: 6- and 9-view), about 8 h each. `touch STOP` each morning frees the laptop. If the A100 becomes available, it can take all of Tier 2 instead. | The laptop is unusable under full GPU load in the daytime, but nights are free. Moving Tier 2 off Colab leaves Colab with Tier 1 only (about 5 GPU-hours per account), which protects the must-have tier from free-tier limits. The 3-view point is rerun on the laptop so the whole views curve comes from one GPU; it also doubles as a cross-GPU check against Tier 1 seed 0. |
| **R3** | **The A100 becomes optional.** Send the admin questions anyway. Build only a light go-bag (git bundle, data tar, conda-pack): about 1 h instead of a day. | Nothing on the critical path needs it any more. It remains the fallback if Colab refuses GPUs, and a bonus for the optional Tier 2b. |
| **R4** | Variants D and E are defined on **DNGaussian's actual structure**: three separate rasterizations per iteration (hard depth, soft depth, photometric). D applies the dropout mask to all three; E applies it only to the photometric pass. | DNGaussian never computes its depth loss on the photometric render, so the playbook's "depth loss on the dropout render" has to mean "the depth passes also see the mask". **E costs no extra render**, since DNGaussian already renders depth separately. The playbook's rule 6 ("one extra render per iteration") does not apply. |
| **R5** | The runner reads **`results_eval.json`** and writes `ssim_sk` as well. On Colab it trains on local disk and copies only small files to Drive. It deletes TensorBoard events everywhere, and checkpoints for every seed except 0. | `metrics.py` writes `results_eval.json`, not `results.json`. One run is 337 MB with spiral renders: 164 MB checkpoint plus 13 MB TensorBoard. 120 such runs would overflow 15 GB Drive quotas. |
| **R6** | The **main table uses the 3DGS SSIM** (`SSIM`). `SSIM_sk` goes in the appendix. | Our reproduction gives 0.594 with 3DGS SSIM and 0.583 with sklearn, against the paper's 0.591. The paper's number matches the 3DGS implementation. |
| **R7** | Tier 2 includes **both D and E**, so no second tag is needed. | Disha's suggestion. Only 32 more runs. |
| **R8** | Dropout: γ = 0.2, linear schedule r(t) = 0.2 · t / T with T = 6000 (our total iterations). | DropGaussian's official code uses `0.2 * (iteration/10000)`, and 10000 is their total. Scaling to our T keeps the published end-of-training rate. |

---

## 3. Decisions for tonight's call

Each decision has a recommendation; the call confirms or changes it. Record the outcome in `RUNLOG.md`, then change this table to "Decided".

| # | Decision | Recommendation | If we disagree |
| --- | --- | --- | --- |
| D1 | Drop rate and schedule | γ = 0.2, r = 0.2 · t / 6000 (R8) | Keep the literal `t/10000` (max 0.12) and state it |
| D2 | SSIM in the main table | 3DGS `SSIM`, with `SSIM_sk` in the appendix; both in `metrics.json` (R6) | — |
| D3 | Who produces `fps` and `model_size_mb` | Aayush (W5, W6) | — |
| D4 | Tier 2 variants | B, C, D, E with `n_views` [2, 3, 6, 9], seed 0 (R2, R7) | If the nights run short, drop 9-view first |
| D5 | Machine per seed | Tier 1, all seeds: Colab T4 (shards 0/2 and 1/2). Tier 2: laptop, overnight Wed and Thu. A100: optional, Aayush confirms access on Tue (R1–R3) | Playbook split: seed 0 Colab, seeds 1–2 A100 |
| D6 | Definition of variant D | One mask per iteration, applied to all three training rasterizations (R4) | — |
| D7 | Opacity penalty under dropout | `opa_pena` uses the **pre-dropout** opacity, so the regularizer is unchanged between variants | — |
| D8 | Figure 1 (D vs E iteration diagram) | Aayush | — |
| D9 | Daily call | 21:30 IST, 15 min | — |
| D10 | Repo visibility | Make it **private** before results land. It now holds the playbook, and the course may not want unpublished work public | Keep it public; then Colab needs no token |
| D11 | Optional Tier 2b (only if Tier 1 is complete by Thu night) | D and E at γ ∈ {0.1, 0.3}, 3-view, seed 0: 32 runs, about 2.6 T4-hours | Skip |

---

## 4. Schedule

### 4.1 Overview

| Day | Aayush | Disha | Gate (checked at the evening call) |
| --- | --- | --- | --- |
| **Mon 12** (tonight) | Call; admin questions; swap laptop data to the tar; start W2 | Call; review PR #3 and #2; fix `aggregate.py` bugs 1–2 | Decisions D1–D11 recorded |
| **Tue 13** | W1 merges → W2–W7 (code) → W8 (runner) → W9 (laptop validation) → **PR by 22:00**; light go-bag if the lab is confirmed | Cell 7a T4 reproduction (8 scenes); plot scripts against the sample `diag.csv`; related work | Laptop smoke matrix passes every V2 check; T4 reproduction within 0.5 dB |
| **Wed 14** | 09:00–12:30 fix review comments; **13:00 tag `v1.0-exp`**; 13:15 start Colab shard 0/2 seed 0, then seed 1; Method section; **~23:30 laptop night 1: Tier 2, 2- and 3-view** | 09:00–11:00 review the PR plus regression check (V3); 13:15 start Colab shard 1/2 seed 0, then seed 1 | First 10 runs have sensible metrics and diagnostics |
| **Thu 15** | ~07:30 `touch STOP` on the laptop; Colab seed 2 (shard 0/2); diagnostics analysis D vs E; draft the verdict; Experimental Setup; **~23:30 laptop night 2: Tier 2, 6- and 9-view** | Colab seed 2 (shard 1/2); reruns; seed-0/1 tables; qualitative grid | All 120 Tier 1 runs done (target: Thu ~14:00); Tier 2 2v/3v done |
| **Fri 16** | ~07:30 Tier 2 done, `touch STOP`, upload to Drive; failure cases (W16); Tier 2 analysis; Figure 1; optional Tier 2b on Colab | Final tables (mean ± std); efficiency table; views plot; Introduction and Results text | **Tier 1 freeze 23:59** (plan to be done a day early) |
| **Sat 17** | Full draft: Method, Setup, Diagnostics + verdict, Discussion | Full draft: Related Work, Introduction, Results | Every figure regenerates from a script |
| **Sun 18** | Swap sections and review; arXiv check; abstract and conclusion together | Same | Complete draft |
| **Mon 19** | Revise; README "how to reproduce"; tag `v1.0-report` | Revise; citations check | Submission-ready |
| **Tue 20** | Buffer: read-through together, submit | Same | Submitted |

**Critical path:** the dropout PR (Tue 22:00) → review and regression (Wed 11:00) → tag (Wed 13:00) → Colab seeds 0–2 (Wed 13:15 to Thu ~14:00) → tables, diagnostics and verdict (Fri) → writing. Tier 2 (laptop, Wed and Thu nights) is off the critical path. A slip in the code freeze delays everything after it. The playbook's fallback stands: the freeze may slip to Wed 20:00, no later.

### 4.2 Hour by hour, the next 48 hours

**Mon 12 Oct, tonight**

| Time | Aayush | Disha |
| --- | --- | --- |
| now–21:30 | Send the admin questions (W13). Start downloading `llff_with_depth.tar` to the laptop (W12.1) | Read #3 and #2 post-merge; review #4 (this plan: §2, §3, §4.2, §6) |
| 21:30–22:00 | **Call:** §3 decisions; the formats in W4–W8 | **Call** |
| 22:00–late | Swap the laptop data and verify the md5 (W12.1). Start W2 (renderer changes) | Fresh Colab session opened from GitHub at `main`: check that Cells 1–6 still work after #3 (one wheel rebuild expected) |

**Tue 13 Oct**

| Time | Aayush | Disha |
| --- | --- | --- |
| 09:00–09:30 | W1 is done except #4; `git pull` | Start Colab Cell 7a (T4 reproduction at `upstream-baseline`, about 40 min) |
| 09:30–12:30 | W2 dropout port + W3 E mode; smoke-test each | Related work; verify `reported_baselines.csv` rows against each paper |
| 12:30–15:00 | W4 `diag.csv`, W5 `train_stats.json`, W6 `render_stats.json`, W7 point-cloud race fix | 7a table vs paper; post it to the channel |
| 15:00 | **Push a sample `diag.csv`** from a smoke run to the channel | — |
| 15:00–18:30 | W8 `run_matrix.py`, `matrix.yaml`, `matrix_tier2.yaml`, `matrix_smoke.yaml` | `experiments/plots/diag_curves.py` and `qual_grid.py` against the sample `diag.csv` and a fake `eval/` folder |
| 18:30–21:30 | W9 validation on the laptop (V1, V2, V4) | Stats script `experiments/stats.py` (§6) |
| 21:30 | Call | Call |
| by 22:00 | **Open PR `feat/dropout` → `main`** with V1/V2/V4 output in the description | — |
| 22:00–late | Light go-bag if the lab is confirmed (W13) | — |

**Wed 14 Oct**

| Time | Aayush | Disha |
| --- | --- | --- |
| 09:00–11:00 | On standby for review questions; Method section outline | Review the PR; **V3 regression** on Colab: variant B, fern and horns, full 6000 iterations at the PR branch |
| 11:00–12:30 | Address comments; rerun V2 if the code changed | Re-review |
| **13:00** | **Merge, tag `v1.0-exp`, push the tag** (W10) | — |
| 13:15 | Colab account A: shard 0/2, seed 0 (W11). Tracker row → "running" | Colab account D: shard 1/2, seed 0. Tracker row → "running" |
| ~15:00 | When seed 0 finishes, start seed 1 on the same account (`SEEDS = '1'`) | Same |
| 17:00–21:30 | Check the first finished `diag.csv` files (W9 checks on real runs); Method section | `aggregate.py` on the first runs; first tables |
| ~23:30 | Laptop night 1: Tier 2 in tmux (W12.3) | — |
| 21:30 | Call: gate "first 10 runs sensible" | Call |

**Thu 15 to Tue 20:** follow §4.1. The details for each task are in §5–§7.

---

## 5. Workstreams: what to build and how

### W1. Merge the open PRs (#1–#3 ✅ merged Mon night; #4 waits for Disha's review)

```bash
# after each PR is approved on GitHub: Squash and merge (web UI), then locally:
git checkout main && git pull
git checkout feat/dropout && git rebase main     # only Aayush works on this branch, so a rebase is safe
git push --force-with-lease
```
Merge order: #3 (it touches `train_llff.py`, which `feat/dropout` also edits), then #2, then #1. If the rebase conflicts in `train_llff.py`, keep both changes: the `load_checkpoint` import from #3 and the `--seed` / `write_run_config` lines from `feat/dropout`.

### W2. Port DropGaussian (Aayush, Tue morning)

**What the official code does** (`DCVL-3D/DropGaussian_release`, `gaussian_renderer/__init__.py`, commit `c13da4a`):

```python
if is_train:
    compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
    drop_rate = 0.2 * (iteration/10000)
    d = torch.nn.Dropout(p=drop_rate)
    compensation = d(compensation)          # zeroes each entry with prob r, scales survivors by 1/(1-r)
    opacity = opacity * compensation[:, None]
```
It is called only from `train.py` (`render(..., is_train=True, iteration=iteration)`). Densification uses that render's `visibility_filter` and `radii` unchanged.

**Our port:**

1. `arguments/__init__.py`, in `OptimizationParams` next to `depth_scale`:
   ```python
   self.drop_rate = 0.0          # DropGaussian max rate (gamma); 0 disables dropout entirely
   self.depth_render = "shared"  # "shared": depth passes see the dropout mask (variant D); "clean": they don't (E)
   ```
2. `gaussian_renderer/__init__.py`: give `render`, `render_for_depth` and `render_for_opa` a new last parameter `drop_comp=None`. Apply it just before the rasterizer call, and **return the pre-dropout opacity** under `"opacity"` (decision D7):
   ```python
   opacity_clean = opacity
   if drop_comp is not None:  # DropGaussian; only the training loop ever passes drop_comp
       opacity = opacity * drop_comp[:, None]
   ...
   return {..., "opacity": opacity_clean, ...}
   ```
   In `render`, place this after the `if inference:` line. In `render_for_depth`, opacity is the constant `0.95`. With compensation, survivors get 0.95/(1−r) > 1, which the rasterizer clamps to per-pixel alpha 0.99 (`forward.cu:347`). This is the faithful result of a naive plug-in; mention it in the Method section. Leave the `_sh` variants alone.
3. `train_llff.py`, at the top of each iteration, just after the camera is picked:
   ```python
   # DropGaussian schedule (official code: 0.2 * iteration / 10000, their total); we scale to our total
   drop_rate_now = opt.drop_rate * iteration / opt.iterations
   drop_comp = None
   if drop_rate_now > 0:
       drop_comp = torch.nn.Dropout(p=drop_rate_now)(torch.ones(gaussians.get_xyz.shape[0], device="cuda"))
   depth_comp = drop_comp if opt.depth_render == "shared" else None
   ```
   Pass `drop_comp=depth_comp` to `render_for_depth` and `render_for_opa`, and `drop_comp=drop_comp` to the photometric `render`. **One mask per iteration** is valid because the Gaussian count only changes in densification, at the end of the iteration.
4. **Leave unchanged:** `clean_views`, `training_report`, `scene.save(... render(...)["color"])` and densification. They call `render` without `drop_comp`, so they never drop. This mirrors DropGaussian: dropped Gaussians still have `radii > 0` (radii don't depend on opacity), so they count as visible with zero gradient that iteration. That slightly lowers their average densification gradient, exactly as in the official code.
5. Use the global torch RNG (as the official code does). `safe_state(seed)` seeds it, so masks differ per seed.

**Tests (laptop, 1100 iterations on fern unless stated):**
- T1: `--drop_rate 0` must create no mask. Add a temporary `assert drop_comp is None` and run variant B; the PSNR must be in the range of Monday's B smoke runs (17.3–17.8).
- T2: `grep -n "drop_comp\|drop_rate" render.py metrics.py spiral.py` prints nothing.
- T3: variant D: temporarily print `(drop_comp == 0).float().mean()` every 200 iterations. It should track `0.2 * t / T` within ±0.01.
- T4: variant E: the same, and confirm the depth passes get `None`.

### W3. Decoupled depth, variant E (Aayush, with W2)

This is covered by `depth_comp` in W2 step 3; there is nothing else to build. Two notes for the Method section:
- In E the dropout mask perturbs only the photometric gradient. The depth passes optimize positions (hard) and opacity (soft) against an un-dropped model.
- E has the **same per-iteration cost as D**. Any timing difference comes from noise and from the diagnostic renders (W4), which are identical in all variants.

### W4. Diagnostics: `diag.csv` (Aayush, Tue afternoon)

The columns are fixed (playbook): `iter, loss_rgb, loss_depth_used, loss_depth_clean, depth_var_masks, n_gaussians, drop_rate_now, iter_ms, vram_peak_mb`. Write one row every 100 iterations: create the file with its header at the start of training and flush each row.

**Where:** immediately after `iter_end.record()`, **before** the `with torch.no_grad():` densification block. Densification runs on the same multiples of 100 and changes the Gaussian count, which would invalidate the masks. Wrap all of it in `torch.no_grad()`.

**How each column is computed:**

| Column | Computation |
| --- | --- |
| `loss_rgb` | `(Ll1 + opt.lambda_dssim * (1 - ssim(image, gt_image))).item()`: the photometric loss of this iteration, without regularizers |
| `loss_depth_used` | `depth_diag(cam, depth_comp)`: the depth loss on the renders that receive gradients in this variant. Masked in C and D; unmasked in A, B and E |
| `loss_depth_clean` | `depth_diag(cam, None)`: the same, with no mask |
| `depth_var_masks` | At a **fixed** view (`scene.getTrainCameras()[0]`), render the soft depth (`render_for_opa`) under K = 4 independent masks at a **fixed probe rate of 0.2** in every variant. Take the per-pixel variance across the 4 maps (`unbiased=False`) and its mean over pixels. The masks use a separate `torch.Generator(device="cuda")` seeded with `1000 + seed`, so the diagnostics never change the training RNG stream |
| `n_gaussians` | `gaussians.get_xyz.shape[0]` |
| `drop_rate_now` | from W2 |
| `iter_ms` | Wall-clock time per training iteration since the previous row, **excluding** time spent in diagnostics. Call `torch.cuda.synchronize()` before reading `time.time()`; subtract the measured diagnostic time |
| `vram_peak_mb` | `torch.cuda.max_memory_reserved() / 2**20` (running peak) |

```python
DIAG_PATCH = 11  # fixed patch size (middle of DNGaussian's 5..17) so used and clean are comparable
def depth_loss_diag(depth, depth_mono, it, tol):
    l = 0.1 * patch_norm_mse_loss(depth[None], depth_mono[None], DIAG_PATCH, tol) \
        + patch_norm_mse_loss_global(depth[None], depth_mono[None], DIAG_PATCH, tol)
    if it > 3000:
        l = l + 0.1 * loss_depth_smoothness(depth[None], depth_mono[None])
    return l

def depth_diag(cam, comp):
    mono = 255.0 - cam.depth_mono
    v = depth_loss_diag(render_for_depth(cam, gaussians, pipe, background, drop_comp=comp)["depth"], mono, iteration, opt.error_tolerance)
    if iteration > opt.soft_depth_start:
        v = v + depth_loss_diag(render_for_opa(cam, gaussians, pipe, background, drop_comp=comp)["depth"], mono, iteration, opt.error_tolerance)
    return v.item()
```
The depth losses involve no randomness apart from the patch size, so fixing the patch size makes `used` and `clean` directly comparable. This mirrors how training composes the losses (0.1 local + 1 global + 0.1 smoothness after iteration 3000, for both passes).

**Sharp checks** (used in V2): `loss_depth_used == loss_depth_clean` exactly in A, B and E. It differs in C and D. `drop_rate_now` is 0 in A and B, and linear to 0.2 in C, D and E. `depth_var_masks > 0` in every variant. Overhead is about 8 extra renders per 100 iterations (about 2–3 %).

**What the diagnostics answer:** the gap `loss_depth_used − loss_depth_clean` in D measures how much dropout perturbs the depth signal DNGaussian learns from. In C it measures the perturbation without depth supervision. `depth_var_masks` across A–E measures how fragile each model's geometry is under dropout.

### W5. `train_stats.json` (Aayush)

Write it at the end of `training()`. Set `t_train0` right before the loop and call `torch.cuda.synchronize()` before reading the clock.

```python
stats = dict(
    train_time_s=round(time.time() - t_train0, 1),       # whole loop, incl. the iteration-6000 eval/save and diagnostics
    n_gaussians=int(gaussians.get_xyz.shape[0]),
    vram_peak_mb=round(torch.cuda.max_memory_reserved() / 2**20, 1),
    model_size_mb=round(sum(p.numel() * p.element_size()
                            for g in gaussians.optimizer.param_groups for p in g["params"]) / 2**20, 2),
    drop_rate=opt.drop_rate, depth_scale=opt.depth_scale, depth_render=opt.depth_render,
    seed=seed, torch_version=torch.__version__)
json.dump(stats, open(os.path.join(dataset.model_path, "train_stats.json"), "w"), indent=2)
```
`model_size_mb` counts every trainable tensor: the Gaussian attributes plus the neural field (grid encoder and MLP). That is everything needed to render.

### W6. FPS: `render_stats.json` (Aayush)

In `render.py`, after the `eval` set is rendered, add a timing loop over the eval cameras: 5 warm-up renders, then 50 timed renders, cycling through the views with `inference=True`, as `render_set` does. Call `torch.cuda.synchronize()` around the timed block. Write `{"fps": 50 / t, "n_gaussians_render": ..., "resolution": [W, H]}` to `<model_path>/render_stats.json`. Fern has only 3 eval views, so timing the per-image saving loop would be noise; that is why there is a separate timed loop. The timed loop runs no dropout (invariant 1).

### W7. Fix the random-point-cloud race (Aayush)

`scene/dataset_readers.py`, `readColmapSceneInfo`, in the `rand_pcd` branch: write the generated cloud to a unique temporary directory instead of `<scene>/sparse/0/points3D_random.ply`:
```python
import tempfile
ply_path = os.path.join(tempfile.mkdtemp(prefix="dngs_pcd_"), "points3D_random.ply")
```
`Scene` then copies it to `<model_path>/input.ply` as before. First `grep -rn points3D_random` to confirm nothing else reads the old path. Without this fix, parallel slots on one scene (A100, or two runs of one scene) could overwrite each other's `input.ply` record. Training itself is unaffected, because the cloud is kept in memory.

### W8. Runner and matrix files (Aayush, Tue afternoon)

**`experiments/matrix.yaml`** (Tier 1):
```yaml
scenes: [fern, flower, fortress, horns, leaves, orchids, room, trex]
seeds: [0, 1, 2]
n_views: [3]
variants:
  A: {depth_scale: 0, drop_rate: 0.0, depth_render: shared}
  B: {depth_scale: 1, drop_rate: 0.0, depth_render: shared}
  C: {depth_scale: 0, drop_rate: 0.2, depth_render: shared}
  D: {depth_scale: 1, drop_rate: 0.2, depth_render: shared}
  E: {depth_scale: 1, drop_rate: 0.2, depth_render: clean}
# identical to scripts/run_llff.sh except: no spiral video, eval-only render, explicit seed
train_flags: >-
  -r 8 --eval --rand_pcd --iterations 6000 --lambda_dssim 0.2
  --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01
  --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500
  --split_opacity_thresh 0.1 --error_tolerance 0.00025 --scaling_lr 0.003
  --shape_pena 0.002 --opa_pena 0.001 --near 10
steps:
  - "python train_llff.py -s {scene_dir} --model_path {work} --n_sparse {n_views} --seed {seed} {variant_flags} {train_flags}"
  - "python render.py -s {scene_dir} --model_path {work} -r 8 --near 10 --skip_train --skip_test"
  - "python metrics.py --model_paths {work}"
```
- **`experiments/matrix_tier2.yaml`:** the same, with `n_views: [2, 3, 6, 9]`, `seeds: [0]`, variants B, C, D, E only. The runner orders runs by `n_views`, so night 1 covers 2- and 3-view and night 2 covers 6- and 9-view. 9-view can be dropped if time runs out.
- **`experiments/matrix_smoke.yaml`:** `scenes: [fern]`, `seeds: [0]`, all five variants, `--iterations 1100 --test_iterations 1100` in `train_flags`. Used by V2 and to produce the sample `diag.csv` for Disha.

**`experiments/run_matrix.py`**, behaviour (start from the playbook sketch and change the following):
1. Enumerate `sorted(product(n_views, seeds, variants, scenes))` and take every N-th from index i for `--shard i/N`. `--seeds` overrides the YAML.
2. `out = {out_root}/llff_{nv}v/{variant}/{scene}/seed{seed}`. Skip the run if `out/metrics.json` exists.
3. Check for a `STOP` file in the repo root before each run. If found, finish cleanly.
4. `--work_root` (default: equal to `--out_root`). On Colab use `/content/work`, so training I/O hits local disk, not Drive.
5. Before a run, delete any leftover `work` and `out` directories (no `metrics.json` means a killed run).
6. Run each step with `subprocess.run(shlex.split(cmd), stdout=log, stderr=subprocess.STDOUT)`. The steps contain no pipes, so no shell is needed, and a scene path with odd characters can't be misread as shell syntax. Append everything to `work/run.log` and print one status line per run to the console.
7. **On failure:** write `work/FAILED` (step index and return code), copy `run.log` to `out/`, and **continue** with the next run.
8. **On success,** build `metrics.json` from `results_eval.json` (the single `ours_*` entry), `train_stats.json`, `render_stats.json` and `config.json`, with these keys:
   `variant, scene, n_views, seed, psnr, ssim, ssim_sk, lpips, train_time_s, fps, n_gaussians, model_size_mb, vram_peak_mb, wall_time_s, git_commit, code_tag, gpu, shared_gpu, torch_version, drop_rate, depth_scale, depth_render`.
9. **Clean up before copying:** delete `events.out.tfevents*`. Delete `chkpnt_latest.pth` unless the seed is in `--keep_ckpt_seeds` (default `0`). Delete `train/` and `test/` if present. Keep `eval/` (about 3 MB, renders plus depth maps), `diag.csv`, the JSON files, `point_cloud/` for kept seeds, `input.ply`, `cfg_args`, `opt_args` and `run.log`.
10. If `work_root != out_root`, copy `work` to `out` **and write `metrics.json` last**, so a half-copied run never counts as done.
11. **Refuse to run unless HEAD is exactly at a tag** (`git describe --tags --exact-match`), unless `--allow_untagged` (smoke tests only). Record the tag as `code_tag`.
12. `--dry_run` prints every command and the shard size without running anything.
13. At the end, print counts of done, skipped and failed runs, and the list of failures.
14. `--shared_gpu` sets `shared_gpu: true` (A100 multi-slot only). Colab and the laptop leave it off.

### W9. Validation before the freeze

| Check | Who, when | How | Pass if |
| --- | --- | --- | --- |
| V1 dry run | Aayush, Tue evening | `run_matrix.py --matrix experiments/matrix.yaml --dry_run --data_root data/llff --out_root /tmp/x --allow_untagged`, then `--shard 0/2` and `1/2` | 120 runs listed (60 + 60), flags correct for each variant |
| V2 smoke matrix | Aayush, Tue evening | `matrix_smoke.yaml` on the laptop, about 7 min for all 5 variants | All 5 `metrics.json` written; `diag.csv` passes the W4 sharp checks; cleanup policy applied; `config.json` has the right flags |
| V3 regression | Disha, Wed morning | Colab, PR branch, variant B, fern and horns, full 6000 iterations | Within ±0.5 dB of her T4 7a numbers for the same scenes (nondeterminism is about ±0.3 dB) |
| V4 eval determinism | Aayush, Tue evening | Run `render.py` twice on a variant D smoke model; `md5sum eval/ours_*/renders/*` | Identical files, which proves evaluation never samples a mask |
| V5 first real runs | Aayush, Wed 17:00 | Read the first 2–3 real `diag.csv` files | Losses finite; Gaussian counts plausible (tens of thousands); `drop_rate_now` reaches 0.2 at iteration 6000 |

### W10. Freeze and tag (Aayush, Wed 13:00)

```bash
git checkout main && git pull
git tag -a v1.0-exp -m "Code freeze for main experiment matrix"
git push origin v1.0-exp
```
Post "v1.0-exp is live" in the channel. Colab Cell 3: `TAG = 'v1.0-exp'`. A bug found later gets a `fix/` branch and tag `v1.1-exp`. Rerun only the affected runs and log them in `RUNLOG.md`.

### W11. Colab operations (both)

**Open the notebook from GitHub** (after PR #1 is merged):
`https://colab.research.google.com/github/aayxsh23/dngs-dropout/blob/main/notebooks/colab_runner.ipynb`
Or in Colab: File → Open notebook → GitHub → `aayxsh23/dngs-dropout` → branch `main` → `notebooks/colab_runner.ipynb`. Don't edit the notebook in Colab and keep the copy. Changes go through git, like any other file.

**Each session:** Runtime → Change runtime type → T4. Run Cells 1, 2, 3 (`TAG = 'v1.0-exp'`), 3b (a no-op after PR #3), 4, 5 and 6. Skip Cell D (depth is generated once and already done). Then Cell 7b:

```python
SHARD = '0/2'   # Aayush: 0/2, Disha: 1/2
SEEDS = '0'     # then '1', then '2'
!python experiments/run_matrix.py --matrix experiments/matrix.yaml --data_root /content/data/llff \
    --out_root {ROOT}/outputs --work_root /content/work --shard {SHARD} --seeds {SEEDS} \
    2>&1 | tee -a {ROOT}/logs/colab_shard_{SHARD.replace('/', 'of')}_seed{SEEDS}.log
```
- **Tracker:** one row per (account, seed) with start time, status and notes. Set it to "running" when you start and "done" or "failed: …" when it ends.
- **Disconnects:** rerun Cells 1–6, then 7b. Finished runs are skipped; only the run in progress is lost.
- **Expected duration:** 20 runs per shard per seed × about 5 min ≈ 1.7 h. Variant A is faster (no depth passes).
- **Order on each account:** seed 0 → seed 1 (Wed) → seed 2 (Thu morning). That is all of Colab's mandatory work, about 5 GPU-hours per account. Spare Colab time on Thu and Fri goes to reruns, then the optional Tier 2b (D11).
- Never run two GPU sessions on one account at the same time. Keep the tab active.

### W12. Laptop operations (Aayush)

**W12.1 Swap in the canonical data (Mon night):**
```bash
cd ~/projects/DNGaussian
# download llff_with_depth.tar from Drive (browser), move it into the repo root on the Linux side
md5sum llff_with_depth.tar                  # must print 0f6bd7bdae837ec58ad757c7c00b0561
mv data/llff data/llff_laptopdepth          # keep the old one until the matrix is done
tar -xf llff_with_depth.tar -C data/        # creates data/llff/
ls data/llff/fern/depth_maps | head -3
```
**W12.2 Make long runs survive (only relevant for the overnight overflow in W12.3):**
- Windows power settings: never sleep when plugged in; lid close does nothing.
- Run inside `tmux` (`sudo apt install tmux`, then `tmux new -s runs`; detach with Ctrl-b d, reattach with `tmux attach -t runs`).
- Keep one Windows Terminal tab running `wsl` open, so WSL doesn't shut the distro down when VS Code closes.

**W12.3 Tier 2 on the laptop, overnight only (revised).** A full-load run makes the laptop unusable, so it trains only at night: about 23:30 to 07:30, roughly 64 runs per night at 7.5 min each.

```bash
# each night, after the day's work is saved (the tag v1.0-exp exists from Wed 13:00)
cd ~/projects/DNGaussian && git fetch --tags && git checkout v1.0-exp
rm -f STOP
tmux new -s tier2          # inside tmux:
python experiments/run_matrix.py --matrix experiments/matrix_tier2.yaml --data_root data/llff \
    --out_root outputs --seeds 0 2>&1 | tee -a logs/laptop_tier2.log
# detach with Ctrl-b d and go to sleep

# each morning
touch STOP                 # finishes the run in progress (≤ 8 min), then exits
pgrep -af run_matrix       # wait until this prints nothing; then the laptop is free
git checkout feat/...      # back to your working branch if needed
```
- **Night 1 (Wed→Thu):** 2- and 3-view (64 runs). **Night 2 (Thu→Fri):** 6- and 9-view (64 runs). The runner skips finished runs, so the same command simply continues. If night 2 falls short, 9-view is what gets dropped.
- **Fri morning:** copy `outputs/llff_{2,3,6,9}v` to `sparse3dgs/outputs/laptop/` on Drive (browser upload or `rclone`). The folders are already slim after cleanup. Update the tracker each night and morning.
- **Emergency use of a night:** if Colab can't finish Tier 1 seed 2 by Thu afternoon, night 2 runs **that whole seed** instead (40 × 7.5 min ≈ 5 h) and Tier 2 keeps only 2- and 3-view. The report then states which seed ran on which GPU.
- **If the A100 works out** (Aayush confirms on Tue): it can take all of Tier 2 in one visit with 4 slots and `--shared_gpu`, and the laptop nights become spare.

### W13. A100 lab (optional; Aayush)

**Send these questions tonight** (copied from the playbook): the operating system and admin rights; WSL2 if Windows; whether USB drives are allowed; lab hours, weekdays and weekend; whether the GPU is shared; the driver version from `nvidia-smi` and free disk space; whether pip, conda and GitHub work (proxy?); whether the home folder persists between visits.

**Light go-bag, only if access is confirmed:** the git bundle (`git bundle create dngs.bundle --all`, refreshed after the tag), `llff_with_depth.tar`, `conda pack -n dngaussian` (Linux), and `go.sh`/`collect.sh` from the playbook. Rebuild the extensions for `TORCH_CUDA_ARCH_LIST="8.0;8.6"` before packing. Do a WSL export only if the admin says Windows.

**Use, in priority order:** (1) overflow if Colab can't finish a seed, but always a **whole seed** on the A100; (2) optional Tier 2b (D11); (3) nothing. It never blocks the plan.

### W14–W18

See §6 (analysis), §7 (figures and writing) and §8 (risks).

---

## 6. Analysis plan and verdict rules (fixed before results)

These rules are written down **before** any Tier 1 result exists. Change them only by PR, with the reason stated.

**Unit of analysis.** For each variant and scene, average over the 3 seeds. That gives 8 scene-level values per variant. Comparisons are **paired by scene**: Δ = X − Y per scene, mean Δ, 95% CI from a t-interval over the 8 scenes (df = 7), and wins out of 8. The table also reports the mean ± std over seeds of the 8-scene average (`aggregate.py`). The script is `experiments/stats.py` (Disha), which reads `results/tables/runs.csv`.

**Comparisons:** B vs A (does depth help?), C vs A (does dropout help?), D vs best(B, C), E vs best(B, C), E vs D (does decoupling help?). Here best(B, C) is whichever has the higher mean PSNR.

**Verdict** (PSNR is primary; SSIM and LPIPS must not contradict it):
- **Complementary:** D or E beats best(B, C) with a 95% CI above 0, and SSIM/LPIPS are not significantly worse.
- **Redundant:** both CIs for D − best(B, C) and E − best(B, C) contain 0.
- **Interfering:** D − best(B, C) has a CI below 0, or D − B has a CI below 0.
- **Mechanism (the report's key evidence):** in D, the mean of `loss_depth_used − loss_depth_clean` over iterations ≥ 1000 is positive across scenes (CI above 0), and larger than in C. If in addition E − D has a CI above 0, the **interference runs through depth supervision**.
- If no difference is detectable, that is still a valid result (redundancy). The report then centres on the diagnostics.

**Power caveat (state it in the report):** run-to-run noise is about 0.3–0.5 dB per run, and averaging 3 seeds × 8 scenes reduces it. Differences much below about 0.2 dB will likely not be resolvable with 8 scenes.

---

## 7. Figures, tables and writing

### 7.1 Figures and tables

| # | Item | Owner | Script → output | Inputs | Ready by |
| --- | --- | --- | --- | --- | --- |
| F1 | One iteration of D vs E: three passes, where the mask applies, which loss goes where | Aayush | draw.io or TikZ → `results/figures/fig1_iteration.pdf` | §5 W2/W3 | Fri |
| T1 | Main table: LLFF 3-view, A–E mean ± std over seeds; published baselines below a rule labelled "reported" | Disha | `aggregate.py` → `results/tables/main_3v.tex` | all `metrics.json`, `reported_baselines.csv` | Fri |
| T2 | Per-scene PSNR (appendix) | Disha | `aggregate.py` → `per_scene_3v.csv` | — | Fri |
| T3 | Paired comparisons with CIs (§6) | Disha | `experiments/stats.py` → `results/tables/paired.tex` | `runs.csv` | Fri |
| F2 | Diagnostic curves: `loss_depth_used − loss_depth_clean` vs iteration for C, D, E (mean ± std over scenes); `depth_var_masks` vs iteration for A–E | Disha (script), Aayush (interpretation) | `experiments/plots/diag_curves.py` → `fig_diag.pdf` | all `diag.csv` | Thu night |
| F3 | Qualitative grid: 3 scenes × [GT, B, C, D, E], RGB row and depth row, seed 0 | Disha | `experiments/plots/qual_grid.py` → `fig_qual.pdf` | `eval/ours_6000/{gt,renders,depth}` | Fri |
| F4 | Failure cases: 2–3 scenes where combining hurt, RGB + depth (W16) | Aayush picks, Disha renders | `qual_grid.py --scenes …` | per-scene Δ from T3 | Fri |
| T4 | Efficiency: train time, FPS, Gaussian count, model size, peak VRAM per variant, **T4 runs only** | Disha | `aggregate.py` (fixed to group by GPU) → `efficiency_3v.tex` | `metrics.json`, `shared_gpu == false` | Fri |
| F5 | PSNR vs number of views (2, 3, 6, 9) for B, C, D, E (laptop, seed 0; all four points from Tier 2) | Disha | `experiments/plots/views.py` → `fig_views.pdf` | `sparse3dgs/outputs/laptop/` | Fri evening |

Every figure must regenerate from its script; no hand edits. Scripts take `--out_root` and write to `results/figures/`.

### 7.2 Failure-case analysis (W16, Aayush, Fri)

1. From T3, take the 2–3 scenes with the most negative D − best(B, C), or E − best(B, C) if E is the worse one.
2. For each: GT, B, D and E renders plus depth maps (`eval/ours_6000/`); that scene's `diag.csv` gap curve; its Gaussian count over time.
3. Look for floaters, depth holes and blurred thin structures. Relate each to the per-scene gap and `depth_var_masks`.
4. One paragraph per scene: what fails, the evidence, and the likely mechanism.

### 7.3 Writing

Overleaf if the course wants LaTeX; otherwise one shared doc. Never edit the same section at the same time.

| Section | Writer | Draft by | Must contain |
| --- | --- | --- | --- |
| Abstract, Conclusion | Both | Sun | The verdict in one sentence, with the number |
| Introduction | Disha | Fri | The cross-family question is open (DNGaussian++ stays within depth regularization) |
| Related work | Disha | Thu | Sparse-view NeRF, sparse-view 3DGS, depth priors (DNGaussian, DNGaussian++, FSGS), dropout-style regularization (DropGaussian, DropoutGS, CoR-GS) |
| Method | Aayush | Fri | Factorial design A–E; the three passes; the dropout port (official code, schedule rescaled 10000 → 6000); D vs E (R4); the alpha clamp in the hard-depth pass; `opa_pena` on pre-dropout opacity; F1 |
| Experimental setup | Aayush | Fri | LLFF 3-view, eval split (every 8th image), resolution `-r 8`; metrics: PSNR, **3DGS SSIM** (sklearn in the appendix), **LPIPS (VGG)**; seeds; Tier 1 on Colab T4 (PyTorch 2.11), Tier 2 on the RTX 3050 Ti laptop (PyTorch 1.12); minutes per run; DPT_Hybrid depth from one archive |
| Results | Disha | Sat | T1–T4, F3, F5 |
| Diagnostics and verdict | Aayush | Sat | F2; the §6 rules applied |
| Discussion, failure cases, limitations | Both | Sat | F4 and the limitations below |

**Limitations to state plainly:**
- The dropout rate was tuned on vanilla 3DGS, and the schedule was rescaled to 6000 iterations.
- **Initialization:** DropGaussian builds its initial point cloud from the sparse views with COLMAP (SfM, `patch_match_stereo` and `stereo_fusion`), while DNGaussian and our variants start from random points. Variant C is therefore not expected to reach DropGaussian's published 20.76 PSNR.
- One dataset (LLFF); 3 seeds; 8 scenes.
- Tier 1 (T4, PyTorch 2.11) and Tier 2 (RTX 3050 Ti, PyTorch 1.12) ran on different GPUs. Each tier is internally consistent, and Tier 2 has one seed only.
- The rasterizer is nondeterministic.
- Hyperparameters are DNGaussian's 3-view settings at every view count.

### 7.4 Before submitting (Mon 19)

- [ ] Every number traces to a `metrics.json` or a cited paper table (`reported_baselines.csv` with `verified_against_own_paper = yes`)
- [ ] Mean ± std, number of seeds, SSIM implementation and LPIPS backbone are all stated
- [ ] GPU types and minutes per run are stated
- [ ] arXiv re-checked on Sun 18 for new work combining depth priors with Gaussian dropout
- [ ] DNGaussian++ volume, issue and pages checked on IEEE Xplore; DropoutGS authors and pages from the CVPR 2025 proceedings; DropGaussian = Park, Ryu and Kim, CVPR 2025, pp. 21600–21609
- [ ] README explains how to regenerate every table and figure from tag `v1.0-exp`
- [ ] Tag `v1.0-report` pushed; Drive `outputs/` left untouched
- [ ] Status table (§1) and `RUNLOG.md` final

---

## 8. Risks and fallbacks

| Risk | Detect by | Fallback |
| --- | --- | --- |
| The dropout PR is not ready by Tue 22:00 | Tue 18:00 progress check | Cut W6 (FPS) and W7 (race fix; only matters for parallel slots) to after the freeze as `v1.1-exp`, since they don't change training. The freeze may slip to Wed 20:00, no later |
| V3 regression fails (B differs from 7a by more than 0.5 dB) | Wed 11:00 | Diff `train_llff.py` against `upstream-baseline`. The usual suspect is a flag default. Spend at most 2 h, then freeze with the fix |
| Colab refuses a GPU or hits a usage limit | Any session | Switch to the other account. Then Kaggle Notebooks (same T4 GPU type, weekly quota; change the paths in Cells 2 and 6). If a seed still can't finish on T4s, the laptop or the A100 runs **that whole seed**, and the report says so |
| Colab is too slow for 3 seeds | Thu 12:00, if fewer than 80 runs are done | Laptop night 2 runs Tier 1 seed 2 as a whole block (W12.3), and Tier 2 keeps 2- and 3-view only. Last resort: cut to 2 seeds |
| Tier 2 doesn't finish in two nights | Fri 07:30 | Drop 9-view (Tier 2 is optional), or run the rest on Fri night. Or the A100 takes the remainder, if available |
| A variant gives NaN or diverges | `run.log`, `diag.csv` | Rerun once. If it repeats on other scenes, it is a finding: report it, don't hide it |
| Drive quota fills | Drive storage page | The cleanup policy keeps runs small (§9). Delete seed-0 checkpoints of scenes not used in figures |
| Data mismatch | md5 check in W12.1 and Cell 6 | Never train on data whose md5 doesn't match |
| No measurable difference between variants | Thu night | A valid "redundant" result; centre the report on the diagnostics (§6) |
| A100 never available | — | Not needed (R3) |

---

## 9. Compute and storage budget

| Block | Runs | Minutes per run | GPU-hours | Where | When |
| --- | --- | --- | --- | --- | --- |
| Tier 1, seed 0 | 40 | ~4.9 (T4) | 3.3 | Colab, 2 accounts | Wed 13:15–~15:00 |
| Tier 1, seed 1 | 40 | ~4.9 | 3.3 | Colab, 2 accounts | Wed ~15:00–~17:00 |
| Tier 1, seed 2 | 40 | ~4.9 | 3.3 | Colab, 2 accounts | Thu morning |
| Tier 2 (views 2, 3, 6, 9) | 128 | ~7.5 (laptop) | 16 | Laptop, nights only | Wed 23:30–Thu 07:30, Thu 23:30–Fri 07:30 |
| V3 regression | 2 | 4.9 | 0.2 | Colab (Disha) | Wed morning |
| Tier 2b (optional) | 32 | 4.9 | 2.6 | Colab or A100 | Fri |

Each Colab account runs 60 Tier 1 runs: about **5 GPU-hours** (about 3.4 h Wed, 1.7 h Thu), comfortably inside free-tier use. If limits still bite, Kaggle is the next T4 source (§8). The laptop runs only at night (Tier 2). The minutes per run come from upstream DNGaussian. Variant A is about 25 % faster (no depth passes); diagnostics add about 2–3 %.

**Storage per run after cleanup:** about 4 MB (`eval/` 3.4 MB plus JSON/CSV) for seeds 1–2, and about 170 MB for seed 0 with checkpoint and point cloud. Tier 1 on Drive ≈ 40 × 170 MB + 80 × 4 MB ≈ 7 GB, split across the two writing accounts. If quota is tight, run seed 0 with `--keep_ckpt_seeds` empty and re-train the 2–3 figure scenes later if needed.

---

## 10. Reference numbers we already have

**Laptop reproduction**: upstream code at `upstream-baseline`, laptop-generated depth, RTX 3050 Ti, one run per scene, 6000 iterations.

| Scene | PSNR | SSIM (3DGS) | SSIM (sklearn) | LPIPS (VGG) |
| --- | --- | --- | --- | --- |
| fern | 20.80 | 0.671 | 0.665 | 0.262 |
| flower | 19.04 | 0.578 | 0.571 | 0.300 |
| fortress | 20.49 | 0.451 | 0.408 | 0.326 |
| horns | 17.77 | 0.605 | 0.590 | 0.359 |
| leaves | 16.66 | 0.562 | 0.577 | 0.269 |
| orchids | 14.70 | 0.408 | 0.410 | 0.346 |
| room | 20.46 | 0.762 | 0.741 | 0.276 |
| trex | 19.93 | 0.711 | 0.703 | 0.241 |
| **Mean** | **18.73** | **0.594** | **0.583** | **0.297** |
| Paper (DNGaussian, Table 1) | 19.12 | 0.591 | — | 0.294 |

**Cross-machine check:** fern on the Colab T4 (Disha's depth, PyTorch 2.11) gives PSNR 20.80, SSIM 0.674, LPIPS 0.257, against the laptop's 20.80, 0.671 and 0.262.

**Nondeterminism:** fern at 1100 iterations, same seed, twice: PSNR 17.75 vs 17.29.

**Timing and memory:** laptop 7.5 min per full run, training footprint 1663 MB; T4 4.9 min per run.

**Published LLFF 3-view numbers** (from DropGaussian's CVPR 2025 table; to be verified against each paper; see `results/reported_baselines.csv`):

| Method | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| DropGaussian | 20.76 | 0.713 | 0.200 |
| CoR-GS | 20.45 | 0.712 | 0.196 |
| FSGS | 20.43 | 0.682 | 0.248 |
| SparseNeRF | 19.86 | 0.624 | 0.328 |
| FreeNeRF | 19.63 | 0.612 | 0.308 |
| 3DGS | 19.22 | 0.649 | 0.229 |
| DNGaussian | 19.12 | 0.591 | 0.294 |
