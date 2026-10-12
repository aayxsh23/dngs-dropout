# Sparse-View 3DGS Project — Team Playbook (You + Disha)

Oct 12, 2026 · @Aayush · **revised Mon 12 Oct evening** after the first measurements

> **What changed in this revision.** The first measurements and a close reading of the DNGaussian and DropGaussian code changed several assumptions. Sections that changed are marked **(revised)**. The hour-by-hour schedule, the full specs and the current status now live in [PLAN.md](PLAN.md); where the two disagree, PLAN.md wins.
>
> - **Measured:** Colab T4 **4.9 min/run**; laptop **7.5 min/run** with a **1663 MB** training footprint (it fits); laptop reproduction **18.73 / 0.594 / 0.297** against the paper's 19.12 / 0.591 / 0.294.
> - **Machines:** Tier 1 runs on Colab T4s (about 9.8 GPU-hours, about 5 h per account, Wed–Thu). Tier 2 runs on the laptop **overnight only** (Wed and Thu nights), since a full-load run makes it unusable in the daytime. The A100 is optional; access is confirmed on Tue.
> - **Variants:** DNGaussian renders depth in **two separate passes** (hard and soft) before the photometric pass, each with its own optimizer step. So D means "the dropout mask also hits the depth passes", and E costs **no extra render**.
> - **Dropout:** the official code is `torch.nn.Dropout(p=0.2*iteration/10000)` applied to a ones vector, then multiplied into opacity. We use γ = 0.2 with the schedule rescaled to our 6000 iterations.
> - **Files:** `metrics.py` writes **`results_eval.json`** (not `results.json`) and renders go to **`eval/`** (not `test/`). Training also needs **shencoder**. Checkpoints must load through `load_checkpoint()`, because the laptop runs **PyTorch 1.12 / Python 3.7** and Colab runs PyTorch 2.11 / Python 3.13.
> - **Storage:** a full upstream run is 337 MB (164 MB checkpoint, 13 MB TensorBoard). The runner deletes these, or 15 GB Drive quotas overflow.
> - **DropGaussian's initialization** is a COLMAP **dense stereo** point cloud from the sparse views, not plain SfM points.

## Read first: we have 8 days, not 22

Today is Monday 12 Oct and the deadline is Tuesday 20 Oct, so 8 working days remain. The handoff plan expected a working environment by 30 Sep, so we are about 12 days behind it and the plan below replaces its Section 9.

Three decisions make 8 days workable:

- **One codebase for every variant.** We add dropout and diagnostics to our own copy of DNGaussian behind command-line flags. Every variant is then the same code with different flags, which is both faster to build and a cleaner controlled experiment.
- **Parallel work from day one.** We agree the file formats today, so Disha can write the analysis scripts while you write the training code.
- **Hard scope tiers (revised).** Tier 1 is the report and runs on Colab T4s. Tier 2 runs on the laptop over two nights. Tier 3 is cut and replaced with published numbers.

| Tier | What it covers | Training runs | Rule |
| --- | --- | --- | --- |
| 1 Must | LLFF 3-view, variants A–E, 8 scenes, 3 seeds (2 minimum), diagnostics, tables, qualitative and failure-case figures | 120 | Non-negotiable. All on Colab T4 (one GPU type) |
| 2 Should | View sparsity **2, 3, 6, 9** for variants **B, C, D and E**, 8 scenes, seed 0 | **128** | **Laptop, overnight:** Wed→Thu (2- and 3-view), Thu→Fri (6- and 9-view). The 3-view point keeps the curve on one GPU. Drop 9-view first if short of time |
| 2b Could | D and E at γ ∈ {0.1, 0.3}, 3-view, seed 0 | 32 | Only if Tier 1 is complete by Thu night (Colab or A100) |
| 3 Cut | Blender 8-view, DTU, Mip-NeRF 360, re-running DropGaussian, DropoutGS, FSGS, CoR-GS, FreeNeRF, SparseNeRF | 0 | Use numbers from their papers, labelled "reported" |

If the real deadline turns out to be later, keep this order and add Tier 3 at the end; do not stretch Tier 1 to fill the time.

## Q1: Coding in VS Code on Windows

Use VS Code installed on Windows with the WSL extension. The window, shortcuts and file explorer are ordinary Windows VS Code, while Python, CUDA and git run inside Ubuntu where your three CUDA extensions are already built. A fully native Windows setup is possible (Option B) but means rebuilding everything with Microsoft's compiler, which is not worth it with 8 days left.

### Option A (recommended): Windows VS Code + WSL extension

1. Install VS Code from code.visualstudio.com using the Windows installer and tick "Add to PATH". Do not install VS Code inside Ubuntu.
2. In VS Code, open Extensions (Ctrl+Shift+X) and install **Remote Development** by Microsoft. It bundles the WSL extension you need here.
3. Keep the repository on the Linux side, for example `~/projects/dngs-dropout`, never under `/mnt/c/...`. Files on `/mnt/c` are much slower for git and Python inside WSL and pick up Windows line endings.
4. Open the project from the Ubuntu terminal with `cd ~/projects/dngs-dropout && code .`. The first launch installs a small VS Code server inside WSL. The bottom-left corner should then read **WSL: Ubuntu**.
5. Alternative way in: press F1, run **WSL: Connect to WSL**, then File > Open Folder and pick the folder under `/home/<you>/`.
6. Install these extensions into WSL (the Extensions panel shows an "Install in WSL: Ubuntu" button): Python, Pylance, Jupyter. Optional: C/C++ for reading the `.cu` rasterizer files.
7. Pick the interpreter: Ctrl+Shift+P, **Python: Select Interpreter**, then choose the `dngaussian` conda environment (path like `~/miniconda3/envs/dngaussian/bin/python`). New terminals will activate it automatically.
8. Open a terminal with Ctrl+\` . It is bash inside WSL, so you train exactly as you did during setup.
9. The Source Control panel uses the git inside WSL. Set your name, email and SSH key inside Ubuntu, not in Windows (see the GitHub section).
10. To view outputs: open rendered PNGs directly in VS Code, or browse `\\wsl$\Ubuntu\home\<you>\projects` in Windows Explorer. TensorBoard started in the WSL terminal opens at http://localhost:6006 in your Windows browser.

Create `.vscode/launch.json` so you can set breakpoints inside training (F5). The `args` below are the full flag set from `scripts/run_llff.sh` with 500 iterations; the laptop already has this file:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug: train fern, 500 iters",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/train_llff.py",
      "args": ["-s", "data/llff/fern", "--model_path", "output/debug/fern",
               "-r", "8", "--eval", "--n_sparse", "3", "--rand_pcd", "--iterations", "500",
               "--lambda_dssim", "0.2", "--densify_grad_threshold", "0.0013", "--prune_threshold", "0.01",
               "--densify_until_iter", "6000", "--percent_dense", "0.01",
               "--position_lr_init", "0.016", "--position_lr_final", "0.00016",
               "--position_lr_max_steps", "5500", "--position_lr_start", "500",
               "--split_opacity_thresh", "0.1", "--error_tolerance", "0.00025",
               "--scaling_lr", "0.003", "--shape_pena", "0.002", "--opa_pena", "0.001", "--near", "10"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {"CUDA_VISIBLE_DEVICES": "0"}
    }
  ]
}
```

### Option B (not recommended): fully native Windows

Only do this if WSL is unusable on your machine. You would need Visual Studio 2022 Build Tools with the "Desktop development with C++" workload, the Windows CUDA 11.8 installer, Miniconda for Windows and PyTorch built for cu118. The extensions must be compiled from the **x64 Native Tools Command Prompt for VS 2022** with `set DISTUTILS_USE_SDK=1` and `set TORCH_CUDA_ARCH_LIST=8.6`, then `pip install --no-build-isolation ./submodules/diff-gaussian-rasterization ./submodules/simple-knn`. Newer VS 2022 releases can be rejected by CUDA 11.8's version check, and the repo's `.sh` scripts will not run natively. You would also be maintaining a second environment that differs from Colab, which runs Linux.

### Disha's machine

Disha does not need CUDA locally. Plain Windows VS Code plus git and a small Python with pandas and matplotlib covers her analysis scripts, and all her GPU work runs on Colab. The `.gitattributes` file in the GitHub section stops her Windows line endings from breaking shell scripts.

## Roles and ownership

You own the training code and Disha owns the data, the Colab pipeline and the analysis. Every workstream has one owner who writes it and one reviewer who checks it, and both of you run experiments on your own Colab accounts.

Splitting by folder keeps you out of each other's files, so merge conflicts become rare:

- **You edit:** `train_llff.py`, `gaussian_renderer/`, `arguments/`, `utils/`, `experiments/run_matrix.py`, `experiments/matrix.yaml`.
- **Disha edits:** `notebooks/`, `experiments/aggregate.py`, `experiments/plots/`, `results/`.
- **Both edit, one at a time:** `README.md` and the report.

| Workstream | Owner | Reviewer | Done when |
| --- | --- | --- | --- |
| Private GitHub repo, `.gitignore`, `.gitattributes` | You | Disha | Disha has cloned it and pushed one test commit |
| Shared Drive folder, LLFF data, DPT depth maps | Disha | You | Depth maps for all 8 LLFF scenes are on Drive |
| Colab notebook with cached extension wheels | Disha | You | A fresh session reaches training in under 10 minutes |
| DNGaussian reproduction, untouched upstream code | Disha | You | 8-scene table compared with the paper's 19.12 PSNR, 0.591 SSIM, 0.294 LPIPS |
| Dropout ported from DropGaussian's official code | You | Disha | `drop_rate 0` reproduces baseline numbers; dropout never active at test time |
| Diagnostics logging | You | Disha | `diag.csv` written every 100 iterations |
| Decoupled-depth mode (variant E) | You | Disha | One flag moves the depth loss onto a clean, no-dropout render |
| Experiment runner `run_matrix.py` | You | Disha | Skips finished runs and resumes after a Colab disconnect |
| `aggregate.py` and result tables | Disha | You | Mean ± std tables built from all `metrics.json` files |
| Figures: diagnostic curves, qualitative grids | Disha | You | Scripts regenerate every figure from the outputs folder |
| Published-baseline numbers with citations | Disha | You | `results/reported_baselines.csv` has a source for every row |
| A100 go-bag disk and lab visits | You | Disha | `nvidia-smi` works at the lab, throughput measured, results copied back after every visit |
| Diagnostics interpretation and the final verdict | You | Disha | One paragraph: complementary, redundant or interfering, with evidence |
| Failure-case analysis | Both | Both | 2–3 scenes with renders and depth maps explained |

## Experiment design the code must support

All five variants come from one DNGaussian fork with four new flags. Depth on or off crossed with dropout on or off gives a 2×2 design, and variant E tests the fix. Because everything else stays identical, any difference between variants is caused by the regularizers, not by different codebases.

**How DNGaussian trains (revised; this shapes D and E).** Every iteration runs three separate rasterizations, each with its own backward pass and optimizer step:
1. **Hard depth** (`render_for_depth`): opacity is fixed at 0.95, and the depth loss moves positions only.
2. **Soft depth** (`render_for_opa`, from iteration 1000): positions, scales and rotations are frozen, and the depth loss moves opacity only.
3. **Photometric** (`render`): L1 + D-SSIM plus shape, scale and opacity penalties.

The depth loss never uses the photometric render. So "depth loss on the dropout render" means the dropout mask is also applied to the two depth passes.

| ID | Name | `--depth_scale` | `--drop_rate` | `--depth_render` | What it isolates |
| --- | --- | --- | --- | --- | --- |
| A | base | 0 | 0 | shared | DNGaussian with depth regularization switched off (both depth passes skipped) |
| B | depth | 1 | 0 | shared | DNGaussian as published |
| C | drop | 0 | 0.2 | shared | Dropout alone, same codebase |
| D | depth + drop, shared | 1 | 0.2 | shared | Naive combination: one mask per iteration hits all three passes |
| E | depth + drop, decoupled | 1 | 0.2 | clean | The mask hits only the photometric pass; both depth passes see the full model. **No extra render** |

How the results map to the research question:

- **Complementary:** D or E beats both B and C on mean PSNR, SSIM and LPIPS across seeds.
- **Redundant:** D is roughly equal to the better of B and C.
- **Interfering:** D is worse than B or C, and the diagnostics show a noisier depth signal under dropout. If E then recovers the loss, the interference runs through depth supervision specifically.

### Implementation rules (you)

1. **Add `--seed`.** ✅ *Done on `feat/dropout`.* Code derived from 3DGS fixes all seeds to 0 inside `safe_state()` in `utils/general_utils.py`. Pass `args.seed` into it, otherwise "three seeds" are three copies of one run. `--rand_pcd` draws the initial cloud from `np.random`, so it now changes per seed.
2. **Add `--depth_scale`.** ✅ *Done.* One multiplier on both depth passes in `train_llff.py`. At `0` both passes are **skipped entirely**. Multiplying by zero would still let Adam momentum move the parameters and would add extra optimizer steps.
3. **Port dropout, do not rewrite it (revised).** The official code ([DCVL-3D/DropGaussian\_release](https://github.com/DCVL-3D/DropGaussian_release), `gaussian_renderer/__init__.py`) is `compensation = torch.nn.Dropout(p=0.2*iteration/10000)(torch.ones(N))`, then `opacity = opacity * compensation[:, None]`. That is the mask and the 1/(1−r) compensation in one step; 10000 is their total iteration count. We use γ = 0.2 and r = 0.2 × t ÷ 6000. Mention in the report that the rate was tuned on vanilla 3DGS and the schedule was rescaled. Exact code: PLAN.md W2.
4. **Training only.** Dropout must be active only inside the training loop. `render.py` and every evaluation path must never apply a mask. Render functions get a `drop_comp=None` argument that only the training loop sets.
5. **Check densification (revised).** DropGaussian does nothing special here: densification uses the dropout render's `visibility_filter` and `radii`. Radii don't depend on opacity, so dropped Gaussians count as visible with zero gradient that iteration. Mirror that, which means changing nothing.
6. **Decoupled mode (revised).** With `--depth_render clean`, the two depth passes get no mask and only the photometric pass is dropped. DNGaussian already renders depth separately, so this costs **no extra render**.
7. **Initialization differs from the DropGaussian paper (revised).** DropGaussian builds its initial cloud by running COLMAP on the sparse views (SfM, then `patch_match_stereo` and `stereo_fusion`: a dense stereo cloud). DNGaussian initializes randomly. Our variant C inherits random initialization, so it will not match DropGaussian's published 20.76 PSNR, and the report must say why.
8. **One SSIM (revised).** `metrics.py` reports both: `SSIM` (3DGS implementation) and `SSIM_sk` (scikit-image). **The main table uses 3DGS SSIM**, which matches the paper: our reproduction gives 0.594 vs the paper's 0.591, while sklearn gives 0.583. Both go into `metrics.json`. LPIPS uses the VGG backbone; name both in the report.
9. **Two Python and PyTorch versions (new).** The laptop runs Python 3.7 / PyTorch 1.12, Colab runs Python 3.13 / PyTorch 2.11. Training code must use Python 3.7 syntax and load checkpoints with `utils.general_utils.load_checkpoint()` (PR #3).
10. **Opacity penalty (new).** `opa_pena` uses the **pre-dropout** opacity, so the regularizer is identical across variants.
11. **Random point-cloud race (new).** `--rand_pcd` writes `sparse/0/points3D_random.ply` into the dataset folder, so parallel runs on one scene race on that file. Write it to a temporary directory instead (PLAN.md W7).

### Diagnostics: `diag.csv`, one row every 100 iterations

| Column | Meaning |
| --- | --- |
| `iter` | Iteration number |
| `loss_rgb` | Photometric loss used for the update |
| `loss_depth_used` | Hard + soft depth loss on the renders that receive gradients in this variant: masked in C and D, unmasked in A, B and E. Computed without gradient at a **fixed patch size of 11**, so it is comparable with the next column |
| `loss_depth_clean` | The same loss on unmasked renders, logged in every variant. Equal to `loss_depth_used` in A, B and E by construction |
| `depth_var_masks` | Mean per-pixel variance of soft-pass depth across 4 random masks at one fixed training view, at a **fixed probe rate of 0.2 in every variant**. A separate RNG keeps it from changing training |
| `n_gaussians` | Current Gaussian count |
| `drop_rate_now` | Current rate from the schedule |
| `iter_ms` | Mean milliseconds per iteration since the previous row |
| `vram_peak_mb` | `torch.cuda.max_memory_reserved()` in MB |

The central piece of evidence is the gap between `loss_depth_used` and `loss_depth_clean` in variant D. It measures directly how much dropout perturbs the depth signal DNGaussian learns from. The diagnostics are computed right after the photometric backward pass and **before** densification, which runs on the same multiples of 100. Full spec: PLAN.md W4.

### Output layout and `metrics.json` (agreed; add keys, never rename) (revised)

```text
outputs/llff_3v/<variant>/<scene>/seed<k>/
    config.json          all arguments + git commit hash + GPU name (written by train_llff.py)
    opt_args, cfg_args   upstream's own argument dumps
    diag.csv
    train_stats.json     train_time_s, n_gaussians, vram_peak_mb, model_size_mb, torch_version
    render_stats.json    fps from a timed render loop (render.py)
    results_eval.json    metrics.py output: {"ours_6000": {"PSNR", "SSIM", "SSIM_sk", "LPIPS"}}
    metrics.json         one flat record built by the runner (below), written last
    run.log              stdout/stderr of all three steps
    eval/ours_6000/      renders/, gt/, depth/ for the held-out views (all seeds, about 3 MB)
    input.ply            the random initial cloud
    point_cloud/, chkpnt_latest.pth   seed 0 only. TensorBoard events are deleted
```

```json
{
  "variant": "D", "scene": "fern", "n_views": 3, "seed": 0,
  "psnr": 0.0, "ssim": 0.0, "ssim_sk": 0.0, "lpips": 0.0,
  "train_time_s": 0.0, "fps": 0.0, "n_gaussians": 0,
  "model_size_mb": 0.0, "vram_peak_mb": 0.0, "wall_time_s": 0.0,
  "git_commit": "abc1234", "code_tag": "v1.0-exp", "gpu": "Tesla T4", "shared_gpu": false,
  "torch_version": "2.11.0+cu130", "drop_rate": 0.2, "depth_scale": 1, "depth_render": "shared"
}
```

Disha writes `aggregate.py` against hand-made files in exactly this shape on Monday, so it works the moment real results land.

## GitHub setup and daily workflow

One private repository, owned by you and built from your already-working DNGaussian clone, holds all code; Disha joins as a collaborator. Data, depth maps and training outputs never go into git; they live on Google Drive.

### Step 1 (you, Monday morning): turn your working clone into the team repo

> ✅ **Done 12 Oct (revised).** The repo is `github.com/aayxsh23/dngs-dropout`, **public** (briefly private on 12 Oct evening; public again so the GitHub ruleset can enforce the merge gate). Tag `upstream-baseline` = `81c3d83`. The nested `.git` folders were three, not two (`third_party/glm` inside the rasterizer too). The tracked `gridencoder`/`shencoder` egg-info folders needed `git rm -r --cached`. The commands below are kept for reference.

The DNGaussian README has you clone `diff-gaussian-rasterization` and `simple-knn` inside `submodules/`, so they are nested git repositories. Pushed as they are, they arrive on GitHub as empty folders. Remove their `.git` folders so the source is committed directly, along with any compile fixes you made during setup.

```bash
# one-time identity and SSH key inside Ubuntu (WSL)
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
ssh-keygen -t ed25519 -C "you@example.com"
cat ~/.ssh/id_ed25519.pub     # paste into GitHub > Settings > SSH and GPG keys
ssh -T git@github.com         # should greet you by username

cd ~/projects/DNGaussian      # the clone you built during setup
git remote rename origin upstream
find submodules -name .git -prune -exec rm -rf {} +

cat >> .gitignore <<'EOF'
data/
output/
outputs/
*.ply
*.pth
*.pt
build/
*.egg-info/
*.so
__pycache__/
.ipynb_checkpoints/
EOF

cat > .gitattributes <<'EOF'
* text=auto
*.sh text eol=lf
*.py text eol=lf
*.yaml text eol=lf
EOF

conda env export --no-builds > environment.lock.yml
git add -A
git status    # submodules/ source must be listed; no images, .ply or .pt files
git commit -m "Import DNGaussian with vendored rasterizer and simple-knn"
git tag upstream-baseline
```

On github.com, create a new **private** repository called `dngs-dropout` and leave README, `.gitignore` and licence unticked. Then push:

```bash
git remote add origin git@github.com:<your-username>/dngs-dropout.git
git push -u origin main
git push origin upstream-baseline
```

Add Disha under the repo's Settings > Collaborators > Add people. She accepts the emailed invite.

### Step 2 (Disha): get the code

1. Install Git for Windows; its Credential Manager handles GitHub login in the browser.
2. Clone with `git clone https://github.com/<your-username>/dngs-dropout.git` and open the folder in VS Code.
3. For Colab, create a classic personal access token with the `repo` scope and a 30-day expiry (GitHub > Settings > Developer settings). Store it only in Colab's Secrets panel as `GH_TOKEN`; never paste it into a notebook cell. You need your own token too for your Colab account.

If you prefer narrower fine-grained tokens, put the repo in a free GitHub organization that you both own, because fine-grained tokens only reach repositories owned by the token holder or their organization.

### Step 3 (both): the daily loop

```bash
git checkout main && git pull
git checkout -b feat/dropout            # one branch per task: feat/, fix/, exp/
# edit, then test on one scene with few iterations
git add -p                              # review every change you stage
git commit -m "Add --drop_rate with DropGaussian mask and schedule"
git push -u origin feat/dropout
```

Open a pull request on GitHub and write "Closes #\<issue>" in it. The other person reviews within 3 hours, then **Squash and merge**, delete the branch, and both run `git pull`.

### Rules that protect the results

- `main` must always train end to end. Untested code stays on a branch.
- Experiments run only from a tag. The freeze tag is `v1.0-exp`, and `config.json` in every run records the commit hash.
- A bug found after the freeze gets a fix branch and tag `v1.1-exp`; rerun only the affected runs and note it in `RUNLOG.md`.
- Small outputs (CSV tables, figure PDFs) are committed under `results/`. Anything large stays on Drive.

```bash
# code freeze (you, Wednesday 14 Oct, 1 PM)
git checkout main && git pull
git tag -a v1.0-exp -m "Code freeze for main experiment matrix"
git push origin v1.0-exp
```

### Folders added to the repo

```text
experiments/matrix.yaml        variants, scenes, seeds (you)
experiments/matrix_tier2.yaml  views 2/3/6/9, variants B-E, seed 0, laptop nights (you)
experiments/matrix_smoke.yaml  fern, 1100 iterations, all variants: pre-freeze check (you)
experiments/run_matrix.py      resumable runner (you)
experiments/stats.py           paired comparisons and verdict rules (Disha)
experiments/fixtures/          fake metrics.json for testing aggregate.py (Disha)
experiments/aggregate.py       metrics.json -> tables (Disha)
experiments/plots/             figure scripts (Disha)
notebooks/colab_runner.ipynb   Colab pipeline (Disha)
results/reported_baselines.csv published numbers + citations (Disha)
results/tables/  results/figures/
RUNLOG.md                      what ran where, problems, reruns
environment.lock.yml           exact laptop environment
setup_env.sh                   laptop install, step by step (PR #2)
pip_freeze_laptop.txt          resolved pip versions on the laptop
scripts/peak_vram.py           peak GPU memory of one training run
docs/PLAN.md                   current plan, status, specs (overrides this playbook)
docs/research_proposal_sparse_view_3dgs_2.md   this playbook
AGENTS.md, CLAUDE.md, GEMINI.md, .github/copilot-instructions.md   instructions for AI coding agents
```

Track work as GitHub Issues: one issue per row of the ownership table, assigned to its owner and closed by the pull request that finishes it.

## Shared Google Drive layout

Disha creates one Drive folder, `sparse3dgs`, and shares it with you as Editor. Every machine reads data from it and writes results into it, so a run finished on any machine is immediately visible to both of you.

```text
sparse3dgs/
    data/llff_with_depth.tar     8 LLFF scenes + DPT depth, one archive
    weights/                     DPT weights file the dpt/ scripts expect
    data/llff_with_depth.manifest.json   md5 0f6bd7bdae837ec58ad757c7c00b0561 and per-scene counts
    wheels/torch-<version>-src-<hash>/   compiled CUDA extension wheels for Colab, keyed by source hash
    outputs/llff_3v/<variant>/<scene>/seed<k>/   Tier 1 (Colab)
    outputs/laptop/llff_{2,3,6,9}v/...          Tier 2 (laptop), copied up Fri morning
    outputs/repro_upstream/<scene>/              T4 reproduction at upstream-baseline (Cell 7a)
    logs/                        one log file per shard
    run_tracker                  Google Sheet: who runs which shard where
```

Four details that save hours:

1. **Add a shortcut, or Colab cannot see it.** A folder shared with you does not appear under `/content/drive/MyDrive`. In Drive, right-click `sparse3dgs`, choose Organize > Add shortcut, and place it in My Drive. It then appears at `/content/drive/MyDrive/sparse3dgs` for both of you.
2. **Store data as one archive.** Thousands of small image files copy very slowly out of Drive. Generate the DPT depth maps once, then `tar -cf llff_with_depth.tar llff/` and upload that single file. Each Colab session extracts it to local disk in a minute or two.
3. **Generate depth exactly once.** Both of you must train on identical DPT depth maps, so never regenerate them per machine. Run `cd dpt && python get_depth_map_for_llff_dtu.py --root_path <llff folder> --benchmark LLFF` once, on Colab or the laptop.
4. **Watch the quota (revised).** In a shared folder each file counts against the quota of whoever created it. A full upstream run is **337 MB** (164 MB checkpoint, 13 MB TensorBoard, 127 MB spiral renders). The runner skips the spiral, deletes TensorBoard events everywhere, and keeps the checkpoint and point cloud for seed 0 only. That leaves about 4 MB per run (seeds 1–2) and about 170 MB (seed 0): roughly 7 GB for Tier 1, split across both accounts. On Colab the runner trains on local disk (`--work_root /content/work`) and copies the slimmed run to Drive at the end.

The run tracker sheet has one row per shard: shard id, machine, person, start time, status and notes. Update it whenever you start or finish a shard, so two machines never run the same jobs.

## Day-by-day plan, 12–20 Oct

> **Revised:** the up-to-date, hour-by-hour version of this schedule is **PLAN.md §4**. In short: code freeze unchanged (Wed 14 Oct, 1 PM). Tier 1 runs on Colab Wed 13:15 to Thu ~14:00. Tier 2 runs on the laptop on Wed and Thu nights (about 23:30 to 07:30). Tier 1 freeze stays Fri midnight but should be met a day early. The weekend is for writing, not runs. The checklists below are kept for reference; Monday's items are ticked.

Code freezes Wednesday 14 Oct at 1 PM. Tier 1 freezes Friday 16 Oct at midnight, with Saturday's lab visit as its only overflow. Tier 2 runs on the A100 at the weekend while you write, and freezes Sunday 18 Oct in the evening. Tuesday 20 is buffer.

&#91;embedded content: team plan · 12–20 Oct, by person\]

Bars are working spans; the dashed line is the code freeze, and no main-matrix run starts before it.

### Mon 12 Oct: set up and measure

You:

- [x] Turn your working clone into the private repo and add Disha (GitHub section, Step 1)
- [x] Commit `setup_env.sh` with the exact install commands that worked on the laptop (PR #2)
- [x] Full `fern` run on the laptop: record minutes per run and peak VRAM (7.5 min, 1663 MB; all 8 scenes reproduced)
- [ ] Ask the lab admin the questions in the A100 section: operating system, WSL2, USB, lab hours (*A100 now optional*)
- [ ] ~~Start the go-bag disk~~ *Reduced to a light go-bag, only if the lab is confirmed (PLAN.md W13)*
- [x] Read DropGaussian's official code and note the exact lines for the mask, compensation and schedule (rule 3 above)

Disha:

- [x] Accept the GitHub invite, clone, open in VS Code
- [x] Create the `sparse3dgs` Drive folder and share it; both add the shortcut to My Drive
- [x] Download LLFF, generate DPT depth once, upload `llff_with_depth.tar` (md5 `0f6bd7bd…`)
- [x] Get the Colab notebook working through Cell 6, with wheels cached on Drive (PR #1; T4 timing 4.9 min/run)

Both:

- [ ] 30-minute call to fix variant names, flags, and the `metrics.json` and `diag.csv` formats; they do not change after today

Gate: one complete train, render and metrics run on at least one machine.

### Tue 13 Oct: build and reproduce

You:

- [ ] Add `--seed`, `--depth_scale`, the dropout port, `--depth_render clean`, `diag.csv` and `train_stats.json`
- [ ] Check that `drop_rate 0` gives baseline-like numbers on `fern` and that `render.py` never drops
- [ ] Write `matrix.yaml`, `matrix_tier2.yaml` and `run_matrix.py` with `--seeds` and the `STOP` file; `--dry_run` prints correct commands
- [ ] Open the pull request by 10 PM
- [ ] Finish the go-bag disk: data archive, `go.sh`, `collect.sh`; test both scripts on the laptop with 2 slots

Disha:

- [ ] Reproduce DNGaussian on all 8 scenes with Cell 7a at tag `upstream-baseline`
- [ ] Per-scene table against the paper's 19.12 PSNR, 0.591 SSIM, 0.294 LPIPS
- [ ] Write `aggregate.py` against hand-made `metrics.json` files
- [ ] Start Related Work and `reported_baselines.csv`

Gate: reproduction mean within about 0.5 dB of the paper, or the fallback applies.

### Wed 14 Oct: freeze and launch

- [ ] Disha reviews the pull request by 11 AM, including a regression check: variant B on 2 scenes lands close to her reproduction
- [ ] You merge and push tag `v1.0-exp` by 1 PM, then make a fresh `dngs.bundle` for the disk
- [ ] Both start Colab with `--seeds 0`, shards 0/2 and 1/2; each run has its T4 to itself, so these runs also supply the efficiency table
- [ ] Lab visit 1, if access is granted: identify the machine, set up from the go-bag, smoke test, measure throughput, launch seed 1
- [ ] Before leaving the lab: `touch STOP`, wait for runs to finish, `collect.sh`, upload to Drive in the evening
- [ ] You sanity-check the first finished `diag.csv` files

Gate: the first 10 runs finish with sensible metrics.

### Thu 15 Oct: seeds and diagnostics

- [ ] Lab visit 2: finish seed 1, then start seed 2; drain with `STOP` and collect before leaving
- [ ] Colab: finish seed 0 and rerun any failures
- [ ] You: plot the depth-loss gap and depth variance for D against E and draft the verdict
- [ ] Disha: seed-0 tables and the qualitative render grid
- [ ] You: start the Method and Experimental Setup sections
- [ ] Tonight: if the A100 is still not working, Colab takes seeds 1 and 2 (cut to 2 seeds if runs are slow)

### Fri 16 Oct: finish experiments

- [ ] Lab visit 3: finish seed 2 and any reruns; collect and upload
- [ ] Pick 2–3 failure-case scenes and render RGB and depth side by side
- [ ] Disha: final mean ± std tables; every figure regenerates from a script
- [ ] Midnight: Tier 1 freeze; anything unfinished goes first in Saturday's lab visit

### Sat 17 to Mon 19 Oct: write, with Tier 2 running at the lab

- [ ] Saturday at the lab: finish any Tier 1 leftovers, then launch Tier 2 with `go.sh <transfer> "0" <slots> experiments/matrix_tier2.yaml`; write a full draft of your sections on the laptop while it runs
- [ ] Saturday: Disha drafts her sections
- [ ] Sunday at the lab: Tier 2 finishes by evening (Tier 2 freeze); collect and upload; figures in place; swap sections and review each other's; repeat the arXiv check
- [ ] Sunday night: Disha adds the views plot from Tier 2
- [ ] Monday: revise, finish the README, check citations, push tag `v1.0-report`

### Tue 20 Oct: buffer and submit

- [ ] Final read-through together, then submit

## Dependencies and the critical path

Your dropout code and Disha's reproduction must both land before the Wednesday freeze, and every result, figure and paragraph after it depends on the main matrix. Agreeing the file formats on Monday is what lets the two tracks run in parallel instead of in sequence.

&#91;embedded content: dependencies · two tracks, one gate\]

The critical path runs down the middle: a slip at the freeze moves every later box by the same amount, so protect Wednesday 1 PM above everything else.

| Who waits | What they need | From | By |
| --- | --- | --- | --- |
| Disha | Access to the GitHub repo | You | Mon 12 Oct, noon |
| Both | Agreed flags, `metrics.json` and `diag.csv` formats | Monday call | Mon 12 Oct, evening |
| Every machine | `llff_with_depth.tar` on Drive, copied onto the go-bag disk | Disha | Mon 12 Oct, night |
| You | A working Colab notebook for your shard | Disha | Tue 13 Oct, noon |
| Disha | Pull request with dropout, diagnostics and runner | You | Tue 13 Oct, 10 PM |
| A100 visits | Go-bag disk: packed env, WSL export, data, scripts | You | Tue 13 Oct, night |
| You | Reproduction numbers and the PR review | Disha | Wed 14 Oct, 11 AM |
| Both | Tag `v1.0-exp`, and a fresh bundle on the disk | You | Wed 14 Oct, 1 PM |
| Disha | A100 results uploaded to Drive | You | Evening of each lab visit |
| Disha | Finished `metrics.json` files | Matrix runs | Thu 15 Oct onward |
| You | Diagnostic plots from Disha's scripts | Disha | Fri 16 Oct |

Work nobody waits on fills the gaps: Disha writes `aggregate.py` against hand-made files and starts Related Work on Tuesday, and you develop the dropout code on your laptop's local data while her depth maps are still being uploaded.

## Q3: Where to run the experiments

> **Revised (12 Oct evening).** Measured: a T4 run takes **4.9 min**, so all of Tier 1 (120 runs, about 9.8 GPU-hours) fits on the two Colab accounts over Wed–Thu, about 5 h each. The new allocation:
>
> | Work | Machine | When |
> | --- | --- | --- |
> | Tier 1, seeds 0, 1, 2 | Colab T4, both accounts (`--shard 0/2` and `1/2`) | Wed 13:15 to Thu evening |
> | Tier 2 (views 2/3/6/9, B–E, seed 0, 128 runs) | Laptop, **nights only**, one run at a time, about 7.5 min each | Wed 23:30–Thu 07:30, Thu 23:30–Fri 07:30 |
> | Development, smoke tests, writing | Laptop, daytime (no long runs: it's unusable under full GPU load) | Throughout |
> | Optional Tier 2b, overflow | A100 if the lab works out; otherwise Colab | Fri |
>
> Every Tier 1 seed then runs on the same GPU type, so the seed std is not mixed with GPU effects. The A100 is no longer on the critical path. The original reasoning is kept below.

Use every machine at once: the laptop for coding and quick tests, two Colab accounts as the always-available runners that also cover nights, and the A100 during lab visits from Wednesday 14 Oct, weekends included. Work is split by whole seeds, so each comparison stays on one GPU type.

| Machine | GPU | Use it for | Limits |
| --- | --- | --- | --- |
| Your laptop (WSL) | RTX 3050 Ti, 4 GB | Coding, debugging, 500-iteration smoke tests; preparing the A100 go-bag disk | 4 GB VRAM; slow; busy while you code |
| Colab free, your account | T4, 16 GB | Half of seed 0 (`--shard 0/2`); nights while the lab is closed | Session cap, idle disconnects, GPU not always granted, hidden usage limits |
| Colab free, Disha's account | T4, 16 GB | Other half of seed 0 (`--shard 1/2`); then any seed the A100 cannot take | Same as above |
| A100 lab machine | A100 | Seeds 1 and 2 in lab visits Wed–Fri; Tier 2 on Sat–Sun | Physical access only, from Wed 14 Oct; runs only while you are there; operating system not yet known |

### Decision rule

1. **Monday:** ask the lab admin the questions in the A100 section, and build the go-bag disk on Monday and Tuesday.
2. **Seed 0 always runs on Colab,** on both accounts from the Wednesday 1 PM freeze, so Tier 1 has one guaranteed seed even if the lab falls through.
3. **A100 working on Wednesday:** it runs seed 1, then seed 2, in visits Wednesday to Friday, and Tier 2 on Saturday and Sunday.
4. **A100 slow or not working:** Colab takes seeds 1 and 2 on Thursday and Friday; cut to 2 seeds if a T4 run takes over 20 minutes. The A100 then only runs Tier 2 at the weekend, if it works by then.
5. **Colab refuses a GPU repeatedly:** Kaggle Notebooks give a weekly quota of free GPU hours and run the same notebook with small path changes. One month of Colab Pro for one of you is the paid fallback.

### Does a full run fit on your laptop?

**Answered: yes.** A full fern run takes **7.5 min**. The training footprint is **1663 MB** over an 846 MB desktop baseline (2509 MB of 4096 MB peak), measured with `scripts/peak_vram.py`; under WSL, nvidia-smi can't show per-process memory. One run at a time only, since two would exceed 4 GB. Run long jobs in `tmux` or with `nohup`: a job started in a VS Code terminal dies when VS Code restarts.

### Two rules for fair comparisons

- **One GPU type per comparison.** The 3DGS rasterizer is not bit-for-bit deterministic (measured: fern, 1100 iterations, same seed twice gave 17.75 and 17.29 PSNR), and different GPUs drift further apart. All five variants of a given seed run on the same GPU type, and `metrics.json` records which.
- **Efficiency numbers from a quiet GPU.** Training time, FPS and VRAM come from one dedicated run per variant with nothing else on the GPU (`shared_gpu: false`). Parallel runs on an A100 are for quality metrics only.

## Colab runbook

> **Revised:** the real notebook is `notebooks/colab_runner.ipynb` in the repo (PR #1). Open it from GitHub: `https://colab.research.google.com/github/aayxsh23/dngs-dropout/blob/main/notebooks/colab_runner.ipynb`, or in Colab go to File → Open notebook → GitHub. Don't use the old copy on Drive. The cells below are the original sketch. Where they differ, the notebook is right: Colab is Python 3.13 / PyTorch 2.11 / CUDA 13; Cell 3b holds the build patches; Cell 4 builds **shencoder** too and keys wheels by source hash; Cell D generated the depth archive once; Cell 7b asserts the `v1.0-exp` tag.

One notebook, `notebooks/colab_runner.ipynb`, run top to bottom at the start of every session. The first session compiles the CUDA extensions and saves them as wheels on Drive; every later session installs those wheels and reaches training in a few minutes.

Before the first run: Runtime > Change runtime type > **T4 GPU**. Open the key icon (Secrets) on the left, add `GH_TOKEN`, and switch on notebook access.

**Cell 1: check the GPU and versions.** The CUDA version printed by `nvcc` must have the same major number as `torch.version.cuda`, or the extension build fails.

```python
!nvidia-smi
import torch, sys
print(sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
!nvcc --version | tail -1
```

**Cell 2: mount Drive.**

```python
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/sparse3dgs'
import os
assert os.path.isdir(ROOT), "Add the shared folder to My Drive as a shortcut first"
```

**Cell 3: clone the code at a fixed tag.** Use `upstream-baseline` for Disha's reproduction runs and `v1.0-exp` for the main matrix.

```python
from google.colab import userdata
tok = userdata.get('GH_TOKEN')
OWNER, TAG = '<your-username>', 'v1.0-exp'
!rm -rf /content/dngs-dropout
!git clone -q https://{tok}@github.com/{OWNER}/dngs-dropout.git /content/dngs-dropout
%cd /content/dngs-dropout
!git checkout -q {TAG} && git log -1 --oneline
```

**Cell 4: install the CUDA extensions, building them only once.** Wheels are stored per PyTorch version, so if Colab upgrades PyTorch this cell rebuilds automatically. Building for 7.5, 8.0 and 8.6 makes the same wheels work on T4, A100 and your laptop's architecture.

```bash
%%bash
set -e
cd /content/dngs-dropout
TV=$(python -c "import torch;print(torch.__version__)")
W=/content/drive/MyDrive/sparse3dgs/wheels/torch-$TV
if ! ls $W/*.whl >/dev/null 2>&1; then
  mkdir -p $W
  export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
  pip wheel --no-build-isolation --no-deps -w $W \
      ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ./gridencoder
fi
pip install -q $W/*.whl
```

Include `./gridencoder` (and `./shencoder`, if training imports it) only if that folder has a `setup.py`. If it compiles itself on first import instead, run `os.environ['TORCH_EXTENSIONS_DIR'] = ROOT + '/torch_ext'` before training so the compiled copy is reused.

**Cell 5: Python packages.** Install the non-PyTorch packages from `environment.yml`. This list is a starting point; add whatever an `ImportError` names, then commit the final list as `requirements_colab.txt`.

```python
!pip install -q plyfile lpips opencv-python-headless imageio timm pyyaml tensorboard
```

**Cell 6: copy data to the fast local disk.**

```python
!mkdir -p /content/data && tar -xf {ROOT}/data/llff_with_depth.tar -C /content/data
!ls /content/data/llff
```

**Cell 7a (Disha, Tuesday): reproduce DNGaussian with the authors' own script.** Use the `upstream-baseline` tag in Cell 3. **(Revised)** `metrics.py` writes `results_eval.json`, not `results.json`.

```python
SCENES = ['fern','flower','fortress','horns','leaves','orchids','room','trex']
for s in SCENES:
    out = f'{ROOT}/outputs/repro_upstream/{s}'
    if os.path.exists(f'{out}/results_eval.json'):
        continue
    !bash scripts/run_llff.sh /content/data/llff/{s} {out} 0
```

**Cell 7b (both, from Wednesday): run your shard of the matrix.** Each person uses a different shard number; Colab runs seed 0, plus any seed the A100 cannot take; record every shard in the run tracker.

```python
SHARD = '0/2'   # you: 0/2, Disha: 1/2
SEEDS = '0'     # revised: then '1', then '2' (all of Tier 1 runs on Colab)
!python experiments/run_matrix.py --matrix experiments/matrix.yaml \
    --data_root /content/data/llff --out_root {ROOT}/outputs --work_root /content/work \
    --shard {SHARD} --seeds {SEEDS} \
    2>&1 | tee -a {ROOT}/logs/colab_shard_{SHARD.replace('/', 'of')}_seed{SEEDS}.log
```

### Known build errors on newer compilers

| Error message | Fix (commit it once, it is harmless on the laptop) |
| --- | --- |
| `FLT_MAX` is undefined, in simple-knn | Add `#include <float.h>` at the top of `submodules/simple-knn/simple_knn.cu`. **Committed in PR #3** |
| `uint32_t` or `uintptr_t` not declared, in the rasterizer | Add `#include <cstdint>` to `cuda_rasterizer/rasterizer_impl.h`. **Already present** in the vendored rasterizer |
| gridencoder/shencoder fail with PyTorch ≥ 2.1 (C++17 required) | `-std=c++14` → `-std=c++17` in their `setup.py` and `backend.py`. **Committed in PR #3**; still builds on the laptop (nvcc 11.3, gcc 9) |
| `_pickle.UnpicklingError` / `weights_only` when `render.py` loads `chkpnt_latest.pth` on PyTorch ≥ 2.6 | Use `utils.general_utils.load_checkpoint()` (**PR #3**). Don't add `weights_only=False` directly: PyTorch 1.12 on the laptop has no such argument and raises `TypeError` |
| nvcc 11.3 rejects gcc 13, or `__s128` errors (laptop, Ubuntu 24.04) | gcc-9 wrapper on `PATH` and Linux 5.4 headers on `CPATH`; see `setup_env.sh` |
| `np.float`, `np.int` or similar missing | Old NumPy aliases: replace with `float` / `int`, or `pip install "numpy<2"` and restart the runtime |
| CUDA version mismatch between `nvcc` and PyTorch | Major versions must match; a minor-version warning is safe to ignore |

### Surviving disconnects

- Keep the tab open and the computer awake, and check in about once an hour; idle disconnects come from browser inactivity.
- Plan for sessions of about 10 hours. The runner skips any run that already has `metrics.json`, so a disconnect loses only the run in progress: rerun Cells 1–6, then 7b.
- Never run two GPU notebooks on the same account at once.

## A100 lab runbook: physical access only

> **Revised: the A100 is now optional** (see Q3). Send the admin questions anyway. Build only a light go-bag (git bundle, data tar, conda-pack), and only if access is confirmed. Uses, in order: (1) a **whole** seed that Colab can't finish; (2) optional Tier 2b. The runbook below stays valid for those cases, with the corrected step 1.

The A100 has no remote access, runs only while you are sitting at it, and is available from Wednesday 14 Oct at the earliest, weekends included. So you prepare everything on your laptop, carry it on your 500 GB disk, set up in minutes, run whole seeds during each visit, stop cleanly before you leave, and let Colab cover the nights.

Three habits make this safe on a shared lab machine: code arrives as a git bundle and data from the disk, so you never log into GitHub or Google there; results go back onto the disk at the end of every visit; and you bring your laptop to code, analyse and write while the A100 trains.

### Ask the admin today

- [ ] Which operating system does the A100 machine run, and will I have admin rights?
- [ ] If it is Windows: is WSL2 enabled, or can it be enabled (needs admin and a reboot)?
- [ ] Are USB drives allowed on that machine?
- [ ] Lab hours on weekdays and on Saturday and Sunday; can I book long blocks?
- [ ] Is the GPU shared with other users while I am there?
- [ ] Driver version from `nvidia-smi`, and free local disk space
- [ ] Do `pip`, `conda` and github.com work from the command line, or only in the browser (proxy)?
- [ ] Is my home folder kept between visits?

### The go-bag disk (you, Mon 12 to Tue 13 Oct, on your laptop)

Format the disk as exFAT so both Windows and Linux can read it. Keep the environment and data as archives, because exFAT loses Linux file permissions on loose files. You don't know the A100's operating system yet, so the disk carries a Linux option and a Windows option.

```text
E:\transfer\
    code\dngs.bundle              the git repo with tags; no credentials needed
    env\dngaussian.tar.gz         conda-pack of your env (if the A100 runs Linux)
    wsl\ubuntu-dngs.tar           export of your whole WSL Ubuntu (if it runs Windows)
    data\llff_with_depth.tar      the same archive as on Drive
    scripts\go.sh  scripts\collect.sh
    results\                      everything you bring back, one folder per visit
```

**1. Rebuild the extensions for both GPUs (revised).** Your extensions were compiled for your laptop's architecture (8.6); the A100 is 8.0. The laptop env is PyTorch **1.12.1 built for CUDA 11.3**, so **do not** install a CUDA 11.8 toolkit into it. Build for both architectures with the laptop's system CUDA 11.3 (which supports sm_80). Then no compiler is needed on site. Use the build environment from `setup_env.sh` step 6, and include all four extensions:

```bash
conda activate dngaussian
export TORCH_CUDA_ARCH_LIST="8.0;8.6" MAX_JOBS=2
pip install --no-build-isolation --no-deps --force-reinstall \
    ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ./gridencoder ./shencoder
(cd ~ && python -c "import torch, diff_gaussian_rasterization, _gridencoder, _shencoder; from simple_knn._C import distCUDA2; print('ok')")
```
If an import fails on site with "no kernel image", recompiling there needs a CUDA 11.x toolkit on that machine; otherwise fall back to Colab.

**2. Linux option: pack the env.** If `conda pack` refuses an editable package, reinstall that package without `-e` and pack again.

```bash
conda install -y -c conda-forge conda-pack
conda pack -n dngaussian -o /mnt/e/transfer/env/dngaussian.tar.gz
```

**3. Windows option: export your whole Ubuntu.** This carries the env, the compiled extensions and your shell setup in one file.

```powershell
wsl -l -v                      # your distro's exact name
wsl --shutdown
wsl --export Ubuntu E:\transfer\wsl\ubuntu-dngs.tar
```

**4. Code.** Make the bundle on your laptop, and again at the lab after the Wednesday freeze, so it contains tag `v1.0-exp`:

```bash
git bundle create /mnt/e/transfer/code/dngs.bundle --all
```

**5. Two scripts.** `go.sh` copies data to the local drive, checks the extensions, and launches several shards with one automatic retry pass each. `collect.sh` copies results to a new dated folder on the disk.

```bash
#!/bin/bash
# go.sh: bash go.sh <transfer folder> "<seeds>" <parallel slots> [matrix file]
# run with the dngaussian env active
set -e
T=$1; SEEDS=$2; N=${3:-4}; M=${4:-experiments/matrix.yaml}
W=$HOME/dngs; mkdir -p $W/outputs
[ -d $W/data/llff ] || { mkdir -p $W/data && tar -xf $T/data/llff_with_depth.tar -C $W/data; }
rm -rf $W/code && git clone -q $T/code/dngs.bundle $W/code && cd $W/code
git checkout -q v1.0-exp && git log -1 --oneline
python -c "import torch, diff_gaussian_rasterization, simple_knn; print(torch.cuda.get_device_name(0))"
mkdir -p logs
for i in $(seq 0 $((N-1))); do
  ( for pass in 1 2; do
      python experiments/run_matrix.py --matrix $M --data_root $W/data/llff \
          --out_root $W/outputs --shard $i/$N --seeds $SEEDS --shared_gpu
    done ) > logs/shard_$i.log 2>&1 &
done
echo "Launched $N shards. Before leaving: touch $W/code/STOP, wait, then run collect.sh"
```

```bash
#!/bin/bash
# collect.sh: bash collect.sh <transfer folder>   (only after all runs have stopped)
T=$1; W=$HOME/dngs; S=$(date +%m%d_%H%M)
mkdir -p $T/results/$S
rsync -rt --exclude '*.ply' $W/outputs/ $T/results/$S/outputs/
cp $W/code/logs/*.log $T/results/$S/ 2>/dev/null || true
echo "Copied to $T/results/$S"
```

Use `rsync -rt`, not `-a`: exFAT cannot store Linux owners and permissions, and `-a` fails trying to set them. Test both scripts on your laptop on Tuesday with 2 slots and a few runs.

### First visit (Wednesday 14 Oct at the earliest)

1. **Identify the machine:** operating system, `nvidia-smi` (GPU, driver, other users), admin rights, free disk. On Linux, also run `ldd --version`.
2. **Copy the transfer folder from the disk to the local drive.**
3. **Set up the environment.** On Linux:

   ```bash
   mkdir -p ~/dngs/env && tar -xzf <local copy>/env/dngaussian.tar.gz -C ~/dngs/env
   source ~/dngs/env/bin/activate && conda-unpack
   ```

   On Windows, in PowerShell, then work inside that Ubuntu exactly as on your laptop:

   ```powershell
   wsl --import DNGS C:\wsl\DNGS <local copy>\wsl\ubuntu-dngs.tar
   wsl -d DNGS -u <your WSL username>
   ```
4. **If an import fails** with a `GLIBC` error or "no kernel image", recompile with the packed compiler: `TORCH_CUDA_ARCH_LIST=8.0 pip install --no-build-isolation --no-deps --force-reinstall ./submodules/diff-gaussian-rasterization ./submodules/simple-knn`. If that fails too, install fresh over the internet with `setup_env.sh`.
5. **Smoke test:** `fern`, 500 iterations.
6. **Measure throughput:** time one run alone, then launch 4 slots and compare the per-run time. Keep the slot count that finishes the most runs per hour; try 6 only if the GPU is clearly underused.
7. **Launch seed 1** after the freeze tag exists: `bash go.sh <transfer> "1" 4`.
8. **Leave cleanly:** about one run-length before you go, `touch ~/dngs/code/STOP`. Running jobs finish and no new ones start. When `pgrep -f run_matrix` prints nothing, run `bash collect.sh <disk transfer folder>`.
9. **That evening:** copy the dated results folder from the disk into `sparse3dgs/outputs/` on Drive so Disha's scripts pick it up.

Every later visit is steps 2, 7, 8 and 9. The runner skips finished runs, so `go.sh` simply continues where the last visit stopped.

### Visit schedule

Each seed is a 40-run block (5 variants × 8 scenes) and runs entirely on one GPU type, so every variant comparison within a seed is fair.

| Day | At the lab (A100) | Colab meanwhile |
| --- | --- | --- |
| Wed 14 Oct | First visit: identify, set up, smoke test, measure; after the 1 PM freeze, launch seed 1 | Both accounts start seed 0 at 1 PM and run overnight |
| Thu 15 Oct | Finish seed 1, then start seed 2 | Finish seed 0; rerun failures |
| Fri 16 Oct | Finish seed 2 and any reruns | Covers any seed the A100 cannot finish |
| Sat 17 Oct | Finish Tier 1 if anything is left, then Tier 2 (`--matrix experiments/matrix_tier2.yaml`); you write on your laptop meanwhile | Idle or reruns |
| Sun 18 Oct | Finish Tier 2 by evening, collect everything | Idle |

Runs per visit ≈ hours at the lab × 60 ÷ minutes per run × parallel slots. If Wednesday's measurement shows a whole seed won't fit in one lab day, the A100 keeps seed 1 and Colab takes seed 2 as well. The report states which seeds ran on which GPU.

## Runner script and compute budget

`run_matrix.py` turns `matrix.yaml` into a fixed list of runs, gives each machine every N-th run, and skips any run that already has `metrics.json`. Any machine can therefore stop, crash or disconnect and restart without losing or duplicating work.

> **Revised:** the sketches below are superseded by the spec in **PLAN.md W8**. The differences: it reads `results_eval.json` and writes `ssim_sk`; no spiral video; `render.py --skip_train --skip_test`; `metrics.py --model_paths`; flags `--seed --depth_scale 1 --drop_rate 0.2 --depth_render shared|clean`; `--work_root` for local-disk training on Colab; cleanup of checkpoints and TensorBoard; refuses to run off a tag; per-run `run.log`; failures don't stop the shard; `metrics.json` written last.

### `experiments/matrix.yaml`

Copy the three commands (train, render, metrics) and all their flags from `scripts/run_llff.sh`, then replace the scene path, output path and GPU id with the placeholders shown. Only the training command gets the variant flags.

```yaml
scenes: [fern, flower, fortress, horns, leaves, orchids, room, trex]
seeds: [0, 1, 2]
n_views: [3]
drop: &P 0.2          # DropGaussian's gamma (official code: 0.2 * iteration / 10000); see PLAN.md W8 for the final file
variants:
  A: {depth_scale: 0, drop_rate: 0,  depth_render: shared}
  B: {depth_scale: 1, drop_rate: 0,  depth_render: shared}
  C: {depth_scale: 0, drop_rate: *P, depth_render: shared}
  D: {depth_scale: 1, drop_rate: *P, depth_render: shared}
  E: {depth_scale: 1, drop_rate: *P, depth_render: clean}
steps:
  - "python train_llff.py -s {scene_dir} --model_path {out} --n_sparse {n_views} --seed {seed} {variant_flags} <other flags from run_llff.sh>"
  - "python render.py -s {scene_dir} --model_path {out} <flags from run_llff.sh>"
  - "python metrics.py --model_path {out} <flags from run_llff.sh>"
```

### `experiments/run_matrix.py`

```python
#!/usr/bin/env python
"""Resumable runner. Example: python experiments/run_matrix.py --data_root D --out_root O --shard 0/2 --seeds 0"""
import argparse, itertools, json, os, shutil, subprocess, time
import torch, yaml

ap = argparse.ArgumentParser()
ap.add_argument("--matrix", default="experiments/matrix.yaml")
ap.add_argument("--data_root", required=True)      # folder holding the 8 scene folders
ap.add_argument("--out_root", required=True)
ap.add_argument("--shard", default="0/1")          # i/N: this process takes runs i, i+N, i+2N...
ap.add_argument("--seeds", type=int, nargs="*")    # e.g. --seeds 1 2; default: all seeds in the yaml
ap.add_argument("--shared_gpu", action="store_true")
ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.matrix))
seeds = a.seeds if a.seeds else cfg["seeds"]
i, n = map(int, a.shard.split("/"))
runs = sorted(itertools.product(cfg["n_views"], seeds, cfg["variants"], cfg["scenes"]))
mine = runs[i::n]                                  # lower seeds finish first
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
gpu = torch.cuda.get_device_name(0)
print(f"shard {a.shard}, seeds {seeds}: {len(mine)} of {len(runs)} runs on {gpu} at {commit}", flush=True)

for nv, seed, var, scene in mine:
    if os.path.exists("STOP"):                     # touch STOP to drain before leaving the lab
        print("STOP file found: no new runs", flush=True)
        break
    out = f"{a.out_root}/llff_{nv}v/{var}/{scene}/seed{seed}"
    if os.path.exists(f"{out}/metrics.json"):
        continue
    shutil.rmtree(out, ignore_errors=True)         # leftovers of a run that was killed
    os.makedirs(out)
    flags = " ".join(f"--{k} {v}" for k, v in cfg["variants"][var].items())
    fill = dict(scene_dir=f"{a.data_root}/{scene}", out=out, n_views=nv, seed=seed, variant_flags=flags)
    t0 = time.time()
    for step in cfg["steps"]:
        cmd = step.format(**fill)
        print(">>", cmd, flush=True)
        if not a.dry_run and subprocess.run(cmd, shell=True).returncode != 0:
            print(f"FAILED {out}", flush=True)
            break
    else:
        if a.dry_run:
            continue
        res = json.load(open(f"{out}/results_eval.json"))   # revised: metrics.py writes results_eval.json
        m = next(iter(res.values()))                      # 3DGS style: {"ours_<iter>": {"PSNR": ...}}
        extra = {}
        if os.path.exists(f"{out}/train_stats.json"):     # written by train_llff.py (your change)
            extra = json.load(open(f"{out}/train_stats.json"))
        json.dump(dict(variant=var, scene=scene, n_views=nv, seed=seed,
                       psnr=m["PSNR"], ssim=m["SSIM"], lpips=m["LPIPS"],
                       wall_time_s=round(time.time() - t0, 1), git_commit=commit,
                       gpu=gpu, shared_gpu=a.shared_gpu, **extra),
                  open(f"{out}/metrics.json", "w"), indent=2)
```

Have `train_llff.py` write `train_stats.json` at the end of training with `train_time_s`, `n_gaussians` and `vram_peak_mb`. Always try `--dry_run` first to check every command prints correctly.

Machines split work by whole seeds, so every variant comparison within a seed uses one GPU type. Both Colab accounts run `--seeds 0` with `--shard 0/2` and `--shard 1/2`; the A100 runs `--seeds 1` and then `--seeds 2` through `go.sh`. Tier 2 uses a second file, `experiments/matrix_tier2.yaml`, with `n_views: [2, 6, 9]`, seed 0 and variants B, C and the better of D/E.

A file called `STOP` in the repo folder lets running jobs finish but starts no new ones; that is how you leave the lab without losing a run. A run killed mid-way leaves no `metrics.json`, so its folder is cleared and it is redone next time.

### Compute budget

> **Revised, measured:** m = **4.9 min** on a T4 (Disha, Monday) and 7.5 min on the laptop. Tier 1 = 120 × 4.9 ≈ **9.8 GPU-hours**, about 5 h per Colab account: **3 seeds, on Colab alone**. Tier 2 (128 runs) ≈ 16 h on the laptop, over two nights. Variant E is **not** slower (no extra render); variant A is about 25 % faster (no depth passes); diagnostics add about 2–3 %. Full table: PLAN.md §9.

Measure **m**, the minutes for one complete run (train, render, metrics) on a T4, during Monday's smoke test. Total GPU hours = runs × m ÷ 60. Tier 1 is 120 runs and Tier 2 is 72; variant E runs are slower because of the extra render.

| Minutes per run on T4 | Tier 1 total | Per Colab account (2 accounts) | Plan |
| --- | --- | --- | --- |
| 10 | 20 h | 10 h | Fits Wednesday–Thursday on Colab alone |
| 20 | 40 h | 20 h | Needs the A100 or the laptop too, otherwise cut to 2 seeds |
| 30 | 60 h | 30 h | Cut to 2 seeds and get the A100; skip Tier 2 |

These rows are scenarios, not estimates. Replace them with your m**ea**sured m on Monday night and decide the seed count then.

## Communication, tracking and fallbacks

A 15-minute call every evening at a fixed time, GitHub Issues for tasks, the run tracker sheet for compute, and an immediate message for anything blocking. With 8 days, a problem that waits for the next call costs a day.

- **Daily call (15 min):** what finished, what is running on which machine, what is blocked, and tomorrow's first task for each person.
- **Blockers:** message at once. Anything blocked for more than 2 hours gets the other person's help or triggers the fallback below.
- **Reviews:** pull requests are reviewed within 3 hours during the day.
- **`RUNLOG.md`:** one line per event, such as "14 Oct, Colab-Disha, shard 1/2 started at v1.0-exp" or "fortress D seed 1 NaN, rerun".

### Fallback triggers

| Trigger | Check by | Fallback |
| --- | --- | --- |
| Laptop cannot finish a full `fern` run | Mon 12 Oct, night | ✅ Not triggered: 7.5 min, 1663 MB. The laptop is unusable under load, so it runs Tier 2 at night only (PLAN.md W12.3) |
| Colab extensions will not build | Tue 13 Oct, noon | Apply the build-error fixes; screen-share debug together; Kaggle as backup |
| Reproduction mean PSNR more than 1 dB below the paper's 19.12 (✅ laptop: −0.39 dB) | Tue 13 Oct, night | Check eval split, SSIM version and error tolerance; spend at most half a day, then continue with our own variant B as the reference and report the gap |
| Dropout and diagnostics not merged | Wed 14 Oct, 1 PM | Freeze may slip to 8 PM the same day, no later; cut to 2 seeds |
| No lab access on Wednesday, or the go-bag environment fails on site | Wed 14 Oct | *Revised:* no effect. Tier 1 is on Colab, Tier 2 on the laptop at night |
| Colab refuses GPUs or hits usage limits | Any session | *New:* switch account, then Kaggle (same T4). If a seed still can't finish on T4s, the laptop or A100 runs that **whole** seed, and the report says so |
| One variant produces NaN or diverges | During the matrix | Rerun once; if it repeats on other scenes, it is a finding, so log it rather than hide it |
| Tier 1 not finished | Thu 15 Oct 12:00 (fewer than 80 runs) / Fri midnight | Laptop night 2 runs Tier 1 seed 2 as a whole block (Tier 2 keeps 2- and 3-view); last resort: cut to 2 seeds; hard stop Sat evening, then report what completed |
| Tier 2 not finished | Sun 18 Oct, evening | Report the view counts that completed; Tier 2 is optional |
| No measurable difference between variants | Thu 15 Oct | Still a valid result (redundancy); centre the report on the diagnostics |

Some gap from the paper is expected: the DNGaussian README warns that PSNR is unstable in some scenes and that its released checkpoints used a different error tolerance than the paper. Within about 0.5 dB counts as a successful reproduction.

## Writing plan and final checklist

The report is written Saturday 17 to Monday 19 Oct, with Related Work and Method started earlier while runs train; Tuesday 20 is buffer only. Use Overleaf if the course wants LaTeX, otherwise one shared document, and never edit the same section at the same time.

| Section | Writer | Start |
| --- | --- | --- |
| Related work: sparse-view NeRF, sparse-view 3DGS, depth priors incl. DNGaussian++, dropout-style regularization | Disha | Tue 13 Oct, while reproduction runs |
| Method: factorial design, dropout port, decoupled render | You | Wed 14 Oct, after the freeze |
| Experimental setup: data split, protocol, metrics, SSIM version, hardware, seeds | You | Thu 15 Oct |
| Introduction | Disha | Fri 16 Oct |
| Results tables and their text | Disha | Fri 16 Oct |
| Diagnostics analysis and the verdict | You | Fri 16 Oct |
| Discussion, failure cases, limitations | Both | Sat 17 Oct |
| Abstract and conclusion | Both | Sun 18 Oct |

### Figures and tables

1. One training iteration in variants D and E: where the mask applies and which render feeds which loss.
2. Main table: LLFF 3-view, variants A–E as mean ± std over seeds, with published baselines below a rule labelled "reported".
3. Diagnostic curves: `loss_depth_used` against `loss_depth_clean` over iterations for D and E, plus `depth_var_masks`.
4. Qualitative grid: 3 scenes × ground truth, B, C, D, E, showing RGB and depth.
5. Failure cases: 2–3 scenes where combining hurt, with their depth maps.
6. Efficiency table: training time, FPS, Gaussian count, model size and peak VRAM per variant, **T4 runs only** (grouped by GPU).
7. Tier 2 only: PSNR against number of views (2, 3, 6, 9) for **B, C, D and E** (laptop, seed 0).

State these limitations plainly (revised):
- The dropout rate was tuned on vanilla 3DGS, and the schedule was rescaled from 10000 to 6000 iterations.
- Our initialization is random, while DropGaussian's is a COLMAP dense stereo cloud from the sparse views.
- In D, compensation inside the constant-opacity hard-depth pass saturates at the rasterizer's alpha cap of 0.99.
- One dataset; 3 seeds.
- Tier 1 on T4 (PyTorch 2.11), Tier 2 on an RTX 3050 Ti (PyTorch 1.12); Tier 2 has one seed.
- Timings come from T4 GPUs.
- The rasterizer is nondeterministic.

### Published LLFF numbers to start `reported_baselines.csv`

Taken from DropGaussian's CVPR 2025 comparison table ([slides](https://cvpr.thecvf.com/media/cvpr-2025/Slides/32481.pdf)). Disha checks every row against that method's own paper before it goes in the report.

| Method | 3-view PSNR | 3-view SSIM | 3-view LPIPS | 6-view PSNR | 9-view PSNR |
| --- | --- | --- | --- | --- | --- |
| DropGaussian | 20.76 | 0.713 | 0.200 | 24.74 | 26.21 |
| CoR-GS | 20.45 | 0.712 | 0.196 | 24.49 | 26.06 |
| FSGS | 20.43 | 0.682 | 0.248 | 24.09 | 25.31 |
| SparseNeRF | 19.86 | 0.624 | 0.328 | not reported | not reported |
| FreeNeRF | 19.63 | 0.612 | 0.308 | 23.73 | 25.13 |
| 3DGS | 19.22 | 0.649 | 0.229 | 23.80 | 25.44 |
| DNGaussian | 19.12 | 0.591 | 0.294 | 22.18 | 23.17 |

### Before submitting

- [ ] Every number in the report traces to a `metrics.json` or a cited paper table
- [ ] Results reported as mean ± std, with the number of seeds stated
- [ ] SSIM implementation and LPIPS backbone named
- [ ] GPU type and minutes per run stated
- [ ] arXiv searched again on Sun 18 Oct for any new paper combining depth priors with Gaussian dropout
- [ ] DNGaussian++ volume, issue and pages checked on IEEE Xplore
- [ ] DropoutGS authors and pages taken from the CVPR 2025 proceedings
- [ ] DropGaussian cited as Park, Ryu and Kim, CVPR 2025, pp. 21600–21609 ([CVF page](https://openaccess.thecvf.com/content/CVPR2025/html/Park_DropGaussian_Structural_Regularization_for_Sparse-view_Gaussian_Splatting_CVPR_2025_paper.html)), which closes one open item in the handoff file
- [ ] README explains how to regenerate every table from tag `v1.0-exp`
- [ ] Final tag `v1.0-report` pushed and the Drive outputs folder left untouched
- [ ] Handoff file Sections 9 and 10 updated with this plan and the current status

### Sources

- [DNGaussian official repository](https://github.com/Fictionarry/DNGaussian): install steps, LLFF depth and training commands, reproducibility notes
- [DropGaussian CVPR 2025 slides](https://cvpr.thecvf.com/media/cvpr-2025/Slides/32481.pdf): comparison table, SfM initialization, training-only progressive dropping
- [DropGaussian on CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Park_DropGaussian_Structural_Regularization_for_Sparse-view_Gaussian_Splatting_CVPR_2025_paper.html): authors and page numbers
- [DropGaussian official code](https://github.com/DCVL-3D/DropGaussian_release): the dropout implementation to port
