# AGENTS.md

Instructions for any AI coding agent working in this repository (Claude Code, Gemini CLI, Codex, Copilot, Cursor and others). `CLAUDE.md`, `GEMINI.md` and `.github/copilot-instructions.md` import or point to this file. Change the rules here, in one place.

## What this project is

This is a two-person course research project (Aayush and Disha) with a deadline of **Tue 20 Oct 2026**. It asks whether two sparse-view 3D Gaussian Splatting regularizers are **complementary, redundant or interfering**:

- **DNGaussian** (CVPR 2024): monocular-depth regularization. This repo is a fork of [Fictionarry/DNGaussian](https://github.com/Fictionarry/DNGaussian).
- **DropGaussian** (CVPR 2025): randomly drops Gaussians during training. It is ported from [DCVL-3D/DropGaussian_release](https://github.com/DCVL-3D/DropGaussian_release).

Every experimental variant is the **same code with different command-line flags**:

| ID | `--depth_scale` | `--drop_rate` | `--depth_render` | Meaning |
| --- | --- | --- | --- | --- |
| A | 0 | 0 | shared | No depth, no dropout |
| B | 1 | 0 | shared | DNGaussian as published |
| C | 0 | 0.2 | shared | Dropout only |
| D | 1 | 0.2 | shared | Naive combination: the dropout mask also hits the depth passes |
| E | 1 | 0.2 | clean | Decoupled: the depth passes never see the mask |
| D_hard | 1 | 0.2 | hard_only | Only the hard-depth pass sees the mask (PLAN_V2 block P) |
| D_soft | 1 | 0.2 | soft_only | Only the soft-depth pass sees the mask (PLAN_V2 block P) |

The main benchmark is LLFF 3-view (8 scenes × 5 variants × 3 seeds = 120 runs).

**Read these before planning any work:**
- [docs/PLAN.md](docs/PLAN.md): the current plan, status, decisions, specs and schedule. **It overrides the playbook wherever they differ.**
- [docs/research_proposal_sparse_view_3dgs_2.md](docs/research_proposal_sparse_view_3dgs_2.md): the original team playbook (12 Oct). Read it for background and runbooks.

## Environments: code must run on all of them

| Machine | Python | PyTorch | CUDA | GPU | Notes |
| --- | --- | --- | --- | --- | --- |
| Laptop (WSL2 Ubuntu 24.04) | **3.7.13** | **1.12.1** | 11.3, gcc 9 | RTX 3050 Ti, 4 GB, sm_86 | conda env `dngaussian`; see `setup_env.sh` |
| Colab | 3.13 | 2.11 | 13.0 | T4, 16 GB, sm_75 | `notebooks/colab_runner.ipynb`; extension wheels cached on Drive |
| A100 lab machine | ? | ? | ? | A100, sm_80 | Physical access only; optional (see PLAN) |

Consequences for the code:
- **Python 3.7 syntax only** in anything the training pipeline imports: no walrus `:=`, no `f"{x=}"`, no `list[int]` or `dict[str, int]` annotations, no `dict | dict`, no `match`.
- **Never call `torch.load` directly** on checkpoints. Use `utils.general_utils.load_checkpoint()`, which passes `weights_only=False` only when the installed PyTorch accepts it.
- The CUDA extensions (`submodules/diff-gaussian-rasterization`, `submodules/simple-knn`, `gridencoder`, `shencoder`) are **vendored source** and are built per machine. Don't edit their CUDA/C++ unless the task is explicitly a build fix. A source change means every machine has to rebuild.

## Commands (laptop)

```bash
conda activate dngaussian        # or use ~/miniconda3/envs/dngaussian/bin/python directly
cd ~/projects/DNGaussian

# smoke test, about 1 min (1100 iterations, covers both depth passes)
python train_llff.py -s data/llff/fern --model_path output/debug/fern -r 8 --eval --n_sparse 3 --rand_pcd \
  --iterations 1100 --test_iterations 1100 --lambda_dssim 0.2 --densify_grad_threshold 0.0013 \
  --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 --position_lr_init 0.016 \
  --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 \
  --split_opacity_thresh 0.1 --error_tolerance 0.00025 --scaling_lr 0.003 --shape_pena 0.002 \
  --opa_pena 0.001 --near 10 --seed 0

# full upstream run: train (6000 it) + render + spiral video + metrics, about 7.5 min on the laptop
bash scripts/run_llff.sh data/llff/fern output/llff/fern 0

render:  python render.py -s data/llff/fern --model_path <out> -r 8 --near 10 --skip_train --skip_test
metrics: python metrics.py --model_paths <out>        # note: --model_paths (plural)
peak VRAM of one training run: python scripts/peak_vram.py train_llff.py <same flags as above>
8-scene reproduction summary:   python avg_llff.py    # reads output/llff/<scene>/results_eval.json
```

Experiment runner (PLAN W8, PLAN_V2 §3): `python experiments/run_matrix.py --matrix experiments/matrix.yaml --data_root <llff dir> --out_root <outputs> --shard i/N --seeds 0 [--dry_run]`. Always try `--dry_run` first.

## How the code is laid out (what matters for this project)

- `train_llff.py`: the training loop. **Each iteration runs three separate rasterizations, each with its own backward pass and optimizer step:**
  1. **Hard depth:** `render_for_depth`. Opacity is fixed at 0.95; gradients flow to positions only.
  2. **Soft depth:** `render_for_opa`, from iteration 1000 on. Positions, scales and rotations are detached; gradients flow to opacity only.
  3. **Photometric:** `render`. Opacity and colour come from the neural field (`gridencoder`/`shencoder`). It computes L1 + D-SSIM plus the shape, scale and opacity penalties.

  The depth losses never use the photometric render. `--depth_scale 0` skips steps 1 and 2 entirely.
- `gaussian_renderer/__init__.py`: `render`, `render_for_depth`, `render_for_opa` (plus `_sh` variants used by Blender/DTU, which we don't touch).
- `arguments/__init__.py`: `ModelParams`, `OptimizationParams`, `PipelineParams`. A new training option goes in `OptimizationParams`, so it is recorded in `opt_args` automatically.
- `utils/general_utils.py`: `safe_state(silent, seed)` seeds `random`, `numpy` and `torch`. `load_checkpoint()`.
- `scene/dataset_readers.py`: the LLFF split. Eval = every 8th image. Train = `--n_sparse` views evenly spaced over the rest. `--rand_pcd` builds a random initial point cloud from `np.random`, so it depends on the seed.
- `render.py`: loads `chkpnt_latest.pth`. With `inference=True` it renders `eval/ours_<iter>/{renders,gt,depth}`. `--skip_train --skip_test` saves time.
- `metrics.py`: evaluates `eval/` and writes **`results_eval.json`** as `{"ours_<iter>": {"PSNR", "SSIM", "SSIM_sk", "LPIPS"}}`. `SSIM` is the 3DGS implementation, `SSIM_sk` is scikit-image, and LPIPS uses VGG. It is **not** `results.json`.
- `dpt/`: monocular depth generation. **Never rerun it.** The canonical depth is `llff_with_depth.tar` on the team Drive (md5 `0f6bd7bdae837ec58ad757c7c00b0561`).
- `experiments/` (runner, matrix files, aggregation), `notebooks/` (Colab), `results/` (small committed outputs), `docs/` (plan and playbook), `scripts/` (shell helpers, `peak_vram.py`).

## Invariants: never break these

1. **Dropout runs only inside the training loop.** `render.py`, `metrics.py`, `spiral.py`, `training_report`, `clean_views` and every evaluation path must never apply a mask. Render functions take an optional per-Gaussian compensation vector that defaults to `None` (no dropout).
2. **Flag defaults reproduce DNGaussian exactly.** `--depth_scale 1 --drop_rate 0 --depth_render shared` must be the upstream behaviour (variant B). When `drop_rate` is 0, do not even create a mask.
3. **The output layout and file formats are frozen** (agreed 12 Oct): `outputs/llff_<n>v/<variant>/<scene>/seed<k>/` containing `config.json`, `diag.csv`, `metrics.json`, `train_stats.json`, `render_stats.json`, `results_eval.json`, `eval/`. You may **add** keys or columns. Never rename or remove them: `experiments/aggregate.py` and the figure scripts depend on them.
4. **One GPU type per seed.** All five variants of a given seed run on the same GPU type. `metrics.json` records `gpu` and `shared_gpu`. Efficiency numbers come only from runs with `shared_gpu: false`.
5. **Experiments run only from a tag** (`v1.0-exp`, then `v1.1-exp` for fixes). `config.json` records the commit. Don't run the matrix from a branch.
6. **Same data everywhere:** `llff_with_depth.tar`, checked by md5.
7. Port DropGaussian, don't reinvent it. The official code (`gaussian_renderer/__init__.py` in their repo) is: `compensation = torch.nn.Dropout(p=r)(torch.ones(N))`, then `opacity = opacity * compensation[:, None]`, with `r = 0.2 * iteration / 10000`, where 10000 is their total iteration count. We keep it verbatim, `r = drop_rate * iteration / 10000`, even though we train for 6000 iterations, so with `--drop_rate 0.2` the rate peaks at 0.12 (decision D1, 13 Oct). The full spec is in PLAN W2.

## Git and ownership

- `origin` is `github.com/aayxsh23/dngs-dropout` and `upstream` is Fictionarry/DNGaussian. Never push to upstream.
- Work on a branch (`feat/`, `fix/`, `exp/`, `docs/`, `chore/`), open a PR, and go through the **Change workflow** below. Squash-merge only. `main` must always train end to end.
- Ownership, to avoid merge conflicts:
  - **Aayush:** `train_llff.py`, `render.py`, `gaussian_renderer/`, `arguments/`, `utils/`, `scene/`, `experiments/run_matrix.py`, `experiments/*.yaml`, `scripts/`.
  - **Disha:** `notebooks/`, `experiments/aggregate.py`, `experiments/plots/`, `experiments/fixtures/`, `results/`.
  - **Shared, one person at a time:** `README.md`, `docs/`, the report.
- **Never commit** `data/`, `output/`, `outputs/`, checkpoints (`*.pth`, `*.pt`), point clouds (`*.ply`), archives, build products or tokens. `.gitignore` covers these; stage files by name (`git add <file>`) and check `git status` before committing.
- An agent must not push, merge, tag or open PRs unless the human asked for it in this session. When asked to merge, it merges only through the Change workflow.

## Change workflow (mandatory for every merge into `main`)

Claude Code runs this as the **`pr-gate` skill** (`.claude/skills/pr-gate/SKILL.md`). Other agents and humans follow the same steps by hand. No unreviewed code, no unverified test result, no unresolved blocking finding, no merge before the final validation.

**Risk class**, from the changed paths in `.github/CODEOWNERS`:

| Class | Paths | Needs before merge |
| --- | --- | --- |
| **Routine** | everything not listed in CODEOWNERS (docs, notebooks, `aggregate.py`, plots, results) | CI `checks` green + agent review Approve on the head commit. **Docs-only** PRs (every file under `docs/` or `*.md`, none in CODEOWNERS; `review_bundle.py` prints the class) need no reviewer: the pr-gate report and `agent-review` status say so |
| **Experiment** | training, rendering, evaluation, extensions, runner, `experiments/*.yaml` | the above + `GPU-SMOKE PASS <head sha>` comment from `scripts/gpu_smoke.sh` + approval from the **other** teammate. After `v1.0-exp`: also a new tag and a RUNLOG entry |
| **Gate** | `.github/`, `.claude/`, `scripts/check.sh`, `scripts/gpu_smoke.sh`, `scripts/pr_ready.py`, `scripts/hooks/`, `scripts/setup_repo_rules.sh`, AGENTS/CLAUDE/GEMINI.md | the routine requirements + approval from the **other** teammate |

**Agents never submit APPROVE reviews** (`gh pr review --approve` is blocked by the hook). Approvals are a human's decision; an agent posts its review report as a comment and sets the `agent-review` status.

**Always name the repo:** this clone has an `upstream` remote (Fictionarry/DNGaussian), and plain `gh` commands can resolve to it. Pass `--repo aayxsh23/dngs-dropout` to every `gh pr` command and use `repos/aayxsh23/dngs-dropout/...` in `gh api` paths. `pr_ready.py` refuses any other repo.

**Pipeline:**
1. **Understand.** Read the goal and acceptance criteria and the code involved; classify the PR.
2. **Implement and test.** Make the smallest change that meets the goal. Add a deterministic check to `scripts/check.sh` for any new invariant. Run `bash scripts/check.sh`. For the experiment class, also run `bash scripts/gpu_smoke.sh` on the clean head and post its last line.
3. **Independent review** by a separate reviewer (the `pr-reviewer` agent in `.claude/agents/`), fed one bundle file from `scripts/review_bundle.py`. Depth and model follow the risk class: docs-only PRs need no reviewer, other routine PRs (code, notebooks, results) get sonnet, experiment and gate get opus. The reviewer works to a tool budget and doesn't re-run what CI proved. The report gets severity-ranked findings and a recommendation: Approve, Request Changes or Blocked. Post it on the PR.
4. **Remediate:** for each blocking finding, add a test that fails on the old code, then fix it. Lows go to a follow-up issue and don't trigger a re-review. Re-run step 2, then do a **delta** re-review (`review_bundle.py --since <reviewed sha>`), which checks only the fixes and new regressions. At most 3 review rounds in total, then escalate to a human.
5. **Record and merge.** Only after an Approve:
   - set the `agent-review` commit status on the reviewed head SHA,
   - wait for CI (`gh pr checks <N> --repo aayxsh23/dngs-dropout --watch`),
   - run `python3 scripts/pr_ready.py <N>`,
   - and only if it prints ELIGIBLE, `gh pr merge <N> --repo aayxsh23/dngs-dropout --squash --delete-branch --match-head-commit <reviewed SHA>`. The SHA pin makes GitHub refuse the merge if anything was pushed after the review; the hook requires it.

   Use `--auto` only while the server-side ruleset is active; `pr_ready.py` checks that. Report "merged" only when `gh pr view` shows MERGED. Never add a "Generated with Claude Code" footer to PR text.

**Blocked, so stop and report, never work around:**
- A check fails, or a result is missing or inconclusive.
- A Critical or High finding is open.
- A secret is exposed.
- There are merge conflicts.
- A required approval is missing.
- A merge would need `--admin`, a push to `main`, or a change to the hook or CI.
- A destructive or irreversible step needs a human's authorization.

**Enforcement.** Agent instructions guide; they don't enforce. What actually enforces the gate:
- **CI job `checks`** (`.github/workflows/ci.yml` runs `scripts/check.sh`). It reports on every PR, but nothing on GitHub stops a merge while it is red unless a ruleset exists.
- **The Claude Code Bash hook** (`.claude/settings.json` → `scripts/hooks/guard_bash.py`). It blocks:
  - pushes to `main` (explicit refspecs, `HEAD` or a bare push while on main, `--all`),
  - `gh pr merge --admin`,
  - `gh pr review --approve`,
  - `gh api .../merge`,
  - any `gh pr merge` that `scripts/pr_ready.py` rejects. `pr_ready.py` is taken from `origin/main` once it exists there.

  The hook script and `.claude/settings.json` themselves are read from the checked-out branch. So branches cut before the gate landed (e.g. an old `feat/dropout`) have no hook: rebase them onto `main`. The hook guards against honest mistakes; it is not a sandbox (`bash -c`, `eval` or a script file can hide commands). It binds only Claude Code sessions in this repo.
- **The GitHub ruleset on `main`** (`scripts/setup_repo_rules.sh --apply`): required checks `checks` + `agent-review`, code-owner review, no direct push or force-push. It enforces the gate server-side for everyone, humans included. Rulesets need a public repo (the repo is public again since 12 Oct night) or GitHub Pro, and an admin applies them once, after the gate PR is merged. The script stops without changing anything when rulesets are unavailable. Until it has run, nothing server-side stops a human (or the owner's token outside Claude Code) from merging a red PR: follow the workflow by hand and never merge a PR whose `checks` or `agent-review` isn't green.

## Practical gotchas

- **The rasterizer is nondeterministic.** The same seed gave 17.75 and 17.29 PSNR on fern at 1100 iterations. Compare variants with means over seeds and scenes, never single runs.
- The laptop GPU has 4 GB, and a training run's footprint is about 1.7 GB. Run **one** training job at a time, and check `pgrep -af "train_llff|render.py|metrics.py"` before you start one. nvidia-smi can't show per-process memory under WSL, so use `scripts/peak_vram.py`.
- Long runs on the laptop: start them in `tmux` or with `nohup`. A job started in a VS Code terminal dies when VS Code restarts.
- `--rand_pcd` writes its random cloud to a unique temp dir (PLAN W7), never into the dataset folder, so parallel runs on one scene don't race. `scripts/check.sh` fails if the old `sparse/0/points3D_random` path comes back.
- The playbook's `run_matrix.py` sketch reads `results.json`. The real file is `results_eval.json`.
- `run_llff.sh` also renders a spiral video with `spiral.py`. The matrix skips it.
- Training writes large files: `chkpnt_latest.pth` is about 164 MB and TensorBoard events are about 13 MB per run. Drive quota is 15 GB per account, so the runner deletes them (PLAN W8).
- Keep the repo on the Linux filesystem (`~/projects`), never `/mnt/c`.

## How to work here

- Read `docs/PLAN.md` first. The status table at its top is the single place where progress is recorded, so update it when you finish something.
- Write code that matches the surrounding style: argparse groups in `arguments/`, the same comment density, minimal diffs. Don't refactor upstream code you don't need to touch; it keeps our diff against DNGaussian reviewable.
- Test changes with `bash scripts/check.sh` (seconds, no GPU) and, for training code, `bash scripts/gpu_smoke.sh` (1100 iterations on `fern` per variant). Then check that the defaults (variant B) still behave like upstream.
- When you change behaviour, say in the PR which variants it affects and how you tested it.
