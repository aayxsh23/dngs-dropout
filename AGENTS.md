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

Experiment runner (being built, see PLAN W8): `python experiments/run_matrix.py --matrix experiments/matrix.yaml --data_root <llff dir> --out_root <outputs> --shard i/N --seeds 0 [--dry_run]`. Always try `--dry_run` first.

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
7. Port DropGaussian, don't reinvent it. The official code (`gaussian_renderer/__init__.py` in their repo) is: `compensation = torch.nn.Dropout(p=r)(torch.ones(N))`, then `opacity = opacity * compensation[:, None]`, with `r = 0.2 * iteration / 10000`, where 10000 is their total iteration count. We use `0.2 * iteration / opt.iterations`. The full spec is in PLAN W2.

## Git and ownership

- `origin` is `github.com/aayxsh23/dngs-dropout` and `upstream` is Fictionarry/DNGaussian. Never push to upstream.
- Work on a branch (`feat/`, `fix/`, `exp/`, `docs/`, `chore/`), open a PR, get a review from the other person, **squash and merge**. `main` must always train end to end.
- Ownership, to avoid merge conflicts:
  - **Aayush:** `train_llff.py`, `render.py`, `gaussian_renderer/`, `arguments/`, `utils/`, `scene/`, `experiments/run_matrix.py`, `experiments/*.yaml`, `scripts/`.
  - **Disha:** `notebooks/`, `experiments/aggregate.py`, `experiments/plots/`, `experiments/fixtures/`, `results/`.
  - **Shared, one person at a time:** `README.md`, `docs/`, the report.
- **Never commit** `data/`, `output/`, `outputs/`, checkpoints (`*.pth`, `*.pt`), point clouds (`*.ply`), archives, build products or tokens. `.gitignore` covers these; stage files by name (`git add <file>`) and check `git status` before committing.
- An agent must not push, merge, tag or open PRs unless the human asked for it in this session.

## Practical gotchas

- **The rasterizer is nondeterministic.** The same seed gave 17.75 and 17.29 PSNR on fern at 1100 iterations. Compare variants with means over seeds and scenes, never single runs.
- The laptop GPU has 4 GB, and a training run's footprint is about 1.7 GB. Run **one** training job at a time, and check `pgrep -af "train_llff|render.py|metrics.py"` before you start one. nvidia-smi can't show per-process memory under WSL, so use `scripts/peak_vram.py`.
- Long runs on the laptop: start them in `tmux` or with `nohup`. A job started in a VS Code terminal dies when VS Code restarts.
- `--rand_pcd` writes `sparse/0/points3D_random.ply` into the **dataset folder**. Parallel runs on the same scene race on that file (PLAN W7 fixes this).
- The playbook's `run_matrix.py` sketch reads `results.json`. The real file is `results_eval.json`.
- `run_llff.sh` also renders a spiral video with `spiral.py`. The matrix skips it.
- Training writes large files: `chkpnt_latest.pth` is about 164 MB and TensorBoard events are about 13 MB per run. Drive quota is 15 GB per account, so the runner deletes them (PLAN W8).
- Keep the repo on the Linux filesystem (`~/projects`), never `/mnt/c`.

## How to work here

- Read `docs/PLAN.md` first. The status table at its top is the single place where progress is recorded, so update it when you finish something.
- Write code that matches the surrounding style: argparse groups in `arguments/`, the same comment density, minimal diffs. Don't refactor upstream code you don't need to touch; it keeps our diff against DNGaussian reviewable.
- Test changes with the 1100-iteration smoke test on `fern`. Then check that the defaults (variant B) still behave like upstream.
- When you change behaviour, say in the PR which variants it affects and how you tested it.
