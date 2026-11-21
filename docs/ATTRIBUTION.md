# Attribution: what is upstream, what is ported, what is ours

## Upstream code (not ours)

- **DNGaussian** (Li et al., CVPR 2024), [Fictionarry/DNGaussian](https://github.com/Fictionarry/DNGaussian): this repository is a fork. Its training scripts (`train_llff.py`, `train_dtu.py`, `train_blender.py`), renderer, scene loading, neural colour field (`gridencoder/`, `shencoder/`, `encoding.py`), DPT depth scripts (`dpt/`), metrics (`metrics.py`, `metrics_dtu.py`, `lpipsPyTorch/`) and run scripts are upstream. The fork point is tag `upstream-baseline` (`81c3d83`). Its original README is kept in [UPSTREAM_README.md](UPSTREAM_README.md).
- **3D Gaussian Splatting** (Kerbl et al., 2023), on which DNGaussian builds, including the vendored rasterizer and `simple-knn` in `submodules/`. Licence: [`LICENSE.md`](../LICENSE.md) (Gaussian-Splatting License, Inria and MPII; non-commercial research use).

## Ported code

- **DropGaussian** (Park, Ryu and Kim, CVPR 2025), [DCVL-3D/DropGaussian_release](https://github.com/DCVL-3D/DropGaussian_release). Only its two dropout lines are ported, verbatim: `compensation = torch.nn.Dropout(p=r)(torch.ones(N))` and `opacity = opacity * compensation[:, None]`, with its schedule `r = 0.2 * iteration / 10000`. In this repository they appear as the optional `drop_comp` argument in `gaussian_renderer/__init__.py` (`render`, `render_for_depth`, `render_for_opa`) and its construction in the three trainers. `scripts/check.sh` checks the verbatim line and the schedule.

## Our changes, per file

From `git diff --name-status upstream-baseline HEAD`, excluding `results/` (our run outputs and tables, all produced by the scripts below), `docs/` (our plans and report drafts) and `report/` (the paper). A = added by us, M = upstream file modified by us. **Author** comes from the commits touching the file: PRs were squash-merged, so the author is the person who opened the PR. Commits made with an AI coding assistant on Aayush's machine are counted as Aayush's.

| | File | Author (commits) |
| --- | --- | --- |
| A | `.claude/agents/pr-reviewer.md` | Aayush ×3 |
| A | `.claude/settings.json` | Aayush ×1 |
| A | `.claude/skills/pr-gate/SKILL.md` | Aayush ×4 |
| A | `.github/CODEOWNERS` | Aayush ×4 |
| A | `.github/copilot-instructions.md` | Aayush ×2 |
| A | `.github/pull_request_template.md` | Aayush ×1 |
| A | `.github/workflows/ci.yml` | Aayush ×2 |
| M | `.gitignore` | Aayush ×1 |
| A | `AGENTS.md` | Aayush ×9 |
| A | `CLAUDE.md` | Aayush ×3 |
| A | `GEMINI.md` | Aayush ×4 |
| A | `LAB_RUNBOOK_11nov.md` | Aayush ×1 |
| M | `README.md` | Disha ×2 |
| A | `RUNLOG.md` | Aayush ×4; Disha ×1 |
| M | `arguments/__init__.py` | Aayush ×2 |
| A | `experiments/aggregate.py` | Disha ×2 |
| A | `experiments/bias_variance.py` | Disha ×2 |
| A | `experiments/eval_dtu.py` | Aayush ×1 |
| A | `experiments/extra_blocks.py` | Disha ×1 |
| A | `experiments/factors.py` | Disha ×2 |
| A | `experiments/fixtures/make_fixtures.py` | Disha ×5 |
| A | `experiments/geometry.py` | Aayush ×1 |
| A | `experiments/interaction_scales.py` | Disha ×1 |
| A | `experiments/matrix.yaml` | Aayush ×1 |
| A | `experiments/matrix_blender.yaml` | Aayush ×1 |
| A | `experiments/matrix_comp.yaml` | Aayush ×1 |
| A | `experiments/matrix_depthdose.yaml` | Aayush ×1 |
| A | `experiments/matrix_dose.yaml` | Aayush ×1 |
| A | `experiments/matrix_dtu.yaml` | Aayush ×1 |
| A | `experiments/matrix_noise.yaml` | Aayush ×1 |
| A | `experiments/matrix_noise2.yaml` | Aayush ×1 |
| A | `experiments/matrix_pass.yaml` | Aayush ×1 |
| A | `experiments/matrix_smoke.yaml` | Aayush ×1 |
| A | `experiments/matrix_smoke_blender.yaml` | Aayush ×1 |
| A | `experiments/matrix_smoke_dtu.yaml` | Aayush ×1 |
| A | `experiments/matrix_views.yaml` | Aayush ×1 |
| A | `experiments/matrix_views_a100.yaml` | Aayush ×1 |
| A | `experiments/plots/diag_curves.py` | Disha ×2 |
| A | `experiments/plots/dose.py` | Disha ×1 |
| A | `experiments/plots/forest.py` | Disha ×1 |
| A | `experiments/plots/pass_loc.py` | Disha ×1 |
| A | `experiments/plots/qual_grid.py` | Disha ×1 |
| A | `experiments/plots/style.py` | Disha ×2 |
| A | `experiments/plots/v2_figures.py` | Disha ×1 |
| A | `experiments/plots/views.py` | Disha ×2 |
| A | `experiments/probes.py` | Aayush (commit 9f8d55f on branch feat/probes, AY-22; restored unchanged here via a81f943) ×1 |
| A | `experiments/probes_summary.py` | Disha ×1 |
| A | `experiments/run_matrix.py` | Aayush ×3 |
| A | `experiments/stats.py` | Disha ×3 |
| A | `experiments/views_contrasts.py` | Aayush ×1 |
| M | `gaussian_renderer/__init__.py` | Aayush ×1 |
| M | `gridencoder/backend.py` | Aayush ×1 |
| M | `gridencoder/setup.py` | Aayush ×1 |
| A | `notebooks/colab_runner.ipynb` | Disha ×3 |
| A | `notebooks/results_v2_colab.ipynb` | Disha ×1 |
| A | `pip_freeze_laptop.txt` | Aayush ×1 |
| M | `render.py` | Aayush ×2 |
| M | `scene/dataset_readers.py` | Aayush ×2 |
| A | `scripts/check.sh` | Aayush ×8; Disha ×1 |
| A | `scripts/gpu_smoke.sh` | Aayush ×3 |
| A | `scripts/hooks/guard_bash.py` | Aayush ×4 |
| A | `scripts/make_submission_zip.sh` | Aayush ×1 |
| A | `scripts/peak_vram.py` | Aayush ×1 |
| A | `scripts/pr_ready.py` | Aayush ×4 |
| A | `scripts/review_bundle.py` | Aayush ×2 |
| A | `scripts/setup_repo_rules.sh` | Aayush ×2 |
| A | `setup_env.sh` | Aayush ×1 |
| M | `shencoder/backend.py` | Aayush ×1 |
| M | `shencoder/setup.py` | Aayush ×1 |
| A | `site/assets/fig_budget.png` | Disha ×1 |
| A | `site/assets/fig_budget_views.png` | Disha ×1 |
| A | `site/assets/fig_ceiling.png` | Disha ×1 |
| A | `site/assets/fig_comp.png` | Disha ×1 |
| A | `site/assets/fig_depthdose.png` | Disha ×1 |
| A | `site/assets/fig_pooled.png` | Disha ×1 |
| A | `site/assets/fig_power.png` | Disha ×1 |
| A | `site/assets/fig_qual.png` | Disha ×1 |
| A | `site/assets/fig_robustness.png` | Disha ×1 |
| A | `site/assets/fig_views.png` | Disha ×1 |
| A | `site/index.html` | Disha ×1 |
| M | `spiral.py` | Aayush ×1 |
| M | `submodules/simple-knn/simple_knn.cu` | Aayush ×1 |
| M | `train_blender.py` | Aayush ×2 |
| M | `train_dtu.py` | Aayush ×2 |
| M | `train_llff.py` | Aayush ×3 |
| A | `utils/dropout.py` | Aayush ×1 |
| M | `utils/general_utils.py` | Aayush ×2 |

## Team contributions (from PR authorship)

| | Aayush Trivedi (`aayxsh23`) | Disha Singh (`DishaSingghh`) |
| --- | --- | --- |
| Infrastructure | fork, environment and build fixes (#2, #3), review/merge gate, CI, hooks (#4–#7, #12) | Colab runner and notebooks (#1, #8, #11) |
| Method code | dropout port, D/E/D_hard/D_soft routing, diagnostics, runner and matrices (#19); clamp-aware and `nc` modes (#27); DTU and Blender port, `eval_dtu.py` (#36) | — |
| Analysis code | `geometry.py`, `probes.py` | `aggregate.py`, `stats.py` v1/v2, plots, fixtures (#9, #20, #22); error budget, pooled stats, N4 (#33); verified baselines (#10) |
| Data | laptop reproduction; DTU download, COLMAP and DPT (RUNLOG 16–18 Nov) | canonical LLFF depth archive; upstream reproduction on T4; Blender depth archive (RUNLOG 12–13 Oct, 18 Nov) |
| Experiments | lab runs and packing (RUNLOG) | lab runs, views seeds 1–2 (RUNLOG 17–18 Nov) |
| Results and writing | research analysis and PLAN_V2 (#16); RUNLOG and docs (#38); report §3–4 (#23); paper (#44) | results v1 and v2 (#29–#31, #37); prereg-v3 (#32); report §5–6, README, website (#39); wording audit fixes (#40) |
