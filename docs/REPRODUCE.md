# Reproducing the paper, end to end

The pipeline for each dataset is: data → poses (DTU only: COLMAP) → monocular depth (DPT) → train, render and evaluate
(the runner, one matrix file per block) → aggregate → statistics → figures. Every command below is the one used for the
paper; the tag in each step is the code version that produced the committed results. Datasets are **not** in this
repository; download them from the official sources below.

## 0. Environment

| Machine | Use | Setup |
| --- | --- | --- |
| Lab PC, RTX A4000 16 GB, PyTorch 1.12.1 | every experiment run | same as the laptop |
| Laptop, RTX 3050 Ti 4 GB (WSL2), Python 3.7.13, PyTorch 1.12.1, CUDA 11.3 | smoke tests, data preparation | [`setup_env.sh`](../setup_env.sh); full package list in [`pip_freeze_laptop.txt`](../pip_freeze_laptop.txt) and [`environment.lock.yml`](../environment.lock.yml) |
| Colab T4 | reproduction check; CPU analysis | [`notebooks/colab_runner.ipynb`](../notebooks/colab_runner.ipynb), [`notebooks/results_v2_colab.ipynb`](../notebooks/results_v2_colab.ipynb) |

Build the vendored CUDA extensions (`submodules/diff-gaussian-rasterization`, `submodules/simple-knn`, `gridencoder`,
`shencoder`) for your GPU, as `setup_env.sh` does. Analysis needs only CPU Python ≥ 3.8 with pandas, numpy, scipy,
matplotlib, pyyaml and statsmodels; `scripts/check.sh` runs it through `uv` if they are missing.

Check the install without a GPU: `bash scripts/check.sh` (seconds). With a GPU: `bash scripts/gpu_smoke.sh` (1.5–7 min).

## 1. Data, poses and depth

The frozen archives (with DPT depth) that every run used are on the team Drive; their md5s are the reference:

| Archive | md5 | Size |
| --- | --- | --- |
| `llff_with_depth.tar` | `0f6bd7bdae837ec58ad757c7c00b0561` | 2.4 GB |
| `dtu_with_depth.tar` | `dc8df470e9cf6e46cd024cb34d19c175` | 2,465,986,560 bytes |
| `blender_with_depth.tar` | `e7597085f8682b660699a39f63d5957f` | 5,168,435,200 bytes |

To rebuild them from the official sources (DNGaussian's own preparation, see [UPSTREAM_README](UPSTREAM_README.md)):

**LLFF.** Download `nerf_llff_data` (the NeRF authors' release, linked in the upstream README) to `data/llff/`. Depth:
```bash
cd dpt && python get_depth_map_for_llff_dtu.py --root_path ../data/llff --benchmark LLFF && cd ..
```

**DTU.** Download "Rectified" from the [DTU MVS site](https://roboimagedata.compute.dtu.dk/?page_id=36/) and the
evaluation masks from RegNeRF (`submission_data.zip`, md5 `7d92576a23549a599fc20d9eab9e2fb7`, linked in the upstream README).
We fetched only the 735 images needed (15 scans × 49 views, light 3) by HTTP range requests (RUNLOG, 16 Nov).
```bash
bash scripts/organize_dtu_dataset.sh <Rectified path>          # 15 scans into data/dtu/scan*/input
for s in data/dtu/scan*; do python convert.py -s $s --no_gpu; done   # COLMAP 4.2.1 on CPU
python convert.py -s data/dtu/scan30 --no_gpu --camera PINHOLE  # the one pre-registered retry; also scan63, scan103, scan110
cd dpt && python get_depth_map_for_llff_dtu.py --root_path ../data/dtu --benchmark DTU && cd ..
```
Rule (prereg-v3): a scan counts only if all 49 images register. All 15 did (`data/dtu/registration.txt`, RUNLOG 17 Nov).

**Blender.** Download `nerf_synthetic` (linked in the upstream README) to `data/nerf_synthetic/`. Depth:
```bash
cd dpt && python get_depth_map_for_blender.py --root_path ../data/nerf_synthetic && cd ..
```

## 2. Train, render and evaluate (GPU)

The runner refuses to run from anything but a clean tag. Each run writes
`<out_root>/<dataset>_<n>v/<variant>/<scene>/seed<k>/` with `config.json`, `metrics.json`, `results_eval.json`,
`diag.csv`, `train_stats.json` and `eval/`. Always dry-run first.

| Block | Tag | Matrix | Seeds | Runs |
| --- | --- | --- | --- | --- |
| T1 | `v1.0-exp` | `experiments/matrix.yaml` | 0 1 2 | 120 |
| P | `v1.0-exp` | `experiments/matrix_pass.yaml` | 0 1 2 | 48 |
| G | `v1.0-exp` | `experiments/matrix_dose.yaml` | 0 | 48 |
| N | `v1.0-exp` | `experiments/matrix_noise.yaml` | 0 | 8 |
| V | `v1.0-exp` | `experiments/matrix_views.yaml` | 0, then 1 2 | 120 + 240 |
| COMP | `v1.1-exp` | `experiments/matrix_comp.yaml` | 0 1 2 | 96 |
| DDOSE | `v1.1-exp` | `experiments/matrix_depthdose.yaml` | 0 1 2 | 96 |
| NOISE2 | `v1.1-exp` | `experiments/matrix_noise2.yaml` | 0 | 48 |
| DTU | `v1.2-exp` | `experiments/matrix_dtu.yaml` | 0 1 2 | 225 |
| Blender | `v1.2-exp` | `experiments/matrix_blender.yaml` | 0 1 2 | 120 |

```bash
git checkout v1.0-exp
python experiments/run_matrix.py --matrix experiments/matrix.yaml --data_root data/llff --out_root outputs \
    --shard 0/1 --seeds 0 1 2 --dry_run         # prints "120 of 120 runs"; drop --dry_run to train
git checkout v1.2-exp
python experiments/run_matrix.py --matrix experiments/matrix_dtu.yaml --data_root data/dtu --out_root outputs_dtu \
    --shard 0/3 --seeds 0 1 2                   # 3 shards = 3 processes on one 16 GB GPU
```
Per run the runner trains for 6000 iterations, renders the held-out views without any mask, and computes metrics
(`metrics.py`; DTU: `experiments/eval_dtu.py` with the object masks). Use `--shared_gpu` when other jobs share the GPU,
so efficiency numbers are not taken from those runs.

**Runtime** (training time per run on a quiet A4000, `results/tables/efficiency_3v.csv`): A and C about 84 s, B, D and E
141–145 s, plus rendering and metrics. The full LLFF run (train, render, video, metrics) takes about 7.5 min on the 4 GB
laptop (RUNLOG, 12 Oct). Shared three ways with another process, a run took about 464 s (RUNLOG, 17 Nov).

A single run by hand, variant E on fern with seed 0 (DNGaussian's LLFF flags as in `scripts/run_llff.sh`):
```bash
python train_llff.py -s data/llff/fern --model_path output/E/fern -r 8 --eval --n_sparse 3 --rand_pcd --iterations 6000 \
  --lambda_dssim 0.2 --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
  --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 \
  --split_opacity_thresh 0.1 --error_tolerance 0.00025 --scaling_lr 0.003 --shape_pena 0.002 --opa_pena 0.001 --near 10 \
  --depth_scale 1 --drop_rate 0.2 --depth_render clean --seed 0
python render.py -s data/llff/fern --model_path output/E/fern -r 8 --near 10 --skip_train --skip_test
python metrics.py --model_paths output/E/fern
```

## 3. Tables, statistics and figures (CPU only)

The per-run metrics of every block are committed (`results/raw/` for v1, `results/v2/tables/*/runs.csv` for v2), so this
step runs without data or a GPU.

- **v1 (LLFF):** the commands in [`results/RESULTS.md`](../results/RESULTS.md#how-to-regenerate) (`aggregate.py` on
  `results/raw/*`, `diag_curves.py`, `stats.py`, plots).
- **v2 (DTU, Blender, compensation, depth dose, noise, power):** the commands in
  [`results/v2/RESULTS_v2.md`](../results/v2/RESULTS_v2.md#how-to-regenerate) (`stats.py` per dataset and pooled,
  `extra_blocks.py`, `v2_figures.py`).
- **Views, 3 seeds:** `python experiments/views_contrasts.py` writes `results/v2/tables/views/contrasts_3seed.csv`.
- **Error budget** (needs the eval renders from the lab tarballs on the team Drive, not in the repo):
  `python experiments/bias_variance.py --out_root <block root> --dataset llff --n_views 3 --variants A B C D E --seeds 0 1 2 --tables <dir>`
  (DTU: add `--mask_root data/dtu/submission_data/idrmasks`); then `experiments/interaction_scales.py` for N4.
- **Lab-only steps** (they need checkpoints): `experiments/probes.py` (test-time robustness, Monte-Carlo depth bias;
  the committed copy is the exact version that ran, md5 `1a2913d0b2fbf219c528cbd87d04b8e7`) and `experiments/geometry.py`
  (MVC@5%). Their outputs are committed under `results/tables/`.

**Regeneration check (21 Nov, this branch).** All v1 tables regenerated from `results/raw/`: 44 files byte-identical.
The `runs.csv` and `factors.csv` files differ only by columns and rows that later code added (`dataset`, `drop_mode`, new
variant IDs); every shared value is identical. All v2 statistics regenerated from `runs.csv`: 23 files byte-identical
(every verdict `.txt` and `.md`), and 8 CSVs differ only by floating-point noise (relative difference ≤ 2 × 10⁻¹¹).
`contrasts_3seed.csv` regenerates byte-identical.

## 4. What is not in the repository, and where it is

| Item | Where |
| --- | --- |
| Datasets and depth archives | official sources above; frozen archives on the team Drive (`sparse3dgs/data/`) |
| Checkpoints (`chkpnt_latest.pth`, about 164 MB each), point clouds, TensorBoard events | deleted by the runner to save space; seed-0 checkpoints of T1 on the lab disk |
| Eval renders of every run | lab tarballs on the team Drive (`sparse3dgs/outputs/`, md5s in RUNLOG 19 Nov) |
| Qualitative grids' source images | `qual_view0.tgz` (about 43 MB), lab copy, 11 Nov |
