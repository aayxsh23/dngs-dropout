# Raw lab results (tag v1.0-exp)

Every run below was trained on the lab PC (NVIDIA RTX A4000, `shared_gpu: false`) from tag `v1.0-exp`
(commit `1040559`), as `metrics.json` records for each run. All runs finished by Fri 6 Nov 23:59 with 0 failures.

| Folder | Block (PLAN_V2 §3) | Lab output root | Matrix file | Runs |
| --- | --- | --- | --- | --- |
| `main/` | T1, A–E, 8 scenes × 3 seeds, 3 views | `outputs/` | `matrix.yaml` | 120 |
| `pass/` | P, D_hard and D_soft, 8 scenes × 3 seeds | `outputs_pass/` | `matrix_pass.yaml` | 48 |
| `dose/` | G, C/D/E at γ = 0.1 and 1/3, 8 scenes, seed 0 | `outputs_dose/` | `matrix_dose.yaml` | 48 |
| `noise/` | N, B_r1–4 and D_r1–4 on fern, seed 0 | `outputs_noise/` | `matrix_noise.yaml` | 8 |
| `views/` | V, A–E at 2, 6 and 9 views, 8 scenes, seed 0 | `outputs_views/` | `matrix_views.yaml` | 120 |
| `smoke/` | GPU smoke test of 1040559 (B and D, 1100 iterations) | `output/smoke/` | – | 2 |
| `logs/` | the runner's console log for each block, and `probes.log` (11 Nov) | `output/lab_matrix*.log`, `output/probes.log` | – | – |

The folders were renamed from the lab's `outputs*/` because `.gitignore` excludes `output*`. Inside each folder the
layout is the frozen one (`llff_<n>v/<variant>/<scene>/seed<k>/`), so `experiments/aggregate.py --out_root
results/raw/<block>` reads it directly.

What is kept per run: `config.json`, `diag.csv`, `metrics.json`, `train_stats.json`, `render_stats.json`,
`results_eval.json`, `per_view_eval.json`, `cameras.json` (and `eval/ours_6000/cameras.json`), and `run.log`,
gzipped to `run.log.gz` (it is about 10× smaller). Renders, depth maps and checkpoints are not here; they are on the
lab disk and Drive. One stray launcher log from the lab (`lab_B%%:*.log`, a failed shell-script start before the
real runs, with a file name that breaks on Windows) was left out.

**Added 11 Nov:** `main/llff_3v/<variant>/<scene>/seed0/probes/` (`robustness.csv`, `depth_bias.json`) for the 40 T1
seed-0 runs, from `experiments/probes.py` at commit `9f8d55f` (md5 `1a2913d0b2fbf219c528cbd87d04b8e7`), run on the lab
A4000 against the kept checkpoints; and `logs/probes.log`. The full backup without checkpoints is
`results_full_nockpt.tgz` on Drive `sparse3dgs` (2,841,905,876 bytes, md5 `2ea56c510519ff48a8a66fc3bae548ae`).
