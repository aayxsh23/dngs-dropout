# CSD361 Part 4 (final code submission): checklist

Zip: `CSD361_2410110012_2410110128.zip`, built by `bash scripts/make_submission_zip.sh <ref> <report PDF>` from the
committed tree only. Its SHA-256 is recorded in the PR that ships it.

| Requirement | File or evidence | Status |
| --- | --- | --- |
| Complete, executable code | Training: `train_llff.py`, `train_dtu.py`, `train_blender.py`. Rendering and metrics: `render.py`, `metrics.py`, `experiments/eval_dtu.py`. Runner and analysis: `experiments/`. `bash scripts/check.sh` passes from a clean clone (CPU, seconds); every matrix dry-runs to its planned count (120, 48, 48, 8, 120, 96, 96, 48, 225, 120); `bias_variance.py --self_test` passes | done |
| Commented code | Every file we added has a purpose header (module docstring or header comment). Upstream files keep their own comments; our edits in them are marked inline (e.g. `# DropGaussian; …` in `gaussian_renderer/__init__.py`). The vendored encoder build files (`gridencoder/`, `shencoder/`) are left as they are | done |
| Data loading, preprocessing, training, prediction, evaluation | [`docs/REPRODUCE.md`](REPRODUCE.md): one path per dataset (download → COLMAP for DTU → DPT depth → runner with the matrix file → render → metrics / `eval_dtu.py` → aggregate → stats → figures), with exact commands | done |
| README | `README.md` (landing page: overview, findings, quickstart, reproduction, repo map, citation, team) | see the README PR |
| Scripts, configs, resources | `experiments/matrix*.yaml`, `scripts/run_*.sh`, `setup_env.sh`, `environment.yml`, `environment.lock.yml`, `pip_freeze_laptop.txt`; data md5s in `docs/REPRODUCE.md` and `RUNLOG.md` | done |
| Output files and logs | `results/raw/` (v1 per-run outputs, logs), `results/tables/`, `results/v2/tables/`, `results/figures/`, `results/v2/figures/`, `RUNLOG.md` | done |
| What is not committed, and where it is | `docs/REPRODUCE.md` §4: datasets (official sources), checkpoints (lab disk), eval renders (team Drive tarballs with md5s) | done |
| Results in the paper match the code | Regenerated on CPU on 21 Nov: v1 44 files byte-identical (5 differ only by later-added columns/rows; shared values identical); v2 23 files byte-identical, 8 CSVs within 2 × 10⁻¹¹ relative; `contrasts_3seed.csv` byte-identical (`docs/REPRODUCE.md` §3). Paper numbers are traced in `report/NUMBERS.md` (added by the paper PR #44) | done |
| No dataset in the zip | `make_submission_zip.sh` uses `git archive` (tracked files only; `data/` holds only `.gitkeep`) and refuses `*.pth`, `*.pt`, `*.ply`, `*.tar`, `*.tgz`, `*.zip` | done |
| Official dataset sources | `docs/REPRODUCE.md` §1, Sec. V of the paper | done |
| Report PDF and LaTeX source in the zip | `report/` (LaTeX, refs, figures; paper PR #44, to be merged before the zip is built); the PDF is added by the zip script | PDF pending (Overleaf compile) |
| Plagiarism and attribution | [`docs/ATTRIBUTION.md`](ATTRIBUTION.md): upstream (DNGaussian fork, 3DGS licence in `LICENSE.md`), the ported DropGaussian lines, a per-file list of our changes from `git diff upstream-baseline` with authors, and team contributions from PR authorship | done |
