# Copilot instructions

The full agent guide is [AGENTS.md](../AGENTS.md), and the current plan is [docs/PLAN.md](../docs/PLAN.md). Read both for anything non-trivial. The rules that matter most when completing code:

- This is a DNGaussian fork used for a controlled experiment. Variants A–E differ **only** by the flags `--depth_scale`, `--drop_rate` and `--depth_render`. Default flags must reproduce upstream DNGaussian exactly.
- Gaussian dropout (ported from DropGaussian) runs **only in the training loop**. Never add it to `render.py`, `metrics.py`, `spiral.py` or evaluation helpers.
- Training code must run on **Python 3.7 / PyTorch 1.12** (laptop) and **Python 3.13 / PyTorch 2.11** (Colab). No syntax newer than Python 3.7; load checkpoints with `utils.general_utils.load_checkpoint()`, never `torch.load`.
- `metrics.py` writes `results_eval.json` (keys `PSNR`, `SSIM`, `SSIM_sk`, `LPIPS` under `ours_<iter>`). The per-run output files and their keys are frozen: add keys, never rename them.
- Don't edit the vendored CUDA sources (`submodules/`, `gridencoder/`, `shencoder/`) unless the task is a build fix.
- Never commit data, outputs, checkpoints, `.ply` files or tokens.
- Every merge goes through AGENTS.md "Change workflow": `scripts/check.sh` (CI job `checks`), the GPU smoke test for experiment code, an independent review, then `scripts/pr_ready.py`. Protected paths are in `.github/CODEOWNERS`.
