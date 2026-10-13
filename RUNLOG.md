# RUNLOG

One line per event: what ran where, problems, reruns, decisions. Newest at the bottom. Format: `date time (IST) · machine/person · event`.

- 12 Oct · laptop/Aayush · Upstream DNGaussian reproduced on 8 LLFF scenes at `upstream-baseline`: mean 18.73 / 0.594 / 0.297 (laptop's own DPT depth)
- 12 Oct · laptop/Aayush · Full fern run 7.5 min; training VRAM footprint 1663 MB (scripts/peak_vram.py)
- 12 Oct · Colab-Disha · `llff_with_depth.tar` uploaded, md5 0f6bd7bdae837ec58ad757c7c00b0561. This is the canonical depth for all machines
- 12 Oct · Colab-Disha · Smoke/timing fern on T4: 4.9 min/run; PSNR 20.80, SSIM 0.674, SSIM_sk 0.669, LPIPS 0.257
- 12 Oct · laptop/Aayush · Same-seed nondeterminism check (fern, 1100 it): PSNR 17.75 vs 17.29
- 13 Oct · Aayush · Decisions D1–D11 (PLAN §3) recorded. All recommendations accepted except D1: keep DropGaussian's official schedule verbatim, r = 0.2 · t / 10000, not rescaled to our 6000 iterations, so the rate peaks at 0.12 (Disha to confirm). D10 settled: repo stays public
