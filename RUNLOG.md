# RUNLOG

One line per event: what ran where, problems, reruns, decisions. Newest at the bottom. Format: `date time (IST) · machine/person · event`.

- 12 Oct · laptop/Aayush · Upstream DNGaussian reproduced on 8 LLFF scenes at `upstream-baseline`: mean 18.73 / 0.594 / 0.297 (laptop's own DPT depth)
- 12 Oct · laptop/Aayush · Full fern run 7.5 min; training VRAM footprint 1663 MB (scripts/peak_vram.py)
- 12 Oct · Colab-Disha · `llff_with_depth.tar` uploaded, md5 0f6bd7bdae837ec58ad757c7c00b0561. This is the canonical depth for all machines
- 12 Oct · Colab-Disha · Smoke/timing fern on T4: 4.9 min/run; PSNR 20.80, SSIM 0.674, SSIM_sk 0.669, LPIPS 0.257
- 12 Oct · laptop/Aayush · Same-seed nondeterminism check (fern, 1100 it): PSNR 17.75 vs 17.29
- 13 Oct · Aayush · Decisions D1–D11 (PLAN §3) recorded. All recommendations accepted except D1: keep DropGaussian's official schedule verbatim, r = 0.2 · t / 10000, not rescaled to our 6000 iterations, so the rate peaks at 0.12 (Disha to confirm). D10 settled: repo stays public
- 13 Oct · Colab-Disha · Upstream DNGaussian reproduced on 8 LLFF scenes on T4 (Cell 7a, `upstream-baseline`, canonical depth tar md5 0f6bd7bd…0561): mean 18.93 / 0.595 / 0.294 vs paper 19.12 / 0.591 / 0.294. Fern 21.08, horns 18.61, so the V3 baseline (mean of the two) is 19.85. Results in Drive `sparse3dgs/outputs/repro_upstream/`
- 13 Oct · Aayush · GitHub ruleset "main: review and CI gate" applied (`scripts/setup_repo_rules.sh --apply`): `checks`, `agent-review` and code-owner review are required on `main`; merges are squash-only
- 14 Oct · Disha · D1 confirmed (0.2 · t / 10000, peak 0.12). D11 changed: the Tier 2b high arm is γ = 1/3 (peaks at exactly 0.2 at iteration 6000, the published end rate) instead of 0.3, and C is added at γ = 1/3 (+8 runs), so the interaction contrast exists at the published dose
- 14 Oct · Aayush · Deadline moved to Sun 15 Nov 2026; the team target is to submit on 10–11 Nov. The plan will be redrafted after a research-scope analysis
- 14 Oct · Aayush · Merged #6 (review bundle), #8 (Colab Cell 3), #9 (analysis scripts) and #10 (verified baselines; FSGS now 20.31 / 0.652 / 0.288)
- *Back-filled 16 Nov from merged PRs, tag messages, `results/raw/README.md` and the team context logs (no entries were written 15 Oct – 15 Nov):*
- 29 Oct · Aayush · PLAN_V2 and RESEARCH_ANALYSIS merged (#16): restructured study, blocks T1/P/G/N/V
- 30 Oct · Disha · Kaggle smoke at `upstream-baseline`: YES, 4.44 min/run, fern 21.07 (context log). #11, #14 and #20 (stats.py v2) merged
- 3 Nov · both · V3 regression at 7633acc: fern 20.74, horns 18.58, mean 19.66 vs 19.85 (PASS, comment on #19). #12, #22 merged; #19 merged as 1040559; tags `v1.0-exp` and `prereg-v1` on 1040559. #27 (clamp-aware dropout) opened, not merged. Gate-tooling freeze
- by 6 Nov · lab RTX A4000 · All blocks ran on the lab PC (`/home/snu/dngs-dropout`, PyTorch 1.12.1) from `v1.0-exp`, `shared_gpu: false`, instead of Colab/A100: T1 120, P 48, G 48, N 8, V 120 (2/6/9 views, seed 0; 3 views = T1 seed 0) = 344 runs, 0 failures, done by Fri 6 Nov 23:59
- 11 Nov · lab · Probes on the 40 T1 seed-0 checkpoints (`probes.py` at 9f8d55f, md5 1a2913d0b2fbf219c528cbd87d04b8e7); MVC@5% geometry on all 344 runs. Backup `results_full_nockpt.tgz` on Drive (2,841,905,876 bytes, md5 2ea56c510519ff48a8a66fc3bae548ae). Results v1 merged (#29–#31)
- 15 Nov · both · New execution plan (docs/EXECUTION_PLAN_15-21Nov.md): error-budget framing, DTU required + Blender, rigor block; submission moved to Sat 21 Nov 20:00
- 16 Nov · Aayush · Run stop moved Thu 12:00 → Fri 12:00, analysis freeze → Fri 20:00, DTU throughput check → Thu 12:00. Lab PC may run unattended overnight in tmux (one visit a day)
- 16 Nov · Aayush · DTU go/no-go not run on Mon (no data yet). DTU prep moved to the laptop: HTTP-range download of only the 735 needed images (15 scans × 49, light 3) from Rectified.zip, COLMAP 4.2.1 on CPU (conda env `colmap`, option-name wrapper `~/dtu_prep/colmap`). Blender is the required second dataset until DTU passes
- 16 Nov · Aayush (Claude) · Prepared, not pushed: `feat/clamp-aware` rebased on main + `--drop_mode nc` + `matrix_comp.yaml` (96) + `matrix_depthdose.yaml` (96) for `v1.1-exp`; `feat/multi-dataset` (DTU/Blender port, eval_dtu.py, dataset key, matrix_dtu 225 / matrix_blender 120) for `v1.2-exp`; `chore/probes-script`. check.sh passes on each
- 17 Nov · laptop (CPU) · DTU COLMAP (convert.py --no_gpu, COLMAP 4.2.1, 4 threads): **15/15 scans registered 49/49**, so no scan is excluded. 11 scans with the default OPENCV camera; scan30, scan63, scan103 and scan110 needed the one pre-registered PINHOLE retry. Log: data/dtu/registration.txt. Next: DPT (DTU), idrmasks, dtu_with_depth.tar + md5, then the go/no-go run on scan8
