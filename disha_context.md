# Disha: context log

Written only by Disha's agents; read by Aayush and his agents. The protocol is in AGENTS.md, "Team context files". The plan and task IDs (AY-nn, DS-nn) are in `docs/PLAN_V2.md` §7.

## Status (rewritten on every update)
- **Updated:** Thu 29 Oct 2026, 23:10 IST
- **Working on:** DS-08 V3 regression at #19's head (Colab T4)
- **Done (PLAN_V2 §7):** DS-04 (Kaggle YES), DS-02 (reports on #11 and #14), DS-03 (PR #20), DS-05 (review of #19)
- **Open PRs:**
  - #11 (Colab: Cell V3, Cell 7b with `SEEDS`, `--work_root` and a `BLOCK` selector for T1/P/G/N): Approve @ 898b7cd
  - #14 (PLAN T1 footnote, FSGS/3DGS note, FSGS arXiv ID): Approve @ bb94fbf
  - #20 (DS-03, `stats.py` v2): CI green, Approve @ 16c6f72
- **Waiting on Aayush:**
  - set `agent-review` and merge #11 @ 898b7cd, #14 @ bb94fbf, #20 @ 16c6f72 (my session can't write commit statuses);
  - fix the High on #17 before I approve it (its CLAUDE.md names your file, so my agents would write `aayush_context.md`);
  - confirm the 5 analysis choices in #20's body before `prereg-v1`;
  - a runner smoke at #19's head before tagging `v1.0-exp` (#19 review, finding 7).
- **Next deadline:** DS-08 V3 + DS-09 approve #19 by Sat 14:00 (earlier if I can)

## Entry template
```
### <Day DD Mon, HH:MM IST> · <short title>
- Done: <what, with task IDs>
- Evidence: <PR links, commits, Drive paths, numbers>
- Needs from Aayush: <what, by when>  (or "nothing")
- Next: <my next task>
```

## Log (newest first)

### Thu 29 Oct, 23:10 IST · DS-04: Kaggle smoke test passes
- Done: DS-04. One fern run at `upstream-baseline` on Kaggle (notebook `kaggle_smoke_ds04.ipynb`, 6000 iterations, 2 x Tesla T4, Python 3.13, torch 2.11+cu128, nvcc 12.8).
- Evidence: **KAGGLE YES.** m = 4.44 min per run (train 4.03 + render/metrics 0.41; Colab T4: 4.9). Fern PSNR 21.07, SSIM 0.679, LPIPS 0.272 (Colab: 20.80). The four extensions built for sm_75. Training needed only the tag, with no token. Kaggle has no cache between sessions, so the build (~10 min) repeats every session.
- Needs from Aayush: nothing. Decision for him: block N (8 runs) can run on Kaggle as PLAN_V2 §3.3 says; with two T4s, block V is an alternative if the A100 falls through (R4). The two-GPU parallel test (Cell 7) is not run yet.
- Next: DS-08 V3 at 4f03a37, then DS-09 approve #19.

### Thu 29 Oct, 22:20 IST · Reviews of #17/#19, Cell 7b block selector, DS-03 PR #20
- Done:
  - DS-02: the reviewer reports on #11 and #14 were already posted on 14 Oct. #11 got the Cell 7b block selector today and a delta review (Approve @ 898b7cd, posted).
  - DS-05: review of #19 @ 4f03a37, Approve with 7 Lows, posted on #19.
  - Review of #17 @ 340922d: Request Changes, one High, posted on #17.
  - DS-03: PR #20 has `stats.py` v2 (§5 rules), the factor table `experiments/factors.py`, sharp checks keyed by `config.json`, and fixtures for blocks P, G and N. Two review rounds: round 1 found 1 Medium and Lows, now fixed; the delta review is Approve @ 16c6f72.
- Evidence:
  - https://github.com/aayxsh23/dngs-dropout/pull/20 (CI `checks` green on 16c6f72); reports on #11, #17, #19 and #20.
  - #20's `diag_curves.py` passes on the 29 Oct sample (`samples/sample_runs_text_29oct.tgz`, `--final_iter 1100`): 5/5 runs keyed from `config.json`, and a tampered `depth_render` is caught.
  - All 7 matrix files on `feat/dropout` match the factor table; `0.333333` is within its tolerance of 1/3.
- Needs from Aayush:
  - `agent-review` + merge of #11, #14 and #20 at the SHAs above;
  - the #17 fix;
  - the 5 choices in #20 at the 21:30 call or before `prereg-v1`;
  - the runner smoke before the tag.
  - #12 and #20 both change the verdict line that `check.sh` step 12 checks. Merge #12 first if you like: I'll rebase #20 and update the line.
- Next: DS-04 (Kaggle, tonight); DS-01 approve #12; DS-08 V3 at 4f03a37 then DS-09 approve #19.

### 12–14 Oct · Summary of earlier work (seeded from repo history)
- Drive folder `sparse3dgs`; `llff_with_depth.tar` (md5 `0f6bd7bd…0561`); Colab notebook, cached wheels and `aggregate.py` (#1).
- T4 timing 4.9 min/run; T4 reproduction (Cell 7a) 18.93 / 0.595 / 0.294; V3 baseline 19.85.
- Colab Cell 3 fix (#8); analysis scripts: `stats.py`, `diag_curves.py`, `qual_grid.py`, `views.py` (#9); verified reported baselines (#10).
- D1 confirmed; D11 changed to γ = 1/3 plus C (14 Oct).
