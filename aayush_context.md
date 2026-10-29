# Aayush: context log

Written only by Aayush's agents; read by Disha and her agents. The protocol is in AGENTS.md, "Team context files". The plan and task IDs (AY-nn, DS-nn) are in `docs/PLAN_V2.md` §7.

## Status (rewritten on every update)
- **Updated:** Thu 29 Oct 2026, evening (IST)
- **Working on:** waiting for approvals; A100 setup visit Fri (AY-10)
- **Done (PLAN_V2 §7):** AY-01 (#16 merged), AY-02, AY-03, AY-06, AY-07, AY-08, AY-09 + AY-11 (one PR, #19), AY-21 (`geometry.py`, early), AY-22 (`probes.py`, early, branch `feat/probes`); AY-05 partly (git bundle and visit kit ready; `conda pack` waits on the lab's OS)
- **Waiting on Disha:** approve #19 after the V3 regression at its head (DS-08/DS-09; the earlier the better, at the latest Sat 14:00); approve #17 and #12; reviewer reports on #11 and #14; DS-03 `stats.py` v2 (Fri 18:00); DS-04 Kaggle yes/no
- **Next deadline:** tags `v1.0-exp` + `prereg-v1` as soon as #19 is approved (plan: Sat 16:00), then Tier 1 starts on Colab

## Entry template
```
### <Day DD Mon, HH:MM IST> · <short title>
- Done: <what, with task IDs>
- Evidence: <PR links, commits, Drive paths, numbers>
- Needs from Disha: <what, by when>  (or "nothing")
- Next: <my next task>
- Message to Disha: <the ready-to-send message>
```

## Log (newest first)

### Thu 29 Oct, evening · Dropout code complete (#19), tested and reviewed
- Done: AY-02, AY-03, AY-06, AY-07, AY-08, AY-09 + AY-11 (merged into one PR, #19), AY-21, AY-22. Merged #16 (PLAN_V2). #17 rebased and approved by the reviewer; it needs Disha's approval.
- Evidence:
  - #19 head 4f03a37: L1 routing test on all 7 variants (B 17.59 PSNR, no mask; D/E/D_hard/D_soft receive the mask exactly where specified); `diag.csv` sharp checks pass; V4 render determinism; GPU-SMOKE PASS 4f03a37.
  - Runner smoke: 5/5 runs, all `metrics.json` keys present.
  - Opus review: 1 blocker (AGENTS invariant 7), fixed; delta review Approve; `agent-review` set.
  - `feat/probes` 9f8d55f: f=0 reproduces the eval PSNR. Sample on Drive: `sparse3dgs/samples/sample_runs_text_29oct.tgz`.
  - A100 kit: `~/dngs-gobag/` (bundle + `A100_VISIT.md`).
- Needs from Disha: V3 regression at 4f03a37 and approval of #19 (at the latest Sat 14:00; earlier lets us start Tier 1 sooner); approve #17 and #12; reviewer reports on #11 and #14; DS-03 Fri 18:00; DS-04 Kaggle.
- Next: A100 visit 1 (AY-10, Fri); tag `v1.0-exp` and `prereg-v1` once #19 is approved (AY-12); then start T1 seed 0 (AY-13).
- Message to Disha: "Hi Disha, the dropout code is done a day early, so we can freeze sooner. (1) PR #19 has everything for the matrix: the dropout port (D1 schedule), D/E plus the new D_hard/D_soft routing, diag.csv, train_stats/render_stats, raw depth for the geometry metric, and the runner with all matrix files. It passed the laptop tests, two GPU smoke tests and two review rounds; it only needs your approval. Please run the V3 regression on Colab at its head first: branch feat/dropout, commit 4f03a37, variant B (--depth_scale 1, --seed 0) on fern and horns at 6000 iterations. The mean must be within 0.5 dB of 19.85. Post the numbers on #19, then approve it (DS-08/DS-09). Today or tomorrow morning if you can, at the latest Sat 14:00. (2) Please also approve #17 (team context files) and #12 (CI for your scripts), and post your reviewer reports on #11 and #14 so I can merge them. (3) Sample outputs of 5 real runs (A-E, fern, 1100 iterations) are on Drive: sparse3dgs/samples/sample_runs_text_29oct.tgz. Your diag_curves.py sharp checks already pass on them with --final_iter 1100. New variant IDs for the stats.py factor table: D_hard/D_soft (depth_render hard_only/soft_only), C_g010/D_g010/E_g010 and C_g033/D_g033/E_g033 (drop_rate 0.1 and 0.333333), B_r1-4/D_r1-4 (noise repeats). (4) Still on for Fri: DS-03 stats.py v2 PR by 18:00, and the Kaggle yes/no. Please don't start any Colab matrix runs until I post that v1.0-exp is tagged."

### Thu 29 Oct · #17 hardened: scripts/team_context.sh, checks 13–14
- Done: Review round 1 on #17 found the multi-line snippet unsafe for agents (it could push the checked-out branch to team-context). It is replaced by `scripts/team_context.sh` (show/open/push), tested end to end in `check.sh` step 13. Step 14 fails while the agent guides link to a missing file. Subagents never touch these files. Round 2: one Low fixed (`show` line count). Issue #18 filed for a hook follow-up.
- Evidence: PR #17 (head 27f46fb), reports on #17, issue #18. CI on #17 is red on step 14 only, by design until #16 merges. #16 is docs-only and eligible.
- Needs from Disha: approve #12 now; approve #17 after it is rebased onto `main` and CI is green (tonight); DS-02 reports on #11 and #14; DS-04 Kaggle yes/no.
- Next: merge #16 (on Aayush's go), rebase #17, then AY-02/AY-03.
- Message to Disha: "Hi Disha, update for Thu 29 Oct. (1) PLAN_V2 (PR #16) is ready to merge; it has your task list DS-01 to DS-24 in §7. (2) The context-file setup (PR #17) went through two review rounds. Agents now use `bash scripts/team_context.sh show aayush` to read my file and `open disha` / `push disha \"<summary>\"` to update yours, instead of the long snippet. #17 can only merge after #16; I'll rebase it tonight and ping you. Please approve it then. Today, please: approve #12; post your reviewer reports on #11 and #14 by 18:00; read PLAN_V2 §0, §2, §5, §7 before the 21:30 call; start DS-03 (stats.py v2, PR by Fri 18:00); and tell me yes/no on Kaggle tonight (DS-04). Please don't start any Colab runs before the tag on Sat 16:00."

### Thu 29 Oct · PLAN_V2 with novelty items; team context files
- Done: PLAN_V2 updated with novelty items 1–5 and a dated task list (§7); research analysis; team-context protocol in AGENTS/CLAUDE/GEMINI/Copilot; this branch and both context files created.
- Evidence: PR #16 (docs-only: RESEARCH_ANALYSIS + PLAN_V2), PR #17 (gate: protocol), branch `team-context`.
- Needs from Disha: DS-01 approve #17 and #12 and read PLAN_V2 §0, §2, §5, §7 by 21:30 today; DS-02 reviewer reports on #11 and #14 by 18:00; DS-03 `stats.py` v2 PR by Fri 18:00; DS-04 Kaggle yes/no tonight.
- Next: AY-02 rebase `feat/dropout`; AY-03 dropout port + `clean`/`hard_only`/`soft_only`, L1 tests by 23:59.
- Message to Disha: "Hi Disha, plan update for Thu 29 Oct. (1) PLAN_V2 is updated (PR #16, docs-only). It adds five novelty items inside our question: D_hard/D_soft variants to find which depth pass carries any interference (48 runs); raw depth plus a multi-view depth-consistency metric; test-time robustness and depth-bias probes; per-scene factors; and the framing 'where should dropout act?'. §7 has a dated task list for each of us; yours are DS-01 to DS-24. (2) PR #17 adds our context files on the `team-context` branch: your agent should read aayush_context.md before working and keep disha_context.md updated (I seeded yours from repo history, please correct it). Today: approve #17 and #12 and read PLAN_V2 §0, §2, §5, §7 by 21:30; post reviewer reports on #11 and #14 by 18:00 so I can merge them; start stats.py v2 with config-based sharp checks and fixtures for the new variant IDs (PR by Fri 18:00); run the Kaggle smoke test and tell me yes/no tonight. Please don't start any Colab runs yet: the code is tagged on Sat at 16:00."

### 12–28 Oct · Summary of earlier work (seeded 29 Oct)
- Laptop environment and DNGaussian reproduction: 18.73 PSNR (laptop depth); 7.5 min per run, 1663 MB VRAM.
- `--seed`, `--depth_scale`, `config.json` on branch `feat/dropout` (no PR yet).
- Merge gate (#5), cheaper reviews (#6), GitHub ruleset on `main` (13 Oct); decisions D1–D11 (#7, #13). #12 (CI runs the analysis scripts; invariant 7 = D1) open, waiting on Disha's approval.
- Research analysis (14 Oct) and PLAN_V2 (28–29 Oct): PR #16. Team-context protocol: PR #17.
