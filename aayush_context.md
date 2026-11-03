# Aayush: context log

Written only by Aayush's agents; read by Disha and her agents. The protocol is in AGENTS.md, "Team context files". The plan and task IDs (AY-nn, DS-nn) are in `docs/PLAN_V2.md` §7.

## Status (rewritten on every update)
- **Updated:** Tue 3 Nov 2026, evening (IST)
- **Working on:** waiting for Disha's V3 + approvals of #19 and #17; then tags and T1 on Colab
- **Done (PLAN_V2 §7):** AY-01–03, AY-06–09, AY-11 (#19), AY-21, AY-22 (`feat/probes`), AY-04 (merged #12, #22), AY-16 + AY-20 drafts (PR #23)
- **Waiting on Disha:** (0) approve #24 (tracker) at baad1be; (1) V3 regression at #19 (any head since 4f03a37; training code unchanged) and approval of #19 at **7633acc**; (2) re-approve #17 at **540a42b**; (3) confirm #20's 5 analysis choices
- **Tracker:** done AY-02..03, AY-05..09, AY-11, AY-12, AY-16, AY-20, AY-21; partial AY-22; working AY-13, AY-10; runs T1=0 P=0 G=0 N=0 V=0
- **A100:** access and runs per day confirmed by Aayush on Wed 4 Nov; block V's plan (full 160, 2/3/6 views only, or Kaggle fallback) is decided then
- **Next deadline:** tags `v1.0-exp` + `prereg-v1` and T1 seed 0 start on both Colab accounts the same evening Disha approves (target Tue 3 Nov night, latest Wed 4 Nov 12:00)

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

### Tue 3 Nov, late · Clamp-aware dropout PR #27 (upgrade 1, gated runs)
- Done: decisions recorded with Aayush: tooling freeze until Fri 23:59 (issues #15, #18, #21, #25, #26 wait); clamp-aware dropout = write code now, run only if the Thu 12:00 gate passes; DTU, second host, extra seeds wait for the core result. PR #27: `--drop_mode clamp_aware` (per-Gaussian rate min(r, 1 - opacity/0.99)), `frac_over_cap` in diag.csv, block CA matrix (C_ca, D_ca, 48 runs), PLAN_V2 §5.1 pre-registration addendum (to be tagged prereg-v2 with v1.1-exp).
- Evidence: check.sh step 18 (Monte Carlo: uniform biased, clamp-aware unbiased; mutation caught). GPU-SMOKE PASS 26a2a33 (B 17.65). Smoke finding: median opacity 0.98-0.998, frac_over_cap 0.56-0.63 already at r = 0.022, so compensation is cut for most Gaussians in this host.
- Needs from Disha: approve #27 after its review; extend factors.py/stats.py for C_ca and D_ca (drop_mode) before block CA runs.
- Next: review of #27; T1 seed 0 shard 0/2 on Colab A.
- Message to Disha: see the chat message of Tue 3 Nov late.

### Tue 3 Nov, ~21:00 IST · #19 merged; v1.0-exp and prereg-v1 tagged; #17 re-reviewed
- Done: AY-12. #19 squash-merged as 1040559 (tree identical to 7633acc: GPU smoke, V3 19.66, Disha's approval); tags `v1.0-exp` and `prereg-v1` on it; CI green on main; runner dry run from the tag OK. #17 conflicted again (check.sh), merged main, delta review Approve, agent-review set on 366bee2. Issue #26: hook fails open when CLAUDE_PROJECT_DIR is unset (found by Disha's cloud setup check).
- Evidence: #19 comment with the tags; #17 report; issue #26.
- Needs from Disha: start T1 seed 0 shard 1/2 now; approve #17 at 366bee2 and #24 at baad1be; do the setup on her local machine.
- Next: T1 seed 0 shard 0/2 on Colab A (AY-13); research-upgrade plan (clamp-aware dropout, DTU, power).
- Message to Disha: see the chat reply of Tue 3 Nov ~21:00.

### Tue 3 Nov, night · Project tracker PR #24
- Done: automated tracker (plan.yaml re-dated: freeze Fri 6, draft Mon 9, submit Wed 11; rubric coverage; prospects FP-1..9). Review round 1: 1 High (tracker.py as gate code, read-only build job) + Medium + 5 Lows, all fixed with check.sh step 10b tests; delta review Approve; agent-review set. Follow-up issue #25.
- Evidence: PR #24 head baad1be, CI green; live summary 23 % done, critical path DS-08 -> DS-09 -> AY-12 -> EXP-T1 -> ... -> SUBMIT.
- Needs from Disha: approve #24; add a `**Tracker:**` line to her Status block.
- Next: tags + T1 once #19 is approved; FP-1 clamp-safe code on Wed while T1 runs.
- Message to Disha: see the chat message of Tue 3 Nov night.

### Tue 3 Nov, evening · Merged #12 and #22; #17 and #19 updated for the new main; Method and Setup drafts
- Done: merged #12 (CI analysis scripts) and #22 (DS-06 plots, after checking Disha's reviewer Approve at 58faa48). Both #17 and #19 conflicted with the new main in `check.sh`; merged main into each (step 12 before 13–14 / 15–17), delta reviews Approve, `agent-review` set. AY-16/AY-20: Method and Experimental setup drafts, PR #23 (docs-only).
- Evidence: #19 head 7633acc: GPU-SMOKE PASS (A 17.04, B 17.52, C 17.31, D 17.87, E 17.67), check.sh and CI green. #17 head 540a42b: check.sh and CI green. #23: `docs/report/03_method.md`, `docs/report/04_setup.md`.
- Needs from Disha: V3 + approve #19 at 7633acc; re-approve #17 at 540a42b (GitHub needs approval on the new head); confirm #20's 5 choices; read #23 §3.4 and §4.6.
- Next: merge #19 and #17, tag `v1.0-exp` + `prereg-v1`, start T1 seed 0 (shard 0/2) on Colab A; open the probes PR; figure F1 (AY-24).
- Message to Disha: see the chat message of Tue 3 Nov.

### Fri 30 Oct, midday · Disha's reviews handled; #12, #17, #19 ready for her approval
- Done: AY-04 (merged #14, #11, #20 after checking her reviewer's Approve); #19 runner Lows 1-6 fixed (check.sh step 17); #17 High fixed (identity-aware `team_context.sh`, step 13); #12 rebased onto #20 (step 12: new verdict line, blocks P/G/N). Follow-ups: issues #15, #18, #21.
- Evidence: #19 head 2957499: GPU-SMOKE PASS (A-E), runner smoke 5 done 0 failed, delta review Approve, agent-review set. #17 head ff2df54 and #12 head b3ef4b7: Approve reports and agent-review set. CI green on all three.
- Needs from Disha: V3 at 2957499 and approval of #19 (by Sat 14:00); approve #17 and #12; confirm #20's 5 analysis choices before prereg-v1.
- Next: A100 visit 1 (AY-10); then tags v1.0-exp and prereg-v1 and T1 seed 0 once #19 is approved (AY-12, AY-13).
- Message to Disha: "Hi Disha, thanks for the Kaggle test and the reviews. Kaggle is a yes: block N goes there, and I'll note block V as the fallback if the A100 falls through. Done on my side: I merged #14, #11 and #20 (stats.py v2) with the agent-review statuses. I fixed your findings on #19 (runner Lows 1-6) and on #17 (your High: the script now refuses any file but your own, and CLAUDE.md follows who the session works for; set TEAM_CONTEXT_ME=disha in your cloud sessions). I also rebased #12 onto #20 and updated its step 12 to the new verdict line and blocks P, G, N. Both smoke tests pass at #19's new head 2957499: GPU-SMOKE PASS (A-E) and the runner smoke, 5 done and 0 failed (posted on #19). What I need from you: (1) The V3 regression at 2957499, with Cell 3 set to TAG='feat/dropout' as you said. The code is the same as 4f03a37, so a result from either commit is valid. Post the two numbers on #19, then approve it (DS-08/09). Fine tomorrow morning, but the earlier the better, at the latest Sat 14:00. (2) Approve #17 (head ff2df54) and #12 (head b3ef4b7): both have an Approve report and the agent-review status. (3) Please confirm the 5 analysis choices in #20's description. I agree with all five; with your OK I tag prereg-v1. Please don't start any Colab matrix runs until I post that v1.0-exp is tagged."

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
