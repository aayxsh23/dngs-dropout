# Aayush: context log

Written only by Aayush's agents; read by Disha and her agents. The protocol is in AGENTS.md, "Team context files". The plan and task IDs (AY-nn, DS-nn) are in `docs/PLAN_V2.md` §7.

## Status (rewritten on every update)
- **Updated:** Thu 29 Oct 2026 (IST), after review round 2 of #17
- **Working on:** AY-02 (rebase `feat/dropout`), AY-03 (dropout port + `clean`/`hard_only`/`soft_only`)
- **Done (PLAN_V2 §7):** — (PLAN_V2 and the context protocol are prep work, not numbered tasks)
- **Waiting on Disha:** DS-01 (read PLAN_V2; approve #12; approve #17 after its rebase), DS-02 (reviewer reports on #11 and #14), DS-04 (Kaggle yes/no)
- **Waiting on Aayush:** go-ahead to merge #16 (docs-only, eligible); #17 then gets rebased, re-checked and handed to Disha
- **Next deadline:** PR-A open Fri 30 Oct 19:00; code freeze and tags Sat 31 Oct 16:00

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
