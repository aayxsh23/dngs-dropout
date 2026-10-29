# Aayush: context log

Written only by Aayush's agents; read by Disha and her agents. The protocol is in AGENTS.md, "Team context files". The plan and task IDs (AY-nn, DS-nn) are in `docs/PLAN_V2.md` §7.

## Status (rewritten on every update)
- **Updated:** Thu 29 Oct 2026 (IST)
- **Working on:** AY-02 (rebase `feat/dropout`), AY-03 (dropout port + `clean`/`hard_only`/`soft_only`)
- **Done (PLAN_V2 §7):** —
- **Waiting on Disha:** DS-01 (approve #17 and #12, read PLAN_V2), DS-02 (reviews on #11 and #14), DS-04 (Kaggle yes/no)
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

### 12–28 Oct · Summary of earlier work (seeded 29 Oct)
- Laptop environment and DNGaussian reproduction: 18.73 PSNR (laptop depth); 7.5 min per run, 1663 MB VRAM.
- `--seed`, `--depth_scale`, `config.json` on branch `feat/dropout` (no PR yet).
- Merge gate (#5), cheaper reviews (#6), GitHub ruleset on `main` (13 Oct); decisions D1–D11 (#7, #13). #12 (CI runs the analysis scripts; invariant 7 = D1) open, waiting on Disha's approval.
- Research analysis (14 Oct) and PLAN_V2 (28–29 Oct): PR #16. Team-context protocol: PR #17.
