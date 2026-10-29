# Disha: context log

Written only by Disha's agents; read by Aayush and his agents. The protocol is in AGENTS.md, "Team context files". The plan and task IDs (AY-nn, DS-nn) are in `docs/PLAN_V2.md` §7.

> Seeded by Aayush's agent on 29 Oct from repo history so the file isn't empty. **Disha: please correct anything wrong and take it over.**

## Status (rewritten on every update)
- **Updated:** (seeded) Thu 29 Oct 2026
- **Working on:** DS-01 – DS-04 (PLAN_V2 §7.1)
- **Done (PLAN_V2 §7):** —
- **Open PRs:** #11 (Colab: Cell V3 regression check; Cell 7b `SEEDS` and `--work_root`), #14 (PLAN T1 footnote, FSGS/3DGS note, FSGS arXiv ID). Both need a review report before merging
- **Waiting on Aayush:** sample run folder (AY-08, Fri 18:00); PR-A for review (AY-09, Fri 19:00)
- **Next deadline:** `stats.py` v2 PR (DS-03) Fri 30 Oct 18:00

## Entry template
```
### <Day DD Mon, HH:MM IST> · <short title>
- Done: <what, with task IDs>
- Evidence: <PR links, commits, Drive paths, numbers>
- Needs from Aayush: <what, by when>  (or "nothing")
- Next: <my next task>
```

## Log (newest first)

### 12–14 Oct · Summary of earlier work (seeded from repo history)
- Drive folder `sparse3dgs`; `llff_with_depth.tar` (md5 `0f6bd7bd…0561`); Colab notebook, cached wheels and `aggregate.py` (#1).
- T4 timing 4.9 min/run; T4 reproduction (Cell 7a) 18.93 / 0.595 / 0.294; V3 baseline 19.85.
- Colab Cell 3 fix (#8); analysis scripts: `stats.py`, `diag_curves.py`, `qual_grid.py`, `views.py` (#9); verified reported baselines (#10).
- D1 confirmed; D11 changed to γ = 1/3 plus C (14 Oct).
