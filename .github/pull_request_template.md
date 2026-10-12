## What and why
<!-- One paragraph: the change, the reason, the issue or PLAN.md item it addresses. -->

## Risk class
<!-- Tick one. Defined in AGENTS.md "Change workflow"; protected paths are listed in .github/CODEOWNERS. -->
- [ ] **Routine**: no protected paths. Can merge after CI + agent review.
- [ ] **Experiment**: touches training, rendering, evaluation or experiment config. Also needs GPU smoke evidence + the other teammate's approval. After `v1.0-exp`: new tag + RUNLOG entry.
- [ ] **Gate**: touches CI, hooks, agent instructions or merge scripts. Needs the other teammate's approval.

## Validation
- [ ] `bash scripts/check.sh` passes locally (CI reruns it as the `checks` job)
- [ ] Experiment class only: `bash scripts/gpu_smoke.sh` on the head commit; its `GPU-SMOKE PASS <sha>` line is posted as a comment
- [ ] Defaults still reproduce DNGaussian (variant B) where training code changed
- Not run, and why:

## Review
- [ ] Agent review report posted as a comment (pr-gate skill); recommendation: Approve / Request Changes / Blocked
- [ ] `agent-review` status set on the head commit
- [ ] Human approval from the other teammate (experiment and gate classes)
