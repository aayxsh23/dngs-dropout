---
name: pr-gate
description: This repo's mandatory review-and-merge pipeline for a pull request or the current branch. It runs the checks, a GPU smoke test where required, an independent pr-reviewer pass, a remediation loop and a review report, sets the agent-review status, and merges only when scripts/pr_ready.py says the PR is eligible. Use it whenever a change is ready to propose or merge, or when asked to review, fix up or merge a PR.
---

# pr-gate: test → independent review → fix → re-verify → merge

Follow these stages in order. Never skip a stage because a previous run "probably" still holds; re-run anything the latest commit could affect. The rules are in `AGENTS.md` ("Change workflow"); this skill is how to execute them.

**Always name the repo.** This clone also has an `upstream` remote (Fictionarry/DNGaussian), and plain `gh` may resolve to it. Every `gh pr` command below takes `--repo aayxsh23/dngs-dropout` (written `-R $R` with `R=aayxsh23/dngs-dropout`), and `gh api` paths use `repos/$R/...`.

## Stage A: Understand
1. Identify the PR (`gh pr view <N> -R $R`) or, for local work, the branch and its base (`main`).
2. Read the goal: the PR text, the linked issue or `docs/PLAN.md` item, and the acceptance criteria. If none are stated, write them down before going on.
3. Classify the PR from the changed files against `.github/CODEOWNERS`:
   - **routine:** no protected path.
   - **experiment:** training, rendering, evaluation or experiment config.
   - **gate:** CI, hooks, agent files or merge scripts.

   A PR can be both experiment and gate.

## Stage B: Validate (the author side)
1. On a clean checkout of the PR head (`gh pr checkout <N> -R $R`), run `bash scripts/check.sh`. Every failure must be fixed; never weaken a check to pass.
2. **Experiment class:** run `bash scripts/gpu_smoke.sh` on the laptop (about 1.5 min per variant).
   - Only one GPU job at a time; check `pgrep -af train_llff` first.
   - The tree must be clean, or the result is marked `-dirty` and doesn't count.
   - Post its last line as a PR comment: `gh pr comment <N> -R $R --body "<GPU-SMOKE PASS ... line>"`.
3. Make sure the PR body follows `.github/pull_request_template.md`.
4. Wait for CI: `gh pr checks <N> -R $R --watch`. If CI fails, read the log (`gh run view <id> -R $R --log-failed`) and fix.

## Stage C: Independent review
Run the **pr-reviewer** subagent (`.claude/agents/pr-reviewer.md`) on the PR with the goal and acceptance criteria. If that agent type isn't available in this session, use a fresh general-purpose or code-reviewer subagent, and paste the pr-reviewer instructions into its prompt. Do not do the review in your own context: the point is independence from the implementation.

## Stage D: Review report
Post the reviewer's report as a PR comment, verbatim, plus a "Validation" line with the check.sh, CI and GPU-smoke results you observed yourself:
`gh pr comment <N> -R $R --body-file <report.md>`. Keep the comment URL that prints.

## Stage E: Remediate and re-review (loop, at most 3 rounds)
- **Merge recommendation = Request Changes:**
  - **Your own PR:** fix every merge-blocking finding. Add or extend a check in `scripts/check.sh`, or a smoke assertion, that would have caught it, where cheap. Commit and push. Then go back to Stage B (all of it), then a **new** Stage C on the new head that includes the previous findings to re-check.
  - **Someone else's PR:** don't push to their branch unless asked. Post the report, then `gh pr review <N> -R $R --request-changes --body "<summary>"`, and stop.
- **Blocked**, or still failing after 3 rounds: stop and report the blocker to the human. No merge.
- **Approve,** with no merge-blocking findings left: continue.

## Stage F: Record approval, then merge only if eligible
1. Set the review status on the **exact head commit you reviewed**:
   ```bash
   SHA=$(gh pr view <N> -R $R --json headRefOid -q .headRefOid)
   gh api -X POST repos/$R/statuses/$SHA -f state=success -f context=agent-review \
     -f description="pr-gate: Approve, 0 blocking findings" -f target_url=<report comment URL>
   ```
   Any later push changes the head SHA, which removes this approval automatically. The review must then be redone.
2. Wait for CI to finish (`gh pr checks <N> -R $R --watch`), then check eligibility: `python3 scripts/pr_ready.py <N>`. If it says BLOCKED, report the listed blockers and stop. Typical blockers:
   - a human approval is still needed (experiment and gate classes),
   - CI failed,
   - the PR is stacked on a non-main base.
3. If ELIGIBLE: `gh pr merge <N> -R $R --squash --delete-branch`. The Bash hook re-runs the gate (from `origin/main`) before letting this through. Use `--auto` only if `pr_ready.py <N> --auto` reports that GitHub enforces the checks: that needs a ruleset, which needs a public repo or GitHub Pro.
4. Confirm. Only report "merged" if `gh pr view <N> -R $R --json state,mergedAt` shows `MERGED`.
5. If the command is denied (by the hook, by Claude Code permissions or by GitHub), do not look for another route. Report exactly what blocked it.

## Never
- `gh pr merge --admin`, pushing to `main`, force-pushing `main`, or disabling or editing the hook or CI to get a merge through.
- `gh pr review --approve`. Agents never approve; humans do.
- Treating a missing CI result, an unavailable reviewer or absent command output as a pass.
- Approving or merging changes to the gate itself (class "gate") without the other teammate's approval.
- Putting tokens or secrets in comments, statuses or logs.
