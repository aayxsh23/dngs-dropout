@./AGENTS.md

# Gemini specifics

Everything in AGENTS.md applies. Gemini CLI loads this file and expands the `@./AGENTS.md` import above. If your Gemini tool doesn't support imports, read AGENTS.md yourself before doing anything.

- **The Gemini assistant inside Colab does not read this repo's files.** When you ask it about the notebook, paste in the "Environments", "Invariants" and "Practical gotchas" sections of AGENTS.md. In particular, it must never write `torch.load(...)` directly; use `load_checkpoint()`. It must not regenerate depth maps, and it must not change `metrics.json` keys.
- **Team context files:** follow AGENTS.md "Team context files". Read the other person's file before starting (`bash scripts/team_context.sh show aayush`). After finishing any task, update your own with `bash scripts/team_context.sh open disha`, then edit, then `push disha "<summary>"`. On Aayush's machines the names swap. Top-level sessions only; subagents never touch these files. The current plan and task IDs are in `docs/PLAN_V2.md`.
- On Colab the repo lives at `/content/dngs-dropout`, and the shared Drive folder at `/content/drive/MyDrive/sparse3dgs`. That path needs the "Add shortcut to My Drive" step from the playbook.
- Colab runs Python 3.13 and PyTorch 2.11, but the same code must also run on the laptop's Python 3.7 / PyTorch 1.12. Don't introduce syntax or APIs newer than Python 3.7 in training code.
- **Merging** follows AGENTS.md "Change workflow": run `bash scripts/check.sh`; for experiment code also `bash scripts/gpu_smoke.sh`. Then an independent review, the fixes and a re-review. Then set the `agent-review` status on the head SHA, wait for CI, and run `python3 scripts/pr_ready.py <N>` before `gh pr merge <N> --repo aayxsh23/dngs-dropout --squash --delete-branch --match-head-commit <reviewed SHA>`. No "Generated with" footers in PR text. Never use `--admin`, push to `main`, or submit `--approve` reviews. Always pass `--repo aayxsh23/dngs-dropout`: plain `gh` can resolve to the upstream repo.
