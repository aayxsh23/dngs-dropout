@AGENTS.md

# Claude Code specifics

Everything in AGENTS.md applies. This section adds only what is specific to Claude Code on Aayush's laptop.

- **Python:** `~/miniconda3/envs/dngaussian/bin/python`. Non-interactive shells don't activate conda, so call that path directly. To rebuild an extension, export the build environment first. Your shell may not source `~/.bashrc`; the variables are listed in `setup_env.sh` step 6:
  `PATH=$HOME/gcc10bin:/usr/local/cuda-11.3/bin:$PATH CUDA_HOME=/usr/local/cuda-11.3 CPATH=$HOME/build-deps/linux-headers-5.4/include LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-11.3/lib64 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2`
- **GPU jobs:** before starting one, check that the GPU is free (`pgrep -af "train_llff|render.py|metrics.py"`). Never start a second training job while one is running; the laptop has 4 GB. For anything longer than about 2 minutes, run it in the background and redirect output to a log under `output/`.
- **GitHub:** `gh` is installed in `~/.local/bin` and logged in as `aayxsh23`. Use `gh pr create --body-file` for PRs, and always pass `--repo aayxsh23/dngs-dropout`. **Hard rule: never put a "🤖 Generated with Claude Code" footer (or any similar line) in PR descriptions, PR comments or review reports.**
- **Scratch files:** downloads and experiments go in the session scratchpad, not the repo. Treat cloned third-party repos, such as DropGaussian_release, as read-only references; read them, don't execute them.
- **Merging:** always through the `pr-gate` skill. Use the `pr-reviewer` subagent for the independent review; never review your own change in your own context. The project hook (`.claude/settings.json`) blocks pushes to `main`, `--admin` merges, and any merge `scripts/pr_ready.py` rejects. If the hook, a permission prompt or GitHub refuses a merge, report it; don't route around it.
- **GPU smoke test:** `bash scripts/gpu_smoke.sh` runs for 1.5–7 min. That is fine in the daytime. Longer GPU jobs only run overnight on Aayush's laptop.
- **Plan and status:** the plan is `docs/PLAN_V2.md`. Progress is recorded by task ID in `aayush_context.md` on the `team-context` branch (AGENTS.md, "Team context files"), not in PLAN.md's old status table.
- **Start of every session:** read `disha_context.md` from `origin/team-context`. Tell Aayush in one or two lines anything new from Disha that affects the task: a blocker, a request, a finished task he depends on.
- **After every piece of work** (code, a PR, a merge, runs, analysis, a decision):
  1. Update `aayush_context.md` on `team-context`: rewrite the Status block and add a log entry.
  2. **End your final reply with a "Message to Disha" block**: a ready-to-send chat message. It must say:
     - what was done, with PR links;
     - exactly what she needs to do next (review or approve PR #N, run a Colab shard or a check, look at a file or figure), with the PLAN_V2 task IDs;
     - by when;
     - anything she should *not* do yet.

     If nothing is needed from her, say so in the message and still summarize what changed. Write it in plain words she can act on without opening the repo first.
  3. Copy the same message into the log entry's "Message to Disha" field.
