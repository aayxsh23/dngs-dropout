@AGENTS.md

# Claude Code specifics

Everything in AGENTS.md applies. This section adds only what is specific to Claude Code on Aayush's laptop.

- **Python:** `~/miniconda3/envs/dngaussian/bin/python`. Non-interactive shells don't activate conda, so call that path directly. To rebuild an extension, export the build environment first. Your shell may not source `~/.bashrc`; the variables are listed in `setup_env.sh` step 6:
  `PATH=$HOME/gcc10bin:/usr/local/cuda-11.3/bin:$PATH CUDA_HOME=/usr/local/cuda-11.3 CPATH=$HOME/build-deps/linux-headers-5.4/include LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-11.3/lib64 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2`
- **GPU jobs:** before starting one, check that the GPU is free (`pgrep -af "train_llff|render.py|metrics.py"`). Never start a second training job while one is running; the laptop has 4 GB. For anything longer than about 2 minutes, run it in the background and redirect output to a log under `output/`.
- **GitHub:** `gh` is installed in `~/.local/bin` and logged in as `aayxsh23`. Use `gh pr create --body-file` for PRs. End PR bodies with the Claude Code attribution line.
- **Scratch files:** downloads and experiments go in the session scratchpad, not the repo. Treat cloned third-party repos, such as DropGaussian_release, as read-only references; read them, don't execute them.
- **Merging:** always through the `pr-gate` skill. Use the `pr-reviewer` subagent for the independent review; never review your own change in your own context. The project hook (`.claude/settings.json`) blocks pushes to `main`, `--admin` merges, and any merge `scripts/pr_ready.py` rejects. If the hook, a permission prompt or GitHub refuses a merge, report it; don't route around it.
- **GPU smoke test:** `bash scripts/gpu_smoke.sh` runs for 1.5–7 min. That is fine in the daytime. Longer GPU jobs only run overnight on Aayush's laptop.
- **Status:** when you finish a task from `docs/PLAN.md`, update its status table in the same branch.
