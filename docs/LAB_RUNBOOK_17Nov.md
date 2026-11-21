# Lab runbook, Tue 17 Nov: LLFF blocks for the error budget and compensation

**Who:** Disha at the lab PC. Aayush works remotely and pushes the `v1.1-exp` tag during the day.
**Where:** lab PC (RTX A4000), repo `/home/snu/dngs-dropout`. All outputs go under this folder.
**What runs today:** 480 LLFF runs, in one queue that runs unattended overnight.

| Order | Block | Runs | Tag | Matrix | Output root | Wall time with 3 slots (estimate) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Views, seeds 1–2 (2/6/9 views, A–E) | 240 | `v1.0-exp` (exists) | `matrix_views.yaml --seeds 1 2` | `outputs_views/` (next to seed 0) | ~4.5 h |
| 2 | Compensation: C_ca, D_ca, C_nc, D_nc | 96 | `v1.1-exp` (pushed today) | `matrix_comp.yaml` | `outputs_comp/` | ~2 h |
| 3 | Depth dose: B/D at depth_scale 0.5 and 2 | 96 | `v1.1-exp` | `matrix_depthdose.yaml` | `outputs_depthdose/` | ~2 h |
| 4 | noise2: A/B/C/D × 4 repeats on room, horns, trex | 48 | `v1.1-exp` | `matrix_noise2.yaml` | `outputs_noise2/` | ~1 h |

**Not today:** DTU and Blender. They need `v1.2-exp` and the two data tars, which are planned for Wednesday.

**The idea.**
- Start one queue with three slots. Views runs first, because its tag already exists.
- Blocks 2–4 **wait for `v1.1-exp`** on their own: the queue checks GitHub every 10 minutes and starts them as soon as Aayush pushes the tag.
- You don't have to be at the lab when the tag lands.

**Copy commands from a terminal, never from a rendered page.** On 11 Nov a renderer ate the `$` signs. Use one of these:
- `git show origin/docs/research-guide:docs/LAB_RUNBOOK_17Nov.md | less` on the lab PC;
- the "Raw" view on GitHub.

---

## 0. Before you leave home (2 min): `prereg-v3` must be tagged first

prereg-v3 governs the views seeds 1–2 results too, so the tag must exist **before** the first run starts.
```bash
git ls-remote --tags https://github.com/aayxsh23/dngs-dropout | grep prereg-v3
```
If this prints nothing, tag `main`'s #32 merge commit (`283250d`) from your laptop:
```bash
git fetch origin && git tag -a prereg-v3 283250d -m "PLAN_V2 §5.2 addendum (#32)" && git push origin prereg-v3
```
Tell Aayush either way.

---

## 1. Check the machine (5 min)

```bash
cd /home/snu/dngs-dropout
pgrep -af "python.*(run_matrix|train_|render\.py|metrics\.py)"
tmux ls
git status --short --untracked-files=no
git describe --tags --exact-match --match 'v*-exp' HEAD
ls data/llff
ls outputs_views/llff_2v/A/fern/
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
df -h /home/snu
```

| Command | Expect | If not |
| --- | --- | --- |
| `pgrep` | nothing | Someone else's job is on the GPU: ask before starting. Never kill a job you didn't start |
| `tmux ls` | no `q0`/`q1`/`q2` | A queue is already running: **don't start a second one** (two copies of a shard train the same run and corrupt it). Go to §4 |
| `git status` | nothing | Stop: someone changed tracked files in the lab repo. Leave it, and message Aayush |
| `git describe` | `v1.0-exp` | Not critical: the queue runs every block from its own worktree, not from this folder |
| `ls data/llff` | the 8 scene folders | Stop: no data |
| `ls outputs_views/...` | `seed0` | Fine if missing; the seed-0 view runs may be elsewhere. Seeds 1–2 still go to `outputs_views/` |
| `nvidia-smi` | RTX A4000, little memory used | Somebody is using the GPU: ask |
| `df -h` | > 20 GB free | Today's blocks need about 7 GB (only depth-dose seed 0 keeps checkpoints) |

## 2. Get the lab kit (2 min)

```bash
mkdir -p ~/lab_kit && cd /home/snu/dngs-dropout && git fetch origin chore/lab-kit
for f in lab_queue.sh lab_check.sh README.txt; do git show FETCH_HEAD:scripts/lab/$f > ~/lab_kit/$f; done
chmod +x ~/lab_kit/*.sh && ls -l ~/lab_kit
```
Fallback if the fetch fails (no GitHub access from the lab): Aayush puts the same three files on Drive as `sparse3dgs/lab_kit/`.

What the kit does:
- `lab_queue.sh <day> <shard>`: one GPU slot. It runs that slot's share of every block for the day, one block after another.
  - It runs each block from a git worktree at that block's tag (`~/dngs-v1.0`, `~/dngs-v1.1`) and adds the worktree itself.
  - It passes absolute data and output paths, and logs to `output/q_<block>_<shard>.log`.
- `lab_check.sh`: what is running, finished runs per block against the expected counts, failed runs, tracebacks, disk.

## 3. Dry run, then start (5 min)

```bash
bash ~/lab_kit/lab_queue.sh tue --dry_run
```
Expected output:
```
added worktree /home/snu/dngs-v1.0 at v1.0-exp            (first time only)
experiments/matrix_views.yaml: shard 0/1, seeds [1, 2]: 240 of 240 runs, tag v1.0-exp, ...
SKIP comp: tag for /home/snu/dngs-v1.1 not available (yet)      <- normal until Aayush pushes v1.1-exp
SKIP depthdose: ...                                             <- same
SKIP noise2: ...                                                <- same
```
If `v1.1-exp` already exists, you see `matrix_comp.yaml ... 96 of 96 runs, tag v1.1-exp` (and 96 and 48) instead of the SKIP lines.

**Stop if any line reads `tag untagged`.** The runner would refuse those runs anyway.

Start the three slots:
```bash
for i in 0 1 2; do tmux new -d -s q$i "LAB_WAIT_HOURS=14 bash $HOME/lab_kit/lab_queue.sh tue $i"; done
tmux ls
```
- `LAB_WAIT_HOURS=14`: after views, each slot waits up to 14 h from now for `v1.1-exp`, then runs comp → depth-dose → noise2.
- If `tmux new` says `duplicate session`, a queue is already running. Don't force it.

## 4. Watch the first runs (15 min, before you leave)

```bash
tail -n 3 -f /home/snu/dngs-dropout/output/q_views_0.log          # Ctrl-C to stop watching
```

**What good looks like.** After about 3–6 minutes per slot you see `done llff_2v/A/fern/seed1  psnr 15.xx  NNN s`.

The first runs of the three slots are A at 2 views, seed 1, on fern, flower and fortress. Seed 0 gave 15.86, 14.28 and 15.52 (`results/tables/views/per_scene_2v.csv`). Within about ±1 dB is normal at 2 views.

**Stop and message Aayush** if any of these happen:
- `FAILED ...` on the first run of a slot;
- a PSNR below 10 or `nan`;
- a `Traceback` in `output/q_views_*.log` or in a run's `run.log`;
- `CUDA out of memory`.

Then:
```bash
bash ~/lab_kit/lab_check.sh
nvidia-smi        # 3 python processes, each roughly 0.6–1.5 GB
```

Optional: watch a slot live with `tmux attach -t q0`, and detach with **Ctrl-b, then d**. Never type Ctrl-c inside it: that kills the slot.

## 5. Before you leave (5 min)

1. **Power:** the PC must not sleep, suspend or reboot overnight. Check the system power settings ("Blank screen" can stay on; "Automatic suspend" must be off).
2. **Lock the screen** (Super+L or the menu). **Don't log out**: logging out can kill the tmux jobs.
3. Run `bash ~/lab_kit/lab_check.sh` and send Aayush the "finished runs" table, plus your start time.
4. RUNLOG text for Aayush to add:
   `17 Nov · lab/Disha · Queue tue started HH:MM, 3 slots, --shared_gpu: views seeds 1–2 (v1.0-exp, 240), then comp/depthdose/noise2 (v1.1-exp, 96/96/48) as soon as the tag exists. prereg-v3 tagged before the first run.`

## 6. What Aayush does today (remotely) so blocks 2–4 can run

1. On the laptop, run `bash scripts/gpu_smoke.sh` on the `feat/clamp-aware` head. Push it to #27.
2. pr-gate: opus reviewer → fixes, if any.
3. **Disha approves #27 on GitHub.** The phone is fine; you don't need the lab PC.
4. Merge with `--match-head-commit`. Tag the merge commit `v1.1-exp`, and `prereg-v2`. Push both tags.
5. Within 10 minutes the waiting slots add `~/dngs-v1.1` and start comp. You'll see `added worktree ... at v1.1-exp` in the tmux output, and `output/q_comp_*.log` appears.

If the tag isn't pushed within 14 hours of the start, blocks 2–4 are skipped. The next visit then reruns `lab_queue.sh tue` (views is already done and skipped).

**Expected finish if `v1.1-exp` lands before views ends:** about 9.5 h after the start.

## 7. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `HEAD must be exactly at a tag` in a log | The worktree isn't at a `v*-exp` tag | `git -C ~/dngs-v1.1 describe --tags`. If it's wrong, remove the worktree (`git -C /home/snu/dngs-dropout worktree remove ~/dngs-v1.1`) and rerun the queue |
| `SKIP ...: no data at ...` | Wrong data path | LLFF must be at `/home/snu/dngs-dropout/data/llff` |
| `FAILED llff_.../seedK (step 0 exited 1)` | Training crashed | Read `outputs_*/.../seedK/run.log` (last 30 lines). One isolated failure: leave it; the next queue run redoes it. The same failure on every run: stop the slots (below) and message Aayush |
| `CUDA out of memory` | Someone else is also on the GPU | Stop one slot (`touch ~/dngs-v1.0/STOP`; it ends after its current run), then remove the STOP file |
| A slot ended early | Its block finished, or it was skipped | `tmux ls` and the slot's last log lines. It's safe to start only the **missing** slot again: `tmux new -d -s q1 "LAB_WAIT_HOURS=14 bash $HOME/lab_kit/lab_queue.sh tue 1"` |
| The PC rebooted overnight | Power settings or updates | Next visit: §1, then §3. Finished runs are skipped; half-done runs are redone |
| You need to stop everything | — | `touch ~/dngs-v1.0/STOP ~/dngs-v1.1/STOP` (current runs finish, no new ones start). Remove the STOP files before the next start. Only as a last resort: `tmux kill-session -t q0` (and q1, q2), then rerun later |

**Never:**
- start a second queue while `q0`/`q1`/`q2` exist;
- run `git pull` or `checkout` in `/home/snu/dngs-dropout`, or edit files in `~/dngs-v1.x`;
- delete anything under `outputs_*`;
- start DTU or Blender today.

## 8. Next visit (Wed), for whoever goes

1. `bash ~/lab_kit/lab_check.sh`: expect views 240/240, comp 96, depth-dose 96, noise2 48 (or still running).
2. Error-budget renders for Disha (N1, BV4), once views seeds 1–2 are complete:
   ```bash
   cd /home/snu/dngs-dropout
   tar czf eval_views.tgz --exclude='*/depth' --exclude='*/depth_raw' outputs_views/llff_*v/*/*/seed*/eval
   md5sum eval_views.tgz > eval_views.tgz.md5
   ```
   Upload both files to Drive `sparse3dgs/outputs/` (expect roughly 0.5–1.5 GB).
3. If `v1.2-exp` and the DTU and Blender tars are ready:
   - copy them to `/home/snu/dngs-dropout`;
   - `md5sum` them and compare with RUNLOG;
   - `tar xf dtu_with_depth.tar -C data` and `tar xf blender_with_depth.tar -C data`;
   - then `bash ~/lab_kit/lab_queue.sh wed --dry_run` (DTU 225, Blender 120);
   - start with `for i in 0 1 2; do tmux new -d -s q$i "LAB_WAIT_HOURS=14 bash $HOME/lab_kit/lab_queue.sh wed $i"; done` (only once no `q*` session is left).
