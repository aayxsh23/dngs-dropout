# Lab runbook, Wed 11 Nov: backup, probes, geometry

**Who:** Disha = GPU (probes). Aayush = CPU and disk (backup, geometry).
**Where:** lab PC, repo at `/home/snu/dngs-dropout`, at tag `v1.0-exp` (commit `1040559`).
**Why at the lab:** the checkpoints and the raw depth maps exist only on the lab disk.

| Step | Who | Terminal | Takes | Uses |
| --- | --- | --- | --- | --- |
| 0. Check the machine | Disha (Aayush watches) | T1 | 2 min | – |
| 1. Backup tar | Aayush | T2 | maybe 10–30 min | disk |
| 2. Probes, 40 runs | Disha | T1 | 40 × fern time | GPU |
| 3. Geometry, 344 runs | Aayush | T3 | unknown, CPU | CPU |
| 4. Before you leave | both | any | 5 min | – |

Steps 1, 2 and 3 run at the same time once step 0 passes. Copy commands from this file and paste them into the terminal
as plain text, never through anything that formats Markdown: last time a script lost its `$` signs, spaces and
underscores that way (`/home/snu/run_all.sh` line 9).

---

## Step 0. Check the machine (Disha, T1, both watch)

```bash
cd ~/dngs-dropout
pgrep -af "train_llff|render.py|metrics.py|run_matrix"
git rev-parse --short HEAD
git status --short --untracked-files=no
ls outputs/llff_3v/*/*/seed0/chkpnt_latest.pth | wc -l
ls -d outputs*/llff_*/*/*/seed*/eval/ours_6000/depth_raw | wc -l
du -sh outputs*
df -h ~
```

| Command | Expect | If not |
| --- | --- | --- |
| `pgrep` | no output | a job is still running: wait or ask before killing it |
| `rev-parse` | `1040559` | `git checkout v1.0-exp` |
| `status` | no output | stop: someone changed tracked files |
| checkpoints | `40` | probes only for the runs that have one; tell Claude |
| `depth_raw` | `344` | geometry covers only the runs that have it; tell Claude |
| `du` + `df` | free space > total of `du` | backup to USB directly instead of `~` |

Write down the `du` numbers. Expected total roughly 2–4 GB (estimate).

---

## Disha: GPU column

### 2a. Get probes.py at the pinned commit (T1)

```bash
cd ~/dngs-dropout
git fetch origin
git show 9f8d55f:experiments/probes.py > experiments/probes.py
md5sum experiments/probes.py
```
Write the md5 down. The record has to show the probes ran from `probes.py` at `9f8d55f`.
The file stays untracked, so the tree stays clean at `v1.0-exp`.

### 2b. One probe on fern, timed (the L6 gate)

```bash
PY=~/miniconda3/envs/dngaussian/bin/python
time $PY experiments/probes.py -s data/llff/fern --model_path outputs/llff_3v/B/fern/seed0 --near 10
head -3 outputs/llff_3v/B/fern/seed0/probes/robustness.csv
cat outputs/llff_3v/B/fern/seed0/results_eval.json
```
- `--near 10` is required: the script defaults to 0, and then its renders don't match `render.py`.
- **Gate:** line 2 of the CSV is `True,0.0,<psnr>,...`. That `<psnr>` must match `PSNR` in the JSON to within a few
  hundredths of a dB. **If it doesn't, stop the probes**, send both outputs to Claude, and don't debug at the lab.
- **Time:** the 40 runs take about 40 × the `real` time printed by `time`. Write it down.

### 2c. Write the loop, check it, start it

Paste this whole block once (from `cat` to `EOF2`):
```bash
cat > ~/run_probes.sh <<'EOF2'
cd ~/dngs-dropout
for r in outputs/llff_3v/*/*/seed0; do
  [ -f "$r/probes/depth_bias.json" ] && continue   # written last, so its presence means the run is done
  s=$(basename "$(dirname "$r")")
  echo "== $r"
  ~/miniconda3/envs/dngaussian/bin/python experiments/probes.py -s data/llff/$s --model_path "$r" --near 10 --quiet || echo "FAILED $r"
done
EOF2
```
Check it before running:
```bash
cat ~/run_probes.sh
```
You must see `$r`, `$s`, `$(dirname "$r")`, `depth_bias.json` and `--model_path`, with every `$` and `_` intact.
If anything is missing, delete the file (`rm ~/run_probes.sh`) and paste again. Then:
```bash
nohup bash ~/run_probes.sh > output/probes.log 2>&1 &
tail -f output/probes.log
```
Ctrl+C stops watching; the job keeps running (also after logout). The loop skips finished runs, so if it is
interrupted, just run the `nohup` line again. Fern B is already done from 2b and is skipped.

### 2d. When it finishes
```bash
grep -c FAILED output/probes.log                                  # expect 0
ls outputs/llff_3v/*/*/seed0/probes/depth_bias.json | wc -l       # expect 40
pgrep -af probes.py                                               # expect nothing
```

---

## Aayush: CPU and disk column

### 1. Backup everything except checkpoints (T2, start first)

```bash
cd ~/dngs-dropout
tar --exclude='*.pth' --exclude='*.ply' -czf ~/results_full_nockpt.tgz \
  outputs outputs_pass outputs_dose outputs_noise outputs_views output/lab_matrix*.log
md5sum ~/results_full_nockpt.tgz
ls -lh ~/results_full_nockpt.tgz
```
- Write the md5 down.
- The files are mostly compressed already, so the tar is slow and the archive is about the size of `du`.
- "file changed as we read it" is harmless if the probes are running: probe outputs go into the second archive.
- **Keep the checkpoints** on the lab disk until the probes have finished.
- Copy the archive to Drive (`sparse3dgs`) or a USB stick before you leave.

### 3. Geometry on all 344 runs (T3, while the tar and probes run)

```bash
cd ~/dngs-dropout
PY=~/miniconda3/envs/dngaussian/bin/python
$PY experiments/geometry.py --run outputs/llff_3v/B/fern/seed0 --self_test
```
**Gate:** it prints JSON and must not say `self-test FAILED` (`median_rel_err` about 0, `mvc5` about 1). If it fails,
skip geometry and tell Claude. If it passes:
```bash
for b in outputs outputs_pass outputs_dose outputs_noise outputs_views; do
  $PY experiments/geometry.py --out_root $b --tables results/tables/geometry/$b
done
ls results/tables/geometry/*/
```
Expect `geometry_3v.csv` in outputs, outputs_pass, outputs_dose and outputs_noise, and `geometry_2v.csv`,
`geometry_6v.csv`, `geometry_9v.csv` in outputs_views. Any "WARNING no valid pixels" line: note which run.

---

## Step 4. Before you leave (both)

Only once the probes are done (2d shows 0 and 40). If they are still running and you must leave, see below.
```bash
cd ~/dngs-dropout
tar -czf ~/probes_geometry.tgz outputs/llff_3v/*/*/seed0/probes results/tables/geometry output/probes.log experiments/probes.py
md5sum ~/probes_geometry.tgz ~/results_full_nockpt.tgz
```
- Copy **both** archives to Drive or USB. Check both md5s at the destination (`md5sum <file>`), and they must
  match the lab's.
- Upload `probes_geometry.tgz` to the Claude thread. It is small (CSVs and JSON).
- Leave the checkpoints on the lab disk.

**If the probes are too slow to finish today:** leave the `nohup` job running. Make the archive without probes
(`tar -czf ~/geometry.tgz results/tables/geometry`), take both archives, and collect the probes next visit. The
backup and geometry matter most; the probes are the planned item to cut.

---

## Write down (for the RUNLOG and the PR)

| Item | Value |
| --- | --- |
| HEAD | 1040559 |
| `probes.py` md5 (from `9f8d55f`) | |
| fern probe time (`real`) | |
| fern f = 0 PSNR vs eval PSNR | |
| `results_full_nockpt.tgz` size and md5 | |
| `probes_geometry.tgz` md5 | |
| probes: FAILED count / depth_bias.json count | 0 / 40 |
| geometry self-test: median_rel_err, mvc5 | |

## Afterwards (Claude, from the uploaded archive)
- Add the geometry CSVs and probe results to PR #29.
- Re-run `stats.py --geometry results/tables/geometry/outputs/geometry_3v.csv` and check the opacity-cap condition 3
  from `depth_bias.json`; update the verdict and RESULTS.md.
- Open the PR that puts `probes.py` (at `9f8d55f`) on `main`: routine, CI plus a sonnet review.
