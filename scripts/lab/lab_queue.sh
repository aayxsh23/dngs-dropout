#!/bin/bash
# Lab run queue (EXECUTION_PLAN A9, ACTION_PLAN_16-17Nov §5). One shard of every block for <day>, one after another,
# so the GPU never idles between visits. Blocks whose tag worktree or matrix file is missing are skipped (printed).
#
#   bash lab_queue.sh <tue|wed|thu|fri> --dry_run          # check counts first, every visit
#   for i in 0 1 2; do tmux new -d -s q$i "LAB_WAIT_HOURS=12 bash $HOME/lab_kit/lab_queue.sh <day> $i"; done
#
# LAB_WAIT_HOURS=h: when a block's tag (v1.1-exp, v1.2-exp) is not checked out yet, poll GitHub every 10 min for up to
# h hours; as soon as the tag exists, add its worktree (~/dngs-v1.x) and run the block. 0 (default) = skip at once.
# So the queue can be left running before a tag is pushed: nobody has to be at the lab when it lands.
# LAB_BLOCKS="comp depthdose noise2": run only these blocks (overrides the day's order).
# LAB_AFTER=matrix_views.yaml: first wait (polling every 5 min) until no run_matrix.py process for that matrix is left,
# e.g. views shards started by hand, so the two never share the GPU at once.
# LAB_AFTER_TMUX="c0 c1 c2": first wait (every 5 min) until none of these tmux sessions exists any more, e.g. an earlier
# day's queue that is still running, so the next blocks start right after it without anyone at the lab.
#
# The runner skips runs whose metrics.json exists, so re-running a day is safe; a STOP file in a worktree root lets
# the run in progress finish and starts no new one in that block.
set -u
DAY=${1:?day: tue, wed, thu or fri}; ARG=${2:?shard 0, 1, 2 or --dry_run}
REPO=${LAB_REPO:-/home/snu/dngs-dropout}                      # data/ and every outputs_* live here (worktrees have no data/)
PY=${LAB_PY:-/home/snu/miniconda3/envs/dngaussian/bin/python}
LOG=$REPO/output; mkdir -p "$LOG"
WT=${LAB_WT:-$HOME}                                 # where the per-tag worktrees are (~/dngs-v1.0, ~/dngs-v1.1, ...)
WAIT_H=${LAB_WAIT_HOURS:-0}
T0=$(date +%s)

ensure_wt() {   # $1 = worktree path ending in dngs-v1.x; returns 0 once it exists
  local wt=$1 tag; tag="$(basename "$1" | sed 's/^dngs-//')-exp"
  while true; do
    [ -d "$wt" ] && return 0
    (   flock 9   # one shard at a time adds the worktree
        [ -d "$wt" ] && exit 0
        git -C "$REPO" fetch -q --tags origin 2>/dev/null
        if git -C "$REPO" rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
          git -C "$REPO" worktree add -q "$wt" "$tag" && echo "$(date '+%a %H:%M') added worktree $wt at $tag"
        fi
    ) 9>"$LOG/.worktree.lock"
    [ -d "$wt" ] && return 0
    [ "$ARG" = "--dry_run" ] && return 1
    [ $(( $(date +%s) - T0 )) -ge $(( WAIT_H * 3600 )) ] && return 1
    echo "$(date '+%a %H:%M') waiting for tag $tag (up to ${WAIT_H} h from the queue start)"
    sleep 600
  done
}

# name | worktree (tag) | matrix | data root | out root | extra runner flags
BLOCKS="
views|$WT/dngs-v1.0|experiments/matrix_views.yaml|$REPO/data/llff|$REPO/outputs_views|--seeds 1 2
comp|$WT/dngs-v1.1|experiments/matrix_comp.yaml|$REPO/data/llff|$REPO/outputs_comp|
depthdose|$WT/dngs-v1.1|experiments/matrix_depthdose.yaml|$REPO/data/llff|$REPO/outputs_depthdose|
noise2|$WT/dngs-v1.1|experiments/matrix_noise2.yaml|$REPO/data/llff|$REPO/outputs_noise2|
dtu|$WT/dngs-v1.2|experiments/matrix_dtu.yaml|$REPO/data/dtu|$REPO/outputs_dtu|
blender|$WT/dngs-v1.2|experiments/matrix_blender.yaml|$REPO/data/nerf_synthetic|$REPO/outputs_blender|
"
case $DAY in
  tue) ORDER="views comp depthdose noise2" ;;
  wed) ORDER="dtu blender views comp depthdose noise2" ;;
  thu|fri) ORDER="dtu blender views comp depthdose noise2" ;;
  *) echo "day must be tue, wed, thu or fri"; exit 2 ;;
esac
[ -n "${LAB_BLOCKS:-}" ] && ORDER=$LAB_BLOCKS

TMUXBIN=$(command -v tmux || echo "$HOME/miniconda3/bin/tmux")
if [ -n "${LAB_AFTER_TMUX:-}" ] && [ "$ARG" != "--dry_run" ]; then
  while true; do
    alive=""; for t in $LAB_AFTER_TMUX; do "$TMUXBIN" has-session -t "$t" 2>/dev/null && alive="$alive $t"; done
    [ -z "$alive" ] && break
    echo "$(date '+%a %H:%M') waiting for tmux session(s)$alive to finish"
    sleep 300
  done
fi
if [ -n "${LAB_AFTER:-}" ] && [ "$ARG" != "--dry_run" ]; then
  while pgrep -f "run_matrix\.py.*${LAB_AFTER}" >/dev/null; do
    echo "$(date '+%a %H:%M') waiting for the running ${LAB_AFTER} jobs to finish"
    sleep 300
  done
fi

for name in $ORDER; do
  line=$(printf '%s\n' "$BLOCKS" | grep "^$name|")
  IFS='|' read -r _ wt matrix data out extra <<< "$line"
  if [ "$name" = noise2 ] && [ ! -f "$wt/$matrix" ] && [ -f "$WT/dngs-v1.2/$matrix" ]; then wt=$WT/dngs-v1.2; fi
  if ! ensure_wt "$wt"; then echo "SKIP $name: tag for $wt not available (yet)"; continue; fi
  if [ "$name" = noise2 ] && [ ! -f "$wt/$matrix" ] && [ -f "$WT/dngs-v1.2/$matrix" ]; then wt=$WT/dngs-v1.2; fi
  if [ ! -f "$wt/$matrix" ]; then echo "SKIP $name: no $wt/$matrix"; continue; fi
  if [ ! -d "$data" ]; then echo "SKIP $name: no data at $data"; continue; fi
  if [ "$ARG" = "--dry_run" ]; then
    (cd "$wt" && $PY experiments/run_matrix.py --matrix "$wt/$matrix" --data_root "$data" --out_root "$out" $extra \
        --dry_run > "$LOG/dry_$name.txt" 2>&1); head -1 "$LOG/dry_$name.txt" | sed "s#^$wt/##"
    continue
  fi
  echo "$(date '+%a %H:%M') start $name shard $ARG/3 ($wt, $(git -C "$wt" describe --tags 2>/dev/null))"
  (cd "$wt" && $PY experiments/run_matrix.py --matrix "$wt/$matrix" --data_root "$data" --out_root "$out" \
      --shard "$ARG/3" --shared_gpu $extra) >> "$LOG/q_${name}_$ARG.log" 2>&1
  echo "$(date '+%a %H:%M') end $name shard $ARG/3: $(tail -1 "$LOG/q_${name}_$ARG.log")"
done
echo "$(date '+%a %H:%M') queue $DAY shard $ARG finished"
