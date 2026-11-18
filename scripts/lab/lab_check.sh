#!/bin/bash
# What finished, what failed, what is running (EXECUTION_PLAN A9: twice a day).   bash lab_check.sh
REPO=${LAB_REPO:-/home/snu/dngs-dropout}
echo "== running (python processes only; the tmux server's argv is not a job)"
ps -eo pid=,comm=,args= | awk '$2 ~ /^python/ && /(run_matrix|train_|render\.py|metrics\.py|eval_dtu)/' | grep . || echo "(nothing)"
TMUXBIN=$(command -v tmux || echo "$HOME/miniconda3/bin/tmux")   # the lab has tmux only in conda base
echo "== tmux"; "$TMUXBIN" ls 2>/dev/null || echo "(no sessions)"
echo "== finished runs (metrics.json) per block"
printf '%-18s %8s %8s\n' block done expected
for b in "outputs_views:llff_*v/*/*/seed[12]:240" "outputs_comp:llff_3v/*/*/seed*:96" \
         "outputs_depthdose:llff_3v/*/*/seed*:96" "outputs_noise2:llff_3v/*/*/seed*:48" \
         "outputs_dtu:dtu_3v/*/*/seed*:225" "outputs_blender:blender_8v/*/*/seed*:120"; do
  IFS=: read -r root pat exp <<< "$b"
  n=$(ls $REPO/$root/$pat/metrics.json 2>/dev/null | wc -l)
  printf '%-18s %8s %8s\n' "$root" "$n" "$exp"
done
echo "== failed runs (FAILED markers; the next queue run redoes them)"
ls $REPO/outputs_*/*_*v/*/*/seed*/FAILED 2>/dev/null | sed "s#$REPO/##;s#/FAILED##" || true
echo "== tracebacks in queue logs"
grep -l Traceback $REPO/output/q_*.log 2>/dev/null || echo "(none)"
echo "== disk"; df -h $REPO | tail -1
