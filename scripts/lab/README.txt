LAB KIT (copy this folder to the lab PC as ~/lab_kit; never paste commands from rendered Markdown)

ONE-TIME SETUP: none needed. lab_queue.sh adds ~/dngs-v1.0, ~/dngs-v1.1, ~/dngs-v1.2 (git worktrees at the tags) itself
  when a tag exists. Make sure the PC never sleeps; lock the screen, do not log out.

WEDNESDAY DATA (before starting the wed queue):
  copy dtu_with_depth.tar and blender_with_depth.tar to /home/snu/dngs-dropout/
  md5sum *_with_depth.tar        -> must match the md5s in RUNLOG
  tar xf dtu_with_depth.tar -C data ; tar xf blender_with_depth.tar -C data
  ls data/dtu | head ; ls data/nerf_synthetic

EVERY VISIT:
  1. bash ~/lab_kit/lab_check.sh                      what is running / done / failed
  2. bash ~/lab_kit/lab_queue.sh DAY --dry_run        DAY = tue, wed, thu or fri; check the run counts and tags
        views 240 (seeds 1-2), comp 96, depthdose 96, noise2 48, dtu 225, blender 120
  3. if nothing of ours is running (tmux ls shows no q0/q1/q2):
        for i in 0 1 2; do tmux new -d -s q$i "LAB_WAIT_HOURS=12 bash $HOME/lab_kit/lab_queue.sh DAY $i"; done
     (LAB_WAIT_HOURS=12: blocks whose tag is not pushed yet wait for it, up to 12 h, and start by themselves)
  4. wait ~10 min:  tail -3 /home/snu/dngs-dropout/output/q_*_0.log   (first runs must say "done ... psnr")
  5. RUNLOG line on the laptop: date, blocks started, tags, shared_gpu true

STOP A BLOCK GRACEFULLY:  touch ~/dngs-v1.1/STOP   (the run in progress finishes; remove the file afterwards)
WATCH A SHARD:            tmux attach -t q0   (detach: Ctrl-b then d)

FRIDAY BEFORE 12:00: last reruns; then copy outputs_* (no *.pth) + eval/ tarballs to Drive with md5sum.
