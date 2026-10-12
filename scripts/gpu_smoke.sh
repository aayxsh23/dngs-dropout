#!/usr/bin/env bash
# GPU smoke test for experiment-affecting changes. Runs on the laptop (or Colab); CI has no GPU.
# Trains fern for 1100 iterations per variant (covers both depth passes), renders the eval set,
# computes metrics and sanity-checks the outputs. About 1.5 min per variant on the laptop.
#
#   bash scripts/gpu_smoke.sh            # all variants the code supports (A-E once --drop_rate exists, else A,B)
#   bash scripts/gpu_smoke.sh B D        # only these
#
# Last line on success: "GPU-SMOKE PASS <full head sha> variants=... psnr=...". Post that line as a PR
# comment; scripts/pr_ready.py requires it, for the PR's head commit, before merging changes to experiment paths.
# A dirty working tree prints "<sha>-dirty", which the gate does not accept.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${DNGS_PY:-$HOME/miniconda3/envs/dngaussian/bin/python}
[ -x "$PY" ] || PY=python
SCENE=${SMOKE_SCENE:-data/llff/fern}

# Only real python processes count (not shells whose command text mentions these scripts).
if pgrep -af "(train_llff|render|metrics)\.py" | awk '$2 ~ /python/ {found=1} END {exit !found}'; then
  echo "GPU-SMOKE FAIL: another training/render job is running (pgrep -af train_llff)"; exit 1
fi
[ -d "$SCENE/images" ] || { echo "GPU-SMOKE FAIL: $SCENE not found (extract llff_with_depth.tar into data/)"; exit 1; }

SHA=$(git rev-parse HEAD)
[ -z "$(git status --porcelain)" ] || SHA="$SHA-dirty"   # untracked non-ignored files count as dirty too
# Use only the flags this commit supports (they arrive in stages on feat/dropout).
HAS_DEPTH=$(grep -q "depth_scale" arguments/__init__.py && echo 1 || echo 0)
HAS_DROP=$(grep -q "drop_rate" arguments/__init__.py && echo 1 || echo 0)
SEED_FLAG=$(grep -q '"--seed"' train_llff.py && echo "--seed 0" || echo "")
if [ "$HAS_DROP" = 1 ]; then ALL=(A B C D E); elif [ "$HAS_DEPTH" = 1 ]; then ALL=(A B); else ALL=(B); fi
VARIANTS=("${@:-${ALL[@]}}")
flags_for() {
  case $1 in
    A) echo "--depth_scale 0" ;;
    B) [ "$HAS_DEPTH" = 1 ] && echo "--depth_scale 1" || echo "" ;;
    C) echo "--depth_scale 0 --drop_rate 0.2 --depth_render shared" ;;
    D) echo "--depth_scale 1 --drop_rate 0.2 --depth_render shared" ;;
    E) echo "--depth_scale 1 --drop_rate 0.2 --depth_render clean" ;;
  esac
}
COMMON="-r 8 --eval --n_sparse 3 --rand_pcd --iterations 1100 --test_iterations 1100 --lambda_dssim 0.2
  --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01
  --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500
  --split_opacity_thresh 0.1 --error_tolerance 0.00025 --scaling_lr 0.003 --shape_pena 0.002 --opa_pena 0.001
  --near 10 $SEED_FLAG"

ROOT=output/smoke/${SHA:0:7}
FAILS=()
SUMMARY=()
for v in "${VARIANTS[@]}"; do
  out=$ROOT/$v; rm -rf "$out"; mkdir -p "$out"
  if [ "$v" != B ] && [ "$HAS_DEPTH" = 0 ]; then echo "variant $v needs --depth_scale, which this commit lacks"; FAILS+=("$v:unsupported"); continue; fi
  echo "== variant $v: $(flags_for "$v") $SEED_FLAG"
  # shellcheck disable=SC2046
  if ! "$PY" train_llff.py -s "$SCENE" --model_path "$out" $COMMON $(flags_for "$v") >"$out/smoke.log" 2>&1 \
     || ! "$PY" render.py -s "$SCENE" --model_path "$out" -r 8 --near 10 --skip_train --skip_test >>"$out/smoke.log" 2>&1 \
     || ! "$PY" metrics.py --model_paths "$out" >>"$out/smoke.log" 2>&1; then
    tail -15 "$out/smoke.log"; FAILS+=("$v:crashed"); continue
  fi
  res=$("$PY" - "$out" <<'EOF'
import csv, json, math, os, sys
out = sys.argv[1]
m = next(iter(json.load(open(os.path.join(out, "results_eval.json"))).values()))
problems = []
if not (12.0 <= m["PSNR"] <= 26.0) or math.isnan(m["PSNR"]):
    problems.append("PSNR %.2f outside the 12-26 sanity range" % m["PSNR"])
diag = os.path.join(out, "diag.csv")
if os.path.exists(diag):
    rows = list(csv.DictReader(open(diag)))
    if not rows:
        problems.append("diag.csv is empty")
    for r in rows:
        if any(v in ("nan", "inf", "") for v in r.values()):
            problems.append("diag.csv has nan/inf/empty at iter %s" % r.get("iter")); break
print("%.2f" % m["PSNR"] + ("" if not problems else " PROBLEM: " + "; ".join(problems)))
EOF
)
  echo "   PSNR $res"
  case $res in *PROBLEM*) FAILS+=("$v:${res#* PROBLEM: }") ;; esac
  SUMMARY+=("$v=${res%% *}")
done

# Variant-specific diagnostic checks (only once diag.csv exists, i.e. after the diagnostics land)
for v in A B E; do
  d=$ROOT/$v/diag.csv
  [ -f "$d" ] || continue
  "$PY" - "$d" "$v" <<'EOF' || FAILS+=("$v:diag-used-vs-clean")
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
bad = [r["iter"] for r in rows if abs(float(r["loss_depth_used"]) - float(r["loss_depth_clean"])) > 1e-6]
if bad:
    print("variant %s: loss_depth_used != loss_depth_clean at iters %s (must be equal by construction)" % (sys.argv[2], bad[:5]))
    sys.exit(1)
EOF
done

if [ ${#FAILS[@]} -gt 0 ]; then
  echo "GPU-SMOKE FAIL $SHA ${FAILS[*]}"; exit 1
fi
echo "GPU-SMOKE PASS $SHA variants=$(IFS=,; echo "${VARIANTS[*]}") psnr=$(IFS=,; echo "${SUMMARY[*]}")"
