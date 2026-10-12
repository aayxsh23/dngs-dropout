#!/usr/bin/env bash
# Deterministic pre-merge checks: the same script runs locally and in CI (.github/workflows/ci.yml).
# No GPU needed. GPU behaviour is covered separately by scripts/gpu_smoke.sh on the laptop.
#
#   bash scripts/check.sh
#
# Exits non-zero if any check fails. In CI (CI=true) a check that cannot run is a failure,
# never a silent pass.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

FAILED=()
SKIPPED=()
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# git grep: exit 1 means "no match"; anything above 1 is an error and must not read as a pass.
# gg runs inside $(...) subshells, so the error is recorded in a file, not a variable.
gg() { local rc; git grep "$@"; rc=$?; if [ "$rc" -gt 1 ]; then echo "git grep error (exit $rc): $*" >&2; : > "$TMP/grep_err"; fi; return 0; }
grep_ok() { [ ! -e "$TMP/grep_err" ]; }
step() { printf '\n== %s\n' "$1"; }
pass() { echo "PASS"; }
fail() { echo "FAIL: $1"; FAILED+=("$1"); }
skip() {
  if [ "${CI:-}" = "true" ]; then fail "$1 (could not run in CI: $2)"; else echo "SKIPPED: $2"; SKIPPED+=("$1"); fi
}

# Tools: use installed ones (CI), otherwise fall back to uvx/uv (laptop).
ruff_() { if command -v ruff >/dev/null 2>&1; then ruff "$@"; else uvx -q ruff@0.6.9 "$@"; fi; }
vermin_() { if command -v vermin >/dev/null 2>&1; then vermin "$@"; else uvx -q --python 3.11 vermin@1.6.0 "$@"; fi; }
if python3 -c "import pandas, yaml" >/dev/null 2>&1; then PYA=(python3)
elif command -v uv >/dev/null 2>&1; then PYA=(uv run -q --no-project --python 3.11 --with "pandas>=2,<3" --with "pyyaml>=6" python)
else PYA=(); fi

# Training-pipeline code that must run on the laptop's Python 3.7 / PyTorch 1.12.
TRAIN_PATHS=(train_llff.py render.py metrics.py spiral.py gaussian_renderer utils scene arguments)
[ -f experiments/run_matrix.py ] && TRAIN_PATHS+=(experiments/run_matrix.py)
EVAL_PATHS=(render.py render_sh.py metrics.py metrics_dtu.py metrics_count.py spiral.py)
step "1. Syntax errors and undefined names (ruff E9,F63,F7,F82)"
if ruff_ check --no-cache --quiet --select E9,F63,F7,F82 --target-version py37 \
     --exclude submodules,dpt,lpipsPyTorch . ; then pass; else fail "ruff critical errors"; fi

step "2. Python 3.7 compatibility of the training pipeline and gate scripts (vermin)"
if out=$(vermin_ -q -t=3.7- --violations --no-tips --eval-annotations "${TRAIN_PATHS[@]}" scripts/pr_ready.py scripts/hooks/guard_bash.py 2>&1); then pass
else echo "$out"; fail "Python 3.7 incompatibility"; fi

step "3. Checkpoints load only through utils.general_utils.load_checkpoint()"
rm -f "$TMP/grep_err"
hits=$(gg -nE '^[^#]*\btorch\.load\(' -- '*.py' ':!submodules' ':!dpt' ':!lpipsPyTorch' ':!utils/general_utils.py')
if [ -z "$hits" ] && grep_ok; then pass; else echo "$hits"; fail "direct torch.load()"; fi

step "4. No dropout in evaluation paths; render functions default to no dropout"
rm -f "$TMP/grep_err"
hits=$(gg -nE 'drop_comp|drop_rate|Dropout' -- "${EVAL_PATHS[@]}")
bad_defs=$(grep -nE '^\s*def .*drop_comp' gaussian_renderer/__init__.py | grep -v 'drop_comp=None')
if [ -z "$hits$bad_defs" ] && grep_ok; then pass
else echo "$hits"; echo "$bad_defs"; fail "dropout reachable from evaluation"; fi

step "5. Repository hygiene: no data, outputs, checkpoints, archives or files > 5 MB"
if hyg=$(git ls-files -z | python3 -c '
import os, re, sys
bad = []
for f in sys.stdin.buffer.read().decode("utf-8", "replace").split("\0"):
    if not f:
        continue
    if re.match(r"(data|output|outputs)/", f) and not f.endswith(".gitkeep"):
        bad.append(f + "  (data/output folder)")
    elif re.search(r"\.(pth|pt|ckpt|ply|tar|zip|npy|npz|pfm)$|events\.out\.tfevents", f):
        bad.append(f + "  (checkpoint/data/archive type)")
    elif os.path.isfile(f) and os.path.getsize(f) > 5 * 1024 * 1024:
        bad.append("%s  (%d bytes > 5 MB)" % (f, os.path.getsize(f)))
print("\n".join(bad))
sys.exit(1 if bad else 0)'); then pass; else echo "$hyg"; fail "forbidden or oversized files tracked"; fi

step "6. No credentials in tracked files"
rm -f "$TMP/grep_err"
secrets=$(gg -nIE 'ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|gho_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|sk-ant-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----' -- . ':!scripts/check.sh')
if [ -z "$secrets" ] && grep_ok; then pass; else echo "$secrets" | cut -c1-120; fail "possible secret committed"; fi

step "7. Notebooks: valid JSON, outputs cleared"
nb_bad=""
while IFS= read -r nb; do
  [ -z "$nb" ] && continue
  python3 - "$nb" <<'EOF' || nb_bad+="$nb "
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
n = sum(len(c.get("outputs", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")
if n:
    print("%s: %d cell outputs committed (clear them: they can leak paths or tokens)" % (sys.argv[1], n))
    sys.exit(1)
EOF
done < <(git ls-files '*.ipynb')
if [ -z "$nb_bad" ]; then pass; else fail "notebook problems: $nb_bad"; fi

step "8. Analysis pipeline: aggregate.py on fixtures"
if [ ! -f experiments/aggregate.py ]; then echo "no experiments/aggregate.py yet"; pass
elif [ ${#PYA[@]} -eq 0 ]; then skip "aggregate fixtures" "pandas/pyyaml unavailable"
elif "${PYA[@]}" experiments/fixtures/make_fixtures.py "$TMP/fake" >/dev/null \
     && "${PYA[@]}" experiments/aggregate.py --out_root "$TMP/fake" --tables "$TMP/tables" >/dev/null 2>"$TMP/agg.err" \
     && test -s "$TMP/tables/main_3v.md" && test -s "$TMP/tables/runs.csv" \
     && head -1 "$TMP/tables/efficiency_3v.csv" | grep -q '^gpu,variant'; then pass
else cat "$TMP/agg.err" 2>/dev/null | tail -5; fail "aggregate.py fixture run"; fi

step "9. Experiment configs parse; runner dry run"
if [ ${#PYA[@]} -eq 0 ]; then skip "matrix configs" "pyyaml unavailable"
else
  cfg_ok=1
  for f in experiments/matrix*.yaml; do
    [ -e "$f" ] || continue
    "${PYA[@]}" - "$f" <<'EOF' || cfg_ok=0
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
missing = [k for k in ("scenes", "seeds", "n_views", "variants", "steps") if k not in cfg]
assert not missing, "%s: missing keys %s" % (sys.argv[1], missing)
for name, v in cfg["variants"].items():
    assert set(v) == {"depth_scale", "drop_rate", "depth_render"}, "%s variant %s: %s" % (sys.argv[1], name, v)
    assert v["depth_render"] in ("shared", "clean"), (sys.argv[1], name)
print("ok", sys.argv[1], len(cfg["scenes"]) * len(cfg["seeds"]) * len(cfg["n_views"]) * len(cfg["variants"]), "runs")
EOF
  done
  if [ -f experiments/run_matrix.py ] && [ -f experiments/matrix.yaml ]; then
    "${PYA[@]}" experiments/run_matrix.py --matrix experiments/matrix.yaml --data_root /nonexistent \
        --out_root "$TMP/out" --dry_run --allow_untagged > "$TMP/dry.txt" 2>&1 || { tail -5 "$TMP/dry.txt"; cfg_ok=0; }
  fi
  if [ $cfg_ok -eq 1 ]; then pass; else fail "experiment config / runner dry run"; fi
fi

step "10. Merge-gate scripts: hook self-test and syntax"
if python3 scripts/hooks/guard_bash.py --selftest >"$TMP/hook.txt" 2>&1 && python3 -m py_compile scripts/pr_ready.py; then pass
else cat "$TMP/hook.txt"; fail "merge-gate scripts"; fi

printf '\n== Summary\n'
[ ${#SKIPPED[@]} -gt 0 ] && echo "Skipped locally: ${SKIPPED[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED (${#FAILED[@]}): ${FAILED[*]}"
  exit 1
fi
echo "ALL CHECKS PASSED"
