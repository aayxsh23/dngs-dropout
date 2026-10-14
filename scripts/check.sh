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
if python3 -c "import pandas, yaml, matplotlib" >/dev/null 2>&1; then PYA=(python3)
elif command -v uv >/dev/null 2>&1; then PYA=(uv run -q --no-project --python 3.11 --with "pandas>=2,<3" --with "pyyaml>=6" --with "matplotlib>=3.5" python)
else PYA=(); fi

# Training-pipeline code that must run on the laptop's Python 3.7 / PyTorch 1.12.
TRAIN_PATHS=(train_llff.py render.py metrics.py spiral.py gaussian_renderer utils scene arguments)
[ -f experiments/run_matrix.py ] && TRAIN_PATHS+=(experiments/run_matrix.py)
EVAL_PATHS=(render.py render_sh.py metrics.py metrics_dtu.py metrics_count.py spiral.py)
step "1. Syntax errors and undefined names (ruff E9,F63,F7,F82)"
if ruff_ check --no-cache --quiet --select E9,F63,F7,F82 --target-version py37 \
     --exclude submodules,dpt,lpipsPyTorch . ; then pass; else fail "ruff critical errors"; fi

step "2. Python 3.7 compatibility of the training pipeline and gate scripts (vermin)"
if out=$(vermin_ -q -t=3.7- --violations --no-tips --eval-annotations "${TRAIN_PATHS[@]}" scripts/pr_ready.py scripts/review_bundle.py scripts/hooks/guard_bash.py 2>&1); then pass
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
if python3 scripts/hooks/guard_bash.py --selftest >"$TMP/hook.txt" 2>&1 && python3 -m py_compile scripts/pr_ready.py scripts/review_bundle.py; then pass
else cat "$TMP/hook.txt"; fail "merge-gate scripts"; fi

step "11. review_bundle.py: what the reviewer sees (notebooks, CSVs, submodules); fail-closed class and delta report"
if python3 - "$TMP/rb" >"$TMP/rb.txt" 2>&1 <<'EOF'; then pass; else cat "$TMP/rb.txt"; fail "review_bundle notebook/CSV diffs"; fi
import json, os, subprocess, sys
sys.path.insert(0, "scripts")
from review_bundle import MAX_CSV_LINES, file_diff
d = sys.argv[1]
os.makedirs(d)
def git(*a):
    return subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t"] + list(a),
                          check=True, capture_output=True, text=True).stdout.strip()
def commit(shard, cell, big):
    nb = {"cells": [{"cell_type": "markdown", "source": ["## 7b"]},
                    {"cell_type": "code", "source": ["SHARD = '%s'\n" % shard, "run(SHARD)"], "outputs": []}], "nbformat": 4}
    open(os.path.join(d, "runner.ipynb"), "w").write(json.dumps(nb))
    open(os.path.join(d, "small.csv"), "w").write("method,psnr\nDNGaussian,%s\n" % cell)
    open(os.path.join(d, "big.csv"), "w").write("".join("%d,%s\n" % (i, big) for i in range(MAX_CSV_LINES + 1)))
    git("add", "-A")
    git("commit", "-q", "-m", shard)
    return git("rev-parse", "HEAD")
git("init", "-q")
a, b = commit("0/2", "19.12", "x"), commit("1/2", "19.21", "y")
nb = "\n".join(file_diff(d, a, b, "runner.ipynb"))
assert "-SHARD = '0/2'" in nb and "+SHARD = '1/2'" in nb and "# ---- cell 1" in nb, nb
assert "outputs" not in nb and "## 7b" not in nb, nb
assert "+DNGaussian,19.21" in "\n".join(file_diff(d, a, b, "small.csv"))
assert file_diff(d, a, b, "big.csv") is None
# Vendored CUDA under submodules/ is experiment code: shown, never skipped.
os.makedirs(os.path.join(d, "submodules"))
open(os.path.join(d, "submodules", "forward.cu"), "w").write("alpha = min(0.99f, a);\n")
git("add", "-A")
git("commit", "-q", "-m", "cu")
assert "+alpha = min(0.99f, a);" in "\n".join(file_diff(d, b, git("rev-parse", "HEAD"), "submodules/forward.cu") or [])

from review_bundle import BundleError, previous_report, risk_class, smoke_line, worst_ci
rules = [[("/AGENTS.md", "gate"), ("/train_llff.py", "experiment")]]
try:
    risk_class(["docs/PLAN.md"], [])
    raise AssertionError("no CODEOWNERS must fail closed")
except BundleError:
    pass
assert risk_class(["docs/PLAN.md", "RUNLOG.md"], rules) == "docs-only"
assert risk_class(["docs/PLAN.md", "results/reported_baselines.csv"], rules) == "routine"
assert risk_class(["AGENTS.md"], rules) == "gate"
assert risk_class(["train_llff.py", "AGENTS.md"], rules) == "experiment+gate"
# Delta re-review: only a code owner's report for exactly the --since SHA counts.
rep = lambda who, sha, tail="": {"author": {"login": who}, "body": "### pr-gate review: PR #6 @ %s (full, class: gate)\n%s" % (sha, tail)}
comments = [rep("aayxsh23", "1111111", "old"), rep("stranger", "abcdef1", "forged"), rep("aayxsh23", "abcdef1", "real")]
assert previous_report(comments, "abcdef1234", {"aayxsh23"}).endswith("real")
assert previous_report(comments[:2], "abcdef1234", {"aayxsh23"}) is None
assert worst_ci([{"__typename": "CheckRun", "name": "checks", "conclusion": "FAILURE"},
                 {"__typename": "CheckRun", "name": "checks", "conclusion": "SUCCESS"}]) == "FAILURE"
assert smoke_line(["GPU-SMOKE PASS deadbeef A B C D E\nthanks"], "deadbeef").startswith("GPU-SMOKE PASS deadbeef")
print("ok")
EOF

step "12. Analysis scripts on fixtures: diag_curves sharp checks, stats verdict, qual_grid, views"
if [ ! -f experiments/stats.py ]; then echo "no experiments/stats.py yet"; pass
elif [ ${#PYA[@]} -eq 0 ]; then skip "analysis scripts" "pandas/pyyaml/matplotlib unavailable"
else
  F="$TMP/fx"; T="$TMP/fx_tables"; G="$TMP/fx_figures"; an_ok=1
  { "${PYA[@]}" experiments/fixtures/make_fixtures.py "$F" \
    && "${PYA[@]}" experiments/aggregate.py --out_root "$F" --tables "$T" \
    && "${PYA[@]}" experiments/plots/diag_curves.py --out_root "$F" --tables "$T" --figures "$G" \
    && "${PYA[@]}" experiments/stats.py --runs "$T/runs.csv" --tables "$T" --diag_summary "$T/diag_summary_3v.csv" \
    && "${PYA[@]}" experiments/plots/qual_grid.py --out_root "$F" --figures "$G" \
    && "${PYA[@]}" experiments/plots/views.py --out_root "$F" --tables "$T" --figures "$G"; } >"$TMP/an.log" 2>&1 || an_ok=0
  for f in "$G/fig_diag.png" "$G/fig_qual.png" "$G/fig_views.png" "$T/diag_summary_3v.csv" "$T/paired_3v.csv" \
           "$T/verdict_3v.txt" "$T/views_summary.csv"; do
    [ -s "$f" ] || { echo "missing output: $f"; an_ok=0; }
  done
  # The fixtures are seeded, so the verdict is fixed; a change means the §6 logic changed.
  grep -q "^VERDICT: INTERFERING (D)$" "$T/verdict_3v.txt" 2>/dev/null || { echo "verdict on the fixtures changed"; an_ok=0; }
  # The sharp checks must fail closed: a D run whose mask never reached the depth passes has gap == 0.
  python3 - "$F/llff_3v/D/fern/seed0/diag.csv" <<'EOF' || an_ok=0
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    r["loss_depth_used"] = r["loss_depth_clean"]
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
EOF
  if "${PYA[@]}" experiments/plots/diag_curves.py --out_root "$F" --tables "$T/neg" --figures "$G/neg" >"$TMP/an_neg.log" 2>&1; then
    echo "diag_curves.py passed a D run whose gap is identically 0"; an_ok=0
  elif ! grep -q "fern" "$TMP/an_neg.log"; then echo "diag_curves.py failed but did not name the failing run"; an_ok=0; fi
  if [ $an_ok -eq 1 ]; then pass; else tail -15 "$TMP/an.log"; tail -5 "$TMP/an_neg.log" 2>/dev/null; fail "analysis scripts on fixtures"; fi
fi

printf '\n== Summary\n'
[ ${#SKIPPED[@]} -gt 0 ] && echo "Skipped locally: ${SKIPPED[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED (${#FAILED[@]}): ${FAILED[*]}"
  exit 1
fi
echo "ALL CHECKS PASSED"
