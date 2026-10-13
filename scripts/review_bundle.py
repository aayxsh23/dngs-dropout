#!/usr/bin/env python3
"""Build the single input file a pr-reviewer needs, so the reviewer reads one file instead of
running a dozen gh/git commands, and never re-runs what CI already proved.

    python3 scripts/review_bundle.py 12 --out /tmp/review_12.md                    # full review
    python3 scripts/review_bundle.py 12 --since 3466f19 --out /tmp/review_12.md    # re-review: only what changed

Contents: PR metadata and risk class (from CODEOWNERS), the goal (PR body), CI / agent-review /
GPU-smoke evidence on the head, the previous pr-gate report (re-reviews), the changed-file list,
and the diff, capped per file and in total so the bundle stays small. Truncated files are listed;
the reviewer opens them with Read only if needed. Notebooks appear as a diff of their code-cell
sources; CSVs appear when they have at most MAX_CSV_LINES lines. Python 3.7 compatible.
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr_ready import CI_CHECK, REVIEW_STATUS, SMOKE_MARK, classify, codeowners_rules, gh_json, origin_repo, run  # noqa: E402

# Files whose diff is noise for review (generated, vendored, binary): listed, not shown.
SKIP_DIFF = re.compile(r"(^submodules/|\.lock$|lock\.yml$|^pip_freeze|\.svg$|\.png$)")
# Notebooks are shown as a diff of their code-cell sources (outputs are cleared, check.sh step 7).
# CSVs are shown when small (hand-edited tables such as reported_baselines.csv), skipped when large.
MAX_CSV_LINES = 200


def show(root, rev, path):
    """Content of path at rev, or "" if it doesn't exist there (added or deleted file)."""
    out = subprocess.run(["git", "show", "%s:%s" % (rev, path)], capture_output=True, text=True, cwd=root, timeout=20)
    return out.stdout if out.returncode == 0 else ""


def notebook_sources(text):
    """Code-cell sources of a notebook as lines, each cell under a header with its index."""
    if not text.strip():
        return []
    try:
        cells = json.loads(text).get("cells", [])
    except ValueError:
        return ["(not valid notebook JSON)"]
    lines = []
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "")
        lines.append("# ---- cell %d" % i)
        lines.extend(("".join(src) if isinstance(src, list) else src).splitlines())
    return lines


def file_diff(root, a, b, path):
    """Diff lines of one file between revs a and b, or None to list it as skipped."""
    if SKIP_DIFF.search(path):
        return None
    if path.endswith(".ipynb"):
        return list(difflib.unified_diff(notebook_sources(show(root, a, path)), notebook_sources(show(root, b, path)),
                                         "a/%s (code cells)" % path, "b/%s (code cells)" % path, n=3, lineterm=""))
    if path.endswith(".csv") and max(len(show(root, r, path).splitlines()) for r in (a, b)) > MAX_CSV_LINES:
        return None
    return run(["git", "diff", "-U3", a, b, "--", path], root).splitlines()


def section(title, body):
    return "## %s\n%s\n" % (title, body.rstrip() + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pr")
    ap.add_argument("--since", default=None, help="last reviewed SHA: bundle only the delta (re-review mode)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-lines", type=int, default=1200, help="total diff lines in the bundle")
    ap.add_argument("--max-file-lines", type=int, default=250, help="diff lines per file")
    a = ap.parse_args()
    root = run(["git", "rev-parse", "--show-toplevel"], os.getcwd()).strip()
    repo = origin_repo(root)
    d = gh_json(["pr", "view", a.pr, "--repo", repo, "--json",
                 "number,title,body,author,baseRefName,headRefOid,files,statusCheckRollup,comments,changedFiles"], root)
    head, base = d["headRefOid"], d["baseRefName"]
    rules, _ = codeowners_rules(root, base, head)
    files = [f["path"] for f in d["files"]]
    hit = classify(files, rules)
    cls = "+".join(sorted(hit)) if hit else "routine"

    ci, status = "missing", "missing"
    for c in d["statusCheckRollup"] or []:
        if c.get("__typename") == "CheckRun" and c.get("name") == CI_CHECK:
            ci = (c.get("conclusion") or c.get("status") or "?").upper()
        elif c.get("__typename") == "StatusContext" and c.get("context") == REVIEW_STATUS:
            status = (c.get("state") or "?").upper()
    bodies = [c.get("body") or "" for c in d["comments"] or []]
    smoke = next((b.strip().splitlines()[-1] for b in bodies if (SMOKE_MARK + " " + head) in b), "none for head")
    prev = [b for b in bodies if b.lstrip().startswith("### pr-gate review")]

    mode = "DELTA re-review since %s" % a.since[:7] if a.since else "FULL review"
    out = ["# Review bundle: %s PR #%d @ %s (%s)\n" % (repo, d["number"], head[:7], mode)]
    out.append(section("Facts", "\n".join([
        "- Title: %s" % d["title"],
        "- Author: %s; base: %s; head: %s" % (d["author"]["login"], base, head),
        "- Risk class (CODEOWNERS): **%s**" % cls,
        "- Changed files: %d" % d["changedFiles"],
    ])))
    out.append(section("Evidence already established (do not re-run)", "\n".join([
        "- CI `%s` on head: %s (runs scripts/check.sh: ruff, vermin py3.7, invariant greps, hygiene, secrets,"
        " notebooks, aggregate fixtures, configs, hook self-test)" % (CI_CHECK, ci),
        "- `%s` status on head: %s" % (REVIEW_STATUS, status),
        "- GPU smoke: %s" % smoke,
    ])))
    goal = (d["body"] or "").strip()
    out.append(section("Goal / acceptance criteria (PR body, truncated)", "\n".join(goal.splitlines()[:40]) or "(empty: ask the author)"))
    if a.since and prev:
        out.append(section("Previous pr-gate report (verify its blocking findings)", "\n".join(prev[-1].splitlines()[:60])))

    if a.since:
        subprocess.run(["git", "fetch", "-q", "origin", head], cwd=root, capture_output=True, timeout=60)
        rng = "%s..%s" % (a.since, head)
        old = a.since
        names = run(["git", "diff", "--name-only", rng], root).splitlines()
    else:
        subprocess.run(["git", "fetch", "-q", "origin", base, head], cwd=root, capture_output=True, timeout=60)
        rng = "origin/%s...%s" % (base, head)
        old = run(["git", "merge-base", "origin/" + base, head], root).strip()
        names = files
    stats = run(["git", "diff", "--numstat", rng], root).splitlines()
    out.append(section("Changed files (%s)" % rng, "\n".join("- `%s`  +%s -%s" % (s.split("\t")[2], s.split("\t")[0], s.split("\t")[1])
                                                           for s in stats if s.count("\t") >= 2)))

    budget, shown, truncated, skipped = a.max_lines, [], [], []
    for f in names:
        diff = file_diff(root, old, head, f)
        if diff is None:
            skipped.append(f)
            continue
        if not diff:
            continue
        cap = min(a.max_file_lines, budget)
        if cap <= 0:
            truncated.append(f + " (not shown: total budget used)")
            continue
        shown.append("### %s\n```diff\n%s\n```" % (f, "\n".join(diff[:cap])))
        if len(diff) > cap:
            truncated.append("%s (%d of %d diff lines shown)" % (f, cap, len(diff)))
        budget -= min(len(diff), cap)
    out.append(section("Diff", "\n\n".join(shown) or "(no reviewable diff)"))
    if truncated or skipped:
        out.append(section("Not fully shown (Read these only if your checklist needs them)",
                           "\n".join(["- " + t for t in truncated] + ["- %s (generated/vendored/binary or a CSV over %d lines: skipped)" % (s, MAX_CSV_LINES) for s in skipped])))
    text = "\n".join(out)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
        print("%s: %d lines, ~%d tokens, class %s, %d truncated, %d skipped"
              % (a.out, text.count("\n"), len(text) // 4, cls, len(truncated), len(skipped)))
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
