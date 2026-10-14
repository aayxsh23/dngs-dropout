#!/usr/bin/env python3
"""Build the single input file a pr-reviewer needs, so the reviewer reads one file instead of
running a dozen gh/git commands, and never re-runs what CI already proved.

    python3 scripts/review_bundle.py 12 --out /tmp/review_12.md                    # full review
    python3 scripts/review_bundle.py 12 --since 3466f19 --out /tmp/review_12.md    # re-review: only what changed

Contents: PR metadata and risk class (from CODEOWNERS), the goal (PR body), CI / agent-review /
GPU-smoke evidence on the head, the previous pr-gate report (re-reviews), the changed-file list,
and the diff, capped per file and in total so the bundle stays small. Truncated files are listed;
the reviewer opens them with Read only if needed. Notebooks appear as a diff of their code-cell
sources; CSVs appear when they have at most MAX_CSV_LINES lines.

Fails closed (non-zero exit, no bundle): CODEOWNERS unreadable, changed files not all listed, or a
--since re-review with no pr-gate report for that SHA from a code owner. Python 3.7 compatible.
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr_ready import (CI_CHECK, PENDING, RANK, REVIEW_STATUS, SMOKE_MARK, classify, codeowners_rules,  # noqa: E402
                      gh_json, origin_repo, run)

# Files whose diff is noise for review (lockfiles, binaries): listed, not shown. Vendored CUDA/C++ under
# submodules/ is shown (capped like any file): it is experiment-class code.
SKIP_DIFF = re.compile(r"(\.lock$|lock\.yml$|^pip_freeze|\.svg$|\.png$)")
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


DOCS_ONLY = re.compile(r"(^docs/|\.md$)")


class BundleError(Exception):
    pass


def risk_class(files, rule_sets):
    """CODEOWNERS class, or "docs-only" (every file under docs/ or *.md, none protected: no reviewer needed).
    No readable CODEOWNERS means the class can't be known: fail closed."""
    if not rule_sets:
        raise BundleError("no readable .github/CODEOWNERS on the base or the head: can't classify")
    hit = classify(files, rule_sets)
    if hit:
        return "+".join(sorted(hit))
    if files and all(DOCS_ONLY.search(f) for f in files):
        return "docs-only"
    return "routine"


def worst_ci(rollup):
    """State of the CI check; with several runs the worst one decides (as in pr_ready.py)."""
    ci = None
    for c in rollup or []:
        if c.get("__typename") == "CheckRun" and c.get("name") == CI_CHECK:
            state = (c.get("conclusion") or c.get("status") or "?").upper()
            if ci is None or RANK.get(state, 2 if state not in PENDING else 1) > RANK.get(ci, 2 if ci not in PENDING else 1):
                ci = state
    return ci or "missing"


def smoke_line(bodies, head):
    return next((ln.strip() for b in bodies for ln in b.splitlines() if (SMOKE_MARK + " " + head) in ln), "none for head")


def previous_report(comments, since, allowed):
    """The last pr-gate report for exactly the `since` commit, posted by a code owner or the repo owner."""
    found = None
    for c in comments or []:
        body = (c.get("body") or "").lstrip()
        login = ((c.get("author") or {}).get("login") or "").lower()
        first = body.splitlines()[0] if body else ""
        if login in allowed and first.startswith("### pr-gate review") and ("@ %s" % since[:7]) in first:
            found = body
    return found


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
    try:
        return build(a)
    except BundleError as e:
        print("review_bundle: %s. No bundle written; fix this before any review." % e, file=sys.stderr)
        return 2


def build(a):
    root = run(["git", "rev-parse", "--show-toplevel"], os.getcwd()).strip()
    repo = origin_repo(root)
    d = gh_json(["pr", "view", a.pr, "--repo", repo, "--json",
                 "number,title,body,author,baseRefName,headRefOid,files,statusCheckRollup,comments,changedFiles"], root)
    head, base = d["headRefOid"], d["baseRefName"]
    rules, owners = codeowners_rules(root, base, head)
    files = [f["path"] for f in d["files"]]
    if len(files) < d["changedFiles"]:  # gh lists at most 100 files; get them all (as pr_ready.py does)
        files = run(["gh", "api", "--paginate", "repos/%s/pulls/%s/files" % (repo, d["number"]),
                     "--jq", ".[].filename"], root, timeout=40).splitlines()
        if len(files) < d["changedFiles"]:
            raise BundleError("could list only %d of %d changed files" % (len(files), d["changedFiles"]))
    cls = risk_class(files, rules)

    status = "missing"
    for c in d["statusCheckRollup"] or []:
        if c.get("__typename") == "StatusContext" and c.get("context") == REVIEW_STATUS:
            status = (c.get("state") or "?").upper()
    bodies = [c.get("body") or "" for c in d["comments"] or []]
    prev = None
    if a.since:
        prev = previous_report(d["comments"], a.since, owners | {repo.split("/")[0].lower()})
        if prev is None:
            raise BundleError("--since %s: no '### pr-gate review ... @ %s' comment from a code owner on this PR"
                              % (a.since[:7], a.since[:7]))

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
        " notebooks, aggregate fixtures, configs, hook self-test, review_bundle self-test, analysis scripts on fixtures:"
        " diag_curves sharp checks, stats verdict, qual_grid, views)" % (CI_CHECK, worst_ci(d["statusCheckRollup"])),
        "- `%s` status on head: %s" % (REVIEW_STATUS, status),
        "- GPU smoke: %s" % smoke_line(bodies, head),
    ])))
    goal = (d["body"] or "").strip().splitlines()
    quoted = ["> " + ln for ln in goal[:40]] + (["> (%d more lines not shown)" % (len(goal) - 40)] if len(goal) > 40 else [])
    out.append(section("Goal / acceptance criteria (PR body, quoted)", "\n".join(quoted) or "(empty: ask the author)"))
    if prev:
        out.append(section("Previous pr-gate report (verify its blocking findings)", "\n".join(prev.splitlines()[:60])))

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
                           "\n".join(["- " + t for t in truncated] + ["- %s (lockfile/binary or a CSV over %d lines: skipped)" % (s, MAX_CSV_LINES) for s in skipped])))
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
