#!/usr/bin/env python3
"""Decide whether a pull request may be merged. Prints the reasons and exits 0 (eligible) or 1 (blocked).

Usage:
    python3 scripts/pr_ready.py 12            # eligible for an immediate merge?
    python3 scripts/pr_ready.py 12 --auto     # eligible to enable GitHub auto-merge? (CI may still be running)

The merge gate (see AGENTS.md "Change workflow") in executable form. The Claude Code hook
scripts/hooks/guard_bash.py runs it before any `gh pr merge`. Requires an authenticated `gh`.
Anything it cannot confirm counts as a blocker: a missing check, an unknown mergeability state or
absent evidence is never treated as a pass.
"""
import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time

CI_CHECK = "checks"            # job name in .github/workflows/ci.yml
REVIEW_STATUS = "agent-review"  # commit status set by the pr-gate skill after an Approve review
SMOKE_MARK = "GPU-SMOKE PASS"   # first words of scripts/gpu_smoke.sh's success line
PENDING = {"QUEUED", "IN_PROGRESS", "PENDING", "WAITING", "REQUESTED", "EXPECTED"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def codeowners(path=os.path.join(ROOT, ".github", "CODEOWNERS")):
    """[(pattern, section)] where section is 'experiment' or 'gate' (from the [..] markers)."""
    rules, section = [], None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s.startswith("# [") and "]" in s:
            section = s[3:s.index("]")]
        elif s and not s.startswith("#"):
            rules.append((s.split()[0], section))
    return rules


def matches(pattern, path):
    p = pattern.lstrip("/")
    if p.endswith("/"):
        return path.startswith(p)
    if "*" in p:  # CODEOWNERS '*' does not cross directories
        return fnmatch.fnmatchcase(path, p) and path.count("/") == p.count("/")
    return path == p


def sections_for(files, rules):
    hit = {}
    for f in files:
        for pattern, section in rules:
            if matches(pattern, f):
                hit.setdefault(section, []).append(f)
                break
    return hit


def gh_json(args):
    out = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("gh " + " ".join(args) + " failed: " + out.stderr.strip())
    return json.loads(out.stdout)


def evaluate(pr, repo, auto):
    fields = "number,state,isDraft,baseRefName,headRefOid,mergeable,author,reviews,files,statusCheckRollup,comments,title"
    blockers, notes = [], []
    for attempt in range(4):  # GitHub computes mergeability lazily; UNKNOWN right after a push
        d = gh_json(["pr", "view", str(pr), "--repo", repo, "--json", fields])
        if d["mergeable"] != "UNKNOWN":
            break
        time.sleep(3)
    head, author = d["headRefOid"], d["author"]["login"]
    files = [f["path"] for f in d["files"]]
    notes.append("PR #%d '%s' head %s by %s, %d files" % (d["number"], d["title"], head[:7], author, len(files)))

    if d["state"] != "OPEN":
        blockers.append("PR is %s, not OPEN" % d["state"])
    if d["isDraft"]:
        blockers.append("PR is a draft")
    if d["baseRefName"] != "main":
        blockers.append("base branch is '%s'; agents merge only into main (retarget or rebase first)" % d["baseRefName"])
    if d["mergeable"] == "CONFLICTING":
        blockers.append("merge conflicts with %s" % d["baseRefName"])
    elif d["mergeable"] != "MERGEABLE":
        blockers.append("mergeability is %s (inconclusive)" % d["mergeable"])

    # CI and the agent-review status, both on the head commit
    ci, review_status = None, None
    for c in d["statusCheckRollup"] or []:
        if c.get("__typename") == "CheckRun":
            name, state = c.get("name"), (c.get("conclusion") or c.get("status") or "").upper()
            if name == CI_CHECK:
                ci = state
            elif state not in ("SUCCESS", "NEUTRAL", "SKIPPED", "") and state not in PENDING:
                blockers.append("check '%s' is %s" % (name, state))
        elif c.get("__typename") == "StatusContext" and c.get("context") == REVIEW_STATUS:
            review_status = (c.get("state") or "").upper()
    if ci is None:
        blockers.append("CI check '%s' has not reported for %s" % (CI_CHECK, head[:7]))
    elif ci in PENDING:
        if auto:
            notes.append("CI '%s' still %s; GitHub auto-merge will wait for it" % (CI_CHECK, ci))
        else:
            blockers.append("CI check '%s' is still %s" % (CI_CHECK, ci))
    elif ci != "SUCCESS":
        blockers.append("CI check '%s' is %s" % (CI_CHECK, ci))
    if review_status != "SUCCESS":
        blockers.append("status '%s' on %s is %s (run the pr-gate review; any new push needs a new review)"
                        % (REVIEW_STATUS, head[:7], review_status or "missing"))

    # Human reviews: latest review per person
    latest = {}
    for r in sorted(d["reviews"] or [], key=lambda r: r.get("submittedAt") or ""):
        latest[r["author"]["login"]] = r
    for who, r in latest.items():
        if r["state"] == "CHANGES_REQUESTED":
            blockers.append("%s requested changes" % who)

    hit = sections_for(files, codeowners())
    if hit:
        for section, fs in sorted(hit.items()):
            notes.append("%s paths touched: %s" % (section, ", ".join(fs[:6]) + (" ..." if len(fs) > 6 else "")))
        approved = [who for who, r in latest.items() if who != author and r["state"] == "APPROVED"
                    and (r.get("commit") or {}).get("oid") == head]
        if not approved:
            blockers.append("protected paths changed: needs an APPROVED review on %s from the other teammate "
                            "(a code owner who is not the author)" % head[:7])
        else:
            notes.append("human approval on head by " + ", ".join(approved))
    else:
        notes.append("routine change: no protected paths touched")
    if "experiment" in hit:
        bodies = [c.get("body") or "" for c in d["comments"] or []]
        if not any((SMOKE_MARK + " " + head) in b for b in bodies):
            blockers.append("experiment paths changed: no '%s %s' comment (run scripts/gpu_smoke.sh on the "
                            "PR head and post its last line)" % (SMOKE_MARK, head[:7]))
    return blockers, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pr", help="PR number or URL")
    ap.add_argument("--repo", default=None, help="owner/name (default: this checkout's GitHub repo)")
    ap.add_argument("--auto", action="store_true", help="checking for `gh pr merge --auto`: CI may be pending")
    a = ap.parse_args()
    repo = a.repo or gh_json(["repo", "view", "--json", "nameWithOwner"])["nameWithOwner"]
    pr = a.pr.rstrip("/").split("/")[-1]
    try:
        blockers, notes = evaluate(pr, repo, a.auto)
    except Exception as e:  # cannot establish eligibility = blocked
        print("BLOCKED: could not evaluate PR %s: %s" % (pr, e))
        return 1
    for n in notes:
        print("  - " + n)
    if blockers:
        print("BLOCKED (%d):" % len(blockers))
        for b in blockers:
            print("  x " + b)
        return 1
    print("ELIGIBLE%s" % (" for auto-merge" if a.auto else " to merge"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
