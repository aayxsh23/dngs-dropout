#!/usr/bin/env python3
"""Decide whether a pull request may be merged. Prints the reasons and exits 0 (eligible) or 1 (blocked).

Usage:
    python3 scripts/pr_ready.py 12                 # eligible for an immediate merge?
    python3 scripts/pr_ready.py 12 --auto          # eligible to enable GitHub auto-merge?

The merge gate (AGENTS.md "Change workflow") in executable form. The Claude Code hook
scripts/hooks/guard_bash.py runs it, taken from origin/main when it exists there, before any
`gh pr merge`. Requires an authenticated `gh`. Anything it cannot confirm is a blocker: a missing
check, an unknown mergeability state, absent evidence or a gh error never counts as a pass.
Must stay Python 3.7 compatible (the laptop's python3).
"""
import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time

CI_CHECK = "checks"            # job name in .github/workflows/ci.yml
REVIEW_STATUS = "agent-review"  # commit status set by the pr-gate skill after an Approve review
SMOKE_MARK = "GPU-SMOKE PASS"   # first words of scripts/gpu_smoke.sh's success line
PENDING = {"QUEUED", "IN_PROGRESS", "PENDING", "WAITING", "REQUESTED", "EXPECTED"}
DECISIVE_REVIEWS = {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}  # COMMENTED never changes a decision


def run(cmd, root, timeout=20):
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=root, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError("%s failed: %s" % (" ".join(cmd[:4]), out.stderr.strip()[:300]))
    return out.stdout


def gh_json(args, root):
    return json.loads(run(["gh"] + args, root))


def origin_repo(root):
    """owner/name of the `origin` remote. gh's own default can resolve to `upstream` in this clone."""
    url = run(["git", "remote", "get-url", "origin"], root).strip()
    m = re.search(r"github\.com[:/]+([^/]+/[^/]+?)(?:\.git)?/?$", url)
    if not m:
        raise RuntimeError("origin remote is not a GitHub URL: %s" % url)
    return m.group(1)


def parse_codeowners(text):
    """([(pattern, section)], {owner logins}); section from the '# [experiment]' / '# [gate]' markers."""
    rules, owners, section = [], set(), None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# [") and "]" in s:
            section = s[3:s.index("]")]
        elif s and not s.startswith("#"):
            parts = s.split()
            rules.append((parts[0], section or "protected"))
            owners.update(p.lstrip("@").lower() for p in parts[1:] if p.startswith("@"))
    return rules, owners


def matches(pattern, path):
    p = pattern.lstrip("/")
    if p.endswith("/"):
        return path.startswith(p)
    if "*" in p:  # CODEOWNERS '*' does not cross directories
        return fnmatch.fnmatchcase(path, p) and path.count("/") == p.count("/")
    return path == p


def classify(files, rule_sets):
    """Union over the base and head CODEOWNERS; within one file the LAST matching rule wins (GitHub semantics)."""
    hit = {}
    for f in files:
        for rules in rule_sets:
            section = None
            for pattern, sec in rules:
                if matches(pattern, f):
                    section = sec
            if section:
                hit.setdefault(section, set()).add(f)
    return hit


def codeowners_rules(root, base, head_sha):
    """Read CODEOWNERS from the base branch and from the PR head, never from the working tree."""
    subprocess.run(["git", "fetch", "-q", "origin", base, head_sha], cwd=root, capture_output=True, timeout=30)
    sets = []
    for ref in ("origin/" + base, head_sha):
        out = subprocess.run(["git", "show", "%s:.github/CODEOWNERS" % ref], cwd=root, capture_output=True,
                             text=True, timeout=15)
        if out.returncode == 0:
            sets.append(parse_codeowners(out.stdout))
    return [r for r, _ in sets], set().union(*[o for _, o in sets]) if sets else set()


def server_requires_ci(repo, root):
    """True only if GitHub itself enforces the CI check on main (rulesets need a public repo or GitHub Pro)."""
    try:
        rules = gh_json(["api", "repos/%s/rules/branches/main" % repo], root)
    except Exception:
        return False
    for r in rules:
        if r.get("type") == "required_status_checks":
            if any(c.get("context") == CI_CHECK for c in r.get("parameters", {}).get("required_status_checks", [])):
                return True
    return False


RANK = {"SUCCESS": 0, "NEUTRAL": 0, "SKIPPED": 0}  # anything else (pending, failure) ranks worse


def evaluate(pr, repo, auto, root, expect_head=None):
    fields = ("number,state,isDraft,baseRefName,headRefOid,mergeable,author,reviews,files,changedFiles,"
              "statusCheckRollup,comments,title")
    blockers, notes = [], []
    for attempt in range(3):  # GitHub computes mergeability lazily; UNKNOWN right after a push
        d = gh_json(["pr", "view", str(pr), "--repo", repo, "--json", fields], root)
        if d["mergeable"] != "UNKNOWN":
            break
        time.sleep(2)
    head, author = d["headRefOid"], d["author"]["login"]
    if expect_head and not head.startswith(expect_head):
        blockers.append("head is %s, not the reviewed %s: a newer push needs its own review" % (head[:7], expect_head[:7]))
    files = [f["path"] for f in d["files"]]
    if len(files) < d["changedFiles"]:  # gh lists at most 100 files; get them all
        files = run(["gh", "api", "--paginate", "repos/%s/pulls/%s/files" % (repo, d["number"]),
                     "--jq", ".[].filename"], root, timeout=40).splitlines()
        if len(files) < d["changedFiles"]:
            blockers.append("could list only %d of %d changed files" % (len(files), d["changedFiles"]))
    notes.append("%s PR #%d '%s' head %s by %s, %d files" % (repo, d["number"], d["title"], head[:7], author, len(files)))

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
    if auto and not server_requires_ci(repo, root):
        notes.append("GitHub does not enforce '%s' on main (no ruleset), so --auto is checked like an immediate merge"
                     % CI_CHECK)
        auto = False
    ci, review_status = None, None
    for c in d["statusCheckRollup"] or []:
        if c.get("__typename") == "CheckRun":
            name, state = c.get("name"), (c.get("conclusion") or c.get("status") or "").upper()
            if name == CI_CHECK:  # several runs named "checks": the worst one decides
                if ci is None or RANK.get(state, 2 if state not in PENDING else 1) > RANK.get(ci, 2 if ci not in PENDING else 1):
                    ci = state
            elif state not in ("SUCCESS", "NEUTRAL", "SKIPPED", "") and state not in PENDING:
                blockers.append("check '%s' is %s" % (name, state))
        elif c.get("__typename") == "StatusContext" and c.get("context") == REVIEW_STATUS:
            review_status = (c.get("state") or "").upper()
    if ci is None:
        blockers.append("CI check '%s' has not reported for %s" % (CI_CHECK, head[:7]))
    elif ci in PENDING:
        if auto:
            notes.append("CI '%s' still %s; GitHub enforces it and auto-merge will wait" % (CI_CHECK, ci))
        else:
            blockers.append("CI check '%s' is still %s (wait: gh pr checks %s --repo %s --watch)" % (CI_CHECK, ci, pr, repo))
    elif ci != "SUCCESS":
        blockers.append("CI check '%s' is %s" % (CI_CHECK, ci))
    if review_status != "SUCCESS":
        blockers.append("status '%s' on %s is %s (run the pr-gate review; any new push needs a new review)"
                        % (REVIEW_STATUS, head[:7], review_status or "missing"))

    # Human reviews: the latest decisive review per person (comments don't clear a decision)
    latest = {}
    for r in sorted(d["reviews"] or [], key=lambda r: r.get("submittedAt") or ""):
        if r["state"] in DECISIVE_REVIEWS:
            latest[r["author"]["login"]] = r
    for who, r in latest.items():
        if r["state"] == "CHANGES_REQUESTED":
            blockers.append("%s requested changes" % who)

    rule_sets, owners = codeowners_rules(root, d["baseRefName"], head)
    if not rule_sets:
        blockers.append("could not read .github/CODEOWNERS from the base or the head; cannot classify the change")
    hit = classify(files, rule_sets)
    if hit:
        for section, fs in sorted(hit.items()):
            fs = sorted(fs)
            notes.append("%s paths touched: %s" % (section, ", ".join(fs[:6]) + (" ..." if len(fs) > 6 else "")))
        approved = [who for who, r in latest.items() if who != author and who.lower() in owners
                    and r["state"] == "APPROVED" and (r.get("commit") or {}).get("oid") == head]
        if not approved:
            blockers.append("protected paths changed: needs an APPROVED review on %s from the other teammate "
                            "(a code owner who is not the author)" % head[:7])
        else:
            notes.append("human approval on head by " + ", ".join(approved))
    elif rule_sets:
        notes.append("routine change: no protected paths touched")
    if "experiment" in hit:
        mark = re.compile(r"%s %s(?=\s|$)" % (re.escape(SMOKE_MARK), re.escape(head)), re.M)
        if not any(mark.search(c.get("body") or "") for c in d["comments"] or []):
            blockers.append("experiment paths changed: no '%s %s' comment from a clean tree (run scripts/gpu_smoke.sh "
                            "on the PR head and post its last line)" % (SMOKE_MARK, head[:7]))
    return blockers, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pr", help="PR number or URL")
    ap.add_argument("--repo", default=None, help="owner/name; must be the origin repo (default: origin)")
    ap.add_argument("--auto", action="store_true", help="checking for `gh pr merge --auto`")
    ap.add_argument("--root", default=None, help="repository checkout (default: the git toplevel of the cwd)")
    ap.add_argument("--expect-head", default=None, help="the head SHA that was reviewed (gh pr merge --match-head-commit)")
    a = ap.parse_args()
    pr = a.pr.rstrip("/").split("/")[-1]
    try:
        root = a.root or run(["git", "rev-parse", "--show-toplevel"], os.getcwd()).strip()
        repo = origin_repo(root)
        if a.repo and a.repo.lower() != repo.lower():
            raise RuntimeError("the merge targets %s, but this checkout's origin is %s" % (a.repo, repo))
        if not a.repo:  # the merge command relies on gh's default repo: it must be origin, not upstream
            out = subprocess.run(["gh", "repo", "set-default", "--view"], capture_output=True, text=True,
                                 cwd=root, timeout=15)
            if out.stdout.strip().lower() != repo.lower():
                raise RuntimeError("gh's default repo is '%s', not origin %s: add `--repo %s` to the merge command"
                                   % (out.stdout.strip() or "unset", repo, repo))
        blockers, notes = evaluate(pr, repo, a.auto, root, a.expect_head)
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
