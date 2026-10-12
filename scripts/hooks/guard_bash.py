#!/usr/bin/env python3
"""Claude Code PreToolUse hook for Bash: enforce the merge gate on the agent's own commands.

Reads the hook JSON on stdin. Exit 0 lets the command run; exit 2 blocks it and the message on
stderr goes back to the agent. Blocks:
  - `gh pr merge` unless scripts/pr_ready.py (taken from origin/main when it exists there) says the PR
    is eligible; `--admin` always
  - `gh pr review --approve` (approvals are for humans) and `gh api .../pulls/N/merge`
  - `git push` that targets main: explicit refspecs, HEAD/@ or a bare push while on main, --all/--mirror
Only real command words count: quoted text and heredoc bodies (commit messages, docs being written)
are ignored. The tokenizer is self-contained, so behaviour does not depend on the Python version
(the laptop's python3 is 3.7). It is a guard against honest mistakes, not a sandbox: `bash -c`,
`eval` or backticks can hide a command. The CI checks and human approvals remain the backstop.
Self-test without network: python3 scripts/hooks/guard_bash.py --selftest
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
PR_READY_LOCAL = os.path.join(REPO_ROOT, "scripts", "pr_ready.py")
GIT_GLOBAL_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
GH_FLAGS_WITH_VALUE = {"-R", "--repo"}


def strip_heredocs(cmd):
    """Drop heredoc bodies: they are data, not commands."""
    out, pending = [], []
    for line in cmd.split("\n"):
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            continue
        out.append(line)
        pending += re.findall(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
    return "\n".join(out)


def segments(cmd):
    """Split into simple commands (lists of words). Separators outside quotes: newline ; & | ( ) `
    and $( . Quoted text, even multi-line, stays inside one word."""
    text = strip_heredocs(cmd)
    segs, words, cur, has_word = [], [], [], False
    quote, i = None, 0

    def end_word():
        nonlocal cur, has_word
        if has_word:
            words.append("".join(cur))
        cur, has_word = [], False

    def end_seg():
        nonlocal words
        end_word()
        if words:
            segs.append(words)
        words = []

    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(text):
                cur.append(text[i + 1]); i += 2; continue
            if ch == quote:
                quote = None
            else:
                cur.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < len(text):
            if text[i + 1] != "\n":
                cur.append(text[i + 1]); has_word = True
            i += 2
            continue
        if ch in "'\"":
            quote, has_word = ch, True
        elif ch == "#" and not has_word:
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        elif ch == "$" and text[i + 1:i + 2] == "(":
            end_seg(); i += 2; continue
        elif ch in "\n;&|()`":
            end_seg()
        elif ch in " \t":
            end_word()
        else:
            cur.append(ch); has_word = True
        i += 1
    end_seg()
    return segs


def find_cmd(words, name):
    """Index of the word that invokes `name` (any path, e.g. ~/.local/bin/gh), or None."""
    for i, w in enumerate(words):
        if os.path.basename(w) == name:
            return i
    return None


def gh_sub(words, sub, action):
    """For `gh [flags] <sub> [flags] <action> args`, return (args, repo) or None."""
    i = find_cmd(words, "gh")
    if i is None:
        return None
    rest, repo, path, j = words[i + 1:], None, [], 0
    while j < len(rest) and len(path) < 2:
        w = rest[j]
        if w in GH_FLAGS_WITH_VALUE and j + 1 < len(rest):
            repo = rest[j + 1]; j += 2; continue
        if w.startswith("--repo="):
            repo = w.split("=", 1)[1]
        elif not w.startswith("-"):
            path.append(w)
        j += 1
    if path != [sub, action]:
        return None
    args = rest[j:]
    for k, w in enumerate(args):
        if w in GH_FLAGS_WITH_VALUE and k + 1 < len(args):
            repo = args[k + 1]
        elif w.startswith("--repo="):
            repo = w.split("=", 1)[1]
    return args, repo


def git_push_args(words):
    """Args after `git [global options] push`, or None."""
    i = find_cmd(words, "git")
    if i is None:
        return None
    j = i + 1
    while j < len(words) and words[j].startswith("-"):
        j += 2 if words[j] in GIT_GLOBAL_WITH_VALUE else 1
    if j < len(words) and words[j] == "push":
        return words[j + 1:]
    return None


def current_branch():
    out = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10)
    return out.stdout.strip()


def run_pr_ready(pr, repo, auto):
    """Run the gate logic from origin/main when it exists there, so a PR can't weaken its own gate."""
    base = subprocess.run(["git", "show", "origin/main:scripts/pr_ready.py"], capture_output=True, text=True,
                          cwd=REPO_ROOT, timeout=15)
    args = [pr] + (["--repo", repo] if repo else []) + (["--auto"] if auto else [])
    if base.returncode == 0:
        return subprocess.run([sys.executable, "-", "--root", REPO_ROOT] + args, input=base.stdout,
                              capture_output=True, text=True, cwd=REPO_ROOT, timeout=75)
    return subprocess.run([sys.executable, PR_READY_LOCAL] + args, capture_output=True, text=True,
                          cwd=REPO_ROOT, timeout=75)


def decide(cmd, online=True, branch=None):
    """Return None to allow, or a reason string to block."""
    to_main = False  # a `git checkout/switch main` earlier in the same command
    for words in segments(cmd):
        gi = find_cmd(words, "git")
        if gi is not None and words[gi + 1:gi + 2] in (["checkout"], ["switch"]) and "main" in words[gi + 2:]:
            to_main = True

        merge = gh_sub(words, "pr", "merge")
        if merge is not None:
            args, repo = merge
            if "--admin" in args:
                return "gh pr merge --admin bypasses the repository rules; never use it (AGENTS.md, Change workflow)."
            if not online:
                return "merge-needs-gate"
            pr = next((w for w in args if not w.startswith("-") and (w.isdigit() or "/pull/" in w)), None)
            if pr is None:
                return ("gh pr merge without an explicit PR number: name the PR "
                        "(`gh pr merge <N> --repo <owner/name> --squash --delete-branch`) so the gate can check it.")
            out = run_pr_ready(pr, repo, "--auto" in args)
            if out.returncode != 0:
                return "Merge gate refused PR %s:\n%s%s" % (pr, out.stdout, out.stderr)

        review = gh_sub(words, "pr", "review")
        if review is not None and any(w in ("--approve", "-a") for w in review[0]):
            return ("Agents never submit APPROVE reviews: approvals of experiment and gate changes must come from "
                    "the other teammate (AGENTS.md, Change workflow). Post the review report as a comment instead.")
        api_i = find_cmd(words, "gh")
        if api_i is not None and words[api_i + 1:api_i + 2] == ["api"] and any(
                re.search(r"pulls/\d+/merge\b", w) for w in words[api_i + 2:]):
            return "Merging through `gh api .../pulls/N/merge` skips the gate; use `gh pr merge` via the pr-gate skill."

        push = git_push_args(words)
        if push is not None:
            if any(w in ("--all", "--mirror") for w in push):
                return "git push --all/--mirror would push main; push one branch and open a PR."
            refs = [w for w in push if not w.startswith("-")][1:]  # first positional is the remote
            cur = branch if branch is not None else (current_branch() if online else "")
            if to_main:
                cur = "main"
            if not refs:
                refs = [cur]
            for r in refs:
                r = r.lstrip("+")
                src, _, dst = r.partition(":")
                dst = dst or src
                if dst in ("HEAD", "@"):
                    dst = cur
                if dst in ("main", "refs/heads/main"):
                    return "Pushing to main is not allowed: push a branch and open a PR (AGENTS.md, Change workflow)."
    return None


def selftest():
    B = "block"
    cases = [  # (command, expectation, current branch)
        ("ls -la", None, "feat/x"),
        ("git push -u origin feat/dropout", None, "feat/x"),
        ("git push --force-with-lease origin feat/dropout", None, "feat/x"),
        ("git push origin HEAD", None, "feat/x"),
        ("git push", None, "feat/x"),
        ("git push origin main", B, "feat/x"),
        ("git push origin HEAD:main", B, "feat/x"),
        ("git push origin refs/heads/main", B, "feat/x"),
        ("git checkout main && git push -f origin +main", B, "feat/x"),
        ("git checkout main && git push", B, "feat/x"),
        ("cd x; git push origin main 2>&1 | tail -2", B, "feat/x"),
        ("git push origin HEAD", B, "main"),
        ("git push -u origin HEAD", B, "main"),
        ("git push", B, "main"),
        ("git -C /repo push origin main", B, "feat/x"),
        ("git -c core.x=1 push origin main", B, "feat/x"),
        ("git push --all origin", B, "feat/x"),
        ("gh pr merge 5 --squash --admin", B, "feat/x"),
        ("echo done;gh pr merge 12 --admin", B, "feat/x"),
        ("(gh pr merge 5 --admin)", B, "feat/x"),
        ("gh pr merge 5 --squash", "merge-needs-gate", "feat/x"),
        ("cd repo&&gh pr merge 12", "merge-needs-gate", "feat/x"),
        ("out=$(gh pr merge 12 --squash --auto 2>&1)", "merge-needs-gate", "feat/x"),
        ("~/.local/bin/gh pr merge 12", "merge-needs-gate", "feat/x"),
        ("gh -R o/r pr merge 5", "merge-needs-gate", "feat/x"),
        ("gh pr -R o/r merge 5", "merge-needs-gate", "feat/x"),
        ("gh pr review 7 --approve", B, "feat/x"),
        ("gh pr review 7 --comment -b ok", None, "feat/x"),
        ("gh api -X PUT repos/o/r/pulls/5/merge", B, "feat/x"),
        ("gh pr view 5 --json state", None, "feat/x"),
        ("git commit -m 'never run gh pr merge --admin or git push origin main'", None, "feat/x"),
        ("git commit -m \"line one\ngh pr merge 5 --admin\ngit push origin main\"", None, "feat/x"),
        ("echo \"docs: gh pr merge 5\" > notes.md", None, "feat/x"),
        ("python3 - <<'EOF'\nprint('gh pr merge 5 --admin')\nsubprocess.run('git push origin main')\nEOF\nls", None, "feat/x"),
        ("cat > f.md <<EOF\ngit push origin main\nEOF\ngit push origin main", B, "feat/x"),
        ("grep -n 'push' x # git push origin main", None, "feat/x"),
    ]
    bad = 0
    for cmd, want, br in cases:
        got = decide(cmd, online=False, branch=br)
        ok = (got is None) if want is None else (got == want if want == "merge-needs-gate"
                                                  else got is not None and got != "merge-needs-gate")
        print(("ok   " if ok else "FAIL ") + repr(cmd)[:72] + " [" + br + "] -> " + ("allow" if got is None else got[:44]))
        bad += not ok
    print("%d/%d cases ok (python %s)" % (len(cases) - bad, len(cases), sys.version.split()[0]))
    return 1 if bad else 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0  # not a hook invocation we understand; don't interfere
    if data.get("tool_name") != "Bash":
        return 0
    cmd = (data.get("tool_input") or {}).get("command") or ""
    try:
        reason = decide(cmd)
    except Exception as e:  # fail closed only for the commands this hook guards
        if re.search(r"\bgh\b.*\b(merge|review)\b|\bgit\b.*\bpush\b", cmd, re.S):
            reason = "Merge/push gate could not evaluate this command (%s: %s); blocked to be safe." % (type(e).__name__, e)
        else:
            return 0
    if reason:
        sys.stderr.write(reason + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
