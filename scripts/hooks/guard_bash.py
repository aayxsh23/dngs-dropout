#!/usr/bin/env python3
"""Claude Code PreToolUse hook for Bash: enforce the merge gate on the agent's own commands.

Reads the hook JSON on stdin. Exit 0 lets the command run; exit 2 blocks it and the message on
stderr goes back to the agent. Blocks:
  - `gh pr merge` unless scripts/pr_ready.py says the PR is eligible (with --auto: CI may be pending)
  - `gh pr merge --admin` (bypasses repository rules), always
  - `git push` that targets main, or force-pushes main
Only real command words count: text inside quotes or heredoc bodies (commit messages, docs being
written) is ignored. Self-test without network: python3 scripts/hooks/guard_bash.py --selftest
"""
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PR_READY = os.path.join(os.path.dirname(HERE), "pr_ready.py")
SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "|&", ";;"}


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
    """Yield the word lists of simple commands, with quoted strings kept as single words."""
    text = strip_heredocs(cmd).replace("\\\n", " ")
    for line in text.split("\n"):
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = "#"
        try:
            tokens = list(lex)
        except ValueError:  # unbalanced quotes: fall back to plain words
            tokens = line.split()
        words = []
        for t in tokens:
            if t in SEPARATORS:
                if words:
                    yield words
                words = []
            else:
                words.append(t)
        if words:
            yield words


def sub_after(words, seq):
    """If words contains the consecutive sequence seq (e.g. gh pr merge), return the words after it."""
    n = len(seq)
    for i in range(len(words) - n + 1):
        if words[i:i + n] == seq:
            return words[i + n:]
    return None


def decide(cmd, online=True):
    """Return None to allow, or a reason string to block."""
    for words in segments(cmd):
        args = sub_after(words, ["gh", "pr", "merge"])
        if args is not None:
            if "--admin" in args:
                return "gh pr merge --admin bypasses the repository rules; never use it (AGENTS.md, Change workflow)."
            if not online:
                return "merge-needs-gate"
            pr = next((w for w in args if not w.startswith("-") and (w.isdigit() or "/pull/" in w)), None)
            if pr is None:
                return ("gh pr merge without an explicit PR number: name the PR (`gh pr merge <N> --squash ...`) "
                        "so the gate can check it.")
            check = [sys.executable, PR_READY, pr] + (["--auto"] if "--auto" in args else [])
            out = subprocess.run(check, capture_output=True, text=True)
            if out.returncode != 0:
                return "Merge gate refused PR %s:\n%s%s" % (pr, out.stdout, out.stderr)
        args = sub_after(words, ["git", "push"])
        if args is not None:
            refs = [w for w in args if not w.startswith("-")][1:]  # first positional is the remote
            hits_main = any(r in ("main", "+main") or r.endswith(":main") or r.endswith(":refs/heads/main")
                            for r in refs)
            if not refs and online:  # bare `git push` pushes the current branch
                cur = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
                hits_main = cur.stdout.strip() == "main"
            if hits_main:
                return "Pushing to main is not allowed: push a branch and open a PR (AGENTS.md, Change workflow)."
    return None


def selftest():
    cases = [
        ("ls -la", None),
        ("git push -u origin feat/dropout", None),
        ("git push --force-with-lease origin feat/dropout", None),
        ("git push origin main", "block"),
        ("git push origin HEAD:main", "block"),
        ("git checkout main && git push -f origin +main", "block"),
        ("cd x; git push origin main 2>&1 | tail -2", "block"),
        ("gh pr merge 5 --squash --admin", "block"),
        ("gh pr merge 5 --squash", "merge-needs-gate"),
        ("echo hi; gh pr merge --squash", "merge-needs-gate"),
        ("gh pr view 5 --json state", None),
        ("git commit -m 'never run gh pr merge --admin or git push origin main'", None),
        ("echo \"docs: gh pr merge 5\" > notes.md", None),
        ("python3 - <<'EOF'\nprint('gh pr merge 5 --admin')\nsubprocess.run('git push origin main')\nEOF\nls", None),
        ("cat > f.md <<EOF\ngit push origin main\nEOF\ngit push origin main", "block"),
    ]
    bad = 0
    for cmd, want in cases:
        got = decide(cmd, online=False)
        ok = (got is None) if want is None else (got == want if want == "merge-needs-gate" else got is not None)
        print(("ok   " if ok else "FAIL ") + repr(cmd)[:70] + " -> " + ("allow" if got is None else got[:50]))
        bad += not ok
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
    reason = decide((data.get("tool_input") or {}).get("command") or "")
    if reason:
        sys.stderr.write(reason + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
