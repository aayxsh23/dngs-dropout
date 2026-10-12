#!/usr/bin/env python3
"""Claude Code PreToolUse hook for Bash: enforce the merge gate on the agent's own commands.

Reads the hook JSON on stdin. Exit 0 lets the command run; exit 2 blocks it and the message on
stderr goes back to the agent. Blocks:
  - `gh pr merge` unless it names the PR and `--match-head-commit <sha>` and scripts/pr_ready.py (taken
    from origin/main when it exists there) says that exact head is eligible; `--admin` always
  - approvals by agents (`gh pr review --approve`, API/GraphQL approve) and merges or main-ref updates
    through `gh api`
  - `git push` that targets main: explicit refspecs, HEAD/@ or a bare push while on main, --all/--mirror
Only real command words count: quoted text, comments and heredoc bodies are data. Command
substitutions ($(...) and backticks), also inside double quotes, are parsed as commands. The
tokenizer is self-contained, so behaviour does not depend on the Python version (the laptop's
python3 is 3.7). It is a guard against honest mistakes, not a sandbox: `bash -c`, `eval` or a
script file can still hide a command. CI, the GitHub ruleset and human approvals are the backstop.
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
PREFIXES = {"env", "sudo", "command", "builtin", "exec", "time", "nohup", "nice", "{", "!", "then", "do", "else"}
REDIR = re.compile(r"^\d*(<<<|<<-?|<>|<&|>&|&>>|&>|>>|>\||<|>)")
API_DANGER = re.compile(r"pulls/\d+/merge\b|/merges\b|git/refs/heads/main\b|mergePullRequest|"
                        r"enablePullRequestAutoMerge|event[=:]\s*\\?\"?APPROVE")


class Unbalanced(ValueError):
    pass


def segments(text):
    """Split a command line into simple commands (lists of words). Separators outside quotes:
    newline ; & | ( ) and the bounds of $( ) and backticks. Quoted text stays inside one word.
    Heredoc bodies and comments are skipped. Substitutions inside double quotes are parsed as
    commands by the same scanner (so heredocs inside them are handled too). Unterminated quotes or
    substitutions raise Unbalanced; main() then blocks guarded commands (fail closed)."""
    return _scan(text, 0, False)[0]


def _scan(text, i, in_subst):
    """Scan from i. If in_subst, stop at the ')' that closes a $( and return (segs, its index)."""
    segs, words, cur, has_word = [], [], [], False
    quote, heredocs, depth = None, [], 0

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
        if quote == "'":
            if ch == "'":
                quote = None
            else:
                cur.append(ch)
            i += 1
            continue
        if quote == '"':
            if ch == "\\" and i + 1 < len(text):
                cur.append(text[i + 1]); i += 2; continue
            if ch == '"':
                quote = None; i += 1; continue
            if ch == "$" and text[i + 1:i + 2] == "(" and text[i + 2:i + 3] != "(":
                inner, j = _scan(text, i + 2, True)
                segs.extend(inner); cur.append("$(...)"); i = j + 1; continue
            if ch == "`":
                j = text.find("`", i + 1)
                if j < 0:
                    raise Unbalanced("unterminated backtick")
                segs.extend(segments(text[i + 1:j])); cur.append("`...`"); i = j + 1; continue
            cur.append(ch); i += 1
            continue
        # unquoted
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
        elif text.startswith("<<<", i):  # here-string: one word of data, not a heredoc
            cur.append("<<<"); has_word = True; i += 3; continue
        elif ch == "<" and text[i:i + 2] == "<<":
            m = re.match(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", text[i:])
            if m:
                end_word()
                heredocs.append(m.group(2))
                i += m.end()
                continue
            cur.append(ch); has_word = True
        elif ch == "$" and text[i + 1:i + 2] == "(":
            end_seg(); depth += 1; i += 2; continue
        elif ch == "&" and ((cur and cur[-1] in "<>") or text[i + 1:i + 2] == ">"):
            cur.append(ch); has_word = True  # part of a redirection: >&  <&  &>
        elif ch == "\n":
            end_seg()
            i += 1
            for delim in heredocs:  # skip heredoc bodies up to each terminator line
                while i < len(text):
                    nl = text.find("\n", i)
                    line = text[i:] if nl < 0 else text[i:nl]
                    i = len(text) if nl < 0 else nl + 1
                    if line.strip() == delim:
                        break
            heredocs = []
            continue
        elif ch == "(":
            end_seg(); depth += 1
        elif ch == ")":
            end_seg()
            if depth == 0 and in_subst:
                return segs, i
            depth = max(depth - 1, 0)
        elif ch in ";&|`":
            end_seg()
        elif ch in " \t":
            end_word()
        else:
            cur.append(ch); has_word = True
        i += 1
    if quote or in_subst:
        raise Unbalanced("unterminated quote or command substitution")
    end_seg()
    return segs, i


def strip_redirs(words):
    out, skip = [], False
    for w in words:
        if skip:
            skip = False
            continue
        m = REDIR.match(w)
        if m:
            skip = m.end() == len(w)  # operator alone: its target is the next word
            continue
        out.append(w)
    return out


def command_index(words, name):
    """Index of the command word if this simple command runs `name` (any path), skipping VAR=x and
    prefixes like env/sudo/time; else None. `echo git push` is not a git command."""
    k = 0
    while k < len(words):
        w = words[k]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w) or w in PREFIXES:
            k += 1
            continue
        if words[k - 1:k] and words[k - 1] in PREFIXES and w.startswith("-") and k > 0:
            k += 1  # flags of a prefix command (env -i, nice -n ...)
            continue
        return k if os.path.basename(w) == name else None
    return None


def gh_sub(words, sub, action):
    """For `gh [flags] <sub> [flags] <action> args`, return (args, repo) or None."""
    i = command_index(words, "gh")
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


def git_sub(words, sub):
    """Args after `git [global options] <sub>`, or None."""
    i = command_index(words, "git")
    if i is None:
        return None
    j = i + 1
    while j < len(words) and words[j].startswith("-"):
        j += 2 if words[j] in GIT_GLOBAL_WITH_VALUE else 1
    if j < len(words) and words[j] == sub:
        return words[j + 1:]
    return None


def checkout_target(args):
    """Branch a `git checkout/switch <args>` leaves you on, or None if it doesn't change branch."""
    if "--" in args:
        return None
    for k, w in enumerate(args):
        if w in ("-b", "-B", "-c", "-C") and k + 1 < len(args):
            return args[k + 1]
    pos = [w for w in args if not w.startswith("-")]
    return pos[0] if pos else None


def current_branch():
    out = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10)
    return out.stdout.strip()


def run_pr_ready(pr, repo, sha):
    """Run the gate logic from origin/main when it exists there, so a PR can't weaken its own gate."""
    base = subprocess.run(["git", "show", "origin/main:scripts/pr_ready.py"], capture_output=True, text=True,
                          cwd=REPO_ROOT, timeout=15)
    args = [pr, "--expect-head", sha] + (["--repo", repo] if repo else [])
    if base.returncode == 0 and "--expect-head" in base.stdout:
        return subprocess.run([sys.executable, "-", "--root", REPO_ROOT] + args, input=base.stdout,
                              capture_output=True, text=True, cwd=REPO_ROOT, timeout=75)
    return subprocess.run([sys.executable, PR_READY_LOCAL, "--root", REPO_ROOT] + args, capture_output=True,
                          text=True, cwd=REPO_ROOT, timeout=75)


def decide(cmd, online=True, branch=None):
    """Return None to allow, or a reason string to block."""
    cur = branch
    for words in segments(cmd):
        co = git_sub(words, "checkout")
        co = co if co is not None else git_sub(words, "switch")
        if co is not None:
            target = checkout_target(co)
            if target is not None:
                cur = target

        merge = gh_sub(words, "pr", "merge")
        if merge is not None:
            args, repo = merge[0], merge[1]
            if "--admin" in args:
                return "gh pr merge --admin bypasses the repository rules; never use it (AGENTS.md, Change workflow)."
            if not online:
                return "merge-needs-gate"
            args = strip_redirs(args)
            pr = next((w for w in args if not w.startswith("-") and (w.isdigit() or "/pull/" in w)), None)
            sha = None
            for k, w in enumerate(args):
                if w == "--match-head-commit" and k + 1 < len(args):
                    sha = args[k + 1]
                elif w.startswith("--match-head-commit="):
                    sha = w.split("=", 1)[1]
            if pr is None or not sha:
                return ("gh pr merge must name the PR and pin the reviewed head: "
                        "`gh pr merge <N> --repo aayxsh23/dngs-dropout --squash --delete-branch "
                        "--match-head-commit <reviewed sha>`.")
            out = run_pr_ready(pr, repo, sha)
            if out.returncode != 0:
                return "Merge gate refused PR %s:\n%s%s" % (pr, out.stdout, out.stderr)

        review = gh_sub(words, "pr", "review")
        if review is not None and any(w in ("-a",) or w.startswith("--approve") for w in review[0]):
            return ("Agents never submit APPROVE reviews: approvals must come from the other teammate "
                    "(AGENTS.md, Change workflow). Post the review report as a comment instead.")
        api = command_index(words, "gh")
        api_args = " ".join(words[api + 2:]) if api is not None and words[api + 1:api + 2] == ["api"] else None
        if api_args is not None and "--input" in api_args and re.search(r"pulls/\d+/reviews|graphql", api_args):
            return ("`gh api ... --input` on reviews/GraphQL can approve or merge outside the gate; "
                    "post review reports with `gh pr comment` instead.")
        if api_args is not None and API_DANGER.search(api_args):
            return ("This `gh api` call would merge, approve or move main outside the gate; "
                    "use the pr-gate skill (approvals are for humans).")

        push = git_sub(words, "push")
        if push is not None:
            push = strip_redirs(push)
            if any(w in ("--all", "--mirror") for w in push):
                return "git push --all/--mirror would push main; push one branch and open a PR."
            here = cur if cur is not None else (current_branch() if online else "")
            refs = [w for w in push if not w.startswith("-")][1:]  # first positional is the remote
            if not refs:
                refs = [here]
            for r in refs:
                src, _, dst = r.lstrip("+").partition(":")
                dst = dst or src
                if dst in ("HEAD", "@"):
                    dst = here
                if dst in ("main", "refs/heads/main"):
                    return "Pushing to main is not allowed: push a branch and open a PR (AGENTS.md, Change workflow)."
    return None


def selftest():
    B, G = "block", "merge-needs-gate"
    apos = "git commit -m \"$(cat <<'EOF'\nfix: don't break (x)\nEOF\n)\""
    heredoc_commit ="git commit -m \"$(cat <<'EOF'\nfix: x\n\ngh pr merge 5 --admin\nEOF\n)\" && git push -u origin HEAD"
    cases = [  # (command, expectation, current branch)
        ("ls -la", None, "feat/x"),
        ("git push -u origin feat/dropout", None, "feat/x"),
        ("git push --force-with-lease origin feat/dropout", None, "feat/x"),
        ("git push origin HEAD", None, "feat/x"),
        ("git push", None, "feat/x"),
        ("git push origin feat/x 2>&1", None, "feat/x"),
        ("git push origin main", B, "feat/x"),
        ("git push origin HEAD:main", B, "feat/x"),
        ("git push origin refs/heads/main", B, "feat/x"),
        ("git checkout main && git push -f origin +main", B, "feat/x"),
        ("git checkout main && git push", B, "feat/x"),
        ("cd x; git push origin main 2>&1 | tail -2", B, "feat/x"),
        ("git push origin HEAD", B, "main"),
        ("git push -u origin HEAD", B, "main"),
        ("git push", B, "main"),
        ("git push origin 2>&1 | tail -3", B, "main"),
        ("git push -u origin >/dev/null", B, "main"),
        ("git push origin > log 2>&1", B, "main"),
        ("git -C /repo push origin main", B, "feat/x"),
        ("git -c core.x=1 push origin main", B, "feat/x"),
        ("env GIT_TRACE=1 git push origin main", B, "feat/x"),
        ("git push --all origin", B, "feat/x"),
        ("echo \"$(git push origin main)\"", B, "feat/x"),
        ("git checkout main && git pull && git checkout -b feat/y && git push -u origin HEAD", None, "feat/x"),
        ("git switch -c feat/y main && git push -u origin HEAD", None, "feat/x"),
        ("git checkout -b fix/y main && git push -u origin HEAD", None, "feat/x"),
        ("git stash && git checkout main && git pull && git checkout feat/x && git stash pop && git push", None, "feat/x"),
        ("echo git push origin main", None, "feat/x"),
        ("gh pr merge 5 --squash --admin", B, "feat/x"),
        ("echo done;gh pr merge 12 --admin", B, "feat/x"),
        ("(gh pr merge 5 --admin)", B, "feat/x"),
        ("out=\"$(gh pr merge 12 --squash --admin 2>&1)\"; echo \"$out\"", B, "feat/x"),
        ("r=\"`gh pr merge 5 --admin`\"", B, "feat/x"),
        ("gh pr merge 5 --squash", G, "feat/x"),
        ("cd repo&&gh pr merge 12", G, "feat/x"),
        ("out=$(gh pr merge 12 --squash --auto 2>&1)", G, "feat/x"),
        ("out=\"$(gh pr merge 12 --squash)\"", G, "feat/x"),
        ("echo \"merged: $(gh pr merge 12 --squash)\"", G, "feat/x"),
        ("~/.local/bin/gh pr merge 12", G, "feat/x"),
        ("gh -R o/r pr merge 5", G, "feat/x"),
        ("gh pr -R o/r merge 5", G, "feat/x"),
        ("gh pr review 7 --approve", B, "feat/x"),
        ("gh pr review 5 --approve=true", B, "feat/x"),
        ("gh pr review 7 --comment -b ok", None, "feat/x"),
        ("gh api -X PUT repos/o/r/pulls/5/merge", B, "feat/x"),
        ("gh api -X POST repos/o/r/pulls/5/reviews -f event=APPROVE", B, "feat/x"),
        ("gh api graphql -f query='mutation { mergePullRequest(input:{}) { clientMutationId } }'", B, "feat/x"),
        ("gh api -X PATCH repos/o/r/git/refs/heads/main -f sha=abc", B, "feat/x"),
        ("gh api repos/o/r/pulls/5/reviews", None, "feat/x"),
        ("gh pr view 5 --json state", None, "feat/x"),
        ("git commit -m 'never run gh pr merge --admin or git push origin main'", None, "feat/x"),
        ("git commit -m \"line one\ngh pr merge 5 --admin\ngit push origin main\"", None, "feat/x"),
        (heredoc_commit, None, "feat/x"),
        ("echo \"docs: gh pr merge 5\" > notes.md", None, "feat/x"),
        ("echo a#b; grep -n '#' f; curl https://x.io/#frag", None, "feat/x"),
        ("python3 - <<'EOF'\nprint('gh pr merge 5 --admin')\nsubprocess.run('git push origin main')\nEOF\nls", None, "feat/x"),
        ("cat > f.md <<EOF\ngit push origin main\nEOF\ngit push origin main", B, "feat/x"),
        ("bc <<< x\ngit push origin main", B, "feat/x"),
        ("git commit -m 'note: cat <<END'\ngh pr merge 5 --admin", B, "feat/x"),
        ("grep -n 'push' x # git push origin main", None, "feat/x"),
        (apos + " && git push origin main", B, "feat/x"),
        (apos + " && gh pr merge 5 --admin", B, "feat/x"),
        (apos + " && gh pr merge 5 --squash", G, "feat/x"),
        (apos + "\ngit push origin main", B, "feat/x"),
        (apos + " && git push -u origin HEAD", None, "feat/x"),
        ("gh pr comment 5 --body \"$(cat <<'EOF'\nIt's ready (really)\nEOF\n)\" && gh pr merge 5 --admin", B, "feat/x"),
        ("echo $((1+2)); x=$(( (1+2)*3 )); [[ -n $x ]] && echo ok", None, "feat/x"),
        ("gh api repos/o/r/pulls/5/reviews --input r.json", B, "feat/x"),
    ]
    bad = 0
    for cmd, want, br in cases:
        got = decide(cmd, online=False, branch=br)
        ok = (got is None) if want is None else (got == want if want == G else got is not None and got != G)
        if not ok or "-v" in sys.argv:
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
        if re.search(r"\bgh\b.*\b(merge|review|api)\b|\bgit\b.*\bpush\b", cmd, re.S):
            reason = "Merge/push gate could not evaluate this command (%s: %s); blocked to be safe." % (type(e).__name__, e)
        else:
            return 0
    if reason:
        sys.stderr.write(reason + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
