#!/usr/bin/env bash
# Team context files (AGENTS.md, "Team context files"): read and update aayush_context.md /
# disha_context.md on the team-context branch without touching the current checkout.
#
#   bash scripts/team_context.sh show <aayush|disha> [lines]       # print a file from origin/team-context
#   bash scripts/team_context.sh open <aayush|disha>               # prepare the worktree; prints the file to edit
#   bash scripts/team_context.sh push <aayush|disha> "<summary>"   # commit that one file and push it
#
# Each command is self-contained, so they can run in separate shells (agents don't keep shell state).
# The worktree lives at a fixed path: TEAM_CONTEXT_WT, default ~/.cache/dngs-team-context.
# push only ever pushes to team-context, refuses any change other than the caller's own file, and keeps the
# worktree with the commit if the push fails, so nothing is lost.
set -euo pipefail
BRANCH=team-context
cmd=${1:-}
name=${2:-}
case "$name" in aayush|disha) ;; *) echo "usage: $0 show|open|push aayush|disha [...]" >&2; exit 2 ;; esac
file="${name}_context.md"
repo=$(git rev-parse --show-toplevel)
WT=${TEAM_CONTEXT_WT:-$HOME/.cache/dngs-team-context}
EXPECTED="aayush_context.md disha_context.md "

ahead() { git -C "$WT" log --format=%H "origin/$BRANCH..HEAD"; }

case "$cmd" in
show)
  git -C "$repo" fetch -q origin "$BRANCH"
  git -C "$repo" show "origin/$BRANCH:$file" | sed -n "1,${3:-80}p"
  ;;
open)
  git -C "$repo" fetch -q origin "$BRANCH"
  if [ -e "$WT/.git" ]; then
    if [ -n "$(git -C "$WT" status --porcelain)" ] || [ -n "$(ahead)" ]; then
      echo "an earlier context update in $WT is not pushed: run '$0 push $name \"...\"' or inspect it" >&2
      exit 1
    fi
    git -C "$WT" checkout -q --detach "origin/$BRANCH"
  elif [ -e "$WT" ]; then
    echo "$WT exists but is not a git worktree: remove it or set TEAM_CONTEXT_WT" >&2
    exit 1
  else
    mkdir -p "$(dirname "$WT")"
    git -C "$repo" worktree add -q --detach "$WT" "origin/$BRANCH"
  fi
  echo "$WT/$file"
  ;;
push)
  msg=${3:-}
  [ -n "$msg" ] || { echo "push needs a one-line summary" >&2; exit 2; }
  [ -e "$WT/.git" ] || { echo "no worktree at $WT: run '$0 open $name' first" >&2; exit 1; }
  tracked=$(git -C "$WT" ls-files | sort | tr '\n' ' ')
  [ "$tracked" = "$EXPECTED" ] || { echo "$WT is not a team-context tree (tracked: $tracked)" >&2; exit 1; }
  dirty=$(git -C "$WT" status --porcelain --untracked-files=all | cut -c4- | sort -u | tr '\n' ' ')
  if [ -n "$dirty" ]; then
    [ "$dirty" = "$file " ] || { echo "only $file may change; found changes in: $dirty" >&2; exit 1; }
    git -C "$WT" add -- "$file"
    git -C "$WT" commit -q -m "context($name): $msg"
  fi
  [ -n "$(ahead)" ] || { echo "nothing to push"; exit 0; }
  changed=$(git -C "$WT" diff --name-only "origin/$BRANCH...HEAD" | sort -u | tr '\n' ' ')
  [ "$changed" = "$file " ] || { echo "unpushed commits touch more than $file: $changed" >&2; exit 1; }
  ok=0
  for _ in 1 2 3; do
    if git -C "$WT" push -q origin "HEAD:refs/heads/$BRANCH"; then ok=1; break; fi
    git -C "$WT" fetch -q origin "$BRANCH"
    if ! git -C "$WT" rebase -q "origin/$BRANCH"; then git -C "$WT" rebase --abort || true; break; fi
  done
  git -C "$WT" fetch -q origin "$BRANCH"
  if [ "$ok" != 1 ] || ! git -C "$WT" merge-base --is-ancestor HEAD "origin/$BRANCH"; then
    echo "context push FAILED; your commit is kept in $WT" >&2
    exit 1
  fi
  echo "pushed $file to $BRANCH"
  ;;
*)
  echo "usage: $0 show|open|push aayush|disha [...]" >&2
  exit 2
  ;;
esac
