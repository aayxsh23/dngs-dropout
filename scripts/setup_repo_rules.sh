#!/usr/bin/env bash
# One-time GitHub configuration that makes the merge gate enforceable by GitHub itself.
# Run by a repository admin AFTER the PR that adds .github/workflows/ci.yml and .github/CODEOWNERS
# is merged into main (otherwise the required checks and code owners don't exist yet).
#
#   bash scripts/setup_repo_rules.sh            # show what would be applied
#   bash scripts/setup_repo_rules.sh --apply    # apply it
#
# Applies:
#  - repo settings: squash-only merges, auto-merge allowed, delete branches after merge
#  - ruleset "main: review and CI gate" on the default branch:
#      PR required; code-owner review (= the other teammate) for paths in .github/CODEOWNERS;
#      stale approvals dismissed on push; required checks "checks" (GitHub Actions only) and
#      "agent-review"; no force-push, no deletion. Repository admins can bypass in an emergency,
#      visibly; agents are forbidden from doing so (AGENTS.md, hook blocks `gh pr merge --admin`).
set -euo pipefail
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
NAME="main: review and CI gate"
GITHUB_ACTIONS_APP_ID=15368

RULESET=$(cat <<EOF
{
  "name": "$NAME",
  "target": "branch",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
  "bypass_actors": [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}],
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {"type": "pull_request", "parameters": {
      "required_approving_review_count": 0,
      "dismiss_stale_reviews_on_push": true,
      "require_code_owner_review": true,
      "require_last_push_approval": false,
      "required_review_thread_resolution": false}},
    {"type": "required_status_checks", "parameters": {
      "strict_required_status_checks_policy": false,
      "required_status_checks": [
        {"context": "checks", "integration_id": $GITHUB_ACTIONS_APP_ID},
        {"context": "agent-review"}]}}
  ]
}
EOF
)

git cat-file -e origin/main:.github/workflows/ci.yml 2>/dev/null || { echo "origin/main has no .github/workflows/ci.yml yet: merge the gate PR first"; exit 1; }
git cat-file -e origin/main:.github/CODEOWNERS 2>/dev/null || { echo "origin/main has no .github/CODEOWNERS yet: merge the gate PR first"; exit 1; }

echo "Repository: $REPO"
echo "Settings : allow_squash_merge=true allow_merge_commit=false allow_rebase_merge=false allow_auto_merge=true delete_branch_on_merge=true"
echo "Ruleset  :"; echo "$RULESET"
if [ "${1:-}" != "--apply" ]; then echo; echo "Dry run. Re-run with --apply to change GitHub."; exit 0; fi

gh api -X PATCH "repos/$REPO" -F allow_squash_merge=true -F allow_merge_commit=false -F allow_rebase_merge=false \
  -F allow_auto_merge=true -F delete_branch_on_merge=true --jq '{allow_auto_merge,delete_branch_on_merge,allow_squash_merge}'
ID=$(gh api "repos/$REPO/rulesets" --jq ".[] | select(.name == \"$NAME\") | .id")
if [ -n "$ID" ]; then
  echo "$RULESET" | gh api -X PUT "repos/$REPO/rulesets/$ID" --input - --jq '{id,name,enforcement}'
else
  echo "$RULESET" | gh api -X POST "repos/$REPO/rulesets" --input - --jq '{id,name,enforcement}'
fi
echo "Done. Check it under Settings > Rules > Rulesets."
