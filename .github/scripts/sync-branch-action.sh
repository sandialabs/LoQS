#!/usr/bin/env bash
# Print what a sync run does to the sync/ branch that heads the sync PR.
#
# usage: sync-branch-action.sh <branch-ref> <sync-ref> <pr-open>
#
# <branch-ref> is the bugfix branch, <sync-ref> its sync/ branch (it may
# not exist), <pr-open> is "true" when a sync PR is open, else "false".
# Prints one word on stdout and exits 0:
#   set           no PR is open: the sync branch is created at, or reset
#                 to, the bugfix tip, whatever it holds
#   keep          a PR is open and its branch already holds the bugfix tip
#   fast-forward  a PR is open and its branch is behind the bugfix tip
#   merge         a PR is open, its branch has commits of its own, and the
#                 bugfix branch merges into it without conflict
#   conflict      as merge, but the merge conflicts
# Exit 2, with nothing on stdout: cannot decide (bad arguments, an
# unresolvable <branch-ref>, a PR open without its sync branch, or
# sync-state.sh could not decide). Reads no environment variable, fetches
# nothing and writes nothing to refs or the worktree.
set -euo pipefail

if [ "$#" -ne 3 ] || { [ "$3" != true ] && [ "$3" != false ]; }; then
  echo "::error::usage: sync-branch-action.sh <branch-ref> <sync-ref> <true|false>" >&2
  exit 2
fi
branch_ref=$1
sync_ref=$2
pr_open=$3

if ! tip=$(git rev-parse --verify --quiet "${branch_ref}^{commit}"); then
  echo "::error::'$branch_ref' does not resolve to a commit; cannot decide the sync branch's action" >&2
  exit 2
fi
if [ "$pr_open" = false ]; then
  echo set
  exit 0
fi
if ! head=$(git rev-parse --verify --quiet "${sync_ref}^{commit}"); then
  echo "::error::a sync PR is open but '$sync_ref' does not resolve to a commit; cannot decide the sync branch's action" >&2
  exit 2
fi

if git merge-base --is-ancestor "$tip" "$head"; then
  echo keep
elif git merge-base --is-ancestor "$head" "$tip"; then
  echo fast-forward
else
  state=$(bash "$(dirname "$0")/sync-state.sh" "$head" "$tip") || exit 2
  case "$state" in
    clean) echo merge ;;
    conflict) echo conflict ;;
    *) exit 2 ;;
  esac
fi
