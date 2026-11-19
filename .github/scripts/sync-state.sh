#!/usr/bin/env bash
# Print what syncing <branch-ref> into <main-ref> would take.
#
# usage: sync-state.sh <main-ref> <branch-ref>
#
# Prints one word on stdout and exits 0: "nothing" when <branch-ref> has
# no commit <main-ref> lacks, else "clean" when merging it into
# <main-ref> has no conflict, else "conflict". Exit 2, with nothing on
# stdout: cannot decide (bad arguments, an unresolvable ref, unrelated
# histories, or a git older than 2.38). Reads no environment variable,
# fetches nothing and writes nothing to refs or the worktree.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "::error::usage: sync-state.sh <main-ref> <branch-ref>" >&2
  exit 2
fi
main_ref=$1
branch_ref=$2

for ref in "$main_ref" "$branch_ref"; do
  if ! git rev-parse --verify --quiet "${ref}^{commit}" > /dev/null; then
    echo "::error::'$ref' does not resolve to a commit; cannot decide the sync state (is the checkout shallow?)" >&2
    exit 2
  fi
done

if ! ahead=$(git rev-list --count "$main_ref..$branch_ref"); then
  echo "::error::git rev-list failed for '$main_ref..$branch_ref'; cannot decide the sync state" >&2
  exit 2
fi
if [ "$ahead" -eq 0 ]; then
  echo nothing
  exit 0
fi

rc=0
git merge-tree --write-tree "$main_ref" "$branch_ref" > /dev/null 2>&1 || rc=$?
case "$rc" in
  0) echo clean ;;
  1) echo conflict ;;
  *)
    echo "::error::git merge-tree failed with status $rc for '$main_ref' and '$branch_ref' (unrelated histories, or git older than 2.38); cannot decide the sync state" >&2
    exit 2
    ;;
esac
