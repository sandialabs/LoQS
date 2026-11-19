#!/usr/bin/env bash
# Print whether a merged sync PR landed as a real merge commit.
#
# usage: sync-merge-kind.sh <merge-commit> <head-sha>
#
# <merge-commit> is the commit GitHub recorded as the PR's merge commit,
# <head-sha> the PR's head commit as a full SHA (it need not exist in this
# clone). Prints one word on stdout and exits 0: "merge-commit" when
# <merge-commit> has exactly two parents and the second is <head-sha>,
# else "other" (a squash or a rebase). Exit 2, with nothing on stdout:
# cannot decide (bad arguments or an unresolvable <merge-commit>). Works
# in a clone of depth 2. Reads no environment variable and writes nothing.
set -euo pipefail

if [ "$#" -ne 2 ] || ! [[ $2 =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::usage: sync-merge-kind.sh <merge-commit> <full-head-sha>" >&2
  exit 2
fi
if ! parents=$(git rev-list --parents -n 1 "${1}^{commit}" 2> /dev/null); then
  echo "::error::'$1' does not resolve to a commit; cannot decide how the PR was merged" >&2
  exit 2
fi
read -r -a words <<< "$parents"
if [ "${#words[@]}" -eq 3 ] && [ "${words[2]}" = "$2" ]; then
  echo merge-commit
else
  echo other
fi
