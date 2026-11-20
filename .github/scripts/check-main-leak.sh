#!/usr/bin/env bash
# Fail when <head> carries commits of <main-ref> that <base-ref> lacks.
#
# usage: check-main-leak.sh <main-ref> <base-ref> <head>
#
# Every merge base of <main-ref> and <head> must already be reachable
# from <base-ref>. Exit 0: no leak. Exit 1: leak. Exit 2: cannot decide
# (bad arguments, an unresolvable ref, or no merge base). Reads no
# environment variable and fetches nothing.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "::error::usage: check-main-leak.sh <main-ref> <base-ref> <head>" >&2
  exit 2
fi
main_ref=$1
base_ref=$2
head=$3

for ref in "$main_ref" "$base_ref" "$head"; do
  if ! git rev-parse --verify --quiet "${ref}^{commit}" > /dev/null; then
    echo "::error::'$ref' does not resolve to a commit; cannot check for main's commits (is the checkout shallow?)" >&2
    exit 2
  fi
done

if ! merge_bases=$(git merge-base --all "$main_ref" "$head"); then
  echo "::error::no merge base between '$main_ref' and '$head'; cannot check for main's commits" >&2
  exit 2
fi
if [ -z "$merge_bases" ]; then
  echo "::error::no merge base between '$main_ref' and '$head'; cannot check for main's commits" >&2
  exit 2
fi

for mb in $merge_bases; do
  if ! git merge-base --is-ancestor "$mb" "$base_ref"; then
    echo "::error::This branch carries commits from main that '$base_ref' lacks (merge base $mb). Rebase your fix commits onto '$base_ref', or cherry-pick them onto a branch cut from it. Do not merge." >&2
    exit 1
  fi
done

echo "OK: every merge base of '$main_ref' and the head is already in '$base_ref'."
