#!/usr/bin/env bash
# Print the name of the bugfix branch that follows a release tag.
#
# usage: next-bugfix-branch.sh <tag>
#
# A plain tag is vX.Y or vX.Y.Z (each number 0 or up to nine digits with
# no leading zero); X.Y is read as X.Y.0. For it, print
# bugfix-release-vX.Y.(Z+1) and exit 0. Exit 1, printing nothing: the tag
# isn't plain. Exit 2: bad arguments. Reads no environment variable and
# touches neither git nor the network.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "::error::usage: next-bugfix-branch.sh <tag>" >&2
  exit 2
fi
tag=$1

num='(0|[1-9][0-9]{0,8})'
if [[ "$tag" =~ ^v${num}\.${num}(\.${num})?$ ]]; then
  major=${BASH_REMATCH[1]}
  minor=${BASH_REMATCH[2]}
  patch=${BASH_REMATCH[4]:-0}
  echo "bugfix-release-v${major}.${minor}.$((patch + 1))"
else
  exit 1
fi
