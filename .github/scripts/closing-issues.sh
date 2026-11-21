#!/usr/bin/env bash
# Print the issue numbers a pull request body closes.
#
# usage: closing-issues.sh < pr-body
#
# One number per line, in order of first appearance, each once; exit 0,
# also when there are none. HTML comments are ignored. A match is one of
# GitHub's closing keywords (close, closes, closed, fix, fixes, fixed,
# resolve, resolves, resolved; any case; optional colon) followed by
# spaces or tabs and #N. Other forms (URLs, owner/repo#N) don't match.
# Exit 2: any argument. Reads no environment variable.
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "::error::usage: closing-issues.sh < pr-body" >&2
  exit 2
fi

perl -0777 -ne '
  s/<!--.*?(?:-->|\z)//gs;
  my %seen;
  while (/(?<![A-Za-z0-9_])(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?):?[ \t]+#([0-9]+)(?![0-9A-Za-z_])/gi) {
    print "$1\n" unless $seen{$1}++;
  }
'
