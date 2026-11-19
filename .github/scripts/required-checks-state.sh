#!/usr/bin/env bash
# Print whether a commit's required checks have passed.
#
# usage: required-checks-state.sh <rules-json> <check-runs-json>
#
# <rules-json> holds the response of GET /repos/{owner}/{repo}/rules/branches/{branch};
# <check-runs-json> holds one or more responses of GET
# /repos/{owner}/{repo}/commits/{ref}/check-runs, as `gh api --paginate`
# prints them. Prints one word on stdout and exits 0: "red" when a
# required check failed, else "pending" when one has no run yet, only
# skipped runs or an unfinished newest run, else "green". One
# "<state> <context>" line per required check goes to stderr. Exit 2, with
# nothing on stdout: cannot decide (bad arguments, a file that is missing
# or not that JSON, or rules requiring no status check). Reads no
# environment variable, calls no API and writes nothing.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "::error::usage: required-checks-state.sh <rules-json> <check-runs-json>" >&2
  exit 2
fi

if ! states=$(jq -rn --slurpfile rules "$1" --slurpfile pages "$2" '
  [$rules[] | arrays | .[] | select(.type == "required_status_checks")
   | .parameters.required_status_checks[]] | unique as $required
  | [$pages[] | .check_runs[]] as $runs
  | if ($required | length) == 0 then error("no required check") else . end
  | $required[]
  | . as $r
  | [$runs[]
     | select(.name == $r.context and .conclusion != "skipped"
              and ($r.integration_id == null or .app.id == $r.integration_id))]
  | (max_by(.id)) as $last
  | (if $last == null or $last.status != "completed" then "pending"
     elif $last.conclusion == "success" then "green"
     else "red" end) as $state
  | "\($state) \($r.context)"
'); then
  echo "::error::could not read the required checks from '$1' and the check runs from '$2' (no required check, or not the API's JSON)" >&2
  exit 2
fi

echo "$states" >&2
case $'\n'"$states" in
  *$'\n'red\ *) echo red ;;
  *$'\n'pending\ *) echo pending ;;
  *) echo green ;;
esac
