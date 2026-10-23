# Codepack validation harness

A pytest tree that generates checks for each shipped codepack from one
declaration (`validation/declarations/<name>.py`). It is not part of the
default test suite (`pytest.ini` ignores it) and needs `pytest-timeout` and
`pytest-xdist`.

## Running

Everything (writes records for fully passing codepacks):

```bash
python -m pytest validation -n auto --timeout 1800 --write-records
```

The full run (7,766 cases) takes about 55 to 70 minutes on 4 workers (`-n 4`,
which `-n auto` gives on a 4-core machine) in an 8 GB container.

Quick run, leaving out the many-qubit (`big`) configs; it writes no record for
a codepack that has `big` cases, because not every case ran:

```bash
python -m pytest validation -m "not big" -n auto --timeout 1800
```

Select with `-k`; case ids are `<codepack>.<config>.<class>.<family>.<detail>`.

## Adding a declaration

1. Write `validation/declarations/<name>.py` with a `DECLARATION`
   (`CodepackDeclaration`, see `declaration.py`): files, configs and
   backends, preps, measures, gates, QEC keys, and every other instruction key
   under `internal` with the reason it has no check of its own. Derive
   stabilizers, logicals and gate actions from the codepack's documentation or
   paper, never from a run.
2. Register it in `validation/declarations/__init__.py`.
3. Run the harness. The `completeness` family fails until every shipped
   instruction key and public multipatch `build_*` function is declared.

## Triaging a failure

A case passes only when the set of checks it fails (`outcome`, `O1` to `O4`)
equals the set its declaration lists in `known_failures`. Each failure is one
of:

- A confirmed bug: add an entry to `bugs.py` (`KNOWN_BUGS`) with a minimal
  reproducer, and a `known_failures` key `<case-id pattern>:<check>` naming it.
  Keep the pattern and check as narrow as the evidence. When the bug is fixed
  the case fails as "declared check now passes" until the key is removed.
- A harness or declaration error: fix it, citing the codepack line or paper.

Never tune a declaration to make a check pass.

## Records and the stale-record test

A record `validation/records/<name>.json` is written only when the codepack's
whole case set ran and every case passed `judge`, and its files are unchanged
since the run started (editing one mid-run refuses the record). It stores the
digest of each of the codepack's files and its declaration module. The
default-suite test `tests/codepacks/test_validation_records.py` fails when any
digest no longer matches, or when a file under `loqs/codepacks/` is in no
record, so changing a codepack or its declaration requires re-running the
harness and committing the refreshed record. The harness's own digests are
recorded but not enforced. CRLF line endings are normalized, so digests agree
across platforms.
