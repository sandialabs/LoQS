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

The static class has these families:

- `completeness`: built keys equal declared keys, every public multipatch
  builder is declared, and the `data-qubits` check that the declaration's
  `data_qubits` equal the code's `template_data_qubits` when it names any.
- `model-coverage`, `labels` and `serialization`: the ideal model covers the
  walked gates, labels can be renamed, and the code survives a round trip.
- `idle-padding`: under `include_idles`, every layer of every circuit leaf
  has an op on each of the circuit's qubits.

## Where a codepack lives

A codepack migrated to the class API has five parts:

| Part | Path | Holds |
|---|---|---|
| Base class | `loqs/codepacks/base.py` | `BaseCodepack`, the gate-name walker and the shared `create_ideal_model()` |
| Codepack | `loqs/codepacks/<name>/` | the subclass: options in `__init__`, the code in `create_qec_code()` |
| Old entry points | `loqs/codepacks/codepack_<name>.py` | deprecated module-level wrappers that call the class |
| Declaration | `validation/declarations/<name>.py` | what the code claims: data qubits, stabilizers, logical operators, each logical gate's Clifford action, preps, measures, QEC keys, internal keys, sweeps and known failures |
| Record | `validation/records/<name>.json` | a hash of the declaration's `files` and every case result of the last full run |

`QECCode` holds instructions and qubits only, so the declaration is the one
place a codepack's stabilizers, logical operators and gate actions are
written down. It stays outside the package so the harness checks the class
against facts the class doesn't supply.

## Adding a declaration

1. Write `validation/declarations/<name>.py` with a `DECLARATION`
   (`CodepackDeclaration`, see `declaration.py`): files (a `BaseCodepack`
   subclass's `files` include `loqs/codepacks/base.py`), configs and
   backends, preps, measures, gates, QEC keys, and every other instruction key
   under `internal` with the reason it has no check of its own. Derive
   stabilizers, logicals and gate actions from the codepack's documentation or
   paper, never from a run.
2. Register it in `validation/declarations/__init__.py`.
3. Run the harness. The `completeness` family fails until every shipped
   instruction key and public multipatch `build_*` function is declared.

## Triaging a failure

A case passes only when the set of checks it fails (`outcome`,
`data-qubits`, `O1` to `O4`) equals the set its declaration lists in
`known_failures`. Each failure is one of:

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
