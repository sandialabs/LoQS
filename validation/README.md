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

The full run (7,752 cases) takes about 55 to 70 minutes on 4 workers (`-n 4`,
which `-n auto` gives on a 4-core machine) in an 8 GB container.

Quick run, leaving out the many-qubit (`big`) configs; it writes no record for
a codepack that has `big` cases, because not every case ran:

```bash
python -m pytest validation -m "not big" -n auto --timeout 1800
```

Select with `-k` on a case id (see "Reading a case id" below).

Every run ends with a verdict table per codepack, one row per class that ran
a case:

```
steane        cases  passed  known  failed  seconds
------------  -----  ------  -----  ------  -------
truth         16     16      0      0       3.4
ft            4      4       0      0       0.9
ft faults: 2 gadgets, 42 distinct single faults, 84 runs, 0 discarded, all corrected
```

- `cases`: cases that ran. `passed`: cases that failed no check and declare
  none. `known`: cases that failed exactly the checks they declare. `failed`:
  every other outcome, including an error or a skip.
- `seconds`: the class's case durations summed, so with `-n` it exceeds wall
  time. `--` means no duration was reported.
- A line under the table names each class's known bugs with their case
  counts.
- The `ft faults` line counts what the fault-tolerance cases injected. A gadget
  is a declared sweep in a config; a distinct single fault is one fault
  location and Pauli of that sweep's target (a weight-2 Pauli after a
  two-qubit gate under `post_twoq`); a run is one fault injected into one
  case's program, so a gadget's faults repeat across its cases. `discarded` is
  runs the sweep dropped. It ends `all corrected` when no run failed, else the
  count of uncorrected runs.

## What each class shows

| Class | Runs | A pass shows | Bugs caught | Mutations the self-test catches |
|---|---|---|---|---|
| `static` | Checks on the built code and model: completeness, labels, model coverage, serialization, idle padding | The declaration and the code agree on keys, data qubits and layout, and the code survives a round trip | `steane-3` to `steane-5`, `five-8`, `five-9`, `five-11`, `five-13`, `surface-5` | none |
| `truth` | Each declared word on the backends, checked against the logical script's measured outcomes (`truth`, `mp-truth`) and between backends (`agreement`) | The measured logical outcome matches the ideal one, and the backends agree | `five-5` (`agreement`), `five-12`, `surface-1`, `surface-2`, `surface-3`, `surface-4` | `s-sdag-swapped`, `x-wrong-support` |
| `tracking` | Words with injected errors, the oracle observing between steps (`gate`, `idle`, `reuse`, `word`, `mp-track`, `mp-reuse`) | The tracked Pauli frame, syndrome and logical state stay right under errors | `steane-1`, `steane-2`, `five-1` to `five-5`, `five-10`, `surface-1`, `surface-3` (O4), `surface-4`, `surface-6`, `surface-7` | `s-bare-circuit`, `h-no-syndrome-swap`, `x-measure-uses-h`, `cx-frame-only`, `surface-cnot-xor-dropped` |
| `independence` | Two logical words on two patches with an error on one | An error on one patch leaves the other patch's results alone | `steane-2`, `five-8`, `surface-1`, `surface-3` | none |
| `ft` | Every single fault of a declared sweep, injected into a gadget between a prep and a measure | One fault anywhere in the gadget is corrected | none yet: `five-6` and `five-7` were found by reading and research runs, and the 5-qubit sweeps that would check them aren't declared yet | `s-bare-circuit` (one `ft` case) |

The static class has these families:

- `completeness`: built keys equal declared keys, every public multipatch
  builder is declared, and the `data-qubits` check that the declaration's
  `data_qubits` equal the code's `template_data_qubits` when it names any.
- `model-coverage`, `labels` and `serialization`: the ideal model covers the
  walked gates, labels can be renamed, and the code survives a round trip.
- `idle-padding`: under `include_idles`, every layer of every circuit leaf
  has an op on each of the circuit's qubits.

## The checks

A case's result is the set of checks it fails:

- `outcome`: the measured logical outcome differs from the logical script's.
- `data-qubits`: the declaration's `data_qubits` differ from the code's
  `template_data_qubits` (static, when the declaration names any).
- O1: every declared stabilizer has expectation +1 or -1.
- O2: the codepack's stored syndrome equals the true stabilizer bits.
- O3: the logical X, Y and Z expectations, corrected by the tracked logical
  Pauli, equal the script's.
- O4: the tracked state re-read from the frame's `patches` after the run
  equals the snapshot taken during it.

O1 and O2 apply to a patch from the first QEC round after its prep until its
measure, and O3 from its prep until its measure, or until a random measure
makes later expectations depend on the drawn outcome (`plan_word`'s
docstring in `engine.py`). O3 is skipped for a patch that took an injected
error under a deferred decoder, and no observation is placed while an
injected error has not yet met a QEC round on its patch or a patch an op
coupled it to. O4 applies after the run.

## Reading a case id

`<codepack>.<config>.<class>.<family>.<detail>`; a `-k` expression such as
`steane and independence` selects by any part. Families by class:

- `static`: `completeness`, `labels`, `model-coverage`, `serialization`,
  `idle-padding`.
- `truth`: `truth`, `mp-truth`, `agreement`.
- `tracking`: `gate`, `idle`, `reuse`, `word`, `mp-track`, `mp-reuse`.
- `independence`: `independence`.
- `ft`: `ft`.

One example per class, with the detail's fields:

| Class | Example id | Detail |
|---|---|---|
| `static` | `steane.default.static.serialization.hdf5` | the format |
| `truth` | `steane.default.truth.truth.FT_Zero_Prep-H+H-FT_Logical_X_Measure-Adaptive_QEC` | prep, gate word (`+` joins gates), measure, QEC key |
| `tracking` | `steane.default.tracking.gate.FT_Zero_Prep-H-H-egq-Adaptive_QEC-FT_Logical_Z_Measure` | prep, input gates, tested gate, order (`egq`: error, gate, QEC; `eqgq` adds a QEC after the error), QEC key, measure |
| `independence` | `steane.default.independence.independence.errL0-FT_Zero_Prep+H+H-FT_Zero_Prep+X-blocks` | the patch with the error, each patch's prep and gate (`prep+gate`), schedule (`blocks` or `interleaved`) |
| `ft` | `steane.default.ft.ft.check-S1-S5-S6-FT_Zero_Prep+H-FT_Logical_X_Measure-none-clean` | sweep, input (prep and gates), measure, suffix word, seed (`clean` or `tracked`) |

## Where a codepack lives

A codepack migrated to the class API has five parts:

| Part | Path | Holds |
|---|---|---|
| Base class | `loqs/codepacks/base.py` | `BaseCodepack`, the gate-name walker and the shared `create_ideal_model()` |
| Codepack | `loqs/codepacks/<name>/` | the subclass: options in `__init__`, the code in `create_qec_code()` |
| Old entry points | `loqs/codepacks/codepack_<name>.py` | deprecated module-level wrappers that call the class |
| Declaration | `validation/declarations/<name>.py` | what the code claims: data qubits, stabilizers, logical operators, each logical gate's Clifford action, preps, measures, QEC keys, internal keys, sweeps and known failures |
| Record | `validation/records/<name>.json` | a hash of the declaration's `files`, the class summary and the known bugs of the last full run (not every case result) |

A generated codepack (one built from a spec) has these parts instead:

| Part | Path | Holds |
|---|---|---|
| Spec | `loqs/codepacks/<name>/spec.py` | the code's facts, circuits and declared gates, with each element's source |
| Custom apply functions | `loqs/codepacks/<name>/custom_apply_fns.py` | the hand-written apply functions the spec names |
| Artifact | `loqs/codepacks/<name>/_generated.py` | the tables derived from the spec; regenerated, never edited |
| Class | `loqs/codepacks/<name>/__init__.py` | a `GeneratedCodepack` subclass setting `spec` and `artifact` |
| Old entry points | `loqs/codepacks/codepack_<name>.py` | deprecated module-level wrappers that call the class |
| Declaration | `validation/declarations/<name>.py` | as above |
| Record | `validation/records/<name>.json` | as above (a class summary and known bugs, not every case result); its `rendering` fingerprint covers the generator's output, so the generator's files need no record |

`docs/codepacks/<name>.md` is the spec rendered as a document, regenerated
with the artifact by `python -m loqs.codepacks.generator`.

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
whole case set ran and every case passed `judge`, and its files and rendering
are unchanged since the run started (editing a file mid-run, or a change to
what the codepack renders, refuses the record). It stores:

- `files`: the digest of each of the codepack's files and its declaration
  module.
- `rendering`: a fingerprint of what the declaration's configs and declared
  multipatch ops build, on the List backend: each code's instructions, data
  and apply-function source (docstrings and comments excluded, same-module
  helpers and the globals a function names included). It replaces the
  generator modules' file hashes, so files under `loqs/codepacks/generator/`
  need no record.
- `classes`: for each class, a block with `cases`, `passed`, `known`,
  `failed` (always 0 in a record), `by_bug` (cases declaring each bug id),
  `seconds` (summed case durations, `null` if unrecorded), and for `ft`,
  `faults` (`gadgets`, `distinct`, `runs`, `discarded`, `uncorrected`, or
  `null`). A class with no case is `"not declared"`.
- `bugs`: each known bug a case lists, with its summary stored once.

There is no per-case list. A record exists only when every case passed as
declared, so each case's result follows from the declaration (hashed in
`files`) and the case generator (hashed in `harness`).

To compare two records across a move (a codepack fix or refactor), results
are identical when each class has the same `cases`, every bug the move
doesn't fix keeps its `by_bug` count, the fixed bugs drop out of `by_bug`,
and `passed` plus `known` is unchanged.

The default-suite test `tests/codepacks/test_validation_records.py` fails when
any digest or the fingerprint no longer matches (the fingerprint test skips
without Stim and pyGSTi), or when a file under `loqs/codepacks/` other than
the generator's is in no record, so changing a codepack or its declaration
requires re-running the harness and committing the refreshed record. The
harness's own digests are recorded but not enforced. CRLF line endings are
normalized, so digests agree across platforms.
