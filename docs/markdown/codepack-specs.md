# Reading a codepack spec

A generated codepack is built from a *spec*: a Python data structure
([](api:loqs.codepacks.generator.schema.CodeSpec)) that holds a code's facts,
its circuits and the logical gates it declares. A generator reads the spec,
checks it with Stim, derives the tables that track the logical frame and the
syndrome through each gate, and writes them to a committed `_generated.py`.
The class, such as [](api:SteaneCodepack), builds its `QECCode` from that file.

This page explains what each part of a spec means, how to read the generated
file (frame tracking above all), and how to regenerate both. The
[Steane page](../codepacks/steane.md) is a spec rendered as a document; keep it
open beside this one.

## What a spec holds

Every element carries a `src` ([](api:loqs.codepacks.generator.schema.Src))
saying where it comes from; see [Provenance](#provenance).

| Element | Meaning |
|---|---|
| [](api:loqs.codepacks.generator.schema.Facts) | The data and auxiliary qubits, the stabilizers and logicals as Pauli strings over the data qubits, and the tracking conventions: the order of the logical frame's bits (`frame_layout`, Steane: `Z`, `X`) and the names of the patch data that hold the frame and the syndrome. |
| [](api:loqs.codepacks.generator.schema.Circuit) | A literal circuit. |
| [](api:loqs.codepacks.generator.schema.CheckCircuit) | The circuit that measures one stabilizer with one auxiliary qubit, written out from the stabilizer. |
| [](api:loqs.codepacks.generator.schema.Parallel) | Circuits merged layer by layer. |
| [](api:loqs.codepacks.generator.schema.Gate) | A logical gate: a physical circuit plus the logical Clifford it declares. |
| [](api:loqs.codepacks.generator.schema.RepeatUntilSuccess) | A reset and a body, repeated until the expected outcomes are read. |
| [](api:loqs.codepacks.generator.schema.Composite) | Instructions run in order. |
| [](api:loqs.codepacks.generator.schema.CustomApplyFn) | An instruction the generator can't derive, such as the decoder, which names a hand-written apply function. Its `branches` list what it pushes onto the instruction stack under each condition, which the page draws as a flowchart beside the function's source. |
| [](api:loqs.codepacks.generator.schema.Role) | What the validation harness reads an instruction as: a prep, a measurement or a QEC round. |
| [](api:loqs.codepacks.generator.schema.LookupTable) | A decoding table from the literature, declared literally. |
| [](api:loqs.codepacks.generator.schema.TwoPatchGate) | A gate across two patches of the code, such as the transversal CX. |

## The circuit grammar

A circuit is text, one layer per non-blank line. A layer is a list of
operations separated by `;`, and an operation is a gate name followed by its
targets, grouped by the gate's arity as Stim does: `Gcnot c1 t1 c2 t2` is two
CNOTs. A `#` starts a comment, and a line `include KEY` splices in the layers
of another circuit of the spec. For example, a layer of Hadamards on the
auxiliary qubits followed by CNOTs from them looks like:

```text
Gh A0 A1 A2
Gcnot A0 D0 A1 D1
```

Gate names come from the generator's gate table, which maps each LoQS gate name
to a Stim gate. `Imrz` is the measure-and-reset, and the page's figures draw it
as `MR`. In a two-patch gate, qubits are named `role.qubit`, such as
`ctrl.D0`.

## Provenance

- **Paper**: the element is taken from a paper. Its cite names the figure,
  table or section, and the paper is cited by its BibTeX key as
  `[@ryananderson_realizing_2021]`, which the docs site renders from
  `docs/references.bib` as a bibliography entry. Its `figures` name crops of
  the paper's figures, tables or pseudocode in the page's `paper/` folder,
  which the page shows under **Paper** beside the **LoQS** circuit,
  flowchart, rows or source, with the spec's `figure_credit`.
- **Ours**: the element is our own choice, and its reason says why (for
  example, which auxiliary qubit each unflagged check uses, which the paper's
  figure leaves open).
- **Derived**: the generator computed the statement with Stim from the circuits
  and the declared logical action; nothing in the spec states it.

## How to read `_generated.py`

The file's header repeats this summary. `GENERATED["tables"]` holds:

- `gates[key]`: a one-patch logical gate. `logical` is its declared action,
  `frame_map` and `syndrome_map` are matrices over GF(2) that update the logical
  frame and the stored syndrome, and `per_qubit` is the rule each data qubit's
  Pauli frame entry follows. `trivial` means both maps are the identity, so no
  bookkeeping step runs.
- `syndrome_index[circuit]`: for each auxiliary qubit a check circuit measures,
  the index of the stabilizer its reading gives.
- `support`, `logical_support`: the data-qubit indices each stabilizer and
  logical operator acts on.
- `lut`: the lookup tables, as literal rows.
- `two_patch[key]`: a gate on two patches, with joint maps and its circuit.

### Frame tracking

A logical Pauli frame is a record of the Pauli correction that has not been
applied, held as two bits in `frame_layout` order. For Steane, the frame is
`[Z, X]`: the pair `[0, 1]` means a pending logical `X`. The stored syndrome
(`latest_syndrome`) is one bit per stabilizer, `S1` to `S6`, the last
measured value of each check. A Clifford gate moves Paulis around, so after the
gate both records must be rewritten as the Pauli they now stand for. Each map
acts as `new = map @ old (mod 2)`: row *i* lists which old bits are XORed into
new bit *i*.

**Hadamard.** `H` swaps `X` and `Z`, so its `frame_map` is `[[0, 1], [1, 0]]`:
the new `Z` bit is the old `X` bit and the new `X` bit is the old `Z` bit. The
transversal `H` also swaps each X-type check with the Z-type check on the same
qubits, so its `syndrome_map` sends `S1`, `S2`, `S3` to `S4`, `S5`, `S6` and
back: new `S1` = old `S4`, new `S4` = old `S1`, and so on. On the data qubits,
`per_qubit` maps `X` to `Z` and `Z` to `X` (`Y` and `I` stay).

**Phase gate.** `S` maps `X` to `Y` and leaves `Z`, so its `frame_map` is
`[[1, 1], [0, 1]]`: the new `Z` bit is old `Z` XOR old `X` (an `X` becomes
`Y`, which has both bits), and the new `X` bit is the old `X` bit. A pending
`Z` is not changed. Each X-type check picks up its Z-type partner: new `S1` =
old `S1` XOR old `S4`, and likewise for `S2`, `S3`, while `S4` to `S6` are
unchanged. `per_qubit` maps `X` to `Y` and `Y` to `X`. Frames track Paulis up
to sign, so `S` and `Sdag` share one rule.

The generator derives each map, checks that the circuit preserves the code, and
refuses the spec when the circuit's action differs from the declared `logical`.

**Reading the support tables.** `support[S1]` lists the data qubits (by index in
the spec's `data` order) that check `S1` acts on, and `syndrome_index[circuit]`
says which stabilizer each auxiliary qubit of a check circuit reports. The
decoder uses both to turn measured bits into syndrome entries. `logical_support`
is the same for the logical operators, used to read a logical measurement.
`lut` holds the decoder's tables: each row of a `data` table is a syndrome
change that flips the logical, and each row of a `flag` table is a hook pattern
(a flag change and a syndrome change) that alters the data table's correction.

**The CX.** A two-patch gate's maps act on both patches' bits concatenated in
`roles` order (`ctrl` first, then `tgt`). The transversal CX's `frame_map` has
the new `ctrl.Z` as `ctrl.Z` XOR `tgt.Z`, and the new `tgt.X` as `ctrl.X` XOR
`tgt.X`: an `X` spreads from control to target, a `Z` from target to control. Its
`syndrome_map` couples the checks the other way round: the control's X-type
checks pick up the target's (`ctrl.S1` = `ctrl.S1` XOR `tgt.S1`, and so on for
`S2` and `S3`) and the target's Z-type checks pick up the control's (`tgt.S4` to
`tgt.S6`). A syndrome bit is the eigenvalue of a fixed measurement, so it follows
the stabilizer group across the gate, which is the opposite pairing to the
frame's. `qubit_frame_map` acts on the per-qubit frames as every `x` bit
followed by every `z` bit. The page's "Derived joint tracking updates" lists
the non-trivial rows.

## Regenerating

One command writes both the artifact and the document:

```text
python -m loqs.codepacks.generator loqs.codepacks.steane:SteaneCodepack
```

It rewrites `_generated.py` and `docs/codepacks/steane.md` with its figures
in `docs/codepacks/steane/`. Commit both together: the test suite fails when
either is stale. The document is written only for the codepacks LoQS ships, and
only from a source checkout.

A codepack of your own works without a lock while you develop it: a subclass
with `locked = False` derives the tables at construction (see
[](api:GeneratedCodepack)), and the same command on your class writes its
artifact when you want to lock it.
