#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Codepack spec schema: pure, canonical data and a circuit grammar.

A spec is a tree of frozen dataclasses holding only str, int, bool, None
and tuples of those. Circuits are text with one layer per line.
Hand-written apply functions are named by `module:function`. Every element carries a `Src` tag:
`paper` (verbatim, with its citation) or `ours` (with a reason). This
module needs neither Stim nor `loqs.core`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from loqs.codepacks.generator.gates import GATES


class SpecError(ValueError):
    """A spec is wrong: a malformed circuit, a bad reference or a failed
    structural check. The message names the element and the cause."""


# --- provenance -------------------------------------------------------------


@dataclass(frozen=True)
class Src:
    """Where a spec element comes from: "paper" with its citation, or
    "ours" with a reason.

    Text may cite a paper by its BibTeX key as `[@key]`, which the docs
    site renders from `docs/references.bib`. `figures` name images of the
    cited figures, files in the spec document's `paper/` folder, drawn
    beside the element so a reader can compare the two.
    """

    kind: str
    cite: str = ""
    reason: str = ""
    figures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("paper", "ours"):
            raise SpecError(f"Src kind {self.kind!r}")
        if self.kind == "paper" and not self.cite:
            raise SpecError("a paper item needs its citation")
        if self.kind == "ours" and not self.reason:
            raise SpecError("an item of ours needs a reason")


def paper(cite: str, reason: str = "", figures: tuple[str, ...] = ()) -> Src:
    """An element taken from a paper, cited by figure, table or section."""
    return Src("paper", cite, reason, figures)


def ours(reason: str, cite: str = "", figures: tuple[str, ...] = ()) -> Src:
    """An element of our own choosing, with the reason."""
    return Src("ours", cite, reason, figures)


# --- facts ------------------------------------------------------------------


@dataclass(frozen=True)
class Tracking:
    """Where a codepack keeps its tracking data.

    `logical_frame` is the patch-data key of the logical Pauli frame, and
    `syndrome` the key holding one stored bit per stabilizer.
    """

    logical_frame: str = "logical_pauli_frame"
    syndrome: tuple[str, ...] = ("latest_syndrome",)


@dataclass(frozen=True)
class Facts:
    """The code's facts. Pauli strings are over `data`, in its order.

    `logicals` is a tuple of (X_L, Z_L) pairs. `frame_layout` and the
    syndrome order are conventions of the tracking data: the logical frame
    holds bits in `frame_layout` order (Steane: `[Z_bit, X_bit]`), and the
    stored syndrome one bit per stabilizer in `stabilizers` order.
    """

    data: tuple[str, ...]
    aux: tuple[str, ...]
    stabilizer_names: tuple[str, ...]
    stabilizers: tuple[str, ...]
    logicals: tuple[tuple[str, str], ...]
    src: Src
    frame_layout: tuple[str, ...] = ("Z", "X")
    conventions_src: Src | None = None
    tracking: Tracking = Tracking()

    @property
    def qubits(self) -> tuple[str, ...]:
        """Every template qubit: the auxiliary qubits, then the data."""
        return self.aux + self.data


# --- items ------------------------------------------------------------------


@dataclass(frozen=True)
class Circuit:
    """A literal circuit. `include KEY` splices another circuit's layers."""

    key: str
    text: str
    src: Src
    name: str = ""


@dataclass(frozen=True)
class CheckCircuit:
    """A one-aux check of one stabilizer, data in `order` (default: the
    stabilizer's support in `data` order). A Z-only check is direct
    parity: `Gcnot q->aux ...; Imrz aux`. Any other check is a Hadamard
    test: `Gh aux; Gcnot aux q (X) or Gcphase aux q (Z) ...; Gh aux; Imrz
    aux`. The circuit is over every template qubit."""

    key: str
    stabilizer: str
    aux: str
    src: Src
    order: tuple[str, ...] = ()
    name: str = ""


@dataclass(frozen=True)
class Parallel:
    """Circuits merged layer by layer from layer 0 (unpadded)."""

    key: str
    parts: tuple[str, ...]
    src: Src
    name: str = ""


@dataclass(frozen=True)
class Gate:
    """A logical gate: a physical circuit and its declared logical Clifford
    (a Stim gate name).

    The generator derives the logical-frame, stored-syndrome and per-qubit
    frame maps and refuses a circuit whose action differs from `logical`.
    With identity bookkeeping the circuit renders under `key` alone;
    otherwise it renders under `circuit_key`, the bookkeeping under
    `update_key` (gates sharing one must derive the same maps) and their
    composite under `key`.
    """

    key: str
    text: str
    logical: str
    src: Src
    circuit_key: str = ""
    update_key: str = ""
    name: str = ""
    circuit_name: str = ""
    update_name: str = ""


@dataclass(frozen=True)
class RepeatUntilSuccess:
    """Reset, clear tracking, run `body` until `expected` is read.

    The reset always clears the patch's tracking data (logical frame,
    stored syndrome, every custom apply function's scratch keys and the
    per-qubit frame),
    so a re-prep can't inherit a previous run's corrections.
    """

    key: str
    reset: str
    body: tuple[str, ...]
    expected: tuple[tuple[str, tuple[int, ...]], ...]
    src: Src
    name: str = ""
    reset_name: str = ""


@dataclass(frozen=True)
class Composite:
    """Instructions run in order, by key."""

    key: str
    parts: tuple[str, ...]
    src: Src
    name: str = ""


@dataclass(frozen=True)
class CustomApplyFn:
    """A hand-written apply function, for an instruction the generator
    can't derive: `fn` is `module:function` of a module-level apply
    function. `data` is literal data; `derived` names generated tables to
    add to `Instruction.data` (`data key -> table name`). `qubit_data`
    lists the data keys holding qubit labels, so the renderer builds the
    `map_qubits_fn`. `scratch` lists patch-data keys the function writes,
    which a prep clears. `branches` lists `(condition, keys)`: the keys
    the function pushes onto the stack when the condition holds, which
    the spec document draws as a flowchart; empty keys push nothing."""

    key: str
    fn: str
    src: Src
    data: tuple[tuple[str, object], ...] = ()
    derived: tuple[tuple[str, str], ...] = ()
    qubit_data: tuple[str, ...] = ()
    param_priorities: tuple[tuple[str, tuple[str, ...]], ...] = ()
    param_aliases: tuple[tuple[str, str], ...] = ()
    scratch: tuple[str, ...] = ()
    branches: tuple[tuple[str, tuple[str, ...]], ...] = ()
    name: str = ""

    @property
    def inserts(self) -> tuple[str, ...]:
        """Every key some branch pushes, in branch order."""
        return tuple(dict.fromkeys(k for _, ks in self.branches for k in ks))


@dataclass(frozen=True)
class Role:
    """What the validation harness reads a key as: "prep" (with `state`),
    "measure" (with `basis`, `ft`) or "qec"."""

    key: str
    role: str
    state: str = ""
    basis: str = ""
    ft: bool = False
    fresh_only: bool = False


@dataclass(frozen=True)
class LookupTable:
    """A literature lookup table, declared literally and passed through.

    `kind="data"`: each row is `(syndrome_diff,)`, a syndrome change over
    one check type that flips the logical. `kind="flag"`: each row is
    `(flag_diff, syndrome_diff)`, a hook pattern that changes the data
    table's correction. `apply_fn_keys` names the custom apply functions'
    data keys the table feeds (`key.data_key`)."""

    name: str
    kind: str
    rows: tuple[tuple[tuple[int, ...], ...], ...]
    src: Src
    apply_fn_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class TwoPatchGate:
    """A gate across patches of this code, derived on the joint system and
    split into per-patch updates. `text` names qubits as `role.qubit`."""

    key: str
    roles: tuple[str, ...]
    text: str
    logical: str
    src: Src
    name: str = ""


Item = (
    Circuit
    | CheckCircuit
    | Parallel
    | Gate
    | RepeatUntilSuccess
    | Composite
    | CustomApplyFn
)


@dataclass(frozen=True)
class CodeSpec:
    """A whole code: facts, items, lookup tables, two-patch gates, roles.
    `figure_credit` is the credit the spec document gives the paper
    figures that `Src.figures` name, such as their licence."""

    name: str
    title: str
    facts: Facts
    items: tuple[Item, ...]
    tables: tuple[LookupTable, ...] = ()
    two_patch: tuple[TwoPatchGate, ...] = ()
    roles: tuple[Role, ...] = field(default=())
    figure_credit: str = ""

    def item(self, key: str) -> Item:
        """The item with this key."""
        for it in self.items:
            if it.key == key:
                return it
        raise KeyError(key)


# --- circuit grammar ----------------------------------------------------------
#
# One layer per non-blank line. A layer is `;`-separated ops; an op is a
# gate name from the gate table followed by targets, grouped by the gate's
# arity (as in Stim: `Gcnot c1 t1 c2 t2`). `#` starts a comment. A line
# `include KEY` splices the layers of circuit item KEY.

Layer = list[tuple[str, tuple[str, ...]]]


def parse_circuit(
    text: str,
    qubits: Any,
    resolve: Callable[[str], list[Layer]] | None = None,
) -> list[Layer]:
    """Parse circuit text into layers `[[(name, (q, ...)), ...], ...]`."""
    layers: list[Layer] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("include "):
            if resolve is None:
                raise SpecError(f"line {lineno}: include needs a resolver")
            layers.extend([list(layer) for layer in resolve(line[8:].strip())])
            continue
        layer: Layer = []
        for op in line.split(";"):
            toks = op.split()
            if not toks:
                raise SpecError(f"line {lineno}: empty op")
            name, targets = toks[0], toks[1:]
            if name not in GATES:
                raise SpecError(f"line {lineno}: unknown gate {name!r}")
            arity = GATES[name].arity
            if not targets or len(targets) % arity:
                raise SpecError(
                    f"line {lineno}: {name} needs targets in groups of {arity}"
                )
            for t in targets:
                if t not in qubits:
                    raise SpecError(f"line {lineno}: unknown qubit {t!r}")
            for i in range(0, len(targets), arity):
                layer.append((name, tuple(targets[i : i + arity])))
        seen = [q for _, qs in layer for q in qs]
        if len(seen) != len(set(seen)):
            raise SpecError(f"line {lineno}: a qubit is used twice")
        layers.append(layer)
    return layers


def check_layers(
    spec: CodeSpec, item: CheckCircuit
) -> tuple[list[Layer], list[str]]:
    """The layers of a `CheckCircuit` and the qubit labels it is over."""
    f = spec.facts
    stabs = dict(zip(f.stabilizer_names, f.stabilizers))
    if item.stabilizer not in stabs:
        raise SpecError(f"unknown stabilizer {item.stabilizer!r}")
    s = stabs[item.stabilizer]
    body = s.lstrip("+-")
    if s.startswith("-"):
        raise SpecError("a negative check needs a sign fix-up")
    pauli_of = dict(zip(f.data, body))
    support = [q for q, p in zip(f.data, body) if p != "I"]
    order = list(item.order) or support
    if sorted(order) != sorted(support):
        raise SpecError("order isn't the stabilizer's support")
    a = item.aux
    if a not in f.aux:
        raise SpecError(f"{a!r} isn't an auxiliary qubit")
    if set(body) <= {"I", "Z"}:
        layers: list[Layer] = [[("Gcnot", (q, a))] for q in order]
        layers.append([("Imrz", (a,))])
        return layers, list(f.qubits)
    if "Y" in body:
        raise NotImplementedError(f"{item.key}: a Y in a check")
    gate = {"X": "Gcnot", "Z": "Gcphase"}
    layers = [[("Gh", (a,))]]
    layers += [[(gate[pauli_of[q]], (a, q))] for q in order]
    layers += [[("Gh", (a,))], [("Imrz", (a,))]]
    return layers, list(f.qubits)


def resolve_circuits(
    spec: CodeSpec,
) -> dict[str, tuple[list[Layer], list[str]]]:
    """`{key: (layers, qubit labels)}` for every circuit-valued item: its
    key, a gate's circuit (under its `circuit_key` or key) and a repeat's
    reset (under `"<key> reset"`)."""
    f = spec.facts
    out: dict[str, tuple[list[Layer], list[str]]] = {}

    def resolve(key: str) -> list[Layer]:
        if key not in out:
            raise SpecError(f"unknown or later circuit {key!r}")
        return out[key][0]

    for it in spec.items:
        try:
            if isinstance(it, Circuit):
                layers = parse_circuit(it.text, f.qubits, resolve)
                out[it.key] = (layers, list(f.qubits))
            elif isinstance(it, CheckCircuit):
                out[it.key] = check_layers(spec, it)
            elif isinstance(it, Parallel):
                merged: list[Layer] = []
                for p in it.parts:
                    for i, layer in enumerate(resolve(p)):
                        if i >= len(merged):
                            merged.append([])
                        merged[i].extend(layer)
                out[it.key] = (merged, list(f.qubits))
            elif isinstance(it, Gate):
                out[it.circuit_key or it.key] = (
                    parse_circuit(it.text, f.qubits, resolve),
                    list(f.qubits),
                )
            elif isinstance(it, RepeatUntilSuccess):
                out[f"{it.key} reset"] = (
                    parse_circuit(it.reset, f.qubits, resolve),
                    list(f.qubits),
                )
        except SpecError as e:
            raise SpecError(f"{it.key}: {e}") from None
    return out


# --- canonical form and hash --------------------------------------------------

_TEXT_FIELDS = {
    ("Circuit", "text"),
    ("Gate", "text"),
    ("RepeatUntilSuccess", "reset"),
    ("TwoPatchGate", "text"),
}


def normalize_text(text: str) -> str:
    """Circuit text in normal form: comments and blank lines dropped, each
    whitespace run one space, the ops of a layer joined by `"; "`."""
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(
                "; ".join(" ".join(op.split()) for op in line.split(";"))
            )
    return "\n".join(lines)


def canonical(obj: Any) -> Any:
    """A JSON-ready canonical form: dataclasses become `{"type", fields}`,
    leaving out a field whose value equals its default and writing each
    circuit-text field in normal form."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        kind = type(obj).__name__
        out: dict[str, Any] = {"type": kind}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if f.default is not dataclasses.MISSING and value == f.default:
                continue
            if (kind, f.name) in _TEXT_FIELDS:
                value = normalize_text(value)
            out[f.name] = canonical(value)
        return out
    if isinstance(obj, (tuple, list)):
        return [canonical(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in sorted(obj.items())}
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    raise TypeError(f"spec holds a non-data value: {obj!r}")


def spec_hash(spec: CodeSpec) -> str:
    """The SHA-256 of the spec's canonical JSON."""
    blob = json.dumps(canonical(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
