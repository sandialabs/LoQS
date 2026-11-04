#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""The generator: a spec to derived tables and structural-check results.

`generate(spec)` runs every structural check and returns JSON-ready
tables. One of two modules that import Stim, so a locked codepack builds
without it.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import stim

from loqs.codepacks.generator import derive
from loqs.codepacks.generator.codepack import GeneratedCodepack, tables_hash
from loqs.codepacks.generator.schema import (
    CodeSpec,
    Circuit,
    Gate,
    RepeatUntilSuccess,
    SpecError,
    TwoPatchGate,
    parse_circuit,
    resolve_circuits,
    spec_hash,
)

GENERATOR_VERSION = 1


def rendered_keys(spec: CodeSpec) -> dict[str, tuple[str, str]]:
    """`{rendered key: (spec element key, part)}` for every instruction key
    the renderer writes. Two elements claiming one key raise `SpecError`;
    only gates may share an `update_key`."""
    out: dict[str, tuple[str, str]] = {}

    def add(key: str, owner: str, part: str) -> None:
        if key in out:
            if part == out[key][1] == "gate bookkeeping":
                return
            raise SpecError(
                f"key {key!r} is claimed by both {out[key][0]!r} and {owner!r}"
            )
        out[key] = (owner, part)

    for it in spec.items:
        add(it.key, it.key, type(it).__name__)
        if isinstance(it, Gate):
            if it.circuit_key:
                add(it.circuit_key, it.key, "gate circuit")
            if it.update_key:
                add(it.update_key, it.key, "gate bookkeeping")
    return out


def _ints(a: Any) -> list[list[int]]:
    return [[int(x) for x in row] for row in np.asarray(a)]


def _is_identity(m: Any) -> bool:
    m = np.asarray(m)
    return bool(
        m.shape[0] == m.shape[1] and (m == np.eye(len(m), dtype=m.dtype)).all()
    )


def _check_references(spec: CodeSpec) -> None:
    keys = rendered_keys(spec)
    for it in spec.items:
        refs = list(getattr(it, "parts", ())) + list(getattr(it, "body", ()))
        refs += list(getattr(it, "inserts", ()))
        for r in refs:
            if r not in keys:
                raise SpecError(f"{it.key}: references unknown key {r!r}")
    for r in spec.roles:
        if r.key not in keys:
            raise SpecError(f"role for unknown key {r.key!r}")
        if r.role == "prep":
            item = next((it for it in spec.items if it.key == r.key), None)
            if not isinstance(item, (Circuit, RepeatUntilSuccess)):
                raise SpecError(
                    f"prep role {r.key!r} must name a Circuit or a "
                    "RepeatUntilSuccess"
                )


def _gate_tables(
    spec: CodeSpec,
    circuits: dict[str, tuple[list, list]],
    alg: derive.Algebra,
) -> dict[str, dict[str, Any]]:
    f = spec.facts
    data_index = {q: i for i, q in enumerate(f.data)}
    out: dict[str, dict[str, Any]] = {}
    shared: dict[str, tuple] = {}
    for it in spec.items:
        if not isinstance(it, Gate):
            continue
        layers, _ = circuits[it.circuit_key or it.key]
        touched = {q for layer in layers for _, qs in layer for q in qs}
        if touched - set(f.data):
            raise SpecError(
                f"{it.key}: a gate circuit touches non-data qubits"
            )
        t = derive.unitary_tableau(layers, data_index)
        try:
            declared = stim.Tableau.from_named_gate(it.logical)
        except (ValueError, IndexError):
            raise SpecError(
                f"{it.key}: unknown logical gate {it.logical!r}"
            ) from None
        try:
            fmap, smap = derive.gate_bookkeeping(
                t, alg, f.frame_layout, declared
            )
        except SpecError as e:
            raise SpecError(f"{it.key}: {e}") from None
        rules = derive.per_qubit_rules(layers, f.data)
        if rules is None:
            raise NotImplementedError(
                f"{it.key}: a multi-qubit op in a gate circuit has no "
                "per-qubit frame rule"
            )
        uniform = {tuple(sorted(r.items())) for r in rules.values()}
        if len(uniform) != 1:
            raise NotImplementedError(
                f"{it.key}: the per-qubit frame rule differs across data "
                "qubits"
            )
        trivial = _is_identity(fmap) and _is_identity(smap)
        if not trivial and not (it.circuit_key and it.update_key):
            raise SpecError(f"{it.key}: needs circuit_key and update_key")
        entry = {
            "logical": it.logical,
            "frame_map": _ints(fmap),
            "syndrome_map": _ints(smap),
            "per_qubit": next(iter(rules.values())),
            "trivial": trivial,
        }
        if it.update_key:
            maps = (entry["frame_map"], entry["syndrome_map"])
            if shared.setdefault(it.update_key, maps) != maps:
                raise SpecError(
                    f"{it.update_key}: gates sharing it derive different maps"
                )
        out[it.key] = entry
    return out


def generate(spec: CodeSpec) -> dict:
    """Derive every table and run every structural check.

    Returns `{"tables": ..., "checks": ...}`. Raises `SpecError` for a
    wrong spec and `NotImplementedError` for one outside what the renderer
    supports.
    """
    f = spec.facts
    checks: dict[str, object] = {}
    tables: dict[str, object] = {}

    if len(f.logicals) > 1:
        raise NotImplementedError(
            "more than one logical qubit in a single code: per-patch "
            "tracking holds one"
        )
    if len(f.tracking.syndrome) > 1:
        raise NotImplementedError(
            "more than one stored-syndrome key: only one snapshot key is "
            "supported"
        )
    alg = derive.Algebra(f.stabilizers, f.logicals)
    checks["facts"] = "commute, independent, logicals anticommute in pairs"

    _check_references(spec)
    checks["references"] = "every part, insert and role resolves"

    circuits = resolve_circuits(spec)
    tables["gates"] = _gate_tables(spec, circuits, alg)
    checks["gates"] = "code-preserving; declared logical action matches"

    # What each measuring circuit reads: {aux: stabilizer index}
    measured = {}
    names = list(f.stabilizer_names)
    for key, (layers, _) in circuits.items():
        _, meas = derive.split_measured(layers)
        if not meas or not set(meas) <= set(f.aux):
            continue
        obs = derive.measured_observables(layers, f)
        measured[key] = {
            a: names.index(s) for a, s in obs.items() if s in names
        }
    tables["syndrome_index"] = measured

    # Lookup tables are literature data, passed through as literal rows
    lut: dict[str, Any] = {}
    for tbl in spec.tables:
        if tbl.kind == "data":
            lut[tbl.name] = [list(r[0]) for r in sorted(tbl.rows)]
        else:
            lut[tbl.name] = [[list(b) for b in r] for r in tbl.rows]
    tables["lut"] = lut

    tables["support"] = {
        n: [i for i, p in enumerate(s.lstrip("+-")) if p != "I"]
        for n, s in zip(f.stabilizer_names, f.stabilizers)
    }
    tables["logical_support"] = {
        "X": [i for i, p in enumerate(f.logicals[0][0]) if p != "I"],
        "Z": [i for i, p in enumerate(f.logicals[0][1]) if p != "I"],
    }

    tables["two_patch"] = {
        g.key: derive_two_patch(spec, g) for g in spec.two_patch
    }
    return {"tables": tables, "checks": checks}


def joint_facts(
    spec: CodeSpec, roles: tuple[str, ...]
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Stabilizers, logicals and data labels of the patches in `roles`,
    over their joint data qubits (labels `role.qubit`)."""
    f = spec.facts
    n = len(f.data)
    stabs: list[str] = []
    logicals: list[tuple[str, str]] = []
    data: list[str] = []
    for r_i, role in enumerate(roles):
        before, after = "I" * (n * r_i), "I" * (n * (len(roles) - r_i - 1))

        def pad(s: str) -> str:
            return before + s.lstrip("+-") + after

        stabs += [pad(s) for s in f.stabilizers]
        logicals += [(pad(x), pad(z)) for x, z in f.logicals]
        data += [f"{role}.{q}" for q in f.data]
    return stabs, logicals, data


def derive_two_patch(spec: CodeSpec, g: TwoPatchGate) -> dict[str, Any]:
    """A two-patch gate's frame, syndrome and per-qubit maps, derived on
    the joint system."""
    f = spec.facts
    stabs, logicals, data = joint_facts(spec, g.roles)
    joint_q = [f"{r}.{q}" for r in g.roles for q in f.qubits]
    try:
        layers = parse_circuit(g.text, joint_q)
    except SpecError as e:
        raise SpecError(f"{g.key}: {e}") from None
    touched = {q for layer in layers for _, qs in layer for q in qs}
    if touched - set(data):
        raise SpecError(f"{g.key}: touches non-data qubits")
    alg = derive.Algebra(stabs, logicals)
    t = derive.unitary_tableau(layers, {q: i for i, q in enumerate(data)})
    try:
        declared = stim.Tableau.from_named_gate(g.logical)
    except (ValueError, IndexError):
        raise SpecError(
            f"{g.key}: unknown logical gate {g.logical!r}"
        ) from None
    try:
        fmap, smap = derive.gate_bookkeeping(t, alg, f.frame_layout, declared)
    except SpecError as e:
        raise SpecError(f"{g.key}: {e}") from None
    # Per-qubit frame: the sign-free symplectic action on the joint data
    # qubits, in [x_0..x_{n-1}, z_0..z_{n-1}] order (Stim's to_numpy)
    n = len(data)
    qmap = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for k in range(n):
        for off, p in ((0, "X"), (n, "Z")):
            ps = stim.PauliString(n)
            ps[k] = p
            qmap[:, off + k] = derive.symp(t(ps))
    return {
        "logical": g.logical,
        "roles": list(g.roles),
        "frame_map": _ints(fmap),
        "syndrome_map": _ints(smap),
        "qubit_frame_map": _ints(qmap),
        "layers": [[[nm, list(qs)] for nm, qs in layer] for layer in layers],
    }


_HEADER = '''"""GENERATED by the codepack generator -- do not edit.

Regenerate with `python -m loqs.codepacks.generator {target}`. A hand edit
fails the lock (`TABLES_HASH`).

How to read `GENERATED["tables"]` (each map acts on a bit column, mod 2):
- `gates[key]`: a logical gate, `logical` its declared action.
  `frame_map` updates the logical frame bits (in the spec's
  `Facts.frame_layout` order) and `syndrome_map` the stored syndrome bits
  (one per stabilizer): new = map @ old. `per_qubit` is the rule each
  data qubit's Pauli frame entry follows. Frames track Paulis up to sign,
  so S and Sdag share one rule (S X S^dag = Y, S^dag X S = -Y).
  `trivial`: both maps are the identity, so no bookkeeping step runs.
- `syndrome_index[circuit]`: each auxiliary qubit the circuit measures,
  to the index of the stabilizer its reading gives.
- `support`, `logical_support`: the data-qubit indices each stabilizer
  and logical operator acts on.
- `lut`: the spec's lookup tables, as literal rows.
- `two_patch[key]`: a gate on two patches. Its maps act on both patches'
  bits, concatenated in `roles` order; `qubit_frame_map` acts on the
  per-qubit frames as every x bit, then every z bit; `layers` is its
  circuit.

`docs/markdown/codepack-specs.md` explains these tables in prose.
"""

from typing import Any

'''

_WIDTH = 79


def _inline(value: Any) -> str:
    """`value` as a one-line Python literal, dict keys sorted."""
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{k!r}: {_inline(value[k])}" for k in sorted(value))
            + "}"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_inline(v) for v in value) + "]"
    return repr(value)


def _layout(value: Any, indent: int = 0, used: int = 0) -> str:
    """`value` as a Python literal laid out for reading.

    It stays on one line when it fits in `_WIDTH` after `used` columns, and
    a list of scalars (a matrix row) always does. Otherwise each item goes
    on its own line, indented four spaces past `indent`.
    """
    flat = _inline(value)
    if not isinstance(value, (dict, list)) or used + len(flat) < _WIDTH:
        return flat
    if isinstance(value, list) and not any(
        isinstance(v, (dict, list)) for v in value
    ):
        return flat
    pad = " " * (indent + 4)
    if isinstance(value, dict):
        lines = [
            f"{pad}{k!r}: "
            + _layout(value[k], indent + 4, len(pad) + len(repr(k)) + 2)
            for k in sorted(value)
        ]
        opener, closer = "{", "}"
    else:
        lines = [pad + _layout(v, indent + 4, len(pad)) for v in value]
        opener, closer = "[", "]"
    return f"{opener}\n" + ",\n".join(lines) + f",\n{' ' * indent}{closer}"


def write_artifact(cls: type[GeneratedCodepack]) -> Path:
    """Write `cls`'s generated module, named by `cls.artifact`, into its
    package's folder, and return its path."""
    package, _, name = cls.artifact.rpartition(".")
    if not package:
        raise SpecError(
            f"{cls.__name__}: artifact {cls.artifact!r} must be a dotted "
            "module name inside a package"
        )
    folder = Path(list(importlib.import_module(package).__path__)[0])
    generated = generate(cls.spec)
    target = f"{cls.__module__}:{cls.__name__}"
    body = _HEADER.format(target=target)
    body += f"INPUT_HASH = {spec_hash(cls.spec)!r}\n"
    body += f"GENERATOR_VERSION = {GENERATOR_VERSION!r}\n"
    body += f"STIM_VERSION = {stim.__version__!r}\n"
    body += f"TABLES_HASH = {tables_hash(generated)!r}\n\n"
    prefix = "GENERATED: dict[str, Any] = "
    body += prefix + _layout(generated, 0, len(prefix)) + "\n"
    path = folder / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path
