#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


"""The renderer: a spec, its generated tables and the options to a `QECCode`.

Deterministic and free of Stim. The generic apply functions are module
level and take their tables from `Instruction.data`, so instructions
serialize by source with no closure. Every one writes into a copy of the
incoming layout.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.codepacks.generator import gates
from loqs.codepacks.generator.schema import (
    CheckCircuit,
    Circuit,
    CodeSpec,
    Composite,
    CustomApplyFn,
    Facts,
    Gate,
    Layer,
    Parallel,
    RepeatUntilSuccess,
    resolve_circuits,
)
from loqs.core import Instruction, QECCode
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchlayout import PatchLayout
from loqs.core.recordables.pauliframe import PauliFrame

CLEAR_NAME = "Clear tracking data"

# --- generic apply functions ----------------------------------------------

_BITS = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
_PAULI = {v: k for k, v in _BITS.items()}


def _mat(m: Sequence[Sequence[int]], v: Sequence[int]) -> list[int]:
    prod = np.array(m, dtype=int) @ np.array(v, dtype=int)
    return [int(b) for b in prod % 2]


def update_tracking(
    patches: PatchLayout,
    labels: Sequence[str],
    tracking: Mapping[str, Any],
    frame_map: Sequence[Sequence[int]],
    syndrome_map: Sequence[Sequence[int]],
    qubit_frame_map: Sequence[Sequence[int]] | None = None,
) -> tuple[PatchLayout, list[int]]:
    """Apply a derived tracking update to one patch or several.

    Joint vectors concatenate the patches in `labels` order. The logical
    frame and the stored syndrome (a missing key reads as zeros and is
    written) take the linear maps. With `qubit_frame_map`, each patch's
    per-qubit frame takes the symplectic map over `[x..., z...]`. Returns
    a new layout, leaving `patches` untouched, and the new logical frame
    (all patches, concatenated).
    """
    nl = len(labels)
    new_data = {lbl: dict(patches[lbl].data) for lbl in labels}

    lf = tracking["logical_frame"]
    nf = len(frame_map) // nl
    f = sum((list(new_data[lbl].get(lf, [0] * nf)) for lbl in labels), [])
    new_f = _mat(frame_map, f)
    for i, lbl in enumerate(labels):
        new_data[lbl][lf] = new_f[i * nf : (i + 1) * nf]

    (sk,) = tracking["syndrome"]
    ns = len(syndrome_map) // nl
    s = sum((list(new_data[lbl].get(sk, [0] * ns)) for lbl in labels), [])
    new_s = _mat(syndrome_map, s)
    for i, lbl in enumerate(labels):
        new_data[lbl][sk] = new_s[i * ns : (i + 1) * ns]

    frames = {lbl: patches[lbl].pauli_frame for lbl in labels}
    if qubit_frame_map is not None:
        xs: list[int] = []
        zs: list[int] = []
        for lbl in labels:
            for p in frames[lbl].pauli_frame:
                xs.append(_BITS[p][0])
                zs.append(_BITS[p][1])
        w = _mat(qubit_frame_map, xs + zs)
        n = len(xs)
        new_p = [_PAULI[(w[i], w[n + i])] for i in range(n)]
        pos = 0
        for lbl in labels:
            fr = frames[lbl]
            m = len(fr.pauli_frame)
            frames[lbl] = PauliFrame(fr.qubit_labels, new_p[pos : pos + m])
            pos += m

    new_patches = patches.copy()
    for lbl in labels:
        new_patch = patches[lbl].copy(pauli_frame=frames[lbl])
        new_patch.data = new_data[lbl]
        new_patches[lbl] = new_patch
    return new_patches, new_f


def apply_bookkeeping(
    patches: PatchLayout,
    patch_label: str,
    frame_map: list[list[int]],
    syndrome_map: list[list[int]],
    tracking: dict[str, Any],
) -> Frame:
    """A one-patch gate's derived logical-frame and syndrome update."""
    new_patches, new_f = update_tracking(
        patches, [patch_label], tracking, frame_map, syndrome_map
    )
    return Frame({"patches": new_patches, "new_logical_pauli_frame": new_f})


def apply_joint_bookkeeping(
    patches: PatchLayout,
    patch_labels: list[str],
    frame_map: list[list[int]],
    syndrome_map: list[list[int]],
    qubit_frame_map: list[list[int]],
    tracking: dict[str, Any],
) -> Frame:
    """The joint-system update of a multi-patch gate, split per patch."""
    new_patches, _ = update_tracking(
        patches,
        list(patch_labels),
        tracking,
        frame_map,
        syndrome_map,
        qubit_frame_map=qubit_frame_map,
    )
    return Frame({"patches": new_patches})


def apply_clear_tracking(
    patches: PatchLayout,
    patch_label: str,
    tracking_keys: list[str],
) -> Frame:
    """Drop a patch's tracking data and reset its per-qubit frame."""
    patch = patches[patch_label]
    new_patch = patch.copy(
        pauli_frame=PauliFrame(patch.pauli_frame.qubit_labels)
    )
    for k in tracking_keys:
        new_patch.data.pop(k, None)
    new_patches = patches.copy()
    new_patches[patch_label] = new_patch
    return Frame({"patches": new_patches})


def map_qubit_data(
    qubit_mapping: Mapping[str | int, str | int],
    qubit_keys: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Map the qubit labels held in `qubit_keys`: a list's items or a
    dict's keys."""
    new = dict(kwargs, qubit_keys=qubit_keys)
    for k in qubit_keys:
        v = kwargs[k]
        if isinstance(v, dict):
            new[k] = {qubit_mapping[q]: x for q, x in v.items()}
        else:
            new[k] = [qubit_mapping[q] for q in v]
    return new


def map_patch_labels(
    qubit_mapping: Mapping[str | int, str | int], **kwargs: Any
) -> dict[str, Any]:
    """A multi-patch update holds patch labels, which don't map."""
    return dict(kwargs)


# --- rendering ----------------------------------------------------------------


def _listify(v: Any) -> Any:
    if isinstance(v, (tuple, list)):
        return [_listify(x) for x in v]
    if isinstance(v, dict):
        return {k: _listify(x) for k, x in v.items()}
    return v


def _tracking_dict(f: Facts) -> dict[str, Any]:
    t = f.tracking
    return {"logical_frame": t.logical_frame, "syndrome": list(t.syndrome)}


def _resolve_ref(spec: CodeSpec, tables: Mapping[str, Any], ref: str) -> Any:
    """The value of a `kind:arg` reference to a generated table."""
    kind, _, arg = ref.partition(":")
    if kind == "syndrome_index":
        return dict(tables["syndrome_index"][arg])
    if kind == "aux_order":
        m = tables["syndrome_index"][arg]
        return sorted(m, key=m.get)  # type: ignore[arg-type]
    if kind == "lut":
        return _listify(tables["lut"][arg])
    if kind == "support":
        f = spec.facts
        ptype = arg.split("-")[0]
        return [
            list(tables["support"][n])
            for n, s in zip(f.stabilizer_names, f.stabilizers)
            if set(s.lstrip("+-")) <= {"I", ptype}
        ]
    if kind == "logical_support":
        return list(tables["logical_support"][arg])
    raise KeyError(ref)


def _default_durations() -> dict[str, int]:
    return {
        n: gates.DEFAULT_DURATIONS[e.duration] for n, e in gates.GATES.items()
    }


def _default_idles() -> dict[int, str]:
    return {
        gates.DEFAULT_DURATIONS[c]: n for c, n in gates.DEFAULT_IDLES.items()
    }


def _custom_apply_fn(path: str) -> Callable:
    mod, _, qual = path.partition(":")
    fn: Any = importlib.import_module(mod)
    for part in qual.split("."):
        fn = getattr(fn, part)
    return fn


def render(
    spec: CodeSpec,
    generated: Mapping[str, Any],
    *,
    circuit_backend: Any = PyGSTiPhysicalCircuit,
    include_idles: bool = False,
    gate_durations: Mapping[str, int] | None = None,
    idle_gates: Mapping[int, str] | None = None,
    ft_state_prep_max_repeats: int = 100,
) -> QECCode:
    """Build the spec's `QECCode` from its generated tables and options."""
    f = spec.facts
    tables = generated["tables"]
    durations = dict(gate_durations or _default_durations())
    idles = dict(idle_gates or _default_idles())
    circuits = resolve_circuits(spec)
    prep_circuits = {r.key for r in spec.roles if r.role == "prep"}

    def circ(layers: list[Layer], labels: Sequence[str]) -> Any:
        flat = [[(n, *qs) for n, qs in layer] for layer in layers]
        c = circuit_backend(flat, qubit_labels=list(labels))
        if include_idles:
            c.pad_single_qubit_idles_by_duration_inplace(idles, durations)
        return c

    custom = [it for it in spec.items if isinstance(it, CustomApplyFn)]
    scratch = sorted({k for it in custom for k in it.scratch})
    tracking_keys = [f.tracking.logical_frame, *f.tracking.syndrome, *scratch]
    clear = Instruction(
        apply_clear_tracking,
        data={"tracking_keys": tracking_keys},
        name=CLEAR_NAME,
    )
    tracking = _tracking_dict(f)

    inst: dict[str, Instruction] = {}
    for it in spec.items:
        name = it.name or it.key
        if isinstance(it, (Circuit, CheckCircuit, Parallel)):
            c_inst = builders.build_physical_circuit_instruction(
                circ(*circuits[it.key]),
                name=f"{name} circuit" if it.key in prep_circuits else name,
            )
            if it.key in prep_circuits:
                inst[it.key] = builders.build_composite_instruction(
                    [c_inst, clear], name=name
                )
            else:
                inst[it.key] = c_inst
        elif isinstance(it, Gate):
            g = tables["gates"][it.key]
            ckey = it.circuit_key or it.key
            c_inst = builders.build_physical_circuit_instruction(
                circ(*circuits[ckey]),
                pauli_frame_update=dict(g["per_qubit"]),
                name=(it.circuit_name if not g["trivial"] else it.name)
                or ckey,
            )
            if g["trivial"]:
                inst[it.key] = c_inst
                continue
            inst[ckey] = c_inst
            if it.update_key not in inst:
                inst[it.update_key] = Instruction(
                    apply_bookkeeping,
                    data={
                        "frame_map": g["frame_map"],
                        "syndrome_map": g["syndrome_map"],
                        "tracking": tracking,
                    },
                    name=it.update_name or it.update_key,
                )
            inst[it.key] = builders.build_composite_instruction(
                [inst[ckey], inst[it.update_key]], name=name
            )
        elif isinstance(it, RepeatUntilSuccess):
            reset = builders.build_physical_circuit_instruction(
                circ(*circuits[f"{it.key} reset"]),
                name=it.reset_name or f"{it.key} reset",
            )
            inst[it.key] = builders.build_repeat_until_success_instruction(
                [reset, clear, *[inst[k] for k in it.body]],
                rus_key=it.key,
                test_frame_key="measurement_outcomes",
                expected=MeasurementOutcomes(
                    {q: list(v) for q, v in it.expected}
                ),
                max_repeats=ft_state_prep_max_repeats,
                name=name,
            )
        elif isinstance(it, Composite):
            inst[it.key] = builders.build_composite_instruction(
                [inst[k] for k in it.parts], name=name
            )
        elif isinstance(it, CustomApplyFn):
            data = {k: _listify(v) for k, v in it.data}
            data.update(
                {k: _resolve_ref(spec, tables, r) for k, r in it.derived}
            )
            kwargs: dict[str, Any] = {}
            if it.qubit_data:
                data["qubit_keys"] = list(it.qubit_data)
                kwargs["map_qubits_fn"] = map_qubit_data
            if it.param_priorities:
                kwargs["param_priorities"] = {
                    k: list(v) for k, v in it.param_priorities
                }
            if it.param_aliases:
                kwargs["param_aliases"] = dict(it.param_aliases)
            inst[it.key] = Instruction(
                _custom_apply_fn(it.fn), data=data, name=name, **kwargs
            )
        else:
            raise TypeError(it)

    return QECCode(inst, list(f.qubits), list(f.data), spec.title)


def render_two_patch(
    spec: CodeSpec,
    generated: Mapping[str, Any],
    key: str,
    geometry: Any,
    *,
    circuit_backend: Any = PyGSTiPhysicalCircuit,
    include_idles: bool = False,
    gate_durations: Mapping[str, int] | None = None,
    idle_gates: Mapping[int, str] | None = None,
) -> Instruction:
    """A two-patch gate on a `PatchGeometry` whose roles are the gate's.

    Raises `ValueError` for a key that isn't one of the spec's two-patch
    gates.
    """
    keys = [g.key for g in spec.two_patch]
    if key not in keys:
        raise ValueError(
            f"{key!r} isn't a two-patch gate of {spec.name!r}; "
            f"its two-patch gates are {keys}"
        )
    g = generated["tables"]["two_patch"][key]
    item = next(t for t in spec.two_patch if t.key == key)
    f = spec.facts
    real: dict[str, str] = {}
    labels: list[str] = []
    for role in g["roles"]:
        qs = list(geometry.qubits(role))
        if len(qs) != len(f.qubits):
            raise ValueError(
                f"role {role}: {len(qs)} qubits, need {len(f.qubits)}"
            )
        for tq, rq in zip(f.qubits, qs):
            real[f"{role}.{tq}"] = rq
        labels += [real[f"{role}.{q}"] for q in f.data]
    layers = [
        [(n, *(real[q] for q in qs)) for n, qs in layer]
        for layer in g["layers"]
    ]
    durations = dict(gate_durations or _default_durations())
    idles = dict(idle_gates or _default_idles())
    c = circuit_backend(layers, qubit_labels=labels)
    if include_idles:
        c.pad_single_qubit_idles_by_duration_inplace(idles, durations)
    name = item.name or key
    c_inst = builders.build_physical_circuit_instruction(
        c, name=f"{name} (physical circuit)"
    )
    b_inst = Instruction(
        apply_joint_bookkeeping,
        data={
            "patch_labels": [geometry.label(r) for r in g["roles"]],
            "frame_map": g["frame_map"],
            "syndrome_map": g["syndrome_map"],
            "qubit_frame_map": g["qubit_frame_map"],
            "tracking": _tracking_dict(f),
        },
        map_qubits_fn=map_patch_labels,
        name=f"{name} (bookkeeping)",
    )
    return builders.build_composite_instruction([c_inst, b_inst], name=name)
