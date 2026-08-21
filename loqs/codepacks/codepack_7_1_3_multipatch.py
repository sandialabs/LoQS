#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Multi-patch machinery for the [[7,1,3]] color code codepack.

This module provides a transversal logical CX between two patches built
from [](api:codepack_7_1_3_quantinuum2021): one physical `Gcnot` per
aligned data-qubit pair, plus a bookkeeping instruction that conjugates
both patches' `logical_pauli_frame` *and* `latest_syndrome` through the
standard CSS transversal-CNOT stabilizer map. The builder takes a
[](api:PatchGeometry) with roles `"ctrl"`/`"tgt"`, no seam.

**Stabilizer coupling and its fix.** A transversal CX couples both
patches' local stabilizer checks going forward: the control's X-type
checks become the product of its own value and the target's
(`S_X^ctrl -> S_X^ctrl * S_X^tgt`), and the target's Z-type checks become
the product of the target's own value and the control's
(`S_Z^tgt -> S_Z^ctrl * S_Z^tgt`); the control's Z-type and the target's
X-type checks are individually unchanged. Because
[](api:codepack_7_1_3_quantinuum2021)'s decoder stores only a single
`latest_syndrome` snapshot per patch (overwritten wholesale by every QEC
round, rather than an append-only history), this coupling can be corrected
with a one-time XOR of that snapshot at CX time -- see
[](api:conjugate_cx_latest_syndromes) -- fully analogous to
[](api:conjugate_cx_logical_pauli_frames)'s frame conjugation. Since every
subsequent round starts from this corrected baseline and overwrites it
completely, this single correction is sufficient for arbitrarily many QEC
rounds afterward, not just one: it restores decoding to the same
correctness the decoder already has for uncoupled single-patch operation
(i.e. its existing single-fault-per-round assumption is not weakened by
having gone through a transversal CX). This differs from
[](api:codepack_surf17_multipatch), which needs a full history-wide XOR
because its decoder defers to a joint MWPM pass over an append-only
space-time graph rather than decoding one snapshot at a time.
"""

from __future__ import annotations

from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.core import Instruction, PatchGeometry
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.recordables.patchlayout import PatchLayout


def build_transversal_cx_circuit_instruction(
    geometry: PatchGeometry,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str = "Transversal CX physical circuit",
) -> Instruction:
    """Build the physical-circuit half of a transversal logical CX.

    One `Gcnot` per data-qubit pair (7, `D0..D6`), control patch -> target
    patch, all in a single layer (every pair acts on disjoint qubits, so
    they commute freely and can run in parallel).

    Parameters
    ----------
    geometry:
        A [](api:PatchGeometry) with roles `"ctrl"`/`"tgt"`, each assigned
        a patch's full qubit list (template order `A0,A1,A2,D0..D6`); no
        seam is used by a transversal CX. Only the last 7 (data) qubits
        of each role participate.

    circuit_backend:
        The circuit backend. Default is PyGSTiPhysicalCircuit.

    include_idles:
        Whether to pad the layer's non-participating qubits with an idle
        gate (True) or leave them genuinely blank (False, default).

    gate_durations, idle_gates:
        See [](api:codepack_7_1_3_quantinuum2021.create_qec_code); defaults
        to that codepack's own duration-2 `Gcnot` convention.

    name:
        Name for logging purposes.

    Returns
    -------
    Instruction
        A physical circuit instruction (no Pauli frame update; that is done
        by the bookkeeping instruction).
    """
    ctrl_data_qubits = geometry.qubits("ctrl")[3:]
    tgt_data_qubits = geometry.qubits("tgt")[3:]
    assert len(ctrl_data_qubits) == len(tgt_data_qubits) == 7, (
        "Transversal CX requires 7 data qubits per role "
        "(template order A0,A1,A2,D0..D6)"
    )

    if gate_durations is None:
        gate_durations = {"Gcnot": 2, "Gi2Q": 2}
    if idle_gates is None:
        idle_gates = {2: "Gi2Q"}

    layers = [
        [("Gcnot", c, t) for c, t in zip(ctrl_data_qubits, tgt_data_qubits)]
    ]
    circuit = circuit_backend(
        layers,  # type: ignore[arg-type]
        qubit_labels=list(ctrl_data_qubits) + list(tgt_data_qubits),
    )
    if include_idles:
        circuit.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    return builders.build_physical_circuit_instruction(circuit, name=name)


def conjugate_cx_logical_pauli_frames(
    frame_ctrl: list[int], frame_tgt: list[int]
) -> tuple[list[int], list[int]]:
    """Conjugate two `logical_pauli_frame`s through a transversal CX.

    Each frame is a 2-element `[Z_bit, X_bit]` list, per
    [](api:codepack_7_1_3_quantinuum2021)'s convention. The control's X-bit
    propagates onto the target and the target's Z-bit propagates onto the
    control, matching the standard CNOT conjugation rule.

    Parameters
    ----------
    frame_ctrl:
        The control patch's `logical_pauli_frame`.

    frame_tgt:
        The target patch's `logical_pauli_frame`.

    Returns
    -------
    tuple[list[int], list[int]]
        Updated copies of `(frame_ctrl, frame_tgt)`.
    """
    new_ctrl = list(frame_ctrl)
    new_tgt = list(frame_tgt)
    new_tgt[1] ^= new_ctrl[1]  # X propagates control -> target
    new_ctrl[0] ^= new_tgt[0]  # Z propagates target -> control
    return new_ctrl, new_tgt


def conjugate_cx_latest_syndromes(
    syndrome_ctrl: list[int], syndrome_tgt: list[int]
) -> tuple[list[int], list[int]]:
    """Conjugate two `latest_syndrome`s through a transversal CX.

    Each syndrome is a 6-element `[S1, S2, S3, S4, S5, S6]` list (X-type
    checks S1-S3, Z-type checks S4-S6), per
    [](api:codepack_7_1_3_quantinuum2021)'s convention. This is the
    syndrome analog of [](api:conjugate_cx_logical_pauli_frames), fixing
    the two patches' checks to reflect the standard CSS transversal-CNOT
    stabilizer map (`S_X^ctrl -> S_X^ctrl * S_X^tgt`,
    `S_Z^tgt -> S_Z^ctrl * S_Z^tgt`): the control's X-type checks pick up
    the target's pre-CX X-type values, and the target's Z-type checks pick
    up the control's pre-CX Z-type values. The control's Z-type and the
    target's X-type checks are individually unchanged. Note this is the
    opposite pairing from the frame conjugation (there, the *target*'s
    X-bit picks up the *control*'s) -- the frame tracks a physical error
    operator whose identity spreads forward through the gate, while a
    syndrome bit is the eigenvalue of a fixed local measurement, which
    shifts due to stabilizer-group closure across the entangling gate.

    Parameters
    ----------
    syndrome_ctrl:
        The control patch's `latest_syndrome`.

    syndrome_tgt:
        The target patch's `latest_syndrome`.

    Returns
    -------
    tuple[list[int], list[int]]
        Updated copies of `(syndrome_ctrl, syndrome_tgt)`.
    """
    new_ctrl = list(syndrome_ctrl)
    new_tgt = list(syndrome_tgt)
    for i in range(3):
        new_ctrl[i] ^= syndrome_tgt[i]  # X-type: ctrl picks up tgt's value
    for i in range(3, 6):
        new_tgt[i] ^= syndrome_ctrl[i]  # Z-type: tgt picks up ctrl's value
    return new_ctrl, new_tgt


def build_cx_bookkeeping_instruction(
    ctrl_patch_label: str,
    tgt_patch_label: str,
    name: str = "Transversal CX frame bookkeeping",
) -> Instruction:
    """Build the Pauli-frame-bookkeeping half of a transversal logical CX.

    Conjugates both patches' `logical_pauli_frame` (see
    [](api:conjugate_cx_logical_pauli_frames)) and `latest_syndrome` (see
    [](api:conjugate_cx_latest_syndromes)) through the standard CSS
    transversal-CNOT rules.

    Parameters
    ----------
    ctrl_patch_label:
        Patch label of the control patch.

    tgt_patch_label:
        Patch label of the target patch.

    name:
        Name for logging purposes.

    Returns
    -------
    Instruction
        The bookkeeping instruction.
    """

    def apply_fn(
        patches: PatchLayout,
        ctrl_patch_label: str,
        tgt_patch_label: str,
    ) -> Frame:
        patch_c = patches[ctrl_patch_label]
        patch_t = patches[tgt_patch_label]

        frame_c, frame_t = conjugate_cx_logical_pauli_frames(
            patch_c.data.get("logical_pauli_frame", [0, 0]),
            patch_t.data.get("logical_pauli_frame", [0, 0]),
        )
        syndrome_c, syndrome_t = conjugate_cx_latest_syndromes(
            patch_c.data.get("latest_syndrome", [0] * 6),
            patch_t.data.get("latest_syndrome", [0] * 6),
        )

        new_patch_c = patch_c.copy()
        new_patch_c.data["logical_pauli_frame"] = frame_c
        new_patch_c.data["latest_syndrome"] = syndrome_c
        new_patch_t = patch_t.copy()
        new_patch_t.data["logical_pauli_frame"] = frame_t
        new_patch_t.data["latest_syndrome"] = syndrome_t

        new_patches = patches.copy()
        new_patches[ctrl_patch_label] = new_patch_c
        new_patches[tgt_patch_label] = new_patch_t

        return Frame({"patches": new_patches})

    data = {
        "ctrl_patch_label": ctrl_patch_label,
        "tgt_patch_label": tgt_patch_label,
    }

    return Instruction(
        apply_fn,
        data=data,
        name=name,
    )


def build_transversal_cx_instruction(
    geometry: PatchGeometry,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str = "Transversal Logical CX",
) -> Instruction:
    """Build a transversal logical CX between two `[[7,1,3]]` patches.

    Composite of [](api:build_transversal_cx_circuit_instruction) followed
    by [](api:build_cx_bookkeeping_instruction). Intended to be placed in a
    program stack as a global instruction, e.g. `(cx_inst, None)`. See this
    module's docstring for the stabilizer-coupling correction this
    instruction applies.

    Parameters
    ----------
    geometry:
        A [](api:PatchGeometry) with roles `"ctrl"`/`"tgt"`, no seam.

    circuit_backend:
        The circuit backend. Default is PyGSTiPhysicalCircuit.

    include_idles, gate_durations, idle_gates:
        See [](api:build_transversal_cx_circuit_instruction).

    name:
        Name for logging purposes.

    Returns
    -------
    Instruction
        The composite transversal logical CX instruction.
    """
    circuit_inst = build_transversal_cx_circuit_instruction(
        geometry,
        circuit_backend=circuit_backend,
        include_idles=include_idles,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
        name=f"{name} (physical circuit)",
    )
    bookkeeping_inst = build_cx_bookkeeping_instruction(
        geometry.label("ctrl"),
        geometry.label("tgt"),
        name=f"{name} (bookkeeping)",
    )
    return builders.build_composite_instruction(
        [circuit_inst, bookkeeping_inst],
        name=name,
    )
