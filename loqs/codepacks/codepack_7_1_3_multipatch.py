#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Deprecated function wrappers for the [[7,1,3]] transversal logical CX.

Use [](api:SteaneCodepack) instead: `SteaneCodepack(...).build_two_patch(
"Transversal CX", geometry)` builds the CX, and its spec document says how
its bookkeeping conjugates the logical Pauli frames and the latest
syndromes. [](api:conjugate_cx_logical_pauli_frames) and
[](api:conjugate_cx_latest_syndromes) stay here, without a warning, because
instructions serialized by older versions call them.
"""

from __future__ import annotations

from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.codepacks.steane import SteaneCodepack
from loqs.core import Instruction, PatchGeometry
from loqs.core.instructions import builders
from loqs.internal.legacy import deprecated

_KEY = "Transversal CX"


def _build(
    geometry: PatchGeometry,
    circuit_backend: type[BasePhysicalCircuit],
    include_idles: bool,
    gate_durations: dict[str, int | float] | None,
    idle_gates: dict[int | float, str] | None,
    name: str,
) -> tuple[Instruction, Instruction]:
    """The class's CX parts, renamed from `name`."""
    cx = SteaneCodepack(
        circuit_backend=circuit_backend,
        include_idles=include_idles,
        gate_durations=gate_durations,  # type: ignore[arg-type]
        idle_gates=idle_gates,  # type: ignore[arg-type]
    ).build_two_patch(_KEY, geometry)
    circuit_inst, bookkeeping_inst = cx.data["instructions"]
    circuit_inst.name = f"{name} (physical circuit)"
    bookkeeping_inst.name = f"{name} (bookkeeping)"
    return circuit_inst, bookkeeping_inst


@deprecated('SteaneCodepack(...).build_two_patch("Transversal CX", geometry)')
def build_transversal_cx_circuit_instruction(
    geometry: PatchGeometry,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str = "Transversal CX physical circuit",
) -> Instruction:
    """Build the physical-circuit half of a transversal logical CX.

    This is the first part of [](api:SteaneCodepack)'s `"Transversal CX"`,
    renamed `name`. `geometry` is a [](api:PatchGeometry) with roles
    `"ctrl"`/`"tgt"` and no seam; the other parameters are the class's
    options of the same names.
    """
    inst = _build(
        geometry,
        circuit_backend,
        include_idles,
        gate_durations,
        idle_gates,
        name,
    )[0]
    inst.name = name
    return inst


def conjugate_cx_logical_pauli_frames(
    frame_ctrl: list[int], frame_tgt: list[int]
) -> tuple[list[int], list[int]]:
    """Conjugate two `logical_pauli_frame`s through a transversal CX.

    Each frame is a 2-element `[Z_bit, X_bit]` list, per
    [](api:SteaneCodepack)'s convention. The control's X-bit
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
    [](api:SteaneCodepack)'s convention. This is the
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


@deprecated('SteaneCodepack(...).build_two_patch("Transversal CX", geometry)')
def build_cx_bookkeeping_instruction(
    ctrl_patch_label: str,
    tgt_patch_label: str,
    name: str = "Transversal CX frame bookkeeping",
) -> Instruction:
    """Build the Pauli-frame-bookkeeping half of a transversal logical CX.

    This is the second part of [](api:SteaneCodepack)'s `"Transversal CX"`
    on the two patch labels, renamed `name`. It conjugates both patches'
    logical Pauli frames, syndromes and per-qubit frames.
    """
    template = SteaneCodepack.spec.facts.qubits
    geometry = PatchGeometry(
        patches={
            "ctrl": (ctrl_patch_label, [f"c{q}" for q in template]),
            "tgt": (tgt_patch_label, [f"t{q}" for q in template]),
        },
        layout="7_1_3",
    )
    inst = _build(geometry, PyGSTiPhysicalCircuit, False, None, None, name)[1]
    inst.name = name
    return inst


@deprecated('SteaneCodepack(...).build_two_patch("Transversal CX", geometry)')
def build_transversal_cx_instruction(
    geometry: PatchGeometry,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str = "Transversal Logical CX",
) -> Instruction:
    """Build a transversal logical CX between two `[[7,1,3]]` patches.

    This is [](api:SteaneCodepack)'s `"Transversal CX"`, composite and parts
    renamed from `name`. `geometry` is a [](api:PatchGeometry) with roles
    `"ctrl"`/`"tgt"` and no seam; the other parameters are the class's
    options of the same names.
    """
    circuit_inst, bookkeeping_inst = _build(
        geometry,
        circuit_backend,
        include_idles,
        gate_durations,
        idle_gates,
        name,
    )
    return builders.build_composite_instruction(
        [circuit_inst, bookkeeping_inst], name=name
    )
