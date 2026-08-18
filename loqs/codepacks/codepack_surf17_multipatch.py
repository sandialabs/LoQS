#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Multi-patch machinery for the Surface-17/13/10 codepack.

This module provides two-patch logical operations for patches built from
[](api:codepack_surf17_tomita2014), which stores its deferred syndrome
histories on each patch's own tracked data (`patch.data["syndrome_history_X"]`
etc.) so multiple patches can decode independently in one program without
needing any patch-label key namespacing. Every builder below takes a
[](api:PatchGeometry) bundling its patches' labels and qubit lists,
instead of separate parameters per patch.

Provided operations
-------------------
- **Transversal logical CNOT** between two patches of the same layout
  (`build_transversal_cnot_instruction`). Alongside the 9 pairwise physical
  CNOTs, a bookkeeping instruction (a) applies the exact pairwise CNOT
  conjugation to both patches' Pauli frames and (b) XORs the round-aligned
  deferred syndrome histories: X errors copy control -> target, so the
  target's Z-check history is XORed with the control's
  (`hist_Z_tgt ^= hist_Z_ctrl`); Z errors copy target -> control, so
  `hist_X_ctrl ^= hist_X_tgt`. This keeps the space-time diff syndromes
  consistent across the CNOT for the deferred MWPM decoder.

- **Ancilla-mediated joint logical parity measurements**
  (`build_joint_parity_zz_instruction`, `build_joint_parity_xx_instruction`).
  A single bare ancilla measures Z_L(A) Z_L(B) (or X_L(A) X_L(B)) directly:
  the ancilla outcome, corrected by the patches' pending Pauli-frame bits on
  the touched data qubits, is stored under the frame key
  `joint_parity_zz_{patch_a_label}_{patch_b_label}`
  (`joint_parity_xx_{patch_a_label}_{patch_b_label}`), scoped by participant
  labels so multiple simultaneous joint measurements between different
  patch pairs cannot collide. These are **not fault tolerant** (a single ancilla or
  data fault can flip the parity or spread onto a patch) but are
  non-destructive: both parities commute with all stabilizers, so QEC
  continues and both can be measured in the same shot with one shared
  ancilla (`Imrz` is measure-and-reset in this codepack's models).

- **Backend feasibility warning** (`warn_if_backend_infeasible`) for dense
  statevector-based backends at multi-patch qubit counts.

Note on reference rounds: in Bell-type protocols one patch is prepared in a
basis that does not stabilize one check type, making that check's round-0
values random; after the CNOT history XOR this randomness spreads to both
patches' histories of that type. Decode such programs with the corresponding
`reference_round_X`/`reference_round_Z` kwarg on the FT measurements
(see [](api:codepack_surf17_tomita2014)).
"""

from __future__ import annotations

from collections.abc import Sequence
import sys
import warnings

from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.codepacks.codepack_surf17_tomita2014 import (
    DEFAULT_GATE_DURATIONS,
    DEFAULT_IDLE_GATES,
    layout_qubits,
)
from loqs.core import Instruction, PatchGeometry
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchlayout import PatchLayout
from loqs.core.recordables.pauliframe import PauliFrame

PATCH_QUBIT_COUNTS = {
    layout: len(layout_qubits(layout)) for layout in ("surf17", "surf13", "surf10")
}
"""Physical qubits (data + auxiliary) per patch for each layout -- sourced
from codepack_surf17_tomita2014.layout_qubits, the single canonical
definition."""


def _pauli_from_bits(x_bit: int, z_bit: int) -> str:
    return {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[
        (x_bit, z_bit)
    ]


def pairwise_cnot_pauli_frames(
    frame_ctrl: PauliFrame,
    frame_tgt: PauliFrame,
    ctrl_qubits: Sequence[str | int],
    tgt_qubits: Sequence[str | int],
) -> tuple[PauliFrame, PauliFrame]:
    """Exact conjugation of two Pauli frames through pairwise physical CNOTs.

    For each pair `(c, t) = (ctrl_qubits[i], tgt_qubits[i])`, the CNOT
    conjugation propagates X components control -> target and Z components
    target -> control:

    - `X_t' = X_t ^ X_c` (X on control copies onto target)
    - `Z_c' = Z_c ^ Z_t` (Z on target copies onto control)

    Parameters
    ----------
    frame_ctrl:
        [](api:PauliFrame) of the control patch.

    frame_tgt:
        [](api:PauliFrame) of the target patch.

    ctrl_qubits:
        Control qubits of the pairwise CNOTs (subset of `frame_ctrl` labels).

    tgt_qubits:
        Target qubits, aligned index-wise with `ctrl_qubits`.

    Returns
    -------
    tuple[PauliFrame, PauliFrame]
        Updated copies of `(frame_ctrl, frame_tgt)`.
    """
    assert len(ctrl_qubits) == len(
        tgt_qubits
    ), "Pairwise CNOT requires equal-length qubit lists"

    new_ctrl = frame_ctrl.copy()
    new_tgt = frame_tgt.copy()
    for c, t in zip(ctrl_qubits, tgt_qubits):
        xc = frame_ctrl.get_bit("X", c)
        zc = frame_ctrl.get_bit("Z", c)
        xt = frame_tgt.get_bit("X", t)
        zt = frame_tgt.get_bit("Z", t)

        new_ctrl.pauli_frame[new_ctrl.qubit_labels.index(c)] = (
            _pauli_from_bits(xc, zc ^ zt)
        )
        new_tgt.pauli_frame[new_tgt.qubit_labels.index(t)] = (
            _pauli_from_bits(xt ^ xc, zt)
        )

    return new_ctrl, new_tgt


def build_transversal_cnot_circuit_instruction(
    ctrl_data_qubits: Sequence[str],
    tgt_data_qubits: Sequence[str],
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str = "Transversal CNOT physical circuit",
) -> Instruction:
    """Build the physical-circuit half of a transversal logical CNOT.

    One `Gcnot` per data-qubit pair, control patch -> target patch, all in
    a single layer (every pair acts on disjoint qubits, so they commute
    freely and can run in parallel). The circuit is defined only on the
    involved data qubits so discrete error-injection tools enumerate a
    minimal set of fault locations.

    Parameters
    ----------
    ctrl_data_qubits:
        The 9 data qubits of the control patch (template order D0..D8).

    tgt_data_qubits:
        The 9 data qubits of the target patch, aligned index-wise.

    circuit_backend:
        The circuit backend. Default is PyGSTiPhysicalCircuit.

    include_idles:
        Whether to pad the layer's non-participating qubits with an idle
        gate (True) or leave them genuinely blank (False, default). With
        all 9 pairs sharing one layer, every one of the 18 involved
        qubits is active, so this has no effect unless the circuit is
        merged/tiled alongside other qubits later.

    gate_durations, idle_gates:
        See [](api:codepack_surf17_tomita2014.create_qec_code); defaults
        to a duration-2 Gcnot padded with Gi2q, matching that codepack's
        convention.

    name:
        Name for logging purposes.

    Returns
    -------
    Instruction
        A physical circuit instruction (no Pauli frame update; that is done
        by the bookkeeping instruction).
    """
    assert len(ctrl_data_qubits) == len(tgt_data_qubits)
    layers = [
        [("Gcnot", c, t) for c, t in zip(ctrl_data_qubits, tgt_data_qubits)]
    ]
    circuit = circuit_backend(
        layers,  # type: ignore[arg-type]
        qubit_labels=list(ctrl_data_qubits) + list(tgt_data_qubits),
    )
    if include_idles:
        circuit.pad_single_qubit_idles_by_duration_inplace(
            idle_gates or DEFAULT_IDLE_GATES,
            gate_durations or DEFAULT_GATE_DURATIONS,
        )
    return builders.build_physical_circuit_instruction(circuit, name=name)


def build_cnot_bookkeeping_instruction(
    ctrl_patch_label: str,
    tgt_patch_label: str,
    name: str = "Transversal CNOT frame/history bookkeeping",
) -> Instruction:
    """Build the decoder-bookkeeping half of a transversal logical CNOT.

    The apply function takes:

    - `patches`, usually from the previous frame
    - the patch labels, from `Instruction.data`

    It (a) conjugates both patches' Pauli frames through the pairwise CNOTs
    (see [](api:pairwise_cnot_pauli_frames)) and (b) XORs the round-aligned
    syndrome histories, tracked on each patch's own `.data`
    (`hist_Z_tgt ^= hist_Z_ctrl` and `hist_X_ctrl ^= hist_X_tgt`), asserting
    that the two patches have accumulated the same number of
    syndrome-extraction rounds. Each patch's own data qubits (needed for
    the pairwise conjugation) are read from [](api:QECCodePatch.data_qubits)
    at apply time, rather than needing to be re-supplied here to match
    whatever was passed to [](api:build_transversal_cnot_circuit_instruction).

    It returns a [](api:Frame) with the two patches' `.data` updated
    in-place inside `patches`.

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

        # 1. Exact pairwise Pauli frame conjugation
        new_frame_c, new_frame_t = pairwise_cnot_pauli_frames(
            patch_c.pauli_frame,
            patch_t.pauli_frame,
            patch_c.data_qubits,
            patch_t.data_qubits,
        )
        new_patch_c = patch_c.copy(pauli_frame=new_frame_c)
        new_patch_t = patch_t.copy(pauli_frame=new_frame_t)

        # 2. XOR round-aligned syndrome histories across the CNOT, read from
        # and written back to each patch's own tracked data (scoped per
        # patch, so multi-patch programs do not collide without needing any
        # key namespacing).
        # X errors copy ctrl -> tgt and are detected by Z checks;
        # Z errors copy tgt -> ctrl and are detected by X checks.
        hist_X_c = list(new_patch_c.data.get("syndrome_history_X", []))
        hist_Z_c = list(new_patch_c.data.get("syndrome_history_Z", []))
        hist_X_t = list(new_patch_t.data.get("syndrome_history_X", []))
        hist_Z_t = list(new_patch_t.data.get("syndrome_history_Z", []))

        assert len(hist_Z_t) == len(hist_Z_c) and len(hist_X_c) == len(
            hist_X_t
        ), (
            "Transversal CNOT requires both patches to have accumulated the "
            "same number of syndrome-extraction rounds "
            f"(ctrl: {len(hist_X_c)}/{len(hist_Z_c)}, "
            f"tgt: {len(hist_X_t)}/{len(hist_Z_t)})"
        )

        new_hist_Z_t = [
            [a ^ b for a, b in zip(round_t, round_c)]
            for round_t, round_c in zip(hist_Z_t, hist_Z_c)
        ]
        new_hist_X_c = [
            [a ^ b for a, b in zip(round_c, round_t)]
            for round_c, round_t in zip(hist_X_c, hist_X_t)
        ]

        new_patch_c.data["syndrome_history_X"] = new_hist_X_c
        new_patch_t.data["syndrome_history_Z"] = new_hist_Z_t

        new_patches = patches.copy()
        new_patches[ctrl_patch_label] = new_patch_c
        new_patches[tgt_patch_label] = new_patch_t

        return Frame({"patches": new_patches})

    data = {
        "ctrl_patch_label": ctrl_patch_label,
        "tgt_patch_label": tgt_patch_label,
    }

    # No map_qubits_fn: nothing qubit-shaped remains in Instruction.data --
    # data qubits are read from the live patches at apply time instead.
    return Instruction(
        apply_fn,
        data=data,
        name=name,
    )


def build_transversal_cnot_instruction(
    geometry: PatchGeometry,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str = "Transversal Logical CNOT",
) -> Instruction:
    """Build a transversal logical CNOT between two same-layout patches.

    Composite of [](api:build_transversal_cnot_circuit_instruction) followed
    by [](api:build_cnot_bookkeeping_instruction). Intended to be placed in a
    program stack as a global instruction, e.g. `(cnot_inst, None)`.

    Parameters
    ----------
    geometry:
        A [](api:PatchGeometry) with roles `"ctrl"`/`"tgt"`, each assigned
        a patch's full qubit list (template order D0..D8 + ancillas); no
        seam is used by a transversal CNOT. Only the first 9 (data)
        qubits of each role participate.

    circuit_backend:
        The circuit backend. Default is PyGSTiPhysicalCircuit.

    include_idles, gate_durations, idle_gates:
        See [](api:build_transversal_cnot_circuit_instruction).

    name:
        Name for logging purposes.

    Returns
    -------
    Instruction
        The composite transversal logical CNOT instruction.
    """
    circuit_inst = build_transversal_cnot_circuit_instruction(
        geometry.qubits("ctrl")[:9],
        geometry.qubits("tgt")[:9],
        circuit_backend=circuit_backend,
        include_idles=include_idles,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
        name=f"{name} (physical circuit)",
    )
    bookkeeping_inst = build_cnot_bookkeeping_instruction(
        geometry.label("ctrl"),
        geometry.label("tgt"),
        name=f"{name} (bookkeeping)",
    )
    return builders.build_composite_instruction(
        [circuit_inst, bookkeeping_inst],
        name=name,
    )


def _build_joint_parity_instruction(
    basis: str,
    geometry: PatchGeometry,
    ancilla: str,
    circuit_backend: type[BasePhysicalCircuit],
    name: str,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
) -> Instruction:
    """Shared builder for the joint ZZ / XX parity instructions."""
    assert basis in ("ZZ", "XX")
    patch_a_label = geometry.label("a")
    patch_b_label = geometry.label("b")
    data_qubits_a = geometry.qubits("a")[:9]
    data_qubits_b = geometry.qubits("b")[:9]

    if basis == "ZZ":
        # Z_L = Z0 Z4 Z8; X errors on these flip the copied Z parity
        support_idx = [0, 4, 8]
        frame_bit_type = "X"
        frame_key = f"joint_parity_zz_{patch_a_label}_{patch_b_label}"
    else:
        # X_L = X2 X4 X6; Z errors on these flip the kicked-back X parity
        support_idx = [2, 4, 6]
        frame_bit_type = "Z"
        frame_key = f"joint_parity_xx_{patch_a_label}_{patch_b_label}"

    supports_a = [data_qubits_a[i] for i in support_idx]
    supports_b = [data_qubits_b[i] for i in support_idx]

    if basis == "ZZ":
        # Ancilla starts in |0>; each data qubit copies its Z value onto it
        layers = [[("Gcnot", d, ancilla)] for d in supports_a + supports_b]
        layers.append([("Imrz", ancilla)])
    else:
        # Ancilla in |+>; X_L X_L phase kicks back onto the ancilla
        layers = (
            [[("Gh", ancilla)]]
            + [[("Gcnot", ancilla, d)] for d in supports_a + supports_b]
            + [[("Gh", ancilla)], [("Imrz", ancilla)]]
        )
    circuit = circuit_backend(
        layers,  # type: ignore[arg-type]
        qubit_labels=supports_a + supports_b + [ancilla],
    )
    if include_idles:
        circuit.pad_single_qubit_idles_by_duration_inplace(
            idle_gates or DEFAULT_IDLE_GATES,
            gate_durations or DEFAULT_GATE_DURATIONS,
        )
    circuit_inst = builders.build_physical_circuit_instruction(
        circuit, name=f"{name} (physical circuit)"
    )

    def decode_apply_fn(
        patches: PatchLayout,
        parity_outcomes: MeasurementOutcomes,
        patch_a_label: str,
        patch_b_label: str,
        support_qubits_a: list[str],
        support_qubits_b: list[str],
        ancilla: str,
        frame_key: str,
        frame_bit_type: str,
    ) -> Frame:
        outcome = parity_outcomes[ancilla][0]
        correction = 0
        for lbl, supports in (
            (patch_a_label, support_qubits_a),
            (patch_b_label, support_qubits_b),
        ):
            pauli_frame = patches[lbl].pauli_frame
            for q in supports:
                correction ^= pauli_frame.get_bit(frame_bit_type, q)
        return Frame({frame_key: outcome ^ correction})

    def map_qubits_fn(
        qubit_mapping, support_qubits_a, support_qubits_b, ancilla, **kwargs
    ):
        new_kwargs = kwargs.copy()
        new_kwargs["support_qubits_a"] = [
            qubit_mapping.get(q, q) for q in support_qubits_a
        ]
        new_kwargs["support_qubits_b"] = [
            qubit_mapping.get(q, q) for q in support_qubits_b
        ]
        new_kwargs["ancilla"] = qubit_mapping.get(ancilla, ancilla)
        return new_kwargs

    decode_inst = Instruction(
        decode_apply_fn,
        data={
            "patch_a_label": patch_a_label,
            "patch_b_label": patch_b_label,
            "support_qubits_a": supports_a,
            "support_qubits_b": supports_b,
            "ancilla": ancilla,
            "frame_key": frame_key,
            "frame_bit_type": frame_bit_type,
        },
        map_qubits_fn=map_qubits_fn,
        param_priorities={"parity_outcomes": ["history[-1]"]},
        param_aliases={"parity_outcomes": "measurement_outcomes"},
        name=f"{name} (decode)",
    )

    return builders.build_composite_instruction(
        [circuit_inst, decode_inst],
        name=name,
    )


def build_joint_parity_zz_instruction(
    geometry: PatchGeometry,
    ancilla: str,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str = "Joint ZZ Parity Measurement",
) -> Instruction:
    """Ancilla-mediated measurement of Z_L(A) Z_L(B) (non fault-tolerant).

    The ancilla is assumed to start in |0> (fresh from state init, or reset
    by a previous `Imrz`). Six `Gcnot`s copy the Z_L supports (D0, D4, D8 of
    each patch) onto the ancilla, which is then measured (and reset) with
    `Imrz`. The decode step XORs the outcome with both patches' pending
    Pauli-frame X bits on the touched qubits and stores the result under the
    frame key `joint_parity_zz_{patch_a_label}_{patch_b_label}` (the two
    patch labels of `geometry`'s `"a"`/`"b"` roles).

    The measured operator commutes with all stabilizers of both patches, so
    the patches remain valid code states afterwards. Single faults on the
    ancilla or data qubits can flip the parity or spread onto a patch: this
    is the deliberately non-FT mode.

    Parameters
    ----------
    geometry:
        A [](api:PatchGeometry) with roles `"a"`/`"b"`, each assigned a
        patch's full qubit list (template order D0..D8 + ancillas); only
        the first 9 (data) qubits of each role participate. No seam is
        used by an ancilla-mediated joint parity measurement.

    ancilla:
        Label of the bare (non-patch) ancilla qubit.

    circuit_backend:
        The circuit backend. Default is PyGSTiPhysicalCircuit.

    include_idles, gate_durations, idle_gates:
        See [](api:codepack_surf17_tomita2014.create_qec_code); pads every
        non-participating qubit at each layer (default: off).

    name:
        Name for logging purposes.

    Returns
    -------
    Instruction
        Composite instruction storing
        `joint_parity_zz_{patch_a_label}_{patch_b_label}` in its final frame.
    """
    return _build_joint_parity_instruction(
        "ZZ",
        geometry,
        ancilla,
        circuit_backend,
        name,
        include_idles=include_idles,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
    )


def build_joint_parity_xx_instruction(
    geometry: PatchGeometry,
    ancilla: str,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    name: str = "Joint XX Parity Measurement",
) -> Instruction:
    """Ancilla-mediated measurement of X_L(A) X_L(B) (non fault-tolerant).

    The ancilla is assumed to start in |0>; `Gh` takes it to |+>, six
    `Gcnot`s (ancilla as control) kick the X_L supports' phase (D2, D4, D6
    of each patch) back onto it, and `Gh` + `Imrz` read it out in the X basis.
    The decode step XORs the outcome with both patches' pending Pauli-frame
    Z bits on the touched qubits and stores the result under the frame key
    `joint_parity_xx_{patch_a_label}_{patch_b_label}`. See
    [](api:build_joint_parity_zz_instruction) for the non-FT caveat;
    parameters are identical.
    """
    return _build_joint_parity_instruction(
        "XX",
        geometry,
        ancilla,
        circuit_backend,
        name,
        include_idles=include_idles,
        gate_durations=gate_durations,
        idle_gates=idle_gates,
    )


def warn_if_backend_infeasible(
    layout: str,
    backend: str | type,
    n_patches: int = 2,
    extra_qubits: int = 0,
) -> int:
    """Warn when a dense-statevector backend is selected at infeasible size.

    STIM (stabilizer) backends scale fine for all layouts and never warn.
    Dense backends (statevector/kraus) store 2^N amplitudes; this warns
    hard above ~8 GiB and gently above ~0.25 GiB.

    Parameters
    ----------
    layout:
        One of "surf17", "surf13", "surf10".

    backend:
        Backend name or class; matched case-insensitively against "stim".

    n_patches:
        Number of surface-code patches. Default 2.

    extra_qubits:
        Additional bare qubits (e.g. a shared joint-parity ancilla, or
        lattice-surgery seam qubits: 3 per seam, 6 for a surgery joint
        strategy or a surgery CNOT). Reference sizes: 3 patches + 2 seams
        (surgery CNOT) = 57/45/36 qubits for surf17/surf13/surf10 (all
        INFEASIBLE dense -> stim only); 2 patches + 2 seams = 40/32/26
        (surf10 gets the gentle warning and is kraus-feasible).

    Returns
    -------
    int
        The total number of physical qubits.
    """
    num_qubits = (
        PATCH_QUBIT_COUNTS[layout] * n_patches + extra_qubits
    )
    backend_str = (
        backend if isinstance(backend, str) else backend.__name__
    ).lower()
    if "stim" in backend_str:
        return num_qubits

    mem_gib = (2**num_qubits) * 16 / 2**30
    if num_qubits >= 30:
        msg = (
            f"Backend '{backend_str}' is INFEASIBLE for {n_patches}x{layout}"
            f"{f' + {extra_qubits} extra qubit(s)' if extra_qubits else ''}: "
            f"{num_qubits} qubits requires ~{mem_gib:,.0f} GiB for a dense "
            "statevector. Use the stim backend, or a smaller layout "
            "(2x surf10 = 20-21 qubits is the intended dense-backend "
            "configuration)."
        )
        print(f"WARNING: {msg}", file=sys.stderr)
        warnings.warn(msg, UserWarning)
    elif num_qubits >= 24:
        msg = (
            f"Backend '{backend_str}' with {n_patches}x{layout}"
            f"{f' + {extra_qubits} extra qubit(s)' if extra_qubits else ''} uses "
            f"{num_qubits} qubits (~{mem_gib:.1f} GiB dense statevector); "
            "expect slow shots and high memory use. Consider stim or "
            "2x surf10."
        )
        print(f"WARNING: {msg}", file=sys.stderr)
        warnings.warn(msg, UserWarning)

    return num_qubits
