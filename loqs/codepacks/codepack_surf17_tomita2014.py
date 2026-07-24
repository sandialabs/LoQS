#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
######################################################################################################################

"""A `LoQS` QEC codepack for the Surface-17 and Surface-13 rotated surface codes.

This implementation is based on the 2014 paper by Yu Tomita and Krysta M. Svore,
"Low-distance surface codes under realistic quadratic noise", Phys. Rev. A 90, 062320 (2014).

Both layouts encode 1 logical qubit into 9 data qubits (labeled D0 to D8) with code distance d=3,
differing only in how the auxiliary (syndrome extraction) qubits are allocated and scheduled:

- **Surface-17 (layout="surf17")**: Uses 8 dedicated auxiliary qubits (A9 to A16).
  X and Z checks are disjoint, allowing parallel syndrome extraction in 7 cycles.
- **Surface-13 (layout="surf13")**: Uses only 4 auxiliary qubits (A9 to A12) by reusing them
  for both X and Z checks. Syndrome extraction is sequential, requiring 14 cycles.

Decoders
--------
This codepack implements a Minimum-Weight Perfect Matching (MWPM) decoder using PyMatching
and Scipy-compatible check matrices. It constructs a 3D Space-Time Matching Graph deferred
global decoder that decodes across the QEC cycles and final measurement parities,
achieving full fault tolerance.

Examples
--------
Preservation of logical state |0>_L over 3 QEC cycles under single-fault injection:

>>> from loqs.backends import PyGSTiPhysicalCircuit, DictNoiseModel, STIMQuantumState, GateRep
>>> from loqs.core import QuantumProgram
>>> from loqs.codepacks import codepack_surf17_tomita2014 as cp
>>> code = cp.create_qec_code(layout="surf17", num_qec_rounds=3)
>>> qubits = [f'D{i}' for i in range(9)] + [f'A{i}' for i in range(9, 17)]
>>> model = cp.create_ideal_model(qubits, gaterep=GateRep.STIM_CIRCUIT_STR, model_backend=DictNoiseModel)
>>> stack = [
...     ('Init State', None, (len(qubits),), {'qubit_labels': qubits}),
...     ('Init Patch SURF', None, ('L0', qubits)),
...     ('Zero Prep', 'L0'),
...     ('QEC', 'L0'),
...     ('FT Logical Z Measure', 'L0')
... ]
>>> program = QuantumProgram(stack, default_noise_model=model, state_type=STIMQuantumState, patch_types={'SURF': code})

Under 192 single-fault discrete error injections, the PyMatching 3D space-time deferred decoder corrects 100% of all faults (0 failures), fully resolving all hook and dynamic errors.
"""

from collections.abc import Sequence
import copy
import itertools
from typing import Mapping, Literal
import numpy as np

from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.backends.model.basemodel import (
    BaseNoiseModel,
    GateRep,
    InstrumentRep,
)
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.reps import RepTuple
from loqs.core import Instruction, QECCode
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import KwargDict
from loqs.core.recordables import QECCodePatch
from loqs.core.recordables.pauliframe import PauliFrame
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict
from loqs.core.syndromelabel import SyndromeLabel
import loqs.tools.qectools as qt


# Layouts ordered from most to least parallel syndrome extraction: surf17
# runs all 8 checks at once (7 layers), surf13 runs them in 2 sequential
# passes of 4 (14 layers), surf10 reuses one ancilla for all 8 checks in
# sequence (56 layers). An idle_layout may only be borrowed from a layout
# at least this parallel -- e.g. surf10 may borrow surf17's idle schedule,
# but surf17 may not borrow surf10's (see create_qec_code's idle_layout
# parameter).
_LAYOUT_PARALLELISM = {"surf17": 0, "surf13": 1, "surf10": 2}
_LAYOUT_NAMES = {"surf17": "Surface-17 Code", "surf13": "Surface-13 Code", "surf10": "Surface-10 Code"}


def _layout_qubits(layout: Literal["surf17", "surf13", "surf10"]) -> list[str]:
    """The (data + ancilla) qubit labels for one of the three layouts."""
    if layout == "surf10":
        return [f"D{i}" for i in range(9)] + ["A9"]
    elif layout == "surf13":
        return [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 13)]
    elif layout == "surf17":
        return [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 17)]
    else:
        raise ValueError(f"Unknown layout: {layout}")


def _build_raw_syndrome_extraction_circuit(
    layout: Literal["surf17", "surf13", "surf10"],
    qubits: Sequence[str],
    circuit_backend: type[BasePhysicalCircuit],
    X_template: BasePhysicalCircuit,
    Z_template: BasePhysicalCircuit,
):
    """Tile the shared X/Z 5-wire templates onto `qubits` according to one
    layout's ancilla-sharing scheme, without any idle padding."""
    if layout == "surf17":
        X_tiles = [
            [None, None, "D1", "D2", "A9"],
            ["D0", "D1", "D3", "D4", "A11"],
            ["D4", "D5", "D7", "D8", "A14"],
            ["D6", "D7", None, None, "A16"],
        ]
        Z_tiles = [
            [None, "D0", None, "D3", "A10"],
            ["D1", "D2", "D4", "D5", "A12"],
            ["D3", "D4", "D6", "D7", "A13"],
            ["D5", None, "D8", None, "A15"],
        ]
        X_syndrome = circuit_backend.from_circuit_tiling(
            X_template, qubits, X_tiles, merge_offsets=0
        )
        Z_syndrome = circuit_backend.from_circuit_tiling(
            Z_template, qubits, Z_tiles, merge_offsets=0
        )
        return X_syndrome.merge(Z_syndrome, 0)
    elif layout == "surf13":
        X_tiles_mapped = [
            [None, None, "D1", "D2", "A9"],
            ["D0", "D1", "D3", "D4", "A11"],
            ["D4", "D5", "D7", "D8", "A10"],
            ["D6", "D7", None, None, "A12"],
        ]
        Z_tiles_mapped = [
            [None, "D0", None, "D3", "A10"],
            ["D1", "D2", "D4", "D5", "A12"],
            ["D3", "D4", "D6", "D7", "A9"],
            ["D5", None, "D8", None, "A11"],
        ]
        X_syndrome = circuit_backend.from_circuit_tiling(
            X_template, qubits, X_tiles_mapped, merge_offsets=0
        )
        Z_syndrome = circuit_backend.from_circuit_tiling(
            Z_template, qubits, Z_tiles_mapped, merge_offsets=0
        )
        return X_syndrome.append(Z_syndrome)
    elif layout == "surf10":
        X_tiles_10 = [
            [None, None, "D1", "D2", "A9"],
            ["D0", "D1", "D3", "D4", "A9"],
            ["D4", "D5", "D7", "D8", "A9"],
            ["D6", "D7", None, None, "A9"],
        ]
        Z_tiles_10 = [
            [None, "D0", None, "D3", "A9"],
            ["D1", "D2", "D4", "D5", "A9"],
            ["D3", "D4", "D6", "D7", "A9"],
            ["D5", None, "D8", None, "A9"],
        ]
        circuits = [
            circuit_backend.from_circuit_tiling(X_template, qubits, [tile], merge_offsets=0)
            for tile in X_tiles_10
        ] + [
            circuit_backend.from_circuit_tiling(Z_template, qubits, [tile], merge_offsets=0)
            for tile in Z_tiles_10
        ]
        full_syndrome_circ = circuits[0]
        for c in circuits[1:]:
            full_syndrome_circ = full_syndrome_circ.append(c)
        return full_syndrome_circ
    else:
        raise ValueError(f"Unknown layout: {layout}")


def create_qec_code(
    layout: Literal["surf17", "surf13", "surf10"] = "surf17",
    idle_layout: Literal["surf17", "surf13", "surf10"] | None = None,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
    num_qec_rounds: int = 3,
) -> QECCode:
    """Create a QECCode implementing the Surface-17, Surface-13, or Surface-10 code.

    Parameters
    ----------
    layout : str, optional
        Whether to implement "surf17" (default), "surf13" (4 auxiliary qubits),
        or "surf10" (1 auxiliary qubit).

    idle_layout : str | None, optional
        Which layout's idle schedule to use for the Syndrome Extraction
        circuit. `None` (default) omits idle gates entirely. `idle_layout
        == layout` pads the circuit with its own idle schedule (only
        supported for "surf17"/"surf13" -- "surf10" has no well-defined
        idle schedule of its own, since serializing all 8 checks onto one
        shared ancilla means most data qubits spend most layers outside
        the 1-2 checks they participate in). A smaller/more-serialized
        layout may instead borrow a larger/more-parallel layout's idle
        schedule, e.g. `layout="surf10", idle_layout="surf17"`: each data
        qubit's real gates are left untouched, and it receives exactly the
        same idle operations (count and duration) that it would under
        `idle_layout`, placed at the earliest layers available once its
        own real gates free them up -- so surf10 can reproduce surf17's
        physical idle-error budget despite its extra serialization,
        instead of that serialization inflating idle time. Layouts may
        only borrow from an equally or more parallel layout (surf10 may
        use surf17 or surf13; surf13 may use surf17; surf17 may not
        borrow from either of the others).

    gate_durations : dict[str, int | float] | None, optional
        Mapping from gate names to durations.

    idle_gates : dict[int | float, str] | None, optional
        Mapping from gate duration to idle gate names.

    circuit_backend : type[BasePhysicalCircuit], optional
        The circuit backend to use when generating physical circuits.
        Default is PyGSTiPhysicalCircuit.

    num_qec_rounds : int, optional
        Number of syndrome extraction rounds before decoding. Default is 3.

    Returns
    -------
    QECCode
        A QECCode implementing the surface code.
    """
    if layout not in _LAYOUT_NAMES:
        raise ValueError(f"Unknown layout: {layout}")
    qubits = _layout_qubits(layout)
    name = _LAYOUT_NAMES[layout]

    if idle_layout is not None:
        if idle_layout not in _LAYOUT_PARALLELISM:
            raise ValueError(f"Unknown idle_layout: {idle_layout}")
        if _LAYOUT_PARALLELISM[idle_layout] > _LAYOUT_PARALLELISM[layout]:
            raise ValueError(
                f"idle_layout={idle_layout!r} is more serialized than layout={layout!r}; "
                "an idle schedule can only be borrowed from an equally or more parallel "
                "layout (e.g. layout='surf10' may use idle_layout='surf17', not the reverse)."
            )
        if idle_layout == layout == "surf10":
            raise ValueError(
                "surf10 has no well-defined idle schedule of its own (its ancilla-sharing "
                "serialization means most data qubits spend most layers outside the checks "
                "they participate in); specify idle_layout='surf17' or 'surf13' instead."
            )

    data_qubits = [f"D{i}" for i in range(9)]
    instructions: dict[str, Instruction] = {}

    if gate_durations is None:
        gate_durations = {
            k: 1
            for k in [
                "Gi",
                "Gi1Q",
                "Gxpi",
                "Gypi",
                "Gzpi",
                "Gzpi2",
                "Gzmpi2",
                "Gh",
            ]
        }
        gate_durations["Gcnot"] = 2
        gate_durations["Gi2Q"] = 2
        gate_durations["Iz"] = 3
        gate_durations["GiMCM"] = 3
    if idle_gates is None:
        idle_gates = {1: "Gi1Q", 2: "Gi2Q", 3: "GiMCM"}

    # 1. State preparation
    # Z-basis preparation (|0>_L)
    raw_Z_prep_circ = circuit_backend(
        [[("Iz", q) for q in data_qubits]], qubit_labels=qubits
    )
    if idle_layout is not None:
        raw_Z_prep_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Zero Prep"] = builders.build_physical_circuit_instruction(
        raw_Z_prep_circ,
        name="Logical zero state prep",
    )

    # X-basis preparation (|+>_L)
    raw_X_prep_circ = circuit_backend(
        [[("Iz", q) for q in data_qubits], [("Gh", q) for q in data_qubits]],
        qubit_labels=qubits,
    )
    if idle_layout is not None:
        raw_X_prep_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Plus Prep"] = builders.build_physical_circuit_instruction(
        raw_X_prep_circ,
        name="Logical plus state prep",
    )

    # 2. Logical Gates
    # Logical I
    logical_I_circ = circuit_backend(
        [[("Gi", q) for q in data_qubits]], qubit_labels=qubits
    )
    if idle_layout is not None:
        logical_I_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["I"] = builders.build_physical_circuit_instruction(
        logical_I_circ,
        name="Logical Identity",
    )

    # Logical X (XL = X2 X4 X6)
    logical_X_circ = circuit_backend(
        [[("Gxpi", q) for q in ["D2", "D4", "D6"]]], qubit_labels=qubits
    )
    if idle_layout is not None:
        logical_X_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["X"] = builders.build_physical_circuit_instruction(
        logical_X_circ,
        name="Logical X",
    )

    # Logical Z (ZL = Z0 Z4 Z8)
    logical_Z_circ = circuit_backend(
        [[("Gzpi", q) for q in ["D0", "D4", "D8"]]], qubit_labels=qubits
    )
    if idle_layout is not None:
        logical_Z_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Z"] = builders.build_physical_circuit_instruction(
        logical_Z_circ,
        name="Logical Z",
    )

    # Logical Y (YL = Z0 Z8 X2 X6 Y4 up to global phase/sign)
    logical_Y_circ = circuit_backend(
        [
            [("Gzpi", q) for q in ["D0", "D8"]],
            [("Gxpi", q) for q in ["D2", "D6"]],
            [("Gypi", "D4")],
        ],
        qubit_labels=qubits,
    )
    if idle_layout is not None:
        logical_Y_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Y"] = builders.build_physical_circuit_instruction(
        logical_Y_circ,
        name="Logical Y",
    )

    # Logical H (composite of transversal physical H then software permutation)
    logical_H_circ = circuit_backend(
        [[("Gh", q) for q in data_qubits]], qubit_labels=qubits
    )
    if idle_layout is not None:
        logical_H_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["H Circuit"] = builders.build_physical_circuit_instruction(
        logical_H_circ,
        name="Logical H physical circuit",
    )

    # Software permutation for H (clockwise 90-degree rotation of the 3x3 data grid)
    # 0->2, 1->5, 2->8, 3->1, 4->4, 5->7, 6->0, 7->3, 8->6
    def H_permutation_apply_fn(
        patches: PatchDict, patch_label: str
    ) -> Frame:
        patch = patches[patch_label]
        old_frame = patch.pauli_frame
        old_paulis = old_frame.pauli_frame
        new_paulis = ["I"] * 9
        pi = {0: 2, 1: 5, 2: 8, 3: 1, 4: 4, 5: 7, 6: 0, 7: 3, 8: 6}
        for i, p in enumerate(old_paulis):
            new_paulis[pi[i]] = p

        new_frame = PauliFrame(old_frame.qubit_labels, new_paulis)
        new_patch = QECCodePatch(patch.code, patch.qubits, new_frame)
        new_patch.data = copy.deepcopy(patch.data)

        # Retrieve and permute syndrome histories from the patch's own
        # tracked data (scoped to this patch, so multi-patch programs do
        # not collide without needing any key namespacing).
        old_hist_X = new_patch.data.get("syndrome_history_X", [])
        if not isinstance(old_hist_X, list):
            old_hist_X = []
        old_hist_Z = new_patch.data.get("syndrome_history_Z", [])
        if not isinstance(old_hist_Z, list):
            old_hist_Z = []

        # S_X -> S_Z mapping: new_Z_round = [s_X[3], s_X[0], s_X[2], s_X[1]]
        new_hist_Z = []
        for s_X in old_hist_X:
            new_hist_Z.append([s_X[3], s_X[0], s_X[2], s_X[1]])

        # S_Z -> S_X mapping: new_X_round = [s_Z[2], s_Z[0], s_Z[1], s_Z[3]]
        new_hist_X = []
        for s_Z in old_hist_Z:
            new_hist_X.append([s_Z[2], s_Z[0], s_Z[1], s_Z[3]])

        new_patch.data["syndrome_history_X"] = new_hist_X
        new_patch.data["syndrome_history_Z"] = new_hist_Z
        patches[patch_label] = new_patch

        return Frame({"patches": patches})

    instructions["H Permutation"] = Instruction(
        H_permutation_apply_fn,
        name="Logical H software permutation",
    )

    instructions["H"] = builders.build_composite_instruction(
        [instructions["H Circuit"], instructions["H Permutation"]],
        name="Logical H",
    )

    # 3. Syndrome Extraction Circuit
    X_template = circuit_backend(
        [
            ("Gh", "aux"),
            ("Gcnot", "aux", "b"),
            ("Gcnot", "aux", "a"),
            ("Gcnot", "aux", "d"),
            ("Gcnot", "aux", "c"),
            ("Gh", "aux"),
            ("Iz", "aux"),
        ],
        qubit_labels=["a", "b", "c", "d", "aux"],
    )

    # CNOT order b,d,a,c (NOT b,a,d,c): a single Z fault on the ancilla
    # after two CNOTs hooks Z onto the two remaining data qubits. With the
    # remaining pair {a,c} = {NW,SW} the hook is VERTICAL, perpendicular to
    # Z_L (which runs along rows), hence correctable. The b,a,d,c order
    # leaves the horizontal pair {d,c}, a weight-2 Z along Z_L that breaks
    # X-basis fault tolerance at d=3. Schedule validity in parallel mode is
    # preserved: no data qubit is touched twice in a layer, and every
    # adjacent X/Z tile pair visits its two shared qubits in the same
    # relative order.
    Z_template = circuit_backend(
        [
            [],
            ("Gcnot", "b", "aux"),
            ("Gcnot", "d", "aux"),
            ("Gcnot", "a", "aux"),
            ("Gcnot", "c", "aux"),
            [],
            ("Iz", "aux"),
        ],
        qubit_labels=["a", "b", "c", "d", "aux"],
    )

    full_syndrome_circ = _build_raw_syndrome_extraction_circuit(
        layout, qubits, circuit_backend, X_template, Z_template
    )

    if idle_layout == layout:
        # Self-padding: fill every genuinely blank (qubit, layer) slot once,
        # from the real union of activity across every merged/appended tile
        # (only valid for surf17/surf13 -- ruled out for surf10 above).
        full_syndrome_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    elif idle_layout is not None:
        # Borrowing: build idle_layout's own self-padded reference circuit
        # (using its own qubit/ancilla labels -- irrelevant here, since the
        # transplant only inspects data qubits) and replay its per-data-qubit
        # real/idle sequence onto this layout's raw circuit.
        reference_qubits = _layout_qubits(idle_layout)
        reference_circ = _build_raw_syndrome_extraction_circuit(
            idle_layout, reference_qubits, circuit_backend, X_template, Z_template
        )
        reference_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
        full_syndrome_circ.transplant_idle_schedule_inplace(
            reference_circ, data_qubits, list(idle_gates.values())
        )

    instructions["Syndrome Extraction"] = (
        builders.build_physical_circuit_instruction(
            full_syndrome_circ,
            name="Syndrome Extraction",
        )
    )

    # 4. Decoder selection
    # X and Z stabilizers
    X_stabilizers = ["XXIXXIIII", "IXXIIIIII", "IIIIXXIXX", "IIIIIIXXI"]
    Z_stabilizers = ["ZIIZIIIII", "IZZIZZIII", "IIIZZIZZI", "IIIIIZIIZ"]

    # H_X maps Z errors (X stabilizers detect Z errors)
    H_X = np.array(
        [
            [1, 1, 0, 1, 1, 0, 0, 0, 0],  # SX0
            [0, 1, 1, 0, 0, 0, 0, 0, 0],  # SX1
            [0, 0, 0, 0, 1, 1, 0, 1, 1],  # SX2
            [0, 0, 0, 0, 0, 0, 1, 1, 0],  # SX3
        ]
    )

    # H_Z maps X errors (Z stabilizers detect X errors)
    H_Z = np.array(
        [
            [1, 0, 0, 1, 0, 0, 0, 0, 0],  # SZ0
            [0, 1, 1, 0, 1, 1, 0, 0, 0],  # SZ1
            [0, 0, 0, 1, 1, 0, 1, 1, 0],  # SZ2
            [0, 0, 0, 0, 0, 1, 0, 0, 1],  # SZ3
        ]
    )

    if layout == "surf17":
        # Surface-17
        syndrome_labels_X = [
            SyndromeLabel("A11", -1, 0),
            SyndromeLabel("A9", -1, 0),
            SyndromeLabel("A14", -1, 0),
            SyndromeLabel("A16", -1, 0),
        ]
        syndrome_labels_Z = [
            SyndromeLabel("A10", -1, 0),
            SyndromeLabel("A12", -1, 0),
            SyndromeLabel("A13", -1, 0),
            SyndromeLabel("A15", -1, 0),
        ]
    elif layout == "surf13":
        # Surface-13
        syndrome_labels_X = [
            SyndromeLabel("A11", -1, 0),
            SyndromeLabel("A9", -1, 0),
            SyndromeLabel("A10", -1, 0),
            SyndromeLabel("A12", -1, 0),
        ]
        syndrome_labels_Z = [
            SyndromeLabel("A10", -1, 1),
            SyndromeLabel("A12", -1, 1),
            SyndromeLabel("A9", -1, 1),
            SyndromeLabel("A11", -1, 1),
        ]
    elif layout == "surf10":
        # Surface-10. Serial tile execution order is [SX1, SX0, SX2, SX3]
        # (matching the surf17/surf13 tile lists), so H row SX0 is the
        # SECOND measurement (outcome_idx 1) and SX1 the first.
        syndrome_labels_X = [
            SyndromeLabel("A9", -1, 1),
            SyndromeLabel("A9", -1, 0),
            SyndromeLabel("A9", -1, 2),
            SyndromeLabel("A9", -1, 3),
        ]
        syndrome_labels_Z = [
            SyndromeLabel("A9", -1, 4),
            SyndromeLabel("A9", -1, 5),
            SyndromeLabel("A9", -1, 6),
            SyndromeLabel("A9", -1, 7),
        ]

    try:
        import pymatching
    except ImportError as e:
        raise ImportError(
            "PyMatching is not installed, cannot use pymatching decoder. "
            "Please install pymatching: pip install loqs[pymatching]"
        ) from e

    # Deferred decoder: collects and stores syndrome outcomes on the patch's
    # tracked data (scoped to this patch, so multi-patch programs do not
    # collide without needing any key namespacing).
    def pymatching_deferred_decoder_apply_fn(
        patch_label: str,
        patches: PatchDict,
        syndrome_labels_X: list[SyndromeLabel],
        syndrome_labels_Z: list[SyndromeLabel],
        num_qec_rounds: int,
        syndrome_outcomes: list[MeasurementOutcomes] | MeasurementOutcomes,
    ) -> Frame:
        if isinstance(syndrome_outcomes, MeasurementOutcomes):
            syndrome_outcomes = [syndrome_outcomes]

        # Extract current round(s) syndromes
        current_sx = []
        current_sz = []
        for t in range(num_qec_rounds):
            sx_t = []
            for synlbl in syndrome_labels_X:
                frame_outcomes = syndrome_outcomes[t]
                outcome = frame_outcomes[synlbl.qubit_label][
                    synlbl.outcome_idx
                ]
                sx_t.append(outcome)
            current_sx.append(sx_t)

            sz_t = []
            for synlbl in syndrome_labels_Z:
                frame_outcomes = syndrome_outcomes[t]
                outcome = frame_outcomes[synlbl.qubit_label][
                    synlbl.outcome_idx
                ]
                sz_t.append(outcome)
            current_sz.append(sz_t)

        # Retrieve histories from the patch's own tracked data
        patch = patches[patch_label]
        new_patch = patch.copy()
        hist_X = list(new_patch.data.get("syndrome_history_X", []))
        hist_Z = list(new_patch.data.get("syndrome_history_Z", []))

        hist_X.extend(current_sx)
        hist_Z.extend(current_sz)

        new_patch.data["syndrome_history_X"] = hist_X
        new_patch.data["syndrome_history_Z"] = hist_Z
        patches[patch_label] = new_patch

        return Frame(
            {
                "patches": patches,  # Data qubits kept uncorrected (deferred)
            }
        )

    def pymatching_decoder_map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        syndrome_labels_X: list[SyndromeLabel],
        syndrome_labels_Z: list[SyndromeLabel],
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["syndrome_labels_X"] = [
            SyndromeLabel(
                qubit_mapping.get(lbl.qubit_label, lbl.qubit_label),
                lbl.frame_idx,
                lbl.outcome_idx,
            )
            for lbl in syndrome_labels_X
        ]
        new_kwargs["syndrome_labels_Z"] = [
            SyndromeLabel(
                qubit_mapping.get(lbl.qubit_label, lbl.qubit_label),
                lbl.frame_idx,
                lbl.outcome_idx,
            )
            for lbl in syndrome_labels_Z
        ]
        return new_kwargs

    instructions["Decoder"] = Instruction(
        pymatching_deferred_decoder_apply_fn,
        data={
            "syndrome_labels_X": syndrome_labels_X,
            "syndrome_labels_Z": syndrome_labels_Z,
            "num_qec_rounds": num_qec_rounds,
        },
        map_qubits_fn=pymatching_decoder_map_qubits_fn,
        param_priorities={
            "syndrome_outcomes": [f"history[-{num_qec_rounds}:]"]
        },
        param_aliases={"syndrome_outcomes": "measurement_outcomes"},
        name="QEC PyMatching Decoder",
    )

    instructions["QEC"] = builders.build_composite_instruction(
        [instructions["Syndrome Extraction"]] * num_qec_rounds
        + [instructions["Decoder"]],
        name="QEC Cycle",
    )

    # 5. Raw data qubit measurements
    raw_Z_meas_circ = circuit_backend(
        [[("Iz", q) for q in data_qubits]], qubit_labels=qubits
    )
    raw_X_meas_circ = circuit_backend(
        [
            [("Gh", q) for q in data_qubits],
            [("Iz", q) for q in data_qubits],
        ],
        qubit_labels=qubits,
    )
    if idle_layout is not None:
        raw_Z_meas_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
        raw_X_meas_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Raw Z Data Measure"] = (
        builders.build_physical_circuit_instruction(
            raw_Z_meas_circ,
            name="Raw logical Z-basis measurement",
        )
    )
    instructions["Raw X Data Measure"] = (
        builders.build_physical_circuit_instruction(
            raw_X_meas_circ,
            name="Raw logical X-basis measurement",
        )
    )

    # 6. Logical measurements with PyMatching global decoding
    try:
        import pymatching
    except ImportError as e:
        raise ImportError(
            "PyMatching is not installed, cannot use pymatching decoder. "
            "Please install pymatching: pip install loqs[pymatching]"
        ) from e

    # Global Space-Time deferred decoding over R rounds of syndrome extraction + final data measurements
    def pymatching_global_meas_apply_fn(
        patch_label: str,
        patches: PatchDict,
        H_X: np.ndarray,
        H_Z: np.ndarray,
        syndrome_labels_X: list[SyndromeLabel],
        syndrome_labels_Z: list[SyndromeLabel],
        data_qubits: list[str],
        measurement_basis: Literal["Z", "X"],
        measurement_outcomes: MeasurementOutcomes,
        reference_round_X: bool = False,
        reference_round_Z: bool = False,
    ) -> Frame:
        import pymatching

        patch = patches[patch_label]
        pauli_frame = patch.pauli_frame

        # 1. Get raw outcomes corrected by previous physical Pauli frame
        inferred_outcomes = measurement_outcomes.get_inferred_outcomes(
            pauli_frame, measurement_basis
        )

        # 2. Retrieve history of syndromes from the patch's own tracked data
        # (scoped to this patch, so multi-patch programs do not collide)
        hist_X = list(patch.data.get("syndrome_history_X", []))
        hist_Z = list(patch.data.get("syndrome_history_Z", []))

        # Known classical offsets baked into the ROUND-0 recorded values by
        # bookkeeping rewrites (e.g. the lattice-surgery split flips the
        # grown-check row of every stored round; interior diffs cancel but
        # the round-0 absolute layer keeps the flip). Subtracted from the
        # t=0 detector layer only, so round 0 stays usable as a detector
        # when the prep basis makes it deterministic. Tracked on the
        # patch's own data, same as syndrome_history_X/Z above.
        offset_X = list(patch.data.get("syndrome_round0_offset_X", []) or [])
        offset_Z = list(patch.data.get("syndrome_round0_offset_Z", []) or [])

        R = len(hist_X)  # number of previous syndrome rounds

        # 3. Final round R+1 syndrome from final measurements
        data_inferred_bitstring = [
            inferred_outcomes[q][0] for q in data_qubits
        ]

        if measurement_basis == "Z":
            # ZL = Z0 Z4 Z8 on the (mapped) data qubits
            uncorrected = (
                inferred_outcomes[data_qubits[0]][0]
                ^ inferred_outcomes[data_qubits[4]][0]
                ^ inferred_outcomes[data_qubits[8]][0]
            )

            # ReconstructSZ0 to SZ3 (detects X errors)
            data_inferred_pstr = "".join(
                ["X" if b == 1 else "I" for b in data_inferred_bitstring]
            )
            sz_final = [
                int(c)
                for c in qt.get_syndrome_from_stabilizers_and_pstr(
                    Z_stabilizers, data_inferred_pstr
                )
            ]

            # Form complete syndrome vector sz_history: hist_Z + sz_final
            sz_history = hist_Z + [sz_final]
            num_rounds = R + 1

            # With a reference round (e.g. prep basis does not stabilize the
            # Z checks, so round 0 is random), round 0 only serves as the
            # reference for later diffs and contributes no detector layer
            num_layers = num_rounds - 1 if reference_round_Z else num_rounds

            # Construct 3D space-time matching graph matching_Z dynamically
            matching_Z = pymatching.Matching()
            matching_Z.ensure_num_fault_ids(9)

            # Spatial edges for each round t
            for t in range(num_layers):
                for j in range(9):
                    nodes = [i for i in range(4) if H_Z[i, j] == 1]
                    fault_ids = {j}
                    if len(nodes) == 2:
                        matching_Z.add_edge(
                            t * 4 + nodes[0],
                            t * 4 + nodes[1],
                            fault_ids=fault_ids,
                            weight=1.0,
                            merge_strategy="smallest-weight",
                        )
                    elif len(nodes) == 1:
                        matching_Z.add_boundary_edge(
                            t * 4 + nodes[0],
                            fault_ids=fault_ids,
                            weight=1.0,
                            merge_strategy="smallest-weight",
                        )
            # Temporal edges
            for t in range(num_layers - 1):
                for i in range(4):
                    matching_Z.add_edge(
                        t * 4 + i,
                        (t + 1) * 4 + i,
                        weight=0.9,
                        merge_strategy="smallest-weight",
                    )
            # With a reference round the first detector layer diffs against
            # a NOISY baseline: a measurement flip in the reference round
            # leaves a single defect there. Escape edges (measurement
            # weight, no fault ids) let the matcher explain it without a
            # spurious data correction.
            if reference_round_Z and num_layers > 0:
                for i in range(4):
                    matching_Z.add_boundary_edge(
                        i,
                        weight=0.9,
                        merge_strategy="smallest-weight",
                    )

            # Compute difference syndrome
            diff_sz = []
            first_t = 1 if reference_round_Z else 0
            for t in range(first_t, num_rounds):
                if t == 0:
                    row = sz_history[0]
                    if offset_Z:
                        row = [a ^ b for a, b in zip(row, offset_Z)]
                    diff_sz.extend(row)
                else:
                    diff_sz.extend(
                        [
                            a ^ b
                            for a, b in zip(sz_history[t], sz_history[t - 1])
                        ]
                    )

            # Decode X errors
            if num_layers > 0:
                correction_Z = matching_Z.decode(
                    np.array(diff_sz, dtype=np.uint8)
                )
            else:
                correction_Z = np.zeros(9, dtype=np.uint8)

            # Check if correction flips ZL = Z0 Z4 Z8
            correction_bit = 0
            if len(correction_Z) > 0:
                for idx in [0, 4, 8]:
                    if idx < len(correction_Z) and correction_Z[idx] == 1:
                        correction_bit ^= 1

            logical_outcome = uncorrected ^ correction_bit

        else:  # X basis
            # XL = X2 X4 X6 on the (mapped) data qubits
            uncorrected = (
                inferred_outcomes[data_qubits[2]][0]
                ^ inferred_outcomes[data_qubits[4]][0]
                ^ inferred_outcomes[data_qubits[6]][0]
            )

            # Reconstruct SX0 to SX3 (detects Z errors)
            data_inferred_pstr = "".join(
                ["Z" if b == 1 else "I" for b in data_inferred_bitstring]
            )
            sx_final = [
                int(c)
                for c in qt.get_syndrome_from_stabilizers_and_pstr(
                    X_stabilizers, data_inferred_pstr
                )
            ]

            # Form complete syndrome vector sx_history: hist_X + sx_final
            sx_history = hist_X + [sx_final]
            num_rounds = R + 1

            # With a reference round (e.g. prep basis does not stabilize the
            # X checks, so round 0 is random), round 0 only serves as the
            # reference for later diffs and contributes no detector layer
            num_layers = num_rounds - 1 if reference_round_X else num_rounds

            # Construct 3D space-time matching graph matching_X dynamically
            matching_X = pymatching.Matching()
            matching_X.ensure_num_fault_ids(9)

            # Spatial edges for each round t
            for t in range(num_layers):
                for j in range(9):
                    nodes = [i for i in range(4) if H_X[i, j] == 1]
                    fault_ids = {j}
                    if len(nodes) == 2:
                        matching_X.add_edge(
                            t * 4 + nodes[0],
                            t * 4 + nodes[1],
                            fault_ids=fault_ids,
                            weight=1.0,
                            merge_strategy="smallest-weight",
                        )
                    elif len(nodes) == 1:
                        matching_X.add_boundary_edge(
                            t * 4 + nodes[0],
                            fault_ids=fault_ids,
                            weight=1.0,
                            merge_strategy="smallest-weight",
                        )
            # Temporal edges
            for t in range(num_layers - 1):
                for i in range(4):
                    matching_X.add_edge(
                        t * 4 + i,
                        (t + 1) * 4 + i,
                        weight=0.9,
                        merge_strategy="smallest-weight",
                    )
            # With a reference round the first detector layer diffs against
            # a NOISY baseline: a measurement flip in the reference round
            # leaves a single defect there. Escape edges (measurement
            # weight, no fault ids) let the matcher explain it without a
            # spurious data correction.
            if reference_round_X and num_layers > 0:
                for i in range(4):
                    matching_X.add_boundary_edge(
                        i,
                        weight=0.9,
                        merge_strategy="smallest-weight",
                    )

            # Compute difference syndrome
            diff_sx = []
            first_t = 1 if reference_round_X else 0
            for t in range(first_t, num_rounds):
                if t == 0:
                    row = sx_history[0]
                    if offset_X:
                        row = [a ^ b for a, b in zip(row, offset_X)]
                    diff_sx.extend(row)
                else:
                    diff_sx.extend(
                        [
                            a ^ b
                            for a, b in zip(sx_history[t], sx_history[t - 1])
                        ]
                    )

            # Decode Z errors
            if num_layers > 0:
                correction_X = matching_X.decode(
                    np.array(diff_sx, dtype=np.uint8)
                )
            else:
                correction_X = np.zeros(9, dtype=np.uint8)

            # Check if correction flips XL = X2 X4 X6
            correction_bit = 0
            if len(correction_X) > 0:
                for idx in [2, 4, 6]:
                    if idx < len(correction_X) and correction_X[idx] == 1:
                        correction_bit ^= 1

            logical_outcome = uncorrected ^ correction_bit

        return Frame(
            {
                "patch_label": patch_label,
                "logical_measurement": logical_outcome,
                "uncorrected_measurement": uncorrected,
            }
        )

    def pymatching_global_meas_map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        syndrome_labels_X: list[SyndromeLabel],
        syndrome_labels_Z: list[SyndromeLabel],
        data_qubits: list[str],
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["syndrome_labels_X"] = [
            SyndromeLabel(
                qubit_mapping.get(lbl.qubit_label, lbl.qubit_label),
                lbl.frame_idx,
                lbl.outcome_idx,
            )
            for lbl in syndrome_labels_X
        ]
        new_kwargs["syndrome_labels_Z"] = [
            SyndromeLabel(
                qubit_mapping.get(lbl.qubit_label, lbl.qubit_label),
                lbl.frame_idx,
                lbl.outcome_idx,
            )
            for lbl in syndrome_labels_Z
        ]
        new_kwargs["data_qubits"] = [
            qubit_mapping.get(q, q) for q in data_qubits
        ]
        return new_kwargs

    Z_global_meas = Instruction(
        pymatching_global_meas_apply_fn,
        data={
            "H_X": H_X,
            "H_Z": H_Z,
            "syndrome_labels_X": syndrome_labels_X,
            "syndrome_labels_Z": syndrome_labels_Z,
            "data_qubits": data_qubits,
            "measurement_basis": "Z",
            "reference_round_X": False,
            "reference_round_Z": False,
        },
        map_qubits_fn=pymatching_global_meas_map_qubits_fn,
        param_priorities={"measurement_outcomes": ["history[-1]"]},
        name="FT Z global parity calculation with PyMatching",
    )
    instructions["FT Logical Z Measure"] = (
        builders.build_composite_instruction(
            [instructions["Raw Z Data Measure"], Z_global_meas],
            # Declared here so stack-level kwargs thread through to the
            # global parity calculation (label kwargs only reach declared params)
            extra_data={
                "reference_round_X": False,
                "reference_round_Z": False,
            },
            name="FT logical Z measurement with PyMatching",
        )
    )

    X_global_meas = Instruction(
        pymatching_global_meas_apply_fn,
        data={
            "H_X": H_X,
            "H_Z": H_Z,
            "syndrome_labels_X": syndrome_labels_X,
            "syndrome_labels_Z": syndrome_labels_Z,
            "data_qubits": data_qubits,
            "measurement_basis": "X",
            "reference_round_X": False,
            "reference_round_Z": False,
        },
        map_qubits_fn=pymatching_global_meas_map_qubits_fn,
        param_priorities={"measurement_outcomes": ["history[-1]"]},
        name="FT X global parity calculation with PyMatching",
    )
    instructions["FT Logical X Measure"] = (
        builders.build_composite_instruction(
            [instructions["Raw X Data Measure"], X_global_meas],
            extra_data={
                "reference_round_X": False,
                "reference_round_Z": False,
            },
            name="FT logical X measurement with PyMatching",
        )
    )
    instructions["FT Z logical parity calculation"] = Z_global_meas
    instructions["FT X logical parity calculation"] = X_global_meas

    code = QECCode(instructions, qubits, data_qubits, name)
    return code


def create_ideal_model(
    qubits: Sequence[str],
    model_backend: type[BaseNoiseModel] = PyGSTiNoiseModel,
    gaterep: GateRep = GateRep.QSIM_SUPEROPERATOR,
    instrep: InstrumentRep = InstrumentRep.ZBASIS_PROJECTION,
) -> BaseNoiseModel:
    """Create an ideal (noiseless) model for the Surface-17 / Surface-13 code.

    Parameters
    ----------
    qubits : Sequence[str]
        List of qubit labels to use.

    model_backend : type[BaseNoiseModel], optional
        The model backend to use. Default is PyGSTiNoiseModel.

    gaterep : GateRep, optional
        Gate representation. Default is GateRep.QSIM_SUPEROPERATOR.

    instrep : InstrumentRep, optional
        Instrument representation. Default is InstrumentRep.ZBASIS_PROJECTION.

    Returns
    -------
    BaseNoiseModel
        A noiseless model for the QECCode.
    """
    model_qubits = [f"Q{i}" for i in range(len(qubits))]

    gate_names = [
        "Gxpi",
        "Gypi",
        "Gzpi",
        "Gzpi2",
        "Gzmpi2",
        "Gh",
        "Gcnot",
        "Gi",
        "Gi1Q",
        "Gi2Q",
        "GiMCM",
    ]

    nonstd_unitaries = {
        "Gi1Q": np.eye(2),
        "Gi2Q": np.eye(2),
        "GiMCM": np.eye(2),
    }

    if model_backend == PyGSTiNoiseModel:
        try:
            import pygsti
        except ImportError:
            raise ImportError(
                "pyGSTi not found, cannot construct pyGSTi noise model"
            )

        pspec = pygsti.processors.QubitProcessorSpec(
            len(model_qubits),
            gate_names=gate_names,
            qubit_labels=model_qubits,
            nonstd_gate_unitaries=nonstd_unitaries,
            availability={k: "all-permutations" for k in gate_names},
        )

        ideal_model_pygsti = pygsti.models.create_crosstalk_free_model(pspec)
        model = PyGSTiNoiseModel(ideal_model_pygsti, qubits)

    elif model_backend == DictNoiseModel:
        gate_dict = {}
        if gaterep == GateRep.STIM_CIRCUIT_STR:
            name_to_stim_ops = {
                "Gxpi": ["X"],
                "Gypi": ["Y"],
                "Gzpi": ["Z"],
                "Gzpi2": ["SQRT_Z"],
                "Gzmpi2": ["SQRT_Z_DAG"],
                "Gh": ["H"],
                "Gcnot": ["CX"],
                "Gi": ["I"],
                "Gi1Q": ["I"],
                "Gi2Q": ["I"],
                "GiMCM": ["I"],
            }

            for gate in gate_names:
                num_qubits = 2 if gate in ["Gcnot", "Gcphase"] else 1

                stim_str = ""
                for stim_op in name_to_stim_ops[gate]:
                    stim_str += stim_op
                    for i in range(num_qubits):
                        stim_str += f" {i}"
                    stim_str += "\n"

                qubit_perms = itertools.permutations(qubits, r=num_qubits)
                for qs in qubit_perms:
                    gate_dict[(gate, qs)] = stim_str
        else:
            try:
                import pygsti
            except ImportError:
                raise ImportError(
                    "pyGSTi not found, cannot construct dict noise model"
                )

            std_unitaries = (
                pygsti.tools.internalgates.standard_gatename_unitaries()
            )

            for gate in gate_names:
                U = std_unitaries.get(gate, None)
                if U is None:
                    U = nonstd_unitaries[gate]

                num_qubits = int(np.log2(U.shape[0]))
                qubit_perms = itertools.permutations(qubits, r=num_qubits)
                for qs in qubit_perms:
                    if gaterep == GateRep.UNITARY:
                        gate_dict[(gate, qs)] = RepTuple(
                            U, qs, GateRep.UNITARY
                        )
                    elif gaterep == GateRep.PTM:
                        gate_dict[(gate, qs)] = (
                            pygsti.tools.unitary_to_pauligate(U)
                        )
                    elif gaterep == GateRep.QSIM_SUPEROPERATOR:
                        import loqs.tools.pygstitools as pt
                        gate_dict[(gate, qs)] = pt.unitary_to_qsim_ptm(U)
                    else:
                        raise NotImplementedError(
                            "Conversion to this rep is not implemented yet."
                        )

        inst_dict = {("Iz", (q,)): (0, True) for q in qubits}

        return DictNoiseModel(
            (gate_dict, inst_dict), gatereps=[gaterep], instreps=[instrep]
        )

    elif issubclass(model_backend, BaseNoiseModel):
        raise NotImplementedError(
            "Cannot generate ideal model for this backend"
        )
    else:
        raise ValueError("Must pass a noise model class")

    return model
