#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
######################################################################################################################

"""A `LoQS` QEC codepack for the Surface-17 and Surface-13 codes.

This implementation is based on the 2014 implementation from
Tomita & Svore in [@tomita_lowdistance_2014].
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
import loqs.tools.pygstitools as pt
import loqs.tools.qectools as qt


def create_qec_code(
    auxiliary_reuse: bool = False,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
) -> QECCode:
    """Create a QECCode implementing the Surface-17 or Surface-13 code.

    Parameters
    ----------
    auxiliary_reuse : bool, optional
        Whether to implement Surface-13 with auxiliary qubit reuse (True)
        or standard Surface-17 (False, default).

    include_idles : bool, optional
        Whether to include (True) or not (False, default) idle gates
        in physical circuits.

    gate_durations : dict[str, int | float] | None, optional
        Mapping from gate names to durations.

    idle_gates : dict[int | float, str] | None, optional
        Mapping from gate duration to idle gate names.

    circuit_backend : type[BasePhysicalCircuit], optional
        The circuit backend to use when generating physical circuits.
        Default is PyGSTiPhysicalCircuit.

    Returns
    -------
    QECCode
        A QECCode implementing the surface code.
    """
    if auxiliary_reuse:
        qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 13)]
        name = "Surface-13 Code"
    else:
        qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 17)]
        name = "Surface-17 Code"

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
    if include_idles:
        raw_Z_prep_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Zero Prep"] = builders.build_physical_circuit_instruction(
        raw_Z_prep_circ,
        name="Logical zero state prep",
    )

    # X-basis preparation (|+>_L)
    raw_X_prep_circ = circuit_backend(
        [
            [("Iz", q) for q in data_qubits],
            [("Gh", q) for q in data_qubits]
        ],
        qubit_labels=qubits
    )
    if include_idles:
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
    if include_idles:
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
    if include_idles:
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
    if include_idles:
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
            [("Gypi", "D4")]
        ],
        qubit_labels=qubits
    )
    if include_idles:
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
    if include_idles:
        logical_H_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["H Circuit"] = builders.build_physical_circuit_instruction(
        logical_H_circ,
        name="Logical H physical circuit",
    )

    # Software permutation for H (clockwise 90-degree rotation of the 3x3 data grid)
    # 0->2, 1->5, 2->8, 3->1, 4->4, 5->7, 6->0, 7->3, 8->6
    def H_permutation_apply_fn(patches: PatchDict, patch_label: str) -> Frame:
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
    X_template = circuit_backend([
        ('Gh', 'aux'),
        ('Gcnot', 'aux', 'b'),
        ('Gcnot', 'aux', 'a'),
        ('Gcnot', 'aux', 'd'),
        ('Gcnot', 'aux', 'c'),
        ('Gh', 'aux'),
        ('Iz', 'aux')
    ], qubit_labels=['a', 'b', 'c', 'd', 'aux'])

    Z_template = circuit_backend([
        [],
        ('Gcnot', 'b', 'aux'),
        ('Gcnot', 'a', 'aux'),
        ('Gcnot', 'd', 'aux'),
        ('Gcnot', 'c', 'aux'),
        [],
        ('Iz','aux')
    ], qubit_labels=['a', 'b', 'c', 'd', 'aux'])

    if include_idles:
        X_template.pad_single_qubit_idles_by_duration_inplace(idle_gates, gate_durations)
        Z_template.pad_single_qubit_idles_by_duration_inplace(idle_gates, gate_durations)

    if not auxiliary_reuse:
        # Surface-17
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
        full_syndrome_circ = X_syndrome.merge(Z_syndrome, 0)
    else:
        # Surface-13
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
        full_syndrome_circ = X_syndrome.append(Z_syndrome)

    instructions["Syndrome Extraction"] = builders.build_physical_circuit_instruction(
        full_syndrome_circ,
        name="Syndrome Extraction",
    )

    # 4. Lookup Decoder
    stabilizers = [
        "XXIXXIIII",  # SX0: X0 X1 X3 X4
        "IXXIIIIII",  # SX1: X1 X2
        "IIIIXXIXX",  # SX2: X4 X5 X7 X8
        "IIIIIIXXI",  # SX3: X6 X7
        "ZIIZIIIII",  # SZ0: Z0 Z3
        "IZZIZZIII",  # SZ1: Z1 Z2 Z4 Z5
        "IIIZZIZZI",  # SZ2: Z3 Z4 Z6 Z7
        "IIIIIZIIZ",  # SZ3: Z5 Z8
    ]

    errors = qt.get_weight_1_errors(9)
    raw_lookup = qt.get_syndrome_dict_from_stabilizers_and_pstrs(stabilizers, errors)
    lookup_table = {
        syndrome: pstrs[0] for syndrome, pstrs in raw_lookup.items()
    }

    if not auxiliary_reuse:
        # Surface-17
        syndrome_labels = [
            SyndromeLabel("A11", -1, 0),  # SX0
            SyndromeLabel("A9",  -1, 0),  # SX1
            SyndromeLabel("A14", -1, 0),  # SX2
            SyndromeLabel("A16", -1, 0),  # SX3
            SyndromeLabel("A10", -1, 0),  # SZ0
            SyndromeLabel("A12", -1, 0),  # SZ1
            SyndromeLabel("A13", -1, 0),  # SZ2
            SyndromeLabel("A15", -1, 0),  # SZ3
        ]
    else:
        # Surface-13
        syndrome_labels = [
            SyndromeLabel("A11", -1, 0),  # SX0
            SyndromeLabel("A9",  -1, 0),  # SX1
            SyndromeLabel("A10", -1, 0),  # SX2
            SyndromeLabel("A12", -1, 0),  # SX3
            SyndromeLabel("A10", -1, 1),  # SZ0
            SyndromeLabel("A12", -1, 1),  # SZ1
            SyndromeLabel("A9",  -1, 1),  # SZ2
            SyndromeLabel("A11", -1, 1),  # SZ3
        ]

    instructions["Decoder"] = builders.build_lookup_decoder_instruction(
        lookup_table=lookup_table,
        syndrome_labels=syndrome_labels,
        raw_syndrome_frame_key="latest_syndrome",
        diff_prev_syndrome=True,
        name="QEC Lookup Decoder",
    )

    instructions["QEC"] = builders.build_composite_instruction(
        [instructions["Syndrome Extraction"], instructions["Decoder"]],
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
    if include_idles:
        raw_Z_meas_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
        raw_X_meas_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Raw Z Data Measure"] = builders.build_physical_circuit_instruction(
        raw_Z_meas_circ,
        name="Raw logical Z-basis measurement",
    )
    instructions["Raw X Data Measure"] = builders.build_physical_circuit_instruction(
        raw_X_meas_circ,
        name="Raw logical X-basis measurement",
    )

    # 6. Logical measurements with classical decoding
    z_stabilizers = ["ZIIZIIIII", "IZZIZZIII", "IIIZZIZZI", "IIIIIZIIZ"]
    z_errors = []
    for i in range(9):
        err = ["I"] * 9
        err[i] = "X"
        z_errors.append("".join(err))
    z_lookup_raw = qt.get_syndrome_dict_from_stabilizers_and_pstrs(z_stabilizers, z_errors)
    z_lookup = {syn: pstrs[0] for syn, pstrs in z_lookup_raw.items()}

    x_stabilizers = ["XXIXXIIII", "IXXIIIIII", "IIIIXXIXX", "IIIIIIXXI"]
    x_errors = []
    for i in range(9):
        err = ["I"] * 9
        err[i] = "Z"
        x_errors.append("".join(err))
    x_lookup_raw = qt.get_syndrome_dict_from_stabilizers_and_pstrs(x_stabilizers, x_errors)
    x_lookup = {syn: pstrs[0] for syn, pstrs in x_lookup_raw.items()}

    def logical_meas_apply_fn(
        patch_label: str,
        patches: PatchDict,
        data_qubits: list[str],
        measurement_basis: Literal["Z", "X"],
        measurement_outcomes: MeasurementOutcomes,
    ) -> Frame:
        patch = patches[patch_label]
        pauli_frame = patch.pauli_frame
        
        # Get inferred outcomes corrected by physical Pauli frame
        inferred_outcomes = measurement_outcomes.get_inferred_outcomes(pauli_frame, measurement_basis)
        
        if measurement_basis == "Z":
            # Uncorrected logical Z
            uncorrected = inferred_outcomes["D0"][0] ^ inferred_outcomes["D4"][0] ^ inferred_outcomes["D8"][0]
            
            # Reconstruct Z-stabilizers
            classical_syndrome = [
                (inferred_outcomes["D0"][0] ^ inferred_outcomes["D3"][0]),
                (inferred_outcomes["D1"][0] ^ inferred_outcomes["D2"][0] ^ inferred_outcomes["D4"][0] ^ inferred_outcomes["D5"][0]),
                (inferred_outcomes["D3"][0] ^ inferred_outcomes["D4"][0] ^ inferred_outcomes["D6"][0] ^ inferred_outcomes["D7"][0]),
                (inferred_outcomes["D5"][0] ^ inferred_outcomes["D8"][0]),
            ]
            syndrome_str = "".join([str(s) for s in classical_syndrome])
            
            # Look up correction
            correction_pstr = z_lookup.get(syndrome_str, "I" * 9)
            
            # Check if correction flips the logical outcome
            correction_bit = int((correction_pstr[0] == "X") ^ (correction_pstr[4] == "X") ^ (correction_pstr[8] == "X"))
            
        else:  # X basis
            # Uncorrected logical X
            uncorrected = inferred_outcomes["D2"][0] ^ inferred_outcomes["D4"][0] ^ inferred_outcomes["D6"][0]
            
            # Reconstruct X-stabilizers
            classical_syndrome = [
                (inferred_outcomes["D0"][0] ^ inferred_outcomes["D1"][0] ^ inferred_outcomes["D3"][0] ^ inferred_outcomes["D4"][0]),
                (inferred_outcomes["D1"][0] ^ inferred_outcomes["D2"][0]),
                (inferred_outcomes["D4"][0] ^ inferred_outcomes["D5"][0] ^ inferred_outcomes["D7"][0] ^ inferred_outcomes["D8"][0]),
                (inferred_outcomes["D6"][0] ^ inferred_outcomes["D7"][0]),
            ]
            syndrome_str = "".join([str(s) for s in classical_syndrome])
            
            # Look up correction
            correction_pstr = x_lookup.get(syndrome_str, "I" * 9)
            
            # Check if correction flips the logical outcome
            correction_bit = int((correction_pstr[2] == "Z") ^ (correction_pstr[4] == "Z") ^ (correction_pstr[6] == "Z"))
            
        logical_outcome = uncorrected ^ correction_bit
        
        return Frame({
            "patch_label": patch_label,
            "logical_measurement": logical_outcome,
            "uncorrected_measurement": uncorrected,
            "classical_syndrome": classical_syndrome,
            "correction_pstr": correction_pstr,
        })

    def logical_meas_map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        data_qubits: list[str],
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["data_qubits"] = [qubit_mapping[q] for q in data_qubits]
        return new_kwargs

    Z_logical_meas = Instruction(
        logical_meas_apply_fn,
        data={"data_qubits": data_qubits, "measurement_basis": "Z"},
        map_qubits_fn=logical_meas_map_qubits_fn,
        name="FT Z logical parity calculation",
    )
    instructions["FT Logical Z Measure"] = builders.build_composite_instruction(
        [instructions["Raw Z Data Measure"], Z_logical_meas],
        name="FT logical Z measurement",
    )

    X_logical_meas = Instruction(
        logical_meas_apply_fn,
        data={"data_qubits": data_qubits, "measurement_basis": "X"},
        map_qubits_fn=logical_meas_map_qubits_fn,
        name="FT X logical parity calculation",
    )
    instructions["FT Logical X Measure"] = builders.build_composite_instruction(
        [instructions["Raw X Data Measure"], X_logical_meas],
        name="FT logical X measurement",
    )

    instructions["FT Z logical parity calculation"] = Z_logical_meas
    instructions["FT X logical parity calculation"] = X_logical_meas

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
