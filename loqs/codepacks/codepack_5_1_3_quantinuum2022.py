#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""A LoQS QEC codepack for the 5-qubit code.

This implementation is based on the 2022 implementation from
Quantinuum in :cite:`ryananderson_implementing_2022`, which is in turn
based on piecewise fault-tolerance from :cite:`yoder_universal_2016`
and flag fault-tolerance from :cite:`chao_quantum_2018`.

As we are using flag fault-tolerance, we require two auxiliary qubits:
a measurement qubit for stabilizer checks, and an additional flag qubit.
Thus, we will have 7 qubits total: 5 data and 2 auxiliary.

.. bibliography::
    :filter: docname in docnames
"""

from collections.abc import Sequence
import itertools
from typing import Mapping
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
from loqs.core import Instruction, QECCode
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import KwargDict
from loqs.core.instructions.instructionlabel import (
    InstructionLabel,
    InstructionLabelCastableTypes,
)
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict
import loqs.tools.pygstitools as pt
import loqs.tools.qectools as qt


def create_qec_code(
    ft_state_prep_max_repeats: int = 100,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
):
    """Create a QECCode implementing the [[5,1,3]] code.
    
    Parameters
    ----------
    ft_state_prep_max_repeats : int, optional
        The number of max repeats to include in the repeat-until-success
        fault-tolerant state prep instruction. Default is 100.
    
    include_idles : bool, optional
        Whether to include (True) or not (False, default) idle gates
        in physical circuits.
    
    gate_durations : dict[str, int | float] | None, optional
        Mapping from gate names to durations. Defaults to None, which uses
        dummy values 1, 2, 3 for 1Q gates, 2Q gates, and mid-circuit
        measurements, respectively.
        See ``durations`` from
        (BasePhysicalCircuit.pad_single_qubit_idles_by_duration_inplace)[api:BasePhysicalCircuit.pad_single_qubit_idles_by_duration_inplace]
        for more details.
    
    idle_gates : dict[int | float, str] | None, optional
        Mapping from gate duration to idle gate names. Defaults to None,
        which maps the dummy values from ``gate_durations`` to ``"Gi1Q"``,
        ``"Gi2Q"``, and ``"GiMCM"``, respectively.
        See ``idle_names`` from
        (BasePhysicalCircuit.pad_single_qubit_idles_by_duration_inplace)[api:BasePhysicalCircuit.pad_single_qubit_idles_by_duration_inplace]
        for more details.
    
    circuit_backend : type[BasePhysicalCircuit], optional
        The circuit backend to use when generating physical circuits.
        Default is (PyGSTiPhysicalCircuit)[api:PyGSTiPhysicalCircuit].
    
    Returns
    -------
    QECCode
        A (QECCode)[api:QECCode] implementing the [[5,1,3]] code.
    """

    # Template qubits for defining one patch
    qubits = ["A0", "A1"] + [f"D{i}" for i in range(5)]
    data_qubits = qubits[2:]

    instructions: dict[str, Instruction] = {}

    # For padding by idles with duration
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
                "Gk",
            ]
        }
        gate_durations["Gcnot"] = 2
        gate_durations["Gcphase"] = 2
        gate_durations["Gi2Q"] = 2
        gate_durations["Iz"] = 3
        gate_durations["GiMCM"] = 3
    if idle_gates is None:
        idle_gates = {1: "Gi1Q", 2: "Gi2Q", 3: "GiMCM"}

    # Non-FT |-> state prep
    # First gray box of Fig 3 of arxiv:2208.01863
    nonft_state_prep_circ = circuit_backend(
        [
            [
                ("Gh", "D0"),
                ("Gh", "D1"),
                ("Gh", "D2"),
                ("Gh", "D3"),
                ("Gh", "D4"),
            ],
            [("Gcphase", "D0", "D1"), ("Gcphase", "D2", "D3")],
            [("Gcphase", "D1", "D2"), ("Gcphase", "D3", "D4")],
            [("Gcphase", "D0", "D4")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        nonft_state_prep_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Non-FT Minus Prep"] = (
        builders.build_physical_circuit_instruction(
            nonft_state_prep_circ,
            name="Non-FT minus state prep",
        )
    )

    ### Try-until-success FT |-> state prep
    # Qubit reset in case of failure
    reset = builders.build_physical_circuit_instruction(
        circuit_backend(
            [[("Iz", q) for q in qubits[2:]]], qubit_labels=qubits
        ),
        name="Reset to all 0 state",
    )

    # First green box of Fig 3 of arxiv:2208.01863
    ft_state_prep_checks_circ = circuit_backend(
        [
            # FT check 1
            ("Gh", "A0"),
            ("Gcnot", "A0", "D0"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D1"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D4"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
            # FT check 2
            ("Gh", "A0"),
            ("Gcphase", "A0", "D0"),
            ("Gcnot", "A0", "A1"),
            ("Gcnot", "A0", "D1"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D2"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
            # FT check 3
            ("Gh", "A0"),
            ("Gcnot", "A0", "D3"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D2"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D4"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        ft_state_prep_checks_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    ft_state_prep_circ = nonft_state_prep_circ.append(
        ft_state_prep_checks_circ
    )
    instructions["Non-FT Minus Prep + Checks"] = (
        builders.build_physical_circuit_instruction(
            ft_state_prep_circ,
            name="Non-FT Minus Prep + Checks",
        )
    )

    # On success, we expect three sets of 0 outcomes on the flag qubits from the check circuit
    rus_success_expected = MeasurementOutcomes(
        {"A0": [0, 0, 0], "A1": [0, 0, 0]}
    )

    instructions["FT Minus Prep"] = (
        builders.build_repeat_until_success_instruction(
            [reset, instructions["Non-FT Minus Prep + Checks"]],
            rus_key="FT Minus Prep",
            test_frame_key="measurement_outcomes",
            expected=rus_success_expected,
            max_repeats=ft_state_prep_max_repeats,
            name="Repeat-until-success FT Minus Prep",
        )
    )

    # Logical X (transversal)
    # Eqn B1 of arxiv:2208.01863
    logical_X_circ = circuit_backend(
        [[("Gypi", "D0"), ("Gxpi", "D2"), ("Gypi", "D4")]],
        qubit_labels=qubits,
    )
    if include_idles:
        logical_X_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["X"] = builders.build_physical_circuit_instruction(
        logical_X_circ,
        name="Logical X",
    )

    # Logical Z (transversal)
    # Eqn B3 of arxiv:2208.01863
    logical_Z_circ = circuit_backend(
        [[("Gxpi", "D0"), ("Gzpi", "D2"), ("Gxpi", "D4")]],
        qubit_labels=qubits,
    )
    if include_idles:
        logical_Z_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Z"] = builders.build_physical_circuit_instruction(
        logical_Z_circ,
        name="Logical Z",
    )

    # Logical K (transversal)
    # Fig 2a of arxiv:1603.03948
    logical_K_circ = circuit_backend(
        [[("Gk", q) for q in qubits[2:]]], qubit_labels=qubits
    )
    if include_idles:
        logical_K_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["K"] = builders.build_physical_circuit_instruction(
        logical_K_circ,
        pauli_frame_update="K",
        name="Logical K",
    )

    # Logical H (transversal + permute)
    # Fig 2b of arxiv:1603.03948
    logical_H_circ = circuit_backend(
        [[("Gh", q) for q in qubits[2:]]], qubit_labels=qubits
    )
    if include_idles:
        logical_H_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Logical H circuit"] = (
        builders.build_physical_circuit_instruction(
            logical_H_circ,
            pauli_frame_update="H",
            name="Logical H circuit",
        )
    )
    instructions["Logical H permutation"] = (
        builders.build_patch_permute_instruction(
            {"D0": "D3", "D1": "D0", "D3": "D4", "D4": "D1"},  # initial: final
            name="Logical H permutation",
        )
    )

    instructions["H"] = builders.build_composite_instruction(
        [
            instructions["Logical H circuit"],
            instructions["Logical H permutation"],
        ],
        name="Logical H",
    )

    logical_I_circ = circuit_backend(
        [[("Gi", q) for q in qubits[2:]]], qubit_labels=qubits
    )
    if include_idles:
        logical_I_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["I"] = builders.build_physical_circuit_instruction(
        logical_I_circ,
        name="Logical I",
    )

    # Rotations to and from the "prime" basis
    # "Local Clifford rotation" gray boxes from Fig 3 of arxiv:2208.01863
    to_prime_basis_circ = circuit_backend(
        [
            [("Gh", "D0"), ("Gypi", "D2"), ("Gh", "D4")],
            [("Gzpi2", "D0"), ("Gzpi2", "D4")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        to_prime_basis_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Logical Prime Basis Transform"] = (
        builders.build_physical_circuit_instruction(
            to_prime_basis_circ,
            name="Local Clifford rotation to prime basis",
        )
    )

    from_prime_basis_circ = circuit_backend(
        [
            [("Gzmpi2", "D0"), ("Gypi", "D2"), ("Gzmpi2", "D4")],
            [("Gh", "D0"), ("Gh", "D4")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        from_prime_basis_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Logical Prime Basis Inverse Transform"] = (
        builders.build_physical_circuit_instruction(
            from_prime_basis_circ,
            name="Local Clifford rotation out of prime basis",
        )
    )

    # In the prime basis, Zbar = ZIZIZ and Xbar = XIXIX
    # So we can do physical Z/X measurements and it makes sense to do so
    raw_Z_meas_circ = circuit_backend(
        [[("Iz", "D0"), ("Iz", "D2"), ("Iz", "D4")]], qubit_labels=qubits
    )
    raw_X_meas_circ = circuit_backend(
        [
            [("Gh", "D0"), ("Gh", "D2"), ("Gh", "D4")],
            [("Iz", "D0"), ("Iz", "D2"), ("Iz", "D4")],
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

    prime_basis_Z_meas = builders.build_physical_circuit_instruction(
        to_prime_basis_circ.append(raw_Z_meas_circ),
        name="Raw logical Z-basis measurement",
    )

    prime_basis_X_meas = builders.build_physical_circuit_instruction(
        to_prime_basis_circ.append(raw_X_meas_circ),
        name="Raw logical X-basis measurement",
    )

    # We can also compute the logical measurement based on the raw logical output
    # In the prime basis, an odd number of 0s is 0 and an odd number of 1s is 1
    # This is because our logical operations are ZIZIZ and XIXIX in the prime basis
    # E.g. all 0s is 0 for ZIZIZ, but so is one 0 and two 1s because the phase on the ones cancels
    # We can define a new operation here which post processes the measurement outcomes of Raw Logical * Measure
    def nonft_logical_meas_apply_fn(
        patch_label: str,
        patches: PatchDict,
        measurement_outcomes: MeasurementOutcomes,
    ) -> Frame:
        """Apply function for non-fault-tolerant logical measurement.
        
        Computes the logical measurement outcome from raw measurement outcomes
        using the Pauli frame to infer corrections. This function implements
        the non-fault-tolerant logical measurement for the [[5,1,3]] code.
        
        The logical measurement is computed by summing the inferred bitstring
        values modulo 2, where an odd parity corresponds to logical 0 and
        even parity corresponds to logical 1.
        
        Parameters
        ----------
        patch_label : str
            Label of the patch being measured.
        patches : PatchDict
            Dictionary of patches containing Pauli frame information for
            tracking Pauli corrections.
        measurement_outcomes : MeasurementOutcomes
            Raw measurement outcomes from the physical qubits, used to infer
            corrected outcomes based on the Pauli frame.
        
        Returns
        -------
        Frame
            Frame containing the computed logical measurement value:
            - logical_measurement : int
                The final logical measurement value (0 or 1) after applying
                Pauli frame corrections and parity computation.
        
        Notes
        -----
        This function is used in composite instructions for non-fault-tolerant
        logical Z and X measurements after transforming to the prime basis.
        """
        # Get pauli frame
        pauli_frame = patches[patch_label].pauli_frame

        # Compute inferred bitstring
        inferred_outcomes = measurement_outcomes.get_inferred_outcomes(
            pauli_frame, "Z"
        )
        inferred_bitstring = [v[0] for v in inferred_outcomes.values()]

        logical_value = sum(inferred_bitstring) % 2
        return Frame({"logical_measurement": logical_value})

    nonft_logical_meas = Instruction(
        nonft_logical_meas_apply_fn,
        name="Non-FT Logical Measurement",
    )

    instructions["Non-FT Logical Z Measure"] = (
        builders.build_composite_instruction(
            [prime_basis_Z_meas, nonft_logical_meas],
            name="Non-FT logical Z measurement (via prime basis measurement)",
        )
    )

    instructions["Non-FT Logical X Measure"] = (
        builders.build_composite_instruction(
            [prime_basis_X_meas, nonft_logical_meas],
            name="Non-FT logical X measurement (via prime basis measurement)",
        )
    )

    ### DECODING CIRCUIT
    state_decoder_circ = circuit_backend(
        [
            [("Gcphase", "D0", "D4")],
            [("Gcphase", "D1", "D2"), ("Gcphase", "D3", "D4")],
            [("Gcphase", "D0", "D1"), ("Gcphase", "D2", "D3")],
            [
                ("Gh", "D0"),
                ("Gh", "D1"),
                ("Gh", "D2"),
                ("Gh", "D3"),
                ("Gh", "D4"),
            ],
            [
                ("Iz", "D0"),
                ("Iz", "D1"),
                ("Iz", "D2"),
                ("Iz", "D3"),
                ("Iz", "D4"),
            ],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        state_decoder_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Non-FT Minus Unprep"] = (
        builders.build_physical_circuit_instruction(
            state_decoder_circ,
            name="Non-FT state decoder circuit",
        )
    )

    # Fig 13 of arxiv:2208.01863
    # Adds the instructions in-place
    _create_adaptive_measure_instruction(
        instructions,
        qubits,
        include_idles,
        gate_durations,
        idle_gates,
        circuit_backend,
    )

    ## Stabilizer circuits (for debugging)
    _create_stabilizer_instructions(
        instructions,
        qubits,
        include_idles,
        gate_durations,
        idle_gates,
        circuit_backend,
    )

    ## QEC
    _create_unflagged_QEC_instruction(
        instructions,
        qubits,
        include_idles,
        gate_durations,
        idle_gates,
        circuit_backend,
    )
    _create_flagged_QEC_instruction(
        instructions,
        qubits,
        include_idles,
        gate_durations,
        idle_gates,
        circuit_backend,
    )

    code = QECCode(instructions, qubits, data_qubits, "Perfect [[5,1,3]] code")
    return code


def create_ideal_model(  # noqa: C901
    qubits: Sequence[str],
    model_backend: type[BaseNoiseModel] = PyGSTiNoiseModel,
    gaterep: GateRep = GateRep.QSIM_SUPEROPERATOR,
    instrep: InstrumentRep = InstrumentRep.ZBASIS_PROJECTION,
):
    """Create an ideal (i.e. noiseless) model for the [[5,1,3]] code.
    
    This model will contain all the instructions needed to run the
    physical circuits in the (QECCode)[api:QECCode] returned by (create_qec_code)[api:create_qec_code].
    
    Parameters
    ----------
    qubits : Sequence[str]
        List of qubit labels to use. It should be have 7 entries,
        and the first two qubits should be the auxiliary qubits.
    
    model_backend : type[BaseNoiseModel], optional
        The model backend to use when generating operations.
        Currently, only (PyGSTiNoiseModel)[api:PyGSTiNoiseModel] is allowed.
        Default is (PyGSTiNoiseModel)[api:PyGSTiNoiseModel].
    
    gaterep : GateRep, optional
        Gate representation to use. Default is GateRep.QSIM_SUPEROPERATOR.
    
    instrep : InstrumentRep, optional
        Instrument representation to use. Default is InstrumentRep.ZBASIS_PROJECTION.
    
    Returns
    -------
    BaseNoiseModel
        A noiseless model for the (QECCode)[api:QECCode] returned by
        (create_qec_code)[api:create_qec_code].
    """
    assert len(qubits) == 7, "Must provide exactly 7 qubit labels"
    model_qubits = [f"Q{i}" for i in range(7)]

    gate_names = [
        "Gxpi",
        "Gypi",
        "Gzpi",
        "Gzpi2",
        "Gzmpi2",
        "Gh",
        "Gk",
        "Gcnot",
        "Gcphase",
        "Gi",
        "Gi1Q",
        "Gi2Q",
        "GiMCM",
    ]

    nonstd_unitaries = {
        "Gk": np.array(
            [
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
                [1j / np.sqrt(2), -1j / np.sqrt(2)],
            ]
        ),
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

        # TODO: Instrument not specified here
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
                "Gk": ["H", "SQRT_Z"],
                "Gcnot": ["CX"],
                "Gcphase": ["CZ"],
                "Gi": ["I"],
                "Gi1Q": ["I"],
                "Gi2Q": ["I"],
                "GiMCM": ["I"],
            }

            for gate in gate_names:
                num_qubits = 2 if gate in ["Gcnot", "Gcphase"] else 1

                # For stim strings, all the representations are "local"
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
            # Currently we use pyGSTi to look up definitions for dense reps
            # TODO: Remove if needed
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
                        gate_dict[(gate, qs)] = U
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

        # Setting the value as (0, True) here means it will reset to 0 state
        # and it will record the outcomes
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

    assert gaterep in model.output_gate_reps
    assert instrep in model.output_instrument_reps

    return model


## Helper functions
def _create_adaptive_measure_instruction(
    instructions,
    qubits,
    include_idles,
    gate_durations,
    idle_gates,
    circuit_backend,
):
    # FT Adaptive X Measurement Scheme
    # Fig 13 of arxiv:2208.01863

    # For an in-depth documentation of this function, check out
    # the Building a Complex Feed-Forward Instruction tutorial
    # TODO: Update this tutorial

    _create_adaptive_measure_instruction_part_I(
        instructions,
        qubits,
        include_idles,
        gate_durations,
        idle_gates,
        circuit_backend,
    )

    _create_adaptive_measure_instruction_part_II(
        instructions,
        qubits,
        include_idles,
        gate_durations,
        idle_gates,
        circuit_backend,
    )

    _create_adaptive_measure_instruction_part_III(
        instructions,
        qubits,
        include_idles,
        gate_durations,
        idle_gates,
        circuit_backend,
    )

    ## CLASSICAL DECODER
    # Not part of the figure, but described in the figure caption
    # More details can also be found in Appendix B.2 of arxiv: 1705.02329
    def classical_decoder_apply_fn(
        patch_label: str,
        patches: PatchDict,
        measurement_outcomes: MeasurementOutcomes,
        flagged_check: str | None,
        flagged_check_order: list[int] | None,
    ) -> Frame:
        """Apply function for classical decoder in fault-tolerant logical measurement.
        
        Performs classical syndrome computation and decoding to determine
        error corrections for logical measurement outcomes. This function
        implements the classical decoder described in Appendix B.2 of
        arxiv:1705.02329 for the [[5,1,3]] quantum error correcting code.
        
        The decoder computes syndromes from stabilizer measurements, looks up
        potential error corrections, applies classical error correction, and
        computes the final logical measurement outcome based on parity.
        
        Parameters
        ----------
        patch_label : str
            Label of the patch being decoded.
        patches : PatchDict
            Dictionary of patches containing Pauli frame information for
            tracking Pauli corrections.
        measurement_outcomes : MeasurementOutcomes
            Raw measurement outcomes from the physical qubits, used to infer
            corrected outcomes based on the Pauli frame.
        flagged_check : str | None
            Name of the flagged check that failed (e.g., "XZIIZ"), or None 
            if no flag failure occurred.
        flagged_check_order : list[int] | None
            Order of qubits in the flagged check, or None if no flag failure.
            Used to determine hook errors when a flag check fails.
        
        Returns
        -------
        Frame
            Frame containing decoded information including:
            - logical_measurement : int
                The final logical measurement value (0 or 1) after error correction.
            - stage : str
                Stage identifier ("Classical Decoder").
            - precorrected_inferred_outcomes : list[int]
                Inferred measurement outcomes before error correction.
            - possible_predecoded_errors : list[str]
                List of possible error Pauli strings before decoding.
            - possible_decoded_errors : list[str]
                List of possible error Pauli strings after decoding.
            - classical_syndrome : tuple[int, ...]
                Computed syndrome from stabilizer measurements.
            - classical_correction : str
                Applied classical error correction Pauli string.
            - corrected_outcomes : list[int]
                Final corrected measurement outcomes after error correction.
        
        Notes
        -----
        The decoder handles two cases:
        1. Measurement mismatches (flagged_check is None): Corrects all weight-1 data errors
        2. Flag failures (flagged_check provided): Corrects hook errors based on failed check
        
        An odd parity of corrected outcomes corresponds to logical 0,
        while even parity corresponds to logical 1.
        """

        # Get pauli frame
        pauli_frame = patches[patch_label].pauli_frame
        qubits = pauli_frame.qubit_labels

        # Compute inferred bitstring
        data_inferred_outcomes = measurement_outcomes.get_inferred_outcomes(
            pauli_frame, "Z"
        )
        data_inferred_bitstring = [
            data_inferred_outcomes[q][0] for q in qubits
        ]

        # Compute syndromes classically
        # We only have Z basis measurements, so any 1s we see is like an X error
        data_inferred_pstr = "".join(
            ["I" if b == 0 else "X" for b in data_inferred_bitstring]
        )

        stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
        if flagged_check is None:
            # This was a measurement mismatch, so we are correcting a data error
            # i.e. all possible weight-1 errors on data qubits
            predecoded_errors = qt.get_weight_1_errors(5)
        else:
            # We are not a data error, but a measurement hook error
            # Compute the hook errors based on the failed flag check
            predecoded_errors = qt.get_hook_errors_in_flagged_check(
                flagged_check, flagged_check_order
            )

        def compute_decoded_bitstring(pstr: str) -> str:
            """Compute the decoded bitstring from a Pauli string.
            
            This function implements the decoding logic for the [[5,1,3]] code by
            propagating Pauli errors through the decoding circuit. It handles X, Y, and Z
            errors according to the error propagation rules of the code.
            
            Parameters
            ----------
            pstr : str
                Input Pauli string to decode (e.g., "IXZXI" for a 5-qubit string).
                Each character represents a Pauli operator on one qubit.
            
            Returns
            -------
            str
                Decoded bitstring representing the corrected Pauli string after
                error propagation through the decoding circuit.
            
            Notes
            -----
            Error propagation rules:
            - X errors: Trigger X errors on adjacent qubits (with periodic boundary conditions)
            - Z errors: Trigger X errors on the same qubit
            - Y errors: Trigger X-like and Z-like errors (X on adjacent and same qubit)
            
            The function uses periodic boundary conditions, so qubit 0 is adjacent to
            qubit 4, and qubit 4 is adjacent to qubit 0.
            """
            decoded_pstr = "I" * len(pstr)
            for i, p in enumerate(pstr):
                if p == "X":
                    # This will trigger X errors on qubits above and below
                    # (with periodic boundary conditions)
                    X_pstr = [
                        "I",
                    ] * len(pstr)
                    X_pstr[(i - 1) % len(pstr)] = "X"
                    X_pstr[(i + 1) % len(pstr)] = "X"

                    decoded_pstr = qt.compose_pstrs(
                        decoded_pstr, "".join(X_pstr)
                    )
                elif p == "Z":
                    # This will trigger an X error on this qubit
                    Z_pstr = [
                        "I",
                    ] * len(pstr)
                    Z_pstr[i] = "X"
                    decoded_pstr = qt.compose_pstrs(
                        decoded_pstr, "".join(Z_pstr)
                    )
                elif p == "Y":
                    # This will trigger an X-like and Z-like error
                    Y_pstr = [
                        "I",
                    ] * len(pstr)
                    Y_pstr[(i - 1) % len(pstr)] = "X"
                    Y_pstr[i] = "X"
                    Y_pstr[(i + 1) % len(pstr)] = "X"
                    decoded_pstr = qt.compose_pstrs(
                        decoded_pstr, "".join(Y_pstr)
                    )
            return decoded_pstr

        # Push possible errors through decoding circuit
        decoded_errors = [
            compute_decoded_bitstring(err) for err in predecoded_errors
        ]

        # Create lookup table from the decoded errors
        syndrome_dict = qt.get_syndrome_dict_from_stabilizers_and_pstrs(
            stabilizers, decoded_errors
        )

        # Compute syndrome
        syndrome = qt.get_syndrome_from_stabilizers_and_pstr(
            stabilizers, data_inferred_pstr
        )

        # Look up correction (directly from syndrome since we applied full frame above)
        classical_correction = syndrome_dict[syndrome][0]

        # Apply correction
        classically_corrected_pstr = qt.compose_pstrs(
            data_inferred_pstr, classical_correction
        )

        # Convert back to bitstring
        final_bitstring = [int(p in "XY") for p in classically_corrected_pstr]

        # An odd parity = logical 0, even parity = logical 1
        logical_value = (sum(final_bitstring) + 1) % 2

        return Frame(
            {
                "logical_measurement": logical_value,
                "stage": "Classical Decoder",
                "precorrected_inferred_outcomes": data_inferred_bitstring,
                "possible_predecoded_errors": predecoded_errors,
                "possible_decoded_errors": decoded_errors,
                "classical_syndrome": syndrome,
                "classical_correction": classical_correction,
                "corrected_outcomes": final_bitstring,
            }
        )

    instructions["FT Logical X Measure Classical Decoder"] = Instruction(
        classical_decoder_apply_fn,
        name="FT Logical X Measure Classical Decoder",
    )

    # Shortcut composite function to kick off the adaptive measurement
    instructions["FT Logical X Measure"] = (
        builders.build_composite_instruction(
            [
                instructions["FT Logical X Measure Part I Circuit"],
                instructions["FT Logical X Measure Part I Feed-Forward"],
            ],
            name="FT Logical X Measure",
        )
    )


def _create_adaptive_measure_instruction_part_I(
    instructions,
    qubits,
    include_idles,
    gate_durations,
    idle_gates,
    circuit_backend,
):
    ## PART I of Fig 13 for arxiv:2208.01863
    measI_circ = circuit_backend(
        [
            ("Gh", "A0"),
            ("Gcphase", "A0", "D4"),
            ("Gcnot", "A0", "A1"),
            ("Gcnot", "A0", "D0"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D1"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        measI_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["FT Logical X Measure Part I Circuit"] = (
        builders.build_physical_circuit_instruction(
            measI_circ,
            name="FT Logical X Measure Part I Circuit",
        )
    )

    def partI_feedforward_apply_fn(
        patch_label: str,
        patches: PatchDict,
        stack: InstructionStack,
        measurement_outcomes: MeasurementOutcomes,
        qubits: list[str],
        ) -> Frame:
        """Apply function for Part I feedforward in fault-tolerant logical X measurement.
        
        Processes measurement outcomes and determines next instructions based on
        flag qubit results. This function implements the first stage of the
        three-part adaptive measurement protocol described in Fig 13 of arxiv:2208.01863.
        
        The function evaluates the flag qubit F1 to determine the next steps:
        - If F1 == 0: Proceeds to Part II of the measurement protocol
        - If F1 == 1: Triggers the classical decoder with flag failure information
        
        When a flag failure occurs, it passes the flagged check name ("XZIIZ")
        and qubit order ([4, 0, 1]) to help the decoder identify hook errors.
        
        Parameters
        ----------
        patch_label : str
            Label of the patch being measured.
        patches : PatchDict
            Dictionary of patches containing Pauli frame information for
            tracking Pauli corrections.
        stack : InstructionStack
            Current instruction stack for feedforward operations that will
            be modified to include next instructions.
        measurement_outcomes : MeasurementOutcomes
            Raw measurement outcomes from the physical qubits, containing
            both measurement and flag results.
        qubits : list[str]
            List of qubit labels involved in the measurement, where
            qubits[0] is the measurement qubit and qubits[1] is the flag qubit.
        
        Returns
        -------
        Frame
            Frame containing updated instruction stack with next instructions
            and measurement information including:
            - stack : InstructionStack (with Part II or decoder instructions)
            - F1 : int (flag qubit value)
            - raw_M1 : int (raw measurement from Part I)
            - inferred_M1 : int (inferred measurement from Part I)
        
        Notes
        -----
        This function implements Part I of the three-part adaptive measurement
        protocol for fault-tolerant logical X measurements. The protocol uses
        flag qubits to detect fault-tolerant violations and adaptively chooses
        between continuing the measurement or triggering error correction.
        """
        pauli_frame = patches[patch_label].pauli_frame

        # Pull out measurement and flag
        meas_qubit = qubits[0]
        flag_qubit = qubits[1]
        M1 = measurement_outcomes[meas_qubit][0]
        F1 = measurement_outcomes[flag_qubit][0]

        # Use the Pauli frame to infer the correction on M1
        # Also, we flip M1 at this point. Measuring 0 means we are in the minus state,
        # but this should correspond to a 1 logical outcome
        # The check is XZIIZ, so Z0, X1, or X4 errors will flip it
        # Qubit indices offset by 2 for A0/A1
        inferred_M1 = (
            sum(
                [
                    M1 + 1,
                    pauli_frame.get_bit("Z", qubits[2]),
                    pauli_frame.get_bit("X", qubits[3]),
                    pauli_frame.get_bit("X", qubits[6]),
                ]
            )
            % 2
        )

        # Do classical feed forward
        if F1 == 0:
            # We go to part II (forward reference, must match key later)
            ilbls: list[InstructionLabelCastableTypes] = [
                ("FT Logical X Measure Part II Circuit", patch_label),
                ("FT Logical X Measure Part II Feed-Forward", patch_label),
            ]
        else:
            # We go to decoding circuit (forward references must match key later)
            # This is a flag failure, so pass that information in
            # We also need to pass a nonstandard order for this circuit
            # (we measure Z_4 first, then X_0 and Z_1)
            kwargs = {
                "flagged_check": "XZIIZ",
                "flagged_check_order": [4, 0, 1],
            }
            ilbls = [
                ("Non-FT Minus Unprep", patch_label),
                (
                    "FT Logical X Measure Classical Decoder",
                    patch_label,
                    (),
                    kwargs,
                ),
            ]

        for i, ilbl in enumerate(ilbls):
            new_label = InstructionLabel.cast(ilbl)
            stack = stack.insert_instruction(i, new_label)

        return Frame(
            {
                "stack": stack,
                "F1": F1,
                "raw_M1": M1,
                "inferred_M1": inferred_M1,
            }
        )

    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        qubits: Sequence[str],
        **kwargs,
    ) -> KwargDict:
        """Map qubits function for Part I feedforward instruction.
        
        Remaps qubit labels according to the provided mapping.
        This function enables the Part I feedforward instruction to work with
        different qubit label schemes by translating between them.
        
        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        qubits : Sequence[str]
            Original qubit labels to be remapped (typically measurement and flag qubits).
        **kwargs : dict
            Additional keyword arguments to preserve unchanged.
        
        Returns
        -------
        KwargDict
            Updated keyword arguments with remapped qubits list,
            preserving all other arguments.
        
        Notes
        -----
        This function is used as the map_qubits_fn parameter when creating
        Instruction objects that need to support qubit relabeling.
        """
        new_kwargs = kwargs.copy()
        new_kwargs["qubits"] = [qubit_mapping.get(q, q) for q in qubits]
        return new_kwargs

    instructions["FT Logical X Measure Part I Feed-Forward"] = Instruction(
        partI_feedforward_apply_fn,
        {"qubits": qubits},
        map_qubits_fn,
        name="FT Logical X Measure Part I Feed-Forward",
    )


def _create_adaptive_measure_instruction_part_II(
    instructions,
    qubits,
    include_idles,
    gate_durations,
    idle_gates,
    circuit_backend,
):
    ## PART II of Fig 13 of arxiv:2208.01863
    measII_circ = circuit_backend(
        [
            ("Gh", "A0"),
            ("Gcphase", "A0", "D0"),
            ("Gcnot", "A0", "A1"),
            ("Gcnot", "A0", "D1"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D2"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        measII_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["FT Logical X Measure Part II Circuit"] = (
        builders.build_physical_circuit_instruction(
            measII_circ,
            name="FT Logical X Measure Part II Circuit",
        )
    )

    def partII_feedforward_apply_fn(
        patch_label: str,
        patches: PatchDict,
        stack: InstructionStack,
        measurement_outcomes: MeasurementOutcomes,
        qubits: list[str],
        inferred_M1: int,
    ) -> Frame:
        """Apply function for Part II feedforward in fault-tolerant logical X measurement.
        
        Processes measurement outcomes from Part II and determines next instructions
        in the adaptive measurement protocol. This function implements the second stage
        of the three-part protocol described in Fig 13 of arxiv:2208.01863.
        
        The function evaluates flag and syndrome outcomes to determine the next steps:
        - If flag F2 != 0: Terminates with logical measurement from M1
        - If inferred_M1 == inferred_M2: Proceeds to Part III
        - Otherwise: Triggers classical decoder for error correction
        
        Parameters
        ----------
        patch_label : str
            Label of the patch being measured.
        patches : PatchDict
            Dictionary of patches containing Pauli frame information.
        stack : InstructionStack
            Current instruction stack for feedforward operations that will
            be modified to include next instructions.
        measurement_outcomes : MeasurementOutcomes
            Raw measurement outcomes from the physical qubits, containing
            both measurement and flag results.
        qubits : list[str]
            List of qubit labels involved in the measurement, where
            qubits[0] is the measurement qubit and qubits[1] is the flag qubit.
        inferred_M1 : int
            Inferred measurement outcome from Part I (0 or 1), retrieved
            from history[-2] frame using param_priorities.
        
        Returns
        -------
        Frame
            Frame containing either:
            - Updated instruction stack with next instructions (Part III or decoder)
            - Final logical measurement outcome and termination information
            
            When continuing, includes updated stack. When terminating, includes:
            - logical_measurement : int (final logical value from M1)
            - stage : str (termination reason)
            - F2 : int (flag qubit value)
            - raw_M2 : int (raw measurement from Part II)
            - inferred_M2 : int (inferred measurement from Part II)
        
        Notes
        -----
        This function implements Part II of the three-part adaptive measurement
        protocol for fault-tolerant logical X measurements.
        """
        pauli_frame = patches[patch_label].pauli_frame

        # Pull out measurement and flag
        meas_qubit = qubits[0]
        flag_qubit = qubits[1]
        M2 = measurement_outcomes[meas_qubit][0]
        F2 = measurement_outcomes[flag_qubit][0]

        # Use the Pauli frame to infer the correction on M2
        # Also, we flip M2 at this point. Measuring 0 means we are in the minus state,
        # but this should correspond to a 1 logical outcome
        # The check is ZXZII, so X0, Z1, or X2 errors will flip it
        # Qubit indices offset by 2 for A0/A1
        inferred_M2 = (
            sum(
                [
                    M2 + 1,
                    pauli_frame.get_bit("X", qubits[2]),
                    pauli_frame.get_bit("Z", qubits[3]),
                    pauli_frame.get_bit("X", qubits[4]),
                ]
            )
            % 2
        )

        # Do classical feed forward
        if F2 != 0:
            # Terminate
            return Frame(
                {
                    "logical_measurement": inferred_M1,
                    "stage": "Part II (F2 != 0)",
                    "F2": F2,
                    "raw_M2": M2,
                    "inferred_M2": inferred_M2,
                }
            )
        elif inferred_M1 == inferred_M2:
            # We go to part III (forward reference, must match key later)
            ilbls: list[InstructionLabelCastableTypes] = [
                ("FT Logical X Measure Part III Circuit", patch_label),
                ("FT Logical X Measure Part III Feed-Forward", patch_label),
            ]
        else:
            # We go to decoding circuit  (forward references, must match key later)
            # This is not a flag failure, so we pass None to these kwargs
            kwargs = {"flagged_check": None, "flagged_check_order": None}
            ilbls = [
                ("Non-FT Minus Unprep", patch_label),
                (
                    "FT Logical X Measure Classical Decoder",
                    patch_label,
                    (),
                    kwargs,
                ),
            ]

        # We need to make sure and feed the patch label forward
        for i, ilbl in enumerate(ilbls):
            new_label = InstructionLabel.cast(ilbl)
            stack = stack.insert_instruction(i, new_label)

        # Return new frame
        frame_data = {
            "stack": stack,
            "F2": F2,
            "raw_M2": M2,
            "inferred_M2": inferred_M2,
        }
        return Frame(frame_data)

    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        qubits: Sequence[str],
        **kwargs,
    ) -> KwargDict:
        """Map qubits function for Part II feedforward instruction.
        
        Remaps qubit labels according to the provided mapping.
        This function enables the Part II feedforward instruction to work with
        different qubit label schemes by translating between them.
        
        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        qubits : Sequence[str]
            Original qubit labels to be remapped (typically measurement and flag qubits).
        **kwargs : dict
            Additional keyword arguments to preserve unchanged.
        
        Returns
        -------
        KwargDict
            Updated keyword arguments with remapped qubits list,
            preserving all other arguments.
        
        Notes
        -----
        This function is used as the map_qubits_fn parameter when creating
        Instruction objects that need to support qubit relabeling.
        """
        new_kwargs = kwargs.copy()
        new_kwargs["qubits"] = [qubit_mapping.get(q, q) for q in qubits]
        return new_kwargs

    instructions["FT Logical X Measure Part II Feed-Forward"] = Instruction(
        partII_feedforward_apply_fn,
        {"qubits": qubits},
        map_qubits_fn,
        param_priorities={
            "inferred_M1": ["history[-2]"]  # Look back 2 frames for M1
        },
            name="FT Logical X Measure Part II Feed-Forward",
    )

    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        qubits: Sequence[str],
        **kwargs,
    ) -> KwargDict:
        """Map qubits function for Part II feedforward instruction.
        
        Remaps qubit labels according to the provided mapping.
        
        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        qubits : Sequence[str]
            Original qubit labels to be remapped.
        **kwargs : dict
            Additional keyword arguments to preserve.
        
        Returns
        -------
        KwargDict
            Updated keyword arguments with remapped qubit labels.
        
        REVIEW_NO_DOCSTRING
        """
    instructions,
    qubits,
    include_idles,
    gate_durations,
    idle_gates,
    circuit_backend,
):
    ## PART III
    measIII_circ = circuit_backend(
        [
            ("Gh", "A0"),
            ("Gcphase", "A0", "D2"),
            ("Gcnot", "A0", "A1"),
            ("Gcnot", "A0", "D3"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D4"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        measIII_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["FT Logical X Measure Part III Circuit"] = (
        builders.build_physical_circuit_instruction(
            measIII_circ,
            name="FT Logical X Measure Part III Circuit",
        )
    )

    def partIII_feedforward_apply_fn(
        patch_label: str,
        patches: PatchDict,
        stack: InstructionStack,
        measurement_outcomes: MeasurementOutcomes,
        qubits: list[str],
        inferred_M1: int,
        inferred_M2: int,
    ) -> Frame:
        """Apply function for Part III feedforward in fault-tolerant logical X measurement.
        
        Processes measurement outcomes from Part III and determines final logical outcome
        or triggers error correction procedures. This function implements the final stage
        of the adaptive measurement protocol described in Fig 13 of arxiv:2208.01863.
        
        The function evaluates three conditions to determine the next steps:
        1. Flag qubit F3 = 0 (no flag error)
        2. Inferred measurements M1 == M2 (consistency between parts I and II)
        3. Inferred measurements M1 != M3 (discrepancy with part III)
        
        When all three conditions are met, it triggers the classical decoder for
        error correction. Otherwise, it terminates with the logical measurement
        value from M1.
        
        Parameters
        ----------
        patch_label : str
            Label of the patch being measured.
        patches : PatchDict
            Dictionary of patches containing Pauli frame information.
        stack : InstructionStack
            Current instruction stack for feedforward operations that will
            be modified to include next instructions.
        measurement_outcomes : MeasurementOutcomes
            Raw measurement outcomes from the physical qubits, containing
            both measurement and flag results.
        qubits : list[str]
            List of qubit labels involved in the measurement, where
            qubits[0] is the measurement qubit and qubits[1] is the flag qubit.
        inferred_M1 : int
            Inferred measurement outcome from Part I (0 or 1).
        inferred_M2 : int
            Inferred measurement outcome from Part II (0 or 1).
        
        Returns
        -------
        Frame
            Frame containing either:
            - Updated instruction stack with decoder instructions (when conditions met)
            - Final logical measurement outcome and termination information (otherwise)
            
            The frame includes:
            - stack : InstructionStack (when continuing)
            - logical_measurement : int (when terminating)
            - stage : str (termination reason)
            - F3 : int (flag qubit value)
            - raw_M3 : int (raw measurement from Part III)
            - inferred_M3 : int (inferred measurement from Part III)
        
        Notes
        -----
        This function implements Part III of the three-part adaptive measurement
        protocol for fault-tolerant logical X measurements.
        """
        pauli_frame = patches[patch_label].pauli_frame

        # Pull out measurement and flag
        meas_qubit = qubits[0]
        flag_qubit = qubits[1]
        M3 = measurement_outcomes[meas_qubit][0]
        F3 = measurement_outcomes[flag_qubit][0]

        # Use the Pauli frame to infer the correction on M3
        # Also, we flip M3 at this point. Measuring 0 means we are in the minus state,
        # but this should correspond to a 1 logical outcome
        # The check is IIZXZ, so X2, Z3, or X4 errors will flip it
        # Qubit indices offset by 2 for A0/A1
        inferred_M3 = (
            sum(
                [
                    M3 + 1,
                    pauli_frame.get_bit("X", qubits[4]),
                    pauli_frame.get_bit("Z", qubits[5]),
                    pauli_frame.get_bit("X", qubits[6]),
                ]
            )
            % 2
        )

        # Do feed forward
        if (
            F3 == 0
            and inferred_M1 == inferred_M2
            and inferred_M1 != inferred_M3
        ):
            # We go to decoding circuit  (forward references, must match key later)
            # This is not a flag failure, so we pass None as these kwargs
            kwargs = {"flagged_check": None, "flagged_check_order": None}
            ilbls = [
                ("Non-FT Minus Unprep", patch_label),
                (
                    "FT Logical X Measure Classical Decoder",
                    patch_label,
                    (),
                    kwargs,
                ),
            ]
        else:
            # Terminate, but let's note which condition caused it
            if F3 != 0:
                cause = "(F3 != 0)"
            else:
                cause = "M1 = M2 = M3"

            return Frame(
                {
                    "logical_measurement": inferred_M1,
                    "stage": f"Part III {cause}",
                    "F3": F3,
                    "raw_M3": M3,
                    "inferred_M3": inferred_M3,
                }
            )

        # We need to make sure and feed the patch label forward
        for i, ilbl in enumerate(ilbls):
            new_label = InstructionLabel.cast(ilbl)
            stack = stack.insert_instruction(i, new_label)

        # Return new frame
        return Frame(
            {
                "stack": stack,
                "F3": F3,
                "raw_M3": M3,
                "inferred_M3": inferred_M3,
            }
        )

    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        qubits: Sequence[str],
        **kwargs,
    ) -> KwargDict:
        """Map qubits function for Part III feedforward instruction.
        
        Remaps qubit labels according to the provided mapping.
        This function enables the Part III feedforward instruction to work with
        different qubit label schemes by translating between them.
        
        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        qubits : Sequence[str]
            Original qubit labels to be remapped (typically measurement and flag qubits).
        **kwargs : dict
            Additional keyword arguments to preserve unchanged.
        
        Returns
        -------
        KwargDict
            Updated keyword arguments with remapped qubits list,
            preserving all other arguments.
        
        Notes
        -----
        This function is used as the map_qubits_fn parameter when creating
        Instruction objects that need to support qubit relabeling.
        """
        new_kwargs = kwargs.copy()
        new_kwargs["qubits"] = [qubit_mapping.get(q, q) for q in qubits]
        return new_kwargs

    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        qubits: Sequence[str],
        **kwargs,
    ) -> KwargDict:
        """Map qubits function for Part III feedforward instruction.
        
        Remaps qubit labels according to the provided mapping.
        
        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        qubits : Sequence[str]
            Original qubit labels to be remapped.
        **kwargs : dict
            Additional keyword arguments to preserve.
        
        Returns
        -------
        KwargDict
            Updated keyword arguments with remapped qubit labels.
        
        REVIEW_NO_DOCSTRING
        """
        param_priorities={
            "inferred_M1": ["history[-4]"],  # Look back 4 frames for M1
            "inferred_M2": ["history[-2]"],  # and 2 frames for M2
        },
        name="FT Logical X Measure Part III Feed-Forward",
    )


def _create_stabilizer_instructions(
    instructions,
    qubits,
    include_idles,
    gate_durations,
    idle_gates,
    circuit_backend,
):
    XZZXI_circ = circuit_backend(
        [[("Gxpi", "D0"), ("Gzpi", "D1"), ("Gzpi", "D2"), ("Gxpi", "D3")]],
        qubit_labels=qubits,
    )
    if include_idles:
        XZZXI_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["XZZXI Stabilizer"] = (
        builders.build_physical_circuit_instruction(
            XZZXI_circ,
            name="XZZXI stabilizer",
        )
    )

    IXZZX_circ = circuit_backend(
        [[("Gxpi", "D1"), ("Gzpi", "D2"), ("Gzpi", "D3"), ("Gxpi", "D4")]],
        qubit_labels=qubits,
    )
    if include_idles:
        IXZZX_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["IXZZX Stabilizer"] = (
        builders.build_physical_circuit_instruction(
            IXZZX_circ,
            name="IXZZX stabilizer",
        )
    )

    XIXZZ_circ = circuit_backend(
        [[("Gxpi", "D0"), ("Gxpi", "D2"), ("Gzpi", "D3"), ("Gzpi", "D4")]],
        qubit_labels=qubits,
    )
    if include_idles:
        XIXZZ_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["XIXZZ Stabilizer"] = (
        builders.build_physical_circuit_instruction(
            XIXZZ_circ,
            name="XIXZZ stabilizer",
        )
    )

    ZXIXZ_circ = circuit_backend(
        [[("Gzpi", "D0"), ("Gxpi", "D1"), ("Gxpi", "D3"), ("Gzpi", "D4")]],
        qubit_labels=qubits,
    )
    if include_idles:
        ZXIXZ_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["ZXIXZ Stabilizer"] = (
        builders.build_physical_circuit_instruction(
            ZXIXZ_circ,
            name="ZXIXZ stabilizer",
        )
    )


def _create_unflagged_QEC_instruction(
    instructions,
    qubits,
    include_idles,
    gate_durations,
    idle_gates,
    circuit_backend,
):
    # These circuits are not explicitly stated in arxiv:2208.01863
    # However, they can be inferred from the Hadamard-test-like circuits of Fig 12
    # and the stabilizer definitions from Eqns B4-B7

    # XZZXI check
    # This actually matches Fig 2c of arXiv:1705.02329 as well
    XZZXI_circ = circuit_backend(
        [
            [("Gh", "A0")],
            [("Gcnot", "A0", "D0")],
            [("Gcphase", "A0", "D1")],
            [("Gcphase", "A0", "D2")],
            [("Gcnot", "A0", "D3")],
            [("Gh", "A0")],
            [("Iz", "A0")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        XZZXI_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Unflagged XZZXI Check"] = (
        builders.build_physical_circuit_instruction(
            XZZXI_circ,
            name="Unflagged XZZXI stabilizer check",
        )
    )

    # IXZZX check
    IXZZX_circ = circuit_backend(
        [
            [("Gh", "A0")],
            [("Gcnot", "A0", "D1")],
            [("Gcphase", "A0", "D2")],
            [("Gcphase", "A0", "D3")],
            [("Gcnot", "A0", "D4")],
            [("Gh", "A0")],
            [("Iz", "A0")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        IXZZX_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Unflagged IXZZX Check"] = (
        builders.build_physical_circuit_instruction(
            IXZZX_circ,
            name="Unflagged IXZZX stabilizer check",
        )
    )

    # XIXZZ check
    XIXZZ_circ = circuit_backend(
        [
            [("Gh", "A0")],
            [("Gcnot", "A0", "D0")],
            [("Gcnot", "A0", "D2")],
            [("Gcphase", "A0", "D3")],
            [("Gcphase", "A0", "D4")],
            [("Gh", "A0")],
            [("Iz", "A0")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        XIXZZ_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Unflagged XIXZZ Check"] = (
        builders.build_physical_circuit_instruction(
            XIXZZ_circ,
            name="Unflagged XIXZZ stabilizer check",
        )
    )

    # ZXIXZ check
    ZXIXZ_circ = circuit_backend(
        [
            [("Gh", "A0")],
            [("Gcphase", "A0", "D0")],
            [("Gcnot", "A0", "D1")],
            [("Gcnot", "A0", "D3")],
            [("Gcphase", "A0", "D4")],
            [("Gh", "A0")],
            [("Iz", "A0")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        ZXIXZ_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Unflagged ZXIXZ Check"] = (
        builders.build_physical_circuit_instruction(
            ZXIXZ_circ,
            name="Unflagged ZXIXZ stabilizer check",
        )
    )

    # Unflagged decoder
    # This is not written out in the references but can be quickly derived from the stabilizers
    # This is now automated in loqs.tools.qectools
    data_errors = qt.get_weight_1_errors(5)
    stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    syndrome_dict = qt.get_syndrome_dict_from_stabilizers_and_pstrs(
        stabilizers, data_errors
    )
    unflagged_lookup_table = {k: v[0] for k, v in syndrome_dict.items()}

    # Take the first measurement from A0 qubit for last 4 instructions as syndrome
    syndrome_labels = [("A0", -4), ("A0", -3), ("A0", -2), ("A0", -1)]
    instructions["Unflagged Decoder"] = (
        builders.build_lookup_decoder_instruction(
            lookup_table=unflagged_lookup_table,
            syndrome_labels=syndrome_labels,
            raw_syndrome_frame_key="raw_syndrome",
            name="Unflagged decoder",
        )
    )

    # QEC is now just the 4 unflagged checks + decoding
    instructions["Unflagged QEC"] = builders.build_composite_instruction(
        [
            instructions["Unflagged XZZXI Check"],
            instructions["Unflagged IXZZX Check"],
            instructions["Unflagged XIXZZ Check"],
            instructions["Unflagged ZXIXZ Check"],
            instructions["Unflagged Decoder"],
        ],
        name="Unflagged QEC",
    )


def _create_flagged_QEC_instruction(
    instructions,
    qubits,
    include_idles,
    gate_durations,
    idle_gates,
    circuit_backend,
):
    # These circuits are not explicitly stated in arxiv:2208.01863
    # However, they can be inferred from the Hadamard-test-like circuits of Fig 12
    # and the stabilizer definitions from Eqns B4-B7

    # XZZXI check
    # This actually matches Fig 2b of arXiv:1705.02329 as well
    XZZXI_circ = circuit_backend(
        [
            [("Gh", "A0")],
            [("Gcnot", "A0", "D0")],
            [("Gcnot", "A0", "A1")],  # Flag
            [("Gcphase", "A0", "D1")],
            [("Gcphase", "A0", "D2")],
            [("Gcnot", "A0", "A1")],  # Flag
            [("Gcnot", "A0", "D3")],
            [("Gh", "A0")],
            [("Iz", "A0"), ("Iz", "A1")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        XZZXI_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Flagged XZZXI Check"] = (
        builders.build_physical_circuit_instruction(
            XZZXI_circ,
            name="Flagged XZZXI stabilizer check",
        )
    )

    # IXZZX check
    IXZZX_circ = circuit_backend(
        [
            [("Gh", "A0")],
            [("Gcnot", "A0", "D1")],
            [("Gcnot", "A0", "A1")],  # Flag
            [("Gcphase", "A0", "D2")],
            [("Gcphase", "A0", "D3")],
            [("Gcnot", "A0", "A1")],  # Flag
            [("Gcnot", "A0", "D4")],
            [("Gh", "A0")],
            [("Iz", "A0"), ("Iz", "A1")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        IXZZX_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Flagged IXZZX Check"] = (
        builders.build_physical_circuit_instruction(
            IXZZX_circ,
            name="Flagged IXZZX stabilizer check",
        )
    )

    # XIXZZ check
    XIXZZ_circ = circuit_backend(
        [
            [("Gh", "A0")],
            [("Gcnot", "A0", "D0")],
            [("Gcnot", "A0", "A1")],  # Flag
            [("Gcnot", "A0", "D2")],
            [("Gcphase", "A0", "D3")],
            [("Gcnot", "A0", "A1")],  # Flag
            [("Gcphase", "A0", "D4")],
            [("Gh", "A0")],
            [("Iz", "A0"), ("Iz", "A1")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        XIXZZ_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Flagged XIXZZ Check"] = (
        builders.build_physical_circuit_instruction(
            XIXZZ_circ,
            name="Flagged XIXZZ stabilizer check",
        )
    )

    # ZXIXZ check
    ZXIXZ_circ = circuit_backend(
        [
            [("Gh", "A0")],
            [("Gcphase", "A0", "D0")],
            [("Gcnot", "A0", "A1")],  # Flag
            [("Gcnot", "A0", "D1")],
            [("Gcnot", "A0", "D3")],
            [("Gcnot", "A0", "A1")],  # Flag
            [("Gcphase", "A0", "D4")],
            [("Gh", "A0")],
            [("Iz", "A0"), ("Iz", "A1")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        ZXIXZ_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Flagged ZXIXZ Check"] = (
        builders.build_physical_circuit_instruction(
            ZXIXZ_circ,
            name="Flagged ZXIXZ stabilizer check",
        )
    )

    # FLAGGED DECODER
    # Unlike the unflagged decoder, this requires feed-forward logic
    # This follows the Error Correction Procedure in Section II of arXiv:1705.02329
    # Much of this is now automated in loqs.tools.qectools

    # First, we create the four different lookup table decoders
    # based on different flags being tripped
    stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    for stab in stabilizers:
        hook_data_errors = qt.get_hook_errors_in_flagged_check(stab)
        syndrome_dict = qt.get_syndrome_dict_from_stabilizers_and_pstrs(
            stabilizers, hook_data_errors
        )

        # Note that this lookup table has many syndromes that correspond to no corrections
        lookup_table = {k: v[0] for k, v in syndrome_dict.items()}

        # If we are using this lookup table, we just did unflagged syndrome checks
        # So the decoder instruction actually looks similar to unflagged, but
        # with a different decoder. We also don't want to look at previous
        # flagged corrections, this correction is based solely on this round info
        syndrome_labels = [("A0", -4), ("A0", -3), ("A0", -2), ("A0", -1)]
        instructions[f"Flagged {stab} Decoder"] = (
            builders.build_lookup_decoder_instruction(
                lookup_table=lookup_table,
                syndrome_labels=syndrome_labels,
                raw_syndrome_frame_key=f"flagged_{stab}_syndrome",
                diff_prev_syndrome=False,
                name=f"Flagged {stab} decoder",
            )
        )

    ## FLAGGED QEC
    # We will only create one additional instruction, as the same procedure is done
    # for all four flagged measurements with different parameters
    def flagged_feedforward_apply_fn(
        stabilizer: str,
        next_stabilizer: str,
        syndrome_qubit: str,
        flag_qubit: str,
        measurement_outcomes: MeasurementOutcomes,
        stack: InstructionStack,
        patch_label: str,
    ):
        """Apply function for flagged feedforward in stabilizer measurements.
        
        Processes flag and syndrome outcomes to determine next error correction instructions
        in the fault-tolerant quantum error correction protocol. This function implements
        the flag-based feedforward logic for the [[5,1,3]] code.
        
        When a flag qubit indicates a potential error (flag=1), the function triggers
        unflagged stabilizer measurements followed by a flagged decoder. When the flag
        indicates no error (flag=0) and syndrome is 0, it proceeds to check the next
        stabilizer. Otherwise, it triggers the appropriate error correction procedure.
        
        Parameters
        ----------
        stabilizer : str
            Current stabilizer being checked (e.g., "XZZXI", "IXZZX").
        next_stabilizer : str
            Next stabilizer to check if current one passes.
        syndrome_qubit : str
            Label of the syndrome qubit used for error detection.
        flag_qubit : str
            Label of the flag qubit used to detect fault-tolerant violations.
        measurement_outcomes : MeasurementOutcomes
            Raw measurement outcomes from the physical qubits, containing
            both syndrome and flag measurement results.
        stack : InstructionStack
            Current instruction stack for feedforward operations that will
            be modified to include next instructions.
        patch_label : str
            Label of the patch being measured.
        
        Returns
        -------
        Frame
            Frame containing the updated instruction stack with the next
            instructions to execute based on the flag and syndrome outcomes.
        
        Notes
        -----
        The function implements the following logic:
        - flag=1: Trigger unflagged measurements + flagged decoder
        - flag=0, syndrome=0: Proceed to next stabilizer check
        - flag=0, syndrome=1: Trigger error correction procedure
        """
        # Pull out flag and syndrome from previous frame
        flag = measurement_outcomes[flag_qubit][0]
        syndrome = measurement_outcomes[syndrome_qubit][0]
        if flag:
            # Follow branch 1a (or analogous) for Error Correction Procedure
            # Do unflagged measurements and pass to flagged decoder
            instructions = [
                "Unflagged XZZXI Check",
                "Unflagged IXZZX Check",
                "Unflagged XIXZZ Check",
                "Unflagged ZXIXZ Check",
                f"Flagged {stabilizer} Decoder",
            ]
        elif syndrome:
            # Follow branch 1b (or analogous) for Error Correction Procedure
            # Do unflagged measurements and pass to unflagged decoder
            instructions = [
                "Unflagged XZZXI Check",
                "Unflagged IXZZX Check",
                "Unflagged XIXZZ Check",
                "Unflagged ZXIXZ Check",
                "Unflagged Decoder",
            ]
        else:
            # Flag and syndrome trivial, move to next stabilizer
            if next_stabilizer == "TERMINATE":
                # Special case for last flagged check, no errors detected
                # We are done with QEC!
                instructions = []
            else:
                # Otherwise, add the next flagged check
                # Note that the Feed-Forward is a forward reference to this type of instruction
                instructions = [
                    f"Flagged {next_stabilizer} Check",
                    f"Flagged {next_stabilizer} Feed-Forward",
                ]

        # In all cases, we are doing a stack update
        # (can be empty for case where no errors detected on last flagged check)
        # This part just looks like a composite instruction
        for i, instruction in enumerate(instructions):
            new_label = InstructionLabel(instruction, patch_label)
            stack = stack.insert_instruction(i, new_label)

        return Frame({"stack": stack})

    def flagged_feedforward_map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        flag_qubit: str,
        syndrome_qubit: str,
        **kwargs,
    ):
        """Map qubits function for flagged feedforward instruction.
        
        Remaps flag and syndrome qubit labels according to the provided mapping.
        This function enables the flagged feedforward instruction to work with
        different qubit label schemes by translating between them.
        
        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        flag_qubit : str
            Original flag qubit label to be remapped.
        syndrome_qubit : str
            Original syndrome qubit label to be remapped.
        **kwargs : dict
            Additional keyword arguments to preserve unchanged.
        
        Returns
        -------
        KwargDict
            Updated keyword arguments with remapped flag_qubit and syndrome_qubit
            labels, preserving all other arguments.
        
        Notes
        -----
        This function is used as the map_qubits_fn parameter when creating
        Instruction objects that need to support qubit relabeling.
        """
        new_kwargs = kwargs.copy()
        new_kwargs["flag_qubit"] = qubit_mapping[flag_qubit]
        new_kwargs["syndrome_qubit"] = qubit_mapping[syndrome_qubit]
        return new_kwargs

    # Now to make our four feed-forward instructions
    instructions["Flagged XZZXI Feed-Forward"] = Instruction(
        flagged_feedforward_apply_fn,
        {
            "stabilizer": "XZZXI",
            "next_stabilizer": "IXZZX",
            "flag_qubit": "A1",
            "syndrome_qubit": "A0",
        },
        flagged_feedforward_map_qubits_fn,
        name="Flagged XZZXI Feed-Forward",
    )

    instructions["Flagged IXZZX Feed-Forward"] = Instruction(
        flagged_feedforward_apply_fn,
        {
            "stabilizer": "IXZZX",
            "next_stabilizer": "XIXZZ",
            "flag_qubit": "A1",
            "syndrome_qubit": "A0",
        },
        flagged_feedforward_map_qubits_fn,
        name="Flagged IXZZX Feed-Forward",
    )

    instructions["Flagged XIXZZ Feed-Forward"] = Instruction(
        flagged_feedforward_apply_fn,
        {
            "stabilizer": "XIXZZ",
            "next_stabilizer": "ZXIXZ",
            "flag_qubit": "A1",
            "syndrome_qubit": "A0",
        },
        flagged_feedforward_map_qubits_fn,
        name="Flagged XIXZZ Feed-Forward",
    )

    instructions["Flagged ZXIXZ Feed-Forward"] = Instruction(
        flagged_feedforward_apply_fn,
        {
            "stabilizer": "ZXIXZ",
            "next_stabilizer": "TERMINATE",
            "flag_qubit": "A1",
            "syndrome_qubit": "A0",
        },
        flagged_feedforward_map_qubits_fn,
        name="Flagged ZXIXZ Feed-Forward",
    )

    # Finally, we can define the QEC instruction
    # This will just be the first stage of the flagged checks
    instructions["Flagged QEC"] = builders.build_composite_instruction(
        [
            instructions["Flagged XZZXI Check"],
            instructions["Flagged XZZXI Feed-Forward"],
        ],
        name="Flagged QEC",
    )
