"""A LoQS QEC codepack for the [[7,1,3]] color code.

This implementation is based on the 2021 implementation from
Quantinuum in :cite:`ryananderson_realizing_2021`.

We require three auxiliary qubits for stabilizer checks.
Thus, we will have 10 qubits total: 7 data and 3 auxiliary.

.. bibliography::
    :filter: docname in docnames
"""

from collections.abc import Sequence
import itertools
from typing import Literal, Mapping
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
    """Create a QECCode implementing the [[7,1,3]] code.

    Parameters
    ----------
    include_idles:
        Whether to include (``True``) or not (``False``, default) idle gates
        in physical circuits.

    gate_durations:
        Mapping from gate names to durations. Defaults to ``None``, which uses
        dummy values 1, 2, 3 for 1Q gates, 2Q gates, and mid-circuit
        measurements, respectively.
        See ``durations`` from
        :meth:`.BasePhysicalCircuit.pad_single_qubit_idles_by_duration_inplace`
        for more details.

    idle_gates:
        Mapping from gate duration to idle gate names. Defaults to ``None``,
        which maps the dummy values from ``gate_durations`` to ``"Gi1Q"``,
        ``"Gi2Q"``, and ``"GiMCM"``, respectively.
        See ``idle_names`` from
        :meth:`.BasePhysicalCircuit.pad_single_qubit_idles_by_duration_inplace`
        for more details.

    circuit_backend:
        The circuit backend to use when generating physical circuits.

    Returns
    -------
        A :class:`.QECCode` implementing the [[5,1,3]] code.
    """

    # Template qubits for defining one patch
    qubits = ["A0", "A1", "A2"] + [f"D{i}" for i in range(7)]
    data_qubits = qubits[2:]

    instructions: dict[str, Instruction] = {}

    # For padding by idles with duration
    # TODO
    if gate_durations is None:
        gate_durations = {
            k: 1
            for k in [
                "Gi",
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
        gate_durations["Iz"] = 3
    if idle_gates is None:
        idle_gates = {1: "Gi1Q", 2: "Gi2Q", 3: "GiMCM"}

    ## PREP
    # Non-FT |0> state prep
    # Encoding circuit box of Fig 10 of 10.1103/PhysRevX.11.041058
    # without auxiliary qubit check
    nonft_state_prep_circ = circuit_backend(
        [
            [
                ("Gh", "D0"),
                ("Gh", "D5"),
                ("Gh", "D7"),
            ],
            [
                ("Gcnot", "D0", "D1"),
                ("Gcnot", "D5", "D6"),
                ("Gcnot", "D7", "D4"),
            ],
            [
                ("Gcnot", "D0", "D3"),
                ("Gcnot", "D5", "D3"),
                ("Gcnot", "D7", "D6"),
            ],
            [
                ("Gcnot", "D0", "D3"),
                ("Gcnot", "D5", "D3"),
                ("Gcnot", "D7", "D6"),
            ],
            [("Gcnot", "D5", "D2"), ("Gcnot", "D4", "D3")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        nonft_state_prep_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Non-FT Zero Prep"] = (
        builders.build_physical_circuit_instruction(
            nonft_state_prep_circ,
            name="Non-FT zero state prep",
        )
    )

    ### Try-until-success FT |0> state prep
    # Qubit reset in case of failure
    reset = builders.build_physical_circuit_instruction(
        circuit_backend(
            [[("Iz", q) for q in qubits[3:]]], qubit_labels=qubits
        ),
        name="Reset to all 0 state",
    )

    # Auxiliary qubit check from Encoding circuit box of Fig 10 of 10.1103/PhysRevX.11.041058
    ft_state_prep_checks_circ = circuit_backend(
        [
            [("Gcnot", "D2", "A0")],
            [("Gcnot", "D4", "A0")],
            [("Gcnot", "D6", "A0")],
            [("Iz", "A0")],
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
    instructions["Non-FT Zero Prep + Checks"] = (
        builders.build_physical_circuit_instruction(
            ft_state_prep_circ,
            name="Non-FT Zero Prep + Checks",
        )
    )

    # On success, we expect 0 outcome on the flag qubit from the check circuit
    rus_success_expected = MeasurementOutcomes({"A0": [0]})

    instructions["FT Zero Prep"] = (
        builders.build_repeat_until_success_instruction(
            [reset, instructions["Non-FT Zero Prep + Checks"]],
            rus_key="FT Minus Prep",
            test_frame_key="measurement_outcomes",
            expected=rus_success_expected,
            max_repeats=ft_state_prep_max_repeats,
            name="Repeat-until-success FT Zero Prep",
        )
    )

    ## GATES
    # Logical Clifford gates (transversal)
    # (or rather, a common subset of them)
    # Table from Fig 1 of 10.1103/PhysRevX.11.041058
    clifford_gates = {
        "X": "Gxpi",
        "Y": "Gypi",
        "Z": "Gzpi",
        "H": "Gh",
        "S": "Gzmpi2",  # Logical S is really all Sdagger...
        "Sdag": "Gzpi2",  # ...and vice versa
        "I": "Gi",
    }
    # Paulis only act on last three data qubits (bottom row)
    edge_qubits = qubits[-3:]
    for n, gn in clifford_gates.items():
        # Bottom row for paulis, all data for others
        active_qubits = edge_qubits if n in "XYZ" else qubits[3:]
        logical_circ = circuit_backend(
            [[(gn, q) for q in active_qubits]],
            qubit_labels=qubits,
        )
        if include_idles:
            logical_circ.pad_single_qubit_idles_by_duration_inplace(
                idle_gates, gate_durations
            )
        instructions[n] = builders.build_physical_circuit_instruction(
            logical_circ,
            name=f"Logical {n}",
        )

    ## QEC
    # This is "First flagged parallel circuit" from Figure 10 of 10.1103/PhysRevX.11.041058
    flagged_QEC_part1_circ = circuit_backend(
        [
            [("Gh", "A0")],
            [
                ("Gcnot", "A0", "D3"),
                ("Gcnot", "D2", "A2"),
                ("Gcnot", "D5", "A1"),
            ],
            [("Gcnot", "A0", "A1")],
            [
                ("Gcnot", "A0", "D0"),
                ("Gcnot", "D3", "A2"),
                ("Gcnot", "D4", "A1"),
            ],
            [
                ("Gcnot", "A0", "D1"),
                ("Gcnot", "D6", "A2"),
                ("Gcnot", "D2", "A1"),
            ],
            [("Gcnot", "A0", "A2")],
            [
                ("Gcnot", "A0", "D2"),
                ("Gcnot", "D5", "A2"),
                ("Gcnot", "D1", "A1"),
            ],
            [("Gh", "A0")],
            [("Iz", "A0"), ("Iz", "A1"), ("Iz", "A2")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        flagged_QEC_part1_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Flagged Parallel S1-S5-S6 Check"] = (
        builders.build_physical_circuit_instruction(
            flagged_QEC_part1_circ,
            name="Flagged S1 Check",
        )
    )

    # This is "Second flagged parallel circuit" from Figure 10 of 10.1103/PhysRevX.11.041058
    flagged_QEC_part2_circ = circuit_backend(
        [
            [("Gh", "A1"), ("Gh", "A2")],
            [
                ("Gcnot", "D3", "A0"),
                ("Gcnot", "A2", "D2"),
                ("Gcnot", "A1", "D5"),
            ],
            [("Gcnot", "A1", "A0")],
            [
                ("Gcnot", "D0", "A0"),
                ("Gcnot", "A2", "D3"),
                ("Gcnot", "A1", "D4"),
            ],
            [
                ("Gcnot", "D1", "A0"),
                ("Gcnot", "A2", "D6"),
                ("Gcnot", "A1", "D2"),
            ],
            [("Gcnot", "A2", "A0")],
            [
                ("Gcnot", "D2", "A0"),
                ("Gcnot", "A2", "D5"),
                ("Gcnot", "A1", "D1"),
            ],
            [("Gh", "A1"), ("Gh", "A2")],
            [("Iz", "A0"), ("Iz", "A1"), ("Iz", "A2")],
        ],
        qubit_labels=qubits,
    )
    if include_idles:
        flagged_QEC_part2_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Flagged Parallel S2-S3-S4 Check"] = (
        builders.build_physical_circuit_instruction(
            flagged_QEC_part2_circ,
            name="Flagged S1 Check",
        )
    )

    # These are the 6 unflagged stabilizer checks from Figure 10 of 10.1103/PhysRevX.11.041058
    # Scheduling less clear, we provide individual and merged Z/X circuits
    # We do this by defining a template circuit and mapping to the various plaquettes
    Z_temp_circ = circuit_backend(
        [[("Gcnot", q, "aux")] for q in "abcd"] + [[("Iz", "aux")]]
    )
    X_temp_circ = circuit_backend(
        [[("Gh", "aux")]]
        + [[("Gcnot", "aux", q)] for q in "abcd"]
        + [[("Gh", "aux")], [("Iz", "aux")]]
    )
    mappings = {
        0: ["D0", "D1", "D2", "D4"],
        1: ["D1", "D2", "D4", "D5"],
        2: ["D2", "D3", "D5", "D6"],
    }
    parallel_X_circ = circuit_backend([], qubit_labels=qubits)
    parallel_Z_circ = circuit_backend([], qubit_labels=qubits)
    for i, m in mappings.items():
        # Map circuits
        mapping = {q1: q2 for q1, q2 in zip("abcd", m)}
        mapping["aux"] = f"A{i}"
        X_circ = X_temp_circ.map_qubit_labels(mapping)
        Z_circ = Z_temp_circ.map_qubit_labels(mapping)

        # Standalone checks
        if include_idles:
            X_circ_padded = X_circ.pad_single_qubit_idles_by_duration(
                idle_gates, gate_durations
            )
            Z_circ_padded = Z_circ.pad_single_qubit_idles_by_duration(
                idle_gates, gate_durations
            )
        else:
            X_circ_padded = X_circ
            Z_circ_padded = Z_circ
        instructions[f"Unflagged S{i+1} Check"] = (
            builders.build_physical_circuit_instruction(
                X_circ_padded, name=f"Unflagged S{i+1} Check"
            )
        )
        instructions[f"Unflagged S{i+4} Check"] = (
            builders.build_physical_circuit_instruction(
                Z_circ_padded, name=f"Unflagged S{i+1} Check"
            )
        )

        # Merge (unpadded) into parallel check
        parallel_X_circ.merge_inplace(X_circ, 0)
        parallel_Z_circ.merge_inplace(Z_circ, 0)
    # Parallel checks
    if include_idles:
        parallel_X_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
        parallel_Z_circ.pad_single_qubit_idles_by_duration_inplace(
            idle_gates, gate_durations
        )
    instructions["Unflagged Parallel S1-S2-S3 Check"] = (
        builders.build_physical_circuit_instruction(
            parallel_X_circ, name="Unflagged Parallel S1-S2-S3 Check"
        )
    )
    instructions["Unflagged Parallel S4-S5-S6 Check"] = (
        builders.build_physical_circuit_instruction(
            parallel_Z_circ, name="Unflagged S4-S5-S6 Check"
        )
    )

    # Now that we have all the circuits, we can make our feed-forward QEC instructions

    ## MEASURE
    # Full data qubit measurements
    raw_Z_meas_circ = circuit_backend(
        [[("Iz", q) for q in qubits[3:]]], qubit_labels=qubits
    )
    raw_X_meas_circ = circuit_backend(
        [
            [("Gh", q) for q in qubits[3:]],
            [("Iz", q) for q in qubits[3:]],
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
    instructions["Raw Z Data Measure"] = (
        builders.build_physical_circuit_instruction(
            raw_Z_meas_circ,
            name="Raw logical Z-basis measurement",
        )
    )
    instructions["Raw X measure circuit"] = (
        builders.build_physical_circuit_instruction(
            raw_X_meas_circ,
            name="Raw logical X-basis measurement",
        )
    )

    # We can also compute the logical measurement based on the raw logical output
    # This is described in Section II.A.4 of 10.1103/PhysRevX.11.041058, summarized here:
    # 1. Compute the non-FT logical outcome as the product of outcomes of bottom edge of plaquette
    # 2. Classically compute the syndrome (we will only get the syndrome for the opposite of the measure basis)
    # 3. Decode the syndrome and determine whether to flip the raw logical output
    #    (multiple ways to do this, we'll just use the syndrome to update the Pauli frame and infer again.
    #    Could also look at weight of correction on edge qubits, odd weight = flip, even weight = don't flip.)
    def logical_meas_apply_fn(
        patch_label: str,
        patches: PatchDict,
        data_qubits: list[str],
        basis: Literal["X", "Z"],
        pauli_frame_per_patch: dict[str, list[int]],
        measurement_outcomes: MeasurementOutcomes,
    ) -> Frame:
        # Get pauli frame
        pauli_frame = patches[patch_label].pauli_frame

        # Compute inferred bitstring
        assert basis in "XZ"
        inferred_outcomes = measurement_outcomes.get_inferred_outcomes(
            pauli_frame, basis
        )
        inferred_bitstring = [inferred_outcomes[q][0] for q in edge_qubits]

        uncorrected_outcome = sum(inferred_bitstring) % 2

        plaq_idxs = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
        classical_syndrome = ""
        for plaq in plaq_idxs:
            plaq_outcomes = [
                inferred_outcomes[data_qubits[q]][0] for q in plaq
            ]
            parity = sum(plaq_outcomes) % 2
            classical_syndrome += str(parity)

        # TODO
        correction = 0

        logical_outcome = (uncorrected_outcome + correction) % 2
        return Frame(
            {
                "logical_measurement": logical_outcome,
                "inferred_outcomes": inferred_outcomes,
                "uncorrected_measurement": uncorrected_outcome,
                "classical_syndrome": classical_syndrome,
            }
        )

    def logical_meas_map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        edge_qubits: list[str],
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["edge_qubits"] = [qubit_mapping[q] for q in edge_qubits]
        return new_kwargs

    Z_logical_meas = Instruction(
        logical_meas_apply_fn,
        data={"data_qubits": qubits[3:], "basis": "Z"},
        map_qubits_fn=logical_meas_map_qubits_fn,
        name="Non-FT Z logical parity calculation",
    )

    instructions["FT Logical Z Measure"] = (
        builders.build_composite_instruction(
            [instructions["Raw Z Data Measure"], Z_logical_meas],
            name="FT logical Z measurement",
        )
    )

    # X_logical_meas = Instruction(
    #     logical_meas_apply_fn,
    #     data={"data_qubits": qubits[3:], "basis": "X"},
    #     map_qubits_fn=logical_meas_map_qubits_fn,
    #     name="Non-FT X logical parity calculation",
    # )

    instructions["FT Logical X Measure"] = (
        builders.build_composite_instruction(
            [
                instructions["H"],
                instructions["Raw Z Data Measure"],
                Z_logical_meas,
            ],
            name="FT logical Z measurement",
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
    physical circuits in the :class:`QECCode` returned by :meth:`create_qec_code()`.


    Parameters
    ----------
    qubits:
        List of qubit labels to use. It should be have 7 entries,
        and the first two qubits should be the auxiliary qubits.

    model_backend:
        The model backend to use when generating operations.
        Currently, only :class:`PyGSTiNoiseModel` is allowed.

    Returns
    -------
        A noiseless model for the `QECCode` returned by
        :meth:`create_qec_code`
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
