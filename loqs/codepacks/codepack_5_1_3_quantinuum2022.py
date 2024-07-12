"""A LoQS QEC codepack for the 5-qubit code.

This implementation is based on the 2022 implementation from
Quantinuum in :cite:`ryananderson_implementing_2022`, which is in turn
based on piecewise fault-tolerance from :cite:`yoder_universal_2016`
and flag fault-tolerance from :cite:`chao_quantum_2018`.

As we are using flag fault-tolerance, we require two auxiliary qubits:
a measurement qubit for stabilizer checks, and an additional flag qubit.
Thus, we will have 7 qubits total: 5 data and 2 auxiliary.

.. bibliography:: ../../docs/references.bib
    :filter: docname in docnames
"""

from collections.abc import Sequence
from typing import Mapping

from loqs.backends import propagate_state
from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.backends.model.basemodel import BaseNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.state.basestate import BaseQuantumState
from loqs.core import Instruction, QECCode
from loqs.core.frame import Frame
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import KwargDict
from loqs.core.instructions.instructionlabel import InstructionLabel
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes


def create_qec_code(
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
):
    """Create a QECCode implementing the [[5,1,3]] code.

    Parameters
    ----------
    circuit_backend:
        The circuit backend to use when generating physical circuits.
        Currently, only :class:`PyGSTiPhysicalCircuit` is allowed.

    Returns
    -------
        A :class:`QECCode` implementing the [[5,1,3]] code.
    """

    # Template qubits for defining one patch
    qubits = ["A0", "A1"] + [f"D{i}" for i in range(5)]

    instructions: dict[str, Instruction] = {}

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
    instructions["Non-FT Minus Prep"] = (
        builders.build_physical_circuit_instruction(
            nonft_state_prep_circ,
            include_outcomes=False,
            name="Non-FT minus state prep",
            fault_tolerant=False,
        )
    )

    # Try-until-success FT |-> state prep
    # First green box of Fig 3 of arxiv:2208.01863
    ft_state_prep_checks_circ = circuit_backend(
        [
            # FT check 1
            ("Gcnot", "A0", "D0"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D1"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D4"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
            # FT check 2
            ("Gcphase", "A0", "D0"),
            ("Gcnot", "A0", "A1"),
            ("Gcnot", "A0", "D1"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D2"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
            # FT check 3
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
    ft_state_prep_circ = nonft_state_prep_circ.append(
        ft_state_prep_checks_circ
    )
    ft_state_prep = builders.build_physical_circuit_instruction(
        ft_state_prep_circ,
        include_outcomes=True,
        reset_mcms=True,
        name="Non-FT Minus Prep + Checks",
        fault_tolerant=False,
    )
    instructions["FT Minus Prep"] = (
        builders.build_repeat_until_success_instruction(
            ft_state_prep,
            name="Repeat-until-success FT Minus Prep",
            fault_tolerant=True,
        )
    )

    # Logical X (transversal)
    # Eqn B1 of arxiv:2208.01863
    logical_X_circ = circuit_backend(
        [[("Gypi", "D0"), ("Gxpi", "D2"), ("Gypi", "D4")]],
        qubit_labels=qubits,
    )
    instructions["X"] = builders.build_physical_circuit_instruction(
        logical_X_circ,
        include_outcomes=False,
        name="Logical X",
        fault_tolerant=True,
    )

    # Logical Z (transversal)
    # Eqn B3 of arxiv:2208.01863
    logical_Z_circ = circuit_backend(
        [[("Gxpi", "D0"), ("Gzpi", "D2"), ("Gxpi", "D4")]],
        qubit_labels=qubits,
    )
    instructions["Z"] = builders.build_physical_circuit_instruction(
        logical_Z_circ,
        include_outcomes=False,
        name="Logical Z",
        fault_tolerant=True,
    )

    # Logical H (transversal + permute)
    # Fig 2b of arxiv:1603.03948
    logical_H_circ = circuit_backend(
        [[("Gh", q) for q in qubits[2:]]], qubit_labels=qubits
    )
    logical_H_circ_inst = builders.build_physical_circuit_instruction(
        logical_H_circ,
        include_outcomes=False,
        name="Logical H circuit",
        fault_tolerant=True,
    )
    logical_H_permutation = builders.build_patch_permute_instruction(
        {"D0": "D3", "D1": "D0", "D3": "D4", "D4": "D1"},  # initial: final
        name="Logical H permutation",
    )

    instructions["H"] = builders.build_composite_instruction(
        [logical_H_circ_inst, logical_H_permutation],
        param_priorities=["patch_label"],
        name="Logical H",
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
    instructions["Logical Prime Basis Transform"] = (
        builders.build_physical_circuit_instruction(
            to_prime_basis_circ,
            include_outcomes=False,
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
    instructions["Logical Prime Basis Inverse Transform"] = (
        builders.build_physical_circuit_instruction(
            from_prime_basis_circ,
            include_outcomes=False,
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
            [("Gh", "D0"), ("Gh", "D2"), ("Gh", "D4")],
        ],
        qubit_labels=qubits,
    )

    prime_basis_Z_meas = builders.build_physical_circuit_instruction(
        to_prime_basis_circ.append(raw_Z_meas_circ),
        include_outcomes=True,
        reset_mcms=False,
        name="Raw logical Z-basis measurement",
        fault_tolerant=False,
    )

    prime_basis_X_meas = builders.build_physical_circuit_instruction(
        to_prime_basis_circ.append(raw_X_meas_circ),
        include_outcomes=True,
        reset_mcms=False,
        name="Raw logical X-basis measurement",
        fault_tolerant=False,
    )

    # We can also compute the logical measurement based on the raw logical output
    # In the prime basis, an odd number of 0s is 0 and an odd number of 1s is 1
    # This is because our logical operations are ZIZIZ and XIXIX in the prime basis
    # E.g. all 0s is 0 for ZIZIZ, but so is one 0 and two 1s because the phase on the ones cancels
    # We can define a new operation here which post processes the measurement outcomes of Raw Logical * Measure
    def nonft_logical_meas_apply_fn(
        measurement_outcomes: MeasurementOutcomes,
    ) -> Frame:
        logical_value = sum([v[0] for v in measurement_outcomes.values()]) % 2
        return Frame({"logical_measurement": logical_value})

    nonft_logical_meas = Instruction(
        nonft_logical_meas_apply_fn,
        ["logical_measurement"],
        name="Non-FT Logical Measurement",
        fault_tolerant=False,
    )

    instructions["Non-FT Logical Z Measure"] = (
        builders.build_composite_instruction(
            [prime_basis_Z_meas, nonft_logical_meas],
            [],
            name="Non-FT logical Z measurement (via prime basis measurement)",
            fault_tolerant=False,
        )
    )

    instructions["Non-FT Logical X Measure"] = (
        builders.build_composite_instruction(
            [prime_basis_X_meas, nonft_logical_meas],
            [],
            name="Non-FT logical X measurement (via prime basis measurement)",
            fault_tolerant=False,
        )
    )

    ### DECODING CIRCUIT
    # TODO: Really this is the nonft state prep reversed. Reuse?
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
    instructions["Non-FT Minus Unprep"] = (
        builders.build_physical_circuit_instruction(
            state_decoder_circ,
            name="Non-FT state decoder circuit",
            reset_mcms=False,
            fault_tolerant=False,
        )
    )

    # Fig 13 of arxiv:2208.01863
    # Adds the instructions in-place
    _create_adaptive_measure_instruction(instructions, qubits, circuit_backend)

    # TODO: QEC instruction

    code = QECCode(instructions, qubits, "Perfect [[5,1,3]] code")
    return code


def create_ideal_model(
    qubits: Sequence[str],
    model_backend: type[BaseNoiseModel] = PyGSTiNoiseModel,
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
        A :class:`QECCode` implementing the [[5,1,3]] code.
    """
    assert len(qubits) == 7, "Must provide exactly 7 qubit labels"
    assert issubclass(
        model_backend, PyGSTiNoiseModel
    ), "Only pyGSTi models can be output by the codepack"

    gate_names = [
        "Gxpi",
        "Gypi",
        "Gzpi",
        "Gzpi2",
        "Gzmpi2",
        "Gh",
        "Gcnot",
        "Gcphase",
    ]

    try:
        import pygsti
    except ImportError:
        raise RuntimeError(
            "pyGSTi not found, cannot construct pyGSTi noise model"
        )
    # TODO: Currently Iz does not need to be set here
    # This is because QSimQuantumState actually does not try to pull the rep
    # Otherwise, this would result in a KeyError in PyGSTiNoiseModel.get_reps(),
    # since it technically should be provided
    pspec = pygsti.processors.QubitProcessorSpec(
        len(qubits),
        gate_names=gate_names,
        qubit_labels=qubits,
        availability={k: "all-permutations" for k in gate_names},
    )

    ideal_model_pygsti = pygsti.models.create_crosstalk_free_model(pspec)

    return model_backend(ideal_model_pygsti)


## Helper functions
def _create_adaptive_measure_instruction(
    instructions, qubits, circuit_backend
):
    # FT Adaptive Measurement Scheme
    # Fig 13 of arxiv:2208.01863
    # Easiest to go from right to left when building up instructions

    # There are two end cases
    # Either we hit the decoding circuit (already defined above), or we terminate

    ### TERMINATION
    # This is a simple operation that returns the previous auxiliary outcome
    def term_apply_fn(outcomes: MeasurementOutcomes, meas_qubit: str) -> Frame:
        return Frame({"logical_measurement": outcomes[meas_qubit][0]})

    term_data = {"meas_qubit": "A0"}

    # Need to map our measure qubit to the patch
    def term_map_qubits_fn(
        qubit_mapping: Mapping[str, str], meas_qubit: str, **kwargs
    ) -> KwargDict:
        return {"meas_qubit": qubit_mapping[meas_qubit]}

    instructions["Adaptive Measure Termination"] = Instruction(
        term_apply_fn,
        ["logical_measurement"],  # Can output dummy frame for DRY_RUN
        term_data,
        term_map_qubits_fn,
        name="Termination for adaptive logical measurement",
        fault_tolerant=True,
    )

    ### Helper functions
    # This is the mapping function used by physical circuit instructions
    # We can use this for all circuit-based instructions
    def map_qubits_fn(
        qubit_mapping: Mapping[str, str],
        circuit: BasePhysicalCircuit,
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["circuit"] = circuit.map_qubit_labels(qubit_mapping)
        return new_kwargs

    ### PART III
    # Part III circuit
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

    # Heart of part III, with two outcomes
    def partIII_apply_fn(
        model: BaseNoiseModel,
        stack: InstructionStack,
        patch_label: str,
        circuit: BasePhysicalCircuit,
        inplace: bool,
        state: BaseQuantumState,
        previous_outcomes: list[MeasurementOutcomes],
    ) -> Frame:
        # Run circuit
        new_state, outcomes = propagate_state(circuit, model, state, inplace)

        assert len(previous_outcomes) == 2

        # Pull measurements/flags
        meas_qubit = circuit.qubit_labels[0]
        flag_qubit = circuit.qubit_labels[1]
        F3 = outcomes[flag_qubit][0]
        M1 = previous_outcomes[-2][meas_qubit][0]
        M2 = previous_outcomes[-1][meas_qubit][0]
        M3 = outcomes[meas_qubit][0]

        # Do feed forward
        if F3 == 0 and M1 == M2 and M1 != M3:
            # If F3 = 0 and M1 = M2 != M3, go to decoding circuit
            next_instruction = "Non-FT Minus Unprep"
        else:
            # Otherwise, we terminate
            next_instruction = "Adaptive Measure Termination"

        # We need to make sure and feed the patch label forward
        new_label = InstructionLabel(next_instruction, patch_label)
        new_stack = stack.insert_instruction(0, new_label)

        # Return new frame
        frame_data = {
            "stack": new_stack,
            "state": new_state,
            "measurement_outcomes": MeasurementOutcomes(outcomes),
        }
        return Frame(frame_data)

    partIII_data = {
        "circuit": measIII_circ,
        "inplace": True,  # TODO: Not hardcode this,
    }

    # The param collection code will mostly work for this instruction
    # The one exception is previous_outcomes, which needs to go back 2 frames
    # and also must be aliased
    paramIII_priorities = {"previous_outcomes": ["history[-2,-1]"]}
    paramIII_aliases = {"previous_outcomes": "measurement_outcomes"}

    # We can now define the whole instruction
    instructions["Adaptive Measure Part III"] = Instruction(
        partIII_apply_fn,
        ["state", "measurement_outcomes"],  # Can use dummy frame for dry run
        partIII_data,
        map_qubits_fn,
        param_priorities=paramIII_priorities,
        param_aliases=paramIII_aliases,
        name="Part III of adaptive logical measurement",
        fault_tolerant=True,
    )

    ### PART II
    # This follows part III closely
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

    # Heart of part II, with three outcomes
    def partII_apply_fn(
        model: BaseNoiseModel,
        stack: InstructionStack,
        patch_label: str,
        circuit: BasePhysicalCircuit,
        inplace: bool,
        state: BaseQuantumState,
        previous_outcome: MeasurementOutcomes,
    ) -> Frame:
        # Run circuit
        new_state, outcomes = propagate_state(circuit, model, state, inplace)

        # Pull measurements/flags
        meas_qubit = circuit.qubit_labels[0]
        flag_qubit = circuit.qubit_labels[1]
        F2 = outcomes[flag_qubit][0]
        M1 = previous_outcome[meas_qubit][0]
        M2 = outcomes[meas_qubit][0]

        # Do classical feed forward
        if F2 != 0:
            # We go to termination
            next_instruction = "Adaptive Measure Termination"
        elif M1 == M2:
            # We go to part III
            next_instruction = "Adaptive Measure Part III"
        else:
            # We go to decoding circuit
            next_instruction = "Non-FT Minus Unprep"

        # We need to make sure and feed the patch label forward
        new_label = InstructionLabel(next_instruction, patch_label)
        new_stack = stack.insert_instruction(0, new_label)

        # Return new frame
        frame_data = {
            "stack": new_stack,
            "state": new_state,
            "measurement_outcomes": MeasurementOutcomes(outcomes),
        }
        return Frame(frame_data)

    partII_data = {
        "circuit": measII_circ,
        "inplace": True,  # TODO: Not hardcode this
    }

    # We don't need more than one frame back in history,
    # but we still need the alias
    paramII_aliases = {"previous_outcome": "measurement_outcomes"}

    # We can now define the whole instruction
    instructions["Adaptive Measure Part II"] = Instruction(
        partII_apply_fn,
        ["state", "measurement_outcomes"],  # Dummy frame is fine for dry run
        partII_data,
        map_qubits_fn,
        param_aliases=paramII_aliases,
        name="Part II of adaptive logical measurement",
        fault_tolerant=True,
    )

    ### PART I
    # Finally we have the first circuit
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

    # Heart of part I, with two outcomes
    def partI_apply_fn(
        model: BaseNoiseModel,
        stack: InstructionStack,
        patch_label: str,
        circuit: BasePhysicalCircuit,
        inplace: bool,
        state: BaseQuantumState,
    ) -> Frame:
        # Run circuit
        new_state, outcomes = propagate_state(circuit, model, state, inplace)

        # Do classical feed forward
        flag_qubit = circuit.qubit_labels[1]
        F1 = outcomes[flag_qubit][0]
        if F1 == 0:
            # We go to part II
            next_instruction = "Adaptive Measure Part II"
        else:
            # We go to decoding circuit
            next_instruction = "Non-FT Minus Unprep"

        # We need to make sure and feed the patch label forward
        new_label = InstructionLabel(next_instruction, patch_label)
        new_stack = stack.insert_instruction(0, new_label)

        # Return new frame
        frame_data = {
            "stack": new_stack,
            "state": new_state,
            "measurement_outcomes": MeasurementOutcomes(outcomes),
        }
        return Frame(frame_data)

    partI_data = {
        "circuit": measI_circ,
        "inplace": True,  # TODO: Not hardcode this
    }

    # We can now define the whole instruction
    instructions["Adaptive Measure Part I"] = Instruction(
        partI_apply_fn,
        ["state", "measurement_outcomes"],  # Dummy frame is ok for dry run
        partI_data,
        map_qubits_fn,
        name="Part I of adaptive logical measurement",
        fault_tolerant=True,
    )

    # For convenience, let's also add an entry for the whole operation
    # For now, this is just part I
    # TODO: Once we have stabilizer frame, this should be composite
    # with part I and then a final operation to do decoding
    instructions["Adaptive Measure"] = builders.build_composite_instruction(
        [instructions["Adaptive Measure Part I"]],
        ["patch_label"],
        name="FT adaptive measure",
        fault_tolerant=True,
    )
