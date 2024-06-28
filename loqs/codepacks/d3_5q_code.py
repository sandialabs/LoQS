"""A LoQS QEC codepack for the 5-qubit code.

TODO
"""

from collections.abc import Sequence
import itertools
from typing import Mapping
import numpy as np

from loqs.backends import propagate_state
from loqs.backends.circuit import PyGSTiPhysicalCircuit as PhysicalCircuit
from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.model.basemodel import BaseNoiseModel
from loqs.backends.state.basestate import BaseQuantumState
from loqs.core import Instruction, QECCode
from loqs.core.frame import Frame
from loqs.core.instructions import common as ic
from loqs.core.instructions.instruction import KwargDict
from loqs.core.instructions.instructionlabel import InstructionLabel
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes


def create_qec_code():
    """TODO"""

    # Template qubits for defining one patch
    qubits = ["A0", "A1"] + [f"D{i}" for i in range(5)]

    operations: dict[str, Instruction] = {}

    # Non-FT |-> state prep
    # First gray box of Fig 3 of arxiv:2208.01863
    nonft_state_prep_circ = PhysicalCircuit(
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

    # TODO: Verify this is actually minus
    operations["Non-FT Minus Prep"] = ic.build_physical_circuit_instruction(
        nonft_state_prep_circ,
        include_outcomes=False,
        name="Non-FT minus state prep",
        fault_tolerant=False,
    )

    # Try-until-success FT |-> state prep
    # First green box of Fig 3 of arxiv:2208.01863
    # ft_state_prep_checks_circ = PhysicalCircuit(
    #     [
    #         # FT check 1
    #         ("Gcnot", "A0", "D0"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcphase", "A0", "D1"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcphase", "A0", "D4"),
    #         ("Gh", "A0"),
    #         [("Iz", "A0"), ("Iz", "A1")],
    #         # FT check 2
    #         ("Gcphase", "A0", "D0"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcnot", "A0", "D1"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcphase", "A0", "D2"),
    #         ("Gh", "A0"),
    #         [("Iz", "A0"), ("Iz", "A1")],
    #         # FT check 3
    #         ("Gcphase", "A0", "D0"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcnot", "A0", "D1"),
    #         ("Gcnot", "A0", "A1"),
    #         ("Gcphase", "A0", "D2"),
    #         ("Gh", "A0"),
    #         [("Iz", "A0"), ("Iz", "A1")],
    #     ],
    #     qubit_labels=qubits,
    # )
    # ft_state_prep_circ = nonft_state_prep_circ.append(
    #     ft_state_prep_checks_circ
    # )
    # ft_state_prep = ic.build_physical_circuit_instruction(
    #     ft_state_prep_circ,
    #     include_outcomes=True,
    #     reset_mcms=True,
    #     name="Non-fault-tolerant minus state prep",
    #     fault_tolerant=True,
    # )

    # operations["FT Minus Prep"] = RepeatUntilSuccess(
    #     ft_state_prep,
    #     name="Repeat-until-success fault-tolerant minus state prep",
    # )

    # Logical X (transversal)
    # Eqn B1 of arxiv:2208.01863
    logical_X_circ = PhysicalCircuit(
        [[("Gypi2", "D0"), ("Gxpi2", "D2"), ("Gypi2", "D4")]],
        qubit_labels=qubits,
    )
    operations["X"] = ic.build_physical_circuit_instruction(
        logical_X_circ,
        include_outcomes=False,
        name="Logical X",
        fault_tolerant=True,
    )

    # Logical Z (transversal)
    # Eqn B3 of arxiv:2208.01863
    logical_Z_circ = PhysicalCircuit(
        [[("Gxpi2", "D0"), ("Gzpi2", "D2"), ("Gxpi2", "D4")]],
        qubit_labels=qubits,
    )
    operations["Z"] = ic.build_physical_circuit_instruction(
        logical_Z_circ,
        include_outcomes=False,
        name="Logical Z",
        fault_tolerant=True,
    )

    # Logical H (transversal + permute)
    # Fig 2b of arxiv:1603.03948
    logical_H_circ = PhysicalCircuit(
        [[("Gh", q) for q in qubits[2:]]], qubit_labels=qubits
    )
    logical_H_circ_inst = ic.build_physical_circuit_instruction(
        logical_H_circ,
        include_outcomes=False,
        name="Logical H circuit",
        fault_tolerant=True,
    )
    logical_H_permutation = ic.build_patch_permute_instruction(
        {"D0": "D3", "D1": "D0", "D3": "D4", "D4": "D1"},  # initial: final
        name="Logical H permutation",
    )

    operations["H"] = ic.build_composite_instruction(
        [logical_H_circ_inst, logical_H_permutation], name="Logical H"
    )

    # Raw physical measurement
    operations["Non-FT Physical Z Measure"] = (
        ic.build_physical_circuit_instruction(
            PhysicalCircuit(
                [[("Iz", q) for q in qubits]], qubit_labels=qubits
            ),
            include_outcomes=True,
            reset_mcms=False,
            name="Z-basis measurement for physical qubits",
            fault_tolerant=False,
        )
    )

    ### DECODING CIRCUIT
    # TODO: Really this is the nonft state prep reversed. Reuse?
    state_decoder_circ = PhysicalCircuit(
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
    operations["Non-FT Minus Unprep"] = ic.build_physical_circuit_instruction(
        state_decoder_circ,
        name="Non-FT state decoder circuit",
        reset_mcms=False,
        fault_tolerant=False,
    )

    # Fig 13 of arxiv:2208.01863
    # Adds the operations in-place
    _create_adaptive_measure_instruction(operations, qubits)

    # TODO: QEC instruction

    # TODO: Logical CZ and CCZ

    return QECCode(operations, qubits, "Perfect [[5,1,3]] code")


## Helper functions
def _create_adaptive_measure_instruction(operations, qubits):
    # FT Adaptive Measurement Scheme
    # Fig 13 of arxiv:2208.01863
    # Easiest to go from right to left when building up instructions

    # There are two end cases
    # Either we hit the decoding circuit (already defined above), or we terminate

    ### TERMINATION
    # This is a simple operation that returns the previous auxiliary outcome
    term_input_spec = [
        ("default", "meas_qubit"),
        ("history", "outcomes", -1, "measurement_outcomes"),
    ]

    term_output_spec = ["logical_measurement"]

    term_defaults = {"meas_qubit": "A0"}

    def term_apply_fn(outcomes: MeasurementOutcomes) -> Frame:
        return Frame({"logical_measurement": outcomes["A0"][0]})

    # Need to map our measure qubit to the patch
    def term_map_qubits_fn(
        qubit_mapping: Mapping[str, str], meas_qubit: str, **kwargs
    ) -> KwargDict:
        return {"meas_qubit": qubit_mapping[meas_qubit]}

    operations["Adaptive Measure Termination"] = Instruction(
        term_apply_fn,
        term_input_spec,
        term_output_spec,
        term_defaults,
        term_map_qubits_fn,
        name="Termination for adaptive logical measurement",
        fault_tolerant=True,
        skip_in_dry_run=False,  # Just metadata updates, can do in dry run
    )

    ### PART III
    # Part III depends on the state and the two previous measurement outcomes,
    # both of which come from the History, as well as the circuit,
    # which we will store in defaults
    partIII_input_spec = [
        ("label", "model"),  # We will take a model from the QuantumProgram
        ("label", "stack"),  # We will take a stack from the QuantumProgram
        (
            "label",
            "patch_label",
        ),  # We will take the patch_label from the QuantumProgram to feed forward
        ("default", "circuit"),  # We will store the circuit in defaults
        ("default", "inplace"),  # and whether to propagate inplace
        ("history", "state", -1),  # We need the previous state
        (
            "history",
            "previous_outcomes",
            [-2, -1],
            "measurement_outcomes",
        ),  # We need the past 2 outcomes
    ]

    # This is the same output spec for all circuit-based instructions here
    # We will output the state and measurement outcomes, as well as
    # feed-forward the next operation onto the stack
    output_spec = ["state", "measurement_outcomes", "stack"]

    # Here is the actual circuit we will run
    measIII_circ = PhysicalCircuit(
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

    partIII_defaults = {
        "circuit": measIII_circ,
        "inplace": True,  # TODO: Not hardcode this,
    }

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

    # We can now define the whole instruction
    operations["Adaptive Measure Part III"] = Instruction(
        partIII_apply_fn,
        partIII_input_spec,
        output_spec,
        partIII_defaults,
        map_qubits_fn,
        name="Part III of adaptive logical measurement",
        fault_tolerant=True,
    )

    ### PART II
    # This follows part III closely, only we only need to reach back one frame
    partII_input_spec = [
        ("label", "model"),  # We will take a model from the QuantumProgram
        ("label", "stack"),  # We will take a stack from the QuantumProgram
        (
            "label",
            "patch_label",
        ),  # We will take the patch_label from the QuantumProgram to feed forward
        ("default", "circuit"),  # We will store the circuit in defaults
        ("default", "inplace"),  # and whether to propagate inplace
        ("history", "state", -1),  # We need the previous state
        (
            "history",
            "previous_outcome",
            -1,
            "measurement_outcomes",
        ),  # We need the past outcome
    ]

    # Here is the actual circuit we will run
    measII_circ = PhysicalCircuit(
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

    partII_defaults = {
        "circuit": measII_circ,
        "inplace": True,  # TODO: Not hardcode this
    }

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

    # We can now define the whole instruction
    operations["Adaptive Measure Part II"] = Instruction(
        partII_apply_fn,
        partII_input_spec,
        output_spec,
        partII_defaults,
        map_qubits_fn,
        name="Part II of adaptive logical measurement",
        fault_tolerant=True,
    )

    ### PART I
    # Finally we have the first circuit, with no outcome dependencies
    partI_input_spec = [
        ("label", "model"),  # We will take a model from the QuantumProgram
        ("label", "stack"),  # We will take a stack from the QuantumProgram
        (
            "label",
            "patch_label",
        ),  # We will take the patch_label from the QuantumProgram to feed forward
        ("default", "circuit"),  # We will store the circuit in defaults
        ("default", "inplace"),  # and whether to propagate inplace
        ("history", "state", -1),  # We need the previous state
    ]

    # Here is the actual circuit we will run
    measI_circ = PhysicalCircuit(
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

    partI_defaults = {
        "circuit": measI_circ,
        "inplace": True,  # TODO: Not hardcode this
    }

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

    # We can now define the whole instruction
    operations["Adaptive Measure Part I"] = Instruction(
        partI_apply_fn,
        partI_input_spec,
        output_spec,
        partI_defaults,
        map_qubits_fn,
        name="Part I of adaptive logical measurement",
        fault_tolerant=True,
    )

    # For convenience, let's also add an entry for the whole operation
    # For now, this is just part I
    # TODO: Once we have stabilizer frame, this should be composite
    # with part I and then a final operation to do decoding
    operations["Adaptive Measure"] = operations["Adaptive Measure Part I"]
