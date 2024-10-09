"""TODO"""

from collections.abc import Sequence
from copy import deepcopy

from tqdm import tqdm
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.core import QuantumProgram
from loqs.core.instructions import Instruction, InstructionLabel


def build_discrete_error_injection_programs(
    base_program: QuantumProgram,
    instruction_to_analyze: Instruction,
    stack_idx_to_modify: int,
    error_circuit_labels: Sequence[str],
) -> list[QuantumProgram]:
    """TODO"""
    # Get possible circuit locations
    try:
        circuit = instruction_to_analyze.data["circuit"]
    except KeyError:
        raise ValueError(
            "Key 'circuit' not available in instruction_to_analyze.data"
        )
    assert isinstance(circuit, BasePhysicalCircuit)

    error_locations = circuit.get_possible_discrete_error_locations()

    # Build instruction label that we will modify
    instruction_label = InstructionLabel.cast(
        base_program.instruction_stack[stack_idx_to_modify]
    )

    # Iterate over all errors during the circuit, i.e. before every gate
    errored_programs = []
    for error_loc in error_locations:
        for eclabel in error_circuit_labels:
            instruction_label = InstructionLabel.cast(
                base_program.instruction_stack[stack_idx_to_modify]
            )
            new_label = deepcopy(instruction_label)

            # Assign error injections to the instrument label
            new_label.inst_kwargs["error_injections"] = [
                (error_loc[0], eclabel, error_loc[1])
            ]

            new_stack = base_program.instruction_stack.delete_instruction(
                stack_idx_to_modify
            )
            new_stack = new_stack.insert_instruction(
                stack_idx_to_modify, new_label
            )

            new_name = f"{base_program.name} + injected error {eclabel} before component {error_loc[0]}"

            new_program = QuantumProgram.from_quantum_program(
                base_program, instruction_stack=new_stack, name=new_name
            )

            errored_programs.append(new_program)

    # Also add every error at the end of the circuit
    for i in range(len(circuit.qubit_labels)):
        for eclabel in error_circuit_labels:
            instruction_label = InstructionLabel.cast(
                base_program.instruction_stack[stack_idx_to_modify]
            )
            new_label = deepcopy(instruction_label)

            # Assign error injections to the instrument label
            new_label.inst_kwargs["error_injections"] = [
                (circuit.depth, eclabel, i)
            ]

            new_stack = base_program.instruction_stack.delete_instruction(
                stack_idx_to_modify
            )
            new_stack = new_stack.insert_instruction(
                stack_idx_to_modify, new_label
            )

            new_name = f"{base_program.name} + injected error {eclabel} at circuit end"

            new_program = QuantumProgram.from_quantum_program(
                base_program, instruction_stack=new_stack, name=new_name
            )

            errored_programs.append(new_program)

    return errored_programs


def run_discrete_error_injected_programs(
    errored_programs: Sequence[QuantumProgram],
    collect_shot_data_args: Sequence[Sequence],
    expected_outcomes: Sequence,
    num_shots: int = 1,
) -> list[QuantumProgram]:
    """TODO"""
    failed = []
    for program in tqdm(
        errored_programs, "Running discrete error injected programs"
    ):
        success = test_program_output(
            program, collect_shot_data_args, expected_outcomes, num_shots
        )
        if not success:
            failed.append(program)

    if len(failed):
        print(f"Failed {len(failed)} programs!")
    else:
        print("All programs succeeded!")

    return failed


def test_program_output(
    test_program: QuantumProgram,
    collect_shot_data_args: Sequence[Sequence],
    expected_outcomes: Sequence,
    num_shots: int = 1,
) -> bool:
    """TODO"""
    test_program.run(shots=num_shots, verbose=False)
    for args, expected in zip(collect_shot_data_args, expected_outcomes):
        # Collect shot data for last shot
        outs = test_program.collect_shot_data(*args)
        for out in outs[-num_shots:]:
            if out != expected:
                return False
    return True
