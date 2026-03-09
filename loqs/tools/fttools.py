"""A collection of functions to help fault-tolerance testing.
"""

from typing import Any
from collections.abc import Sequence
from copy import deepcopy
from tqdm import tqdm

try:
    from dask.distributed import Client, progress
except ImportError:
    Client = Any  # type: ignore

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.core import QuantumProgram
from loqs.core.history import HistoryCollectDataArgsType
from loqs.core.instructions import Instruction, InstructionLabel
from loqs.tools.dasktools import run_program_list


def build_discrete_error_injection_programs(
    base_program: QuantumProgram,
    instruction_to_analyze: Instruction,
    stack_idx_to_modify: int,
    error_circuit_labels: Sequence[str],
    post_twoq_gates: bool = False,
) -> list[QuantumProgram]:
    """Create a series of programs with one discrete error injected each.

    This will take a (presumably physical circuit) :class:`.Instruction`,
    use :meth:`.BasePhysicalCircuit.get_possible_discrete_error_locations`
    to collect the possible error locations, and then create new programs
    where the error will be injected via ``error_injections`` (see
    :meth:`.build_physical_circuit_instruction` for more) as a kwarg
    to the relevant :class:`.InstructionLabel`.

    Parameters
    ----------
    base_program:
        The base program to modify

    instruction_to_analyze:
        The :class:`.Instruction` to get all possible discrete errors for

    stack_idx_to_modify:
        The entry in the :class:`.InstructionStack` of the ``base_program``
        to modify with ``error_injections`` as a label kwarg.

    error_circuit_labels:
        The labels for possible errors to insert.

    post_twoq_gates:
        Whether to inject weight-1 errors before every gate (``False``, default)
        or all weight-2 errors after 2Q gates (``True``). Also see
        :meth:`.BasePhysicalCircuit.get_possible_discrete_error_locations`.

    Returns
    -------
    list[QuantumProgram]
        A list of programs, one for each possible discrete error
    """
    # Get possible circuit locations
    try:
        circuit = instruction_to_analyze.data["circuit"]
    except KeyError:
        raise ValueError(
            "Key 'circuit' not available in instruction_to_analyze.data"
        )
    assert isinstance(circuit, BasePhysicalCircuit)

    error_locations = circuit.get_possible_discrete_error_locations(
        post_twoq_gates=post_twoq_gates
    )

    # Build instruction label that we will modify
    instruction_label = InstructionLabel.cast(
        base_program.instruction_stack[stack_idx_to_modify]
    )

    # TODO: Split these out so we can inject one error at will at a higher level of API
    def insert_2q_error(layer, eclabel1, eclabel2, qubit1, qubit2):
        new_label = deepcopy(instruction_label)

        # Inject a weight-2 error
        new_label.inst_kwargs["error_injections"] = [
            (layer, eclabel1, qubit1),
            (layer, eclabel2, qubit2),
        ]

        new_stack = base_program.instruction_stack.delete_instruction(
            stack_idx_to_modify
        )
        new_stack = new_stack.insert_instruction(
            stack_idx_to_modify, new_label
        )

        # Name with weight-2 error
        new_name = f"{base_program.name} + injected error {eclabel1}/{eclabel2} on qubit indices {(qubit1, qubit2)} after component {layer} of stack location {stack_idx_to_modify}"

        new_program = QuantumProgram.from_quantum_program(
            base_program,
            instruction_stack=new_stack,
            name=new_name,
        )

        errored_programs.append(new_program)

    def insert_1q_error(layer, eclabel, qubit, end=False):
        new_label = deepcopy(instruction_label)

        # Inject a weight-1 error
        new_label.inst_kwargs["error_injections"] = [(layer, eclabel, qubit)]

        new_stack = base_program.instruction_stack.delete_instruction(
            stack_idx_to_modify
        )
        new_stack = new_stack.insert_instruction(
            stack_idx_to_modify, new_label
        )

        # Name with weight-1 error
        if end:
            new_name = f"{base_program.name} + injected error {eclabel} on qubit index {qubit} at end of stack location {stack_idx_to_modify}"
        else:
            new_name = f"{base_program.name} + injected error {eclabel} on qubit index {qubit} before component {layer} of stack location {stack_idx_to_modify}"

        new_program = QuantumProgram.from_quantum_program(
            base_program, instruction_stack=new_stack, name=new_name
        )

        errored_programs.append(new_program)

    # Iterate over all errors during the circuit, i.e. before every gate
    errored_programs: list[QuantumProgram] = []
    for error_loc in error_locations:
        for eclabel in error_circuit_labels:
            if post_twoq_gates:
                assert (
                    isinstance(error_loc[1], tuple) and len(error_loc[1]) == 2
                )

                # We have two qubit gate errors, we need an extra loop to create weight-2 errors
                for eclabel2 in error_circuit_labels:
                    insert_2q_error(
                        error_loc[0],
                        eclabel,
                        eclabel2,
                        error_loc[1][0],
                        error_loc[1][1],
                    )
            else:
                assert isinstance(error_loc[1], int)

                # We only have single qubit errors, create the new program at this loop level
                insert_1q_error(error_loc[0], eclabel, error_loc[1])

    # Also add every error at the end of the circuit in the case of single qubit errors
    # Two qubit errors don't need this because they are already post errors

    # TODO: Don't do this, instead insert before readout as well
    # if not post_twoq_gates:
    #     for eclabel in error_circuit_labels:
    #         for i in range(len(circuit.qubit_labels)):
    #             insert_1q_error(circuit.depth, eclabel, i, end=True)

    return errored_programs


def run_discrete_error_injected_programs(
    errored_programs: Sequence[QuantumProgram],
    collect_shot_data_args: Sequence[HistoryCollectDataArgsType],
    expected_outcomes: Sequence,
    num_shots: int = 1,
    dask_client: Client | None = None,  # type: ignore
) -> list[QuantumProgram]:
    """Call :meth:`.test_program_output` on many programs.

    Parameters
    ----------
    errored_programs:
        A list of programs to test, usually the output of
        :meth:`.build_discrete_error_injection_programs`.

    collect_shot_data_args:
        See :meth:`.test_program_output`.

    expected_outcomes:
        See :meth:`.test_program_output`.

    num_shots:
        See :meth:`.test_program_output`.

    dask_client:
        A Dask client to use for parallelizing over programs
        (as this is likely a better strategy than parallelizing
        over small number of shots per program).
        Defaults to ``None``, which runs shots in serial.

    Returns
    -------
    list[QuantumProgram]
        The failed programs
    """
    failed = []

    if dask_client is None:
        tasks = [
            (p, collect_shot_data_args, expected_outcomes, num_shots)
            for p in errored_programs
        ]
        for task in tqdm(tasks, "Running discrete error injected programs"):
            success = test_program_output(*task)
            if not success:
                failed.append(task[0])
    else:
        print("Running discrete error injected programs in parallel with Dask")
        run_program_list(errored_programs, dask_client, num_shots)

        for program in errored_programs:
            success = test_program_output(
                program,
                collect_shot_data_args,
                expected_outcomes,
                num_shots,
                skip_run=True,
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
    collect_shot_data_args: Sequence[HistoryCollectDataArgsType],
    expected_outcomes: Sequence,
    num_shots: int = 1,
    verbose: bool = False,
    skip_run: bool = False,
) -> bool:
    """Test a program against expected output.

    Parameters
    ----------
    test_program:
        The :class:`.QuantumProgram` to test

    collect_shot_data_args:
        A list of arguments to :meth:`.History.collect_shot_data`.

    expected_outcomes:
        A list of the expected results to the
        :meth:`.History.collect_shot_data` calls.

    num_shots:
        The number of shots to run and test.

    verbose:
        Whether to print the failed entry, if one occurs.
        Will only print the first failed entry, if more than
        one fails.

    Returns
    -------
    bool
        ``True`` if all outputs match expected, ``False`` on failure
    """
    if not skip_run:
        test_program.run(num_shots=num_shots, verbose=False)

    for args, expected in zip(collect_shot_data_args, expected_outcomes):
        # Collect shot data for last shot
        outs = test_program.collect_shot_data(*args)
        for out in outs[-num_shots:]:
            if out != expected:
                if verbose:
                    print(f"Output:   {out}")
                    print(f"Expected: {expected}")
                return False
    return True
