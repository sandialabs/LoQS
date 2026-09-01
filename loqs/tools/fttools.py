#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""A collection of functions to help fault-tolerance testing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.core import QuantumProgram
from loqs.core.executors import SubmitExecutor
from loqs.core.historydatacollector import (
    HistoryDataCollector,
    HistoryDataCollectorLike,
)
from loqs.core.instructions import Instruction, InstructionLabel
from loqs.tools.paralleltools import ParallelStrategy
from loqs.tools.multiprogramrunner import MultiProgramRunner


def build_discrete_error_injection_program_for_combo(
    base_program: QuantumProgram,
    stack_idx_to_modify: int,
    error_injections: Sequence[tuple[int, str, int]],
) -> QuantumProgram:
    """Build a single discrete-error-injected program for one explicit
    combination of `(layer, error_circuit_label, qubit)` injections.

    A lower-level building block factored out of the per-combo body of
    [](api:build_discrete_error_injection_programs), for callers that
    have already decided exactly which fault combination(s) to build
    (e.g. one representative per Pauli-propagation equivalence class)
    rather than the full location x label enumeration.

    Parameters
    ----------
    base_program : QuantumProgram
        The base program to modify.

    stack_idx_to_modify : int
        The entry in the [](api:InstructionStack) of `base_program` to
        modify with `error_injections` as a label kwarg.

    error_injections : Sequence[tuple[int, str, int]]
        One entry for a weight-1 error, two for a correlated weight-2
        error: `(layer, error_circuit_label, qubit)`.

    Returns
    -------
    QuantumProgram
        The program with this exact fault combination injected.
    """
    instruction_label = base_program.instruction_stack[stack_idx_to_modify]
    assert isinstance(instruction_label, InstructionLabel)

    new_label = deepcopy(instruction_label)
    new_label["error_injections"] = list(error_injections)

    new_stack = base_program.instruction_stack.delete_instruction(
        stack_idx_to_modify
    )
    new_stack = new_stack.insert_instruction(stack_idx_to_modify, new_label)

    tag = "/".join(f"{lbl} on qubit {q}" for _, lbl, q in error_injections)
    layer = error_injections[0][0] if error_injections else "?"
    new_name = (
        f"{base_program.name} + injected error {tag} at layer {layer} "
        f"of stack location {stack_idx_to_modify}"
    )

    return QuantumProgram.from_quantum_program(
        base_program, instruction_stack=new_stack, name=new_name
    )


# ---------------------------------------------------------------------------
# Pauli-propagation equivalence-class pruning.
#
# For a circuit built entirely from Clifford gates (no mid-circuit
# measurement in the segment being swept), propagating an injected Pauli
# fault forward to the end of that segment is exact, deterministic
# algebra: two fault locations that propagate to the identical final
# Pauli support are guaranteed to have identical downstream
# detection/correction behavior (same effect on every instruction after
# this one in the stack), so only one representative per class needs to
# actually be simulated. This uses STIM's own tested tableau machinery
# (`stim.PauliString.after`) rather than hand-derived propagation rules,
# and is skipped entirely (falling back to the untouched exhaustive
# combo list) if `stim` isn't importable, since it's the thing doing the
# propagation math.
#
# Using this relaxes a sweep from "test every circuit location" to "test
# every distinct propagated error" -- a narrower guarantee than a raw
# exhaustive sweep, so it's opt-in (see `build_pruned_discrete_error_
# injection_programs`), not part of the default
# `build_discrete_error_injection_programs` behavior below.
# ---------------------------------------------------------------------------

PAULI_PROPAGATION_GATE_MAP: dict[str, str] = {"Gh": "H", "Gcnot": "CX"}
"""LoQS gate name -> STIM gate name, for gates this propagation utility
knows how to conjugate a Pauli through. Extend as needed for other
Clifford gates; anything not here or in
[](api:PAULI_PROPAGATION_IDLE_GATES) raises rather than silently
mis-pruning."""

PAULI_PROPAGATION_IDLE_GATES: frozenset[str] = frozenset(
    {"Imrz", "Gi1Q", "Gi2Q", "GiMCM"}
)
"""LoQS idle-gate names, which never change a Pauli's support/type and
so are skipped during propagation."""

_PAULI_LABEL_TO_CHAR: dict[str, str] = {
    "Gxpi": "X",
    "Gypi": "Y",
    "Gzpi": "Z",
}


def is_stim_pauli_propagation_available() -> bool:
    """Whether the STIM-tableau-based propagation utility can run."""
    try:
        import stim  # noqa: F401
    except ImportError:
        return False
    return True


def propagate_pauli_signature(
    circuit: BasePhysicalCircuit,
    start_layer: int,
    seed: dict[int, str],
) -> tuple[tuple[int, str], ...]:
    """Propagate a seed Pauli fault (`{qubit_idx: 'X'/'Y'/'Z'}`, applied
    starting at `start_layer`, inclusive) forward through `circuit`'s
    remaining gates using STIM's tableau machinery, and return a
    hashable, sign-independent signature of the resulting Pauli support
    at the circuit's end (empty tuple if it propagates away entirely).

    Only supports `PyGSTiPhysicalCircuit`-backed circuits built entirely
    from gates in [](api:PAULI_PROPAGATION_GATE_MAP)/
    [](api:PAULI_PROPAGATION_IDLE_GATES) (this needs to be verified by
    the caller, e.g. no mid-circuit measurement in the segment being
    propagated through); raises on any other gate so an unrecognized
    one fails loudly rather than silently mis-pruning.
    """
    import stim

    qubit_labels = circuit.qubit_labels
    p = stim.PauliString(len(qubit_labels))
    for qidx, pauli in seed.items():
        p[qidx] = pauli
    for lidx in range(start_layer, circuit.depth):
        for comp in circuit._circuit._layer_components(lidx):
            name = comp.name
            if name in PAULI_PROPAGATION_IDLE_GATES:
                continue
            stim_name = PAULI_PROPAGATION_GATE_MAP.get(name)
            if stim_name is None:
                raise ValueError(
                    f"No Pauli-propagation rule for gate {name!r}; "
                    "extend PAULI_PROPAGATION_GATE_MAP or disable pruning."
                )
            idxs = [qubit_labels.index(q) for q in comp.qubits]
            p = p.after(stim.CircuitInstruction(stim_name, idxs))
    return tuple(
        (i, "IXYZ"[p[i]]) for i in range(len(qubit_labels)) if p[i] != 0
    )


def prune_error_combos_by_propagation(
    circuit: BasePhysicalCircuit,
    error_labels: Sequence[str],
    post_twoq_gates: bool = False,
) -> tuple[list[list[tuple[int, str, int]]], int]:
    """(representative combos, total combos before pruning).

    Each representative combo is in `error_injections` format (one
    `(layer, label, qubit)` entry for weight-1, two for weight-2/
    post-2-qubit-gate errors), one per distinct propagated-Pauli
    equivalence class, per [](api:propagate_pauli_signature). If STIM
    isn't available, returns every combo unpruned (see module comment
    above).
    """
    locations = circuit.get_possible_discrete_error_locations(
        post_twoq_gates=post_twoq_gates
    )
    all_combos: list[list[tuple[int, str, int]]] = []
    for layer, target in locations:
        if post_twoq_gates:
            q1, q2 = target
            for lbl1 in error_labels:
                for lbl2 in error_labels:
                    all_combos.append([(layer, lbl1, q1), (layer, lbl2, q2)])
        else:
            for lbl in error_labels:
                all_combos.append([(layer, lbl, target)])

    if not is_stim_pauli_propagation_available():
        return all_combos, len(all_combos)

    seen_signatures: set[tuple] = set()
    representatives: list[list[tuple[int, str, int]]] = []
    for combo in all_combos:
        seed = {qubit: _PAULI_LABEL_TO_CHAR[lbl] for _, lbl, qubit in combo}
        layer = combo[0][0]
        signature = propagate_pauli_signature(circuit, layer, seed)
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            representatives.append(combo)
    return representatives, len(all_combos)


def build_pruned_discrete_error_injection_programs(
    base_program: QuantumProgram,
    instruction_to_analyze: Instruction,
    stack_idx_to_modify: int,
    error_circuit_labels: Sequence[str],
    post_twoq_gates: bool = False,
) -> tuple[list[QuantumProgram], int]:
    """(programs, total combos before pruning) -- like
    [](api:build_discrete_error_injection_programs), but builds only one
    program per Pauli-propagation equivalence-class representative (see
    [](api:prune_error_combos_by_propagation)) instead of every
    location x label combination. Relaxes "test every circuit location"
    to "test every distinct propagated error"; use only where that's an
    acceptable substitution for the exhaustive sweep.
    """
    circuit = instruction_to_analyze.data["circuit"]
    assert isinstance(circuit, BasePhysicalCircuit)
    representatives, total = prune_error_combos_by_propagation(
        circuit, error_circuit_labels, post_twoq_gates
    )
    programs = [
        build_discrete_error_injection_program_for_combo(
            base_program, stack_idx_to_modify, combo
        )
        for combo in representatives
    ]
    return programs, total


def build_discrete_error_injection_programs(
    base_program: QuantumProgram,
    instruction_to_analyze: Instruction,
    stack_idx_to_modify: int,
    error_circuit_labels: Sequence[str],
    post_twoq_gates: bool = False,
) -> list[QuantumProgram]:
    """Create a series of programs with one discrete error injected each.

    This will take a [](api:Instruction),
    use [](api:BasePhysicalCircuit.get_possible_discrete_error_locations)
    to collect the possible error locations, and then create new programs
    where the error will be injected via `error_injections` (see
    `build_physical_circuit_instruction` for more) as a kwarg
    to the relevant [](api:InstructionLabel).

    Parameters
    ----------
    base_program : QuantumProgram
        The base program to modify

    instruction_to_analyze : Instruction
        The [](api:Instruction) to get all possible discrete errors for

    stack_idx_to_modify : int
        The entry in the [](api:InstructionStack) of the `base_program`
        to modify with `error_injections` as a label kwarg.

    error_circuit_labels : Sequence[str]
        The labels for possible errors to insert.

    post_twoq_gates : bool, optional
        Whether to inject weight-1 errors before every gate (`False`, default)
        or all weight-2 errors after 2Q gates (`True`). Also see
        [](api:BasePhysicalCircuit.get_possible_discrete_error_locations), by default False

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

    # Build instruction label that we will modify. Always already an
    # InstructionLabel: InstructionStack.__getitem__ guarantees it.
    instruction_label = base_program.instruction_stack[stack_idx_to_modify]
    assert isinstance(instruction_label, InstructionLabel)

    # TODO: Split these out so we can inject one error at will at a higher level of API
    def insert_2q_error(layer, eclabel1, eclabel2, qubit1, qubit2):
        new_label = deepcopy(instruction_label)

        # Inject a weight-2 error
        new_label["error_injections"] = [
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
        new_label["error_injections"] = [(layer, eclabel, qubit)]

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


def _fault_checkpoint_subdir(
    shot_checkpoint_dir: str | Path, index: int
) -> Path:
    """Isolated subdirectory for one program's shots, keyed by index."""
    return Path(shot_checkpoint_dir) / f"fault_{index}"


def _run_one_program(
    program: QuantumProgram,
    index: int,
    *,
    shot_executor: SubmitExecutor | None,
    n_shot_batches: int | None,
    collect_shot_data_args: Sequence[HistoryDataCollectorLike],
    expected_outcomes: Sequence,
    num_shots: int,
    shot_checkpoint_dir: str | Path | None,
    checkpoint_batch_size: int | None,
    lazy_loading: bool,
    keep_shot_results: bool = False,
) -> bool | tuple[bool, Any]:
    """Run one program via test_program_output, returning success flag.

    When `keep_shot_results=False` (default), returns the bare bool.
    When `keep_shot_results=True`, returns (success, program_results).

    Matches _run_one_circuit's shape: (item, index, *, shot_executor,
    **static_kwargs) -> result_bool. Each program gets its own isolated
    shot-checkpoint subdirectory keyed by index.
    """
    item_shot_checkpoint_dir = None
    if shot_checkpoint_dir is not None:
        item_shot_checkpoint_dir = _fault_checkpoint_subdir(
            shot_checkpoint_dir, index
        )

    return test_program_output(
        program,
        collect_shot_data_args,
        expected_outcomes,
        num_shots=num_shots,
        shot_executor=shot_executor,
        n_shot_batches=n_shot_batches,
        checkpoint_batch_size=checkpoint_batch_size,
        checkpoint_dir=item_shot_checkpoint_dir,
        lazy_loading=lazy_loading,
        return_program_results=keep_shot_results,
    )


class FaultInjectionRunner(MultiProgramRunner):
    """Runner for testing discrete-error-injected programs with checkpoint/resume.

    Encapsulates all configuration needed to test error-injected programs,
    including parallel/checkpoint settings, in a serializable object that can
    be recovered after a crash via `FaultInjectionRunner.read(runner_path).run()`.
    Checkpoint/resume behavior is controlled by explicit `checkpoint`/`resume`
    flags applied against on-disk state -- see `MultiProgramRunner.run`.
    """

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "errored_programs",
        "collect_shot_data_args",
        "expected_outcomes",
        "num_shots",
    ]

    def __init__(
        self,
        errored_programs: Sequence[QuantumProgram],
        collect_shot_data_args: Sequence[HistoryDataCollectorLike],
        expected_outcomes: Sequence,
        num_shots: int = 1,
        parallel_strategy: ParallelStrategy | None = None,
        item_checkpoint_dir: str | Path | None = None,
        checkpoint: bool = False,
        resume: bool = False,
        force_resume: bool = False,
        checkpoint_batch_size: int | None = None,
        shot_checkpoint_dir: str | Path | None = None,
        lazy_loading: bool = True,
        keep_shot_results: bool = False,
        poll_interval: float = 1.0,
        show_progress: bool = True,
    ):
        super().__init__(
            parallel_strategy=parallel_strategy,
            item_checkpoint_dir=item_checkpoint_dir,
            checkpoint=checkpoint,
            resume=resume,
            force_resume=force_resume,
            checkpoint_batch_size=checkpoint_batch_size,
            shot_checkpoint_dir=shot_checkpoint_dir,
            lazy_loading=lazy_loading,
            keep_shot_results=keep_shot_results,
            poll_interval=poll_interval,
            show_progress=show_progress,
        )
        self.errored_programs = errored_programs
        self.collect_shot_data_args = collect_shot_data_args
        self.expected_outcomes = expected_outcomes
        self.num_shots = num_shots

    def _get_items(self) -> Sequence:
        """Return programs to test."""
        return self.errored_programs

    def _item_key_fn(self) -> Callable[[QuantumProgram], str] | None:
        """No stable identity beyond position."""
        return None

    def _process_item_fn(self) -> Callable:
        """Return the _run_one_program function."""
        return _run_one_program

    def _static_kwargs(self) -> dict[str, Any]:
        """Return static kwargs for _run_one_program."""
        return {
            "collect_shot_data_args": self.collect_shot_data_args,
            "expected_outcomes": self.expected_outcomes,
            "num_shots": self.num_shots,
            "shot_checkpoint_dir": self.shot_checkpoint_dir,
            "checkpoint_batch_size": self.checkpoint_batch_size,
            "lazy_loading": self.lazy_loading,
        }

    def _make_on_item_done(
        self,
    ) -> Callable[[int, QuantumProgram, bool], None]:
        """Return a closure that merges each program's success bool into the
        shared `_reduced_results` accumulator."""

        def on_item_done(
            index: int, program: QuantumProgram, success: bool
        ) -> None:
            self._merge_reduced_result(index, success)

        return on_item_done

    def _finalize(self, result_list: list[bool]) -> list[QuantumProgram]:
        """Return failed programs (same objects from input, matched by index).

        Builds the failed list from `_reduced_results`, which holds success
        bools indexed by program position, in original program order.
        """
        failed = [
            prog
            for i, prog in enumerate(self.errored_programs)
            if not self._reduced_results.get(i, True)
        ]

        if len(failed):
            print(f"Failed {len(failed)} programs!")
        else:
            print("All programs succeeded!")

        return failed

    def _desc(self) -> str:
        """Return description for progress bar."""
        return "Running discrete error injected programs"

    def _mismatch_check_fields(self) -> list[str]:
        """Return fields to check for resume mismatch."""
        return [
            "num_shots",
            "collect_shot_data_args",
            "expected_outcomes",
        ]

    def _shot_checkpoint_subdir(self, index: int) -> Path | None:
        """Return the checkpoint directory for a specific program's shots.

        Returns a per-program subdirectory under shot_checkpoint_dir keyed by
        the program index, or None if shot-level checkpointing is not enabled.
        """
        if (
            self.shot_checkpoint_dir is not None
            and self.checkpoint_batch_size is not None
        ):
            return _fault_checkpoint_subdir(self.shot_checkpoint_dir, index)
        return None


def test_program_output(
    test_program: QuantumProgram,
    collect_shot_data_args: Sequence[HistoryDataCollectorLike],
    expected_outcomes: Sequence,
    num_shots: int = 1,
    verbose: bool = False,
    shot_executor: SubmitExecutor | None = None,
    n_shot_batches: int | None = None,
    checkpoint_batch_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    lazy_loading: bool = True,
    return_program_results: bool = False,
) -> bool | tuple[bool, Any]:
    """Test a program against expected output.

    Parameters
    ----------
    test_program : QuantumProgram
        The [](api:QuantumProgram) to test

    collect_shot_data_args : Sequence[HistoryDataCollectorLike]
        A list of arguments to [](api:ProgramResults.collect_shot_data).

    expected_outcomes : Sequence
        A list of the expected results to the
        [](api:ProgramResults.collect_shot_data) calls.

    num_shots : int, optional
        The number of shots to run and test, by default 1

    verbose : bool, optional
        Whether to print the failed entry, if one occurs.
        Will only print the first failed entry, if more than
        one fails, by default False

    shot_executor : SubmitExecutor | None, optional
        Forwarded to [](api:QuantumProgram.run) for shot-level
        parallelism. Defaults to `None`, which runs shots serially.

    n_shot_batches : int | None, optional
        Forwarded to [](api:QuantumProgram.run) for shot-level
        parallel dispatch batching. Defaults to `None`.

    checkpoint_batch_size : int | None, optional
        Forwarded to [](api:QuantumProgram.run) for shot checkpointing.
        Defaults to `None`, which disables batch checkpointing.

    checkpoint_dir : str | Path | None, optional
        Forwarded to [](api:QuantumProgram.run) for shot checkpointing.
        Defaults to `None`, which disables checkpointing.

    lazy_loading : bool, optional
        Forwarded to [](api:QuantumProgram.run) for lazy loading.
        Defaults to `True`.

    return_program_results : bool, optional
        If `True`, return `(success, program_results)` instead of the bare bool.
        Defaults to `False`.

    Returns
    -------
    bool | tuple[bool, ProgramResults]
        When `return_program_results=False` (default): `True` if all outputs
        match expected, `False` on failure.
        When `return_program_results=True`: `(success, program_results)` tuple
        with the same success boolean and the full `ProgramResults` object.
    """
    run_kwargs = {
        "num_shots": num_shots,
        "shot_executor": shot_executor,
        "n_shot_batches": n_shot_batches,
        "verbose": False,
        "lazy_loading": lazy_loading,
    }
    if checkpoint_batch_size is not None:
        run_kwargs["checkpoint"] = True
        run_kwargs["checkpoint_batch_size"] = checkpoint_batch_size
    if checkpoint_dir is not None:
        run_kwargs["checkpoint_dir"] = checkpoint_dir
        # Cascade resume: only True if this specific item has prior shot state
        if checkpoint_batch_size is not None:
            run_kwargs["resume"] = (
                Path(checkpoint_dir) / "results.h5"
            ).exists()

    program_results = test_program.run(**run_kwargs)

    success = True
    for args, expected in zip(collect_shot_data_args, expected_outcomes):
        # Collect shot data for last shot
        outs = HistoryDataCollector.from_raw(args).collect(program_results)
        for out in outs[-num_shots:]:
            if out != expected:
                if verbose:
                    print(f"Output:   {out}")
                    print(f"Expected: {expected}")
                success = False
                break
        if not success:
            break

    if return_program_results:
        return success, program_results
    return success
