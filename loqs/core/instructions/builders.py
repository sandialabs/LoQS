"""Functions to construct common :class:`.Instruction` objects.

Each function documents both how to use it, as well
as providing the following information about the created
:class:`.Instruction`:

- The apply function
    - The parameters it pulls, including the typical source
    - What keys are in the returned :class:`.Frame`
- The map qubits function (if needed)
- The parameter priorities (if not default)
- The parameter aliases (if provided)

If information is omitted, it is implied to be the default.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import inspect as ins
import numpy as np
import typing

from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.state import BaseQuantumState, STIMQuantumState
from loqs.core.frame import Frame
from loqs.core.history import History
from loqs.core.instructions import Instruction
from loqs.core.instructions.instruction import DEFAULT_PRIORITIES, KwargDict
from loqs.core.instructions.instructionlabel import InstructionLabel
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.qeccode import QECCode, QECCodePatch
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict
from loqs.core.syndrome import (
    PauliFrame,
    SyndromeLabel,
    SyndromeLabelCastableTypes,
)


def build_composite_instruction(
    instructions: Sequence[Instruction],
    name: str = "(Unnamed composite instruction)",
) -> Instruction:
    """Build a composite instruction that updates the stack.

    The composite instruction holds a sequence of instructions
    that will be inserted into the stack. It is primarily intended
    for convenience and organization purposes, i.e. being able
    to define instructions in small modular pieces but still
    providing a high-level sequence of instructions via the
    composite instruction.

    The apply function takes:

    - ``patch_label``, usually from the :attr:`.InstructionLabel.patch_label`
    - `stack`, usually from the :class:`.QuantumProgram`
    - `instructions`, usually from the :attr:`.Instruction.data`

    It returns a :class:`.Frame` where ``instructions`` have been inserted
    onto the front of :class:`.InstructionStack` stored at ``"stack"``.

    There is a map qubits function, which calls the map qubits
    functions for the underlying `instructions`.

    Parameters
    ----------
    instructions:
        A list of instructions to be inserted onto the stack

    name:
        Name for logging purposes

    Returns
    -------
        The built composite instruction
    """

    def apply_fn(
        patch_label: str | None,
        stack: InstructionStack,
        instructions: Sequence[Instruction],
    ) -> Frame:
        for i, instruction in enumerate(instructions):
            new_label = InstructionLabel(instruction, patch_label)
            stack = stack.insert_instruction(i, new_label)

        return Frame({"stack": stack})

    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        instructions: Sequence[Instruction],
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["instructions"] = [
            instruction.map_qubits(qubit_mapping)
            for instruction in instructions
        ]
        return new_kwargs

    # We will need to store the instructions and param priorities
    data = {"instructions": instructions}

    return Instruction(
        apply_fn=apply_fn,
        data=data,
        map_qubits_fn=map_qubits_fn,
        name=name,
        type="Composite",
    )


def build_lookup_decoder_instruction(
    lookup_table: Mapping[str, str],
    syndrome_labels: Sequence[SyndromeLabelCastableTypes],
    raw_syndrome_frame_key: str,
    diff_prev_syndrome: bool = True,
    name: str = "(Unnamed lookup decoder)",
) -> Instruction:
    """Build a lookup table decoder instruction.

    The lookup table decoder instruction takes a lookup table
    with syndrome keys and correction values and a sequence
    of `SyndromeLabel` objects so that it can pull the syndrome
    from previous measurement outcomes. It then applies
    a Pauli frame update to the relevant patch.

    The apply function takes:

    - `patch_label`, usually from the `InstructionLabel`
    - `lookup_table`, usually from the `Instruction.data`
    - `syndrome_labels`, usually from the `Instruction.data`
    - `raw_syndrome_frame_key`, usually from the `Instruction.data`
    - `diff_prev_syndrome`, usually from the `Instruction.data`
    - `patches`, usually from the previous frame
    - `syndrome_outcomes`, an alias for `measurement_outcomes`
      from the previous frames
    - `history`, usually from the `QuantumProgram`

    It returns a `Frame` with a `PatchDict` that has an updated `PauliFrame`,
    the raw syndrome, and some other debugging information.

    There is a map qubits function, which maps the `syndrome_labels`.

    There is a non-standard parameter priority for `syndrome_labels`,
    as this can require more than just the previous single frame.

    Parameters
    ----------
    lookup_table:
        A dict with syndrome keys (as strings of "0" and "1") and
        data corrections (as Pauli strings with entries "IXYZ" and
        length = number of data qubits).

    syndrome_labels:
        List of `SyndromeLabels` (or tuples to be cast as `SydromeLabels`)
        that describe which entries in the previous measurement outcomes
        correspond to which syndrome bit

    raw_syndrome_frame_key:
        The key to use in the output Frame for the raw syndrome information,
        as well as the key used when searching for previous syndrome information.

    diff_prev_syndrome:
        Whether to do a bitwise XOR to the previous syndrome (True, default)
        or to use this round's syndrome information directly (False).

    name:
        Name for logging purposes

    Returns
    -------
        The built lookup table decoder instruction
    """

    # Sanity check: Have specified a syndrome label for each element of lookup_table
    assert all(
        [len(k) == len(syndrome_labels) for k in lookup_table]
    ), "Lookup table syndromes must match number of syndrome labels"

    # Standard apply_fn construction
    def apply_fn(
        patch_label: str,
        lookup_table: dict[str, str],
        syndrome_labels: list[SyndromeLabel],
        raw_syndrome_frame_key: str,
        diff_prev_syndrome: bool,
        patches: PatchDict,
        syndrome_outcomes: list[MeasurementOutcomes] | MeasurementOutcomes,
        history: History,
    ) -> Frame:
        if isinstance(syndrome_outcomes, MeasurementOutcomes):
            syndrome_outcomes = [syndrome_outcomes]

        # Look up PauliFrame in patch
        patch = patches[patch_label]

        # Extract our syndrome measurements based on the qubit labels
        syndrome: list[int] = []
        for synlbl in syndrome_labels:
            frame_outcomes = syndrome_outcomes[synlbl.frame_idx]
            outcome = frame_outcomes[synlbl.qubit_label][synlbl.outcome_idx]
            syndrome.append(outcome)

        # We need to diff against the last measured syndrome to see if anything has flipped
        # First, find the last time we measured a syndrome with this tag
        # TODO: This will not distinguish tags between patches. We have the history,
        # so can figure it out by comparing patch_labels and patch objects
        # but punting on this for now. Need to fix before production 2Q runs
        prev_syndrome = None
        prev_frame_info = None
        if diff_prev_syndrome:
            for i, frame in enumerate(history[::-1]):
                prev_syndrome = typing.cast(
                    list[int] | None, frame.get(raw_syndrome_frame_key, None)
                )
                if prev_syndrome is None:
                    continue
                assert isinstance(prev_syndrome, list)
                assert all([isinstance(i, int) for i in prev_syndrome])

                # If we got one, record logging info and break
                prev_frame_info = (frame.log, -i - 1)
                break

        # If we have a previous syndrome, XOR it to get the diff
        if prev_syndrome is None:
            syndrome_diff = syndrome
        else:
            syndrome_diff = [
                int(b) for b in np.bitwise_xor(syndrome, prev_syndrome)
            ]

        # Look up data error based on changed syndromes
        syndrome_str = "".join([str(s) for s in syndrome_diff])
        data_error_str = lookup_table[syndrome_str]

        # Update pauli frame
        new_pauli_frame = patch.pauli_frame.update_from_pauli_str(
            data_error_str
        )

        # Update patches
        new_patches = patches.copy()
        new_patches[patch_label] = QECCodePatch(
            patch.code, patch.qubits, new_pauli_frame
        )

        frame = Frame(
            {
                "patches": new_patches,
                raw_syndrome_frame_key: syndrome,
                "decoded_error": data_error_str,
                "prev_pauli_frame": "".join(patch.pauli_frame.pauli_frame),
                "new_pauli_frame": "".join(new_pauli_frame.pauli_frame),
            }
        )

        if diff_prev_syndrome:
            frame = frame.update(
                {
                    "prev_syndrome_frame": prev_frame_info,
                    "syndrome_diff": syndrome_diff,
                }
            )

        return frame

    # We store lookup table and syndrome labels
    cast_labels = [SyndromeLabel.cast(sl) for sl in syndrome_labels]
    data = {
        "lookup_table": dict(lookup_table),
        "syndrome_labels": cast_labels,
        "raw_syndrome_frame_key": raw_syndrome_frame_key,
        "diff_prev_syndrome": diff_prev_syndrome,
    }

    # We need to be able to map the qubit_labels
    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        syndrome_labels: list[SyndromeLabel],
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["syndrome_labels"] = [
            SyndromeLabel(
                qubit_mapping[sl.qubit_label], sl.frame_idx, sl.outcome_idx
            )
            for sl in syndrome_labels
        ]
        return new_kwargs

    # We also need param priorities and aliases
    # For priorities, we need as many measurement outcomes as requested by syndrome labels
    frame_idxes = [sl.frame_idx for sl in cast_labels]
    param_priorities = {"syndrome_outcomes": [f"history[{min(frame_idxes)}:]"]}

    # For aliases, we just have syndrome -> measurement
    param_aliases = {"syndrome_outcomes": "measurement_outcomes"}

    return Instruction(
        apply_fn=apply_fn,
        data=data,
        map_qubits_fn=map_qubits_fn,
        param_priorities=param_priorities,
        param_aliases=param_aliases,
        name=name,
        type="Lookup Decoder",
    )


def build_object_builder_instruction(
    frame_key: str,
    obj_class: type,
    name: str = "(Unnamed object builder)",
) -> Instruction:
    """Build an instruction that can initialize LoQS objects.

    This is a sort of meta-instruction that can build LoQS
    objects and then store them into a `Frame`. This is currently
    used primarily to initialize the `BaseQuantumState` if no
    `initial_history` is provided to a `QuantumProgram`.
    The constructor arguments should typically be provided
    in the `InstructionLabel` as args or kwargs.

    The apply function takes variadic kwargs, since we do not
    know the constructor arguments until runtime. However, it takes
    at least the following:

    - `frame_key`, taken from `Instruction.data`
    - `obj_class`, taken from `Instruction.data`

    It also must take all required args to the `obj_class` constructor, and
    returns a `Frame` with the constructed object stored under `frame_key`.

    The parameter priorities are generated programatically using
    the `inspect` module for function signature introspection.

    Parameters
    ----------
    frame_key:
        The key used to store the resulting object in the `Frame`

    obj_class:
        The LoQS object to construct

    name:
        Name for logging purposes

    Returns
    -------
        The built object builder instruction
    """

    # This is also an odd apply_fn because we do not know the args a priori
    # Here we define the generic apply_fn using variadic kwargs
    def apply_fn(**kwargs) -> Frame:
        frame_key = kwargs.pop("frame_key")
        obj_class = kwargs.pop("obj_class")
        try:
            obj = obj_class(**kwargs)
        except Exception as e:
            raise ValueError(
                "Failed to create object in object builder"
            ) from e
        return Frame({frame_key: obj})

    data = {"frame_key": frame_key, "obj_class": obj_class}

    # Because we don't know the signature, let's set the priorities ourselves
    # Let's grab all the args from the constructor, excluding self
    param_priorities = {}
    sig = ins.signature(obj_class)
    for param in list(sig.parameters):
        param_priorities[param] = DEFAULT_PRIORITIES
    # And add on our instruction data information
    for k in data:
        param_priorities[k] = ["instruction"]

    return Instruction(
        apply_fn=apply_fn,
        data=data,
        param_priorities=param_priorities,
        param_error_behavior="continue",  # Suppress variadic kwargs warning
        name=name,
        type="Object builder",
    )


def build_patch_builder_instruction(
    qec_code: QECCode,
    name: str = "(Unnamed patch builder)",
) -> Instruction:
    """Build an instruction that can make patches from a `QECCode`.

    This is a sort of meta-instruction that can build `QECCodePatch`
    objects and then store them into the main `PatchDict`.
    The qubit labels for the new patch should typically be provided
    in the `InstrumentLabel` as args or kwargs.

    The apply function takes:

    - `patch_label`, usually taken from `InstructionLabel`
    - `qubits`, usually taken from `InstructionLabel`
    - `qec_code`, usually taken from `Instruction.data`
    - `patches`, usually taken from the previous frame,
      but can be taken from `Instruction.data` as a default fallback

    It returns a `Frame` with an updated `patches` containing the new
    `QECCodePatch` under `patch_label`.

    The parameter priorities for `patches` are not default,
    because we want to prioritize a true `PatchDict` from
    the `History` over the default one provided in the
    `Instruction.data`.

    Parameters
    ----------
    qec_code:
        The `QECCode` to use when constructing new patches.

    name:
        Name for logging purposes

    Returns
    -------
        The built patch builder instruction
    """

    # Standard apply_fn construction
    def apply_fn(
        patch_label: str,
        qubits: Sequence[str],
        qec_code: QECCode,
        patches: PatchDict | None,
    ) -> Frame:
        if patches is None:
            patches = PatchDict()

        all_patch_qubits = patches.all_qubit_labels

        # Disjoint patch checks
        assert all(
            [q not in all_patch_qubits for q in qubits]
        ), f"Patch builder failed, requesting overlapping patches for {patch_label}"
        assert (
            patch_label not in patches
        ), f"Patch builder failed, already have existing patch {patch_label}"

        try:
            patch = qec_code.create_patch(qubits)
        except Exception as e:
            raise ValueError("Failed to create patch in patch builder") from e

        patches[patch_label] = patch

        return Frame({"patches": patches})

    data = {"qec_code": qec_code, "patches": None}

    # We are providing a default patches dict here in the instruction
    # However, we would like it to be loaded from History first if possible
    param_priorities = {"patches": ["history[-1]", "instruction"]}

    return Instruction(
        apply_fn=apply_fn,
        data=data,
        param_priorities=param_priorities,
        name=name,
        type="Patch Builder",
    )


def build_patch_remover_instruction(
    name: str = "(Unnamed patch builder)",
) -> Instruction:
    """Build an instruction that can make delete patches.

    This is a sort of meta-instruction that can remove patches
    from the main `PatchDict`. The patch label should typically
    be provided in the `InstrumentLabel` as args or kwargs.

    The apply function takes:

    - `patch_label`, usually taken from `InstructionLabel`
    - `patches`, usually taken from the previous frame

    It returns a `Frame` with an updated `patches` without the
    `QECCodePatch` under `patch_label`.

    Parameters
    ----------
    name:
        Name for logging purposes

    Returns
    -------
        The built patch remover instruction
    """

    def apply_fn(
        patch_label: str,
        patches: PatchDict,
    ) -> Frame:
        assert (
            patch_label in patches
        ), f"Patch remover failed, could not find patch {patch_label}"

        del patches[patch_label]

        return Frame({"patches": patches})

    return Instruction(apply_fn=apply_fn, name=name, type="Patch Remover")


def build_patch_permute_instruction(
    mapping: Mapping[str | int, str | int],
    name: str = "(Unnamed patch permutation)",
) -> Instruction:
    """Build an instruction that permute patch qubits.

    This can permute the qubits in a patch, i.e. doing SWAP
    operations but in software.

    The apply function takes:

    - `patch_label`, usually taken from `InstructionLabel`
    - `mapping`, usually taken from `Instruction.data`
    - `patches`, usually taken from the previous frame

    It returns a `Frame` with an updated `patches` where the
    `QECCodePatch` stored under `patch_label` has had the qubits
    mapped via `mapping`.

    There is a map qubits function that updates both the keys
    and values of `mapping`.

    Parameters
    ----------
    mapping:
        The qubit mapping to apply to the code patch

    name:
        Name for logging purposes

    Returns
    -------
        The built patch permuter instruction
    """

    # Standard apply_fn construction
    def apply_fn(
        patch_label: str,
        mapping: Mapping[str | int, str | int],
        patches: PatchDict,
    ) -> Frame:
        assert (
            patch_label in patches
        ), f"Patch permute failed, could not find patch {patch_label}"

        patch = patches[patch_label]

        code = patch.code
        qubits = patch.qubits

        # get(q, q) ensures non-specified qubits are unchanged
        mapped_qubits = [mapping.get(q, q) for q in qubits]

        permuted_patch = code.create_patch(mapped_qubits)

        patches[patch_label] = permuted_patch

        return Frame({"patches": patches})

    # We store the mapping
    data = {"mapping": mapping}

    # In this case, we do need to be able to update the mapping if qubits change
    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        mapping: Mapping[str | int, str | int],
    ) -> KwargDict:
        new_mapping = {
            qubit_mapping[k]: qubit_mapping[v] for k, v in mapping.items()
        }
        return {"mapping": new_mapping}

    return Instruction(
        apply_fn=apply_fn,
        data=data,
        map_qubits_fn=map_qubits_fn,
        name=name,
        type="Patch Permuter",
    )


def build_physical_circuit_instruction(
    circuit: BasePhysicalCircuit,
    inplace: bool = True,
    model: BaseNoiseModel | None = None,
    pauli_frame_update: (
        str | Sequence[str] | Mapping[str | int, str | int] | None
    ) = None,
    name: str = "(Unnamed physical circuit)",
) -> Instruction:
    """Build an instruction that applies a physical circuit to a state.

    This is the computational workhorse for LoQS. This stores a circuit
    that can propagate the quantum state forward in time.
    However, it can do some other common pre-/post-processing tasks.
    It can insert discrete errors into the circuit prior to application,
    and it can update the Pauli frame if the circuit represents the action
    of a Clifford gate.

    The apply function takes:

    - `model`, usually from the `QuantumProgram.default_noise_model`,
      but also from `InstructionLabel` and `Instruction.data`
    - `circuit`, usually from the `Instruction.data`
    - `state`, usually from the previous frame
    - `inplace`, usually from the `Instruction.data`
    - `error_injections`, usually from the `InstructionLabel`
    - `pauli_frame_update`, usually from the `Instruction.data`
    - `patch_label`, usually from the `InstructionLabel`
    - `patches`, usually from the previous frame

    It returns a `Frame` with an updated `state`, observed
    `measurement_outcomes` if requested, an updated `patches` if a
    `PauliFrame` update was requested, and some other debugging information.

    There is a map qubits function calls `circuit.map_qubit_labels()`.

    Parameters
    ----------
    circuit:
        The physical circuit to run

    inplace:
        Whether to propagate the state in-place (True, default) or make a copy

    model:
        A model to use when converting the circuit into reps to apply to the state

    pauli_frame_update: str | Sequence[str] | Mapping[str|int, str|int] | None = None,
        Either a string that is passed to `PauliFrame.update_from_transversal_clifford()`,
        a list of strings that is passed to `PauliFrame.update_from_clifford_conjugation()`,
        a mapping that is passed to `PauliFrame.map_frame()`, or None (the default) if no
        Pauli frame update is required.

    name:
        Name for logging purposes

    Returns
    -------
        The built physical circuit instruction
    """

    # Standard apply_fn construction
    def apply_fn(
        model: BaseNoiseModel,
        circuit: BasePhysicalCircuit,
        state: BaseQuantumState,
        inplace: bool,
        error_injections: list[tuple[int, str, int]] | None,
        pauli_frame_update: str | list[str] | dict[str, str] | None,
        patch_label: str,
        patches: PatchDict,
    ) -> Frame:

        # Modify circuit for injected errors
        qubits = circuit.qubit_labels
        errored_circuit = circuit.copy()

        # Inject errors from the back
        if error_injections is None:
            error_injections = []
        rev_sorted_errors = sorted(
            error_injections, key=lambda x: x[0], reverse=True
        )
        for error in rev_sorted_errors:
            circuit_backend = type(circuit)
            error_circuit = circuit_backend(
                [(error[1], qubits[error[2]])], qubit_labels=qubits
            )
            errored_circuit.insert_inplace(error_circuit, error[0])

        new_state, outcomes = propagate_state(
            errored_circuit, model, state, inplace
        )

        data: dict[str, object] = {"state": new_state}

        # Update pauli frame, if needed
        if pauli_frame_update is not None:
            patch = patches[patch_label]

            if isinstance(pauli_frame_update, dict):
                new_pauli_frame = patch.pauli_frame.map_frame(
                    pauli_frame_update
                )
            elif isinstance(pauli_frame_update, str):
                new_pauli_frame = (
                    patch.pauli_frame.update_from_transversal_clifford(
                        pauli_frame_update
                    )
                )
            elif isinstance(pauli_frame_update, Sequence):
                new_pauli_frame = (
                    patch.pauli_frame.update_from_clifford_conjugation(
                        pauli_frame_update
                    )
                )
            else:
                raise ValueError("Invalid pauli frame mapping")

            # Update patches
            new_patches = patches.copy()
            new_patches[patch_label] = QECCodePatch(
                patch.code, patch.qubits, new_pauli_frame
            )

            data["patches"] = new_patches

        if len(outcomes):
            data["measurement_outcomes"] = MeasurementOutcomes(outcomes)
        if len(error_injections):
            data["errored_circuit"] = errored_circuit
        # TODO: Make this more general, maybe models have a "save_to_frame_attrs" or somethign
        if isinstance(state, STIMQuantumState):
            data["applied_stim_circuit"] = state.latest_applied_circuit

        return Frame(data)

    # We store circuit and flags as defaults
    data = {
        "circuit": circuit,
        "inplace": inplace,
        "error_injections": None,  # Must be specified in label only
        "pauli_frame_update": pauli_frame_update,
    }
    if model is not None:
        data["model"] = model

    # We need to be able to map the circuit if qubits change
    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        circuit: BasePhysicalCircuit,
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["circuit"] = circuit.map_qubit_labels(qubit_mapping)
        return new_kwargs

    return Instruction(
        apply_fn=apply_fn,
        data=data,
        map_qubits_fn=map_qubits_fn,
        name=name,
        type="Physical circuit",
    )


def build_repeat_until_success_instruction(
    instruction: Instruction,
    reset_label_key: Instruction | str,
    rus_key: str,
    target_outcomes: MeasurementOutcomes | None = None,
    max_repeats: int = 100,
    name: str = "(Unnamed repeat-until-success instruction)",
) -> Instruction:
    """Build an instruction that repeats a physical circuit instruction until success.

    This is a special case feed-forward type of instruction.
    It first applies the underlying instruction (presumed to be a physical
    circuit instruction, although this is not actively checked),
    and then compares the measurement outcomes to a set of target outcomes.
    If the outcomes do not match, this places itself on top of the stack.

    The apply function takes variadic kwargs since we do not
    know the underlying instruction arguments until runtime. However, it takes
    at least the following:

    - `instruction`, usually from the `Instruction.data`
    - `target_outcomes`, usually from the `Instruction.data`
    - `max_repeats`, usually from the `Instruction.data`
    - `repeat_count`, usually from the `Instruction.data`
    - `rus_key`, usually from the `Instruction.data`
    - `reset_key`, usually from the `Instruction.data`
    - `patch_label`, usually from the `InstructionLabel`
    - `patches`, usually from the previous frame
    - `stack`, usually from the `QuantumProgram`

    It returns a `Frame` with the outcome of the underlying
    `instruction`, as well as an updated stack with a reset and
    this operation again in the case of failure, and some debugging information.

    There is a map qubits function that maps the qubits of
    `instruction` and `reset_key`, in the case that `reset_key`
    is an `Instruction`.

    Parameters
    ----------
    instruction:
        The underlying instruction to run and repeat until success

    reset_label_key:
        The key for the `InstructionLabel` needed to reset the
        quantum state in the case of failure

    rus_key:
        The key for the `InstructionLabel` needed to run this
        operation again in the case of failure.
        We are not simply providing `self` to avoid loops
        during operations such as serialization.

    target_outcomes:
        The target outcomes to compare to. If None (the default),
        this assumes that all measurement outcomes should be 0.

    max_repeats:
        The number of repeats after which an error is thrown.
        Defaults to 100.

    name:
        Name for logging purposes

    Returns
    -------
        The built repeat-until-success instruction
    """

    # We do not know all the params for the underlying instruction,
    # so take variadic kwargs here
    def apply_fn(**kwargs) -> Frame:
        # Pull some args out of kwargs
        instruction = kwargs.pop("instruction")
        assert isinstance(instruction, Instruction)
        target_outcomes = kwargs.pop("target_outcomes")
        if target_outcomes is not None:
            target_outcomes = MeasurementOutcomes.cast(target_outcomes)
        max_repeats = int(kwargs.pop("max_repeats"))
        repeat_count = int(kwargs.pop("repeat_count"))
        rus_key = kwargs.pop("rus_key")
        reset_key = kwargs.pop("reset_key")

        if "patch_label" in instruction.param_priorities:
            patch_label = kwargs["patch_label"]
        else:
            patch_label = kwargs.pop("patch_label")
        if "patches" in instruction.param_priorities:
            patches = kwargs["patches"]
        else:
            patches = kwargs.pop("patches")
        if "stack" in instruction.param_priorities:
            stack = InstructionStack.cast(kwargs["stack"])
        else:
            stack = InstructionStack.cast(kwargs.pop("stack"))

        # All the remaining kwargs should go straight into the instruction
        applied_frame = instruction.apply(**kwargs)

        # Run success function to see if we are terminated
        try:
            patch = patches[patch_label]

            pauli_frame = PauliFrame.cast(
                applied_frame.get("pauli_frame", patch.qubits)
            )
            outcomes = MeasurementOutcomes.cast(
                applied_frame["measurement_outcomes"]
            )
            inferred_outcomes = outcomes.get_inferred_outcomes(
                pauli_frame, "Z"
            )
            if target_outcomes is None:
                # If target outcomes not provided, use all 0s
                success = True
                for _, v in outcomes.items():
                    if any([bit for bit in v]):
                        success = False
                        break
            else:
                success = target_outcomes == inferred_outcomes
        except KeyError:
            raise RuntimeError(
                "Try-until-success instruction does not output outcomes"
            )

        if not success:  # We have failed, need to add this back again
            # Check if we have hit limit
            repeat_count += 1
            if repeat_count >= max_repeats:
                raise RuntimeError(
                    "Try-until-success instruction hit max repeats"
                )

            # Otherwise, we need to reset and redo the RUS instruction
            reset_label = InstructionLabel(reset_key, patch_label)

            # We need to at least put repeat_counts into label args
            rus_label = InstructionLabel(
                rus_key,
                patch_label,
                inst_kwargs={"repeat_count": repeat_count},
            )

            stack = stack.insert_instruction(0, rus_label)
            stack = stack.insert_instruction(0, reset_label)

        # Return frame with the stack update
        return applied_frame.update(
            {
                "stack": stack,
                "inferred_outcomes": inferred_outcomes,
                "rus_success": success,
            }
        )

    # We need to store the instruction, target outcomes, and repeat information
    # To avoid recursion, we also store the key for the RUS instruction
    data: dict[str, object] = {
        "instruction": instruction,
        "target_outcomes": target_outcomes,
        "max_repeats": max_repeats,
        "repeat_count": 0,
        "rus_key": rus_key,
        "reset_key": reset_label_key,
    }
    # We also need to pull out any data from the instruction
    for k, v in instruction.data.items():
        assert (
            k not in data
        ), "Have key collision between RUS and underlying instructions"
        data[k] = v

    # We need to map the instruction
    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        instruction: Instruction,
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["instruction"] = instruction.map_qubits(qubit_mapping)
        # Reset any data we pulled from instruction
        for k, v in new_kwargs["instruction"].data.items():
            new_kwargs[k] = v
        if isinstance(new_kwargs["reset_key"], Instruction):
            new_kwargs["reset_key"] = new_kwargs["reset_key"].map_qubits(
                qubit_mapping
            )
        return new_kwargs

    # Since we have variadic kwargs, we'll set the param priority ourselves
    # Default order is OK for new params. We'll pick up most of what we need
    # from instruction, but repeat count will be from label arg after first iteration
    param_priorities = instruction.param_priorities.copy()
    for k in data:
        param_priorities[k] = DEFAULT_PRIORITIES
    # We also need the patch_label for new labels, and the stack to update it
    param_priorities["patch_label"] = param_priorities.get(
        "patch_label", DEFAULT_PRIORITIES
    )
    param_priorities["patches"] = param_priorities.get(
        "patches", DEFAULT_PRIORITIES
    )
    param_priorities["stack"] = param_priorities.get(
        "stack", DEFAULT_PRIORITIES
    )

    return Instruction(
        apply_fn=apply_fn,
        data=data,
        map_qubits_fn=map_qubits_fn,
        param_priorities=param_priorities,
        param_error_behavior="continue",  # Skip warning for variadic kwargs
        name=name,
        type="Repeat-until-success",
    )
