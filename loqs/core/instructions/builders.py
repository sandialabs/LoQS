#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Functions to construct common [Instruction](api:Instruction) objects.

Each function documents both how to use it, as well
as providing the following information about the created
[Instruction](api:Instruction):

- The apply function
    - The parameters it pulls, including the typical source
    - What keys are in the returned [Frame](api:Frame)
- The map qubits function (if needed)
- The parameter priorities (if not default)
- The parameter aliases (if provided)

If information is omitted, it is implied to be the default.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import inspect as ins
import numpy as np
from typing import TYPE_CHECKING, Any

from loqs.backends import is_backend_available, propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import (
    BaseNoiseModel,
    TimeDependentBaseNoiseModel,
)
from loqs.backends.state import BaseQuantumState
from loqs.core.frame import Frame
from loqs.core.history import History
from loqs.core.instructions import Instruction
from loqs.core.instructions.instruction import DEFAULT_PRIORITIES, KwargDict
from loqs.core.instructions.instructionlabel import (
    InstructionLabel,
    InstructionLabelCastableTypes,
)
from loqs.core.instructions.instructionstack import (
    InstructionStack,
    InstructionStackCastableTypes,
)
from loqs.core.qeccode import QECCode
from loqs.core.recordables import (
    MeasurementOutcomes,
    PatchDict,
    PauliFrame,
    QECCodePatch,
)
from loqs.core.syndromelabel import (
    SyndromeLabel,
    SyndromeLabelCastableTypes,
)

# Conditional imports for PyGSTi
if TYPE_CHECKING:
    # Type checking imports - these won't be executed at runtime
    from loqs.backends import (
        STIMQuantumState,
        STIMPhysicalCircuit,
        PyGSTiNoiseModel,
    )
else:
    # Runtime imports - these will be attempted only when needed
    try:
        from loqs.backends import PyGSTiNoiseModel
    except ImportError:
        PyGSTiNoiseModel = Any  # type: ignore

    try:
        from loqs.backends import STIMQuantumState, STIMPhysicalCircuit
    except ImportError:
        STIMQuantumState = Any  # type: ignore
        STIMPhysicalCircuit = Any  # type: ignore


def build_composite_instruction(
    instructions: Sequence[InstructionLabelCastableTypes],
    extra_data: KwargDict | None = None,
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

    - ``patch_label``, usually from the [patch_label](api:InstructionLabel.patch_label)
    - `stack`, usually from the QuantumProgram
    - `instructions`, usually from the [data](api:Instruction.data)

    It returns a [Frame](api:Frame) where ``instructions`` have been inserted
    onto the front of InstructionStack stored at ``"stack"``.

    There is a map qubits function, which calls the map qubits
    functions for the underlying `instructions`.

    Parameters
    ----------
    instructions:
        A list of instructions or instruction labels to be inserted
        onto the stack

    name:
        Name for logging purposes

    Returns
    -------
        The built composite instruction
    """

    def apply_fn(
        patch_label: str | None,
        stack: InstructionStack,
        instructions: Sequence[Instruction | InstructionLabel],
        **kwargs,
    ) -> Frame:
        """Apply function for composite instruction.

        Inserts instructions into the instruction stack and returns updated frame.

        Parameters
        ----------
        patch_label : str | None
            Patch label for the instruction.
        stack : InstructionStack
            Current instruction stack.
        instructions : Sequence[Instruction | InstructionLabel]
            Instructions to insert into the stack.
        **kwargs
            Additional keyword arguments for the instructions.

        Returns
        -------
        Frame
            Updated frame with modified instruction stack.

        REVIEW_NO_DOCSTRING
        """
        for i, inst_or_label in enumerate(instructions):
            if isinstance(inst_or_label, Instruction):
                new_label = InstructionLabel(
                    inst_or_label, patch_label, inst_kwargs=kwargs
                )
            else:
                inst_or_label = InstructionLabel.cast(inst_or_label)
                new_kwargs = kwargs.copy()
                new_kwargs.update(inst_or_label.inst_kwargs)
                first_entry = (
                    inst_or_label.instruction
                    if inst_or_label.instruction is not None
                    else inst_or_label.inst_label
                )
                assert first_entry is not None
                new_label = InstructionLabel(
                    first_entry,
                    inst_or_label.patch_label,
                    inst_or_label.inst_args,
                    new_kwargs,
                )
            stack = stack.insert_instruction(i, new_label)

        return Frame({"stack": stack})

    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        instructions: Sequence[Instruction | InstructionLabel],
        **kwargs,
    ) -> KwargDict:
        """Map qubits function for composite instruction.

        Maps qubits in the instruction sequence according to the provided mapping.

        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        instructions : Sequence[Instruction | InstructionLabel]
            Instructions to map qubits for.
        **kwargs
            Additional keyword arguments to preserve.

        Returns
        -------
        KwargDict
            Dictionary containing updated instructions with mapped qubits.

        REVIEW_NO_DOCSTRING
        """
        new_kwargs = kwargs.copy()
        new_kwargs["instructions"] = [
            (
                instruction.map_qubits(qubit_mapping)
                if isinstance(instruction, Instruction)
                else instruction
            )
            for instruction in instructions
        ]
        return new_kwargs

    if extra_data is None:
        extra_data = {}
    data = extra_data.copy()
    data["instructions"] = instructions

    # Make sure all extra data gets pulled in
    param_priorities = {k: DEFAULT_PRIORITIES for k in data.keys()}

    # We will need to store the instructions
    return Instruction(
        apply_fn=apply_fn,
        data=data,
        map_qubits_fn=map_qubits_fn,
        param_priorities=param_priorities,
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
        """Apply lookup table decoder instruction.

        Parameters
        ----------
        patch_label : str
            Label of the patch to apply corrections to.
        lookup_table : dict[str, str]
            Mapping from syndrome strings to correction Pauli strings.
        syndrome_labels : list[SyndromeLabel]
            List of syndrome labels describing measurement outcomes.
        raw_syndrome_frame_key : str
            Key for storing raw syndrome information in the output frame.
        diff_prev_syndrome : bool
            Whether to XOR with previous syndrome (True) or use current syndrome directly (False).
        patches : PatchDict
            Dictionary of patches containing Pauli frames.
        syndrome_outcomes : list[MeasurementOutcomes] | MeasurementOutcomes
            Measurement outcomes from previous frames.
        history : History
            History of previous frames for reference.

        Returns
        -------
        Frame
            Updated frame with corrected Pauli frame and syndrome information.
        """
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
                prev_syndrome = frame.get(raw_syndrome_frame_key, None)
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
        """Map qubits function for lookup decoder instruction.

        Updates syndrome labels to reflect new qubit mapping.

        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        syndrome_labels : list[SyndromeLabel]
            List of syndrome labels to be updated.
        **kwargs : dict
            Additional keyword arguments to preserve.

        Returns
        -------
        KwargDict
            Dictionary containing updated syndrome labels and preserved kwargs.
        """
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
        """Apply object builder instruction.

        This function takes variadic kwargs since the constructor arguments
        are not known until runtime. It constructs an object of the specified
        class and stores it in a frame.

        Parameters
        ----------
        **kwargs : dict
            Must contain 'frame_key' (str) and 'obj_class' (type), plus any
            additional arguments required by the obj_class constructor.

        Returns
        -------
        Frame
            Frame containing the updated patches dictionary.
        """
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
        """Apply patch builder instruction.

        Creates a new patch from the QEC code and adds it to the patches dictionary.

        Parameters
        ----------
        patch_label : str
            Label for the new patch.
        qubits : Sequence[str]
            List of qubit labels for the new patch.
        qec_code : QECCode
            Quantum error correction code to use for creating the patch.
        patches : PatchDict | None
            Existing patches dictionary, or None to create a new one.

        Returns
        -------
        Frame
            Frame containing the updated patches dictionary.

        REVIEW_NO_DOCSTRING
        """
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
        """Apply patch remover instruction.

        Removes a patch from the patches dictionary.

        Parameters
        ----------
        patch_label : str
            Label of the patch to remove.
        patches : PatchDict
            Dictionary of patches to remove from.

        Returns
        -------
        Frame
            Frame containing the updated patches dictionary.

        REVIEW_NO_DOCSTRING
        """
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
        """Apply patch permute instruction.

        Permutes the qubits in a patch according to the provided mapping.

        Parameters
        ----------
        patch_label : str
            Label of the patch to permute.
        mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        patches : PatchDict
            Dictionary of patches containing the patch to permute.

        Returns
        -------
        Frame
            Frame containing the updated patches dictionary with permuted patch.
        """
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
        """Map qubits function for patch permute instruction.

        Updates the qubit mapping to reflect the new qubit labels.

        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        mapping : Mapping[str | int, str | int]
            Original mapping to be updated.

        Returns
        -------
        KwargDict
            Dictionary containing the updated mapping.
        """
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
        """Apply physical circuit instruction.

        Executes a physical circuit on the quantum state and updates the Pauli frame.

        Parameters
        ----------
        model : BaseNoiseModel
            Noise model to use for circuit execution.
        circuit : BasePhysicalCircuit
            Physical circuit to execute.
        state : BaseQuantumState
            Quantum state to operate on.
        inplace : bool
            Whether to modify the state in-place.
        error_injections : list[tuple[int, str, int]] | None
            List of error injections to apply to the circuit.
        pauli_frame_update : str | list[str] | dict[str, str] | None
            Pauli frame update to apply after circuit execution.
        patch_label : str
            Label of the patch to update.
        patches : PatchDict
            Dictionary of patches containing the patch to update.

        Returns
        -------
        Frame
            Frame containing the updated state and patches.
        """

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
            if isinstance(circuit, STIMPhysicalCircuit):
                error_circuit = circuit_backend(
                    f"{error[1]} {error[2]}\nTICK\n", qubit_labels=qubits
                )
            else:
                error_circuit = circuit_backend(
                    [(error[1], qubits[error[2]])], qubit_labels=qubits  # type: ignore
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
                if pauli_frame_update.lower() == "reset":
                    # Reset back to trivial frame
                    new_pauli_frame = PauliFrame(patch.qubits)
                else:
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

        if isinstance(model, TimeDependentBaseNoiseModel):
            data["current_model_time"] = model.current_time
        if len(outcomes):
            data["measurement_outcomes"] = MeasurementOutcomes(outcomes)
        if len(error_injections):
            data["errored_circuit"] = errored_circuit
        # TODO: Make this more general, maybe models have a "save_to_frame_attrs" or somethign
        if not is_backend_available("stim_state") and isinstance(
            state, STIMQuantumState
        ):
            data["applied_stim_circuit_str"] = str(
                state.latest_applied_circuit
            )

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
        """Map qubits function for physical circuit instruction.

        Updates the circuit to reflect new qubit mapping.

        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        circuit : BasePhysicalCircuit
            Circuit to be updated with new qubit labels.
        **kwargs : dict
            Additional keyword arguments to preserve.

        Returns
        -------
        KwargDict
            Dictionary containing updated circuit and preserved kwargs.
        """
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
    instructions: InstructionStackCastableTypes,
    rus_key: str,
    test_frame_key: str = "rus_success",
    expected: object = True,
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

    - `observed`, usually from the previous frame
    - `expected`, usually from the `Instruction.data`
    - `rus_key`, usually from the `Instruction.data`
    - `patch_label`, usually from the `InstructionLabel`
    - `repeat_count`, usually (except first time) from the `InstructionLabel`
    - `instructions`, usually from the `Instruction.data`
    - `max_repeats`, usually from the `Instruction.data`
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
        The underlying instruction to run and repeat until success.
        This should include reset to 0 as the first instruction,
        and the final frame should contain a `success_frame_key`
        as True if successful and False if not.

    success_frame_key:
        The key to check in the final frame for success.

    max_repeats:
        The number of repeats after which an error is thrown.
        Defaults to 100.

    name:
        Name for logging purposes

    Returns
    -------
        The built repeat-until-success instruction
    """

    def apply_fn(
        observed: object,
        expected: object,
        rus_key: str,
        patch_label: str,
        repeat_count: int,
        instructions: InstructionStackCastableTypes,
        max_repeats: int,
        stack: InstructionStack,
    ) -> Frame:
        """Apply repeat-until-success instruction.

        Repeats the underlying instruction until it succeeds or max_repeats is reached.

        Parameters
        ----------
        observed : object
            Observed outcome from the instruction execution.
        expected : object
            Expected outcome for successful execution.
        rus_key : str
            Key for the repeat-until-success instruction.
        patch_label : str
            Label of the patch being operated on.
        repeat_count : int
            Current repeat count.
        instructions : InstructionStackCastableTypes
            Instructions to execute.
        max_repeats : int
            Maximum number of repeats before giving up.
        stack : InstructionStack
            Current instruction stack.

        Returns
        -------
        Frame
            Frame containing success status and repeat count information.
        """
        # If we were successful, return empty frame (with debug info)
        # TODO: If these are measurement_outcomes, how do we get inferred_outcomes from Pauli frame?
        if observed == expected:
            return Frame(
                {"total_rus_count": repeat_count, "rus_success": True}
            )

        repeat_count += 1
        if repeat_count > max_repeats:
            raise RuntimeError(
                "Hit max repeats in repeat-until-success instruction"
            )

        # TODO: InstructionStack cast misbehaved, track that down
        # new_labels = InstructionStack.cast(instructions)._instructions
        new_labels = [InstructionLabel.cast(ilbl) for ilbl in instructions]

        # Create a new RUS label with updated count
        rus_label = InstructionLabel(
            rus_key, patch_label, None, {"repeat_count": repeat_count}
        )
        new_labels.append(rus_label)

        # Update stack
        stack = stack.insert_instructions(0, new_labels)

        # Return frame with the stack update
        return Frame({"stack": stack, "rus_success": False})

    # We need to store the instruction, target outcomes, and repeat information
    # To avoid recursion, we also store the key for the RUS instruction
    data: dict[str, object] = {
        "instructions": instructions,
        test_frame_key: (
            None if expected is not None else False
        ),  # Will always fail first round so we do prep
        "expected": expected,
        "max_repeats": max_repeats,
        "repeat_count": 0,
        "rus_key": rus_key,
    }

    # Need to map underlying instructions also
    def map_qubits_fn(
        qubit_mapping: Mapping[str | int, str | int],
        instructions: Sequence[Instruction | InstructionLabel],
        **kwargs,
    ) -> KwargDict:
        """Map qubits function for repeat-until-success instruction.

        Maps qubits in the instruction sequence and expected outcomes.

        Parameters
        ----------
        qubit_mapping : Mapping[str | int, str | int]
            Mapping from old qubit labels to new qubit labels.
        instructions : Sequence[Instruction | InstructionLabel]
            Instructions to map qubits for.
        **kwargs
            Additional keyword arguments including expected outcomes.

        Returns
        -------
        KwargDict
            Dictionary containing updated instructions and expected outcomes with mapped qubits.

        REVIEW_NO_DOCSTRING
        """
        new_kwargs = kwargs.copy()
        new_kwargs["instructions"] = [
            (
                instruction.map_qubits(qubit_mapping)
                if isinstance(instruction, Instruction)
                else instruction
            )
            for instruction in instructions
        ]
        # Often expected will be a MeasurementOutcomes, so we want to map that as well
        # TODO: Is it always?
        new_kwargs["expected"] = new_kwargs["expected"].map_qubits(
            qubit_mapping
        )
        return new_kwargs

    # The success argument is being collected from the previous frame by alias
    return Instruction(
        apply_fn=apply_fn,
        data=data,
        map_qubits_fn=map_qubits_fn,
        param_priorities={
            "observed": ["label", "history[-1]", "instruction"]
        },  # prioritize history over instruction data
        param_aliases={"observed": test_frame_key},
        name=name,
        type="Repeat-until-success",
    )
