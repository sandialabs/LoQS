"""TODO
"""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
import inspect as ins
import numpy as np

from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.state import BaseQuantumState
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
    param_priorities: (
        Sequence[str] | Mapping[str, Sequence[str]] | None
    ) = None,
    name: str = "(Unnamed composite instruction)",
) -> Instruction:
    """TODO"""

    # We don't know what args we need, so let's take variadic kwargs
    def apply_fn(**kwargs) -> Frame:
        # These will always be the last two
        assert "stack" in kwargs
        assert "instructions" in kwargs
        stack = InstructionStack.cast(kwargs["stack"])
        instructions: list[Instruction] = list(kwargs["instructions"])

        assert param_priorities is not None

        for i, instruction in enumerate(instructions):
            patch_label = None
            label_kwargs = {}
            for k, v in kwargs.items():
                if k in param_priorities:
                    # We want to forward this via the label
                    if k == "patch_label":
                        patch_label = v
                    elif k in ["stack", "patches", "instructions"]:
                        # We want these from the program or not at all
                        pass
                    else:
                        label_kwargs[k] = v

            new_label = InstructionLabel(
                instruction, patch_label, inst_kwargs=label_kwargs
            )

            stack = stack.insert_instruction(i, new_label)

        return Frame({"stack": stack})

    # We will need to store the instructions
    data = {"instructions": instructions}

    def map_qubits_fn(
        qubit_mapping: Mapping[str, str], instructions: Sequence[Instruction]
    ) -> KwargDict:
        new_kwargs: dict[str, object] = {
            "instructions": [
                instruction.map_qubits(qubit_mapping)
                for instruction in instructions
            ]
        }
        return new_kwargs

    # Because we don't have a fixed function signature above,
    # we need to pass in priorities. For convenience,
    # we allow a Sequence of keys that we will turn into default priorities here
    if param_priorities is None:
        param_priorities = []
    if not isinstance(param_priorities, Mapping):
        param_priorities = {k: DEFAULT_PRIORITIES for k in param_priorities}
    param_priorities = dict(param_priorities)
    # Also get the patch label and pass it forward if needed
    param_priorities["patch_label"] = param_priorities.get(
        "patch_label", DEFAULT_PRIORITIES
    )
    # We also need the stack and instructions
    param_priorities["stack"] = param_priorities.get(
        "stack", DEFAULT_PRIORITIES
    )
    param_priorities["instructions"] = param_priorities.get(
        "instructions", DEFAULT_PRIORITIES
    )

    return Instruction(
        apply_fn=apply_fn,
        data=data,
        map_qubits_fn=map_qubits_fn,
        param_priorities=param_priorities,
        param_error_behavior="continue",  # Suppress the warning for variadic kwargs
        name=name,
    )


def build_lookup_decoder_instruction(
    lookup_table: Mapping[str, str],
    syndrome_labels: Sequence[SyndromeLabelCastableTypes],
    raw_syndrome_frame_key: str,
    name: str = "(Unnamed lookup decoder)",
) -> Instruction:
    """TODO"""
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
        patches: PatchDict,
        syndrome_outcomes: list[MeasurementOutcomes] | MeasurementOutcomes,
        history: History,
    ) -> Frame:
        if isinstance(syndrome_outcomes, MeasurementOutcomes):
            syndrome_outcomes = [syndrome_outcomes]

        # Look up PauliFrame in patch
        patch = patches[patch_label]

        # Extract our syndrome measurements based on the qubit labels
        syndrome = []
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
        for i, frame in enumerate(history[::-1]):
            prev_syndrome = frame.get(raw_syndrome_frame_key, None)
            if prev_syndrome is None:
                continue

            # If we got one, record logging info and break
            prev_frame_info = (frame.log, -i - 1)
            break

        # If we have a previous syndrome, XOR it to get the diff
        if prev_syndrome is None:
            syndrome_diff = syndrome
        else:
            syndrome_diff = [
                int(b) for b in np.logical_xor(syndrome, prev_syndrome)
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

        return Frame(
            {
                "patches": new_patches,
                raw_syndrome_frame_key: syndrome,
                "prev_syndrome_frame": prev_frame_info,
                "syndrome_diff": syndrome_diff,
                "decoded_error": data_error_str,
                "prev_pauli_frame": "".join(patch.pauli_frame.pauli_frame),
                "new_pauli_frame": "".join(new_pauli_frame.pauli_frame),
            }
        )

    # We store lookup table and syndrome labels
    data = {
        "lookup_table": dict(lookup_table),
        "syndrome_labels": [SyndromeLabel.cast(sl) for sl in syndrome_labels],
        "raw_syndrome_frame_key": raw_syndrome_frame_key,
    }

    # We need to be able to map the qubit_labels
    def map_qubits_fn(
        qubit_mapping: Mapping[str, str],
        syndrome_labels: list[tuple[str, int]],
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

    # Get our expected output frame keys for use in dry run
    frame_keys = [
        "patches",
        raw_syndrome_frame_key,
        "syndrome_diff",
        "prev_syndrome_frame",
        "decoded_error",
        "prev_pauli_frame",
        "new_pauli_frame",
    ]

    # We also need param priorities and aliases
    # For priorities, we need as many measurement outcomes as requested by syndrome labels
    frame_idxes = [sl.frame_idx for sl in data["syndrome_labels"]]
    param_priorities = {"syndrome_outcomes": [f"history[{min(frame_idxes)}:]"]}

    # For aliases, we just have syndrome -> measurement
    param_aliases = {"syndrome_outcomes": "measurement_outcomes"}

    return Instruction(
        apply_fn=apply_fn,
        dry_run_apply_fn=frame_keys,  # Skip apply and just return DRY_RUN for these
        data=data,
        map_qubits_fn=map_qubits_fn,
        param_priorities=param_priorities,
        param_aliases=param_aliases,
        name=name,
    )


def build_object_builder_instruction(
    frame_key: str,
    obj_class: type,
    name: str = "(Unnamed object builder)",
) -> Instruction:
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
    sig = ins.signature(obj_class.__init__)
    for param in list(sig.parameters)[1:]:  # Skipping self
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
    )


def build_patch_builder_instruction(
    qec_code: QECCode,
    name: str = "(Unnamed patch builder)",
) -> Instruction:
    # Standard apply_fn construction
    def apply_fn(
        patch_label: str,
        qubits: Sequence[str],
        qec_code: QECCode,
        patches: PatchDict,
    ) -> Frame:
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

    data = {"qec_code": qec_code, "patches": PatchDict()}

    # We are providing a default patches dict here in the instruction
    # However, we would like it to be loaded from History first if possible
    param_priorities = {"patches": ["history[-1]", "instruction"]}

    return Instruction(
        apply_fn=apply_fn,
        data=data,
        param_priorities=param_priorities,
        name=name,
    )


def build_patch_remover_instruction(
    name: str = "(Unnamed patch builder)",
) -> Instruction:
    def apply_fn(
        patch_label: str,
        patches: PatchDict,
    ) -> Frame:
        assert (
            patch_label in patches
        ), f"Patch remover failed, could not find patch {patch_label}"

        del patches[patch_label]

        return Frame({"patches": patches})

    return Instruction(
        apply_fn=apply_fn,
        name=name,
    )


def build_patch_permute_instruction(
    mapping: Mapping[str, str],
    name: str = "(Unnamed patch permutation)",
) -> Instruction:
    """TODO"""

    # Standard apply_fn construction
    def apply_fn(
        patch_label: str,
        mapping: Mapping[str, str],
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
        qubit_mapping: Mapping[str, str],
        mapping: Mapping[str, str],
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
    )


def build_physical_circuit_instruction(
    circuit: BasePhysicalCircuit,
    include_outcomes: bool = False,
    inplace: bool = True,
    reset_mcms: bool = True,
    name: str = "(Unnamed physical circuit)",
) -> Instruction:
    """TODO"""

    # Standard apply_fn construction
    def apply_fn(
        model: BaseNoiseModel,
        circuit: BasePhysicalCircuit,
        state: BaseQuantumState,
        include_outcomes: bool,
        inplace: bool,
        reset_mcms: bool,
    ) -> Frame:

        new_state, outcomes = propagate_state(
            circuit, model, state, inplace, reset_mcms
        )

        data: dict[str, object] = {"state": new_state}
        if include_outcomes:
            data["measurement_outcomes"] = MeasurementOutcomes(outcomes)

        return Frame(data)

    # We store circuit and flags as defaults
    data = {
        "circuit": circuit,
        "include_outcomes": include_outcomes,
        "inplace": inplace,
        "reset_mcms": reset_mcms,
    }

    # We need to be able to map the circuit if qubits change
    def map_qubits_fn(
        qubit_mapping: Mapping[str, str],
        circuit: BasePhysicalCircuit,
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["circuit"] = circuit.map_qubit_labels(qubit_mapping)
        return new_kwargs

    # Get our expected output frame keys for use in dry run
    frame_keys = ["state"]
    if include_outcomes:
        frame_keys.append("measurement_outcomes")

    return Instruction(
        apply_fn=apply_fn,
        dry_run_apply_fn=frame_keys,  # Skip apply and just return DRY_RUN for these
        data=data,
        map_qubits_fn=map_qubits_fn,
        name=name,
    )


def _default_success_fn(outcomes: MeasurementOutcomes) -> bool:
    """Default all-0 success function for repeat-until-success.

    Parameters
    ----------
    outcomes:
        Measurement outcomes from previous frame

    Returns
    -------
        True if all measures bits are 0, False otherwise
    """
    for _, v in outcomes.items():
        if any([bit for bit in v]):
            return False

    return True


def build_repeat_until_success_instruction(
    instruction: Instruction,
    reset_label_key: Instruction | str,
    rus_key: str,
    success_fn: Callable[[MeasurementOutcomes], bool] = _default_success_fn,
    max_repeats: int = 100,
    name: str = "(Unnamed repeat-until-success instruction)",
) -> Instruction:
    """TODO"""

    # We do not know all the params for the underlying instruction,
    # so take variadic kwargs here
    def apply_fn(**kwargs) -> Frame:
        # Pull some args out of kwargs
        instruction = kwargs.pop("instruction")
        assert isinstance(instruction, Instruction)
        success_fn = kwargs.pop("success_fn")
        max_repeats = int(kwargs.pop("max_repeats"))
        repeat_count = int(kwargs.pop("repeat_count"))
        rus_key = kwargs.pop("rus_key")
        reset_key = kwargs.pop("reset_key")

        patch_label = kwargs.pop("patch_label")
        stack = InstructionStack.cast(kwargs.pop("stack"))

        # All the remaining kwargs should go straight into the instruction
        applied_frame = instruction.apply(**kwargs)

        # Run success function to see if we are terminated
        try:
            pauli_frame = applied_frame.get("pauli_frame", None)
            outcomes = MeasurementOutcomes.cast(
                applied_frame["measurement_outcomes"]
            )
            inferred_outcomes = outcomes.get_inferred_outcomes(
                "Z", pauli_frame
            )
            success = success_fn(inferred_outcomes)
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
        return applied_frame.update({"stack": stack, "rus_success": success})

    # We need to store the instruction, success fn, and repeat information
    # To avoid recursion, we also store the key for the RUS instruction
    data = {
        "instruction": instruction,
        "success_fn": success_fn,
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
        qubit_mapping: Mapping[str, str], instruction: Instruction, **kwargs
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
    param_priorities = instruction.param_priorities
    for k in data:
        param_priorities[k] = DEFAULT_PRIORITIES
    # We also need the patch_label for new labels, and the stack to update it
    param_priorities["patch_label"] = param_priorities.get(
        "patch_label", DEFAULT_PRIORITIES
    )
    param_priorities["stack"] = param_priorities.get(
        "stack", DEFAULT_PRIORITIES
    )

    def dry_run_fn(**kwargs) -> Frame:
        # Just put the instruction on the stack so that it can deal with it
        # Pull some args out of kwargs
        instruction = kwargs.pop("instruction")
        assert isinstance(instruction, Instruction)

        stack = InstructionStack.cast(kwargs.pop("stack"))

        del kwargs["rus_key"]
        del kwargs["success_fn"]
        del kwargs["max_repeats"]
        del kwargs["repeat_count"]

        new_label = InstructionLabel(instruction, inst_kwargs=kwargs)
        stack = stack.insert_instruction(0, new_label)

        return Frame({"stack": stack})

    return Instruction(
        apply_fn=apply_fn,
        dry_run_apply_fn=dry_run_fn,
        data=data,
        map_qubits_fn=map_qubits_fn,
        param_priorities=param_priorities,
        param_error_behavior="continue",  # Skip warning for variadic kwargs
        name=name,
    )
