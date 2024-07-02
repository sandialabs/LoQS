"""TODO
"""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
import inspect as ins

from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core.frame import Frame
from loqs.core.instructions import Instruction
from loqs.core.instructions.instruction import DEFAULT_PRIORITIES, KwargDict
from loqs.core.instructions.instructionlabel import InstructionLabel
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.qeccode import QECCode
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict


def build_composite_instruction(
    instructions: Sequence[Instruction],
    param_priorities: Sequence[str] | Mapping[str, Sequence[str]],
    name: str = "(Unnamed composite instruction)",
    parent: object | None = None,
    fault_tolerant: bool = True,
) -> Instruction:
    """TODO"""

    # We don't know what args we need, so let's take variadic kwargs
    def apply_fn(**kwargs) -> Frame:
        # These will always be the last two
        assert "stack" in kwargs
        assert "instructions" in kwargs
        stack = InstructionStack.cast(kwargs["stack"])
        instructions: list[Instruction] = list(kwargs["instructions"])

        for i, instruction in enumerate(instructions):
            patch_label = None
            label_kwargs = {}
            for k, v in kwargs.items():
                if k in param_priorities:
                    # We want to forward this via the label
                    if k == "patch_label":
                        patch_label = v
                    else:
                        label_kwargs[k] = v

            new_label = InstructionLabel(
                instruction, patch_label, inst_kwargs=kwargs
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
    if not isinstance(param_priorities, Mapping):
        param_priorities = {k: DEFAULT_PRIORITIES for k in param_priorities}
    # We also need the stack and instructions
    param_priorities = dict(param_priorities)
    param_priorities["stack"] = DEFAULT_PRIORITIES
    param_priorities["instructions"] = DEFAULT_PRIORITIES

    composite_instruction = Instruction(
        apply_fn=apply_fn,
        dry_run_apply_fn=apply_fn,  # Just stack updates, can run in dry_run
        map_qubits_fn=map_qubits_fn,
        data=data,
        param_priorities=param_priorities,
        param_error_behavior="continue",  # Suppress the warning for variadic kwargs
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
    )

    # Set all instructions to have this composite as parent
    data_instructions = composite_instruction.data["instructions"]
    for instruction in data_instructions:  # type: ignore
        instruction.parent = composite_instruction

    return composite_instruction


def build_lookup_decoder_instruction() -> None:
    """TODO"""
    raise NotImplementedError("TODO")


def build_object_builder_instruction(
    frame_key: str,
    obj_class: type,
    name: str = "(Unnamed object builder)",
    parent: object | None = None,
    fault_tolerant: bool = True,
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
        param_priorities[param] = ["label"]
    # And add on our instruction data information
    for k in data:
        param_priorities[k] = ["instruction"]

    return Instruction(
        apply_fn=apply_fn,
        dry_run_apply_fn=apply_fn,  # Object creation can be done in dry runs
        data=data,
        param_priorities=param_priorities,
        param_error_behavior="continue",  # Suppress variadic kwargs warning
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
    )


def build_patch_builder_instruction(
    qec_code: QECCode,
    name: str = "(Unnamed patch builder)",
    parent: object | None = None,
    fault_tolerant: bool = True,
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
        dry_run_apply_fn=apply_fn,  # Patch manip is dry-run safe
        data=data,
        param_priorities=param_priorities,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
    )


def build_patch_remover_instruction(
    name: str = "(Unnamed patch builder)",
    parent: object | None = None,
    fault_tolerant: bool = True,
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
        dry_run_apply_fn=apply_fn,  # Patch manip is dry-run safe
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
    )


def build_patch_permute_instruction(
    mapping: Mapping[str, str],
    name: str = "(Unnamed patch permutation)",
    parent: object | None = None,
    fault_tolerant: bool = True,
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
        dry_run_apply_fn=apply_fn,  # Patch manip is dry-run safe
        map_qubits_fn=map_qubits_fn,
        data=data,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
    )


def build_physical_circuit_instruction(
    circuit: BasePhysicalCircuit,
    include_outcomes: bool = False,
    inplace: bool = True,
    reset_mcms: bool = True,
    name: str = "(Unnamed physical circuit)",
    parent: object | None = None,
    fault_tolerant: bool | None = None,
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
        map_qubits_fn=map_qubits_fn,
        data=data,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
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
    for _, v in outcomes:
        if any([bit != 0 for bit in v]):
            return False

    return True


def build_repeat_until_success_instruction(
    instruction: Instruction,
    success_fn: Callable[[MeasurementOutcomes], bool] = _default_success_fn,
    max_repeats: int = 100,
    name: str = "(Unnamed repeat-until-success instruction)",
    parent: object | None = None,
    fault_tolerant: bool | None = None,
) -> Instruction:
    """TODO"""
    if fault_tolerant is None:
        fault_tolerant = instruction.fault_tolerant

    # We do not know all the params for the underlying instruction,
    # so take variadic kwargs here
    def apply_fn(**kwargs) -> Frame:
        # Pull some args out of kwargs
        instruction = kwargs.pop("instruction")
        assert isinstance(instruction, Instruction)

        self = kwargs.pop("self")
        assert isinstance(self, Instruction)

        success_fn = kwargs.pop("success_fn")
        max_repeats = int(kwargs.pop("max_repeats"))
        repeat_count = int(kwargs.pop("repeat_count"))

        stack = InstructionStack.cast(kwargs.pop("stack"))

        # All the remaining kwargs should go straight into the instruction
        applied_frame = instruction.apply(**kwargs)

        # Run success function to see if we are terminated
        try:
            success = success_fn(applied_frame["measurement_outcomes"])
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

            # Otherwise, let's add another copy onto the stack
            # We need to at least put repeat_counts into label args
            new_label = InstructionLabel(
                self, inst_kwargs={"repeat_count": repeat_count}
            )

            stack = stack.insert_instruction(0, new_label)

        # Return frame with the stack update
        return applied_frame.update({"stack": stack})

    # We need to store the instruction, success fn, and repeat information
    # One weird note: I want to store self in this, but we need the object first...
    # So we will have some post-processing
    data = {
        "instruction": instruction,
        "success_fn": success_fn,
        "max_repeats": max_repeats,
        "repeat_count": 1,
    }

    # We need to map the instruction
    def map_qubits_fn(
        qubit_mapping: Mapping[str, str], instruction: Instruction, **kwargs
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["instruction"] = instruction.map_qubits(qubit_mapping)
        return new_kwargs

    # Since we have variadic kwargs, we'll set the param priority ourselves
    # Default order is OK for new params. We'll pick up most of what we need
    # from instruction, but repeat count will be from label arg after first iteration
    param_priorities = instruction.param_priorities
    for k in data:
        param_priorities[k] = DEFAULT_PRIORITIES

    def dry_run_fn(**kwargs) -> Frame:
        # Just put the instruction on the stack so that it can deal with it
        # Pull some args out of kwargs
        instruction = kwargs.pop("instruction")
        assert isinstance(instruction, Instruction)

        stack = InstructionStack.cast(kwargs.pop("stack"))

        del kwargs["self"]
        del kwargs["success_fn"]
        del kwargs["max_repeats"]
        del kwargs["repeat_count"]

        new_label = InstructionLabel(instruction, inst_kwargs=kwargs)
        stack = stack.insert_instruction(0, new_label)

        return Frame({"stack": stack})

    rus_instruction = Instruction(
        apply_fn=apply_fn,
        dry_run_apply_fn=dry_run_fn,
        map_qubits_fn=map_qubits_fn,
        data=data,
        param_priorities=param_priorities,
        param_error_behavior="continue",  # Skip warning for variadic kwargs
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
    )

    # Add self and set up collection from rus_instruction.data
    rus_instruction.data["self"] = rus_instruction
    rus_instruction.param_priorities["self"] = ["instruction"]

    return rus_instruction
