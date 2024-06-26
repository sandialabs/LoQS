"""TODO
"""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence

from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core.frame import Frame
from loqs.core.instructions import Instruction
from loqs.core.instructions.inputspec import InputParam
from loqs.core.instructions.instruction import KwargDict
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.qeccode import QECCode
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.patchdict import PatchDict


def build_composite_instruction(
    instructions: Sequence[Instruction],
    name: str = "(Unnamed composite instruction)",
    parent: object | None = None,
    fault_tolerant: bool = True,
) -> Instruction:
    """TODO"""
    # We'll just unfold our instruction into the stack
    input_spec = [
        InputParam("stack", "history", -1),
        InputParam("instructions", "default"),
    ]

    # We'll output a new stack to use
    output_spec = ["stack"]

    # We need to store the instructions
    default_kwargs = {"instructions": instructions}

    def apply_fn(
        stack: InstructionStack, instructions: Sequence[Instruction]
    ) -> Frame:
        for instruction in instructions:
            stack.append_instruction(instruction)
        return Frame({"stack": stack})

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

    return Instruction(
        apply_fn=apply_fn,
        input_spec=input_spec,
        output_spec=output_spec,
        map_qubits_fn=map_qubits_fn,
        default_kwargs=default_kwargs,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
    )


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

    # We do not know all the parameters incoming
    # However, we will at least need to pass in the defaults
    input_spec = [
        InputParam("frame_key", "default"),
        InputParam("obj_class", "default"),
    ]

    # Output includes the new object
    output_spec = [frame_key]

    # Defaults include the frame key and object class
    default_kwargs = {"frame_key": frame_key, "obj_class": obj_class}

    # This apply_fn is a bit of an oddity. Normally I like to full specify
    # the args needed. However, in this case, we have no idea what the kwargs
    # needed by the object constructor are. So in this case, we let them
    # pass through without enforcement
    def apply_fn(frame_key: str, obj_class: type, **kwargs) -> Frame:
        try:
            obj = obj_class(**kwargs)
        except Exception as e:
            raise ValueError(
                "Failed to create object in object builder"
            ) from e

        return Frame({frame_key: obj})

    # No need for map_qubits

    return Instruction(
        apply_fn=apply_fn,
        input_spec=input_spec,
        output_spec=output_spec,
        default_kwargs=default_kwargs,
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

    # We require patch name and qubits to be passed in
    input_spec = [
        InputParam("qec_code", "default"),
        InputParam("patches", ["history", "default"], -1),
        InputParam("patch_name", "label"),
        InputParam("qubits", "label"),
    ]

    # Output includes a PatchDict
    output_spec = ["patches"]

    # Default is the QECCode and an empty PatchDict
    # This should only be used if one does not exist in history,
    # since our load order is ["history", "default"]
    default_kwargs = {"qec_code": qec_code, "patches": PatchDict()}

    def apply_fn(
        qec_code: QECCode,
        patches: PatchDict,
        patch_name: str,
        qubits: Sequence[str],
    ) -> Frame:
        all_patch_qubits = patches.all_qubit_labels

        # Disjoint patch checks
        assert all(
            [q not in all_patch_qubits for q in qubits]
        ), f"Patch builder failed, requesting overlapping patches for {patch_name}"
        assert (
            patch_name not in patches
        ), f"Patch builder failed, already have existing patch {patch_name}"

        try:
            patch = qec_code.create_patch(qubits)
        except Exception as e:
            raise ValueError("Failed to create patch in patch builder") from e

        patches[patch_name] = patch

        return Frame({"patches": patches})

    # No need for map_qubits

    return Instruction(
        apply_fn=apply_fn,
        input_spec=input_spec,
        output_spec=output_spec,
        default_kwargs=default_kwargs,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
    )


def build_patch_remover_instruction(
    name: str = "(Unnamed patch builder)",
    parent: object | None = None,
    fault_tolerant: bool = True,
) -> Instruction:

    # We require patch name and qubits to be passed in
    input_spec = [
        InputParam("patches", "history", -1),
        InputParam("patch_name", "label"),
    ]

    # Output includes a PatchDict
    output_spec = ["patches"]

    # No defaults provided
    default_kwargs = {}

    def apply_fn(
        patches: PatchDict,
        patch_name: str,
    ) -> Frame:
        assert (
            patch_name in patches
        ), f"Patch remover failed, could not find patch {patch_name}"

        del patches[patch_name]

        return Frame({"patches": patches})

    # No need for map_qubits

    return Instruction(
        apply_fn=apply_fn,
        input_spec=input_spec,
        output_spec=output_spec,
        default_kwargs=default_kwargs,
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
    # We require the state from history, a passed-in model on apply,
    # and everything else is stored as defaults from this constructor
    input_spec = [
        InputParam("mapping", "default"),
        InputParam("patches", "history", -1),
        InputParam("patch_name", "label"),
    ]

    # Output includes a PatchDict
    output_spec = ["patches"]

    # We store the mapping
    default_kwargs = {"mapping": mapping}

    def apply_fn(
        patches: PatchDict,
        patch_name: str,
        mapping: Mapping[str, str],
    ) -> Frame:
        assert (
            patch_name in patches
        ), f"Patch permute failed, could not find patch {patch_name}"

        patch = patches[patch_name]

        code = patch.code
        qubits = patch.qubits

        mapped_qubits = [mapping[q] for q in qubits]

        permuted_patch = code.create_patch(mapped_qubits)

        patches[patch_name] = permuted_patch

        return Frame({"patches": patches})

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
        input_spec=input_spec,
        output_spec=output_spec,
        default_kwargs=default_kwargs,
        map_qubits_fn=map_qubits_fn,
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
    # We require the state from history, a passed-in model on apply,
    # and everything else is stored as defaults from this constructor
    input_spec = [
        InputParam("circuit", "default"),
        InputParam("model", "label"),
        InputParam("state", "history", -1),
        InputParam("include_outcomes", "default"),
        InputParam("inplace", "default"),
        InputParam("reset_mcms", "default"),
    ]

    # We output state and optionally outcomes
    output_spec = ["state"]
    if include_outcomes:
        output_spec.append("measurement_outcomes")

    # We store circuit and flags as defaults
    default_kwargs = {
        "circuit": circuit,
        "include_outcomes": include_outcomes,
        "inplace": inplace,
        "reset_mcms": reset_mcms,
    }

    def apply_fn(
        circuit: BasePhysicalCircuit,
        model: BaseNoiseModel,
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
            data["measurement_outcomes"] = MeasurementOutcomes.cast(outcomes)

        return Frame(data)

    def map_qubits_fn(
        qubit_mapping: Mapping[str, str],
        circuit: BasePhysicalCircuit,
        **kwargs,
    ) -> KwargDict:
        new_kwargs = kwargs.copy()
        new_kwargs["circuit"] = circuit.map_qubit_labels(qubit_mapping)
        return new_kwargs

    return Instruction(
        apply_fn=apply_fn,
        input_spec=input_spec,
        output_spec=output_spec,
        default_kwargs=default_kwargs,
        map_qubits_fn=map_qubits_fn,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
    )


def build_repeat_until_success_instruction(
    instruction_to_repeat: Instruction,
    success_fn: Callable[[MeasurementOutcomes], bool],
    max_repeats: int = 100,
    name: str = "(Unnamed repeat-until-success instruction)",
    parent: object | None = None,
    fault_tolerant: bool | None = None,
) -> None:
    """TODO"""
    raise NotImplementedError("TODO")
