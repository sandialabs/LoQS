"""TODO
"""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
import inspect

from loqs.backends import propagate_state
from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core.frame import Frame
from loqs.core.instructions import Instruction
from loqs.core.instructions.inputspec import InputSpec
from loqs.core.instructions.instruction import KwargDict
from loqs.core.instructions.instructionlabel import InstructionLabel
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
    # Anything that is needed from labels in subsequent instructions,
    # we have to ask for now so that we can set the label properly
    input_spec = []
    for instruction in instructions:
        for param in instruction.input_spec:
            if "label" in param.sources:
                new_label = (param.sources, param.key)
                input_spec.append(new_label)

    # We'll just unfold our instruction into the stack
    # Order matters (should match apply_fn)
    # We want most up to date stack from QuantumProgram, i.e. with popped Instruction already
    # Thus we ask for label and not history, -1 as you might expect
    input_spec.extend(
        [
            ("label", "stack"),
            ("default", "instructions"),
        ]
    )

    input_spec = InputSpec.cast(input_spec)

    # We'll output a new stack to use
    output_spec = ["stack"]

    # We need to store the instructions
    defaults = {"instructions": instructions}

    # We do not know all the params for the underlying instructions
    # We have ensured that all label sources are in the input_spec,
    # so we assume they are all passed in and we can pull them out by
    # position here to set the created labels properly
    def apply_fn(*args) -> Frame:
        # These will always be the last two
        stack = InstructionStack.cast(args[-2])
        instructions: list[Instruction] = list(args[-1])

        for i, instruction in enumerate(instructions):
            kwargs = {}
            subspec = instruction.input_spec
            for subparam in subspec:
                if "label" in subparam.sources:
                    # Check what position this mapped to in the full spec
                    assert subparam.key is not None
                    param = input_spec.get_by_key(subparam.key)
                    kwargs[param.key] = args[param.position]

            new_label = InstructionLabel(instruction, inst_kwargs=kwargs)

            stack = stack.insert_instruction(i, new_label)

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

    # Composite instructions are only stack updates
    # These are one of the things that must run in dry runs
    # to verify the correct codepath is followed
    composite_instruction = Instruction(
        apply_fn=apply_fn,
        input_spec=input_spec,
        output_spec=output_spec,
        map_qubits_fn=map_qubits_fn,
        defaults=defaults,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
        skip_in_dry_run=False,
    )

    # Set all instructions to have this composite as parent
    for instruction in composite_instruction.defaults["instructions"]:
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

    # This is sort of a special input_spec because we don't know
    # all the args a priori.
    # First, we use inspect to introspect the __init__ signature
    input_spec = []
    init_sig = inspect.signature(obj_class)
    for param in init_sig.parameters.values():
        input_spec.append(("label", param.name))

    # Then we set the defaults for the metadata we need
    input_spec.extend([("default", "frame_key"), ("default", "obj_class")])

    # Output includes the new object
    output_spec = [frame_key]

    # Defaults include the frame key and object class
    defaults = {"frame_key": frame_key, "obj_class": obj_class}

    # This apply_fn is a bit of an oddity. Normally I like to full specify
    # the args needed. However, in this case, we have no idea what the kwargs
    # needed by the object constructor are. So in this case, we let them
    # pass through without enforcement
    def apply_fn(*args) -> Frame:

        frame_key = args[-2]
        obj_class = args[-1]
        obj_args = args[:-2]
        try:
            obj = obj_class(*obj_args)
        except Exception as e:
            raise ValueError(
                "Failed to create object in object builder"
            ) from e

        return Frame({frame_key: obj})

    # No need for map_qubits

    # Object builders are only metadata updates
    # These are one of the things that must run in dry runs
    # to verify the correct codepath is followed
    return Instruction(
        apply_fn=apply_fn,
        input_spec=input_spec,
        output_spec=output_spec,
        defaults=defaults,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
        skip_in_dry_run=False,
    )


def build_patch_builder_instruction(
    qec_code: QECCode,
    name: str = "(Unnamed patch builder)",
    parent: object | None = None,
    fault_tolerant: bool = True,
) -> Instruction:

    # We require patch name and qubits to be passed in
    # Order matters (should match apply_fn)
    # Labels should be first to enable passing by args in InstrumentLabel
    input_spec = [
        ("label", "patch_label"),
        ("label", "qubits"),
        ("default", "qec_code"),
        (["history", "default"], "patches", -1),
    ]

    # Output includes a PatchDict
    output_spec = ["patches"]

    # Default is the QECCode and an empty PatchDict
    # This should only be used if one does not exist in history,
    # since our load order is ["history", "default"]
    defaults = {"qec_code": qec_code, "patches": PatchDict()}

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

    # No need for map_qubits

    # Patch builders are only patch metadata operations
    # These are one of the things that must run in dry runs
    # to verify the correct codepath is followed
    return Instruction(
        apply_fn=apply_fn,
        input_spec=input_spec,
        output_spec=output_spec,
        defaults=defaults,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
        skip_in_dry_run=False,
    )


def build_patch_remover_instruction(
    name: str = "(Unnamed patch builder)",
    parent: object | None = None,
    fault_tolerant: bool = True,
) -> Instruction:

    # We require patch name and qubits to be passed in
    # Order matters (should match apply_fn)
    # Labels should be first to enable passing by args in InstrumentLabel
    input_spec = [
        ("label", "patch_label"),
        ("history", "patches", -1),
    ]

    # Output includes a PatchDict
    output_spec = ["patches"]

    # No defaults

    def apply_fn(
        patch_label: str,
        patches: PatchDict,
    ) -> Frame:
        assert (
            patch_label in patches
        ), f"Patch remover failed, could not find patch {patch_label}"

        del patches[patch_label]

        return Frame({"patches": patches})

    # No need for map_qubits

    # Patch removal are only patch metadata operations
    # These are one of the things that must run in dry runs
    # to verify the correct codepath is followed
    return Instruction(
        apply_fn=apply_fn,
        input_spec=input_spec,
        output_spec=output_spec,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
        skip_in_dry_run=False,
    )


def build_patch_permute_instruction(
    mapping: Mapping[str, str],
    name: str = "(Unnamed patch permutation)",
    parent: object | None = None,
    fault_tolerant: bool = True,
) -> Instruction:
    """TODO"""
    # Order matters (must match apply_fn)
    # Labels should be first to enable passing by args in InstrumentLabel
    input_spec = [
        ("label", "patch_label"),
        ("default", "mapping"),
        ("history", "patches", -1),
    ]

    # Output includes a PatchDict
    output_spec = ["patches"]

    # We store the mapping
    defaults = {"mapping": mapping}

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

    def map_qubits_fn(
        qubit_mapping: Mapping[str, str],
        mapping: Mapping[str, str],
    ) -> KwargDict:
        new_mapping = {
            qubit_mapping[k]: qubit_mapping[v] for k, v in mapping.items()
        }
        return {"mapping": new_mapping}

    # Patch permutations are only patch metadata operations
    # These are one of the things that must run in dry runs
    # to verify the correct codepath is followed
    return Instruction(
        apply_fn=apply_fn,
        input_spec=input_spec,
        output_spec=output_spec,
        defaults=defaults,
        map_qubits_fn=map_qubits_fn,
        name=name,
        parent=parent,
        fault_tolerant=fault_tolerant,
        skip_in_dry_run=False,
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
    # Order matters (must match apply_fn)
    # Labels should be first to enable passing by args in InstrumentLabel
    input_spec = [
        ("label", "model"),
        ("default", "circuit"),
        ("history", "state", -1),
        ("default", "include_outcomes"),
        ("default", "inplace"),
        ("default", "reset_mcms"),
    ]

    # We output state and optionally outcomes
    output_spec = ["state"]
    if include_outcomes:
        output_spec.append("measurement_outcomes")

    # We store circuit and flags as defaults
    defaults = {
        "circuit": circuit,
        "include_outcomes": include_outcomes,
        "inplace": inplace,
        "reset_mcms": reset_mcms,
    }

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
        defaults=defaults,
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
