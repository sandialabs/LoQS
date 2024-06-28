"""Class definition for QuantumProgram
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import warnings

from loqs.backends.model import BaseNoiseModel
from loqs.backends.state import BaseQuantumState
from loqs.core import Instruction, InstructionStack, History
from loqs.core.history import Frame, HistoryCastableTypes
from loqs.core.instructions import common as ic
from loqs.core.instructions import InstructionLabel
from loqs.core.instructions.instructionlabel import (
    InstructionLabelCastableTypes,
)
from loqs.core.instructions.instructionstack import (
    InstructionStackCastableTypes,
)
from loqs.core.qeccode import QECCode
from loqs.core.recordables import PatchDict


class QuantumProgram:
    """A container for the main quantum program to be executed."""

    def __init__(
        self,
        instruction_stack: InstructionStackCastableTypes = None,
        initial_history: HistoryCastableTypes = None,
        default_noise_model: BaseNoiseModel | None = None,
        expiring_state: bool = True,
        global_instructions: Mapping[str, Instruction] | None = None,
        state_type: type[BaseQuantumState] | None = None,
        patch_types: Mapping[str, QECCode] | None = None,
        override_global_instructions: bool = False,
        name: str = "(Unnamed quantum program)",
    ) -> None:
        """Initialize a QuantumProgram from a list of operations.

        TODO
        """
        # Do history before instruction stack in case it already has one
        self.initial_history = History.cast(initial_history)
        if instruction_stack is None and (
            initial_history is None
            or len(self.initial_history) < 1
            or "stack" not in self.initial_history[-1]
        ):
            raise ValueError(
                "Must provide either initial instruction stack or history with a stack"
            )

        self.default_noise_model = default_noise_model

        if expiring_state:
            self.initial_history.expiring_keys.add("state")

        # Create the instruction stack and add it to the history
        if instruction_stack is not None:
            instruction_stack = InstructionStack.cast(instruction_stack)

            if len(self.initial_history):
                last_frame = Frame.cast(self.initial_history[-1])
            else:
                last_frame = Frame()

            new_frame = last_frame.update(
                {"stack": instruction_stack},
                new_log=f"Adding InstructionStack from new QuantumProgram {name}",
            )

            if len(last_frame) == 0:
                # This was an empty frame
                # Let's just start with the new stack frame
                self.initial_history = History.cast(new_frame)
            else:
                # Let's append to the existing history
                self.initial_history.append(new_frame)

        if global_instructions is None:
            global_instructions = {}
        self.global_instructions = {
            k: v for k, v in global_instructions.items()
        }
        assert all(
            [
                isinstance(v, Instruction)
                for v in self.global_instructions.values()
            ]
        )

        # Add state initialization, if requested
        if state_type is not None:
            if (
                "Init State" in self.global_instructions
                and not override_global_instructions
            ):
                warnings.warn(
                    "state_type provided, but 'Init State' already exists "
                    + "and override_global_instructions is False. Consider "
                    + "renaming the existing 'Init State' or "
                    + "setting override_global_instruction to True."
                )
            else:
                builder = ic.build_object_builder_instruction(
                    "state",
                    state_type,
                    name=f"{state_type.__qualname__} state builder",
                )
                self.global_instructions["Init State"] = builder

        # Add patch initializations/removals, if requested
        if patch_types is not None:
            for patch_name, patch_code in patch_types.items():
                label = f"Init Patch {patch_name}"

                if (
                    label in self.global_instructions
                    and not override_global_instructions
                ):
                    warnings.warn(
                        f"patch_types['{patch_name}'] provided, "
                        + f"but '{label}' already exists "
                        + "and override_global_instructions is False. Consider "
                        + f"renaming the existing '{label}' or "
                        + "setting override_global_instruction to True."
                    )
                builder = ic.build_patch_builder_instruction(
                    patch_code,
                    name=f"{patch_name} patch builder",
                )
                self.global_instructions[label] = builder

            if (
                "Remove Patch" in self.global_instructions
                and not override_global_instructions
            ):
                warnings.warn(
                    "patch_types provided, but 'Remove Patch' already exists "
                    + "and override_global_instructions is False. Consider "
                    + "renaming the existing 'Remove Patch' or "
                    + "setting override_global_instruction to True."
                )
            else:
                builder = ic.build_patch_remover_instruction(
                    name="Global patch remover"
                )
                self.global_instructions["Remove Patch"] = builder

        self.run_histories = []

    def run(
        self, dry_run: bool = False, max_frame_limit: int = 100
    ) -> History:
        """TODO"""
        num_frames = 0

        history = copy.deepcopy(self.initial_history)

        stack = InstructionStack.cast(history[-1]["stack"])

        print(f"Executing program run {len(self.run_histories)}")

        while num_frames < max_frame_limit and len(stack):
            print(f"Working on frame {num_frames+1}")

            inst, stack = stack.pop_instruction()
            patch_label = inst.patch_label
            args = inst.inst_args
            kwargs = inst.inst_kwargs

            inst = self._resolve_instruction(inst, history[-1])

            # Some label information we can actually provide
            # Check to see if it is needed, and provide if not available
            available_params = {
                "model": self.default_noise_model,
                "patch_label": patch_label,
                "stack": stack,
            }
            for k, v in available_params.items():
                try:
                    param_spec = inst.input_spec.get_by_key(k)
                except KeyError:
                    # Not needed because not in input spec
                    continue

                # Check if in args or kwargs
                if len(args) > param_spec.position or k in kwargs:
                    continue

                # For model specifically, if we are providing it, should not be None
                if k == "model" and v is None:
                    raise ValueError(
                        "Model being requested and not provided. Either "
                        + "provide it as label arg/kwarg or set default_noise_model."
                    )

                # If not otherwise provided, lets provide it in kwargs
                kwargs[k] = v

            applied_frame = inst.apply(history, dry_run, *args, **kwargs)

            # Only update stack if the instruction did not
            if "stack" not in inst.output_spec:
                new_frame = applied_frame.update({"stack": stack})
            else:
                new_frame = applied_frame

            history.append(new_frame)

            stack = InstructionStack.cast(new_frame["stack"])

            num_frames += 1

        if len(stack):
            warnings.warn(
                f"Terminated run due to `max_frame_limit` of {max_frame_limit}"
            )

        if dry_run:
            print("Dry run completed successfully!")

        self.run_histories.append(history)

        return History.cast(history[-num_frames:])

    def _resolve_instruction(
        self, inst_lbl: InstructionLabelCastableTypes, frame: Frame
    ) -> Instruction:
        ilbl = InstructionLabel.cast(inst_lbl)

        if ilbl.instruction is not None:
            return ilbl.instruction

        # If we are here, we need the inst_label to resolve
        assert ilbl.inst_label is not None

        # First check global
        if ilbl.patch_label is None:
            try:
                inst = self.global_instructions[ilbl.inst_label]
            except KeyError:
                raise RuntimeError(
                    f"Could not resolve global instruction from {ilbl}"
                )

            return inst

        # Otherwise, we must be a patch instruction
        try:
            patchdict = PatchDict.cast(frame["patches"])
        except KeyError:
            raise RuntimeError(
                f"'patches' not available in last frame for resolving {ilbl}"
            )

        try:
            patch = patchdict.patches[ilbl.patch_label]
        except KeyError:
            raise RuntimeError(
                f"Patch {ilbl.patch_label} not available for resolving {ilbl}"
            )

        try:
            inst = patch[ilbl.inst_label]
        except KeyError:
            raise RuntimeError(
                f"{ilbl.inst_label} not available in patch for resolving {ilbl}"
            )

        return inst
