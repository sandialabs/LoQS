"""Class definition for QuantumProgram
"""

from __future__ import annotations

from collections.abc import Mapping
import warnings

from loqs.backends.state import BaseQuantumState
from loqs.core import Instruction, InstructionStack, HistoryStack
from loqs.core.history import HistoryFrame, HistoryStackCastableTypes
from loqs.core.instruction import (
    InstructionLabel,
    InstructionStackCastableTypes,
)
from loqs.core.instructions import ObjectBuilder
from loqs.core.instructions.patchoperations import PatchBuilder, PatchRemover
from loqs.core.qeccode import QECCode
from loqs.core.recordables import PatchDict


class QuantumProgram:
    """A container for the main quantum program to be executed."""

    def __init__(
        self,
        instruction_stack: InstructionStackCastableTypes = None,
        initial_history: HistoryStackCastableTypes = None,
        state_key: str = "state",
        expiring_state: bool = True,
        patch_key: str = "patches",
        stack_key: str = "stack",
        global_instructions: Mapping[str, Instruction] | None = None,
        state_type: type[BaseQuantumState] | None = None,
        patch_types: Mapping[str, QECCode] | None = None,
    ) -> None:
        """Initialize a QuantumProgram from a list of operations.

        TODO
        """
        self.history = HistoryStack.cast(initial_history)
        if instruction_stack is None and (
            initial_history is None
            or len(self.history) < 1
            or stack_key not in self.history[-1]
        ):
            raise ValueError(
                "Must provide either initial instruction stack or history with a stack"
            )

        self.state_key = state_key
        if expiring_state:
            self.history.expiring_keys.add(self.state_key)
        self.patch_key = patch_key
        self.stack_key = stack_key

        # Create the instruction stack and add it to the history
        if instruction_stack is not None:
            instruction_stack = InstructionStack.cast(instruction_stack)

            if len(self.history):
                last_frame = HistoryFrame.cast(self.history[-1])
            else:
                last_frame = HistoryFrame()

            new_frame = last_frame.update(
                {self.stack_key: instruction_stack},
                new_log="Adding InstructionStack from new QuantumProgram",
            )

            self.history.append(new_frame)

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
        if (
            "Init State" not in self.global_instructions
            and state_type is not None
        ):
            builder = ObjectBuilder(
                "state",
                state_type,
                name=f"{state_type.__qualname__} state builder",
            )
            self.global_instructions["Init State"] = builder

        # Add patch initializations/removals, if requested
        if (
            "Init Patch" not in self.global_instructions
            and patch_types is not None
        ):
            for patch_name, patch_code in patch_types.items():
                builder = PatchBuilder(
                    patch_code,
                    self.patch_key,
                    name=f"{patch_name} patch builder",
                )
                self.global_instructions[f"Init {patch_name} Patch"] = builder

            self.global_instructions["Remove Patch"] = PatchRemover(
                self.patch_key
            )

    def run(self, max_frame_limit: int = 100) -> HistoryStack:
        """TODO"""
        num_frames = 0

        stack = InstructionStack.cast(self.history[-1][self.stack_key])

        while num_frames < max_frame_limit and len(stack):
            print(f"Working on frame {num_frames+1}")

            inst, stack = stack.pop_instruction()
            if isinstance(inst, InstructionLabel):
                args = inst.inst_args
                kwargs = inst.inst_kwargs
            else:
                args = []
                kwargs = {}
            print(f"Working on {inst}")
            print(f"  with args {args} and kwargs {kwargs}\n")

            inst = self._resolve_instruction(inst)
            print(f"Resolved to {inst}")

            applied_frame = inst.apply(self.history, *args, **kwargs)
            print(f"Applied frame: {applied_frame}")

            new_frame = applied_frame.update({"stack": stack})
            print(f"Updated stack frame: {new_frame}")

            self.history.append(new_frame)
            num_frames += 1

        if len(stack):
            warnings.warn(
                f"Terminated run due to `max_frame_limit` of {max_frame_limit}"
            )

        return HistoryStack.cast(self.history[-num_frames:])

    def _resolve_instruction(
        self, inst_or_lbl: Instruction | InstructionLabel
    ) -> Instruction:
        if isinstance(inst_or_lbl, Instruction):
            return inst_or_lbl

        ilbl = InstructionLabel.cast(inst_or_lbl)

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
            patchdict = PatchDict.cast(self.history[-1][self.patch_key])
        except KeyError:
            raise RuntimeError(
                f"{self.patch_key} not available in last frame for resolving {ilbl}"
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
