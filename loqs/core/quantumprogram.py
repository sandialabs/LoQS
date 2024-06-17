"""Class definition for QuantumProgram
"""

from __future__ import annotations

from collections.abc import Mapping
import warnings

from loqs.core import Instruction, InstructionStack, HistoryStack
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.recordables import PatchDict


class QuantumProgram:
    """A container for the main quantum program to be executed."""

    def __init__(
        self,
        initial_history: HistoryStackCastableTypes,
        patch_key: str = "patches",
        stack_key: str = "stack",
        global_instructions: Mapping[str, Instruction] | None = None,
    ) -> None:
        """Initialize a QuantumProgram from a list of operations."""
        self.history = HistoryStack.cast(initial_history)
        self.patch_key = patch_key
        self.stack_key = stack_key

        # TODO: Add patch creation to global instruction if not available
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

        # Check that required keys are available
        assert len(
            self.history
        ), "Must provide starting frame with at least initial stack"
        last_frame = self.history[-1]

        assert (
            stack_key in last_frame
        ), f"`stack_key` {stack_key} not available in initial history"
        try:
            InstructionStack.cast(last_frame[stack_key])
        except Exception as e:
            raise ValueError("Cannot create stack") from e

        try:
            PatchDict.cast(last_frame.get(patch_key, None))
        except Exception as e:
            raise ValueError("Cannot create patches") from e

    def run(self, max_frame_limit: int = 100) -> HistoryStack:
        """TODO"""
        num_frames = 0

        stack = InstructionStack.cast(self.history[-1][self.stack_key])

        while num_frames < max_frame_limit and len(stack):
            # Resolve instruction

            # Perform instruction

            # Update stack

            num_frames += 1

        if len(stack):
            warnings.warn(
                f"Terminated run due to `max_frame_limit` of {max_frame_limit}"
            )

        return HistoryStack.cast(self.history[-num_frames:])
