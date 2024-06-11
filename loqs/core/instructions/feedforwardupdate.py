"""TODO
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

from loqs.core import Instruction, InstructionStack, HistoryStack, HistoryFrame
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.instruction import InstructionParentTypes
from loqs.core.recordables import MeasurementOutcomes, StabilizerFrame
from loqs.internal import Bit, PauliStr


FeedForwardUpdateCallable: TypeAlias = Callable[
    [MeasurementOutcomes, InstructionStack], InstructionStack
]
"""Type alias for decoding functions :class:`Decoder` can take.
"""
FFUCallable = FeedForwardUpdateCallable

FeedForwardUpdateTable: TypeAlias = Mapping[tuple[Bit, ...], Instruction]
"""TODO
"""
FFUTable = FeedForwardUpdateTable

FeedForwardUpdateCastableTypes: TypeAlias = (
    "FeedForwardUpdate | FFUCallable | FFUTable"
)
"""TODO"""


class FeedForwardUpdate(Instruction):
    """TODO"""

    update_fn: FFUCallable
    """TODO
    """

    update_table: FFUTable | None
    """TODO
    """

    def __init__(
        self,
        ffupdate: FeedForwardUpdateCastableTypes,
        name: str = "(Unnamed decoder)",
        parent: InstructionParentTypes = None,
    ) -> None:
        """TODO"""
        if isinstance(ffupdate, FeedForwardUpdate):
            self.update_fn = ffupdate.update_fn
            self.update_table = ffupdate.update_table
        elif callable(ffupdate):
            self.update_fn = ffupdate
            self.update_table = None
        elif isinstance(ffupdate, Mapping):
            self.update_table = ffupdate
            self.update_fn = self._lookup_table
        else:
            raise NotImplementedError(
                f"Cannot create a feed-forward update from {ffupdate}"
            )

        super().__init__(name, parent)

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return {
            "outcomes": MeasurementOutcomes,
            "stack": InstructionStack,
        }

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {"stack": InstructionStack, "instruction": FeedForwardUpdate}

    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
        """TODO"""
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        outcomes = MeasurementOutcomes.cast(last_frame["outcomes"])
        stack = InstructionStack.cast(last_frame["stack"])

        new_stack = self.update_fn(outcomes, stack)

        new_data = {
            "stack": new_stack,
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame

    def _lookup_table(
        self, outcomes: MeasurementOutcomes, stack: InstructionStack
    ) -> InstructionStack:
        """TODO"""
        assert self.update_table is not None

        new_stack = stack  # .copy()

        # TODO
        # self.decoder_table[syn.bitstring]

        return new_stack
