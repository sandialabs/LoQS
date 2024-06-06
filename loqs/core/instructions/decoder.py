""":class:`MockOperations` definition.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from loqs.core import Instruction, HistoryStack, HistoryFrame
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.recordables.measurementoutcomes import MeasurementOutcomes
from loqs.core.recordables.stabilizerframe import StabilizerFrame


DecoderCallable: TypeAlias = Callable[
    [MeasurementOutcomes, StabilizerFrame], StabilizerFrame
]
"""Type alias for decoding functions :class:`Decoder` can take.
"""

DecoderCastableTypes: TypeAlias = "Decoder | DecoderCallable"
"""TODO"""


class Decoder(Instruction):
    """TODO"""

    def __init__(self, decoder: DecoderCastableTypes) -> None:
        """TODO"""
        if isinstance(decoder, Decoder):
            self.decoder_fn = decoder.decoder_fn
        else:
            self.decoder_fn = decoder

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return {
            "measurement_outcomes": MeasurementOutcomes,
            "stabilizer_frame": StabilizerFrame,
        }

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {"stabilizer_frame": StabilizerFrame, "instruction": Decoder}

    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
        """TODO"""
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        outcomes = MeasurementOutcomes.cast(last_frame["measurement_outcomes"])
        old_stab_frame = StabilizerFrame.cast(last_frame["stabilizer_frame"])

        new_stab_frame = self.decoder_fn(outcomes, old_stab_frame)

        new_data = {
            "stabilizer_frame": new_stab_frame,
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame
