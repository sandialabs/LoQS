"""TODO
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

from loqs.core import Instruction, HistoryStack, HistoryFrame
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.recordables import StabilizerFrame, Syndrome
from loqs.internal import Bit, PauliStr


DecoderCallable: TypeAlias = Callable[
    [Syndrome, StabilizerFrame], StabilizerFrame
]
"""Type alias for decoding functions :class:`Decoder` can take.
"""

DecoderTable: TypeAlias = Mapping[tuple[Bit, ...], tuple[PauliStr, str]]
"""TODO
"""

DecoderCastableTypes: TypeAlias = "Decoder | DecoderCallable | DecoderTable"
"""TODO"""


class Decoder(Instruction):
    """TODO"""

    decoder_fn: DecoderCallable
    """TODO
    """

    decoder_table: DecoderTable | None
    """TODO
    """

    def __init__(self, decoder: DecoderCastableTypes) -> None:
        """TODO"""
        if isinstance(decoder, Decoder):
            self.decoder_fn = decoder.decoder_fn
            self.decoder_table = decoder.decoder_table
        elif callable(decoder):
            self.decoder_fn = decoder
            self.decoder_table = None
        elif isinstance(decoder, Mapping):
            self.decoder_table = decoder
            self.decoder_fn = self._lookup_table
        else:
            raise NotImplementedError(
                f"Cannot create a decoder from {decoder}"
            )

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return {
            "syndrome": Syndrome,
            "stabilizer_frame": StabilizerFrame,
        }

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {"stabilizer_frame": StabilizerFrame, "instruction": Decoder}

    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
        """TODO"""
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        syndrome = Syndrome.cast(last_frame["syndrome"])
        old_stab_frame = StabilizerFrame.cast(last_frame["stabilizer_frame"])

        new_stab_frame = self.decoder_fn(syndrome, old_stab_frame)

        new_data = {
            "stabilizer_frame": new_stab_frame,
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame

    def _lookup_table(
        self, syn: Syndrome, sf: StabilizerFrame
    ) -> StabilizerFrame:
        """TODO"""
        assert self.decoder_table is not None

        new_sf = sf.copy()

        # TODO
        # self.decoder_table[syn.bitstring]

        return new_sf
