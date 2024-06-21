""":class:`MeasurementOutcomes` definition.
"""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from loqs.internal import Bit, PauliStr, Recordable


SyndromeCastableTypes: TypeAlias = (
    "Syndrome | Sequence[Bit] | Mapping[PauliStr, Bit]"
)
"""TODO"""


class Syndrome(Recordable):
    """TODO"""

    bitstring: tuple[Bit, ...]
    """TODO
    """

    stabilizers: tuple[PauliStr, ...] | None
    """TODO
    """

    def __init__(
        self,
        bitstring: SyndromeCastableTypes,
        stabilizers: Sequence[PauliStr] | None = None,
    ) -> None:
        """TODO"""
        if isinstance(bitstring, Syndrome):
            self.bitstring = bitstring.bitstring
            self.stabilizers = bitstring.stabilizers
        elif isinstance(bitstring, Mapping):
            self.bitstring = tuple(bitstring.values())
            self.stabilizers = tuple(bitstring.keys())
        elif isinstance(bitstring, Sequence):
            self.bitstring = tuple(bitstring)
            self.stabilizers = None
        else:
            raise NotImplementedError(
                f"Cannot create a syndrome from {bitstring}"
            )

        if stabilizers is not None:
            assert len(stabilizers) == len(
                self.bitstring
            ), "Must provide one stabilizer for each bitstring"
            self.stabilizers = tuple(stabilizers)
