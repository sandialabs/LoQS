"""Class definition for StabilizerFrame
"""

from __future__ import annotations

from typing import Sequence, TypeAlias

from loqs.core.recordable import Recordable
from loqs.internal import Bit


StabilizerFrameCastableTypes: TypeAlias = "StabilizerFrame | Sequence[str]"


class StabilizerFrame(Recordable):
    """TODO"""

    qubit_labels: list[str]
    """TODO"""

    num_qubits: int
    """TODO"""

    x_bits: list[Bit]
    """TODO"""

    z_bits: list[Bit]
    """TODO"""

    def __init__(
        self,
        frame_or_labels: StabilizerFrameCastableTypes,
        x_bits: Sequence[Bit] | None = None,
        z_bits: Sequence[Bit] | None = None,
    ) -> None:
        """TODO"""
        if isinstance(frame_or_labels, StabilizerFrame):
            self.qubit_labels = frame_or_labels.qubit_labels
            self.num_qubits = frame_or_labels.num_qubits
            self.x_bits = frame_or_labels.x_bits
            self.z_bits = frame_or_labels.z_bits
        else:
            self.qubit_labels = list(frame_or_labels)
            self.num_qubits = len(frame_or_labels)
            self.x_bits = [
                0,
            ] * self.num_qubits
            self.z_bits = [
                0,
            ] * self.num_qubits

        if x_bits is not None:
            assert (
                len(x_bits) == self.num_qubits
            ), "Must provide complete stabilizer frame"
            self.x_bits = list(x_bits)

        if z_bits is not None:
            assert (
                len(z_bits) == self.num_qubits
            ), "Must provide complete stabilizer frame"
            self.z_bits = list(z_bits)

    def copy(self) -> StabilizerFrame:
        return StabilizerFrame(self.qubit_labels, self.x_bits, self.z_bits)

    def flip_bit(self, type: str, qubit: str) -> None:
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        bits = self.x_bits if type == "X" else self.z_bits

        new_val = (bits[self.qubit_labels.index(qubit)] + 1) % 2
        bits[self.qubit_labels.index(qubit)] = new_val

    def get_bit(self, type: str, qubit: str) -> int:
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        bits = self.x_bits if type == "X" else self.z_bits

        return bits[self.qubit_labels.index(qubit)]

    def set_bit(self, type: str, qubit: str, bit: int) -> None:
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        bits = self.x_bits if type == "X" else self.z_bits

        bits[self.qubit_labels.index(qubit)] = bit
