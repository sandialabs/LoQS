"""Class definition for StabilizerFrame
"""

from __future__ import annotations

from typing import Sequence, TypeAlias

from loqs.internal import Castable


StabilizerFrameCastableTypes: TypeAlias = "StabilizerFrame | Sequence[str]"


class StabilizerFrame(Castable):
    """TODO"""

    qubit_labels: list[str]
    """TODO"""

    num_qubits: int
    """TODO"""

    x_bits: list[int]
    """TODO"""

    z_bits: list[int]
    """TODO"""

    def __init__(
        self,
        frame_or_labels: StabilizerFrameCastableTypes,
        x_bits: Sequence[int] | None = None,
        z_bits: Sequence[int] | None = None,
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
            assert all(
                [x in [0, 1] for x in x_bits]
            ), "Bitvalues must be 0 or 1"
            self.x_bits = list(x_bits)

        if z_bits is not None:
            assert (
                len(z_bits) == self.num_qubits
            ), "Must provide complete stabilizer frame"
            assert all(
                [z in [0, 1] for z in z_bits]
            ), "Bitvalues must be 0 or 1"
            self.z_bits = list(z_bits)

    def copy(self) -> StabilizerFrame:
        return StabilizerFrame(self.qubit_labels, self.x_bits, self.z_bits)

    def get_bit(self, type: str, qubit: str) -> int:
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        bits = self.x_bits if type == "X" else self.z_bits

        return bits[self.qubit_labels.index(qubit)]

    def update_from_pauli_str(self, pstr: str) -> StabilizerFrame:
        assert len(pstr) == self.num_qubits

        new_frame = self.copy()
        for i, P in enumerate(pstr):
            if P == "X":
                new_val = (new_frame.x_bits[i] + 1) % 2
                new_frame.x_bits[i] = new_val
            elif P == "Y":
                new_val = (new_frame.x_bits[i] + 1) % 2
                new_frame.x_bits[i] = new_val

                new_val = (new_frame.z_bits[i] + 1) % 2
                new_frame.z_bits[i] = new_val
            elif P == "Z":
                new_val = (new_frame.z_bits[i] + 1) % 2
                new_frame.z_bits[i] = new_val

            # Otherwise we must be I, no action needed

        return new_frame
