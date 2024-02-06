"""Class definition for StabilizerFrame
"""

from typing import Iterable, Optional


class StabilizerFrame:
    """ """

    def __init__(
        self,
        qubit_labels: Iterable[str],
        x_bits: Optional[Iterable[int]] = None,
        z_bits: Optional[Iterable[int]] = None,
    ) -> None:
        """ """
        self.qubit_labels = qubit_labels
        self.num_qubits = len(qubit_labels)

        if x_bits is None:
            self.x_bits = [
                0,
            ] * self.num_qubits
        else:
            assert (
                len(x_bits) == self.num_qubits
            ), "Must provide complete stabilizer frame"
            self.x_bits = x_bits

        if z_bits is None:
            self.z_bits = [
                0,
            ] * self.num_qubits
        else:
            assert (
                len(z_bits) == self.num_qubits
            ), "Must provide complete stabilizer frame"
            self.z_bits = z_bits

    def flip_bit(self, type: str, qubit: str) -> None:
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        bits = self.x_bits if type == "X" else self.z_bits

        new_val = (bits[self.qubit_labels.index[qubit]] + 1) % 2
        bits[self.qubit_labels.index[qubit]] = new_val

    def get_bit(self, type: str, qubit: str) -> int:
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        bits = self.x_bits if type == "X" else self.z_bits

        return bits[self.qubit_labels.index[qubit]]

    def set_bit(self, type: str, qubit: str, bit: int) -> None:
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        bits = self.x_bits if type == "X" else self.z_bits

        bits[self.qubit_labels.index[qubit]] = bit
