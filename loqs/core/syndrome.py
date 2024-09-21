"""TODO
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from loqs.core.instructions import Instruction
from loqs.internal import Castable

SyndromeLabelCastableTypes: TypeAlias = (
    "str | tuple[str] | tuple[str, int] | tuple[str, int, int] | SyndromeLabel"
)


@dataclass
class SyndromeLabel(Castable):
    """Label that indicates which past outcome was a syndrome bit."""

    qubit_label: str
    """The qubit label."""

    frame_idx: int = -1
    """The frame index.

    Defaults to -1, i.e. the previous frame.
    """

    outcome_idx: int = 0
    """The outcome index.

    Defaults to 0, the first outcome on `qubit_label`.
    Could be >0 if multiple checks were measured on `qubit_label`.
    """

    @classmethod
    def cast(cls: SyndromeLabel, obj: object) -> SyndromeLabel:
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, dict):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        elif isinstance(obj, (tuple, list)):
            assert len(obj) < 4
            return cls(*obj)

        # Otherwise, assume this is the first __init__ argument
        return cls(obj)


# TODO: Long term location?
PauliFrameCastableTypes: TypeAlias = "PauliFrame | Sequence[str]"
PauliLiterals: TypeAlias = (
    Literal["I"] | Literal["X"] | Literal["Y"] | Literal["Z"]
)


class PauliFrame(Castable):
    """TODO"""

    qubit_labels: list[str]
    """TODO"""

    pauli_frame: list[PauliLiterals]
    """TODO"""

    def __init__(
        self,
        frame_or_labels: PauliFrameCastableTypes,
        initial_paulis: Sequence[PauliLiterals] | None = None,
    ) -> None:
        """TODO"""
        if isinstance(frame_or_labels, PauliFrame):
            self.qubit_labels = frame_or_labels.qubit_labels
            self.x_bits = frame_or_labels.x_bits
            self.z_bits = frame_or_labels.z_bits
        else:
            self.qubit_labels = list(frame_or_labels)
            self.pauli_frame = ["I"] * self.num_qubits

        if initial_paulis is not None:
            assert (
                len(initial_paulis) == self.num_qubits
            ), "Must provide complete initial pauli frame"

    def __str__(self) -> str:
        s = f"PauliFrame on [{self.qubit_labels[0]},...,{self.qubit_labels[-1]}] qubits:\n"
        s += f"  Paulis: {self.pauli_frame}"
        return s

    @property
    def num_qubits(self) -> int:
        return len(self.qubit_labels)

    def copy(self) -> PauliFrame:
        return PauliFrame(self.qubit_labels, self.pauli_frame)

    def get_bit(self, type: str, qubit: str) -> int:
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        pauli = self.pauli_frame[self.qubit_labels.index(qubit)]
        if (type == "X" and pauli in "XY") or (type == "Z" and pauli in "YZ"):
            return 1

        return 0

    def update_from_pauli_str(
        self, pstr: Sequence[PauliLiterals]
    ) -> PauliFrame:
        assert len(pstr) == self.num_qubits

        new_frame = self.copy()
        for i, (Pold, P) in enumerate(self.pauli_frame, pstr):
            if P == "X":
                old_to_new = {"I": "X", "X": "I", "Y": "Z", "Z": "Y"}
            elif P == "Y":
                old_to_new = {"I": "Y", "X": "Z", "Y": "I", "Z": "X"}
            elif P == "Z":
                old_to_new = {"I": "Z", "X": "Y", "Y": "X", "Z": "I"}
            else:
                old_to_new = {k: k for k in "IXYZ"}

            new_frame.pauli_frame[i] = old_to_new[Pold]

        return new_frame
