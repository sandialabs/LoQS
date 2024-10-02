"""TODO
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

from loqs.internal import Castable, Serializable

T = TypeVar("T", bound="SyndromeLabel")
U = TypeVar("U", bound="PauliFrame")


SyndromeLabelCastableTypes: TypeAlias = (
    "str | tuple[str] | tuple[str, int] | tuple[str, int, int] | SyndromeLabel"
)


@dataclass
class SyndromeLabel(Castable, Serializable):
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
    def cast(cls: type[SyndromeLabel], obj: object) -> SyndromeLabel:
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, dict):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        elif isinstance(obj, (tuple, list)):
            assert len(obj) < 4
            return cls(*obj)
        elif isinstance(obj, str):
            return cls(obj)

        raise ValueError(f"Cannot cast {obj} to a SyndromeLabel")

    @classmethod
    def _from_serialization(cls: type[T], state: Mapping) -> T:
        qubit_label = state["qubit_label"]
        frame_idx = state["frame_idx"]
        outcome_idx = state["outcome_idx"]
        return cls(qubit_label, frame_idx, outcome_idx)

    def _to_serialization(self) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "qubit_label": self.qubit_label,
                "frame_idx": self.frame_idx,
                "outcome_idx": self.outcome_idx,
            }
        )
        return state


# TODO: Long term location?
PauliFrameCastableTypes: TypeAlias = "PauliFrame | Sequence[str]"


class PauliFrame(Castable, Serializable):
    """TODO"""

    qubit_labels: list[str]
    """TODO"""

    pauli_frame: list[str]
    """TODO"""

    def __init__(
        self,
        frame_or_labels: PauliFrameCastableTypes,
        initial_paulis: Sequence[str] | str | None = None,
    ) -> None:
        """TODO"""
        if isinstance(frame_or_labels, PauliFrame):
            self.qubit_labels = frame_or_labels.qubit_labels
            self.pauli_frame = frame_or_labels.pauli_frame
        else:
            self.qubit_labels = list(frame_or_labels)
            self.pauli_frame = ["I"] * self.num_qubits

        if initial_paulis is not None:
            assert (
                len(initial_paulis) == self.num_qubits
            ), "Must provide complete initial pauli frame"
            assert all([ip in "IXYZ" for ip in initial_paulis])

            self.pauli_frame = list(initial_paulis)

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

    def map_frame(self, map: dict) -> PauliFrame:
        new_paulis = [map[P] for P in self.pauli_frame]
        return PauliFrame(self.qubit_labels, new_paulis)

    def update_from_pauli_str(self, pstr: str) -> PauliFrame:
        assert len(pstr) == self.num_qubits

        new_frame = self.copy()
        for i, (Pold, P) in enumerate(zip(self.pauli_frame, pstr)):
            old_to_new = self._clifford_mapping_dict(P)
            new_frame.pauli_frame[i] = old_to_new[Pold]

        return new_frame

    def update_from_transversal_clifford(self, clifford: str) -> PauliFrame:
        old_to_new = self._clifford_mapping_dict(clifford)
        return self.map_frame(old_to_new)

    def _clifford_mapping_dict(self, clifford: str) -> dict[str, str]:
        if clifford == "I":
            old_to_new = {k: k for k in "IXYZ"}
        elif clifford == "X":
            old_to_new = {"I": "X", "X": "I", "Y": "Z", "Z": "Y"}
        elif clifford == "Y":
            old_to_new = {"I": "Y", "X": "Z", "Y": "I", "Z": "X"}
        elif clifford == "Z":
            old_to_new = {"I": "Z", "X": "Y", "Y": "X", "Z": "I"}
        elif clifford == "H":
            old_to_new = {"I": "I", "X": "Z", "Y": "Y", "Z": "X"}
        elif clifford in ["S", "Sdag"]:
            old_to_new = {"I": "I", "X": "Y", "Y": "X", "Z": "Z"}
        else:
            raise NotImplementedError(f"{clifford} is not implemented")

        return old_to_new

    @classmethod
    def _from_serialization(cls: type[U], state: Mapping) -> U:
        qubit_labels = state["qubit_labels"]
        pauli_frame = list(state["pauli_frame"])
        return cls(qubit_labels, pauli_frame)

    def _to_serialization(self) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "qubit_labels": self.qubit_labels,
                "pauli_frame": "".join(self.pauli_frame),
            }
        )
        return state
