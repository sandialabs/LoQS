"""TODO
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

from loqs.internal import Castable, SeqCastable, Displayable

T = TypeVar("T", bound="SyndromeLabel")
U = TypeVar("U", bound="PauliFrame")


SyndromeLabelCastableTypes: TypeAlias = (
    "str | tuple[str] | tuple[str, int] | tuple[str, int, int] | SyndromeLabel"
)
"""Objects that can be cast to :class:`.SyndromeLabel` objects."""


@dataclass
class SyndromeLabel(Castable, Displayable):
    """Label that indicates which past outcome was a syndrome bit."""

    qubit_label: str
    """The qubit label."""

    frame_idx: int = -1
    """The frame index.

    Defaults to -1, i.e. the previous frame.
    """

    outcome_idx: int = 0
    """The outcome index.

    Defaults to 0, the first outcome on :attr:`.qubit_label`.
    Could be >0 if multiple checks were measured on :attr:`.qubit_label`.
    """

    def __hash__(self) -> int:
        return hash((self.qubit_label, self.frame_idx, self.outcome_idx))

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        qubit_label = state["qubit_label"]
        frame_idx = state["frame_idx"]
        outcome_idx = state["outcome_idx"]
        return cls(qubit_label, frame_idx, outcome_idx)

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
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
PauliFrameCastableTypes: TypeAlias = "PauliFrame | Sequence[str | int]"
"""Types that can be cast into a :class:`.PauliFrame`."""


class PauliFrame(SeqCastable, Displayable):
    """Tracks a Pauli frame on a set of qubits.

    Commonly this is used to track data errors without applying
    active correction, and can be used in conjunction with
    :class:`.MeasurementOutcomes` to provide inferred outcomes.
    """

    qubit_labels: list[str | int]
    """Qubit labels being tracked by this :class:`.PauliFrame`."""

    pauli_frame: list[str]
    """A list of Pauli errors on the given :attr:`.qubit_labels`."""

    def __init__(
        self,
        frame_or_labels: PauliFrameCastableTypes,
        initial_paulis: Sequence[str] | str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        frame_or_labels:
            Either an existing :class:`.PauliFrame` or a set of
            qubit labels to be used for :attr:`.qubit_labels`.
            If qubit labels and no ``initial_paulis``, the frame
            is initialized to all ``"I"``.

        initial_paulis:
            An initial set of Pauli corrections to track. If
            ``frame_or_labels`` was an existing :class:`.PauliFrame`,
            then ``initial_paulis`` will override the copied value of
            :attr:`.pauli_frame`.
        """
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

    def __hash__(self) -> int:
        return hash((tuple(self.qubit_labels), "".join(self.pauli_frame)))

    @property
    def num_qubits(self) -> int:
        """Number of qubits tracked by this :class:`.PauliFrame`."""
        return len(self.qubit_labels)

    def copy(self) -> PauliFrame:
        """Return a copy of this :class:`.PauliFrame`."""
        return PauliFrame(self.qubit_labels, self.pauli_frame)

    def get_bit(self, type: str, qubit: str | int) -> int:
        """Get the bit value of this frame on a given qubit in a given basis.

        Parameters
        ----------
        type:
            One of ``["X", "Z"]``, indicating which basis to return.

        qubit:
            The qubit label to check.

        Returns
        -------
        int
            The bit value of :attr:`.pauli_frame` in basis ``type`` on qubit ``qubit``
        """
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        pauli = self.pauli_frame[self.qubit_labels.index(qubit)]
        if (type == "X" and pauli in "XY") or (type == "Z" and pauli in "YZ"):
            return 1

        return 0

    def map_frame(self, map: dict) -> PauliFrame:
        """Map every element of the :class:`.PauliFrame`.

        Parameters
        ----------
        map:
            A dict with current Paulis as keys and new Paulis
            as values. Both keys and values should be in ``'IXYZ'``.
        """
        new_paulis = [map[P] for P in self.pauli_frame]
        return PauliFrame(self.qubit_labels, new_paulis)

    def update_from_pauli_str(self, pstr: str) -> PauliFrame:
        """Update the :class:`.PauliFrame` by multiplication.

        This is commonly used to update a :class:`.PauliFrame`
        from a correction coming from a lookup table.

        Formally, we are doing :math:`F \\rightarrow F P`, where
        :math:`F` is the :attr:`.pauli_frame` and :math:`P` is the
        multi-qubit Pauli represented by ``pstr``.

        Parameters
        ----------
        pstr:
            A Pauli string to be multiplied into :attr:`.pauli_frame`.

        Returns
        -------
        :class:`.PauliFrame`
            A copied and updated :class:`.PauliFrame`
        """
        assert len(pstr) == self.num_qubits

        new_frame = self.copy()
        for i, (Pold, P) in enumerate(zip(self.pauli_frame, pstr)):
            old_to_new = self._pauli_product_mapping_dict(P)
            new_frame.pauli_frame[i] = old_to_new[Pold]

        return new_frame

    def update_from_clifford_conjugation(
        self, cliffords: Sequence[str]
    ) -> PauliFrame:
        """Update the :class:`.PauliFrame` by Clifford conjugation.

        Formally, we are doing :math:`F_i \\rightarrow C_i^{-1} F_i C_i`, where
        :math:`F_i` is element :math:`i` of the :attr:`.pauli_frame` and
        :math:`C_i` is element :math:`i` of the ``cliffords``.

        Parameters
        ----------
        cliffords:
            A set of Cliffords to conjugate the elements of
            :attr:`.pauli_frame` element-wise.

        Returns
        -------
        :class:`.PauliFrame`
            A copied and updated :class:`.PauliFrame`
        """
        assert len(cliffords) == len(self.pauli_frame)

        new_frame = self.copy()
        for i, (Pold, C) in enumerate(zip(self.pauli_frame, cliffords)):
            old_to_new = self._clifford_mapping_dict(C)
            new_frame.pauli_frame[i] = old_to_new[Pold]

        return new_frame

    def update_from_transversal_clifford(self, clifford: str) -> PauliFrame:
        """Update the :class:`.PauliFrame` by Clifford conjugation.

        This is commonly used to update a :class:`.PauliFrame`
        after a logical Clifford gate has been applied.

        Formally, we are doing :math:`F_i \\rightarrow C^{-1} F_i C`, where
        :math:`F_i` is element :math:`i` of the :attr:`.pauli_frame` and
        :math:`C` is the ``clifford``.

        Parameters
        ----------
        clifford:
           The Clifford to conjugate all elements of
            :attr:`.pauli_frame`.

        Returns
        -------
        :class:`.PauliFrame`
            A copied and updated :class:`.PauliFrame`
        """
        old_to_new = self._clifford_mapping_dict(clifford)
        return self.map_frame(old_to_new)

    def _pauli_product_mapping_dict(self, pauli: str) -> dict[str, str]:
        if pauli == "I":
            old_to_new = {k: k for k in "IXYZ"}
        elif pauli == "X":
            old_to_new = {"I": "X", "X": "I", "Y": "Z", "Z": "Y"}
        elif pauli == "Y":
            old_to_new = {"I": "Y", "X": "Z", "Y": "I", "Z": "X"}
        elif pauli == "Z":
            old_to_new = {"I": "Z", "X": "Y", "Y": "X", "Z": "I"}
        else:
            raise NotImplementedError(f"{pauli} is not a Pauli")

        return old_to_new

    def _clifford_mapping_dict(self, clifford: str) -> dict[str, str]:
        if clifford in ["I", "X", "Y", "Z"]:
            old_to_new = {k: k for k in "IXYZ"}
        elif clifford == "H":
            old_to_new = {"I": "I", "X": "Z", "Y": "Y", "Z": "X"}
        elif clifford in ["S", "Sdag"]:
            old_to_new = {"I": "I", "X": "Y", "Y": "X", "Z": "Z"}
        elif clifford in ["K"]:
            old_to_new = {"I": "I", "X": "Y", "Y": "Z", "Z": "X"}
        else:
            raise NotImplementedError(f"{clifford} is not implemented")

        return old_to_new

    @classmethod
    def _from_serialization(
        cls: type[U], state: Mapping, serial_id_to_obj_cache=None
    ) -> U:
        qubit_labels = state["qubit_labels"]
        pauli_frame = list(state["pauli_frame"])
        return cls(qubit_labels, pauli_frame)

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "qubit_labels": self.qubit_labels,
                "pauli_frame": "".join(self.pauli_frame),
            }
        )
        return state
