#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

from __future__ import annotations

from typing import Sequence, TypeAlias, TypeVar

from loqs.internal import SeqCastable, Displayable


U = TypeVar("U", bound="PauliFrame")

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

    SERIALIZE_ATTRS = ["pauli_frame", "qubit_labels"]

    SERIALIZE_ATTRS_MAP = {
        "qubit_labels": "frame_or_labels",
        "pauli_frame": "initial_paulis",
    }

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

    @property
    def num_qubits(self) -> int:
        """Number of qubits tracked by this (PauliFrame)[api:PauliFrame]."""
        return len(self.qubit_labels)

    def copy(self) -> PauliFrame:
        """Return a copy of this (PauliFrame)[api:PauliFrame]."""
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
            The bit value of (pauli_frame)[api:PauliFrame.pauli_frame] in basis ``type`` on qubit ``qubit``
        """
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        pauli = self.pauli_frame[self.qubit_labels.index(qubit)]
        if (type == "X" and pauli in "XY") or (type == "Z" and pauli in "YZ"):
            return 1

        return 0

    def map_frame(self, map: dict) -> PauliFrame:
        """Map every element of the (PauliFrame)[api:PauliFrame].

        Parameters
        ----------
        map:
            A dict with current Paulis as keys and new Paulis
            as values. Both keys and values should be in ``'IXYZ'``.
        """
        new_paulis = [map[P] for P in self.pauli_frame]
        return PauliFrame(self.qubit_labels, new_paulis)

    def update_from_pauli_str(self, pstr: str) -> PauliFrame:
        """Update the (PauliFrame)[api:PauliFrame] by multiplication.

        This is commonly used to update a (PauliFrame)[api:PauliFrame]
        from a correction coming from a lookup table.

        Formally, we are doing :math:`F \rightarrow F P`, where
        :math:`F` is the (pauli_frame)[api:PauliFrame.pauli_frame] and :math:`P` is the
        multi-qubit Pauli represented by ``pstr``.

        Parameters
        ----------
        pstr:
            A Pauli string to be multiplied into (pauli_frame)[api:PauliFrame.pauli_frame].

        Returns
        -------
        PauliFrame
            A copied and updated (PauliFrame)[api:PauliFrame]
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
        """Update the (PauliFrame)[api:PauliFrame] by Clifford conjugation.

        Formally, we are doing :math:`F_i \rightarrow C_i^{-1} F_i C_i`, where
        :math:`F_i` is element :math:`i` of the (pauli_frame)[api:PauliFrame.pauli_frame] and
        :math:`C_i` is element :math:`i` of the ``cliffords``.

        Parameters
        ----------
        cliffords:
            A set of Cliffords to conjugate the elements of
            (pauli_frame)[api:PauliFrame.pauli_frame] element-wise.

        Returns
        -------
        PauliFrame
            A copied and updated (PauliFrame)[api:PauliFrame]
        """
        assert len(cliffords) == len(self.pauli_frame)

        new_frame = self.copy()
        for i, (Pold, C) in enumerate(zip(self.pauli_frame, cliffords)):
            old_to_new = self._clifford_mapping_dict(C)
            new_frame.pauli_frame[i] = old_to_new[Pold]

        return new_frame

    def update_from_transversal_clifford(self, clifford: str) -> PauliFrame:
        """Update the (PauliFrame)[api:PauliFrame] by Clifford conjugation.

        This is commonly used to update a (PauliFrame)[api:PauliFrame]
        after a logical Clifford gate has been applied.

        Formally, we are doing :math:`F_i \rightarrow C^{-1} F_i C`, where
        :math:`F_i` is element :math:`i` of the (pauli_frame)[api:PauliFrame.pauli_frame] and
        :math:`C` is the ``clifford``.

        Parameters
        ----------
        clifford:
           The Clifford to conjugate all elements of
            (pauli_frame)[api:PauliFrame.pauli_frame].

        Returns
        -------
        PauliFrame
            A copied and updated (PauliFrame)[api:PauliFrame]
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
        elif clifford in ["H", "SY", "SYdag"]:
            old_to_new = {"I": "I", "X": "Z", "Y": "Y", "Z": "X"}
        elif clifford in ["S", "Sdag"]:
            old_to_new = {"I": "I", "X": "Y", "Y": "X", "Z": "Z"}
        elif clifford in ["K"]:
            old_to_new = {"I": "I", "X": "Y", "Y": "Z", "Z": "X"}
        else:
            raise NotImplementedError(f"{clifford} is not implemented")

        return old_to_new
