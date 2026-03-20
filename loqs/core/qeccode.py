#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`.QECCode` and :class:`.QECCodePatch` definitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, TypeVar

from loqs.core.instructions import Instruction
from loqs.core.recordables.pauliframe import PauliFrameCastableTypes
from loqs.core.recordables.qeccodepatch import QECCodePatch
from loqs.internal import Displayable
from loqs.internal.serializable import Serializable


T = TypeVar("T", bound="QECCode")


class QECCode(Displayable):
    """A set of :class:`.Instruction` objects that implement a QEC code.

    All qubit-specific quantities are defined with respect to a set of
    template qubits that can then be replaced with real qubit labels
    at runtime.
    """

    CACHE_ON_SERIALIZE: ClassVar[bool] = True

    SERIALIZE_ATTRS = [
        "instructions",
        "template_qubits",
        "template_data_qubits",
        "name",
    ]

    def __init__(
        self,
        instructions: Mapping[str, Instruction],
        template_qubits: Sequence[str | int],
        template_data_qubits: Sequence[str | int],
        name: str = "(Unnamed QEC code)",
    ):
        """
        Parameters
        ----------
        instructions:
            See :attr:`.instructions`.

        template_qubits:
            See :attr:`.template_qubits`.

        template_data_qubits:
            See :attr:`.template_data_qubits`.

        name:
            See :attr:`.name`.
        """
        self.instructions = dict(instructions)
        """A mapping from name keys to :class:`.Instruction` values."""

        self.template_qubits = list(template_qubits)
        """All template qubits used in :attr:`instructions`."""

        self.template_data_qubits = list(template_data_qubits)
        """The entries of :attr:`template_qubits` corresponding to data qubits."""

        assert all(
            [tdq in self.template_qubits for tdq in self.template_data_qubits]
        ), "Data qubits must a subset of all template qubits"

        self.name = name
        """Name for logging"""

    def __str__(self) -> str:
        return f"QECCode {self.name}"

    def create_patch(
        self,
        qubits: Sequence[str | int],
        pauli_frame: PauliFrameCastableTypes | None = None,
    ) -> QECCodePatch:
        """Create a :class:`.QECCodePatch` based on this :class:`QECCode`.

        Parameters
        ----------
        qubits:
            Qubit labels to replace :attr:`.template_qubits`.

        pauli_frame:
            An initial :class:`.PauliFrame` to assign to the patch.
            Defaults to ``None``, which assigns the trivial Pauli frame
            of all ``"I"`` entries.

        Returns
        -------
        QECCodePatch
            The constructed :class:`.QECCodePatch`
        """
        if pauli_frame is None:
            # Map template data qubits to real qubits
            data_qubits = [
                qubits[self.template_qubits.index(tdq)]
                for tdq in self.template_data_qubits
            ]
            # Initialize an empty PauliFrame on our data qubits
            pauli_frame = data_qubits
        return QECCodePatch(self, qubits, pauli_frame)

    @classmethod
    def from_decoded_attrs(cls, attr_dict) -> "QECCode":
        """Create a QECCode from decoded attributes dictionary."""
        return cls(
            attr_dict["instructions"],
            attr_dict["template_qubits"],
            attr_dict["template_data_qubits"],
            name=attr_dict.get("name", "(Unnamed QEC code)"),
        )
