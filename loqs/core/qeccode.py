#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################



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
    """A set of [](api:Instruction) objects that implement a QEC code.

    All qubit-specific quantities are defined with respect to a set of
    template qubits that can then be replaced with real qubit labels
    at runtime.
    """

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    _SERIALIZE_ATTRS = [
        "instructions",
        "template_qubits",
        "template_data_qubits",
        "name",
    ]

    instructions: dict[str, Instruction]
    """A mapping from name keys to [Instruction](api:Instruction) values."""

    template_qubits: list[str | int]
    """All template qubits used in [instructions](api:QECCode.instructions)."""

    template_data_qubits: list[str | int]
    """The entries of [template_qubits](api:QECCode.template_qubits) corresponding to data qubits."""

    name: str = "(Unnamed QEC code)"
    """Name for logging"""

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
            See [instructions](api:QECCode.instructions).

        template_qubits:
            See [template_qubits](api:QECCode.template_qubits).

        template_data_qubits:
        See [template_data_qubits](api:QECCode.template_data_qubits).

        name:
            See [name](api:QECCode.name).
        """
        self.instructions = dict(instructions)

        self.template_qubits = list(template_qubits)

        self.template_data_qubits = list(template_data_qubits)

        assert all(
            [tdq in self.template_qubits for tdq in self.template_data_qubits]
        ), "Data qubits must a subset of all template qubits"

        self.name = name

    def __str__(self) -> str:
        return f"QECCode {self.name}"

    def create_patch(
        self,
        qubits: Sequence[str | int],
        pauli_frame: PauliFrameCastableTypes | None = None,
    ) -> QECCodePatch:
        """Create a [QECCodePatch](api:QECCodePatch) based on this [QECCode](api:QECCode).

        Parameters
        ----------
        qubits:
            Qubit labels to replace [template_qubits](api:QECCode.template_qubits).

        pauli_frame:
            An initial [PauliFrame](api:PauliFrame) to assign to the patch.
            Defaults to `None`, which assigns the trivial Pauli frame
            of all `"I"` entries.

        Returns
        -------
        QECCodePatch
            The constructed [QECCodePatch](api:QECCodePatch)
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
    def _from_decoded_attrs(cls, attr_dict) -> "QECCode":
        """Create a QECCode from decoded attributes dictionary."""
        return cls(
            attr_dict["instructions"],
            attr_dict["template_qubits"],
            attr_dict["template_data_qubits"],
            name=attr_dict.get("name", "(Unnamed QEC code)"),
        )
