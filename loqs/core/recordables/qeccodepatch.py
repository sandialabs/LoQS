#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, ClassVar, Sequence, TypeVar

from loqs.core.instructions.instruction import Instruction
from loqs.core.recordables.pauliframe import (
    PauliFrame,
    PauliFrameCastableTypes,
)
from loqs.internal.displayable import Displayable
from loqs.internal.serializable import Serializable

if TYPE_CHECKING:
    # Can import for typechecking without circular import issue
    from loqs.core import QECCode

U = TypeVar("U", bound="QECCodePatch")


class QECCodePatch(Mapping[str, Instruction], Displayable):
    """An instantiation of a [](api:QECCode) on a set of qubits.

    This object acts like a `dict`, where instruction names are the
    keys and the appropriate [](api:Instruction) (mapped to the patch
    qubits) is returned.
    It also stores the [](api:PauliFrame) for the data qubits, as this
    is the natural place for it.
    """

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    _SERIALIZE_ATTRS = ["code", "qubits", "pauli_frame", "data"]

    def __init__(
        self,
        code: "QECCode",  # to avoid circular import
        qubits: Sequence[str | int],
        pauli_frame: PauliFrameCastableTypes,
    ):
        """
        Parameters
        ----------
        code:
            See :attr:`.code`.

        qubits:
            See :attr:`.qubits`.

        pauli_frame:
            See :attr:`.pauli_frame`.
        """
        assert len(qubits) == len(code.template_qubits), (
            f"Patch must have {len(code.template_qubits)} qubits "
            + f"to match code {code}, not {len(qubits)}"
        )

        self.code = code
        """The [](api:QECCode) being used on this patch of qubits."""

        self.qubits = qubits
        """The qubits this patch acts on."""

        self.pauli_frame = PauliFrame.cast(pauli_frame)
        """The Pauli frame tracking errors on these qubits."""

        self.data = {}
        """Extra patch-specific data to be tracked."""

    def __getitem__(self, key: str) -> Instruction:
        try:
            template_op = self.code.instructions[key]
        except KeyError:
            raise KeyError(
                f"Operation {key} not available in code {self.code}"
            )

        mapping = {
            k: v for k, v in zip(self.code.template_qubits, self.qubits)
        }
        return template_op.map_qubits(mapping)

    def __len__(self) -> int:
        return len(self.code.instructions)

    def __iter__(self) -> Iterator[str]:
        return iter(self.code.instructions)

    def __str__(self) -> str:
        s = f"QECCodePatch for {self.code.name} on qubits "
        s += f"[{self.qubits[0]},...,{self.qubits[-1]}]" + "\n"
        s += f"  Current frame: {self.pauli_frame.pauli_frame}"
        return s

    @classmethod
    def _from_decoded_attrs(cls, attr_dict) -> "QECCodePatch":
        """Create a QECCodePatch from decoded attributes dictionary."""
        from loqs.core import QECCode

        code = attr_dict["code"]
        if not isinstance(code, QECCode):
            raise ValueError(f"Expected QECCode, got {type(code)}")

        qubits = attr_dict["qubits"]
        pauli_frame = attr_dict["pauli_frame"]

        obj = cls(code, qubits, pauli_frame)
        obj.data = attr_dict.get("data", {})
        return obj
