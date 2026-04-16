#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`.PatchDict` definition.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from typing import ClassVar, TypeAlias, TypeVar

from loqs.core.recordables.qeccodepatch import QECCodePatch
from loqs.internal import MapCastable, Displayable
from loqs.internal.serializable import Serializable


T = TypeVar("T", bound="PatchDict")

PatchDictCastableTypes: TypeAlias = (
    "PatchDict | Mapping[str, QECCodePatch] | None"
)
"""Objects that can be cast to a :class:`.PatchDict`."""


class PatchDict(MutableMapping[str, QECCodePatch], MapCastable, Displayable):
    """A collection of :class:`.QECCodePatch` objects.

    This is a dict-like object where the keys are patch labels (literally,
    as any ``patch_label`` usage in an :class:`.Instruction` apply function
    refers to these keys) and the values are :class:`.QECCodePatch` objects.

    Unlike many other LoQS objects, this is a mutable object to make it easy
    to manipulate patches. Users should be careful to first use :attr:`.copy`
    to avoid messing up previous :class:`.Frame` objects (or use
    :meth:`.Frame.expire` properly).
    """

    CACHE_ON_SERIALIZE: ClassVar[bool] = True

    SERIALIZE_ATTRS = ["patches"]

    patches: dict[str, QECCodePatch]
    """Underlying dict of patch labels and :class:`.QECCodePatch` objects.
    """

    def __init__(self, patches: PatchDictCastableTypes = None) -> None:
        """
        Parameters
        ----------
        patches:
            See :attr:`.patches`. Defaults to ``None``, which uses
            an empty ``dict``.
        """
        if patches is None:
            patches = {}

        if isinstance(patches, PatchDict):
            self.patches = self.patches
        else:
            assert all([isinstance(k, str) for k in patches.keys()])
            assert all([isinstance(v, QECCodePatch) for v in patches.values()])
            self.patches = {k: v for k, v in patches.items()}

    def __getitem__(self, key: str) -> QECCodePatch:
        return self.patches[key]

    def __len__(self) -> int:
        return len(self.patches)

    def __iter__(self) -> Iterator[str]:
        return iter(self.patches)

    def __setitem__(self, key: str, value: QECCodePatch) -> None:
        self.patches[key] = value

    def __delitem__(self, key: str) -> None:
        del self.patches[key]

    def __str__(self) -> str:
        str_dict = {k: str(v) for k, v in self.patches.items()}
        return f"PatchDict({str_dict})"

    @property
    def all_qubit_labels(self) -> list[str | int]:
        """All qubits managed by patches in this (PatchDict)[api:PatchDict]."""
        qubits: list[str | int] = []
        for patch in self.patches.values():
            qubits.extend(patch.qubits)
        return qubits

    def copy(self) -> PatchDict:
        """Return a copy of this (PatchDict)[api:PatchDict].

        Returns
        -------
        PatchDict
            The copied (PatchDict)[api:PatchDict]
        """
        return PatchDict(self.patches.copy())
