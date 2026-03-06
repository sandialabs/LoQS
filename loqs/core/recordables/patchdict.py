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

from loqs.core import QECCodePatch
from loqs.internal import MapCastable, Displayable


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

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.patches.keys()),
                tuple(hash(p) for p in self.patches.values()),
            )
        )

    @property
    def all_qubit_labels(self) -> list[str | int]:
        """All qubits managed by patches in this :class:`.PatchDict`."""
        qubits: list[str | int] = []
        for patch in self.patches.values():
            qubits.extend(patch.qubits)
        return qubits

    def copy(self) -> PatchDict:
        """Return a copy of this :class:`.PatchDict`.

        Returns
        -------
        PatchDict
            The copied :class:`.PatchDict`
        """
        return PatchDict(self.patches.copy())

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        patches = cls.deserialize(state["patches"], serial_id_to_obj_cache)
        assert isinstance(patches, dict)
        return cls(patches)

    def _to_serialization(
        self, hash_to_serial_id_cache=None, ignore_no_serialize_flags=False
    ) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "patches": self.serialize(
                    self.patches,
                    hash_to_serial_id_cache,
                    ignore_no_serialize_flags,
                )
            }
        )
        return state
