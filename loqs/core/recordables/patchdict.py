""":class:`MeasurementOutcomes` definition.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from typing import ClassVar, TypeAlias, TypeVar

from loqs.core import QECCodePatch
from loqs.internal import Castable, Displayable


T = TypeVar("T", bound="PatchDict")

PatchDictCastableTypes: TypeAlias = (
    "PatchDict | Mapping[str, QECCodePatch] | None"
)


class PatchDict(MutableMapping[str, QECCodePatch], Castable, Displayable):
    """TODO"""

    CACHE_ON_SERIALIZE: ClassVar[bool] = True

    patches: dict[str, QECCodePatch]
    """TODO
    """

    def __init__(self, patches: PatchDictCastableTypes = None) -> None:
        """TODO"""
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
    def all_qubit_labels(self) -> list[str]:
        """TODO"""
        qubits: list[str] = []
        for patch in self.patches.values():
            qubits.extend(patch.qubits)
        return qubits

    def copy(self) -> PatchDict:
        return PatchDict(self.patches.copy())

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        patches = cls.deserialize(state["patches"], serial_id_to_obj_cache)
        assert isinstance(patches, dict)
        return cls(patches)

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()
        state.update(
            {"patches": self.serialize(self.patches, hash_to_serial_id_cache)}
        )
        return state
