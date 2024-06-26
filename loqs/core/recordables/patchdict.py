""":class:`MeasurementOutcomes` definition.
"""

from collections.abc import Iterator, Mapping, MutableMapping
from typing import TypeAlias

from loqs.core import QECCodePatch
from loqs.internal import Castable


PatchDictCastableTypes: TypeAlias = (
    "PatchDict | Mapping[str, QECCodePatch] | None"
)


class PatchDict(MutableMapping[str, QECCodePatch], Castable):
    """TODO"""

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

    @property
    def all_qubit_labels(self) -> list[str]:
        """TODO"""
        qubits = []
        for patch in self.patches.values():
            qubits.extend(patch.qubits)
        return qubits
