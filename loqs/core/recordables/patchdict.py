""":class:`MeasurementOutcomes` definition.
"""

from collections.abc import Mapping
from typing import TypeAlias

from loqs.core import QECCodePatch, Recordable


PatchDictCastableTypes: TypeAlias = (
    "PatchDict | Mapping[str, QECCodePatch] | None"
)


class PatchDict(Recordable):
    """TODO"""

    patches: dict[str, QECCodePatch]
    """TODO
    """

    def __init__(self, patches: PatchDictCastableTypes) -> None:
        """TODO"""
        if patches is None:
            patches = {}

        if isinstance(patches, PatchDict):
            self.patches = self.patches
        else:
            assert all([isinstance(k, str) for k in patches.keys()])
            assert all([isinstance(v, QECCodePatch) for v in patches.values()])
            self.patches = {k: v for k, v in patches.items()}

    @property
    def all_qubit_labels(self) -> list[str]:
        """TODO"""
        qubits = []
        for patch in self.patches.values():
            qubits.extend(patch.qubits)
        return qubits
