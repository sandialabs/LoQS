""":class:`MeasurementOutcomes` definition.
"""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from loqs.core import Recordable


PatchesCastableTypes: TypeAlias = "PatchDict | Mapping[str, Sequence[int]]"


class PatchDict(Recordable):
    """TODO"""

    patches: dict[str, list[int]]
    """TODO
    """

    def __init__(self, patches: PatchesCastableTypes) -> None:
        """TODO"""
        if isinstance(patches, PatchDict):
            self.patches = self.patches
        else:
            self.patches = {k: list(v) for k, v in patches.items()}
