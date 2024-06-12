"""TODO
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import TypeAlias

from loqs.core import Instruction, HistoryStack, HistoryFrame
from loqs.core.instruction import InstructionParentTypes
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.recordables import PatchDict


PermutePatchCastable: TypeAlias = "PermutePatch | Mapping[str, str]"


class PermutePatch(Instruction):
    """TODO"""

    def __init__(
        self,
        mapping: PermutePatchCastable,
        name: str = "(Unnamed patch permutation)",
        parent: InstructionParentTypes = None,
    ) -> None:
        """TODO

        Parameters
        ----------
        """
        if isinstance(mapping, PermutePatch):
            self.mapping = mapping.mapping
        elif isinstance(mapping, Mapping):
            self.mapping = {k: v for k, v in mapping.items()}
        else:
            raise NotImplementedError(
                "Cannot create PermutePatch from given mapping"
            )

        super().__init__(name, parent)

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return {"patches": PatchDict}

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {"patches": PatchDict, "instruction": Instruction}

    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
        """Map the input :class:`MockState` forward.

        This

        Parameters
        ----------
        input:
            The input frame/history information

        Returns
        -------
        output_frame:
            The new output frame
        """
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        patches = PatchDict.cast(last_frame["patches"])

        # Iterate through patches and apply mapping?

        new_data = {
            "patches": patches,
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame

    def map_qubits(self, qubit_mapping: Mapping[str, str]) -> PermutePatch:
        new_mapping = {
            qubit_mapping[k]: qubit_mapping[v] for k, v in self.mapping.items()
        }
        return PermutePatch(new_mapping, self.name, self.parent)
