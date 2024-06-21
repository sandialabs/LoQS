"""TODO
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from loqs.core import Instruction, HistoryStack, HistoryFrame
from loqs.core.instruction import InstructionParentTypes
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.qeccode import QECCode
from loqs.core.recordables import PatchDict


PatchBuilderCastableTypes: TypeAlias = (
    "PatchBuilder | tuple[str, type[QECCode]] | tuple[str, type[QECCode], str]"
)

PermutePatchCastable: TypeAlias = "PermutePatch | Mapping[str, str]"


class PatchBuilder(Instruction):
    """TODO"""

    def __init__(
        self,
        qec_code: QECCode,
        patch_key: str = "patches",
        name: str = "(Unnamed patch builder)",
        parent: InstructionParentTypes = None,
        fault_tolerant: bool | None = None,
    ) -> None:
        """TODO

        Parameters
        ----------
        """
        super().__init__(name, parent, fault_tolerant)

        if self.fault_tolerant is None:
            # Assume FT is True unless explicitly set to False
            self.fault_tolerant = True

        self.patch_key = patch_key

        assert isinstance(
            qec_code, QECCode
        ), "PatchBuilder should only build from QECCodes"
        self.qec_code = qec_code

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return {}

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {self.patch_key: PatchDict, "instruction": Instruction}

    def apply_unsafe(
        self,
        input: HistoryStackCastableTypes,
        patch_name: str,
        qubits: Sequence[str],
    ) -> HistoryFrame:
        """TODO"""
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        patches = PatchDict.cast(last_frame.get(self.patch_key, None))
        all_patch_qubits = patches.all_qubit_labels

        # Disjoint patch checks
        assert all(
            [q not in all_patch_qubits for q in qubits]
        ), f"PatchBuilder failed, requesting overlapping patches for {patch_name}"
        assert (
            patch_name not in patches
        ), f"PatchBuilder failed, already have existing patch {patch_name}"

        try:
            patch = self.qec_code.create_patch(qubits)
        except Exception as e:
            raise ValueError("Failed to create patch in PatchBuilder") from e

        patches[patch_name] = patch

        new_data = {
            self.patch_key: patches,
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame


class PatchRemover(Instruction):
    """TODO"""

    def __init__(
        self,
        patch_key: str = "patches",
        name: str = "(Unnamed patch remover)",
        parent: InstructionParentTypes = None,
        fault_tolerant: bool | None = None,
    ) -> None:
        """TODO

        Parameters
        ----------
        """
        super().__init__(name, parent, fault_tolerant)

        if self.fault_tolerant is None:
            # Assume FT is True unless explicitly set to False
            self.fault_tolerant = True

        self.patch_key = patch_key

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return {self.patch_key: PatchDict}

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {self.patch_key: PatchDict, "instruction": Instruction}

    def apply_unsafe(
        self, input: HistoryStackCastableTypes, patch_name: str
    ) -> HistoryFrame:
        """TODO"""
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        patches = PatchDict.cast(last_frame.get(self.patch_key, None))
        assert (
            patch_name in patches
        ), f"PatchRemover failed, could not find patch {patch_name}"

        del patches[patch_name]

        new_data = {
            self.patch_key: patches,
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame


class PermutePatch(Instruction):
    """TODO"""

    def __init__(
        self,
        mapping: PermutePatchCastable,
        name: str = "(Unnamed patch permutation)",
        parent: InstructionParentTypes = None,
        fault_tolerant: bool | None = None,
    ) -> None:
        """TODO

        Parameters
        ----------
        """
        super().__init__(name, parent, fault_tolerant)

        if self.fault_tolerant is None:
            # Assume FT unless explicitly set to False
            self.fault_tolerant = True

        if isinstance(mapping, PermutePatch):
            self.mapping = mapping.mapping
        elif isinstance(mapping, Mapping):
            self.mapping = {k: v for k, v in mapping.items()}
        else:
            raise ValueError("Cannot create PermutePatch from given mapping")

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
