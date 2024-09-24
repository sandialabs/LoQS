"""TODO
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TypeAlias

from loqs.core.instructions import Instruction
from loqs.core.syndrome import PauliFrame, PauliFrameCastableTypes
from loqs.internal import Castable


class QECCode:
    """TODO"""

    def __init__(
        self,
        instructions: Mapping[str, Instruction],
        template_qubits: Sequence[str],
        template_data_qubits: Sequence[str],
        name: str = "(Unnamed QEC code)",
    ):
        """TODO"""
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
        qubits: Sequence[str],
        pauli_frame: PauliFrameCastableTypes | None = None,
    ):
        """TODO"""
        if pauli_frame is None:
            # Map template data qubits to real qubits
            data_qubits = [
                qubits[self.template_qubits.index(tdq)]
                for tdq in self.template_data_qubits
            ]
            # Initialize an empty PauliFrame on our data qubits
            pauli_frame = data_qubits
        return QECCodePatch(self, qubits, pauli_frame)


class QECCodePatch(Mapping[str, Instruction]):
    """TODO"""

    def __init__(
        self,
        code: QECCode,
        qubits: Sequence[str],
        pauli_frame: PauliFrameCastableTypes,
    ):
        """TODO"""
        assert len(qubits) == len(code.template_qubits), (
            f"Patch must have {len(code.template_qubits)} qubits "
            + f"to match code {code}, not {len(qubits)}"
        )

        self.code = code
        self.qubits = qubits
        self.pauli_frame = PauliFrame.cast(pauli_frame)

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
        s += f"[{self.qubits[0]},...,{self.qubits[-1]}]"
        return s
