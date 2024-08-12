"""TODO
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TypeAlias

from loqs.core.instructions import Instruction
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


PauliFrameCastableTypes: TypeAlias = "PauliFrame | Sequence[str]"


# TODO: Find long-term place for this
class PauliFrame(Castable):
    """TODO"""

    qubit_labels: list[str]
    """TODO"""

    num_qubits: int
    """TODO"""

    x_bits: list[int]
    """TODO"""

    z_bits: list[int]
    """TODO"""

    def __init__(
        self,
        frame_or_labels: PauliFrameCastableTypes,
        x_bits: Sequence[int] | None = None,
        z_bits: Sequence[int] | None = None,
    ) -> None:
        """TODO"""
        if isinstance(frame_or_labels, PauliFrame):
            self.qubit_labels = frame_or_labels.qubit_labels
            self.num_qubits = frame_or_labels.num_qubits
            self.x_bits = frame_or_labels.x_bits
            self.z_bits = frame_or_labels.z_bits
        else:
            self.qubit_labels = list(frame_or_labels)
            self.num_qubits = len(frame_or_labels)
            self.x_bits = [
                0,
            ] * self.num_qubits
            self.z_bits = [
                0,
            ] * self.num_qubits

        if x_bits is not None:
            assert (
                len(x_bits) == self.num_qubits
            ), "Must provide complete stabilizer frame"
            assert all(
                [x in [0, 1] for x in x_bits]
            ), "Bitvalues must be 0 or 1"
            self.x_bits = list(x_bits)

        if z_bits is not None:
            assert (
                len(z_bits) == self.num_qubits
            ), "Must provide complete stabilizer frame"
            assert all(
                [z in [0, 1] for z in z_bits]
            ), "Bitvalues must be 0 or 1"
            self.z_bits = list(z_bits)

    def __str__(self) -> str:
        s = f"PauliFrame on [{self.qubit_labels[0]},...,{self.qubit_labels[-1]}] qubits:\n"
        s += f"  X bits: {self.x_bits}"
        s += f"  Z bits: {self.z_bits}"
        return s

    def copy(self) -> PauliFrame:
        return PauliFrame(self.qubit_labels, self.x_bits, self.z_bits)

    def get_bit(self, type: str, qubit: str) -> int:
        type = type.upper()
        assert type in ("X", "Z"), "Can only get X or Z type bits"

        bits = self.x_bits if type == "X" else self.z_bits

        return bits[self.qubit_labels.index(qubit)]

    def update_from_pauli_str(self, pstr: str) -> PauliFrame:
        assert len(pstr) == self.num_qubits

        new_frame = self.copy()
        for i, P in enumerate(pstr):
            if P == "X":
                new_val = (new_frame.x_bits[i] + 1) % 2
                new_frame.x_bits[i] = new_val
            elif P == "Y":
                new_val = (new_frame.x_bits[i] + 1) % 2
                new_frame.x_bits[i] = new_val

                new_val = (new_frame.z_bits[i] + 1) % 2
                new_frame.z_bits[i] = new_val
            elif P == "Z":
                new_val = (new_frame.z_bits[i] + 1) % 2
                new_frame.z_bits[i] = new_val

            # Otherwise we must be I, no action needed

        return new_frame
