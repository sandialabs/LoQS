"""TODO
"""

from collections.abc import Iterator, Mapping, Sequence

from loqs.core.instruction import Instruction


class QECCode:
    """TODO"""

    def __init__(
        self,
        instructions: Mapping[str, Instruction],
        template_qubits: Sequence[str],
    ):
        """TODO"""
        self.instructions = {k: v for k, v in instructions.items()}
        self.template_qubits = list(template_qubits)

    def create_patch(self, qubits: Sequence[str]):
        """TODO"""
        return QECCodePatch(self, qubits)


class QECCodePatch(Mapping[str, Instruction]):
    """TODO"""

    def __init__(self, code: QECCode, qubits: Sequence[str]):
        """TODO"""
        assert len(qubits) == len(code.template_qubits), (
            f"Patch must have {len(code.template_qubits)} qubits "
            + f"to match code {code}, not {len(qubits)}"
        )

        self.code = code
        self.qubits = qubits

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
