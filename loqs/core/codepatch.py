"""TODO
"""

from collections.abc import Mapping, Sequence

from loqs.core.instruction import Instruction


class CodePatch:
    """TODO"""

    def __init__(
        self,
        logical_operations: Mapping[str, Instruction],
        qubit_labels: Sequence[str],
    ):
        """TODO"""
        self.logical_operations = {k: v for k, v in logical_operations.items()}
        self.qubit_labels = list(qubit_labels)
