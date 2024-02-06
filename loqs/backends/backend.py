"""Base backend for simulating quantum systems.
"""

from typing import Iterable, Optional

# from loqs.operations import BaseOperation


class Backend:
    """Base backend for simulating physical quantum systems."""

    def __init__(self, qubit_labels: Iterable[str]):
        self.qubit_labels = qubit_labels
        self._state = None

    # def apply_operation(self, operation: BaseOperation) -> None:
    #    raise NotImplementedError("Derived classes must implement this")
