"""TODO
"""

from collections.abc import Mapping

from loqs.core.instruction import Instruction


class QECCode:
    """TODO"""

    def __init__(self, logical_operations: Mapping[str, Instruction]):
        """TODO"""
        self.logical_operations = logical_operations
