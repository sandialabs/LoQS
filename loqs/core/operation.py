"""TODO
"""

from collections import abc

from loqs.core import IsRecordable
from loqs.utils import IsCastable


class OperationSpec(IsCastable):
    def __init__(self):
        pass


class Operation(IsRecordable):
    def __init__(self):
        pass


class CompositeOperation(Operation):
    def __init__(self):
        pass


class OperationStack(abc.Sequence[Operation], IsRecordable):
    def __init__(self):
        pass
