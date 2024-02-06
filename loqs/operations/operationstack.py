"""TODO
"""

from collections import UserList
from typing import TYPE_CHECKING

from loqs.operations import Operation


OperationList = UserList[Operation] if TYPE_CHECKING else UserList


class OperationStack(OperationList):
    def __init__(self):
        pass

    # TODO: Other list functions
