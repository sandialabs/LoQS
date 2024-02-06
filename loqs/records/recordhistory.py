"""TODO
"""

from collections import UserList
from typing import TYPE_CHECKING

from loqs.records import Record


RecordList = UserList[Record] if TYPE_CHECKING else UserList


class RecordHistory(RecordList):
    def __init__(self):
        pass

    # TODO: Other list functions
