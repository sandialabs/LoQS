""":class:`Recordable` definition.
"""

from abc import ABC
from pprint import pformat, saferepr

from loqs.internal.castable import Castable


class Recordable(ABC, Castable):
    """Base class for things that can be added to a :class:`TrajectoryFrame`.

    Currently empty, will eventually probably have logging/printing/analysis
    functionality at least.
    """

    def __str__(self):
        """TODO"""
        return pformat(vars(self), indent=2)

    def __repr__(self):
        """TODO"""
        return saferepr(vars(self))
