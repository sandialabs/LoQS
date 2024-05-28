""":class:`Recordable` definition.
"""

from abc import ABC


class Recordable(ABC):
    """Base class for things that can be added to a :class:`TrajectoryFrame`.

    Currently empty, will eventually probably have logging/printing/analysis
    functionality at least.
    """

    # TODO
