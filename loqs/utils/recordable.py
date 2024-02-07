"""Definition of IsRecordable utility class.
"""

from abc import ABC


class IsRecordable(ABC):
    """Base class for things that can be added to a :class:`Record`.

    Currently empty, will eventually probably have logging/printing/analysis
    functionality at least.
    """
