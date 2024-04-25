"""Definition of IsCastable utility class.
"""

from __future__ import annotations

from abc import ABC
from typing import TypeAlias, TypeVar

from .roclassproperty import roclassproperty

# Generic type variable to stand-in for derived class below
T = TypeVar("T")


class IsCastable(ABC):
    """Utility class for objects that are castable."""

    @roclassproperty
    def Castable(cls) -> TypeAlias:
        """A type alias for the allowed inputs to cast().

        Typically the same as allowed inputs to the derived class's constructor.
        """
        raise NotImplementedError("Derived classes should implement this")

    @classmethod
    def cast(cls: T, obj: Castable) -> T:
        """Cast to the derived class.

        This is the base implementation, which either returns
        `obj` if it matches the class or simply passes it to the
        constructor. Derived classes should reimplement this if
        additional casting logic is desired.

        Parameters
        ----------
        obj: Castable
            A castable object

        Returns
        -------
        T
            An object with type T (matching the derived class)
        """
        if isinstance(obj, cls):
            return obj

        return cls(obj)
