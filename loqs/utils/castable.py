"""Definition of IsCastable utility class.
"""

from __future__ import annotations

from typing import TypeAlias, TypeVar

from .classproperty import (
    abstractroclassproperty,
    ABCWithROClassProperties,
)

# Generic type variable to stand-in for derived class below
T = TypeVar("T")


class IsCastable(ABCWithROClassProperties):
    """Utility class for objects that are castable.

    This inherits from :class:`ABCWithROClassProperties`
    when technically :class:`HasROClassProperties` would have been
    sufficient for this interface.
    However, so many derived classes would also need :class:`abc.ABC`
    so it is just easier to sort out the metaclasses here.
    """

    @abstractroclassproperty
    def Castable(cls) -> TypeAlias:
        """A type alias for the allowed inputs to cast().

        Typically a superset of allowed inputs to the derived class's constructor.
        """
        pass

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
