"""Definition of IsCastable utility class.
"""

from __future__ import annotations

from typing import TypeVar

from .classproperty import (
    abstractroclassproperty,
    ABCWithROClassProperties,
)

# Generic type variable to stand-in for derived class below
T = TypeVar("T")


class Castable(ABCWithROClassProperties):
    """Utility class for objects that are "castable".

    By default, a :class:`Castable` object is one that can be
    initialized from a single argument, i.e. the object
    can be passed as the first argument to the constructor and any
    remaining arguments have sensible defaults.

    :class:`Castable` objects have a :attr:`CastableTypes` class property
    which is a type for inputs that can be cast. This is
    also handy for typing the first arg of :meth:`__init__`.
    They also have a :meth:`cast` function, which either does nothing
    if the object matches the correct type, or initializes it by calling
    the constructor.
    """

    @abstractroclassproperty
    def CastableTypes(cls) -> type:
        """A type alias for the allowed inputs to cast().

        Typically a subset of allowed inputs to the derived class's constructor.
        """
        pass

    @classmethod
    def cast(cls: T, obj: Castable) -> T:
        """Cast to the derived class.

        This is the base implementation. Derived classes should
        reimplement this if additional casting logic is desired.

        Parameters
        ----------
        obj:
            A castable object that is either:
            - Already the derived class type, in which case `obj`
            is returned
            - A kwarg dict that is passed into the derived class
            constructor
            - The first argument of the derived class constructor

        Returns
        -------
            An object with type T (matching the derived class)
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, dict):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)

        # Otherwise, assume this is the first __init__ argument
        return cls(obj)
