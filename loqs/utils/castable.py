"""Definition of IsCastable utility class.
"""

from abc import ABC, abstractmethod
from typing import TypeAlias, TypeVar

# Generic type variable to stand-in for derived class below
T = TypeVar("T")


class IsCastable(ABC):
    """Utility class for objects that are castable."""

    @property
    @abstractmethod
    def Castable(self) -> TypeAlias:
        """A type alias for the allowed inputs to cast().

        Typically the same as allowed inputs to the derived class's constructor.
        """
        pass

    @classmethod
    def cast(cls: T, obj: "IsCastable.Castable") -> T:
        """Cast to the derived class.

        Parameters
        ----------
        obj: Castable
            A castable object

        Returns
        -------
        T
            An object with type T (matching the derived class)
        """
        # If we are already of the same type as the class, return
        if isinstance(obj, cls):
            return obj

        # Derived classes could do additional casting logic here

        # Otherwise, assume the constructor can handle the object
        return cls(obj)
