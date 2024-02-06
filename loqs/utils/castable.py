"""Utility class for objects that are castable.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeAlias


class IsCastable(ABC):
    """ """

    @property
    @abstractmethod
    def Castable(self) -> TypeAlias:
        """A type alias for the allowed inputs to cast().

        Typically the same as allowed inputs to the derived class's constructor.
        """
        pass

    @classmethod
    def cast(cls, obj: "IsCastable.Castable") -> Any:
        """Cast to the derived class.

        Parameters
        ----------
        obj: Castable
            A castable object

        Returns
        -------
        Any
            An object with the type of the derived class
        """
        # If we are already of the same type as the class, return
        if isinstance(obj, cls):
            return obj

        # Derived classes could do additional casting logic here

        # Otherwise, assume the constructor can handle the object
        return cls(obj)
