"""TODO
"""

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Type, TypeAlias


class HasStateBackend(ABC):
    """Utility class for objects that contain a StateBackend."""

    @property
    @abstractmethod
    def state_backend(self) -> "StateBackend":
        pass


class StateBackend(ABC):
    """Base class for an object that can store physical quantum states."""

    @property
    @abstractmethod
    def StateType(self) -> TypeAlias:
        pass
