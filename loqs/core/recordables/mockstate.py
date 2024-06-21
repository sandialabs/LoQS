""":class:`MockState` definition.
"""

from typing import TypeAlias

from loqs.internal import Recordable


MockStateCastableTypes: TypeAlias = "MockState | str"


class MockState(Recordable):
    """A "mock" state.

    This simply holds a string that indicates a "state".
    Primarily used for testing and demonstrating
    the high-level flow of a class:`QuantumProgram`.
    """

    state: str
    """Underlying "state"."""

    def __init__(self, state: MockStateCastableTypes) -> None:
        """Initialize a :class:`MockState`.

        Parameters
        ----------
        """
        if isinstance(state, MockState):
            self.state = state.state
        else:
            self.state = state
