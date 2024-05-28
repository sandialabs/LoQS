""":class:`MockState` definition.
"""

from loqs.core import Recordable


class MockState(Recordable):
    """A "mock" state.

    This simply holds a string that indicates a "state".
    Primarily used for testing and demonstrating
    the high-level flow of a class:`QuantumProgram`.
    """

    def __init__(self, state: str) -> None:
        """Initialize a :class:`MockState`.

        Parameters
        ----------
        """
        self.state = state
