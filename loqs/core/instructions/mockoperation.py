""":class:`MockOperations` definition.
"""

from __future__ import annotations

from collections.abc import Mapping

from loqs.core import Instruction, Recordable, HistoryStack, HistoryFrame
from loqs.core.recordables import MockState
from loqs.internal.classproperty import roclassproperty


class MockOperation(Instruction):
    """A "mock" operation acting on :class:`MockState`.

    This simply generates a new :class:`MockState` based
    on the :attr:`MockState.state` attribute.
    Primarily used for testing and demonstrating
    the high-level flow of a class:`QuantumProgram`.
    """

    def __init__(self, state_map: CastableTypes) -> None:
        """Initialize a :class:`MockOperation`.

        Parameters
        ----------
        """
        self.state_map = state_map

    @roclassproperty
    def CastableTypes(self) -> type:
        return MockOperation | Mapping[str, str]

    @roclassproperty
    def input_frame_spec(self) -> dict[str, type[Recordable]]:
        return {"mock_state": MockState}

    @roclassproperty
    def output_frame_spec(self) -> dict[str, type[Recordable]]:
        return {"mock_state": MockState, "instruction": MockOperation}

    def apply_unsafe(self, input: HistoryStack.Castable) -> HistoryFrame:
        """Map the input :class:`MockState` forward.

        This

        Parameters
        ----------
        input:
            The input frame/history information

        Returns
        -------
        output_frame:
            The new output frame
        """
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        old_state = last_frame["mock_state"]

        try:
            new_state_str = self.state_map[old_state.state]
        except KeyError as e:
            raise KeyError(
                f"MockOperation has no mapped value for {old_state.state}"
            ) from e

        new_data = {
            "mock_state": MockState(new_state_str),
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame
