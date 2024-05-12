""":class:`MockOperations` definition.
"""

from collections.abc import Mapping
from typing import TypeAlias

from loqs.core.instruction import Instruction
from loqs.core.recordables import MockState
from loqs.core.trajectory import Trajectory, TrajectoryFrame
from loqs.utils.classproperty import roclassproperty
from loqs.utils.recordable import IsRecordable


class MockOperation(Instruction):
    """A "mock" operation acting on :class:`MockState`.

    This simply generates a new :class:`MockState` based
    on the :attr:`MockState.state` attribute.
    Primarily used for testing and demonstrating
    the high-level flow of a class:`QuantumProgram`.
    """

    def __init__(self, state_map: Mapping[str, str]) -> None:
        """Initialize a :class:`MockOperation`.

        Parameters
        ----------
        """
        self.state_map = state_map

    @roclassproperty
    def Castable(self) -> TypeAlias:
        return Mapping[str, str] | MockOperation

    @roclassproperty
    def input_frame_spec(self) -> dict[str, type[IsRecordable]]:
        return {"mock_state": MockState}

    @roclassproperty
    def output_frame_spec(self) -> dict[str, type[IsRecordable]]:
        return {"mock_state": MockState, "instruction": MockOperation}

    def apply_unsafe(self, input: Trajectory.Castable) -> TrajectoryFrame:
        """Map the input :class:`MockState` forward.

        This

        Parameters
        ----------
        input:
            The input frame/trajectory information

        Returns
        -------
        output_frame:
            The new output frame
        """
        input = Trajectory.cast(input)

        last_frame: TrajectoryFrame = input[-1]

        old_state = last_frame["mock_state"]

        try:
            new_state_str = self.state_map[old_state.state]
        except KeyError as e:
            raise KeyError(
                f"MockOperation has no mapped value for {old_state.state}"
            ) from e

        new_state = MockState(new_state_str)

        output_frame = last_frame.copy()
        output_frame["mock_state"] = new_state
        output_frame["instruction"] = self
        output_frame.finalize_inplace()

        return output_frame
