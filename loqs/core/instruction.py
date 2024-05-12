"""TODO
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, MutableSequence
from typing import TypeAlias

from loqs.core import TrajectoryFrame, Trajectory
from loqs.utils import IsCastable, IsRecordable
from loqs.utils.classproperty import abstractroclassproperty


class Instruction(IsCastable, IsRecordable):

    def __init__(
        self,
        name: str = "(Unnamed)",
        parent: Instruction | InstructionStack | None = None,
    ) -> None:
        self.name = name
        self.parent = parent

    # Derived classes should define Castable also

    @abstractroclassproperty
    def input_frame_spec(self) -> dict[str, type[IsRecordable]]:
        """Minimum specification of an input :class:`Trajectory`."""
        pass

    def num_req_input_frames(self) -> int:
        """Minimum number of frames needed in the input :class:`Trajectory`.

        Defaults to 1, which only looks at the previous frame and allows
        a single :class:`TrajectoryFrame` to be passed in.
        An :class:`Instruction` that requires more history should override
        this to specify the number of frames needed.
        """
        return 1

    @abstractroclassproperty
    def output_frame_spec(self) -> dict[str, type[IsRecordable]]:
        """Minimum specification of the returned :class:`TrajectoryFrame`."""
        pass

    def check_frame(
        self,
        traj_obj: Trajectory.Castable,
        check_input: bool = True,
        check_output: bool = False,
    ) -> bool:
        """Checks whether a frame matches the input or output frame spec.

        Parameters
        ----------
        traj_obj:
            A Trajectory-type object that holds input state information

        check_input:
            Whether to check keys/types against the :meth:`InputFrameSpec`.
            By default, this is True (i.e. we are only checking input frames).

        check_output:
            Whether to check keys/types against the :meth:`OutputFrameSpec`.
            By default, this is False (i.e. we are only checking input frames).

        Returns
        -------
            True if the Trajectory-type object matches in frame specification,
            False otherwise
        """
        try:
            traj = Trajectory.cast(traj_obj)
        except Exception as e:
            raise ValueError(
                "Instruction input must be castable to a Trajectory"
            ) from e

        frame_spec = traj.std_frame_spec

        if check_input:
            if len(traj) < self.num_req_input_frames:
                # We don't have a long enough history for this Instruction
                return False

            for rec_key, rec_class in self.input_frame_spec.items():
                if rec_key in frame_spec and issubclass(
                    frame_spec[rec_key], rec_class
                ):
                    # We don't have a piece of information this Instruction needs
                    return False

        if check_output:
            for rec_key, rec_class in self.output_frame_spec.items():
                if rec_key in frame_spec and issubclass(
                    frame_spec[rec_key], rec_class
                ):
                    # We don't have a piece of information this Instruction would have outputted
                    return False

        return True

    @abstractmethod
    def apply_unsafe(self, input: Trajectory.Castable) -> TrajectoryFrame:
        """Workhorse function for generating a new :class:`TrajectoryFrame`.

        This is an application of the :class:`Instruction` with no safety checks.
        Derived classes should implement this method to enact whatever transformation
        they would like on the state.

        Parameters
        ----------
        input:
            The input frame/trajectory information

        Returns
        -------
        output_frame:
            The new output frame
        """
        pass

    def apply(self, input: Trajectory.Castable, **kwargs) -> TrajectoryFrame:
        """Generate a new :class:`TrajectoryFrame` from the input :class:`Trajectory`.

        Parameters
        ----------
        input:
            The input frame/trajectory information

        kwargs:
            Additional kwargs to be passed on to the underlying :meth:`apply_unsafe`
            call. See that function documentation for more details.

        Returns
        -------
        output_frame:
            The new output frame
        """
        assert self.check_frame(
            input, check_input=True, check_output=False
        ), "Input frame does not match required specification"

        output_frame = self._apply(input, **kwargs)

        assert self.check_frame(
            input, check_input=True, check_output=False
        ), "Output frame does not match required specification"

        return output_frame


class CompositeInstruction(Instruction):

    @property
    def Castable(self) -> TypeAlias:
        return CompositeInstruction | Iterable[Instruction]

    def __init__(
        self,
        instructions: CompositeInstruction.Castable,
        parent: Instruction | None = None,
    ) -> None:
        if isinstance(instructions, CompositeInstruction):
            self.instructions = instructions.instructions
            self.parent = instructions.parent if parent is None else parent
        else:
            self.instructions = instructions
            self.parent = parent


class InstructionStack(MutableSequence[Instruction], IsRecordable, IsCastable):

    @property
    def Castable(self) -> TypeAlias:
        return InstructionStack | Iterable[Instruction] | Instruction

    def __init__(
        self,
        instructions: InstructionStack.Castable | None = None,
        static: bool = True,
    ) -> None:
        """Initialize an InstructionStack."""
        self.static = False  # Just for initialization

        if isinstance(instructions, InstructionStack):
            self._instructions = instructions._instructions
        else:
            if isinstance(instructions, Instruction):
                instructions = [instructions]

            for inst in instructions:
                # This should use .insert under the hool and have proper logic
                self.append(inst)

        self.static = static

    def __getitem__(self, i):
        return self._instructions[i]

    def __setitem__(self, i, item):
        if self.static:
            raise RuntimeError(
                "Cannot set an item in a static "
                + "InstructionStack. First set .static to False."
            )
        self._instructions[i] = item

    def __delitem__(self, i):
        if self.static:
            raise RuntimeError(
                "Cannot delete an item in a static "
                + "InstructionStack. First set .static to False."
            )
        del self._instructions[i]

    def __iter__(self):
        return iter(self._instructions)

    def __len__(self):
        return len(self._instructions)

    def insert(self, i, item):
        if self.static:
            raise RuntimeError(
                "Cannot insert an item into a static "
                + "InstructionStack. First set .static to False."
            )

        assert isinstance(
            item, Instruction
        ), "InstructionStack can only hold Instructions"

        return self._instructions.insert(i, item)

    def reverse(self):
        raise RuntimeError("Cannot reverse an InstructionStack")
