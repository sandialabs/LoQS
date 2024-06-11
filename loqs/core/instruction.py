"""TODO
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

from loqs.core import HistoryFrame, HistoryStack, Recordable
from loqs.core.history import HistoryStackCastableTypes


# Type aliases for static type checking
InstructionParentTypes: TypeAlias = "Instruction | InstructionStack | None"
"""TODO"""

CompositeInstructionCastableTypes: TypeAlias = (
    "CompositeInstruction | Sequence[Instruction]"
)

InstructionStackCastableTypes: TypeAlias = (
    "InstructionStack | Sequence[Instruction] | Instruction | None"
)


class Instruction(Recordable):

    def __init__(
        self,
        name: str = "(Unnamed)",
        parent: InstructionParentTypes = None,
    ) -> None:
        self.name = name
        self.parent = parent

    @property
    @abstractmethod
    def input_frame_spec(self) -> dict[str, type]:
        """Minimum specification of an input :class:`Trajectory`."""
        pass

    @property
    def num_req_input_frames(self) -> int:
        """Minimum number of frames needed in the input :class:`Trajectory`.

        Defaults to 1, which only looks at the previous frame and allows
        a single :class:`TrajectoryFrame` to be passed in.
        An :class:`Instruction` that requires more history should override
        this to specify the number of frames needed.
        """
        return 1

    @property
    @abstractmethod
    def output_frame_spec(self) -> dict[str, type]:
        """Minimum specification of the returned :class:`TrajectoryFrame`."""
        pass

    def check_frame(
        self,
        traj_obj: HistoryStackCastableTypes,
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
            traj = HistoryStack.cast(traj_obj)
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
    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
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

    def apply(
        self, input: HistoryStackCastableTypes, **kwargs
    ) -> HistoryFrame:
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

        output_frame = self.apply_unsafe(input, **kwargs)

        assert self.check_frame(
            input, check_input=True, check_output=False
        ), "Output frame does not match required specification"

        return output_frame


class CompositeInstruction(Instruction):

    instructions: list[Instruction]
    """Set of instructions in this :class:`CompositeInstruction`."""

    def __init__(
        self,
        instructions: CompositeInstructionCastableTypes,
        name: str = "(Unnamed composite)",
        parent: Instruction | None = None,
    ) -> None:
        super().__init__(name=name, parent=parent)

        if isinstance(instructions, CompositeInstruction):
            self.instructions = instructions.instructions
            self.parent = instructions.parent if parent is None else parent
        else:
            # TODO: Type-check
            self.instructions = list(instructions)

    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
        """Workhorse function for generating a new :class:`TrajectoryFrame`.

        This is an application of the :class:`Instruction` with no safety checks.

        For :class:`CompositeInstruction`, this simply calls the underlying
        :meth:`apply_unsafe` methods of the contained :class:`Instruction` objects,
        feeding forward the resulting frames as needed.

        Parameters
        ----------
        input:
            The input frame/trajectory information

        Returns
        -------
        output_frame:
            The new output frame
        """
        stack = HistoryStack.cast(input)
        for instruction in self.instructions:
            stack.append(instruction.apply_unsafe(stack))
        return stack[-1]


class InstructionStack(Sequence[Instruction], Recordable):

    _instructions: list[Instruction]
    """Internal list of instructions"""

    def __init__(
        self, instructions: InstructionStackCastableTypes = None
    ) -> None:
        """Initialize an InstructionStack."""
        if isinstance(instructions, InstructionStack):
            self._instructions = instructions._instructions
        elif isinstance(instructions, Instruction):
            self._instructions = [instructions]
        elif isinstance(instructions, Sequence):
            assert all(
                [isinstance(inst, Instruction) for inst in instructions]
            ), "InstructionStack can only hold Instructions"

            self._instructions = list(instructions)
        else:
            self._instructions = []

        for inst in self._instructions:
            inst.parent = self

    def __getitem__(self, i):
        return self._instructions[i]

    def __len__(self):
        return len(self._instructions)

    def append_instruction(self, item):
        return self.insert_instruction(len(self), item)

    def delete_instruction(self, i):
        instructions = self._instructions.copy()
        del instructions[i]
        return InstructionStack(instructions)

    def insert_instruction(self, i, item):
        instructions = self._instructions.copy()
        instructions.insert(i, item)
        return InstructionStack(instructions)
