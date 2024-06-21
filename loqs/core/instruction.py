"""TODO
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
import textwrap
from typing import TypeAlias, TypeVar

from loqs.core import HistoryFrame, HistoryStack
from loqs.core.history import HistoryStackCastableTypes
from loqs.internal import Castable, Recordable


T = TypeVar("T", bound="Instruction")

# Type aliases for static type checking
InstructionParentTypes: TypeAlias = "Instruction | InstructionStack | None"
"""TODO"""


InstructionLabelCastableTypes: TypeAlias = (
    "tuple[str, str | list[str]] | tuple[str, str | list[str], str | None] | InstructionLabel"
)
"""TODO"""


InstructionStackCastableTypes: TypeAlias = (
    "InstructionStack | Instruction | InstructionLabelCastableTypes | Sequence[Instruction | InstructionLabelCastableTypes] | None"
)


class Instruction(Recordable):

    def __init__(
        self,
        name: str = "(Unnamed)",
        parent: InstructionParentTypes = None,
        fault_tolerant: bool | None = None,
    ) -> None:
        self.name = name
        self.parent = parent
        self.fault_tolerant = fault_tolerant

    def __str__(self) -> str:
        s = f"Instruction {self.name}\n"
        s += f"  Parent={self.parent}\n"
        s += f"  Fault-tolerant={self.fault_tolerant}\n"
        return s

    @property
    @abstractmethod
    def input_frame_spec(self) -> dict[str, type[Recordable]]:
        """Minimum specification of an input :class:`HistoryStack`."""
        pass

    @property
    def num_req_input_frames(self) -> int:
        """Minimum number of frames needed in the input :class:`HistoryStack`.

        Defaults to 1, which only looks at the previous frame and allows
        a single :class:`HistoryFrame` to be passed in.
        An :class:`Instruction` that requires more history should override
        this to specify the number of frames needed.
        """
        return 1

    @property
    @abstractmethod
    def output_frame_spec(self) -> dict[str, type[Recordable]]:
        """Minimum specification of the returned :class:`HistoryFrame`."""
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
    def apply_unsafe(
        self, input: HistoryStackCastableTypes, *args, **kwargs
    ) -> HistoryFrame:
        """Workhorse function for generating a new :class:`HistoryFrame`.

        This is an application of the :class:`Instruction` with no safety checks.
        Derived classes should implement this method to enact whatever transformation
        they would like on the state.

        Parameters
        ----------
        input:
            The input frame/trajectory information

        *args:
            Any additional args needed for application

        **kwargs:
            Any additional kwargs needed for application

        Returns
        -------
        output_frame:
            The new output frame
        """
        pass

    def apply(
        self, input: HistoryStackCastableTypes, *args, **kwargs
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

        output_frame = self.apply_unsafe(input, *args, **kwargs)

        assert self.check_frame(
            input, check_input=True, check_output=False
        ), "Output frame does not match required specification"

        return output_frame

    def map_qubits(self: T, qubit_mapping: Mapping[str, str]) -> T:
        """TODO"""
        # Many instructions don't have qubits that need to be mapped
        # so default to just returning unmodified object
        return self


class InstructionLabel(Castable):
    """Key type for an InstructionSet."""

    inst_label: str
    """Instruction name.

    This should be the key to look up either in the
    :attr:`InstructionSet.instructions` or a
    :attr:`QECCode.instructions`.
    """

    patch_label: str | None
    """Target patch label."""

    inst_args: tuple
    """Additional args to pass on.
    """

    inst_kwargs: dict[str, object]
    """Additional kwargs to pass on.
    """

    def __init__(
        self,
        inst_label: str,
        patch_label: str | None = None,
        inst_args: Sequence | None = None,
        inst_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize an :class:`InstructionLabel`.

        TODO
        """
        self.inst_label = inst_label
        self.patch_label = patch_label

        if inst_args is None:
            inst_args = []
        self.inst_args = tuple(inst_args)

        if inst_kwargs is None:
            inst_kwargs = {}
        self.inst_kwargs = dict(inst_kwargs)

    def __str__(self) -> str:
        """TODO"""
        return self.__repr__()

    def __repr__(self) -> str:
        """TODO"""
        s = f"InstructionLabel({self.inst_label},{self.patch_label},"
        s += f"{self.inst_args},{self.inst_kwargs})\n"
        return s

    @classmethod
    def cast(cls, obj: object) -> InstructionLabel:
        """Cast to a :class:`InstructionLabel` object.

        Unlike most castable objects, :class:`InstructionLabel`
        requires at least two inputs. This version of cast additionally
        allows a tuple/list variant for the multiple arguments and
        disallows a single object being passed in.

        Parameters
        ----------
        obj:
            A castable object that is either:
            - Already a :class:`InstructionLabel` object,
            in which case `obj` is returned
            - A kwarg dict that is passed into the constructor
            - A sequence of the arguments of the
            :class:`InstructionLabel` constructor

        Returns
        -------
            A :class:`SyndromeExtraction` object
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, dict):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        elif isinstance(obj, tuple):
            # Assume this is a tuple of arguments, pass all in
            return cls(*obj)

        # Else we can't handle this
        raise ValueError(
            "InstructionLabel requires at least two arguments to cast. "
            + "Use a tuple of arguments or kwarg dict when casting."
        )


class InstructionStack(Sequence[Instruction | InstructionLabel], Recordable):

    _instructions: list[Instruction | InstructionLabel]
    """Internal list of instructions"""

    def __init__(
        self, instructions: InstructionStackCastableTypes = None
    ) -> None:
        """Initialize an InstructionStack."""
        self._instructions = []
        if isinstance(instructions, InstructionStack):
            self._instructions = instructions._instructions
        elif isinstance(instructions, Instruction):
            self._instructions = [instructions]
        elif isinstance(instructions, Sequence):
            for inst in instructions:
                if not isinstance(inst, Instruction):
                    try:
                        inst = InstructionLabel.cast(inst)
                    except ValueError as e:
                        raise ValueError(
                            f"Failed to cast {inst} to InstructionLabel"
                        ) from e

                self._instructions.append(inst)

        for inst in self._instructions:
            if isinstance(inst, Instruction):
                inst.parent = self

    def __getitem__(self, i):
        return self._instructions[i]

    def __len__(self):
        return len(self._instructions)

    def __str__(self):
        s = f"InstructionStack with {len(self)} items:\n"
        for i, inst in enumerate(self._instructions):
            si = str(inst)
            si = textwrap.indent(si, "  ")
            s += si
        return s

    def append_instruction(self, item) -> InstructionStack:
        return self.insert_instruction(len(self), item)

    def delete_instruction(self, i) -> InstructionStack:
        instructions = self._instructions.copy()
        del instructions[i]
        return InstructionStack(instructions)

    def insert_instruction(self, i, item) -> InstructionStack:
        instructions = self._instructions.copy()
        instructions.insert(i, item)
        return InstructionStack(instructions)

    def pop_instruction(
        self,
    ) -> tuple[Instruction | InstructionLabel, InstructionStack]:
        return self._instructions[0], InstructionStack(self._instructions[1:])
