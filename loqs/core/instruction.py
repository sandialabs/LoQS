"""Base classes for Instructions.

Instructions are generally objects that take state information (i.e. from one or
several Records or a RecordHistory) and propagate or generate new state
information into a new Record.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableSequence
from typing import Optional, Type, TypeAlias, Union, get_args

from loqs.utils import IsCastable, IsRecordable
from loqs.core import Record, RecordHistory, RecordSpec


class InstructionSpec:
    def __init__(
        self,
        input_type: Type,
        input_spec: RecordSpec.Castable,
        output_spec: RecordSpec.Castable,
    ):
        assert input_type in [Record] + get_args(
            RecordHistory.Castable
        ), "Input type Must be Record or RecordHistory.Castable"
        self._input_type = input_type
        self._input_spec = RecordSpec.cast(input_spec)
        self._output_spec = RecordSpec.cast(output_spec)

    @property
    def minimum_input_spec(self) -> RecordSpec:
        return self._input_spec

    @property
    def minimum_output_spec(self) -> RecordSpec:
        return self._output_spec

    def check_record(
        self,
        record_object: Union[RecordSpec, Record, RecordHistory],
        check_input: bool = True,
        check_output: bool = True,
    ) -> bool:
        if isinstance(record_object, Record):
            spec = record_object.spec
        elif isinstance(record_object, RecordHistory):
            spec = record_object.standard_record_spec
        else:
            spec = record_object

        if check_input:
            for rec_key, rec_class in self.inputs:
                if not spec.check_class(rec_key, rec_class):
                    return False

        if check_output:
            for rec_key, rec_class in self.outputs:
                if not spec.check_class(rec_key, rec_class):
                    return False

        # If everything checked out, we can act on this Record/RecordSpec
        return True


class Instruction(IsRecordable, ABC):
    def __init__(
        self,
        instruction_spec: InstructionSpec,
        name: Optional[str] = None,
        parent: Optional[Instruction] = None,
    ) -> None:
        self._spec = instruction_spec
        self._name = name
        self._parent = parent

    @property
    def name(self) -> str:
        return "(Unnamed)" if self._name is None else self._name

    @property
    def instruction_spec(self) -> InstructionSpec:
        return self._spec

    @property
    def parent(self) -> Optional[Instruction]:
        return self._parent

    @abstractmethod
    def apply(self, input: Union[Record, RecordHistory.Castable]) -> Record:
        pass


class CompositeInstruction(Instruction, IsCastable):

    @property
    def Castable(self) -> TypeAlias:
        return Union[CompositeInstruction, Iterable[Instruction]]

    def __init__(
        self,
        instructions: CompositeInstruction.Castable,
        parent: Optional[Instruction] = None,
    ) -> None:
        if isinstance(instructions, CompositeInstruction):
            self.instructions = instructions.instructions
            self.parent = instructions.parent if parent is None else parent
        else:
            self.instructions = instructions
            self.parent = parent


class InstructionStack(MutableSequence[Instruction], IsCastable, IsRecordable):

    @property
    def Castable(self) -> TypeAlias:
        return Union[InstructionStack, Iterable[Instruction], Instruction]

    def __init__(
        self,
        instructions: Optional[InstructionStack.Castable] = None,
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
