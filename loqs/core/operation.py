"""Base classes for Operations.

Operations are generally objects that take state information (i.e. from one or
several Records or a RecordHistory) and propagate or generate new state
information into a new Record.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableSequence
from typing import Optional, Type, TypeAlias, Union, get_args

from loqs.utils import IsCastable, IsRecordable
from loqs.core import Record, RecordHistory, RecordSpec


class OperationSpec:
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


class Operation(IsRecordable, ABC):
    def __init__(
        self,
        operation_spec: OperationSpec,
        name: Optional[str] = None,
        parent: Optional["Operation"] = None,
    ) -> None:
        self._spec = operation_spec
        self._name = name
        self._parent = parent

    @property
    def name(self) -> str:
        return "(Unnamed)" if self._name is None else self._name

    @property
    def operation_spec(self) -> OperationSpec:
        return self._spec

    @property
    def parent(self) -> Optional["Operation"]:
        return self._parent

    @abstractmethod
    def apply(self, input: Union[Record, RecordHistory.Castable]) -> Record:
        pass


class CompositeOperation(Operation, IsCastable):

    Castable: TypeAlias = Union["CompositeOperation", Iterable[Operation]]

    def __init__(
        self,
        operations: "CompositeOperation.Castable",
        parent: Optional[Union["Operation", "OperationStack"]] = None,
    ) -> None:
        if isinstance(operations, CompositeOperation):
            self.operations = operations.operations
            self.parent = operations.parent if parent is None else parent
        else:
            self.operations = operations
            self.parent = parent


class OperationStack(MutableSequence[Operation], IsCastable, IsRecordable):

    Castable = Union["OperationStack", Iterable[Operation], Operation]

    def __init__(
        self,
        operations: Optional["OperationStack.Castable"] = None,
        static: bool = True,
    ) -> None:
        """Initialize an OperationStack."""
        self.static = False  # Just for initialization

        if isinstance(operations, OperationStack):
            self._ops = operations._ops
        else:
            if isinstance(operations, Operation):
                operations = [operations]

            for op in operations:
                # This should use .insert under the hool and have proper logic
                self.append(op)

        self.static = static

    def __getitem__(self, i):
        return self._ops[i]

    def __setitem__(self, i, item):
        if self.static:
            raise RuntimeError(
                "Cannot set an item in a static "
                + "OperationStack. First set .static to False."
            )
        self._ops[i] = item

    def __delitem__(self, i):
        if self.static:
            raise RuntimeError(
                "Cannot delete an item in a static "
                + "OperationStack. First set .static to False."
            )
        del self._ops[i]

    def __iter__(self):
        return iter(self._ops)

    def __len__(self):
        return len(self._ops)

    def insert(self, i, item):
        if self.static:
            raise RuntimeError(
                "Cannot insert an item into a static "
                + "OperationStack. First set .static to False."
            )

        assert isinstance(
            item, Operation
        ), "OperationStack can only hold Operations"

        return self._ops.insert(i, item)

    def reverse(self):
        raise RuntimeError("Cannot reverse an OperationStack")
