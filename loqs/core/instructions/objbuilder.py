"""TODO
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from loqs.core import Instruction, HistoryStack, HistoryFrame
from loqs.core.instruction import InstructionParentTypes
from loqs.core.history import HistoryStackCastableTypes


ObjectBuilderCastableTypes: TypeAlias = (
    "ObjectBuilder | tuple[str, type] | tuple[str, type, Sequence | None] | tuple[str, type, Sequence | None, Mapping[str, object] | None]"
)


class ObjectBuilder(Instruction):
    """TODO"""

    obj_class: type
    """TODO"""

    obj_args: tuple
    """TODO"""

    obj_kwargs: dict[str, object]
    """TODO"""

    def __init__(
        self,
        frame_key: str,
        obj_class: type,
        obj_args: Sequence | None = None,
        obj_kwargs: Mapping | None = None,
        name: str = "(Unnamed object builder)",
        parent: InstructionParentTypes = None,
        fault_tolerant: bool | None = None,
    ) -> None:
        """TODO

        Parameters
        ----------
        """
        super().__init__(name, parent, fault_tolerant)

        if self.fault_tolerant is None:
            # Assume FT unless explicitly set to False
            self.fault_tolerant = True

        self.frame_key = frame_key
        self.obj_class = obj_class
        if obj_args is None:
            obj_args = []
        self.obj_args = tuple(obj_args)
        if obj_kwargs is None:
            obj_kwargs = {}
        self.obj_kwargs = dict(obj_kwargs)

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return {}

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {self.frame_key: self.obj_class, "instruction": Instruction}

    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
        """TODO"""
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        obj = self.obj_class(*self.obj_args, **self.obj_kwargs)

        new_data = {
            self.frame_key: obj,
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame

    # This is one of the cases where we do not have a nice single argument cast
    # Override the cast method to accept a tuple of the first two args
    @classmethod
    def cast(cls, obj: object) -> ObjectBuilder:
        """Cast to a :class:`ObjectBuilder` object.

        Unlike most castable objects, :class:`ObjectBuilder`
        requires at least 2 inputs. This version of cast additionally allows
        a tuple/list variant for the arguments and disallows
        a single object being passed in.

        Parameters
        ----------
        obj:
            A castable object that is either:
            - Already a :class:`ObjectBuilder` object,
            in which case `obj` is returned
            - A kwarg dict that is passed into the constructor
            - A sequence of the first two to four arguments of the
            :class:`ObjectBuilder` constructor

        Returns
        -------
            A :class:`ObjectBuilder` object
        """
        if isinstance(obj, cls):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, dict):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        elif isinstance(obj, Sequence) and len(obj) > 1 and len(obj) < 5:
            # Assume this is a tuple/list of first several args
            return cls(*obj)

        # Else we can't handle this
        raise ValueError(
            "SyndromeExtraction requires two arguments to cast. "
            + "Use a 2-tuple or kwarg dict when casting."
        )
