"""TODO
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from loqs.core import Instruction, HistoryStack, HistoryFrame
from loqs.core.instruction import InstructionParentTypes
from loqs.core.history import HistoryStackCastableTypes
from loqs.internal import Recordable


ObjectBuilderCastableTypes: TypeAlias = (
    "ObjectBuilder | tuple[str, type] | tuple[str, type, Sequence | None] | tuple[str, type, Sequence | None, Mapping[str, object] | None]"
)


class ObjectBuilder(Instruction):
    """TODO"""

    def __init__(
        self,
        frame_key: str,
        obj_class: type[Recordable],
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
            # Assume FT is True unless explicitly set to False
            self.fault_tolerant = True

        self.frame_key = frame_key

        assert issubclass(
            obj_class, Recordable
        ), "ObjectBuilder should only build Recordable classes"
        self.obj_class = obj_class

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return {}

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {self.frame_key: self.obj_class, "instruction": Instruction}

    def apply_unsafe(
        self, input: HistoryStackCastableTypes, *args, **kwargs
    ) -> HistoryFrame:
        """TODO"""
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        try:
            obj = self.obj_class(*args, **kwargs)
        except Exception as e:
            raise ValueError("Failed to create object in ObjectBuilder") from e

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
        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            # Assume this is a tuple/list of first several args
            return cls(*obj)

        # Else we can't handle this
        raise ValueError(
            "ObjectBuilder requires two arguments to cast. "
            + "Use a 2-tuple or kwarg dict when casting."
        )
