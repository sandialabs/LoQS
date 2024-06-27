"""TODO
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from loqs.internal import Castable

InstructionLabelCastableTypes: TypeAlias = (
    "tuple[str, str | list[str] | None, str | None] | InstructionLabel"
)


class InstructionLabel(Castable):
    """TODO"""

    inst_label: str
    """Instruction name.

    This should be the key to look up either in the
    :attr:`InstructionSet.instructions` or a
    :attr:`QECCode.instructions`.
    """

    patch_label: str | None
    """Target patch label.

    Can be None to use an entry in
    :attr:`InstructionStack.global_instructions`.
    Otherwise, should be a key into the
    :class:`PatchDict` stored in 'patches' in the
    last :class:`Frame` of the :class:`History`.
    """

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
        return repr(self)

    def __repr__(self) -> str:
        """TODO"""
        s = f"InstructionLabel({self.inst_label},{self.patch_label},"
        s += f"{self.inst_args}," + "{"
        for i, (k, v) in enumerate(self.inst_kwargs.items()):
            vstr = str(v)
            if vstr.endswith("\n"):
                vstr = vstr[:-1]
            s += f"{k}: {vstr}"
            if i != len(self.inst_kwargs) - 1:
                s += ","
        s += "})\n"
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
