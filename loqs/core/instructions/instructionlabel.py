"""TODO
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypeVar

from loqs.core.instructions.instruction import Instruction
from loqs.internal import Castable, Serializable


T = TypeVar("T", bound="InstructionLabel")

InstructionLabelCastableTypes: TypeAlias = (
    "Instruction | str | tuple[Instruction | str, str | None, str | None] | InstructionLabel"
)


class InstructionLabel(Castable, Serializable):
    """TODO"""

    instruction: Instruction | None
    """Instruction.

    Either :attr:`instruction` or
    :attr:`inst_label` must be defined.
    """

    inst_label: str | None
    """Instruction name, if needs to be resolved.

    Either :attr:`instruction` or
    :attr:`inst_label` must be defined.

    This should be the key to look up either in the
    :attr:`InstructionSet.instructions` or a
    :attr:`QECCode.instructions`.
    """

    patch_label: str | None
    """Target patch label, if needs to be resolved.

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
        inst_or_label: Instruction | str,
        patch_label: str | None = None,
        inst_args: Sequence | None = None,
        inst_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize an :class:`InstructionLabel`.

        TODO
        """
        self.instruction = None
        self.inst_label = None
        if isinstance(inst_or_label, Instruction):
            self.instruction = inst_or_label
        else:
            self.inst_label = inst_or_label
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
        if self.inst_label is None:
            assert self.instruction is not None
            inst_label = self.instruction.name
        else:
            inst_label = self.inst_label

        s = f"InstructionLabel({inst_label},{self.patch_label},"
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
        if isinstance(obj, InstructionLabel):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, Mapping):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        elif isinstance(obj, (Instruction, str)):
            return cls(obj)

        # Assume this is a tuple of arguments, pass all in
        return cls(*obj)  # type: ignore

    @classmethod
    def _from_serialization(cls: type[T], state: Mapping) -> T:
        inst_label = cls.deserialize(state["instruction"])
        assert isinstance(inst_label, Instruction | None)
        if inst_label is None:
            inst_label = state["inst_label"]

        patch_label = state["patch_label"]
        inst_args = cls.deserialize(state["inst_args"])
        assert isinstance(inst_args, list)
        inst_kwargs = cls.deserialize(state["inst_kwargs"])
        assert isinstance(inst_kwargs, dict)
        return cls(inst_label, patch_label, inst_args, inst_kwargs)

    def _to_serialization(self) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "instruction": self.serialize(self.instruction),
                "inst_label": self.inst_label,
                "patch_label": self.patch_label,
                "inst_args": self.serialize(self.inst_args),
                "inst_kwargs": self.serialize(self.inst_kwargs),
            }
        )
        return state
