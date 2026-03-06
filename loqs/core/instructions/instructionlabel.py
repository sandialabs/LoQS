#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`InstructionLabel` definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypeVar

from loqs.core.instructions.instruction import Instruction
from loqs.internal import SeqCastable, Displayable


T = TypeVar("T", bound="InstructionLabel")

InstructionLabelCastableTypes: TypeAlias = (
    "Instruction | str | tuple[Instruction | str, str | None] | tuple[Instruction | str, str | None, Sequence | None] | tuple[Instruction | str, str | None, Sequence | None, Mapping | None] | InstructionLabel"
)
"""Objects that can be cast to a :class:`.InstructionLabel`."""


class InstructionLabel(SeqCastable, Displayable):
    """Instruction labels intended to be elements of an :class:`.InstructionStack`.

    These are also castable from 1- to 4-tuples, so users
    can just specify a stack as a list of tuples and labels
    will be cast into these under the hood.
    """

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
        """
        Parameters
        ----------
        inst_or_label:
            Either an :class:`Instruction` or string, setting
            one of :attr:`.instruction` or :attr:`.inst_label`.

        patch_label:
            See :attr:`.patch_label`. Defaults to ``None``.

        inst_args:
            See :attr:`.inst_args`. Default to ``None``, which
            just sets it to be an empty list.

        inst_kwargs:
            See :attr:`.inst_kwargs`. Default to ``None``, which
            just sets it to be an empty dict.
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
        return repr(self)

    def __repr__(self) -> str:
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

    def __hash__(self) -> int:
        return hash(
            (
                hash(self.instruction),
                hash(self.inst_label),
                self.patch_label,
                self.hash(self.inst_args),
                self.hash(self.inst_kwargs),
            )
        )

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
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        inst_label = cls.deserialize(
            state["instruction"], serial_id_to_obj_cache
        )
        assert isinstance(inst_label, Instruction | None)
        if inst_label is None:
            inst_label = state["inst_label"]

        patch_label = state["patch_label"]
        inst_args = cls.deserialize(state["inst_args"], serial_id_to_obj_cache)
        assert isinstance(inst_args, list)
        inst_kwargs = cls.deserialize(
            state["inst_kwargs"], serial_id_to_obj_cache
        )
        assert isinstance(inst_kwargs, dict)
        return cls(inst_label, patch_label, inst_args, inst_kwargs)

    def _to_serialization(
        self, hash_to_serial_id_cache=None, ignore_no_serialize_flags=False
    ) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "instruction": self.serialize(
                    self.instruction,
                    hash_to_serial_id_cache,
                    ignore_no_serialize_flags,
                ),
                "inst_label": self.inst_label,
                "patch_label": self.patch_label,
                "inst_args": self.serialize(
                    self.inst_args,
                    hash_to_serial_id_cache,
                    ignore_no_serialize_flags,
                ),
                "inst_kwargs": self.serialize(
                    self.inst_kwargs,
                    hash_to_serial_id_cache,
                    ignore_no_serialize_flags,
                ),
            }
        )
        return state
