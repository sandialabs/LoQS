#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from collections.abc import Mapping, Sequence
import h5py
import textwrap
from typing import ClassVar, TypeAlias, TypeVar

from loqs.core.instructions import Instruction, InstructionLabel
from loqs.core.instructions.instructionlabel import InstructionLabelLike
from loqs.internal import Displayable
from loqs.internal.encoder.hdf5encoder import HDF5Encoder
from loqs.internal.encoder.jsonencoder import JSONEncoder
from loqs.internal.serializable import Serializable

T = TypeVar("T", bound="InstructionStack")

InstructionStackLike: TypeAlias = (
    "InstructionStack | InstructionLabelLike | Sequence[InstructionLabelLike] | None"
)
"""Objects that can be cast to a [](api:InstructionStack)."""


class InstructionStack(Sequence[InstructionLabel], Displayable):
    """A list of [](api:InstructionLabel) objects to execute.

    This is intended to be an immutable list of [](api:InstructionLabel)
    objects to execute. Stack manipulations return a modified copy.

    Each entry is cast through [](api:InstructionLabel.from_raw), so a
    stack can freely mix every supported raw shape: a bare global
    instruction name, the succinct `(instruction, patch_label)` 2-tuple,
    a full dict for anything needing more (extra kwargs, or multiple
    patches via `"patch_labels"`).

    Examples
    --------
    >>> from loqs.core.instructions import InstructionStack
    >>> stack = InstructionStack([
    ...     "Init State",                                 # bare global instruction
    ...     ("H", "L0"),                                   # tuple sugar, single patch
    ...     {
    ...         "instruction": "FT Logical X Measure Classical Decoder",
    ...         "patch_label": "L0",
    ...         "flagged_check": "XZIIZ",
    ...         "flagged_check_order": [4, 0, 1],
    ...     },                                              # dict, single patch + extra kwargs
    ...     {
    ...         "instruction": "CNOT Bookkeeping",
    ...         "patch_labels": {"ctrl": "L0", "tgt": "L1"},
    ...     },                                              # dict, multi-patch
    ... ])
    >>> len(stack)
    4
    >>> stack[0]["instruction"]
    'Init State'
    >>> stack[1]["patch_label"]
    'L0'
    >>> stack[2]["flagged_check"]
    'XZIIZ'
    >>> stack[3]["patch_labels"]
    {'ctrl': 'L0', 'tgt': 'L1'}
    """

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    _SERIALIZE_ATTRS = ["_instructions"]

    _SERIALIZE_ATTRS_MAP = {"_instructions": "instructions"}

    _instructions: list[InstructionLabel]
    """Internal list of [](api:InstructionLabels)"""

    def __init__(
        self, instructions: InstructionStackLike = None
    ) -> None:
        """
        Parameters
        ----------
        instructions:
            A sequence of [](api:InstructionLabel) castable things.
            Defaults to `None`, which creates an empty list.
        """
        self._instructions = []
        if isinstance(instructions, InstructionStack):
            self._instructions = instructions._instructions
            return
        if instructions is None or (
            isinstance(instructions, Sequence) and not len(instructions)
        ):
            self._instructions = []
            return
        if isinstance(instructions, (Instruction, str, InstructionLabel)):
            self._instructions = [InstructionLabel.from_raw(instructions)]
            return

        # If we are here, we are a sequence of some kind. A tuple is always
        # one InstructionLabel's own raw form (matching InstructionLabelLike's
        # tuple sugar); anything else (e.g. a list) is a sequence of raw
        # items to convert individually. This can't instead be decided by
        # inspecting instructions[0]'s type, as done previously: a list of
        # multiple bare Instruction/str labels (e.g. ["LabelA", "LabelB"])
        # would then be misread as one InstructionLabel's own raw form,
        # silently dropping every entry past the first two.
        if isinstance(instructions, tuple):
            self._instructions = [InstructionLabel.from_raw(instructions)]
            return

        for inst in instructions:
            self._instructions.append(InstructionLabel.from_raw(inst))

    def __getitem__(self, i):
        return self._instructions[i]

    def __len__(self):
        return len(self._instructions)

    @classmethod
    def _from_decoded_attrs(cls, attr_dict) -> "InstructionStack":
        """Build from decoded items, casting each through
        `InstructionLabel.from_raw` -- still needed since a modern
        `InstructionLabel` decodes as a plain `dict`, with no
        `encode_type` of its own."""
        obj = cls()
        obj._instructions = [
            InstructionLabel.from_raw(item) for item in attr_dict["_instructions"]
        ]
        return obj

    def __str__(self):
        if len(self):
            s = f"InstructionStack with {len(self)} items:\n"
            for i, inst in enumerate(self._instructions):
                si = str(inst)
                si = textwrap.indent(si, "  ")
                s += si
            return s
        else:
            return "Empty InstructionStack"

    def append_instruction(
        self, item: InstructionLabelLike
    ) -> InstructionStack:
        """Add an entry to the end of the stack.

        Parameters
        ----------
        item:
            The item to add

        Returns
        -------
        InstructionStack
            The modified stack
        """
        return self.insert_instruction(len(self), item)

    def append_instructions(
        self, items: Sequence[InstructionLabelLike]
    ) -> InstructionStack:
        """Add a series of entries to the end of the stack.

        Parameters
        ----------
        items:
            The items to add

        Returns
        -------
        InstructionStack
            The modified stack
        """
        return self.insert_instructions(len(self), items)

    def delete_instruction(self, i: int) -> InstructionStack:
        """Remove an entry from the stack.

        Parameters
        ----------
        i:
            The index to remove

        Returns
        -------
        InstructionStack
            The modified stack
        """
        instructions = self._instructions.copy()
        del instructions[i]
        return InstructionStack(instructions)

    def insert_instruction(
        self, i: int, item: InstructionLabelLike
    ) -> InstructionStack:
        """Add an entry to a position in the stack.

        Parameters
        ----------
        i:
            The index of insertion

        item:
            The item to add

        Returns
        -------
        InstructionStack
            The modified stack
        """
        instructions = self._instructions.copy()
        instructions.insert(i, InstructionLabel.from_raw(item))
        return InstructionStack(instructions)

    def insert_instructions(
        self, i: int, items: Sequence[InstructionLabelLike]
    ) -> InstructionStack:
        """Add a series of entries to a position in the stack.

        Parameters
        ----------
        i:
            The index of insertion

        items:
            The items to add

        Returns
        -------
        InstructionStack
            The modified stack
        """
        instructions = self._instructions.copy()
        items_to_add = [InstructionLabel.from_raw(item) for item in items]
        instructions = (
            self._instructions[:i] + items_to_add + self._instructions[i:]
        )
        return InstructionStack(instructions)

    def pop_instruction(
        self,
    ) -> tuple[InstructionLabel, InstructionStack]:
        """Remove and return the first entry on the stack.

        Returns
        -------
        InstructionLabel, InstructionStack
            The first instruction and modified stack without
            the first element
        """
        return self._instructions[0], InstructionStack(self._instructions[1:])
