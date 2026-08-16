#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""[](api:InstructionLabel) definition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from loqs.core.instructions.instruction import Instruction

InstructionLabelLike: TypeAlias = (
    "Instruction | str | tuple[Instruction | str] | tuple[Instruction | str, str] | Mapping | InstructionLabel"
)
"""Objects that can be cast to a [](api:InstructionLabel)."""


class InstructionLabel(dict):
    """A dict-based instruction label, an element of an [](api:InstructionStack).

    Canonical shape: `{"instruction": Instruction | str, **kwargs}`. The
    reserved `"instruction"` key holds either an already-built
    [](api:Instruction), or the `str` name of one to resolve later
    (globally, or within a patch). Every other key is forwarded as a kwarg
    candidate for the resolved instruction's apply function -- there is no
    hardcoded positional slot for anything, and no separate "args" vs.
    "kwargs" split. By convention (not by hardcoded position, and with no
    dedicated accessor on this class -- read keys directly), two keys are
    given special meaning by the rest of the framework:

    - `"patch_label"`: a single patch this instruction targets.
    - `"patch_labels"`: a `Mapping[str, str]` naming *multiple* patches
      this instruction targets (for multi-patch instructions).

    Both are entirely optional, ordinary dict keys -- an instruction with
    neither is a "global" instruction, resolved from
    [](api:InstructionStack.global_instructions) rather than a specific
    patch's own instruction set.

    A handful of shorter forms are accepted by [](api:from_raw) as sugar
    for the common single-patch case: a bare `str`/[](api:Instruction) (a
    global instruction with no extra kwargs), or a `(instruction,
    patch_label)` 2-tuple. Anything needing more than that -- multiple
    patches, or any other kwarg -- must use the dict form.

    Examples
    --------
    >>> from loqs.core.instructions import InstructionLabel

    The dict form and the `(instruction, patch_label)` tuple sugar are
    equivalent for the common single-patch case:

    >>> InstructionLabel.from_raw({"instruction": "H", "patch_label": "L0"}) == \\
    ...     InstructionLabel.from_raw(("H", "L0"))
    True

    Any other kwarg an instruction's apply function needs is just another
    key, with no positional placeholder required to reach it:

    >>> label = InstructionLabel(
    ...     "FT Logical X Measure Classical Decoder",
    ...     patch_label="L0",
    ...     flagged_check="XZIIZ",
    ...     flagged_check_order=[4, 0, 1],
    ... )
    >>> label["flagged_check"]
    'XZIIZ'

    A multi-patch instruction names its patches under `"patch_labels"`
    instead of the single-patch `"patch_label"`:

    >>> multi = InstructionLabel(
    ...     "CNOT Bookkeeping", patch_labels={"ctrl": "L0", "tgt": "L1"}
    ... )
    >>> multi.get("patch_label") is None
    True
    >>> multi.get("patch_labels")
    {'ctrl': 'L0', 'tgt': 'L1'}

    Tuples longer than the 2-element sugar are rejected -- use the dict
    form instead:

    >>> InstructionLabel.from_raw(("H", "L0", (), {"flagged_check": "XZIIZ"}))
    Traceback (most recent call last):
        ...
    TypeError: Tuples longer than 2 elements are no longer supported -- use the dict form instead, e.g. {"instruction": ..., "patch_label": ..., <other kwargs>}.
    """

    def __init__(self, instruction: Instruction | str, **kwargs: object) -> None:
        """
        Parameters
        ----------
        instruction:
            Either an [](api:Instruction) or a `str` name to resolve
            later, stored under the reserved `"instruction"` key.

        **kwargs:
            Any other data this label carries, forwarded as kwarg
            candidates for the resolved instruction's apply function.
            `patch_label`/`patch_labels` are conventionally-recognized
            keys (see the class docstring) but are otherwise ordinary.
        """
        if not isinstance(instruction, (Instruction, str)):
            raise TypeError(
                f"instruction must be an Instruction or str, got {type(instruction)!r}"
            )
        super().__init__(instruction=instruction, **kwargs)

    def __repr__(self) -> str:
        return f"InstructionLabel({dict.__repr__(self)})"

    @classmethod
    def from_raw(cls, obj: object) -> InstructionLabel:
        """Build an [](api:InstructionLabel) from a loosely-typed raw value.

        Several call sites hand this class a genuinely ambiguous raw
        blob -- a bare `str`/[](api:Instruction), a short tuple to unpack,
        an already-built [](api:InstructionLabel), or a kwarg dict --
        which a constructor's positional-argument signature alone cannot
        disambiguate (in particular, it cannot un-blob a tuple handed to
        it as a single object).

        Parameters
        ----------
        obj:
            A raw value that is either:
            - Already an [](api:InstructionLabel) object
            - A kwarg dict that is passed into the constructor
            - A bare `str`/[](api:Instruction), or a `(instruction,
              patch_label)` 2-tuple

        Returns
        -------
            An [](api:InstructionLabel) object
        """
        if isinstance(obj, InstructionLabel):
            # We are already the correct class, perform no copy
            return obj
        if isinstance(obj, Mapping):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        if isinstance(obj, (Instruction, str)):
            return cls(obj)
        if isinstance(obj, Sequence):
            # Only the succinct (instruction, patch_label) 2-tuple is kept
            # as positional sugar -- anything needing more (multiple
            # patches, or any other kwarg) must use the dict form, since a
            # historical inst_args position could only ever be mapped
            # onto the right keyword by knowing the target instruction's
            # real parameter order, which isn't available here.
            if len(obj) == 1:
                return cls(obj[0])
            if len(obj) == 2:
                return cls(obj[0], patch_label=obj[1])
            raise TypeError(
                "Tuples longer than 2 elements are no longer supported -- "
                'use the dict form instead, e.g. {"instruction": ..., '
                '"patch_label": ..., <other kwargs>}.'
            )
        raise TypeError(f"Cannot cast {obj!r} to an InstructionLabel")
