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
from dataclasses import dataclass
from typing import TypeAlias
import warnings

from loqs.core.instructions.instruction import Instruction

InstructionLabelLike: TypeAlias = (
    "Instruction | str | tuple[Instruction | str] | tuple[Instruction | str, str] | Mapping | InstructionLabel"
)
"""Objects that can be cast to a [](api:InstructionLabel)."""


def _remap_legacy_positional_args(
    instruction: Instruction, inst_args: tuple, inst_kwargs: dict
) -> dict:
    """Map pre-#104 positional `inst_args` onto their real keyword names.

    The historical convention (confirmed directly from the deleted
    pre-#104 `QuantumProgram._run_shot`/`_collect_kwarg`, not guessed):
    positional index *i* corresponds to the *i*-th key of
    `instruction.param_priorities` (dict insertion order), stored under
    that key's *aliased* name (`instruction.param_alias(key)`) -- the same
    name `inst_kwargs` already used. A positional value takes precedence
    over a same-key `inst_kwargs` entry, matching the old `_collect_kwarg`,
    which checked the positional slot before falling back to a keyword.
    """
    merged = dict(inst_kwargs)
    for i, key in enumerate(instruction.param_priorities.keys()):
        if i < len(inst_args):
            merged[instruction.param_alias(key)] = inst_args[i]
    return merged


@dataclass
class _PendingLegacyInstructionLabel:
    """A not-yet-resolvable pre-#104 `InstructionLabel`, found while
    decoding an old file.

    Produced by `InstructionLabel._from_decoded_attrs` when the label's
    `instruction` is a bare string name rather than an already-resolved
    `Instruction` -- resolving a name requires sibling `global_instructions`/
    a current patch layout that a nested object's own decode never sees
    (mirrors `Serializable`'s existing `DeferredRef` pattern for the same
    kind of "needs context from further up the decode" problem). Finished
    by `QuantumProgram`'s own `_from_decoded_attrs`, which does have that
    context once all of its own sibling attributes are decoded.
    """

    inst_label: str
    patch_label: str | None
    inst_args: tuple
    inst_kwargs: dict

    def _unresolved_error(self) -> RuntimeError:
        return RuntimeError(
            f"This old-format InstructionLabel (referencing "
            f"{self.inst_label!r}) was never resolved during decode -- a "
            "bare InstructionStack has no global_instructions/patch "
            "context to resolve an old string-named label against. "
            "Decode it as part of its original QuantumProgram instead, "
            "which finishes resolving any such label once its sibling "
            "attributes are available."
        )

    def __getitem__(self, key: object) -> object:
        raise self._unresolved_error()

    def get(self, key: object, default: object = None) -> object:
        raise self._unresolved_error()


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

    def __init__(
        self,
        instruction: Instruction | str,
        *legacy_positional_args: object,
        **kwargs: object,
    ) -> None:
        """
        Parameters
        ----------
        instruction:
            Either an [](api:Instruction) or a `str` name to resolve
            later, stored under the reserved `"instruction"` key.

        *legacy_positional_args:
            Deprecated. Pre-#104 code called this
            `(patch_label, inst_args, inst_kwargs)` positionally; passing
            any of these warns and remaps them onto their real keyword
            names (see [](api:InstructionLabel._from_decoded_attrs)),
            requiring `instruction` to already be a resolved
            [](api:Instruction) (a bare name can't be remapped without
            knowing its parameters). New code should use keyword
            arguments directly instead.

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
        if legacy_positional_args:
            if not isinstance(instruction, Instruction):
                raise TypeError(
                    "Old-style positional InstructionLabel(name, patch_label, "
                    "inst_args, inst_kwargs) construction found a bare "
                    "instruction name, which can't be remapped without already "
                    "knowing the target Instruction's parameters -- pass an "
                    "already-resolved Instruction object instead, or migrate to "
                    "the modern keyword form."
                )
            warnings.warn(
                "Old-style positional InstructionLabel(instruction, patch_label, "
                "inst_args, inst_kwargs) construction is deprecated; use keyword "
                "arguments instead. Remapping positional arguments on your behalf.",
                DeprecationWarning,
                stacklevel=2,
            )
            padded = list(legacy_positional_args) + [None, (), {}]
            patch_label, inst_args, inst_kwargs = padded[0], padded[1], padded[2]
            remapped = _remap_legacy_positional_args(
                instruction, tuple(inst_args or ()), dict(inst_kwargs or {})
            )
            if patch_label is not None:
                remapped["patch_label"] = patch_label
            remapped.update(kwargs)
            super().__init__(instruction=instruction, **remapped)
            return
        super().__init__(instruction=instruction, **kwargs)

    def __repr__(self) -> str:
        return f"InstructionLabel({dict.__repr__(self)})"

    @classmethod
    def _from_decoded_attrs(
        cls, attr_dict: Mapping
    ) -> "InstructionLabel | _PendingLegacyInstructionLabel":
        """Decode-only compatibility for the pre-#104 tuple-attrs shape.

        Old files recorded `InstructionLabel` as a genuine `Serializable`
        object with 5 attrs (`instruction`, `inst_label`, `patch_label`,
        `inst_args`, `inst_kwargs`) rather than today's plain dict shape --
        confirmed directly against real fixture bytes, not assumed. Modern
        `InstructionLabel` is a plain `dict` subclass, never encoded with
        `encode_type: "Serializable"`, so this classmethod is only ever
        reached for genuinely old data (safe to add even though the class
        no longer inherits `Serializable` -- decode dispatches to it by
        duck-typing alone).

        When `instruction` is already a resolved [](api:Instruction), the
        full remap can happen right here, standalone. When it's `None`
        (a bare `inst_label` string instead), resolving it needs sibling
        `global_instructions`/patch context this nested object's own
        decode never sees -- a [](api:_PendingLegacyInstructionLabel)
        placeholder is returned instead, finished later by
        [](api:QuantumProgram._from_decoded_attrs).
        """
        instruction = attr_dict.get("instruction")
        inst_label = attr_dict.get("inst_label")
        patch_label = attr_dict.get("patch_label")
        inst_args = tuple(attr_dict.get("inst_args") or ())
        inst_kwargs = dict(attr_dict.get("inst_kwargs") or {})

        if instruction is not None:
            remapped = _remap_legacy_positional_args(
                instruction, inst_args, inst_kwargs
            )
            if patch_label is not None:
                remapped["patch_label"] = patch_label
            return cls(instruction, **remapped)

        assert isinstance(inst_label, str)
        return _PendingLegacyInstructionLabel(
            inst_label=inst_label,
            patch_label=patch_label,
            inst_args=inst_args,
            inst_kwargs=inst_kwargs,
        )

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
            # Only the (instruction, patch_label) 2-tuple is kept as sugar;
            # a longer tuple can't be resolved to the right keywords
            # without already knowing the target instruction's parameters.
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


def _finish_pending_legacy_label(
    pending: _PendingLegacyInstructionLabel, instruction: Instruction
) -> InstructionLabel:
    """Complete a [](api:_PendingLegacyInstructionLabel) once its target
    [](api:Instruction) has been resolved (see
    `QuantumProgram._from_decoded_attrs`)."""
    remapped = _remap_legacy_positional_args(
        instruction, pending.inst_args, pending.inst_kwargs
    )
    if pending.patch_label is not None:
        remapped["patch_label"] = pending.patch_label
    return InstructionLabel(instruction, **remapped)
