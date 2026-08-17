#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Rewrite pre-#104 positional `InstructionLabel` construction to the
modern keyword form.

Two source shapes are recognized, both matching pre-#104's
`(instruction, patch_label, inst_args, inst_kwargs)` convention:

1. A direct `InstructionLabel(instruction, patch_label, inst_args,
   inst_kwargs)` call with more than one positional argument.
2. A bare tuple literal of length 3 or 4, matching the same heuristic used
   for [](api:loqs.internal.legacy.detect_legacy_construction)'s decode-time
   cousin: a 4-tuple whose 3rd element is itself an empty tuple (the
   `inst_args` placeholder), or a 3-tuple whose 2nd element is `None` (a
   global, unpatched instruction with non-empty `inst_args`). A 3-tuple
   with a *string* 2nd element is deliberately **not** treated as a
   candidate, even though it matches pre-#104's shape for a per-patch
   instruction: real testing against `docs/notebooks/buildinstruction.md`
   found this collides constantly with an unrelated, far more common
   pattern -- a raw pyGSTi circuit-layer gate-label tuple, e.g.
   `("Gcphase", "A0", "D4")`, has the exact same 3-string-element shape.
   Since that collision is expected to dominate real source files (pyGSTi
   circuit layers appear throughout the codebase, this specific legacy
   shape does not), this case is left undetected rather than flooding
   every report with false positives.

Rewriting either shape requires knowing the target [](api:Instruction)'s
`param_priorities`/`param_alias` (to map a positional slot onto its real
keyword name) -- the same algorithm [](api:InstructionLabel._from_decoded_attrs)
already uses at decode time, reimplemented here to work on unevaluated CST
expression nodes instead of runtime values, so an argument's *source text*
can be carried over verbatim without needing its runtime value. This is
only possible when:

- the instruction is named by a string literal that resolves against the
  caller-supplied instruction registry (see
  [](api:loqs.tools.migrate.config)) -- an already-resolved `Instruction`
  object (a bare name/attribute reference) already works fine as-is today,
  with only a `DeprecationWarning` (see `InstructionLabel.__init__`), so
  this is flagged as a low-priority style note rather than an error; and
- `inst_args`/`inst_kwargs` are themselves literal tuple/list/dict
  expressions (their *elements'* values may be arbitrary expressions,
  spliced through unevaluated -- only their own shape needs to be static).

Anything else is flagged for manual review rather than guessed, per this
tool's overall design goal (see the `feat-97` plan's Sub-problem B).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from loqs.core.instructions.instruction import Instruction
from loqs.tools.migrate.report import ManualReviewItem, MigrationResult


def _func_name(node: cst.BaseExpression) -> str | None:
    """The bare name at the end of a `Call`'s `func`, for a plain `Name`
    or a dotted `Attribute` (e.g. `loqs.core.InstructionLabel`)."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return node.attr.value
    return None


def _is_empty_tuple(node: cst.BaseExpression) -> bool:
    return isinstance(node, cst.Tuple) and len(node.elements) == 0


def _is_none(node: cst.BaseExpression) -> bool:
    return isinstance(node, cst.Name) and node.value == "None"


def _string_value(node: cst.BaseExpression) -> str | None:
    if isinstance(node, cst.SimpleString):
        try:
            value = node.evaluated_value
        except Exception:
            return None
        return value if isinstance(value, str) else None
    return None


def _literal_sequence(
    node: cst.BaseExpression | None,
) -> list[cst.BaseExpression] | None:
    """A literal tuple/list's element expressions, or `None` if `node`
    isn't one (or contains a `*`-unpacked element, whose contribution to
    the sequence can't be known statically)."""
    if node is None:
        return []
    if not isinstance(node, (cst.Tuple, cst.List)):
        return None
    elements = []
    for el in node.elements:
        if not isinstance(el, cst.Element):
            return None
        elements.append(el.value)
    return elements


def _literal_str_keyed_dict(
    node: cst.BaseExpression | None,
) -> dict[str, cst.BaseExpression] | None:
    """A literal dict's `{str_literal_key: value_expr}` entries, or `None`
    if `node` isn't a dict literal with exclusively string-literal keys
    (a `**`-unpacked entry or a non-literal key means the real key set
    can't be known statically)."""
    if node is None:
        return {}
    if not isinstance(node, cst.Dict):
        return None
    result: dict[str, cst.BaseExpression] = {}
    for el in node.elements:
        if not isinstance(el, cst.DictElement):
            return None
        key = _string_value(el.key)
        if key is None:
            return None
        result[key] = el.value
    return result


def _remap_positional(
    instruction: Instruction,
    inst_args: Sequence[cst.BaseExpression],
    inst_kwargs: Mapping[str, cst.BaseExpression],
) -> dict[str, cst.BaseExpression]:
    """CST-node analog of `InstructionLabel._remap_legacy_positional_args`:
    same positional-index -> aliased-keyword mapping, but over unevaluated
    expression nodes rather than runtime values."""
    merged = dict(inst_kwargs)
    for i, key in enumerate(instruction.param_priorities.keys()):
        if i < len(inst_args):
            merged[instruction.param_alias(key)] = inst_args[i]
    return merged


_NO_SPACE_EQUAL = cst.AssignEqual(
    whitespace_before=cst.SimpleWhitespace(""),
    whitespace_after=cst.SimpleWhitespace(""),
)


def _build_call(
    func: cst.BaseExpression,
    instruction_expr: cst.BaseExpression,
    keyword_args: Mapping[str, cst.BaseExpression],
    extra_args: Sequence[cst.Arg] = (),
) -> cst.Call:
    args = [cst.Arg(value=instruction_expr)]
    for key, value in keyword_args.items():
        args.append(
            cst.Arg(keyword=cst.Name(key), value=value, equal=_NO_SPACE_EQUAL)
        )
    # Any `**kwargs`-style unpack from the original call is carried
    # through verbatim, last, matching its original precedence.
    args.extend(
        a.with_changes(comma=cst.MaybeSentinel.DEFAULT) for a in extra_args
    )
    return cst.Call(func=func, args=args)


class _InstructionLabelRewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, instructions: Mapping[str, Instruction]) -> None:
        self._instructions = instructions
        self.changed = False
        self.manual_review: list[ManualReviewItem] = []

    def _flag(self, node: cst.CSTNode, message: str) -> None:
        line = self.get_metadata(PositionProvider, node).start.line
        self.manual_review.append(ManualReviewItem(line=line, message=message))

    def _resolve_and_rewrite(
        self,
        original_node: cst.CSTNode,
        func: cst.BaseExpression,
        instruction_expr: cst.BaseExpression,
        patch_label_expr: cst.BaseExpression | None,
        inst_args_expr: cst.BaseExpression | None,
        inst_kwargs_expr: cst.BaseExpression | None,
        extra_keywords: Mapping[str, cst.BaseExpression],
        extra_args: Sequence[cst.Arg] = (),
    ) -> cst.BaseExpression | None:
        """Returns the rewritten node, or `None` if this candidate was
        only flagged (caller should leave the original node unchanged)."""
        instr_name = _string_value(instruction_expr)
        if instr_name is None:
            self._flag(
                original_node,
                "Old-style positional InstructionLabel construction with "
                "an already-resolved Instruction (not a string name) -- "
                "this already works today via a DeprecationWarning, but "
                "wasn't auto-modernized; rewrite to keyword form by hand "
                "if desired.",
            )
            return None
        instruction = self._instructions.get(instr_name)
        if instruction is None:
            self._flag(
                original_node,
                f"Could not resolve instruction {instr_name!r} against the "
                "configured instruction registry for this file -- add or "
                "correct the migration config, or migrate by hand.",
            )
            return None
        literal_args = _literal_sequence(inst_args_expr)
        literal_kwargs = _literal_str_keyed_dict(inst_kwargs_expr)
        if literal_args is None or literal_kwargs is None:
            self._flag(
                original_node,
                "inst_args/inst_kwargs aren't literal tuple/list/dict "
                "expressions -- can't statically remap positions to "
                "keyword names; migrate by hand.",
            )
            return None

        remapped = _remap_positional(instruction, literal_args, literal_kwargs)
        if patch_label_expr is not None and not _is_none(patch_label_expr):
            remapped["patch_label"] = patch_label_expr
        remapped.update(extra_keywords)
        self.changed = True
        return _build_call(func, instruction_expr, remapped, extra_args)

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        if _func_name(updated_node.func) != "InstructionLabel":
            return updated_node

        if any(a.star == "*" for a in updated_node.args):
            # A genuine `InstructionLabel(*expr)` positional splat --
            # unlike `**kwargs` below, its contribution to positional-arg
            # count can't be known without evaluating `expr`.
            self._flag(
                original_node,
                "InstructionLabel(*expr) splat call -- can't statically "
                "resolve a starred argument's contents; migrate by hand.",
            )
            return updated_node

        positional = [
            a.value
            for a in updated_node.args
            if a.keyword is None and a.star == ""
        ]
        keyword_args = {
            a.keyword.value: a.value
            for a in updated_node.args
            if a.keyword is not None
        }
        # A `**kwargs`-style unpack (e.g. InstructionLabel(instruction,
        # **kwargs)) is already the modern calling convention -- its
        # Arg has keyword=None like a positional one, but doesn't affect
        # positional-arg counting or need remapping; carried through
        # verbatim on any rewrite instead of merged into named keys,
        # since its contents can't be known statically either.
        double_starred = [a for a in updated_node.args if a.star == "**"]
        if len(positional) < 2:
            return updated_node  # already modern (0-1 positional arg)

        instruction_expr, *rest = positional
        patch_label_expr = rest[0] if len(rest) >= 1 else None
        inst_args_expr = rest[1] if len(rest) >= 2 else None
        inst_kwargs_expr = rest[2] if len(rest) >= 3 else None

        rewritten = self._resolve_and_rewrite(
            original_node,
            updated_node.func,
            instruction_expr,
            patch_label_expr,
            inst_args_expr,
            inst_kwargs_expr,
            keyword_args,
            double_starred,
        )
        return rewritten if rewritten is not None else updated_node

    def leave_Tuple(
        self, original_node: cst.Tuple, updated_node: cst.Tuple
    ) -> cst.BaseExpression:
        if not all(isinstance(e, cst.Element) for e in updated_node.elements):
            return updated_node
        values = [e.value for e in updated_node.elements]

        if len(values) == 4 and _is_empty_tuple(values[2]):
            (
                instruction_expr,
                patch_label_expr,
                inst_args_expr,
                inst_kwargs_expr,
            ) = values
        elif len(values) == 3 and _is_none(values[1]):
            instruction_expr, patch_label_expr, inst_args_expr = values
            inst_kwargs_expr = None
        else:
            return updated_node

        rewritten = self._resolve_and_rewrite(
            original_node,
            cst.Name("InstructionLabel"),
            instruction_expr,
            patch_label_expr,
            inst_args_expr,
            inst_kwargs_expr,
            {},
        )
        return rewritten if rewritten is not None else updated_node


def migrate_instruction_labels(
    source: str, instructions: Mapping[str, Instruction]
) -> MigrationResult:
    """Detect and rewrite pre-#104 positional `InstructionLabel`
    construction throughout `source` (see the module docstring for exactly
    which shapes are recognized and when a confident rewrite is possible).

    Parameters
    ----------
    source:
        The full text of a `.py` file (or an extracted code cell).
    instructions:
        A `{name: Instruction}` registry (typically a `QECCode.instructions`
        dict, or several merged together) used to resolve a string-named
        instruction's `param_priorities`/`param_alias`, per
        [](api:loqs.tools.migrate.config).
    """
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    transformer = _InstructionLabelRewriter(instructions)
    new_module = wrapper.visit(transformer)
    new_source = new_module.code
    return MigrationResult(
        source=new_source,
        changed=transformer.changed,
        manual_review=transformer.manual_review,
    )
