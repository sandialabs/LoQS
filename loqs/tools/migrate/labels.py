#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Rewrite pre-1.2 positional `InstructionLabel` construction to the
modern keyword form.

Two source shapes are recognized, both matching pre-1.2's `(instruction,
patch_label, inst_args, inst_kwargs)` convention: a direct
`InstructionLabel(...)` call with more than one positional argument, or a
bare 3-/4-tuple literal. A direct call is rewritten to a modern call
(`InstructionLabel("Increment", increment_by=2, patch_label="L0")`); a
bare tuple is rewritten to a bare `{"instruction": ..., **kwargs}` dict
instead, matching `InstructionLabel`'s own modern raw-dict shape, since
building a call there would introduce a new `InstructionLabel` reference
into a file that may never have needed to import it before. A 3-tuple
with a *string* 2nd element is deliberately **not** treated as a
candidate, even though it matches this shape for a per-patch instruction,
since it collides constantly with an unrelated, far more common pattern
-- a raw pyGSTi circuit-layer gate-label tuple, e.g. `("Gcphase", "A0",
"D4")`, has the exact same 3-string-element shape and is expected to
dominate real source files.

`instruction` and `patch_label` are carried through verbatim regardless
of their own shape -- modern keyword-form `InstructionLabel` accepts
either a bare name string or an already-resolved `Instruction` object
identically, so neither needs to be known statically. `inst_args`
similarly never needs to be inspected: if it isn't provably empty, it's
carried through verbatim under the reserved
[](api:LEGACY_PENDING_INST_ARGS) key, the same mechanism
[](api:QuantumProgram._label_kwargs) already uses to remap a pending
positional-args stash once the real instruction is available at run
time -- no static lookup needed either way. Only `inst_kwargs` must be a
literal dict with string-literal keys, since its individual entries are
spliced in as real keyword arguments; anything else is flagged for
manual review rather than guessed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from loqs.core.instructions.instructionlabel import LEGACY_PENDING_INST_ARGS
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
    the sequence can't be known statically). A literal `None` also counts
    as empty, matching `_remap_legacy_positional_args`'s own
    `tuple(inst_args or ())` runtime handling -- needed since
    `QuantumProgram_v1.json.gz`'s frozen source contains exactly this
    shape (`InstructionLabel(rus_key, patch_label, None, {...})`)."""
    if node is None or _is_none(node):
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
    can't be known statically). A literal `None` counts as empty too (see
    `_literal_sequence`)."""
    if node is None or _is_none(node):
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


def _build_dict(
    instruction_expr: cst.BaseExpression,
    keyword_args: Mapping[str, cst.BaseExpression],
) -> cst.Dict:
    """A bare `{"instruction": ..., **kwargs}` dict, matching
    `InstructionLabel`'s own modern raw-dict shape -- used instead of an
    explicit `InstructionLabel(...)` call so a rewritten bare tuple never
    needs a new `InstructionLabel` import added to the file."""
    elements = [
        cst.DictElement(
            key=cst.SimpleString('"instruction"'), value=instruction_expr
        )
    ]
    for key, value in keyword_args.items():
        elements.append(
            cst.DictElement(key=cst.SimpleString(f'"{key}"'), value=value)
        )
    return cst.Dict(elements=elements)


class _InstructionLabelRewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self) -> None:
        self.changed = False
        self.manual_review: list[ManualReviewItem] = []

    def _flag(self, node: cst.CSTNode, message: str) -> None:
        line = self.get_metadata(PositionProvider, node).start.line
        self.manual_review.append(ManualReviewItem(line=line, message=message))

    def _resolve_and_rewrite(
        self,
        original_node: cst.CSTNode,
        func: cst.BaseExpression | None,
        instruction_expr: cst.BaseExpression,
        patch_label_expr: cst.BaseExpression | None,
        inst_args_expr: cst.BaseExpression | None,
        inst_kwargs_expr: cst.BaseExpression | None,
        extra_keywords: Mapping[str, cst.BaseExpression],
        extra_args: Sequence[cst.Arg] = (),
    ) -> cst.BaseExpression | None:
        """Returns the rewritten node, or `None` if this candidate was
        only flagged (caller should leave the original node unchanged).

        Builds an `InstructionLabel(...)` call when `func` is given (an
        explicit call is being rewritten -- `func` is that call's own
        original callee, preserved verbatim), or a bare
        `{"instruction": ...}` dict when `func` is `None` (a bare tuple
        is being rewritten instead, where building a call would
        introduce a new `InstructionLabel` reference the file may not
        already import)."""
        literal_kwargs = _literal_str_keyed_dict(inst_kwargs_expr)
        if literal_kwargs is None:
            self._flag(
                original_node,
                "inst_kwargs isn't a literal dict with string-literal "
                "keys -- can't confidently splice its entries as keyword "
                "arguments; migrate by hand.",
            )
            return None

        remapped = dict(literal_kwargs)
        if patch_label_expr is not None and not _is_none(patch_label_expr):
            remapped["patch_label"] = patch_label_expr
        remapped.update(extra_keywords)
        # inst_args is carried through verbatim rather than statically
        # remapped to keyword names -- see the module docstring.
        if _literal_sequence(inst_args_expr) != []:
            remapped[LEGACY_PENDING_INST_ARGS] = inst_args_expr
        self.changed = True
        if func is None:
            return _build_dict(instruction_expr, remapped)
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
        # A `**kwargs`-style unpack already uses the modern convention --
        # its Arg has keyword=None like a positional one, but doesn't
        # count as positional or need remapping; carried through verbatim
        # rather than merged into named keys, since its contents aren't
        # known statically.
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
            None,
            instruction_expr,
            patch_label_expr,
            inst_args_expr,
            inst_kwargs_expr,
            {},
        )
        return rewritten if rewritten is not None else updated_node


def migrate_instruction_labels(source: str) -> MigrationResult:
    """Detect and rewrite pre-1.2 positional `InstructionLabel`
    construction throughout `source` (see the module docstring for exactly
    which shapes are recognized and when a confident rewrite is possible).

    Parameters
    ----------
    source:
        The full text of a `.py` file (or an extracted code cell).
    """
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    transformer = _InstructionLabelRewriter()
    new_module = wrapper.visit(transformer)
    new_source = new_module.code
    return MigrationResult(
        source=new_source,
        changed=transformer.changed,
        manual_review=transformer.manual_review,
    )
