#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Rewrite a confidently-resolvable `RepTuple(rep, qubits, reptype)` call
to the modern concrete [](api:OperationRep) subclass it actually meant,
instead of only flagging it (see [](api:RENAMES)'s own `RepTuple` entry).

`reptype` is only resolvable when it's a literal `GateRep.<NAME>` or
`InstrumentRep.<NAME>` attribute reference -- one of the ten pre-refactor
enum members `loqs/backends/reps/legacy.py`'s `_upgrade_legacy_gaterep`/
`_upgrade_legacy_instrumentrep` already dispatch on for decode, reused
here as this pass's own mapping table. `GateRep`/`InstrumentRep` are
matched by bare name only, not by resolving the actual import (unlike
[](api:rewrite_renames)'s `QualifiedNameProvider`-based matching) --
deliberately, since these names refer to the *old*, now-nonexistent enum
in this specific context, not the modern classes of the same name a real
import would resolve to. A bare int `reptype`, or any other non-literal
expression, is left unresolved: gate values 1-6 and instrument values
1-4 overlap, so a bare int is genuinely ambiguous without knowing which
enum it came from, and guessing wrong would construct the wrong class
outright rather than merely fail to rewrite.

For every `GateRep` target, and one `InstrumentRep` target
(`STIM_CIRCUIT_STR`), `rep` passes straight through as that class's own
single payload argument. The other three `InstrumentRep` targets'
`rep` was always a nested tuple that decode destructures positionally
(e.g. `reset, include_outcome = rep`); `NewClass(*rep,
qubit_labels=qubits)` reproduces that destructuring exactly for *any*
expression, not just a literal tuple, since Python's `*`-unpack enforces
the same arity a tuple-assignment would. `KrausGateRep` specifically
skips decode's `tp_check_abstol=None` (used there to avoid re-validating
already-accepted data) -- a live rewrite is fresh construction, not a
decode, so it gets the class's own default trace-preservation check like
any new `KrausGateRep(...)` call would.

The old `RepTuple` import is dropped once every reference to it in the
file has been resolved this way (see [](api:RemoveImportsVisitor));
`GateRep`/`InstrumentRep`'s own (old-enum-style) import is deliberately
left alone, since this pass doesn't know which module a bare `GateRep`/
`InstrumentRep` name was actually imported from without resolving it
properly, and removing the wrong one would be worse than leaving an
unused import behind.
"""

from __future__ import annotations

from collections.abc import Sequence

import libcst as cst
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import AddImportsVisitor, RemoveImportsVisitor
from libcst.metadata import MetadataWrapper, PositionProvider

from loqs.tools.migrate.report import (
    ManualReviewItem,
    MigrationResult,
    RewriteItem,
    remap_manual_review,
    remap_rewrites,
)

_OLD_REPTUPLE_MODULE = "loqs.backends.reps"
_GATEREP_MODULE = "loqs.backends.reps.gatereps"
_INSTRUMENTREP_MODULE = "loqs.backends.reps.instrumentreps"

_GATE_REPTYPES: dict[str, tuple[str, bool]] = {
    "UNITARY": ("UnitaryGateRep", False),
    "PTM": ("PTMGateRep", False),
    "QSIM_SUPEROPERATOR": ("QSimSuperopGateRep", False),
    "STIM_CIRCUIT_STR": ("StimCircuitGateRep", False),
    "PROBABILISTIC_STIM_OPERATIONS": ("ProbabilisticStimGateRep", False),
    "KRAUS_OPERATORS": ("KrausGateRep", False),
}
"""Old `GateRep.<NAME>` enum member -> `(modern concrete class, needs
destructuring)`, mirroring `legacy.py`'s `_upgrade_legacy_gaterep` --
`rep` always passes straight through as the single payload argument for
every gate type."""

_INSTRUMENT_REPTYPES: dict[str, tuple[str, bool]] = {
    "ZBASIS_PROJECTION": ("ZBasisProjectionInstrumentRep", True),
    "ZBASIS_PRE_POST_OPERATIONS": ("ZBasisPrePostInstrumentRep", True),
    "ZBASIS_OUTCOME_OPERATION_DICT": ("OutcomeOperationDictInstrumentRep", True),
    "STIM_CIRCUIT_STR": ("StimCircuitInstrumentRep", False),
}
"""Old `InstrumentRep.<NAME>` enum member -> `(modern concrete class,
needs destructuring)`, mirroring `legacy.py`'s
`_upgrade_legacy_instrumentrep`. Three of the four destructure `rep` into
multiple constructor arguments; `STIM_CIRCUIT_STR` passes `rep` straight
through instead, same as every gate type -- it also coincidentally
shares a name with a `_GATE_REPTYPES` entry, disambiguated by which
receiver (`GateRep`/`InstrumentRep`) it appears on, same as the original
pre-refactor enums were two distinct types."""

_ARG_NAMES = ("rep", "qubits", "reptype")

_NO_SPACE_EQUAL = cst.AssignEqual(
    whitespace_before=cst.SimpleWhitespace(""),
    whitespace_after=cst.SimpleWhitespace(""),
)


def _func_name(node: cst.BaseExpression) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return node.attr.value
    return None


def _extract_positional_or_keyword(
    node: cst.Call, names: Sequence[str]
) -> list[cst.BaseExpression] | None:
    """`node`'s arguments matched against a fixed `names` parameter list,
    positionally, by keyword, or a mix of both -- the same flexibility a
    real Python call has. `None` if a `*`/`**` splat is present, an
    unrecognized keyword is given, a name is filled both positionally and
    by keyword, or any name ends up unfilled."""
    if any(a.star for a in node.args):
        return None
    positional = [a.value for a in node.args if a.keyword is None]
    if len(positional) > len(names):
        return None
    keyword = {a.keyword.value: a.value for a in node.args if a.keyword is not None}
    if any(k not in names for k in keyword) or any(
        name in keyword for name in names[: len(positional)]
    ):
        return None

    values = list(positional)
    for name in names[len(positional) :]:
        if name not in keyword:
            return None
        values.append(keyword[name])
    return values


def _destructured_positional_args(rep_expr: cst.BaseExpression) -> list[cst.Arg]:
    """The positional `Arg`s a destructuring reptype's `rep` unpacks into.
    A literal tuple/list is destructured directly into separate
    arguments (more readable than a splat); any other expression falls
    back to a `*`-unpack, which reproduces the same destructuring for any
    expression that unpacks to the right arity, not just a literal."""
    if isinstance(rep_expr, (cst.Tuple, cst.List)) and all(
        isinstance(el, cst.Element) for el in rep_expr.elements
    ):
        return [cst.Arg(value=el.value) for el in rep_expr.elements]
    return [cst.Arg(value=rep_expr, star="*")]


def _resolve_reptype(node: cst.BaseExpression) -> tuple[str, str, bool] | None:
    """`(new_module, new_class, needs_destructure)` for a literal
    `GateRep.<NAME>`/`InstrumentRep.<NAME>` reptype (however the
    `GateRep`/`InstrumentRep` receiver itself was accessed, e.g. a dotted
    `some_module.GateRep.<NAME>`), or `None` if `node` isn't one of the
    ten recognized attribute references."""
    if not isinstance(node, cst.Attribute):
        return None
    receiver = _func_name(node.value)
    member = node.attr.value
    if receiver == "GateRep" and member in _GATE_REPTYPES:
        new_class, needs_destructure = _GATE_REPTYPES[member]
        return (_GATEREP_MODULE, new_class, needs_destructure)
    if receiver == "InstrumentRep" and member in _INSTRUMENT_REPTYPES:
        new_class, needs_destructure = _INSTRUMENT_REPTYPES[member]
        return (_INSTRUMENTREP_MODULE, new_class, needs_destructure)
    return None


class _RepTupleRewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, context: CodemodContext) -> None:
        self.context = context
        self.changed = False
        self.manual_review: list[ManualReviewItem] = []
        self.rewrites: list[RewriteItem] = []

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        if _func_name(updated_node.func) != "RepTuple":
            return updated_node

        extracted = _extract_positional_or_keyword(updated_node, _ARG_NAMES)
        if extracted is None:
            return updated_node
        rep_expr, qubits_expr, reptype_expr = extracted

        resolved = _resolve_reptype(reptype_expr)
        if resolved is None:
            line = self.get_metadata(PositionProvider, original_node).start.line
            self.manual_review.append(
                ManualReviewItem(
                    line=line,
                    message=(
                        "RepTuple(...) call found, but reptype isn't a "
                        "literal GateRep.<NAME>/InstrumentRep.<NAME> "
                        "reference -- can't confidently pick a concrete "
                        "OperationRep subclass to construct; migrate by "
                        "hand."
                    ),
                )
            )
            return updated_node

        new_module, new_class, needs_destructure = resolved
        AddImportsVisitor.add_needed_import(self.context, new_module, new_class)
        RemoveImportsVisitor.remove_unused_import(
            self.context, _OLD_REPTUPLE_MODULE, "RepTuple"
        )

        args = (
            _destructured_positional_args(rep_expr)
            if needs_destructure
            else [cst.Arg(value=rep_expr)]
        )
        args.append(
            cst.Arg(
                keyword=cst.Name("qubit_labels"),
                value=qubits_expr,
                equal=_NO_SPACE_EQUAL,
            )
        )
        self.changed = True
        line = self.get_metadata(PositionProvider, original_node).start.line
        self.rewrites.append(
            RewriteItem(line=line, message=f"RepTuple(...) -> {new_class}(...)")
        )
        return cst.Call(func=cst.Name(new_class), args=args)


def rewrite_reptuple_construction(source: str) -> MigrationResult:
    """Rewrite every confidently-resolvable `RepTuple(...)` call in
    `source` to the modern [](api:OperationRep) subclass it meant (see the
    module docstring), leaving anything else about `RepTuple` -- an
    unresolvable call, or a bare reference with no call at all -- for
    [](api:rewrite_renames)'s own generic deleted-name handling.

    Parameters
    ----------
    source:
        The full text of a `.py` file (or an extracted code cell).
    """
    # A cheap textual pre-filter before ever parsing, same reasoning
    # `rewrite_renames` already applies to its own rename table: avoids
    # real CST/import-machinery overhead for files that never mention
    # `RepTuple` at all -- the overwhelming majority of files scanned.
    if "RepTuple" not in source:
        return MigrationResult(source=source, changed=False, manual_review=[], rewrites=[])

    context = CodemodContext()
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    transformer = _RepTupleRewriter(context)
    new_module = wrapper.visit(transformer)
    new_module = AddImportsVisitor(context).transform_module(new_module)
    new_module = RemoveImportsVisitor(context).transform_module(new_module)
    new_source = new_module.code

    return MigrationResult(
        source=new_source,
        changed=transformer.changed,
        manual_review=remap_manual_review(source, new_source, transformer.manual_review),
        rewrites=remap_rewrites(source, new_source, transformer.rewrites),
    )
