#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Straight `(module, name)` rename/removal rewriting for old source files.

Reuses [](api:Serializable)'s own `IMPORT_LOCATION_CHANGES_BY_VERSION` table
(the single source of truth for every version's class relocations/renames)
as its data, via `_get_cumulative_changes(0)`, plus a small number of
additional entries that never needed a decode-time compatibility entry at
all (e.g. a class renamed before its old name ever shipped), but are still
worth fixing in a user's own source files.

Two entries are deliberately *overridden* to a deleted-outright (`None`)
target rather than reused as-is, even though a real decode-time target
exists for each: `RepTuple` and `STIMDictNoiseModel`. Decoding
already-serialized data only ever needs to know *where the modern class
lives*, since the decoded attribute values are already correctly shaped
for it. Rewriting live source code is a different problem -- it needs the
old *constructor call* to also make sense as a call to the new class, and
for these two specifically it doesn't: `RepTuple(rep, qubits, reptype)`'s
`reptype` dispatches across ~10 differently-shaped concrete `GateRep`/
`InstrumentRep` classes (no single new class to rewrite the call to), and
`STIMDictNoiseModel(model_or_dicts, ...)` takes its gate/instrument data
as one positional tuple where `DictNoiseModel(gate_dict, inst_dict, ...)`
takes two separate positional arguments -- a blind text rewrite of either
would produce code that runs but silently misbehaves or breaks outright.
Both are still *constructible* today via a deprecated compatibility shim
(see `loqs.backends.model.__init__` for `STIMDictNoiseModel`; `RepTuple`
has no such shim, since even its shim's own realistic call pattern -- an
old `GateRep`/`InstrumentRep` enum member as `reptype` -- fails on that
attribute access before a shim could ever run), so flagging rather than
either rewriting or hard-erroring is the honest outcome here.

[](api:Serializable._update_imports) is a thin wrapper around this same
function, so frozen function source and whole user source files share one
rewrite mechanism, built on `libcst`'s `RenameCommand`
(`libcst.codemod.commands.rename`). `RenameCommand` resolves real code
references via `QualifiedNameProvider`, so it can tell an actual usage of
a renamed class apart from the same text appearing in a docstring,
comment, or unrelated identifier, and updates both the import statement
and every usage site together.

A few real `RenameCommand` behaviors worth knowing about before trusting
this pass blindly:

1. It always adds the new import at module scope, even when the old one
   was function-local (e.g. a lazy import used to avoid a circular
   import, a real pattern in this codebase). Still valid Python, but
   changes where the import happens.
2. It drops a renamed import entirely if the renamed name ends up with no
   remaining real code reference (i.e. it was only ever mentioned in a
   comment/docstring, or never used at all) -- it does not distinguish
   "genuinely unused" from "moved but the rest of this rename batch
   already removed its only usage".
3. Chaining several `RenameCommand` passes over the same module (one per
   renamed name, done here since each has its own old/new location) can
   leave a stray blank line and trailing comma behind in a multi-name
   `from ... import (...)` block when an earlier pass's partial cleanup
   isn't reformatted by a later pass. Still valid Python, just untidy.

None of these are incorrect Python, but review the diff before committing
rather than applying a whole-file rewrite unseen, same as any other
automated rewrite.
"""

from __future__ import annotations

from collections.abc import Mapping

import libcst as cst
from libcst.codemod import CodemodContext
from libcst.codemod.commands.rename import RenameCommand
from libcst.helpers import get_full_name_for_node
from libcst.metadata import (
    MetadataWrapper,
    PositionProvider,
    QualifiedNameProvider,
)

from loqs.internal.serializable import Serializable
from loqs.tools.migrate.report import ManualReviewItem, MigrationResult

_OLD_INSTRUMENT_REP_NAME = "ZBasisOutcomeOperationDictInstrumentRep"
"""The pre-1.2 name being renamed away from -- named as its own constant
so this occurrence reads as intentional, not a stray leftover reference.
"""

RENAMES: dict[tuple[str, str], tuple[str, str] | None] = {
    **Serializable._get_cumulative_changes(0),
    # ZBasis...InstrumentRep -> OutcomeOperationDictInstrumentRep: no
    # decode-time entry exists for this one, since the old name never
    # appeared in a shipped release -- but a user working off an
    # unreleased checkout could still have it in their own source.
    (
        "loqs.backends.reps.instrumentreps",
        _OLD_INSTRUMENT_REP_NAME,
    ): (
        "loqs.backends.reps.instrumentreps",
        "OutcomeOperationDictInstrumentRep",
    ),
    # RepTuple/STIMDictNoiseModel: overridden to a deleted-outright entry
    # for *this* table specifically, even though a real decode-time
    # target exists in IMPORT_LOCATION_CHANGES_BY_VERSION -- see the
    # module docstring for why a blind rewrite would be wrong for both.
    ("loqs.backends.reps", "RepTuple"): None,
    ("loqs.backends.model.stimdictmodel", "STIMDictNoiseModel"): None,
}
"""`(old_module, old_name) -> (new_module, new_name) | None` (`None` means
deleted outright, not relocated -- rewriting flags real remaining code
references for manual review instead of guessing a replacement).
"""


class _DeletedNameFinder(cst.CSTVisitor):
    """Find real code references (not docstring/comment mentions) to a
    deleted `(module, name)`: both an `import` of it (via a plain module/
    name comparison -- `QualifiedNameProvider` doesn't resolve anything
    useful for an import statement's own name node) and any later usage
    of it (via the same `QualifiedNameProvider` resolution
    [](api:RenameCommand) itself uses)."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider, PositionProvider)

    def __init__(self, module: str, name: str) -> None:
        self._module = module
        self._name = name
        self._qualified_name = f"{module}.{name}"
        self.lines: list[int] = []

    def _line(self, node: cst.CSTNode) -> int:
        return self.get_metadata(PositionProvider, node).start.line

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:  # a relative `from . import x`
            return
        if get_full_name_for_node(node.module) != self._module:
            return
        if isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            if get_full_name_for_node(alias.name) == self._name:
                self.lines.append(self._line(alias.name))

    def visit_Name(self, node: cst.Name) -> None:
        try:
            qualified_names = self.get_metadata(QualifiedNameProvider, node)
        except KeyError:
            return
        if any(qn.name == self._qualified_name for qn in qualified_names):
            self.lines.append(self._line(node))


def rewrite_renames(
    source: str,
    renames: Mapping[tuple[str, str], tuple[str, str] | None] | None = None,
) -> MigrationResult:
    """Rewrite every real code reference (import or usage) to a renamed
    `(module, name)` in `renames` throughout `source`, and flag every
    real code reference to a name deleted outright (a `None` entry) for
    manual review instead.

    Parameters
    ----------
    source:
        The full text of a `.py` file (or an extracted code cell -- see
        [](api:loqs.tools.migrate.notebook)).
    renames:
        The `(old_module, old_name) -> (new_module, new_name) | None`
        table to rewrite against. Defaults to [](api:RENAMES); overriding
        it is mainly useful for testing against a synthetic table.
    """
    if renames is None:
        renames = RENAMES

    # A cheap textual pre-filter before a real CST transform per rename:
    # with ~25 table entries, running each one through libcst against
    # every file in a large tree is too slow, and a name absent from the
    # file's text can't be referenced in its code either.
    applicable_renames = {k: v for k, v in renames.items() if k[1] in source}
    if not applicable_renames:
        return MigrationResult(source=source, changed=False, manual_review=[])

    module = cst.parse_module(source)

    for (old_module, old_name), new_loc in applicable_renames.items():
        if new_loc is None:
            continue
        new_module, new_name = new_loc
        context = CodemodContext()
        command = RenameCommand(
            context,
            f"{old_module}.{old_name}",
            f"{new_module}.{new_name}",
        )
        module = command.transform_module(module)

    manual_review: list[ManualReviewItem] = []
    deleted_names = {
        (old_module, old_name)
        for (old_module, old_name), new_loc in applicable_renames.items()
        if new_loc is None
    }
    if not deleted_names:
        new_source = module.code
        return MigrationResult(
            source=new_source,
            changed=new_source != source,
            manual_review=[],
        )
    wrapper = MetadataWrapper(module)
    for old_module, old_name in deleted_names:
        finder = _DeletedNameFinder(old_module, old_name)
        wrapper.visit(finder)
        for line in finder.lines:
            manual_review.append(
                ManualReviewItem(
                    line=line,
                    message=(
                        f"References {old_name!r}, which was removed "
                        "outright (not relocated) with no automatic "
                        "replacement -- see the CHANGELOG for what "
                        "replaced it."
                    ),
                )
            )

    new_source = module.code
    return MigrationResult(
        source=new_source,
        changed=new_source != source,
        manual_review=manual_review,
    )
