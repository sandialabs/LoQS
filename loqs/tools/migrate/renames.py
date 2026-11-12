#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3.1                                                                         #
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
rewrite mechanism. Both rewrite passes resolve real code references via
`libcst`'s `QualifiedNameProvider`, so they can tell an actual usage of a
renamed class apart from the same text appearing in a docstring,
comment, or unrelated identifier, and update both the import statement
and every usage site together. Neither rewrites or flags a name inside
a quoted annotation (`x: "PatchDict"`), which keeps its old name.

Same-module renames (the `*CastableTypes` -> `*Like` renames, the
instrument-rep rename and the `PatchDict` re-exports) go through LoQS's
own pass, which keeps the user's import style: module aliases, `as`
names and import placement. `RenameCommand` writes such a rename reached
through `import a.b as x` as `a.b.New` without binding `a` (#161), and
drops the from-import when the file also has `import a.b`. Module-changing
renames use `libcst`'s `RenameCommand`
(`libcst.codemod.commands.rename`); LoQS adds the from-import it drops
when the file has both `import <old module>` and a from-import of the
old name.

A few real `RenameCommand` behaviors, which now apply only to
module-changing renames, worth knowing about before trusting this pass
blindly:

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
   renamed name the file really references, done here since each has its
   own old/new location) can leave a stray blank line and trailing comma
   behind in a multi-name `from ... import (...)` block when an earlier
   pass's partial cleanup isn't reformatted by a later pass. Still valid
   Python, just untidy.

None of these are incorrect Python, but review the diff before committing
rather than applying a whole-file rewrite unseen, same as any other
automated rewrite.
"""

from __future__ import annotations

from collections.abc import Mapping

import libcst as cst
from libcst.codemod import CodemodContext
from libcst.codemod.commands.rename import RenameCommand
from libcst.codemod.visitors import AddImportsVisitor
from libcst.helpers import get_full_name_for_node
from libcst.metadata import (
    MetadataWrapper,
    PositionProvider,
    QualifiedNameProvider,
)

from loqs.internal.serializable import Serializable
from loqs.tools.migrate.report import (
    ManualReviewItem,
    MigrationResult,
    RewriteItem,
    remap_rewrites,
)

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
    # PatchDict -> PatchLayout: IMPORT_LOCATION_CHANGES_BY_VERSION only
    # records where the class was actually *defined* pre-1.2
    # (loqs.core.recordables.patchdict), but real source overwhelmingly
    # imports it from one of the two convenience re-exports instead
    # (both of which already export PatchLayout under its own name too,
    # so each rewrites to a same-module class-name swap rather than a
    # module change).
    ("loqs.core.recordables", "PatchDict"): (
        "loqs.core.recordables",
        "PatchLayout",
    ),
    ("loqs.core", "PatchDict"): ("loqs.core", "PatchLayout"),
}
"""`(old_module, old_name) -> (new_module, new_name) | None` (`None` means
deleted outright, not relocated -- rewriting flags real remaining code
references for manual review instead of guessing a replacement).
"""


class _QualifiedNameFinder(cst.CSTVisitor):
    """Find real code references (not docstring/comment mentions) to a
    `(module, name)`, whether it's being renamed or deleted outright:
    both an `import` of it (via a plain module/name comparison --
    `QualifiedNameProvider` doesn't resolve anything useful for an import
    statement's own name node) and any later usage of it (via the same
    `QualifiedNameProvider` resolution [](api:RenameCommand) itself
    uses). A usage through a module import, such as `loqs.core.PatchDict()`
    after `import loqs.core`, is an `Attribute` node and is found too."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider, PositionProvider)

    def __init__(self, module: str, name: str) -> None:
        self._module = module
        self._name = name
        self._qualified_name = f"{module}.{name}"
        self.lines: list[int] = []
        # Whether the file has `import <module>` (any `as`) and a
        # `from <module> import <name>`, respectively.
        self.has_module_import = False
        self.has_from_import = False

    def _line(self, node: cst.CSTNode) -> int:
        return self.get_metadata(PositionProvider, node).start.line

    def visit_Import(self, node: cst.Import) -> None:
        if any(
            get_full_name_for_node(alias.name) == self._module
            for alias in node.names
        ):
            self.has_module_import = True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:  # a relative `from . import x`
            return
        if get_full_name_for_node(node.module) != self._module:
            return
        if isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            if get_full_name_for_node(alias.name) == self._name:
                self.has_from_import = True
                self.lines.append(self._line(alias.name))

    def visit_Name(self, node: cst.Name | cst.Attribute) -> None:
        try:
            qualified_names = self.get_metadata(QualifiedNameProvider, node)
        except KeyError:
            return
        if any(qn.name == self._qualified_name for qn in qualified_names):
            self.lines.append(self._line(node))

    visit_Attribute = visit_Name


class _SameModuleRenamer(cst.CSTTransformer):
    """Rename names that stay in their module, keeping the user's import
    style. `table` maps `"<module>.<old_name>"` to the new name. Renames
    the name in `from <module> import <old_name> [as y]` (keeping `as y`),
    each `Name` resolving to a table entry whose text is the old name, and
    the `attr` of each `Attribute` resolving to one (`lc.PatchDict`)."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self, table: Mapping[str, str]) -> None:
        self._table = table

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        if original_node.module is None:
            return updated_node
        if isinstance(updated_node.names, cst.ImportStar):
            return updated_node
        module = get_full_name_for_node(original_node.module)
        names = [
            (
                alias.with_changes(name=cst.Name(self._table[key]))
                if isinstance(alias.name, cst.Name)
                and (key := f"{module}.{alias.name.value}") in self._table
                else alias
            )
            for alias in updated_node.names
        ]
        return updated_node.with_changes(names=names)

    def _new_name(
        self, node: cst.Name | cst.Attribute, text: str
    ) -> str | None:
        for qn in self.get_metadata(QualifiedNameProvider, node, set()):
            if qn.name.rsplit(".", 1)[-1] == text and qn.name in self._table:
                return self._table[qn.name]
        return None

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.Name:
        new = self._new_name(original_node, original_node.value)
        return updated_node.with_changes(value=new) if new else updated_node

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.Attribute:
        new = self._new_name(original_node, original_node.attr.value)
        if new is None:
            return updated_node
        return updated_node.with_changes(attr=cst.Name(new))


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

    # Two pre-filters before a real CST transform per rename, since each
    # `RenameCommand` recomputes libcst's scope metadata for the whole
    # file. First a substring check: a name absent from the file's text
    # can't be referenced in its code. Then the finder below, which skips
    # the transform for a name that appears only as text (a docstring, a
    # comment, part of a longer identifier).
    applicable_renames = {k: v for k, v in renames.items() if k[1] in source}
    if not applicable_renames:
        return MigrationResult(
            source=source, changed=False, manual_review=[], rewrites=[]
        )

    module = cst.parse_module(source)

    # Every renamed (not deleted-outright) entry's own real references,
    # found against the original, not-yet-rewritten module -- remapped
    # to their final positions once every pass below has actually run
    # (module-level line counts can shift, e.g. a new import added).
    original_wrapper = MetadataWrapper(module)
    rewrite_items: list[RewriteItem] = []
    same_module: dict[str, str] = {}
    module_changing: list[tuple[str, str, str, str, bool]] = []
    for (old_module, old_name), new_loc in applicable_renames.items():
        if new_loc is None:
            continue
        new_module, new_name = new_loc
        finder = _QualifiedNameFinder(old_module, old_name)
        original_wrapper.visit(finder)
        rewrite_items.extend(
            RewriteItem(line=line, message=f"{old_name} -> {new_name}")
            for line in finder.lines
        )
        if not finder.lines:
            continue
        if new_module == old_module:
            same_module[f"{old_module}.{old_name}"] = new_name
        else:
            # `RenameCommand` drops the from-import when the file also
            # has `import <old_module>`, so it's re-added below.
            both = finder.has_module_import and finder.has_from_import
            module_changing.append(
                (old_module, old_name, new_module, new_name, both)
            )

    if same_module:
        # Visits `original_wrapper`, reusing the finders' metadata.
        module = original_wrapper.visit(_SameModuleRenamer(same_module))
    for old_module, old_name, new_module, new_name, both in module_changing:
        context = CodemodContext()
        if both:
            AddImportsVisitor.add_needed_import(context, new_module, new_name)
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
            rewrites=remap_rewrites(source, new_source, rewrite_items),
        )
    wrapper = MetadataWrapper(module)
    for old_module, old_name in deleted_names:
        finder = _QualifiedNameFinder(old_module, old_name)
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
        rewrites=remap_rewrites(source, new_source, rewrite_items),
    )
