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

This is deliberately **not** implemented on top of
[](api:Serializable._update_imports) itself, even though that function
already does this exact rewrite for frozen function source: it's a plain
line-based text scan, and two real problems with reusing it at the
whole-file level were found empirically, not theorized:

1. Its multi-line-import-continuation detection
   (`line.startswith("from") and line.endswith("(")`) false-triggers on
   any unrelated line shaped like one -- confirmed directly, it corrupts
   `codepack_5_1_3_quantinuum2022.py`, which has a local variable named
   `from_prime_basis_circ` with a method call spanning multiple lines.
2. Its "replace this name throughout the rest of the file" pass is a
   blind text substitution, with no regard for whether a match is inside
   a string or comment -- confirmed directly against
   `tests/backends/fixtures/generate_reps_fixtures.py`, whose docstring
   *narrates* the `RepTuple` -> `OperationRep` history in prose; a blind
   substitution corrupts that prose into a false statement (e.g. "issue
   #97 later removed `OperationRep` entirely" -- backwards).

Both failure modes are avoided here by using `libcst`'s
`RenameCommand` (`libcst.codemod.commands.rename`), which resolves real
code references via `QualifiedNameProvider` -- so it can tell an actual
usage of a renamed class apart from the same text appearing in a
docstring, comment, or unrelated identifier, and updates both the import
statement and every usage site together.

One real behavior worth knowing about before trusting this pass blindly:
`RenameCommand` always adds the new import at module scope, even when the
old one was function-local (e.g. a lazy import used to avoid a circular
import, a real pattern in this codebase) -- confirmed directly against
`tests/core/recordables/test_patchdict.py`. This is still valid Python,
but changes where the import happens; review the diff before committing
rather than applying it unseen, same as any other automated rewrite.
"""

from __future__ import annotations

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


def rewrite_renames(source: str) -> MigrationResult:
    """Rewrite every real code reference (import or usage) to a renamed
    `(module, name)` in [](api:RENAMES) throughout `source`, and flag
    every real code reference to a name deleted outright (a `None` entry)
    for manual review instead.

    Parameters
    ----------
    source:
        The full text of a `.py` file (or an extracted code cell -- see
        [](api:loqs.tools.migrate.notebook)).
    """
    # A cheap textual pre-filter before paying for a real CST transform
    # per rename: with ~25 table entries, running every one of them
    # through libcst against every file in a large tree is far too slow
    # to be practical, and the overwhelming majority of files reference
    # none of them at all. A name that doesn't appear as text anywhere in
    # the file certainly isn't referenced in its code either.
    applicable_renames = {k: v for k, v in RENAMES.items() if k[1] in source}
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
