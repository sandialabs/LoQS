#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Extract/splice Python code cells out of a MyST Markdown tutorial, so
[](api:migrate_source) can run against `docs/notebooks/*.md`'s embedded
code the same way it runs against a plain `.py` file.

`docs/notebooks/*.md` is jupytext-formatted MyST Markdown, not bare
Python: an `` ```{code-cell} ipython3 `` fence opens a code cell, a bare
`` ``` `` closes it, and everything else (prose, other fence types like
`` ```{note} ``, front matter) is untouched. This is the only committed
representation (the paired `.ipynb` is gitignored and regenerated locally
via `jupytext --to ipynb`), so it's also the only one this tool needs to
touch directly.

A code cell may open with one or more `:key: value` MyST field lines
(e.g. `:tags: [scroll-output]`) before its actual code, as
`docs/notebooks/workflow.md` does -- these aren't parseable as bare
Python on their own. These lines are passed through unchanged, alongside
the fence itself, never handed to `migrate_source`.

Known limitation: each cell is migrated independently, with no import
context carried over from earlier cells -- unlike a real notebook kernel,
where state (including imports) persists across cells. A renamed class
used bare in a later cell, relying on an import done in an earlier one,
is neither rewritten nor flagged, since [](api:rewrite_renames)'s
`QualifiedNameProvider`-based resolution can't see that import from
inside this cell alone. Not currently a problem for `docs/notebooks/*.md`
itself, but worth knowing about for a user's own notebook.
"""

from __future__ import annotations

import re

from loqs.tools.migrate import migrate_source
from loqs.tools.migrate.report import ManualReviewItem, MigrationResult

_CELL_OPEN = re.compile(r"^```\{code-cell\}.*$")
_FENCE_CLOSE = re.compile(r"^```\s*$")
_CELL_FIELD_LINE = re.compile(r"^:[\w-]+:.*$")


def migrate_notebook_source(source: str) -> MigrationResult:
    """Run [](api:migrate_source) over every code cell in a MyST Markdown
    document, leaving everything else untouched.

    A [](api:ManualReviewItem)'s line number is relative to the whole
    document (not the cell), matching [](api:migrate_source)'s own
    per-file convention.
    """
    lines = source.splitlines(keepends=True)
    output: list[str] = []
    manual_review: list[ManualReviewItem] = []
    changed = False

    i = 0
    while i < len(lines):
        line = lines[i]
        if not _CELL_OPEN.match(line.rstrip("\n")):
            output.append(line)
            i += 1
            continue

        # Found a code-cell fence: collect its body up to the closing fence.
        output.append(line)
        cell_start = i + 1

        # Pass any leading `:key: value` MyST field lines through untouched
        # -- they aren't code, and migrate_source can't parse them as such.
        code_start = cell_start
        while code_start < len(lines) and _CELL_FIELD_LINE.match(
            lines[code_start].rstrip("\n")
        ):
            output.append(lines[code_start])
            code_start += 1

        j = code_start
        while j < len(lines) and not _FENCE_CLOSE.match(lines[j].rstrip("\n")):
            j += 1
        cell_lines = lines[code_start:j]
        cell_source = "".join(cell_lines)

        result = migrate_source(cell_source)
        if result.changed:
            changed = True
        for item in result.manual_review:
            manual_review.append(
                ManualReviewItem(
                    line=item.line + code_start,
                    message=item.message,
                )
            )
        output.append(result.source)

        if j < len(lines):
            output.append(lines[j])  # the closing fence itself
        i = j + 1

    return MigrationResult(
        source="".join(output),
        changed=changed,
        manual_review=manual_review,
    )
