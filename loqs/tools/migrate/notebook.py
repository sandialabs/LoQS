#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
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
"""

from __future__ import annotations

from collections.abc import Mapping
import re

from loqs.core.instructions.instruction import Instruction
from loqs.tools.migrate import migrate_source
from loqs.tools.migrate.report import ManualReviewItem, MigrationResult

_CELL_OPEN = re.compile(r"^```\{code-cell\}.*$")
_FENCE_CLOSE = re.compile(r"^```\s*$")


def migrate_notebook_source(
    source: str, *, instructions: Mapping[str, Instruction] | None = None
) -> MigrationResult:
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
        j = cell_start
        while j < len(lines) and not _FENCE_CLOSE.match(lines[j].rstrip("\n")):
            j += 1
        cell_lines = lines[cell_start:j]
        cell_source = "".join(cell_lines)

        result = migrate_source(cell_source, instructions=instructions)
        if result.changed:
            changed = True
        for item in result.manual_review:
            manual_review.append(
                ManualReviewItem(
                    line=item.line + cell_start,
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
