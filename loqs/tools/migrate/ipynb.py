#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Rewrite Python code cells embedded in a Jupyter `.ipynb` notebook, so
[](api:migrate_source) can run against a user's own real notebooks the
same way it runs against a plain `.py` file.

Unlike `docs/notebooks/*.md` (jupytext MyST Markdown, handled by
[](api:loqs.tools.migrate.notebook)), a plain `.ipynb` file has no
paired, separately-edited source format -- it *is* the notebook. Only
`code` cells are touched; `markdown`/`raw` cells, outputs, execution
counts, and all other notebook metadata are left exactly as they were.

A notebook is ordinary JSON, and every real `.ipynb` file round-trips
byte-identically through `json.dumps(nb, indent=1, sort_keys=True,
ensure_ascii=False)` plus a trailing newline when untouched -- the same
formatting Jupyter's own writer produces. Re-serializing the whole
notebook this way after mutating only the changed cells' `source`
therefore produces a minimal, cell-scoped diff rather than a spurious
whole-file reformat.

Known limitation, shared with [](api:migrate_notebook_source): each code
cell is migrated independently, with no import context carried over from
an earlier cell, unlike a real kernel where state persists across cells.

A real notebook's code cells routinely include IPython magics (`%timeit`,
`%matplotlib inline`), shell escapes (`!pip install ...`), or `?`/`??`
help syntax -- none of which are valid Python, so [](api:migrate_source)
can't parse them directly. Rather than letting one such line abort the
whole cell (real code often surrounds it, e.g. `%%time` followed by an
ordinary statement), every line the parser rejects is temporarily
replaced with an inert placeholder, the resulting valid Python is
migrated normally, and the original line is substituted back in
afterward -- see `_migrate_cell_text` for the mechanics. A cell that
still can't be made to parse this way (e.g. a `%%bash` cell whose entire
body is shell script, not Python) is left untouched and flagged instead.
"""

from __future__ import annotations

import json
import re

import libcst as cst

from loqs.tools.migrate import migrate_source
from loqs.tools.migrate.report import (
    ManualReviewItem,
    MigrationResult,
    RewriteItem,
    annotate_manual_review,
)

_PARSE_ERROR_LOCATION = re.compile(r"error at (\d+):\d+")


class _NotPythonCell(Exception):
    """Raised when every non-blank line of a cell needed stripping just
    to get it to parse -- signals a cell that isn't meaningfully Python
    at all (e.g. a `%%bash`/`%%html` cell-magic body), which should be
    flagged like any other unparseable cell rather than accepted as a
    migration that legitimately found nothing to report."""


def _cell_source_text(cell: dict) -> str:
    """A code cell's `source` as one string, whether stored on disk as a
    single string or (Jupyter's own convention) a list of lines."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _text_to_cell_source(text: str, was_list: bool) -> list[str] | str:
    """The inverse of `_cell_source_text`, matching the cell's original
    on-disk shape so an untouched cell's representation never changes."""
    return text.splitlines(keepends=True) if was_list else text


def _bad_line_number(exc: cst.ParserSyntaxError) -> int | None:
    """The 1-indexed source line libcst's parser choked on. Its own
    `raw_line`/`raw_column` attributes don't reliably point at this same
    line across every failure shape (off by one in some, exactly right
    in others) -- the location embedded in the error's own message text
    does, consistently."""
    match = _PARSE_ERROR_LOCATION.search(exc.message)
    return int(match.group(1)) if match else None


def _strip_unparseable_lines(text: str) -> tuple[str, dict[str, str]] | None:
    """Replace whichever line libcst's parser rejects with an inert
    `pass` placeholder, one line at a time, until the rest parses (or
    every distinct failing line has already been tried once, at which
    point this gives up rather than loop). Returns the patched text and
    a `{placeholder_line: original_line}` map for restoring the real
    content verbatim afterward, or `None` if no amount of stripping got
    the cell to parse.

    Restoring by exact text substitution (not by line number) is
    deliberate: a later rewrite pass can add or remove lines elsewhere
    in the same text (e.g. a new import), which would shift any
    position recorded by number.
    """
    lines = text.splitlines(keepends=True)
    placeholders: dict[str, str] = {}
    substituted: set[int] = set()

    while True:
        candidate = "".join(lines)
        try:
            cst.parse_module(candidate)
            return candidate, placeholders
        except cst.ParserSyntaxError as exc:
            bad_line = _bad_line_number(exc)
            if bad_line is None or not (1 <= bad_line <= len(lines)):
                return None
            index = bad_line - 1
            if index in substituted:
                return None  # placeholder itself didn't resolve the error; give up
            substituted.add(index)

            original_line = lines[index]
            indent = original_line[: len(original_line) - len(original_line.lstrip())]
            token = f"__LOQS_MIGRATE_IPYNB_PLACEHOLDER_{len(placeholders)}__"
            placeholders[token] = original_line
            lines[index] = f"{indent}pass  # {token}\n"


def _migrate_cell_text(
    cell_text: str, *, rename_iz: bool = False, rename_patch_label: str | None = None
) -> MigrationResult:
    """[](api:migrate_source) a code cell's text, first substituting out
    any line the parser can't handle on its own (see module docstring)
    and restoring it verbatim afterward. Raises `cst.ParserSyntaxError`
    if the cell still can't be made to parse this way, or
    `_NotPythonCell` if stripping succeeded only by replacing every
    real line in it."""
    stripped = _strip_unparseable_lines(cell_text)
    if stripped is None:
        cst.parse_module(cell_text)  # re-raise the original parse error
        raise AssertionError("unreachable: parse_module should have raised")

    patched_text, placeholders = stripped
    nonblank_lines = sum(1 for line in cell_text.splitlines() if line.strip())
    if placeholders and len(placeholders) >= nonblank_lines:
        raise _NotPythonCell(f"stripped {len(placeholders)}/{nonblank_lines} non-blank lines")

    result = migrate_source(
        patched_text, rename_iz=rename_iz, rename_patch_label=rename_patch_label
    )
    if not placeholders:
        return result

    restored_source = result.source
    for token, original_line in placeholders.items():
        restored_source = restored_source.replace(f"pass  # {token}\n", original_line)
    return MigrationResult(
        source=restored_source,
        changed=result.changed,
        manual_review=result.manual_review,
        rewrites=result.rewrites,
    )


def migrate_ipynb_source(
    source: str, *, rename_iz: bool = False, rename_patch_label: str | None = None
) -> MigrationResult:
    """Run [](api:migrate_source) over every `code` cell in a Jupyter
    notebook, leaving markdown/raw cells, outputs, and all other
    notebook structure untouched.

    A [](api:ManualReviewItem)/[](api:RewriteItem)'s `line` is relative
    to the cell it was found in (a whole-file line number has no
    meaningful counterpart in a JSON notebook); its `cell` is set to that
    cell's 1-indexed position among all cells, so the two together locate
    it unambiguously (see [](api:ManualReviewItem.location)). Whenever
    anything in the notebook changes, every remaining flagged cell also
    gets its own [](api:annotate_manual_review) pass, even a cell that
    had no rewrite of its own -- the notebook is being rewritten either
    way, so there's no reason to leave some of its flagged spots
    undocumented in the file itself.

    Parameters
    ----------
    source:
        The full text of an `.ipynb` file.

    rename_iz, rename_patch_label:
        Forwarded to every cell's own [](api:migrate_source) call
        unchanged.
    """
    notebook = json.loads(source)
    unparseable_review: list[ManualReviewItem] = []
    changed = False
    cell_results: list[tuple[int, dict, bool, MigrationResult]] = []

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        original_source = cell.get("source", "")
        was_list = isinstance(original_source, list)

        try:
            result = _migrate_cell_text(
                _cell_source_text(cell),
                rename_iz=rename_iz,
                rename_patch_label=rename_patch_label,
            )
        except (cst.ParserSyntaxError, _NotPythonCell):
            unparseable_review.append(
                ManualReviewItem(
                    line=1,
                    message=(
                        "couldn't parse as plain Python (IPython magic or "
                        "shell syntax?) -- skipped; review by hand."
                    ),
                    cell=index,
                )
            )
            continue

        changed = changed or result.changed
        cell_results.append((index, cell, was_list, result))

    if not changed:
        # Nothing is being written back, so nothing gets annotated either
        # -- report every cell's own (un-annotated) manual-review/rewrite
        # lines exactly as migrate_source found them.
        manual_review = list(unparseable_review)
        rewrites: list[RewriteItem] = []
        for index, _cell, _was_list, result in cell_results:
            for item in result.manual_review:
                manual_review.append(
                    ManualReviewItem(line=item.line, message=item.message, cell=index, kind=item.kind)
                )
            for item in result.rewrites:
                rewrites.append(RewriteItem(line=item.line, message=item.message, cell=index))
        return MigrationResult(
            source=source, changed=False, manual_review=manual_review, rewrites=rewrites
        )

    manual_review = list(unparseable_review)
    rewrites = []
    for index, cell, was_list, result in cell_results:
        cell_source = result.source
        cell_manual_review = result.manual_review
        if result.manual_review and not result.changed:
            # This cell had no rewrite of its own (so migrate_source's
            # own annotate step never fired), but the notebook overall
            # is being rewritten anyway -- annotate it too.
            cell_source, cell_manual_review = annotate_manual_review(
                cell_source, result.manual_review
            )
        cell["source"] = _text_to_cell_source(cell_source, was_list)
        for item in cell_manual_review:
            manual_review.append(
                ManualReviewItem(line=item.line, message=item.message, cell=index, kind=item.kind)
            )
        for item in result.rewrites:
            rewrites.append(RewriteItem(line=item.line, message=item.message, cell=index))

    new_source = json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False)
    if source.endswith("\n"):
        new_source += "\n"
    return MigrationResult(
        source=new_source, changed=True, manual_review=manual_review, rewrites=rewrites
    )
