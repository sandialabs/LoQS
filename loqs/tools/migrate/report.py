#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Shared result/reporting types for [](api:loqs.tools.migrate)."""

from __future__ import annotations

import difflib
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar


@dataclass
class _LocatedItem:
    """Shared shape/formatting for [](api:ManualReviewItem) and
    [](api:RewriteItem): both are one thing this tool noticed at one
    place in a file, differing only in whether it also rewrote it."""

    line: int
    """1-indexed line number the item starts at (relative to the cell it
    was found in, if `cell` is set -- a whole-file line number has no
    meaningful counterpart in a JSON notebook)."""

    message: str
    """A human-readable description of what was found (and, for a
    [](api:RewriteItem), what it was rewritten to)."""

    cell: int | None = None
    """1-indexed notebook cell this item was found in, or `None` for a
    plain `.py`/MyST Markdown file, where `line` alone already locates it
    in the whole document."""

    _TAG: ClassVar[str]
    """`"FLAG"`/`"REWRITE"`, set by each concrete subclass -- printed as
    part of `__str__` so a block mixing both kinds of item (or a reader
    grepping raw CLI output) can tell them apart at a glance."""

    @property
    def location(self) -> str:
        """A human-readable locator, e.g. `"Line 25"` or `"Cell 8, Line 3"`."""
        if self.cell is not None:
            return f"Cell {self.cell}, Line {self.line}"
        return f"Line {self.line}"

    def __str__(self) -> str:
        return f"{self._TAG} {self.location}: {self.message}"


@dataclass
class ManualReviewItem(_LocatedItem):
    """One thing the migration tool found but couldn't confidently
    auto-rewrite, needing a human to look at it instead."""

    _TAG: ClassVar[str] = "FLAG"

    kind: str | None = None
    """A stable category tag for items an opt-in CLI flag can act on
    (currently `"iz"`/`"patch_label_kwarg"`), used to decide whether to
    suggest one of those flags at the end of a run; `None` for anything
    else."""


@dataclass
class RewriteItem(_LocatedItem):
    """One thing the migration tool confidently rewrote on its own,
    reported the same `"<TAG> <location>: <message>"` way an unresolved
    [](api:ManualReviewItem) is -- e.g. `"REWRITE Line 3: PatchDict ->
    PatchLayout"` -- so a dry run's output reads as one flat account of
    what happened at each location instead of a diff a reader has to
    reconstruct meaning from by hand."""

    _TAG: ClassVar[str] = "REWRITE"


@dataclass
class MigrationResult:
    """The result of migrating one file's source text."""

    source: str
    """The (possibly rewritten) source text."""

    changed: bool
    """Whether `source` differs from the original input."""

    manual_review: list[ManualReviewItem] = field(default_factory=list)
    """Anything found that needs a human to look at, whether or not
    `source` was also changed elsewhere."""

    rewrites: list[RewriteItem] = field(default_factory=list)
    """Everything this pass confidently rewrote on its own, one item per
    location -- always safe to have happened alongside `manual_review`,
    since nothing already flagged is ever also rewritten."""

    def merge(self, other: "MigrationResult") -> "MigrationResult":
        """Combine this result with another pass's result over the same
        (already-updated) source, concatenating manual-review items and
        rewrite items."""
        return MigrationResult(
            source=other.source,
            changed=self.changed or other.changed,
            manual_review=self.manual_review + other.manual_review,
            rewrites=self.rewrites + other.rewrites,
        )


def _remap_located_items(
    old_source: str, new_source: str, items: Sequence[_LocatedItem]
) -> list:
    """The shared implementation behind [](api:remap_manual_review) and
    [](api:remap_rewrites): re-locate each item's `line` (given relative
    to `old_source`) to its corresponding line in `new_source`, via their
    textual diff, preserving each item's own concrete type
    ([](api:ManualReviewItem) or [](api:RewriteItem)) and any of its
    other fields untouched.

    Needed whenever a rewrite changes the surrounding line count (e.g.
    collapsing a multi-line call onto one line) somewhere before an item
    reported relative to the unmodified line numbering -- without this,
    such an item's reported line silently drifts out of sync with the
    file it's actually reported against.
    """
    if old_source == new_source or not items:
        return list(items)

    old_lines = old_source.splitlines()
    new_lines = new_source.splitlines()
    opcodes = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False).get_opcodes()

    def remap(line: int) -> int:
        index = line - 1
        for tag, i1, i2, j1, j2 in opcodes:
            if i1 <= index < i2:
                offset = index - i1
                if tag == "equal" or (i2 - i1) == (j2 - j1):
                    # Either genuinely unchanged, or a like-for-like
                    # same-line-count replacement (the common case for a
                    # rewrite item, e.g. one substituted identifier per
                    # line) -- either way, each line's own position
                    # within the block is preserved, so several items in
                    # the same block don't all collapse onto its start.
                    return j1 + offset + 1
                return j1 + 1  # a line-count-changing region: best-effort, start of it
        return len(new_lines)  # fell past the end of a shrunk file

    return [replace(item, line=remap(item.line)) for item in items]


def remap_manual_review(
    old_source: str, new_source: str, manual_review: Sequence[ManualReviewItem]
) -> list[ManualReviewItem]:
    """[](api:_remap_located_items) for [](api:ManualReviewItem)s."""
    return _remap_located_items(old_source, new_source, manual_review)


def remap_rewrites(
    old_source: str, new_source: str, rewrites: Sequence[RewriteItem]
) -> list[RewriteItem]:
    """[](api:_remap_located_items) for [](api:RewriteItem)s."""
    return _remap_located_items(old_source, new_source, rewrites)


_MANUAL_REVIEW_COMMENT_PREFIX = "# LOQS-MIGRATE (pre-1.2 API): "
_MANUAL_REVIEW_COMMENT_MAX_LINES = 2
_MANUAL_REVIEW_COMMENT_WIDTH = 88


def annotate_manual_review(
    source: str, manual_review: Sequence[ManualReviewItem]
) -> tuple[str, list[ManualReviewItem]]:
    """Insert a short comment directly above each flagged line in
    `source`, wrapped to at most two lines and explicitly naming v1.2 as
    the API-transition point, so a later reader of the migrated file
    itself -- not just this tool's own stdout, which is easy to lose
    track of once a run is over -- can find what still needs a look.

    `item.line` in `manual_review` must already be relative to `source`
    itself (see [](api:remap_manual_review) if it isn't). Returns the
    annotated source alongside a matching, re-located copy of
    `manual_review`: inserting comment lines necessarily pushes every
    flagged code line further down, so a caller that reports or further
    processes these items afterward needs their corrected positions, not
    the ones relative to the un-annotated `source` passed in.
    """
    if not manual_review:
        return source, list(manual_review)

    wrapped_by_id = {
        id(item): textwrap.wrap(
            item.message,
            width=_MANUAL_REVIEW_COMMENT_WIDTH,
            max_lines=_MANUAL_REVIEW_COMMENT_MAX_LINES,
            placeholder=" [...]",
        )
        for item in manual_review
    }

    # Two items (e.g. from two independent passes) can flag the exact
    # same original line -- grouped here so their comments stack as one
    # combined insertion, rather than each being applied independently
    # and misplacing the second relative to the first.
    items_by_line: dict[int, list[ManualReviewItem]] = {}
    for item in manual_review:
        items_by_line.setdefault(item.line, []).append(item)

    # Every line's own flagged code shifts down by that line's own
    # combined comment-line count plus every earlier line's, since all
    # of those insert above it in the final text. Items sharing a line
    # necessarily end up reporting the same (correct) final position.
    final_line_for: dict[int, int] = {}
    cumulative_shift = 0
    for line in sorted(items_by_line):
        cumulative_shift += sum(len(wrapped_by_id[id(i)]) for i in items_by_line[line])
        final_line_for[line] = line + cumulative_shift
    remapped = [
        ManualReviewItem(
            line=final_line_for[item.line],
            message=item.message,
            cell=item.cell,
            kind=item.kind,
        )
        for item in manual_review
    ]

    # Mutate from the bottom up so an earlier insertion never invalidates
    # a not-yet-processed (larger) original line index.
    lines = source.splitlines(keepends=True)
    for line in sorted(items_by_line, reverse=True):
        if not (1 <= line <= len(lines)):
            continue
        target_line = lines[line - 1]
        indent = target_line[: len(target_line) - len(target_line.lstrip())]
        comment_lines = [
            f"{indent}{_MANUAL_REVIEW_COMMENT_PREFIX}{text}\n"
            for item in items_by_line[line]
            for text in wrapped_by_id[id(item)]
        ]
        lines[line - 1 : line - 1] = comment_lines

    return "".join(lines), remapped


_REPORT_RULE_WIDTH = 88


def _format_located_block(label: str, items: Sequence[_LocatedItem]) -> str:
    """A `label`-headed block listing `items`, one per line by
    `location`, bracketed above and below by a horizontal rule -- so a
    run reporting several files reads as clearly file-by-file rather than
    one undifferentiated stream of `file:line:` entries.
    """
    rule = "=" * _REPORT_RULE_WIDTH
    lines = [rule, label, rule]
    lines.extend(str(item) for item in items)
    return "\n".join(lines)


def format_manual_review_block(label: str, items: Sequence[ManualReviewItem]) -> str:
    """[](api:_format_located_block) for [](api:ManualReviewItem)s."""
    return _format_located_block(label, items)


def format_rewrite_block(label: str, items: Sequence[RewriteItem]) -> str:
    """[](api:_format_located_block) for [](api:RewriteItem)s."""
    return _format_located_block(label, items)
