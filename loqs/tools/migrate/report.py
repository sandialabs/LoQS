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
from dataclasses import dataclass, field


@dataclass
class ManualReviewItem:
    """One thing the migration tool found but couldn't confidently
    auto-rewrite, needing a human to look at it instead."""

    line: int
    """1-indexed line number the item starts at."""

    message: str
    """A human-readable description of what was found and why it wasn't
    auto-rewritten."""

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


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

    def merge(self, other: "MigrationResult") -> "MigrationResult":
        """Combine this result with another pass's result over the same
        (already-updated) source, concatenating manual-review items."""
        return MigrationResult(
            source=other.source,
            changed=self.changed or other.changed,
            manual_review=self.manual_review + other.manual_review,
        )


def remap_manual_review(
    old_source: str, new_source: str, manual_review: Sequence[ManualReviewItem]
) -> list[ManualReviewItem]:
    """Re-locate each item's `line` (given relative to `old_source`) to
    its corresponding line in `new_source`, via their textual diff.

    Needed whenever a rewrite changes the surrounding line count (e.g.
    collapsing a multi-line call onto one line) somewhere before an item
    that was flagged, but not itself rewritten, relative to the
    unmodified line numbering -- without this, such an item's reported
    line silently drifts out of sync with the file it's actually
    reported against.
    """
    if old_source == new_source or not manual_review:
        return list(manual_review)

    old_lines = old_source.splitlines()
    new_lines = new_source.splitlines()
    opcodes = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False).get_opcodes()

    def remap(line: int) -> int:
        index = line - 1
        for tag, i1, i2, j1, j2 in opcodes:
            if i1 <= index < i2:
                if tag == "equal":
                    return j1 + (index - i1) + 1
                return j1 + 1  # inside a changed region: best-effort, start of it
        return len(new_lines)  # fell past the end of a shrunk file

    return [
        ManualReviewItem(line=remap(item.line), message=item.message)
        for item in manual_review
    ]


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
        ManualReviewItem(line=final_line_for[item.line], message=item.message)
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
