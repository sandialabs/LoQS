#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Shared result/reporting types for [](api:loqs.tools.migrate)."""

from __future__ import annotations

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
