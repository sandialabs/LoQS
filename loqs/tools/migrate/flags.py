#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Detect-only patterns: old APIs whose replacement changes meaning, not
just name, so guessing a rewrite would be dishonest rather than helpful.

Each pattern here is a plain regex scan, deliberately not auto-rewritten:

- `.cast(...)`: removed entirely (issue #96/#107); its replacement depends
  on which class the call was on (`InstructionStack.cast(x)` ->
  `InstructionStack(x)`, `InstructionLabel.cast(x)` ->
  `InstructionLabel.from_raw(x)`), which isn't generally inferable from
  the call site alone (an arbitrary receiver expression's static type
  isn't known).
- `include_idles=`/`reference_round_Z=`/`reference_round_X=`: replaced by
  `idle_layout=`/`reference_round_mode_Z=`/`reference_round_mode_X=`
  (issue #108), but these are semantic changes, not pure renames (a
  boolean doesn't map onto the new parameter's meaning automatically).
- A bare `"Iz"` string literal: likely an old instrument-name reference to
  what's now `"Imrz"` (issue #101), but this is a plain string, not an
  identifier reference, so a blind rewrite risks matching unrelated text
  that just happens to contain the same two characters.
"""

from __future__ import annotations

import re

from loqs.tools.migrate.report import ManualReviewItem

_DETECT_ONLY_PATTERNS: dict[str, re.Pattern] = {
    ".cast(...) call (removed -- issue #96/#107)": re.compile(r"\.cast\("),
    "include_idles= kwarg (replaced by idle_layout= -- issue #108)": re.compile(
        r"\binclude_idles\s*="
    ),
    "reference_round_Z/X= kwarg (replaced by reference_round_mode_Z/X= -- issue #108)": re.compile(
        r"\breference_round_[ZX]\s*="
    ),
    '"Iz" string literal (likely the old instrument name, renamed to "Imrz" -- issue #101)': re.compile(
        r"""(['"])Iz\1"""
    ),
}


def detect_flagged_patterns(source: str) -> list[ManualReviewItem]:
    """Scan `source` for every pattern in [](api:_DETECT_ONLY_PATTERNS),
    returning one [](api:ManualReviewItem) per match (not per pattern --
    a pattern appearing 3 times produces 3 items, each with its own line
    number)."""
    items = []
    lines = source.splitlines()
    for name, pattern in _DETECT_ONLY_PATTERNS.items():
        for lineno, line in enumerate(lines, start=1):
            if pattern.search(line):
                items.append(ManualReviewItem(line=lineno, message=name))
    return items
