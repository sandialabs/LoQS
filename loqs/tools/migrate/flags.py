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

- `<OldCastableClass>.cast(...)`: removed entirely in v1.2; its
  replacement depends on which class the call was on
  (`InstructionStack.cast(x)` -> `InstructionStack(x)`,
  `InstructionLabel.cast(x)` -> `InstructionLabel.from_raw(x)`), which
  isn't generally inferable from the call site alone. Scoped to the
  specific 14 classes that had a real `.cast()` (derived from
  [](api:RENAMES)'s own `*CastableTypes` entries, not a bare `.cast(`
  scan), since a bare scan collides with unrelated third-party `.cast()`
  methods with the exact same name (`pygsti.circuits.Circuit.cast`,
  `pygsti.evotypes.Evotype.cast`), which aren't LoQS's removed API at all.
- `include_idles=`/`reference_round_Z=`/`reference_round_X=` passed to a
  `create_qec_code(...)`-style call: replaced in v1.2 by `idle_layout=`/
  `reference_round_mode_Z=`/`reference_round_mode_X=`, but these are
  semantic changes, not pure renames (a boolean doesn't map onto the new
  parameter's meaning automatically). Scoped to calls whose function name
  ends in `create_qec_code` specifically, not a bare `include_idles=`
  scan, since several *other*, unrelated, still-current
  `include_idles: bool` parameters exist on lower-level circuit-building
  helpers throughout `loqs/codepacks/codepack_surf17_multipatch.py`/
  `codepack_surf17_surgery.py` that this rename doesn't touch.
- A bare `"Iz"` string literal: likely an old instrument-name reference to
  what's now `"Imrz"` (renamed in v1.2), but this is a plain string, not
  an identifier reference, so a blind rewrite risks matching unrelated
  text that just happens to contain the same two characters.

`RepTuple`/`STIMDictNoiseModel` real code references are flagged too, but
via [](api:loqs.tools.migrate.renames)'s own `RENAMES` table (both are
overridden there to a deleted-outright entry) rather than a pattern here
-- see that module's docstring for why a blind rename would be actively
wrong for these two specifically.
"""

from __future__ import annotations

import re

import libcst as cst

from loqs.tools.migrate.renames import RENAMES
from loqs.tools.migrate.report import ManualReviewItem

_CASTABLE_CLASS_NAMES = sorted(
    {
        old_name[: -len("CastableTypes")]
        for (_, old_name) in RENAMES
        if old_name.endswith("CastableTypes")
    }
)

_LINE_PATTERNS: dict[str, re.Pattern] = {
    f"{cls}.cast(...) call (removed in v1.2)": re.compile(
        rf"\b{re.escape(cls)}\.cast\("
    )
    for cls in _CASTABLE_CLASS_NAMES
} | {
    '"Iz" string literal (likely the old instrument name, renamed to "Imrz" in v1.2)': re.compile(
        r"""(['"])Iz\1"""
    ),
}

_QEC_CODE_KWARGS = {
    "include_idles": "idle_layout",
    "reference_round_Z": "reference_round_mode_Z",
    "reference_round_X": "reference_round_mode_X",
}


def _detect_line_patterns(source: str) -> list[ManualReviewItem]:
    items = []
    lines = source.splitlines()
    for name, pattern in _LINE_PATTERNS.items():
        for lineno, line in enumerate(lines, start=1):
            if pattern.search(line):
                items.append(ManualReviewItem(line=lineno, message=name))
    return items


def _func_name(node: cst.BaseExpression) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return node.attr.value
    return None


class _QECCodeKwargFinder(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self) -> None:
        self.items: list[ManualReviewItem] = []

    def visit_Call(self, node: cst.Call) -> None:
        if _func_name(node.func) != "create_qec_code":
            return
        for arg in node.args:
            if arg.keyword is None:
                continue
            new_name = _QEC_CODE_KWARGS.get(arg.keyword.value)
            if new_name is None:
                continue
            line = self.get_metadata(
                cst.metadata.PositionProvider, node
            ).start.line
            self.items.append(
                ManualReviewItem(
                    line=line,
                    message=(
                        f"{arg.keyword.value}= kwarg to create_qec_code() "
                        f"(replaced by {new_name}= in v1.2)"
                    ),
                )
            )


def _detect_qec_code_kwargs(source: str) -> list[ManualReviewItem]:
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return []  # not a standalone-parseable file; nothing to check here
    wrapper = cst.metadata.MetadataWrapper(module)
    finder = _QECCodeKwargFinder()
    wrapper.visit(finder)
    return finder.items


def detect_flagged_patterns(source: str) -> list[ManualReviewItem]:
    """Scan `source` for every pattern described in this module's
    docstring, returning one [](api:ManualReviewItem) per match (not per
    pattern -- a pattern appearing 3 times produces 3 items, each with
    its own line number)."""
    return _detect_line_patterns(source) + _detect_qec_code_kwargs(source)
