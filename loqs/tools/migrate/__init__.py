#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Rewrite `.py`/`.ipynb`/MyST Markdown source files still using pre-1.2
LoQS APIs.

`loqs.tools.migrate` and the `loqs-migrate` console script (see
[](api:loqs.tools.migrate.cli)) address a different problem than
[](api:Serializable)'s own decode-time compatibility machinery: that
machinery keeps *already-serialized data* readable; this tool rewrites a
user's own *source code* -- experiment scripts, a Jupyter notebook, or a
custom `apply_fn`/`map_qubits_fn` that will be frozen into future
serialized data -- so it stops relying on removed/renamed APIs going
forward.

Four independent passes, run in order, together making up [](api:migrate_source):

1. [](api:loqs.tools.migrate.reptuple): confidently-resolvable
   `RepTuple(...)` construction calls, rewritten to the concrete
   `OperationRep` subclass they actually meant.
2. [](api:loqs.tools.migrate.renames): straight `(module, name)` renames,
   always confidently rewritable -- including anything `reptuple`
   couldn't resolve, still bare `RepTuple` references at this point.
3. [](api:loqs.tools.migrate.labels): pre-1.2 positional `InstructionLabel`
   construction, rewritten to modern keyword form.
4. [](api:loqs.tools.migrate.flags): patterns whose replacement is a
   semantic change, not a pure rename (`.cast()`, `include_idles=`, an
   `"Iz"` string) -- flagged by default, never guessed at automatically.

Two of the above are opt-in exceptions to "never guessed at automatically",
each gated behind its own [](api:migrate_source) keyword argument (and the
CLI's matching `--rename-Iz`/`--rename-patch-label` flags): rewriting a
bare `"Iz"` string to `"Imrz"` (pass 4), and renaming a colliding
`inst_kwargs["patch_label"]` key (pass 3) -- both still risk being wrong
in a way this tool can't verify itself, which is why they default to off.

A pass's own manual-review and rewrite items are always relative to
*its own* input line numbering; since an earlier pass may itself change
the surrounding line count (e.g. collapsing a multi-line call onto one
line), those items are remapped forward through each later pass via
[](api:remap_manual_review)/[](api:remap_rewrites) before being combined,
so every reported line in the final result is accurate relative to the
file actually written.
"""

from __future__ import annotations

from loqs.tools.migrate.flags import detect_flagged_patterns, rewrite_iz_literal
from loqs.tools.migrate.labels import migrate_instruction_labels
from loqs.tools.migrate.renames import rewrite_renames
from loqs.tools.migrate.reptuple import rewrite_reptuple_construction
from loqs.tools.migrate.report import (
    ManualReviewItem,
    MigrationResult,
    RewriteItem,
    annotate_manual_review,
    remap_manual_review,
    remap_rewrites,
)

__all__ = [
    "ManualReviewItem",
    "MigrationResult",
    "RewriteItem",
    "migrate_source",
]


def _chain(result: MigrationResult, next_result: MigrationResult) -> MigrationResult:
    """Combine `result` with the next pass's own result over `result`'s
    (already-updated) source, remapping `result`'s own manual-review and
    rewrite items forward in case `next_result` changed the surrounding
    line count."""
    return MigrationResult(
        source=next_result.source,
        changed=result.changed or next_result.changed,
        manual_review=(
            remap_manual_review(result.source, next_result.source, result.manual_review)
            + next_result.manual_review
        ),
        rewrites=(
            remap_rewrites(result.source, next_result.source, result.rewrites)
            + next_result.rewrites
        ),
    )


def migrate_source(
    source: str, *, rename_iz: bool = False, rename_patch_label: str | None = None
) -> MigrationResult:
    """Run every migration pass over `source`, returning the combined result.

    Parameters
    ----------
    source:
        The full text of a `.py` file (or an extracted MyST code cell --
        see [](api:loqs.tools.migrate.notebook)).

    rename_iz:
        If `True`, confidently rewrite every bare `"Iz"`/`'Iz'` string
        literal to `"Imrz"`/`'Imrz'` (see
        [](api:loqs.tools.migrate.flags.rewrite_iz_literal)) instead of
        only flagging it. Off by default, since a blind string rewrite
        can't rule out an unrelated match.

    rename_patch_label:
        If given, rewrite a legacy `InstructionLabel`'s colliding
        `inst_kwargs["patch_label"]` key (see
        [](api:loqs.tools.migrate.labels)) to this name instead of only
        flagging it. Still flagged either way, as a reminder that the
        corresponding `Instruction`'s `apply_fn` parameter needs the same
        rename by hand.

    Returns
    -------
    A [](api:MigrationResult): `.source` is safe to write back to disk
    even when `.manual_review` is non-empty -- only confidently-resolvable
    rewrites are ever applied, so nothing already flagged was also
    (possibly wrongly) rewritten. Whenever `.changed` is true, `.source`
    also has a short explanatory comment inserted above each remaining
    `.manual_review` line (see [](api:annotate_manual_review)), so the
    file itself still documents what needs a look even without this
    call's own return value kept around.
    """
    result = rewrite_reptuple_construction(source)
    result = _chain(result, rewrite_renames(result.source))
    result = _chain(
        result,
        migrate_instruction_labels(result.source, rename_patch_label=rename_patch_label),
    )
    if rename_iz:
        result = _chain(result, rewrite_iz_literal(result.source))
    result.manual_review.extend(detect_flagged_patterns(result.source, rename_iz=rename_iz))
    if result.changed:
        pre_annotate_source = result.source
        result.source, result.manual_review = annotate_manual_review(
            result.source, result.manual_review
        )
        result.rewrites = remap_rewrites(pre_annotate_source, result.source, result.rewrites)
    return result
