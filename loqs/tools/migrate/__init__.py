#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Rewrite `.py`/MyST Markdown source files still using pre-1.2 LoQS APIs.

`loqs.tools.migrate` and the `loqs-migrate` console script (see
[](api:loqs.tools.migrate.cli)) address a different problem than
[](api:Serializable)'s own decode-time compatibility machinery: that
machinery keeps *already-serialized data* readable; this tool rewrites a
user's own *source code* -- experiment scripts, or a custom
`apply_fn`/`map_qubits_fn` that will be frozen into future serialized data
-- so it stops relying on removed/renamed APIs going forward.

Three independent passes, run in order, together making up [](api:migrate_source):

1. [](api:loqs.tools.migrate.renames): straight `(module, name)` renames,
   always confidently rewritable.
2. [](api:loqs.tools.migrate.labels): pre-1.2 positional `InstructionLabel`
   construction, rewritten to modern keyword form.
3. [](api:loqs.tools.migrate.flags): patterns whose replacement is a
   semantic change, not a pure rename (`.cast()`, `include_idles=`, an
   `"Iz"` string) -- always flagged, never auto-rewritten.
"""

from __future__ import annotations

from loqs.tools.migrate.flags import detect_flagged_patterns
from loqs.tools.migrate.labels import migrate_instruction_labels
from loqs.tools.migrate.renames import rewrite_renames
from loqs.tools.migrate.report import ManualReviewItem, MigrationResult

__all__ = [
    "ManualReviewItem",
    "MigrationResult",
    "migrate_source",
]


def migrate_source(source: str) -> MigrationResult:
    """Run every migration pass over `source`, returning the combined result.

    Parameters
    ----------
    source:
        The full text of a `.py` file (or an extracted MyST code cell --
        see [](api:loqs.tools.migrate.notebook)).

    Returns
    -------
    A [](api:MigrationResult): `.source` is safe to write back to disk
    even when `.manual_review` is non-empty -- only confidently-resolvable
    rewrites are ever applied, so nothing already flagged was also
    (possibly wrongly) rewritten.
    """
    result = rewrite_renames(source)
    result = result.merge(migrate_instruction_labels(result.source))
    result.manual_review.extend(detect_flagged_patterns(result.source))
    return result
