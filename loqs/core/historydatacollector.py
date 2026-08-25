#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""[](api:HistoryDataCollector) definition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from loqs.core.history import HistoryCollectDataIndexTypes
from loqs.core.programresults import ProgramResults

HistoryDataCollectorLike: TypeAlias = (
    "HistoryDataCollector | str | Mapping[str, object] | "
    "tuple[str] | tuple[str, HistoryCollectDataIndexTypes]"
)
"""Objects that can be cast to a single [](api:HistoryDataCollector).

A bare `list` is deliberately excluded: it instead means "combine several
of these into one joined outcome per shot" wherever this alias is used in
a `list[HistoryDataCollectorLike]` context, e.g.
[](api:ProgramResults.collect_shot_data)'s callers in `loqs.tools`.
"""


@dataclass
class HistoryDataCollector:
    """A recipe for pulling one series of values out of a [](api:ProgramResults).

    Bundles the arguments to [](api:ProgramResults.collect_shot_data) into a
    single, reusable object instead of a raw positional tuple.
    """

    key: str
    """See `key` in [](api:History.collect_data)."""

    indices: HistoryCollectDataIndexTypes = -1
    """See `indices` in [](api:History.collect_data)."""

    frame_filter: Mapping[str, object] | None = field(
        default=None, kw_only=True
    )
    """See `frame_filter` in [](api:History.collect_data)."""

    strip_none_entries: bool = field(default=False, kw_only=True)
    """See `strip_none_entries` in [](api:History.collect_data)."""

    @classmethod
    def from_raw(cls, raw: HistoryDataCollectorLike) -> HistoryDataCollector:
        """Build a [](api:HistoryDataCollector) from a loosely-typed raw value.

        Parameters
        ----------
        raw : HistoryDataCollectorLike
            Either an already-built [](api:HistoryDataCollector); a bare `str`,
            taken as `key` alone; a `Mapping`, unpacked as constructor kwargs;
            or a `tuple` of positional constructor arguments (`(key,)` or
            `(key, indices)`).

        Returns
        -------
        HistoryDataCollector
            The resulting collector.

        Raises
        ------
        TypeError
            If `raw` is a `list` (see [](api:HistoryDataCollectorLike)), or is
            some other type this cannot cast.
        """
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str):
            return cls(key=raw)
        if isinstance(raw, Mapping):
            return cls(**raw)
        if isinstance(raw, list):
            raise TypeError(
                "A list combines several collectors; use a tuple for the "
                "(key, indices) sugar, e.g. ('logical_measurement', -1)."
            )
        if isinstance(raw, Sequence):
            return cls(*raw)
        raise TypeError(f"Cannot cast {raw!r} to a HistoryDataCollector")

    def collect(self, program_results: ProgramResults) -> list:
        """Collect this recipe's data from a [](api:ProgramResults)' shots.

        Scoped to [](api:ProgramResults) since that is the only thing any
        current caller needs.
        """
        return program_results.collect_shot_data(
            self.key,
            self.indices,
            strip_none_entries=self.strip_none_entries,
            frame_filter=self.frame_filter,
        )
