#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""[](api:OutputResults) definition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from loqs.internal import Displayable


class OutputResults(Displayable):
    """The declared outputs of a run, one entry per shot, and nothing else.

    It keeps only a run's declared outputs, with no per-shot
    [](api:History), so it is the outputs-only counterpart of
    [](api:ProgramResults). Each output is stored as one 1-D array of
    values plus an `int64` array of offsets, which holds regular and
    ragged outputs alike. Nothing in the class is tied to a backend.
    """

    _SERIALIZE_ATTRS = ["name", "_values", "_offsets"]

    @classmethod
    def _from_decoded_attrs(cls, attr_dict) -> "OutputResults":
        # Stored arrays were validated when first built, so skip `outputs`
        obj = cls.__new__(cls)
        obj.name = attr_dict["name"]
        obj._values = attr_dict["_values"]
        obj._offsets = attr_dict["_offsets"]
        return obj

    def __init__(
        self,
        outputs: Mapping[str, Any],
        name: str = "(Unnamed output results)",
    ) -> None:
        """
        Parameters
        ----------
        outputs:
            Maps an output name to a 2-D array `(shots, n)`, or to a
            sequence of per-shot 1-D arrays of any lengths.

        name:
            Name for logging.

        Raises
        ------
        ValueError
            If the outputs don't all have the same number of shots, or an
            output isn't 2-D or a sequence of 1-D arrays.
        """
        self.name = name
        self._values: dict[str, NDArray] = {}
        self._offsets: dict[str, NDArray] = {}
        for key, out in outputs.items():
            if isinstance(out, np.ndarray) and out.ndim == 2:
                shots, width = out.shape
                values = out.reshape(-1)
                offsets = np.arange(shots + 1, dtype=np.int64) * width
            else:
                records = [np.asarray(r) for r in out]
                if any(r.ndim != 1 for r in records):
                    raise ValueError(
                        f"Output {key!r} must be a 2-D array or a sequence "
                        "of 1-D arrays"
                    )
                lengths = [len(r) for r in records]
                offsets = np.zeros(len(records) + 1, dtype=np.int64)
                np.cumsum(lengths, out=offsets[1:])
                values = (
                    np.concatenate(records)
                    if records
                    else np.zeros(0, dtype=np.float64)
                )
            self._values[key] = values
            self._offsets[key] = offsets

        counts = {k: len(o) - 1 for k, o in self._offsets.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(f"Outputs have different shot counts: {counts}")

    @property
    def num_shots(self) -> int:
        """The number of shots, shared by every output (0 if none)."""
        for offsets in self._offsets.values():
            return len(offsets) - 1
        return 0

    def keys(self) -> list[str]:
        """The output names."""
        return list(self._values)

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def records(self, name: str) -> list[NDArray]:
        """Output `name` as a list of per-shot 1-D arrays."""
        values, offsets = self._values[name], self._offsets[name]
        return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]

    def __getitem__(self, name: str) -> NDArray:
        """Output `name` as a `(shots, n)` array, without a copy.

        Raises
        ------
        ValueError
            If the shots don't all have the same number of entries; use
            [](api:OutputResults.records) then.
        """
        values, offsets = self._values[name], self._offsets[name]
        widths = np.diff(offsets)
        shots = len(widths)
        if shots and not (widths == widths[0]).all():
            raise ValueError(
                f"Output {name!r} has shots of different lengths; "
                "use records() instead"
            )
        width = int(widths[0]) if shots else 0
        return values.reshape(shots, width)
