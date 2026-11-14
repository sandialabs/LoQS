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
    [](api:ProgramResults). Each output has a depth, the nesting of one
    shot's record: 0 is a scalar, 1 a 1-D array, and 2 a list of 1-D
    arrays (such as one per frame). Outputs are given either as dense
    arrays of shape `(shots,)`, `(shots, n)` or `(shots, f, n)`, or as a
    sequence of per-shot records, which may be ragged. Storage is flat:
    one 1-D array of values plus `int64` offsets, with a second level of
    offsets for depth 2. Nothing in the class is tied to a backend.
    """

    _SERIALIZE_ATTRS = [
        "name",
        "_values",
        "_offsets",
        "_inner_offsets",
        "_depths",
    ]

    @classmethod
    def _from_decoded_attrs(cls, attr_dict) -> "OutputResults":
        # Stored arrays were validated when first built, so skip `outputs`
        obj = cls.__new__(cls)
        obj.name = attr_dict["name"]
        obj._values = attr_dict["_values"]
        obj._offsets = attr_dict["_offsets"]
        obj._inner_offsets = attr_dict["_inner_offsets"]
        obj._depths = attr_dict["_depths"]
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
            Maps an output name to either a dense `np.ndarray` of shape
            `(shots,)` (depth 0, scalars), `(shots, n)` (depth 1) or
            `(shots, f, n)` (depth 2), or a sequence of per-shot records.
            A record that is a `list` or `tuple` whose items are all
            arrays, lists or tuples is depth 2, each item a 1-D array;
            any other record is a 1-D array (depth 1). Records may have
            different lengths, and an empty record takes the depth of the
            output's other records.

        name:
            Name for logging.

        Raises
        ------
        ValueError
            If the outputs don't all have the same number of shots, an
            output's records differ in depth, or an output has any other
            shape.
        """
        self.name = name
        self._values: dict[str, NDArray] = {}
        self._offsets: dict[str, NDArray] = {}
        self._inner_offsets: dict[str, NDArray] = {}
        self._depths: dict[str, int] = {}
        for key, out in outputs.items():
            if isinstance(out, np.ndarray):
                depth, values, offsets, inner = self._from_dense(key, out)
            else:
                depth, values, offsets, inner = self._from_records(key, out)
            self._values[key] = values
            self._offsets[key] = offsets
            self._depths[key] = depth
            if inner is not None:
                self._inner_offsets[key] = inner

        counts = {k: len(o) - 1 for k, o in self._offsets.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(f"Outputs have different shot counts: {counts}")

    @staticmethod
    def _from_dense(key: str, out: NDArray):
        if out.ndim not in (1, 2, 3):
            raise ValueError(
                f"Output {key!r} must have 1 to 3 dimensions, "
                f"not {out.ndim}"
            )
        shots = out.shape[0]
        values = out.reshape(-1)
        if out.ndim < 3:
            width = 1 if out.ndim == 1 else out.shape[1]
            offsets = np.arange(shots + 1, dtype=np.int64) * width
            return out.ndim - 1, values, offsets, None
        f, n = out.shape[1:]
        offsets = np.arange(shots + 1, dtype=np.int64) * f
        inner = np.arange(shots * f + 1, dtype=np.int64) * n
        return 2, values, offsets, inner

    @staticmethod
    def _from_records(key: str, out: Any):
        records = list(out)
        nested = (list, tuple)
        depths = set()
        for r in records:
            if isinstance(r, (np.ndarray, *nested)) and len(r) == 0:
                continue
            is_two = isinstance(r, nested) and all(
                isinstance(i, (np.ndarray, *nested)) for i in r
            )
            depths.add(2 if is_two else 1)
        if len(depths) > 1:
            raise ValueError(f"Output {key!r} has records of different depths")
        depth = depths.pop() if depths else 1

        if depth == 1:
            arrays = [np.asarray(r) for r in records]
            if any(a.ndim != 1 for a in arrays):
                raise ValueError(
                    f"Output {key!r} must be a dense array or a sequence "
                    "of 1-D arrays or lists of 1-D arrays"
                )
            inner = None
            lengths = [len(a) for a in arrays]
        else:
            arrays = []
            lengths = []
            for r in records:
                items = [np.asarray(i) for i in r]
                if any(a.ndim != 1 for a in items):
                    raise ValueError(
                        f"Output {key!r} must be a dense array or a "
                        "sequence of 1-D arrays or lists of 1-D arrays"
                    )
                arrays.extend(items)
                lengths.append(len(items))
            inner = np.zeros(len(arrays) + 1, dtype=np.int64)
            np.cumsum([len(a) for a in arrays], out=inner[1:])

        offsets = np.zeros(len(records) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        values = (
            np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float64)
        )
        return depth, values, offsets, inner

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

    def records(self, name: str) -> list:
        """Output `name` as a list of per-shot records at its depth.

        Each record is a numpy scalar (depth 0), a 1-D array (depth 1) or
        a list of 1-D arrays (depth 2).
        """
        values, offsets = self._values[name], self._offsets[name]
        depth = self._depths[name]
        spans = zip(offsets[:-1], offsets[1:])
        if depth == 0:
            return [values[a] for a, _ in spans]
        if depth == 1:
            return [values[a:b] for a, b in spans]
        inner = self._inner_offsets[name]
        return [
            [values[inner[i] : inner[i + 1]] for i in range(a, b)]
            for a, b in spans
        ]

    def __getitem__(self, name: str) -> NDArray:
        """Output `name` as a dense array, without a copy.

        The shape is `(shots,)`, `(shots, n)` or `(shots, f, n)` by depth.

        Raises
        ------
        ValueError
            If the shots, or a shot's inner arrays, don't all have the
            same length; use [](api:OutputResults.records) then.
        """
        values, offsets = self._values[name], self._offsets[name]
        depth = self._depths[name]
        widths = np.diff(offsets)
        shots = len(widths)
        if shots and not (widths == widths[0]).all():
            raise ValueError(
                f"Output {name!r} has shots of different lengths; "
                "use records() instead"
            )
        width = int(widths[0]) if shots else 0
        if depth == 0:
            return values.reshape(shots)
        if depth == 1:
            return values.reshape(shots, width)
        lengths = np.diff(self._inner_offsets[name])
        if len(lengths) and not (lengths == lengths[0]).all():
            raise ValueError(
                f"Output {name!r} has inner arrays of different lengths; "
                "use records() instead"
            )
        n = int(lengths[0]) if len(lengths) else 0
        return values.reshape(shots, width, n)
