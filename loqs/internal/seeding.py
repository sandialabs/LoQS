#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3.1                                                                         #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Helpers for building and saving NumPy `SeedSequence` streams."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import SeedSequence

STATE_STREAM = 0
"""Consumer key of a shot's quantum states."""

INSTRUCTION_STREAM = 1
"""Consumer key of the seed handed to a shot's instructions."""


def as_seed_sequence(seed: int | SeedSequence | None) -> SeedSequence:
    """Turn a seed into a `SeedSequence`.

    A `SeedSequence` is returned as the same object. Anything else goes to
    `SeedSequence(seed)`, so None draws fresh entropy and a bad value raises
    NumPy's own error.
    """
    if isinstance(seed, SeedSequence):
        return seed
    return SeedSequence(seed)


def child_seed_sequence(parent: SeedSequence, *key: int) -> SeedSequence:
    """Derive the child of `parent` at `key`, independent of call order.

    The result equals what `parent.spawn()` would give for the matching
    index, but never touches the parent's spawn counter.
    """
    return SeedSequence(
        parent.entropy,
        spawn_key=tuple(parent.spawn_key) + tuple(key),
        pool_size=parent.pool_size,
    )


def seed_word(seq: SeedSequence) -> int:
    """A 64-bit integer drawn from `seq`, for APIs that take only an integer."""
    return int(seq.generate_state(1, np.uint64)[0])


def seed_sequence_to_dict(seq: SeedSequence) -> dict[str, Any]:
    """The saved form of a sequence. Equal dicts mean the same seed."""
    return {
        "entropy": seq.entropy,
        "spawn_key": list(seq.spawn_key),
        "pool_size": seq.pool_size,
    }


def seed_sequence_from_dict(d: dict[str, Any]) -> SeedSequence:
    """Inverse of `seed_sequence_to_dict`."""
    return SeedSequence(
        d["entropy"],
        spawn_key=tuple(d["spawn_key"]),
        pool_size=d["pool_size"],
    )
