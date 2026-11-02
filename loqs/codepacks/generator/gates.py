#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""The gate table: one entry per gate name a codepack spec may use.

Each Clifford gate's unitary is a key of `STANDARD_GATE_UNITARIES`;
`loqs.codepacks.generator.derive` turns it into a Stim tableau. This module
needs no Stim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loqs.backends.reps import STANDARD_GATE_UNITARIES


@dataclass(frozen=True)
class GateEntry:
    """A gate name's kind, unitary source and duration class.

    `kind` is "gate", "idle" or "instrument". `unitary` names a key of
    `STANDARD_GATE_UNITARIES` (gates and idles); instruments have none.
    `duration` is a class name ("1q", "2q", "mcm") that the renderer maps
    to a number and an idle name. `arity` is the number of targets of one
    application.
    """

    name: str
    kind: str
    unitary: str | None
    duration: str
    arity: int


GATES: dict[str, GateEntry] = {
    e.name: e
    for e in (
        GateEntry("Gxpi", "gate", "X", "1q", 1),
        GateEntry("Gypi", "gate", "Y", "1q", 1),
        GateEntry("Gzpi", "gate", "Z", "1q", 1),
        GateEntry("Gzpi2", "gate", "S", "1q", 1),
        GateEntry("Gzmpi2", "gate", "S_DAG", "1q", 1),
        GateEntry("Gh", "gate", "H", "1q", 1),
        GateEntry("Gcnot", "gate", "CX", "2q", 2),
        GateEntry("Gcphase", "gate", "CZ", "2q", 2),
        GateEntry("Gi", "gate", "I", "1q", 1),
        GateEntry("Gi1Q", "idle", "I", "1q", 1),
        GateEntry("Gi2Q", "idle", "I", "2q", 1),
        GateEntry("GiMCM", "idle", "I", "mcm", 1),
        GateEntry("Imrz", "instrument", None, "mcm", 1),
    )
}

DEFAULT_DURATIONS = {"1q": 1, "2q": 2, "mcm": 3}
"""Duration of each duration class, in the old Steane module's units."""

DEFAULT_IDLES = {"1q": "Gi1Q", "2q": "Gi2Q", "mcm": "GiMCM"}
"""The idle gate name of each duration class."""


def unitary(name: str) -> np.ndarray:
    """The unitary of a gate or idle name (big-endian over its targets)."""
    e = GATES[name]
    if e.unitary is None:
        raise ValueError(f"{name} is an instrument and has no unitary")
    return np.asarray(STANDARD_GATE_UNITARIES[e.unitary])
