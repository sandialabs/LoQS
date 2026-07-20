#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Gate/instrument operation representation classes.

An operation representation ([](api:OperationRep)) describes how a gate or
instrument (mid-circuit measurement) operation should be applied to a state:
e.g. as a unitary matrix, a set of Kraus operators, or a STIM circuit
string. [](api:GateRep) and [](api:InstrumentRep) are the abstract base
classes for gate-level and instrument-level representations, respectively;
each concrete subclass (e.g. [](api:UnitaryGateRep),
[](api:KrausGateRep), [](api:ZBasisProjectionInstrumentRep)) carries its own
descriptive, named payload field(s) rather than a single untyped `rep`
attribute paired with an enum tag.

[](api:convert) converts a raw, unwrapped payload or an
already-constructed [](api:OperationRep) into a requested target class (or
the closest of several candidate classes), via `matches`/`from_raw` and/or
a shortest-path search over a registry of pairwise numeric/structural
converters between concrete classes (e.g. [](api:UnitaryGateRep) to
[](api:PTMGateRep) to [](api:KrausGateRep)).

[](api:RepTuple) is retained only so that `.json`/`.h5` files serialized
before this class hierarchy existed continue to load correctly; it cannot
be constructed by new code (see its docstring).

[](api:StimCircuitPayloadMixin) factors out the storage/construction logic
shared by [](api:StimCircuitGateRep) and [](api:StimCircuitInstrumentRep),
which otherwise sit in unrelated branches of the [](api:GateRep)/
[](api:InstrumentRep) hierarchy but wrap the exact same STIM
circuit-string payload shape.

[](api:STANDARD_GATE_UNITARIES) is a small, fixed set of well-known
single/two-qubit gate unitaries (used internally to build the N-qubit
Pauli basis the pure-numpy `GateRep` conversions are built on), exposed
publicly so other callers needing common gate matrices don't need a
second hand-copied set.
"""

from loqs.backends.reps.base import (
    OperationRep,
    RepConstructionError,
    StimCircuitPayloadMixin,
    is_rep_compatible,
)
from loqs.backends.reps.gatereps import (
    GateRep,
    KrausGateRep,
    PTMGateRep,
    ProbabilisticStimGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    UnitaryGateRep,
)
from loqs.backends.reps.instrumentreps import (
    InstrumentRep,
    StimCircuitInstrumentRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.backends.reps.conversion import STANDARD_GATE_UNITARIES, convert
from loqs.backends.reps.legacy import RepTuple

__all__ = [
    "OperationRep",
    "RepConstructionError",
    "StimCircuitPayloadMixin",
    "is_rep_compatible",
    "GateRep",
    "UnitaryGateRep",
    "PTMGateRep",
    "QSimSuperopGateRep",
    "StimCircuitGateRep",
    "ProbabilisticStimGateRep",
    "KrausGateRep",
    "InstrumentRep",
    "ZBasisProjectionInstrumentRep",
    "ZBasisPrePostInstrumentRep",
    "ZBasisOutcomeOperationDictInstrumentRep",
    "StimCircuitInstrumentRep",
    "STANDARD_GATE_UNITARIES",
    "convert",
    "RepTuple",
]
