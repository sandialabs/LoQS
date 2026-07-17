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

[](api:upgrade_gate_rep)/[](api:upgrade_instrument_rep) convert a raw,
pre-refactor-style payload (e.g. a bare `numpy.ndarray` or `str`) into the
appropriate concrete class, given a list of candidate classes to try.

[](api:RepTuple) is retained only so that `.json`/`.h5` files serialized
before this class hierarchy existed continue to load correctly; it cannot
be constructed by new code (see its docstring).
"""

from loqs.backends.reps.base import (
    OperationRep,
    RepConstructionError,
    is_rep_compatible,
)
from loqs.backends.reps.gatereps import (
    GateRep,
    KrausGateRep,
    PTMGateRep,
    ProbabilisticStimGateRep,
    QSimSuperoperatorGateRep,
    StimCircuitGateRep,
    UnitaryGateRep,
)
from loqs.backends.reps.instrumentreps import (
    GateUpgrader,
    InstrumentRep,
    StimCircuitInstrumentRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.backends.reps.construction import upgrade_gate_rep, upgrade_instrument_rep
from loqs.backends.reps.legacy import RepTuple

__all__ = [
    "OperationRep",
    "RepConstructionError",
    "is_rep_compatible",
    "GateRep",
    "UnitaryGateRep",
    "PTMGateRep",
    "QSimSuperoperatorGateRep",
    "StimCircuitGateRep",
    "ProbabilisticStimGateRep",
    "KrausGateRep",
    "InstrumentRep",
    "GateUpgrader",
    "ZBasisProjectionInstrumentRep",
    "ZBasisPrePostInstrumentRep",
    "ZBasisOutcomeOperationDictInstrumentRep",
    "StimCircuitInstrumentRep",
    "upgrade_gate_rep",
    "upgrade_instrument_rep",
    "RepTuple",
]
