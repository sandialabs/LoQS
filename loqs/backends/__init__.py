#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Quantum simulation backends for LoQS
"""

from .reps import RepEnum, GateRep, InstrumentRep, RepTuple

from .circuit import (
    BasePhysicalCircuit,
    ListPhysicalCircuit,
    PyGSTiPhysicalCircuit,
    STIMPhysicalCircuit,
)

# Needs to be after circuit import but before state so that we have OpRep
from .model import (
    BaseNoiseModel,
    DictNoiseModel,
    PyGSTiNoiseModel,
    STIMDictNoiseModel,
    TimeDependentBaseNoiseModel,
)

from .state import (
    BaseQuantumState,
    OutcomeDict,
    NumpyStatevectorQuantumState,
    QSimQuantumState,
    STIMQuantumState,
)


def propagate_state(
    circuit: BasePhysicalCircuit,
    model: BaseNoiseModel,
    state: BaseQuantumState,
    inplace: bool = True,
) -> tuple[BaseQuantumState, OutcomeDict]:
    """Given a circuit and model, propagate a state forward in time.

    This is a wrapper for :meth:`.BaseNoiseModel.get_reps`
    and :meth:`.BaseQuantumState.apply_reps_inplace` (or
    the non-inplace version if ``inplace=False``).
    It does also try to find compatible reptypes by
    searching for a match in output reps from ``model``
    and input reps from ``state``.

    Parameters
    ----------
    circuit:
        The circuit to run

    model:
        The noise model to use to convert circuit operations
        into representations the state can apply

    state:
        The state to move forward in time

    inplace:
        Whether to modify the state in-place (``True``, default)
        or propagate a copy forward (``False``).
        This should probably remain ``True`` for memory reasons.

    Returns
    -------
    BaseQuantumState, OutcomeDict
        The output of :meth:`.BaseQuantumState.apply_reps`.
        If ``inplace=True``, then the state is also returned
        to provide a consistent API.
    """
    # Find a compatible model/state oprep
    opreps = []
    for oprep in model.output_gate_reps:
        if oprep in state.input_reps:
            opreps.append(oprep)
    assert (
        len(opreps) > 0
    ), "Could not find matching gate rep between model output and state input"

    instreps = []
    for instrep in model.output_instrument_reps:
        if instrep in state.input_reps:
            instreps.append(instrep)
    assert (
        len(instreps) > 0
    ), "Could not find matching instrument rep between model output and state input"

    # Look up reps from model
    reps = model.get_reps(circuit, list(opreps), list(instreps))

    # Apply operator reps to state
    if inplace:
        outcomes = state.apply_reps_inplace(reps)
    else:
        state, outcomes = state.apply_reps(reps)

    return state, outcomes
