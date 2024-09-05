"""Quantum simulation backends for LoQS
"""

from .circuit import BasePhysicalCircuit, PyGSTiPhysicalCircuit

# Needs to be after circuit import but before state so that we have OpRep
from .model import (
    BaseNoiseModel,
    GateRep,
    InstrumentRep,
    PyGSTiNoiseModel,
    DictNoiseModel,
)

from .state import BaseQuantumState, OutcomeDict, QSimQuantumState


def propagate_state(
    circuit: BasePhysicalCircuit,
    model: BaseNoiseModel,
    state: BaseQuantumState,
    inplace: bool = True,
    reset_mcms: bool = True,
) -> tuple[BaseQuantumState, OutcomeDict]:
    """TODO"""
    # Find a compatible model/state oprep
    oprep: GateRep | None = None
    for rep in model.output_gate_reps:
        if rep in state.input_reps:
            oprep = rep
    assert (
        oprep is not None
    ), "Could not find matching gate rep between model output and state input"

    instrep: InstrumentRep | None = None
    for rep in model.output_instrument_reps:
        if rep in state.input_reps:
            instrep = rep
    assert (
        instrep is not None
    ), "Could not find matching instrument rep between model output and state input"

    # Look up reps from model
    reps = model.get_reps(circuit, oprep, instrep)

    # Apply operator reps to state
    if inplace:
        outcomes = state.apply_reps_inplace(reps, reset_mcms)
    else:
        state, outcomes = state.apply_reps(reps, reset_mcms)

    return state, outcomes
