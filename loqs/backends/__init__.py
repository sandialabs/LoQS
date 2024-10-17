"""Quantum simulation backends for LoQS
"""

from .reps import RepEnum, GateRep, InstrumentRep, RepTuple

from .circuit import (
    BasePhysicalCircuit,
    PyGSTiPhysicalCircuit,
    ListPhysicalCircuit,
)

# Needs to be after circuit import but before state so that we have OpRep
from .model import (
    BaseNoiseModel,
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

    reset_mcms:
        Whether to reset the qubit state after an instrument
        (``True``, default) or not.

    Returns
    -------
    BaseQuantumState, OutcomeDict
        The output of :meth:`.BaseQuantumState.apply_reps`.
        If ``inplace=True``, then the state is also returned
        to provide a consistent API.
    """
    # Find a compatible model/state oprep
    oprep: GateRep | None = None
    for grep in model.output_gate_reps:
        if grep in state.input_reps:
            oprep = grep
            break
    assert (
        oprep is not None
    ), "Could not find matching gate rep between model output and state input"

    instrep: InstrumentRep | None = None
    for irep in model.output_instrument_reps:
        if irep in state.input_reps:
            instrep = irep
            break
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
