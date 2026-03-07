"""Quantum simulation backends for LoQS
"""

from typing import TYPE_CHECKING, Any
from importlib import import_module
from dataclasses import dataclass

from .reps import RepEnum, GateRep, InstrumentRep, RepTuple

from .circuit import BasePhysicalCircuit
from .model import BaseNoiseModel
from .state import BaseQuantumState
from .state.basestate import OutcomeDict


@dataclass
class BackendAvailability:
    """Class to track backend availability"""

    name: str
    available: bool
    error: str | None = None


# Backend availability tracking
_backend_availability: dict[str, BackendAvailability] = {}


def _check_backend_availability(backend_name: str, import_path: str) -> bool:
    """Check if a backend is available and update availability tracking"""
    try:
        import_module(import_path)
        _backend_availability[backend_name] = BackendAvailability(
            backend_name, True
        )
        return True
    except ImportError as e:
        _backend_availability[backend_name] = BackendAvailability(
            backend_name, False, str(e)
        )
        return False


def get_available_backends() -> list[str]:
    """Get list of available backend names"""
    return [
        name
        for name, avail in _backend_availability.items()
        if avail.available
    ]


def is_backend_available(backend_name: str) -> bool:
    """Check if a specific backend is available"""
    return _backend_availability.get(
        backend_name, BackendAvailability(backend_name, False)
    ).available


def get_backend_error(backend_name: str) -> str | None:
    """Get the error message for an unavailable backend"""
    return _backend_availability.get(
        backend_name, BackendAvailability(backend_name, False)
    ).error


# Check availability of all backends at import time
_check_backend_availability("pygsti_circuit", "pygsti.circuits")
_check_backend_availability("pygsti_model", "pygsti.objects")
_check_backend_availability("stim_circuit", "stim")
_check_backend_availability("stim_state", "stim")
_check_backend_availability("qsim", "quantumsim")


# Import concrete backend classes with conditional availability
def __getattr__(name: str) -> Any:
    """Lazy import of backend classes based on availability"""
    if name == "PyGSTiPhysicalCircuit":
        if is_backend_available("pygsti_circuit"):
            from .circuit.pygsticircuit import PyGSTiPhysicalCircuit

            return PyGSTiPhysicalCircuit
        else:
            raise ImportError(
                f"PyGSTi circuit backend is not available. "
                f"Error: {get_backend_error('pygsti_circuit')}"
            )
    elif name == "STIMPhysicalCircuit":
        if is_backend_available("stim_circuit"):
            from .circuit.stimcircuit import STIMPhysicalCircuit

            return STIMPhysicalCircuit
        else:
            raise ImportError(
                f"STIM circuit backend is not available. "
                f"Error: {get_backend_error('stim_circuit')}"
            )
    elif name == "PyGSTiNoiseModel":
        if is_backend_available("pygsti_model"):
            from .model.pygstimodel import PyGSTiNoiseModel

            return PyGSTiNoiseModel
        else:
            raise ImportError(
                f"PyGSTi model backend is not available. "
                f"Error: {get_backend_error('pygsti_model')}"
            )
    elif name == "STIMQuantumState":
        if is_backend_available("stim_state"):
            from .state.stimstate import STIMQuantumState

            return STIMQuantumState
        else:
            raise ImportError(
                f"STIM state backend is not available. "
                f"Error: {get_backend_error('stim_state')}"
            )
    elif name == "QSimQuantumState":
        if is_backend_available("qsim"):
            from .state.qsimstate import QSimQuantumState

            return QSimQuantumState
        else:
            raise ImportError(
                f"QSim backend is not available. "
                f"Error: {get_backend_error('qsim')}"
            )

    # Always available backends
    elif name == "ListPhysicalCircuit":
        from .circuit.listcircuit import ListPhysicalCircuit

        return ListPhysicalCircuit
    elif name == "DictNoiseModel":
        from .model.dictmodel import DictNoiseModel

        return DictNoiseModel
    elif name == "NumpyStatevectorQuantumState":
        from .state.npsvstate import NumpyStatevectorQuantumState

        return NumpyStatevectorQuantumState

    # Base classes
    elif name == "BasePhysicalCircuit":
        from .circuit.basecircuit import BasePhysicalCircuit

        return BasePhysicalCircuit
    elif name == "BaseNoiseModel":
        from .model.basemodel import BaseNoiseModel

        return BaseNoiseModel
    elif name == "BaseQuantumState":
        from .state.basestate import BaseQuantumState

        return BaseQuantumState
    elif name == "OutcomeDict":
        from .state.basestate import OutcomeDict

        return OutcomeDict

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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
