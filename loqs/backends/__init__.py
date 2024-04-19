"""Quantum simulation backends for LoQS
"""

from typing import TypeAlias, Union, Literal


### CIRCUIT BACKENDS ###
from .circuit import BaseCircuitBackend
from .circuit.pygstibackend import PyGSTiCircuitBackend

### MODEL BACKENDS ###
from .model import BaseNoiseModel
from .model.pygstimodel import PyGSTiNoiseModel

### STATE BACKENDS ###
from .state import BaseQuantumState
from .state.qsimstate import QSimQuantumState


### Circuit Backend Utilities
### TODO: Get rid of?
CircuitBackendCastable: TypeAlias = Union[
    BaseCircuitBackend, Literal["pygsti"]
]


def cast_circuit_backend(
    backend: CircuitBackendCastable,
) -> BaseCircuitBackend:
    """Helper function to create CircuitBackends from strings.

    Parameters
    ----------
    backend: CircuitBackend or {"pygsti"}
        Backend object to cast. Defaults to PyGSTiCircuitBackend

    Returns
    -------
    CircuitBackend
        The created circuit backend
    """
    if isinstance(backend, BaseCircuitBackend):
        return backend
    elif backend is None or backend == "pygsti":
        return PyGSTiCircuitBackend()
    else:
        raise ValueError(f"Cannot cast a CircuitBackend from {backend}")
