"""Quantum simulation backends for LoQS
"""

from typing import TypeAlias, Union, Literal


### CIRCUIT BACKENDS ###
from .circuit import CircuitBackend
from .circuit.pygsticircuitbackend import PyGSTiCircuitBackend

### MODEL BACKENDS ###
from .model import ModelBackend
from .model.pygstimodelbackend import PyGSTiModelBackend

### STATE BACKENDS ###

CircuitBackendCastable: TypeAlias = Union[CircuitBackend, Literal["pygsti"]]


def cast_circuit_backend(backend: CircuitBackendCastable) -> CircuitBackend:
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
    if isinstance(backend, CircuitBackend):
        return backend
    elif backend is None or backend == "pygsti":
        return PyGSTiCircuitBackend()
    else:
        raise ValueError(f"Cannot cast a CircuitBackend from {backend}")
