""":class:`PyGSTiCircuitBackend` definition.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional, Type, TypeAlias, Union

from pygsti.models.model import OpModel
from pygsti.circuits import Circuit as PyGSTiCircuit

from loqs.backends.circuit import CircuitBackend, PyGSTiCircuitBackend
from loqs.backends.model import ModelBackend, OpRep
from loqs.core.physicalcircuit import PhysicalCircuit


class PyGSTiModelBackend(ModelBackend):
    """Model backend for handling :class:`pygsti.model.OpModel`s."""

    def __init__(self, model: OpModel) -> None:
        """Initialize a PyGSTiModelBackend.

        Parameters
        ----------
        model:
            A pyGSTi model to use when looking up operations
        """
        self.model = model

        # TODO: Crosstalk specification?

    @property
    def name(self) -> str:
        return "pyGSTi"

    @property
    def CircuitBackendInputs(self) -> Iterable[CircuitBackend]:
        """PyGSTi backend circuit type (pygsti.circuits.Circuit)"""
        return [PyGSTiCircuitBackend]

    @property
    def OpRepOutputs(self) -> Iterable[OpRep]:
        return [OpRep.UNITARY, OpRep.PTM, OpRep.PTM_QSIM]

    def get_operator_reps(
        self, circuit: PhysicalCircuit, reptype: OpRep
    ) -> Iterable:
        super().get_operator_reps(circuit, reptype)

        raise NotImplementedError("TODO")
