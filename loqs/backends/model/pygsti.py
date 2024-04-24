""":class:`PyGSTiNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Iterable
import functools
import itertools
from typing import TypeAlias, Union
import numpy as np

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.circuit.pygsti import PyGSTiPhysicalCircuit
from loqs.backends.model import BaseNoiseModel, OpRep


class PyGSTiNoiseModel(BaseNoiseModel):
    """Model backend for handling ``pygsti.model.OpModel`` objects."""

    @property
    def AllowedModelTypes(self) -> TypeAlias:
        try:
            from pygsti.models import ExplicitOpModel, ImplicitOpModel
        except ImportError as e:
            raise ImportError(
                "Failed import, cannot use pyGSTi as backend"
            ) from e
        return Union[ExplicitOpModel, ImplicitOpModel]

    def __init__(self, model: AllowedModelTypes) -> None:
        """Initialize a PyGSTiModelBackend.

        Parameters
        ----------
        model:
            A pyGSTi model to use when looking up operations
        """
        try:
            from pygsti.models import ExplicitOpModel, ImplicitOpModel
        except ImportError as e:
            raise ImportError(
                "Failed import, cannot use pyGSTi as backend"
            ) from e

        self.model = model
        if isinstance(self.model, ExplicitOpModel):
            self.gate_dict = self.model.operations
        elif isinstance(self.model, ImplicitOpModel):
            self.gate_dict = self.model.operation_blks["gates"]
        else:
            raise TypeError("Can only take Explicit or Implicit OpModels")

        # TODO: Crosstalk specification?

    @property
    def name(self) -> str:
        return "pyGSTi"

    @property
    def CircuitBackendInputs(self) -> Iterable[BasePhysicalCircuit]:
        """PyGSTi backend circuit type (pygsti.circuits.Circuit)"""
        return [PyGSTiPhysicalCircuit]

    @property
    def OpRepOutputs(self) -> Iterable[OpRep]:
        return [OpRep.UNITARY, OpRep.PTM, OpRep.PTM_QSIM]

    def get_operator_reps(
        self, circuit: PyGSTiPhysicalCircuit, reptype: OpRep
    ) -> Iterable:
        try:
            from pygsti.baseobjs import ExplicitBasis
            from pygsti.tools import basistools as bt
        except ImportError as e:
            raise ImportError(
                "Failed import, cannot use pyGSTi as backend"
            ) from e

        # Check we can process it
        super().get_operator_reps(circuit, reptype)

        pygsti_circuit = circuit.get_bare_circuit()

        # Prep QuantumSim bases
        sig0q = np.array([[1.0, 0], [0, 0]], dtype="complex")
        sigXq = np.array([[0, 1], [1, 0]], dtype="complex") / np.sqrt(2)
        sigYq = (
            np.array([[0, -1], [1, 0]], dtype="complex") * 1.0j / np.sqrt(2.0)
        )
        sig1q = np.array([[0, 0], [0, 1]], dtype="complex")

        q1basis = [sig0q, sigXq, sigYq, sig1q]
        qsim1 = ExplicitBasis(
            q1basis,
            ["myEl%d" % i for i in range(4**1)],
            name="qsim_1q",
            longname="QuantumSim_1qubit",
        )

        q2basis = itertools.product(q1basis, repeat=2)
        q2basis = [functools.reduce(np.kron, *x) for x in q2basis]
        qsim2 = ExplicitBasis(
            q1basis,
            ["myEl%d" % i for i in range(4**2)],
            name="qsim_2q",
            longname="QuantumSim_2qubit",
        )

        # Iterate through circuit and pull out representations
        # TODO: What to do about instruments
        op_reps = []
        for layer in pygsti_circuit:
            for comp in layer.components:
                qubits = comp.qubits
                op = self.gate_dict[comp]
                if reptype == OpRep.UNITARY:
                    try:
                        rep = op.to_dense(on_space="Hilbert")
                    except ValueError as e:
                        raise ValueError(
                            "Failed to cast operation as a unitary. Consider ",
                            "using process matrices instead.",
                        ) from e
                elif reptype == OpRep.PTM:
                    rep = op.to_dense(on_space="HilbertSchmidt")
                elif reptype == OpRep.PTM_QSIM:
                    rep = op.to_dense(on_space="HilbertSchmidt")

                    if comp.num_qubits == 1:
                        rep = bt.change_basis(rep, self.model.basis, qsim1)
                    elif comp.num_qubits == 2:
                        rep = bt.change_basis(rep, self.model.basis, qsim2)
                    else:
                        raise ValueError(
                            "Cannot change more than a 2 qubit operation into",
                            " the QuantumSim basis",
                        )

                op_reps.append((rep, qubits))

        return op_reps
