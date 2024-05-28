""":class:`PyGSTiNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Iterable
import functools
import itertools
import numpy as np

from loqs.backends import OpRep
from loqs.backends.circuit import BasePhysicalCircuit, PyGSTiPhysicalCircuit
from loqs.internal.castable import Castable
from loqs.internal.classproperty import roclassproperty


class PyGSTiNoiseModel(Castable):
    """Model backend for handling ``pygsti.model.OpModel`` objects."""

    @roclassproperty
    def CastableTypes(self) -> type:
        try:
            from pygsti.models import ExplicitOpModel, ImplicitOpModel
        except ImportError as e:
            raise ImportError(
                "Failed import, cannot use pyGSTi as backend"
            ) from e
        return ExplicitOpModel | ImplicitOpModel

    def __init__(self, model: CastableTypes) -> None:
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

    @roclassproperty
    def name(self) -> str:
        return "pyGSTi"

    @roclassproperty
    def CircuitBackendInputs(self) -> type[BasePhysicalCircuit]:
        """PyGSTi backend circuit type (pygsti.circuits.Circuit)"""
        return PyGSTiPhysicalCircuit

    @roclassproperty
    def OpRepOutputs(self) -> type[OpRep]:
        return OpRep.UNITARY | OpRep.PTM | OpRep.QSIM_SUPEROPERATOR

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
                elif reptype == OpRep.QSIM_SUPEROP:
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
