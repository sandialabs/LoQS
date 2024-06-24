""":class:`PyGSTiNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Sequence
import functools
import itertools
from typing import TypeAlias
import numpy as np

from loqs.backends.circuit import PyGSTiPhysicalCircuit
from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel, GateRep, InstrumentRep


try:
    from pygsti.baseobjs import ExplicitBasis
    from pygsti.models import ExplicitOpModel, ImplicitOpModel
    from pygsti.tools import basistools as bt
except ImportError as e:
    raise ImportError("Failed import, cannot use pyGSTi as backend") from e

# Type aliases for static type checking
PyGSTiModelCastableTypes: TypeAlias = (
    ExplicitOpModel | ImplicitOpModel
)  # TODO: Take other LoQS models
"""Types of pyGSTi models this backend can handle"""


class PyGSTiNoiseModel(BaseNoiseModel):
    """Model backend for handling ``pygsti.model.OpModel`` objects."""

    def __init__(self, model: PyGSTiModelCastableTypes) -> None:
        """Initialize a PyGSTiModelBackend.

        Parameters
        ----------
        model:
            A pyGSTi model to use when looking up operations
        """
        self.model = model
        if isinstance(self.model, ExplicitOpModel):
            self.gate_dict = self.model.operations
            self.inst_dict = self.model.instructions  # type: ignore
        elif isinstance(self.model, ImplicitOpModel):
            self.gate_dict = self.model.operation_blks["gates"]
            self.inst_dict = self.model.instruction_blks["gates"]  # type: ignore
        else:
            raise TypeError("Can only take Explicit or Implicit OpModels")

        # TODO: Crosstalk specification?

    @property
    def input_circuit_types(self) -> Sequence[type[BasePhysicalCircuit]]:
        return [PyGSTiPhysicalCircuit]

    @property
    def output_gate_reps(self) -> Sequence[GateRep]:
        return [GateRep.UNITARY, GateRep.PTM, GateRep.QSIM_SUPEROPERATOR]

    @property
    def output_instrument_reps(self) -> Sequence[InstrumentRep]:
        return [InstrumentRep.ZBASISPROJECTION]  # TODO: Can do more

    def get_operator_reps(
        self,
        circuit: BasePhysicalCircuit,
        gaterep: GateRep,
        instrep: InstrumentRep,
    ) -> Sequence:
        # Get bare circuit
        circuit = PyGSTiPhysicalCircuit.cast(circuit)
        pygsti_circuit = circuit.circuit

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

        q2basis = [
            functools.reduce(np.kron, *x)
            for x in itertools.product(q1basis, repeat=2)
        ]
        qsim2 = ExplicitBasis(
            q2basis,
            ["myEl%d" % i for i in range(4**2)],
            name="qsim_2q",
            longname="QuantumSim_2qubit",
        )

        # Iterate through circuit and pull out representations
        # TODO: What to do about instruments
        reps = []
        for layer in pygsti_circuit:
            for comp in layer.components:  # type: ignore
                qubits = comp.qubits
                if comp.name.startswith("G"):
                    op = self.gate_dict[comp]
                    reptype = gaterep
                    if gaterep == GateRep.UNITARY:
                        try:
                            rep = op.to_dense(on_space="Hilbert")
                        except ValueError as e:
                            raise ValueError(
                                "Failed to cast gate as a unitary. Consider ",
                                "using process matrices instead.",
                            ) from e
                    elif gaterep == GateRep.PTM:
                        rep = op.to_dense(on_space="HilbertSchmidt")
                    elif gaterep == GateRep.QSIM_SUPEROPERATOR:
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
                    else:
                        raise NotImplementedError(
                            f"Cannot create gate rep for {gaterep}"
                        )
                elif comp.name.startswith("I"):
                    # inst = self.inst_dict[comp]
                    assert comp.name == "Iz", "Can only handle Z-basis MCMs"

                    reptype = instrep

                    # TODO: Do more for not just projections
                    if instrep == InstrumentRep.ZBASISPROJECTION:
                        rep = None
                    else:
                        raise NotImplementedError(
                            f"Cannot create instrument rep for {instrep}"
                        )

                reps.append((rep, qubits, reptype))
        return reps
