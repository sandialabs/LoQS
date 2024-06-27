""":class:`PyGSTiNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Sequence
import functools
import itertools
from typing import ClassVar, TypeAlias
import numpy as np

from loqs.backends.circuit import PyGSTiPhysicalCircuit
from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel, GateRep, InstrumentRep


try:
    from pygsti.baseobjs import ExplicitBasis, TensorProdBasis
    from pygsti.modelmembers.operations import EmbeddedOp
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

    name: ClassVar[str] = "pyGSTi"

    def __init__(self, model: PyGSTiModelCastableTypes) -> None:
        """Initialize a PyGSTiModelBackend.

        Parameters
        ----------
        model:
            A pyGSTi model to use when looking up operations
        """
        self.model = model
        self.use_embedded_op = False
        if isinstance(self.model, ExplicitOpModel):
            self.gate_dict = self.model.operations
            self.inst_dict = self.model.instruments
        elif isinstance(self.model, ImplicitOpModel):
            self.gate_dict = self.model.operation_blks.get("layers", {})
            self.inst_dict = self.model.instrument_blks.get("layers", {})
            self.use_embedded_op = True
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

    def get_reps(
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

        qsim_bases = {}
        for n_qubits in [1, 2]:
            qbasis = itertools.product(
                [sig0q, sigXq, sigYq, sig1q], repeat=n_qubits
            )
            qbasis = [functools.reduce(np.kron, x) for x in qbasis]
            qsim_bases[n_qubits] = ExplicitBasis(
                qbasis,
                ["myEl%d" % i for i in range(4**n_qubits)],
                name=f"qsim_{n_qubits}q",
                longname=f"QuantumSim_{n_qubits}qubit",
            )

        # Iterate through circuit and pull out representations
        # TODO: What to do about instruments
        reps = []
        for layer in pygsti_circuit.layertup:  # type: ignore
            for comp in layer.components:  # type: ignore
                qubits = comp.qubits
                if comp.name.startswith("G"):
                    op = self.gate_dict[comp]
                    basis = self.model.basis

                    if self.use_embedded_op and isinstance(op, EmbeddedOp):
                        # Pull out the relevant part of the tensor prod basis
                        assert isinstance(self.model.basis, TensorProdBasis)
                        assert op.target_labels is not None
                        target_indices = [
                            op.state_space.qubit_labels.index(q)
                            for q in op.target_labels
                        ]
                        target_bases = [
                            self.model.basis.component_bases[i]
                            for i in target_indices
                        ]
                        basis = TensorProdBasis(target_bases)

                        op = op.embedded_op
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

                        if comp.num_qubits in [1, 2]:
                            rep = bt.change_basis(
                                rep, basis, qsim_bases[comp.num_qubits]
                            )
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
                else:
                    raise NotImplementedError("Can only handle G/I prefixes")

                reps.append((rep, qubits, reptype))
        return reps
