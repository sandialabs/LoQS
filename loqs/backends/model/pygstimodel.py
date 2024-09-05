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
    from pygsti.baseobjs import TensorProdBasis, ExplicitBasis
    from pygsti.modelmembers.operations import EmbeddedOp
    from pygsti.models import ExplicitOpModel, ImplicitOpModel
    from pygsti.tools import basistools as bt
except ImportError as e:
    raise ImportError("Failed import, cannot use pyGSTi as backend") from e


def compute_qsim_bases(num_qubits: int):
    """TODO"""
    # Prep QuantumSim bases
    sig0q = np.array([[1.0, 0], [0, 0]], dtype="complex")
    sigXq = np.array([[0, 1], [1, 0]], dtype="complex") / np.sqrt(2)
    sigYq = np.array([[0, -1], [1, 0]], dtype="complex") * 1.0j / np.sqrt(2.0)
    sig1q = np.array([[0, 0], [0, 1]], dtype="complex")

    qbasis = itertools.product([sig0q, sigXq, sigYq, sig1q], repeat=num_qubits)
    qbasis = [functools.reduce(np.kron, x) for x in qbasis]

    return ExplicitBasis(
        qbasis,
        ["myEl%d" % i for i in range(4**num_qubits)],
        name=f"qsim_{num_qubits}q",
        longname=f"QuantumSim_{num_qubits}qubit",
    )


PYGSTI_QSIM_BASES = {nq: compute_qsim_bases(nq) for nq in [1, 2]}
"""Precomputed 1- and 2-qubit basis for QSim PTMs"""


# Type aliases for static type checking
PyGSTiModelCastableTypes: TypeAlias = (
    ExplicitOpModel | ImplicitOpModel | BaseNoiseModel
)
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
        from loqs.backends.model import DictNoiseModel

        self.use_embedded_op = False
        if isinstance(model, ExplicitOpModel):
            self.model = model
            self.gate_dict = self.model.operations
            self.inst_dict = self.model.instruments
        elif isinstance(model, ImplicitOpModel):
            self.model = model
            self.gate_dict = self.model.operation_blks.get("layers", {})
            self.inst_dict = self.model.instrument_blks.get("layers", {})
            self.use_embedded_op = True
        elif isinstance(model, PyGSTiNoiseModel):
            self.model = model.model
            self.gate_dict = model.gate_dict
            self.inst_dict = model.inst_dict
            self.use_embedded_op = model.use_embedded_op
        elif isinstance(model, DictNoiseModel):
            raise NotImplementedError("TODO: Build explicit op model")
        else:
            raise TypeError(f"Cannot cast {type(model)} to PyGSTiNoiseModel")

        # TODO: Crosstalk specification?

    @property
    def gate_keys(self) -> list:
        return list(self.gate_dict.keys())

    @property
    def instrument_keys(self) -> list:
        return list(self.inst_dict.keys())

    @property
    def output_gate_reps(self) -> list[GateRep]:
        return [GateRep.UNITARY, GateRep.PTM, GateRep.QSIM_SUPEROPERATOR]

    @property
    def output_instrument_reps(self) -> list[InstrumentRep]:
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
                                rep, basis, PYGSTI_QSIM_BASES[comp.num_qubits]
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
