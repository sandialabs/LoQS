""":class:`.PyGSTiNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Mapping
import functools
import itertools
import numpy as np
from typing import ClassVar, Sequence, TypeAlias, TypeVar

from loqs.backends.circuit import BasePhysicalCircuit, PyGSTiPhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.reps import GateRep, InstrumentRep, RepEnum, RepTuple

try:
    from pygsti.baseobjs import TensorProdBasis, ExplicitBasis
    from pygsti.modelmembers.operations import EmbeddedOp
    from pygsti.models import Model, ExplicitOpModel, ImplicitOpModel
    from pygsti.tools import basistools as bt
except ImportError as e:
    raise ImportError("Failed import, cannot use pyGSTi as backend") from e


T = TypeVar("T", bound="PyGSTiNoiseModel")


def compute_qsim_bases(num_qubits: int):
    """TODO"""
    # Prep QuantumSim bases
    sig0q = np.array([[1.0, 0], [0, 0]], dtype="complex")
    sigXq = np.array([[0, 1], [1, 0]], dtype="complex") / np.sqrt(2)
    sigYq = np.array([[0, -1], [1, 0]], dtype="complex") * 1.0j / np.sqrt(2.0)
    sig1q = np.array([[0, 0], [0, 1]], dtype="complex")

    qbasis_prod = itertools.product(
        [sig0q, sigXq, sigYq, sig1q], repeat=num_qubits
    )
    qbasis = [functools.reduce(np.kron, x) for x in qbasis_prod]

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

    def __init__(
        self,
        model: PyGSTiModelCastableTypes,
        qubit_aliases: Mapping | Sequence | None = None,
        zbasis_proj_resets: bool = True,
    ) -> None:
        """Initialize a PyGSTiModelBackend.

        TODO: Choices are made about instrument reset/outcomes.
        Document this.

        Parameters
        ----------
        model:
            A pyGSTi model to use when looking up operations
        """
        from loqs.backends.model import DictNoiseModel

        # Currently there is a pyGSTi bug deserializing models that have
        # non-int or strs that do not start with "Q"
        # We will enforce this upon the user here for the model itself,
        # but allow qubit label aliasing
        model_qubits = model.state_space.qubit_labels  # type: ignore
        assert all(
            [
                isinstance(q, int)
                or (isinstance(q, str) and q.startswith("Q"))
                for q in model_qubits
            ]
        ), (
            "Model must use int or str starting with Q labels for qubits. ",
            "For qubit labels outside of these restrictions, use `qubit_aliases` "
            "to map from model qubits to your desired qubit labels.",
        )

        if qubit_aliases is None:
            self.qubit_aliases = {k: k for k in model_qubits}
        elif isinstance(qubit_aliases, Mapping):
            assert all([q in qubit_aliases for q in model_qubits])
            self.qubit_aliases = dict(qubit_aliases)
        elif isinstance(qubit_aliases, Sequence):
            assert len(qubit_aliases) == len(model_qubits)
            self.qubit_aliases = {
                k: v for k, v in zip(model_qubits, qubit_aliases)
            }
        else:
            raise TypeError("Invalid type for qubit aliases")
        self._rev_qubit_aliases = {v: k for k, v in self.qubit_aliases.items()}

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

        self.zbasis_proj_resets = zbasis_proj_resets

        # TODO: Crosstalk specification?

    def __hash__(self) -> int:
        return hash((hash(self.model), self.hash(self.qubit_aliases)))

    @property
    def gate_keys(self) -> list:
        keys = []
        for key in self.gate_dict.keys():
            name = key.name
            aliased_qubits = [self.qubit_aliases[q] for q in key.qubits]
            keys.append((name, aliased_qubits))
        return keys

    @property
    def instrument_keys(self) -> list:
        keys = []
        for key in self.inst_dict.keys():
            name = key.name
            aliased_qubits = [self.qubit_aliases[q] for q in key.qubits]
            keys.append((name, aliased_qubits))
        return keys

    @property
    def output_gate_reps(self) -> list[GateRep]:
        return [GateRep.UNITARY, GateRep.PTM, GateRep.QSIM_SUPEROPERATOR]

    # TODO: This is not quite right. It's probably one or the other,
    # depending on whether instruments are defined or not
    @property
    def output_instrument_reps(self) -> list[InstrumentRep]:
        return [
            InstrumentRep.ZBASIS_PROJECTION,
            InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT,
        ]

    def get_reps(
        self,
        circuit: BasePhysicalCircuit,
        gatereps: Sequence[GateRep],
        instreps: Sequence[InstrumentRep],
    ) -> list[RepTuple]:
        # Get bare circuit
        circuit = PyGSTiPhysicalCircuit.cast(circuit)
        pygsti_circuit = circuit.circuit

        # Iterate through circuit and pull out representations
        reps = []
        for layer in pygsti_circuit.layertup:  # type: ignore
            for comp in layer.components:  # type: ignore
                name = comp.name
                aliased_qubits = comp.qubits  # The circuit is already aliased
                qubits = [self._rev_qubit_aliases[q] for q in aliased_qubits]
                if name.startswith("G"):
                    # TODO: Currently this is only using first rep. Fix?
                    rep = self._get_gate_rep(comp.name, qubits, gatereps[0])
                    reptype: RepEnum = gatereps[0]
                elif comp.name.startswith("I"):
                    # TODO: Currently this is only using first rep. Fix?
                    rep = self._get_instrument_rep(
                        comp.name, qubits, instreps[0]
                    )
                    reptype = instreps[0]
                else:
                    raise NotImplementedError("Can only handle G/I prefixes")

                # We need to save with original (aliased) qubits
                reps.append(RepTuple(rep, aliased_qubits, reptype))
        return reps

    def _get_gate_rep(self, name, qubits, gaterep) -> object:
        op = self.gate_dict[(name,)+qubits]  # Look up using unaliased qubits
        basis = self.model.basis

        if self.use_embedded_op and isinstance(op, EmbeddedOp):
            # Pull out the relevant part of the tensor prod basis
            assert isinstance(self.model.basis, TensorProdBasis)
            assert op.target_labels is not None
            target_indices = [
                op.state_space.qubit_labels.index(q) for q in op.target_labels
            ]
            target_bases = [
                self.model.basis.component_bases[i] for i in target_indices
            ]
            basis = TensorProdBasis(target_bases)

            op = op.embedded_op

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

            if len(qubits) in [1, 2]:
                rep = bt.change_basis(
                    rep, basis, PYGSTI_QSIM_BASES[len(qubits)]
                )
            else:
                raise ValueError(
                    "Cannot change more than a 2 qubit operation into",
                    " the QuantumSim basis",
                )
        else:
            raise NotImplementedError(f"Cannot create gate rep for {gaterep}")

        return rep

    def _get_instrument_rep(self, name, qubits, instrep) -> object:
        if instrep == InstrumentRep.ZBASIS_PROJECTION:
            rep: None | int | dict = 0 if self.zbasis_proj_resets else None
        elif instrep == InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT:
            # TODO: What to do with key error?
            op = self.inst_dict[
                (name,) + qubits
            ]  # Look up using unaliased qubits

            rep = {}
            for k, v in op.items():
                if isinstance(k, str):
                    try:
                        if len(k) > 1:
                            label = tuple([int(c) for c in k])
                        else:
                            label = (int(k),)
                    except ValueError as e:
                        raise ValueError(
                            "Failed to cast instrument keys to outcome labels"
                        ) from e
                else:
                    label = k

                assert isinstance(label, tuple)
                assert all([c in [0, 1] for c in label])

                rep[label] = v
        else:
            raise NotImplementedError(
                f"Cannot create instrument rep for {instrep}"
            )

        return (rep, True)

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        model = Model.from_nice_serialization(state["model"])
        qubit_aliases = state["qubit_aliases"]
        return cls(model, qubit_aliases)

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "model": self.model.to_nice_serialization(),
                "qubit_aliases": self.qubit_aliases,
            }
        )
        return state
