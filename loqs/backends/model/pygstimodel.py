#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`.PyGSTiNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Mapping
import functools
import itertools
import numpy as np
from typing import ClassVar, Sequence, TypeAlias, TypeVar, TYPE_CHECKING, Any

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel, TimeDependentBaseNoiseModel
from loqs.backends.reps import GateRep, InstrumentRep, RepEnum, RepTuple
from loqs.internal.serializable import Serializable

# Conditional imports for PyGSTi
_pygsti_available = True
if TYPE_CHECKING:
    # Type checking imports - these won't be executed at runtime
    from pygsti.baseobjs import TensorProdBasis, ExplicitBasis
    from pygsti.baseobjs.label import (
        Label,
        LabelStr,
        LabelTupTupWithTime,
        LabelTupWithTime,
    )
    from pygsti.modelmembers.operations import EmbeddedOp, DenseOperator
    from pygsti.models import Model, ExplicitOpModel, ImplicitOpModel
    from pygsti.tools import basistools as bt, superop_to_unitary
else:
    # Runtime imports - these will be attempted only when needed
    try:
        from pygsti.baseobjs import TensorProdBasis, ExplicitBasis
        from pygsti.baseobjs.label import (
            Label,
            LabelStr,
            LabelTupTupWithTime,
            LabelTupWithTime,
        )
        from pygsti.modelmembers.operations import EmbeddedOp, DenseOperator
        from pygsti.models import Model, ExplicitOpModel, ImplicitOpModel
        from pygsti.tools import basistools as bt, superop_to_unitary
    except ImportError:
        _pygsti_available = False


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
        qbasis,  # type: ignore
        ["myEl%d" % i for i in range(4**num_qubits)],
        name=f"qsim_{num_qubits}q",
        longname=f"QuantumSim_{num_qubits}qubit",
    )


# Module-level code that depends on PyGSTi must be conditional
if _pygsti_available:
    PYGSTI_QSIM_BASES = {nq: compute_qsim_bases(nq) for nq in [1, 2]}
    """Precomputed 1- and 2-qubit basis for QSim PTMs"""

    PyGSTiModelCastableTypes: TypeAlias = (
        ExplicitOpModel | ImplicitOpModel | BaseNoiseModel
    )
else:
    PYGSTI_QSIM_BASES = {}
    PyGSTiModelCastableTypes = Any  # type: ignore

"""Types of pyGSTi models this backend can handle"""


class PyGSTiNoiseModel(TimeDependentBaseNoiseModel):
    """Model backend for handling ``pygsti.model.OpModel`` objects.

    PyGSTi models are inherently time-dependent, so this inherits from
    :class:`TimeDependentBaseNoiseModel` rather than
    :class:`BaseNoiseModel`.
    However, time-dependent features are opt-in and require the user
    to specify ``use_time_dependence=True`` during initialization.
    """

    name: ClassVar[str] = "pyGSTi"

    SERIALIZE_ATTRS = ["model", "qubit_aliases"]

    def __init__(
        self,
        model: PyGSTiModelCastableTypes,
        qubit_aliases: Mapping | Sequence | None = None,
        zbasis_proj_resets: bool = True,
        use_time_dependence: bool = False,
        default_gate_durations: (
            Mapping[Label | str, int | float] | None
        ) = None,
        default_instrument_durations: (
            Mapping[Label | str, int | float] | None
        ) = None,
    ) -> None:
        """Initialize a PyGSTiModelBackend.

        TODO: Choices are made about instrument reset/outcomes.
        Document this.

        Parameters
        ----------
        model:
            A pyGSTi model to use when looking up operations
        """
        from loqs.backends import is_backend_available

        if not is_backend_available("pygsti_model"):
            raise ImportError(
                "PyGSTi model backend is not available. "
                "Please install pygsti: pip install loqs[pygsti]"
            )
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
        self.model_qubit_aliases = {
            v: k for k, v in self.qubit_aliases.items()
        }

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

        self.use_time_dependence = use_time_dependence
        self.default_gate_durations = default_gate_durations
        self.default_instrument_durations = default_instrument_durations

        # TODO: Crosstalk specification?

        self._gate_rep_cache = {}
        self._inst_rep_cache = {}

    @property
    def gate_keys(self) -> list:
        """Get the list of gate keys this model can take in circuits.

        Returns
        -------
        list
            List of gate keys, where each key is a tuple of (gate_name, qubit_labels)
            or (gate_name,) for gates without specific qubit labels.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        keys = []
        for key in self.gate_dict.keys():
            name = key.name

            if isinstance(key, LabelStr):
                # LabelStr does not have any qubits with it
                keys.append((name,))
                continue

            aliased_qubits = [self.qubit_aliases[q] for q in key.qubits]
            keys.append((name, aliased_qubits))
        return keys

    @property
    def instrument_keys(self) -> list:
        """Get the list of instrument keys this model can take in circuits.

        Returns
        -------
        list
            List of instrument keys, where each key is a tuple of (instrument_name, qubit_labels).

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        keys = []
        for key in self.inst_dict.keys():
            name = key.name
            aliased_qubits = [self.qubit_aliases[q] for q in key.qubits]
            keys.append((name, aliased_qubits))
        return keys

    @property
    def output_gate_reps(self) -> list[GateRep]:
        """Get the list of gate representations this model can output.

        Returns
        -------
        list[GateRep]
            List of gate representations that this model can output.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        return [
            GateRep.UNITARY,
            GateRep.KRAUS_OPERATORS,
            GateRep.PTM,
            GateRep.QSIM_SUPEROPERATOR,
        ]

    # TODO: This is not quite right. It's probably one or the other,
    # depending on whether instruments are defined or not
    @property
    def output_instrument_reps(self) -> list[InstrumentRep]:
        """Get the list of instrument representations this model can output.

        Returns
        -------
        list[InstrumentRep]
            List of instrument representations that this model can output.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        return [
            InstrumentRep.ZBASIS_PROJECTION,
            InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT,
        ]

    def get_gate_duration(self, gate_label) -> int | float:
        """Get the duration of a gate operation.

        This method retrieves the duration of a gate operation from either the gate label
        itself (if time-dependent) or from the default gate durations mapping.

        Parameters
        ----------
        gate_label : Label | str
            The gate label to get the duration for.

        Returns
        -------
        int | float
            Duration of the gate operation.

        Raises
        ------
        ValueError
            If time dependence is not enabled and no default gate durations are provided,
            or if a LayerTupTupWithTime is unexpectedly provided.

        KeyError
            If the gate label is not found in the default gate durations.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        if not self.use_time_dependence:
            return 0

        if isinstance(gate_label, LabelTupWithTime):
            return gate_label.time  # type: ignore
        elif isinstance(gate_label, LabelTupTupWithTime):
            # This represents a layer. Technically this should be fine to extract the time,
            # but we almost certainly didn't mean to pass a whole layer here
            raise ValueError(
                "Unexpectedly got a LayerTupTupWithTime (i.e., layer) in get_gate_duration"
            )

        # If we are here, we got a normal (probably LabelTup/LabelTupWithArgs)
        # and need to be looking up in the defaults
        if self.default_gate_durations is None:
            raise ValueError(
                "Time not specified in pyGSTi circuit label and no default gate durations given!"
            )

        duration = self.default_gate_durations.get(gate_label, None)
        if duration is None:
            # Failed specific lookup, let's try by name only
            duration = self.default_gate_durations.get(gate_label.name, None)  # type: ignore

        if duration is None:
            raise KeyError(
                f"{gate_label} not available by label or name in default gate durations!"
            )

        return duration

    def get_instrument_duration(self, inst_label) -> int | float:
        """Get the duration of an instrument operation.

        This method retrieves the duration of an instrument operation from either the
        instrument label itself (if time-dependent) or from the default instrument
        durations mapping.

        Parameters
        ----------
        inst_label : Label | str
            The instrument label to get the duration for.

        Returns
        -------
        int | float
            Duration of the instrument operation.

        Raises
        ------
        ValueError
            If time dependence is not enabled and no default instrument durations are provided,
            or if a LayerTupTupWithTime is unexpectedly provided.

        KeyError
            If the instrument label is not found in the default instrument durations.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        if not self.use_time_dependence:
            return 0

        if isinstance(inst_label, LabelTupWithTime):
            return inst_label.time  # type: ignore
        elif isinstance(inst_label, LabelTupTupWithTime):
            # This represents a layer. Technically this should be fine to extract the time,
            # but we almost certainly didn't mean to pass a whole layer here
            raise ValueError(
                "Unexpectedly got a LayerTupTupWithTime (i.e., layer) in get_instrument_duration"
            )

        # If we are here, we got a normal (probably LabelTup/LabelTupWithArgs)
        # and need to be looking up in the defaults
        if self.default_instrument_durations is None:
            raise ValueError(
                "Time not specified in pyGSTi circuit label and no default instrument durations given!"
            )

        duration = self.default_instrument_durations.get(inst_label, None)
        if duration is None:
            # Failed specific lookup, let's try by name only
            duration = self.default_instrument_durations.get(inst_label.name, None)  # type: ignore

        if duration is None:
            raise KeyError(
                f"{inst_label} not available by label or name in default instrument durations!"
            )

        return duration

    def get_reps(
        self,
        circuit: BasePhysicalCircuit,
        gatereps: Sequence[GateRep],
        instreps: Sequence[InstrumentRep],
    ) -> list[RepTuple]:
        """Get list of operator representations that can be applied.

        This method processes a circuit and returns a list of operation representations
        (RepTuples) that can be applied to a quantum state.

        Parameters
        ----------
        circuit : BasePhysicalCircuit
            Physical circuit to get the representations for.

        gatereps : Sequence[GateRep]
            Output representations for gate operations.

        instreps : Sequence[InstrumentRep]
            Output representations for instrument operations.

        Returns
        -------
        list[RepTuple]
            List of operation representations for the circuit.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        # Get bare circuit
        from loqs.backends import PyGSTiPhysicalCircuit

        circuit = PyGSTiPhysicalCircuit.cast(circuit)
        pygsti_circuit = circuit.circuit

        # Iterate through circuit and pull out representations
        reps = []
        for layer in pygsti_circuit.layertup:  # type: ignore
            for comp in layer.components:  # type: ignore
                name = comp.name
                aliased_qubits = comp.qubits  # The circuit is already aliased
                qubits = [self.model_qubit_aliases[q] for q in aliased_qubits]
                if name.startswith("G"):
                    rep, reptype = self._get_gate_rep(
                        comp.name, qubits, gatereps
                    )
                    duration = self.get_gate_duration(comp)
                elif comp.name.startswith("I"):
                    rep, reptype = self._get_instrument_rep(
                        comp.name, qubits, instreps
                    )
                    duration = self.get_instrument_duration(comp)
                else:
                    raise NotImplementedError("Can only handle G/I prefixes")

                # We need to save with original (aliased) qubits
                reps.append(RepTuple(rep, aliased_qubits, reptype))

                # If using time-dependence, update layer time
                if self.use_time_dependence:
                    self.add_gate_duration_to_layer(duration)
            # If using time-dependence, move simulation time forward
            if self.use_time_dependence:
                self.add_layer_duration_to_current_time()
        return reps

    def _get_gate_rep(self, name, qubits, gatereps):  # noqa: C901
        op_key = (name,) + tuple(qubits)
        # Check cache
        for gaterep in gatereps:
            if (op_key, gaterep) in self._gate_rep_cache:
                return (self._gate_rep_cache[op_key, gaterep], gaterep)

        # Look up using unaliased qubits
        op = self.gate_dict[op_key]
        basis = self.model.basis

        # if using time-dependence, update operator rep and clear cache
        if self.use_time_dependence:
            op.set_time(self.current_time)

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

        rep = None

        def _get_rep(gaterep):
            if gaterep == GateRep.UNITARY:
                try:
                    rep = op.to_dense(on_space="Hilbert")
                except ValueError:
                    # Failed, could be because we have the densitymx evotype
                    # Try to manually convert down to unitary
                    try:
                        rep = superop_to_unitary(op.to_dense())

                    except (ValueError, IndexError) as e:
                        raise ValueError(
                            "Failed to cast gate as a unitary. Consider "
                            + "using process matrices instead. PyGSTi error: "
                            + str(e),
                        ) from e
            elif gaterep == GateRep.KRAUS_OPERATORS:
                try:
                    # We'll upcast to DenseOperator to get access to the kraus property
                    # TODO for pygsti: This should probably be moved to optools instead of DenseOperator
                    dense_op = DenseOperator(
                        op.to_dense(), basis, self.model.evotype
                    )
                    Ks = dense_op.kraus_operators

                    rep = []
                    # Pre-compute probabilities (if unitary)
                    for K in Ks:
                        KKdag = K @ K.conjugate().T
                        prob = KKdag[0, 0]
                        if np.all(
                            np.isclose(KKdag / prob, np.eye(KKdag.shape[0]))
                        ):
                            # This was the identity when we pulled the probability out
                            assert np.isreal(prob)
                            rep.append((K, abs(prob.real)))
                        else:
                            # Not the identity, so store None (signal states to compute on the fly)
                            rep.append((K, None))
                except (ValueError, AttributeError, ZeroDivisionError) as e:
                    raise ValueError(
                        "Failed to cast gate as Kraus operators. Consider "
                        + "using process matrices instead."
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
                raise ValueError(f"Cannot create gate rep for {gaterep}")

            return rep

        repidx = 0
        errors = []
        while repidx < len(gatereps):
            try:
                rep = _get_rep(gatereps[repidx])
                break
            except ValueError as e:
                # Try next one
                repidx += 1

                errors.append(e)

        if repidx == len(gatereps):
            raise ValueError(
                f"Failed to create gate rep for any of {gatereps}, with errors:"
                + "\n".join([str(e) for e in errors])
            )

        if not self.use_time_dependence:
            self._gate_rep_cache[op_key, gatereps[repidx]] = rep

        return rep, gatereps[repidx]

    def _get_instrument_rep(self, name, qubits, instreps):
        inst_key = (name,) + tuple(qubits)
        # Check cache
        for instrep in instreps:
            if (inst_key, instrep) in self._inst_rep_cache:
                return (self._inst_rep_cache[inst_key, instrep], instrep)

        rep = None

        def _get_rep(instrep):
            if instrep == InstrumentRep.ZBASIS_PROJECTION:
                rep: None | int | dict = 0 if self.zbasis_proj_resets else None
            elif instrep == InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT:
                # TODO: What to do with key error?
                # Look up using unaliased qubits
                op = self.inst_dict[inst_key]

                # if using time-dependence, update operator rep
                if self.use_time_dependence:
                    op.set_time(self.current_time)

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
                raise ValueError(f"Cannot create instrument rep for {instrep}")

            return rep

        repidx = 0
        while repidx < len(instreps):
            try:
                rep = _get_rep(instreps[repidx])
                break
            except ValueError:
                # Try next one
                repidx += 1

        if repidx == len(instreps):
            raise ValueError(
                f"Failed to create instrument rep for any of {instreps}"
            )

        if not self.use_time_dependence:
            self._inst_rep_cache[inst_key, instreps[repidx]] = (rep, True)

        return (rep, True), instreps[repidx]

    def get_encoding_attr(self, attr, ignore_no_serialize_flags=False):
        """Get the encoding attribute for serialization.

        This method returns the serialized representation of the model attribute
        for encoding purposes.

        Parameters
        ----------
        attr : str
            Name of the attribute to encode.

        ignore_no_serialize_flags : bool, optional
            Whether to ignore no-serialize flags during encoding. Default is False.

        Returns
        -------
        object
            Serialized representation of the requested attribute.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        if attr == "model":
            return self.model.to_nice_serialization()
        return super().get_encoding_attr(attr, ignore_no_serialize_flags)

    @classmethod
    def from_decoded_attrs(cls: type[T], attr_dict: Mapping) -> T:
        """Create a PyGSTiNoiseModel from decoded attributes.

        This class method reconstructs a PyGSTiNoiseModel instance from a dictionary
        of decoded attributes, typically used during deserialization.

        Parameters
        ----------
        attr_dict : Mapping
            Dictionary containing the decoded attributes. Expected to have keys:
            'model' and 'qubit_aliases'.

        Returns
        -------
        T
            A new PyGSTiNoiseModel instance initialized with the provided attributes.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        model = Model.from_nice_serialization(attr_dict["model"])
        qubit_aliases = attr_dict["qubit_aliases"]
        return cls(model, qubit_aliases)
