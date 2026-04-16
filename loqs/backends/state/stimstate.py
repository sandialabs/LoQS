#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`.STIMQuantumState` definition.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import h5py
import numpy as np
from typing import ClassVar, TypeAlias, TypeVar, TYPE_CHECKING, Any

from loqs.backends import GateRep, is_backend_available
from loqs.backends.model.basemodel import InstrumentRep
from loqs.backends.reps import RepTuple
from loqs.backends.state import BaseQuantumState, OutcomeDict
from loqs.internal.encoder.hdf5encoder import HDF5Encoder
from loqs.internal.encoder.jsonencoder import JSONEncoder
from loqs.internal.serializable import Serializable
from loqs.types import Float

# Conditional imports for STIM
if TYPE_CHECKING:
    # Type checking imports - these won't be executed at runtime
    from stim import Circuit as _Circuit
    from stim import Tableau as _Tableau
    from stim import TableauSimulator as _TableauSimulator
else:
    # Runtime imports - these will be attempted only when needed
    try:
        from stim import Circuit as _Circuit
        from stim import Tableau as _Tableau
        from stim import TableauSimulator as _TableauSimulator
    except ImportError:
        _Circuit = Any  # type: ignore
        _Tableau = Any  # type: ignore
        _TableauSimulator = Any  # type: ignore


T = TypeVar("T", bound="STIMQuantumState")

# Type aliases for static type checking
STIMStateCastableTypes: TypeAlias = (
    "STIMQuantumState | _TableauSimulator | _Tableau | int | Sequence[int]"
)
"""Types that this backend can cast to an underlying state object."""

QubitTypes: TypeAlias = str | int
"""Types this backend can use for qubit labels.

Note that this is technically not a true restriction of SparseDM,
but keeping it simple as other types are unlikely.
"""


class STIMQuantumState(BaseQuantumState):
    """Base class for an object that holds a STIM Tableau."""

    name: ClassVar[str] = "STIM Tableau"

    SERIALIZE_ATTRS = ["qubit_labels", "seed", "_stim_state_vector"]

    _state: _TableauSimulator
    """Underlying state object."""

    qubit_labels: list[QubitTypes]
    """Qubit labels.

    These are used to map local ints
    to global ints in
    :attr:`.GateRep.STIM_CIRCUIT_STR` reps.
    """

    @property
    def state(self) -> _TableauSimulator:
        """Get the underlying STIM TableauSimulator state object.

        Returns
        -------
        _TableauSimulator
            The internal STIM TableauSimulator object that represents the quantum state.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.

        This property provides access to the raw STIM TableauSimulator state object,
        which contains the actual quantum state representation using STIM's tableau
        simulation approach.
        """
        return self._state

    @property
    def input_reps(self) -> list[GateRep | InstrumentRep]:
        """Get the list of supported operation representation types.

        Returns
        -------
        list[GateRep | InstrumentRep]
            List of operation representation types that this quantum state backend
            can process and apply.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.

        The STIM backend supports STIM circuit strings, probabilistic STIM operations,
        Z-basis projections, Z-basis pre/post operations, and STIM circuit strings
        for instruments as input representations.
        """
        return [
            GateRep.STIM_CIRCUIT_STR,
            GateRep.PROBABILISTIC_STIM_OPERATIONS,
            InstrumentRep.ZBASIS_PROJECTION,
            InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
            InstrumentRep.STIM_CIRCUIT_STR,
        ]

    def __init__(
        self,
        state: STIMStateCastableTypes,
        qubit_labels: Sequence[QubitTypes] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        state:
            A representation of the underlying state. If an integer is passed,
            an all-0 state with that number of qubits is passed

        qubit_labels:
            Optional qubit labels. If not provided, the default range of ints
            is used.
        """
        if not is_backend_available("stim_state"):
            raise ImportError(
                "STIM backend is not available. "
                "Please install stim: pip install loqs[stim]"
            )
        self.qubit_labels = []
        if isinstance(state, STIMQuantumState):
            # If we are setting a seed here, do not copy internal RNG
            # Otherwise, DO copy internal RNG
            self._state = state._state.copy(copy_rng=seed is None, seed=seed)
            self.qubit_labels = state.qubit_labels
        elif isinstance(state, _TableauSimulator):
            self._state = state.copy(copy_rng=seed is None, seed=seed)
        elif isinstance(state, _Tableau):
            self._state = _TableauSimulator(seed=seed)
            self._state.set_inverse_tableau(state)
        elif isinstance(state, int):
            self._state = _TableauSimulator(seed=seed)
            self._state.set_num_qubits(state)
        elif isinstance(state, Sequence) and all(
            [el in [0, 1] for el in state]
        ):
            self._state = _TableauSimulator(seed=seed)
            self._state.set_num_qubits(len(state))
            # Flip specified bits
            for bit, val in enumerate(state):
                if val:
                    self.state.x(bit)
        else:
            raise ValueError(
                f"Cannot initialize TableauSimulator from {state}"
            )

        if qubit_labels is not None:
            self.qubit_labels = list(qubit_labels)
        if (
            len(self.qubit_labels) == 0
        ):  # We haven't set it yet, default to ints
            self.qubit_labels = list(range(self.state.num_qubits))
        assert (
            len(self.qubit_labels) == self.state.num_qubits
        ), "Must specify a qubit label for every qubit"

        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.latest_applied_circuit = _Circuit()
        self.latest_measurement_labels = []

    def __str__(self) -> str:
        s = f"Physical {self.name} state:\n"
        s += f"  STIM state on {self.state.num_qubits} qubits"
        s += f" ([{self.qubit_labels[0]},...,{self.qubit_labels[-1]}])\n"
        return s

    def apply_reps_inplace(
        self, reps: Sequence, reset_latest_circ: bool = True
    ) -> OutcomeDict:
        """Apply operation representations to the quantum state in-place.

        This method applies a sequence of operation representations (RepTuples)
        directly to the current quantum state, modifying it in-place, and returns
        any measurement outcomes.

        Parameters
        ----------
        reps : Sequence
            Sequence of operation representations to apply to the state.

        reset_latest_circ : bool, optional
            Whether to reset the latest applied circuit before processing.
            Default is True.

        Returns
        -------
        OutcomeDict
            Dictionary of measurement outcomes from applying the operations.
            Outcomes can be empty if no measurements were performed.

        Raises
        ------
        NotImplementedError
            If an unknown or unsupported operation representation type is encountered.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.

        This method processes both gate operations (which modify the state directly)
        and instrument operations (which may produce measurement outcomes).
        When reset_latest_circ is True, it clears the previously applied circuit
        before processing new operations.
        """
        outcomes: OutcomeDict = defaultdict(list)

        if reset_latest_circ:
            self.latest_applied_circuit = _Circuit()

        for reptuple in reps:
            reptype = reptuple.reptype
            if isinstance(reptype, GateRep):
                self._apply_gate_rep(reptuple)
            elif isinstance(reptype, InstrumentRep):
                rep_outcomes = self._apply_instrument_rep(reptuple)

                # Merge outcomes with already observed outcomes
                for k, v in rep_outcomes.items():
                    outcomes[k].extend(v)
            else:
                raise NotImplementedError(
                    f"Cannot apply unknown reptype {reptype}"
                )

        return outcomes

    def apply_reps(
        self, reps: Sequence
    ) -> tuple[STIMQuantumState, OutcomeDict]:
        """Apply operation representations to the quantum state.

        This method applies a sequence of operation representations (RepTuples)
        to the quantum state and returns a new state with the operations applied
        along with any measurement outcomes.

        Parameters
        ----------
        reps : Sequence
            Sequence of operation representations to apply to the state.

        Returns
        -------
        tuple[STIMQuantumState, OutcomeDict]
            A tuple containing a new quantum state with the operations applied
            and a dictionary of measurement outcomes.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.

        This method delegates to the parent class implementation for the actual
        operation application logic.
        """
        return super().apply_reps(reps)

    def copy(self) -> STIMQuantumState:
        """Create a deep copy of the quantum state.

        Returns
        -------
        STIMQuantumState
            A new quantum state object that is a deep copy of the current state,
            including the STIM circuit state, qubit labels, random number generator
            state, and all other attributes.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.

        This method creates a complete independent copy of the quantum state,
        ensuring that modifications to the copy do not affect the original state.
        """
        new_state = STIMQuantumState(self.state, self.qubit_labels, self.seed)
        new_state._rng = deepcopy(self._rng)
        return new_state

    def reset_seed(self, new_seed: int | None) -> None:
        """Reset the random seed for the quantum state.

        Parameters
        ----------
        new_seed : int | None
            The new random seed to use. If None, the random number generator
            will use its default seeding behavior.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.

        This method updates both the stored seed value and the internal random
        number generator, ensuring reproducible behavior when the same seed is used.
        Unlike some other implementations, this method explicitly does not copy
        the RNG state, forcing a fresh RNG initialization with the new seed.
        """
        # We explicitly don't want to copy RNG here, force a new RNG seed
        self._state = self._state.copy(copy_rng=False, seed=new_seed)
        self.seed = new_seed
        self._rng = np.random.default_rng(new_seed)

    def _apply_gate_rep(self, reptuple: RepTuple):
        rep = reptuple.rep

        qubits = reptuple.qubits
        assert isinstance(qubits, (tuple, list)) and all(
            [isinstance(q, str) for q in qubits]
        )

        reptype = reptuple.reptype
        assert isinstance(reptype, GateRep)

        if len(qubits) == 0:
            # This is a STIM annotation or comment, pass it on to applied circuit directly
            assert isinstance(rep, str) and reptype == GateRep.STIM_CIRCUIT_STR
            self.latest_applied_circuit += _Circuit(rep)
            return

        if reptype == GateRep.STIM_CIRCUIT_STR:
            assert isinstance(rep, str)

            # We have three types of indices here
            # Local: The placeholder/template qubit used in the rep
            # Global: The qubit label
            # Internal: The qubit label index
            local_to_global = {}
            local_to_internal = {}
            for i, q in enumerate(qubits):
                assert isinstance(q, str)
                negated = q.startswith("!")
                global_label = q.strip("!")
                try:
                    index = self.qubit_labels.index(global_label)
                except ValueError:
                    index = self.qubit_labels.index(int(global_label))
                local_to_internal[str(i)] = f"{'!' if negated else ''}{index}"
                local_to_global[str(i)] = q

            # Split string for easy processing
            internal_lines = []
            global_lines = []
            for line in rep.split("\n"):
                if len(line) == 0:
                    # Skip empty line
                    continue

                entries = line.split()

                internal_entries = [entries[0]]  # instruction is unchanged
                internal_entries += [local_to_internal[e] for e in entries[1:]]

                # Pull measurement labels, if they exist
                command = entries[0].split("(")[0]
                # Subset of measure/reset gates that we want to record
                include_outcomes = [
                    "M",
                    "MX",
                    "MY",
                    "MZ",
                    "MR",
                    "MRX",
                    "MRY",
                    "MRZ",
                ]
                if command in include_outcomes:
                    noneg_internal_entries = [
                        self.qubit_labels[int(me.strip("!"))]
                        for me in internal_entries[1:]
                    ]
                    self.latest_measurement_labels.extend(
                        noneg_internal_entries
                    )

                internal_lines.append(" ".join(internal_entries))

                global_entries = [entries[0]]  # instruction is unchanged
                global_entries += [local_to_global[e] for e in entries[1:]]
                global_lines.append(" ".join(global_entries))

            internal_circuit_str = "\n".join(internal_lines)
            internal_circuit = _Circuit(internal_circuit_str)
            self.state.do_circuit(internal_circuit)

            # Save executed circuit, needed for decoding via pymatching
            # This one we do in global labels since we don't need the smaller internal space,
            # and this is less confusing to read off
            try:
                global_circuit_str = "\n".join(global_lines)
                self.latest_applied_circuit += _Circuit(global_circuit_str)
            except ValueError:
                # STIM failed to convert, our global labels are probably non-int strings
                # Fall back to internal representation
                self.latest_applied_circuit += internal_circuit
        elif reptype == GateRep.PROBABILISTIC_STIM_OPERATIONS:
            assert isinstance(rep, (list, tuple))
            probs = [r[1] for r in rep]
            assert abs(1 - sum(probs)) < 1e-12, "Probabilities should sum to 1"
            assert all(
                [p >= 0 for p in probs]
            ), "Probabilities should be positive"

            # Pick an op to apply
            idx_to_apply = self._rng.choice(list(range(len(rep))), p=probs)

            rep_to_apply = RepTuple(
                rep[idx_to_apply][0], qubits, GateRep.STIM_CIRCUIT_STR
            )

            # Apply chosen op
            self.apply_reps_inplace([rep_to_apply], reset_latest_circ=False)
        else:
            raise NotImplementedError(f"Cannot apply GateRep {reptype}")

    def _apply_instrument_rep(self, reptuple: RepTuple) -> OutcomeDict:
        rep = reptuple.rep

        qubits = reptuple.qubits
        assert isinstance(qubits, (tuple, list)) and len(qubits) > 0

        reptype = reptuple.reptype
        assert isinstance(reptype, InstrumentRep)

        outcomes: OutcomeDict = defaultdict(list)

        if reptype == InstrumentRep.ZBASIS_PROJECTION:
            assert isinstance(rep, (tuple, list)) and len(rep) > 1
            reset = rep[0]
            include_outcomes = rep[1]

            for qbit in qubits:
                cbit = self._measure_and_reset(qbit, reset)
                if include_outcomes:
                    outcomes[qbit].append(cbit)
        elif reptype == InstrumentRep.ZBASIS_PRE_POST_OPERATIONS:
            assert isinstance(rep, (tuple, list)) and len(rep) > 1
            reset = rep[0]
            include_outcomes = rep[1]

            # Check we can apply the reps
            preop = rep[2]
            postop = rep[3]
            assert reset in [None, 0, 1]
            assert preop.reptype in self.input_reps
            assert postop.reptype in self.input_reps
            assert isinstance(preop.reptype, GateRep)
            assert isinstance(postop.reptype, GateRep)
            # TODO: Strict subsets is OK too
            assert preop.qubits == qubits
            assert postop.qubits == qubits

            # Apply the pre-op
            self.apply_reps_inplace([preop])

            # Do perfect measurement
            for qbit in qubits:
                cbit = self._measure_and_reset(qbit, reset)
                if include_outcomes:
                    outcomes[qbit].append(cbit)

            # Apply the post-op
            self.apply_reps_inplace([postop])
        elif reptype == InstrumentRep.STIM_CIRCUIT_STR:
            assert isinstance(rep, str)

            self.latest_measurement_labels = []

            # We'll reuse the gate apply code...
            self.apply_reps_inplace(
                [RepTuple(rep, qubits, GateRep.STIM_CIRCUIT_STR)],
                reset_latest_circ=False,
            )

            # but then post-process to grab the outcomes from the measurement record
            current_mr = self.state.current_measurement_record()
            # print(current_mr[-len(self.latest_measurement_labels):])
            mr_entries = [
                int(mre)
                for mre in current_mr[-len(self.latest_measurement_labels) :]
            ]
            # print(mr_entries)
            # print()

            for qbit, cbit in zip(self.latest_measurement_labels, mr_entries):
                outcomes[qbit].append(cbit)
        else:
            raise NotImplementedError(f"Cannot apply InstrumentRep {reptype}")

        return outcomes

    def _measure_and_reset(
        self, qubit: QubitTypes, reset: int | None = None
    ) -> int:
        qidx = self.qubit_labels.index(qubit)
        cbit = int(self.state.measure(qidx))

        if reset is not None:
            # Reset to 0
            self.state.reset(qidx)

            # Reset to 1, if needed
            if reset == 1:
                self.state.x(qidx)

        return cbit

    def get_encoding_attr(self, attr, ignore_no_serialize_flags=False):
        """Get an attribute for encoding/serialization purposes.

        This method retrieves specific attributes from the quantum state that are
        needed for serialization, including STIM-specific state information.

        Parameters
        ----------
        attr : str
            The name of the attribute to retrieve.

        ignore_no_serialize_flags : bool, optional
            Whether to ignore serialization flags. Default is False.

        Returns
        -------
        object
            The value of the requested attribute, or the result from the parent
            class if the attribute is not found in this class.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.

        This method handles retrieval of STIM-specific attributes such as the
        state vector, which are needed for proper serialization and deserialization
        of the quantum state.
        """
        # Retrieve STIM state vector
        if attr == "_stim_state_vector":
            return self.state.state_vector(endian="little")

        # Otherwise fallback
        return super().get_encoding_attr(attr, ignore_no_serialize_flags)

    @classmethod
    def from_decoded_attrs(cls: type[T], attr_dict: Mapping) -> T:
        """Create a quantum state from decoded attributes.

        This class method reconstructs a quantum state object from a dictionary
        of decoded attributes, typically used during deserialization.

        Parameters
        ----------
        attr_dict : Mapping
            Dictionary containing the decoded attributes needed to reconstruct
            the quantum state.

        Returns
        -------
        T
            A new quantum state object initialized with the provided attributes.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.

        This method is typically used during deserialization to reconstruct a
        quantum state from stored data, including qubit labels and state vector
        information. The state vector is converted to a STIM Tableau representation.
        """
        qubit_labels = attr_dict["qubit_labels"]
        seed = attr_dict["seed"]
        state_vector = attr_dict["_stim_state_vector"]
        assert isinstance(state_vector, np.ndarray)

        obj = cls(
            _Tableau.from_state_vector(state_vector, endian="little"),
            qubit_labels=qubit_labels,
        )
        obj.reset_seed(seed)
        return obj
