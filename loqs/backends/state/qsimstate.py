#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################



from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence, Collection
from copy import deepcopy
import numpy as np
from typing import ClassVar, TypeAlias, TypeVar


from loqs.backends import GateRep
from loqs.backends.model.basemodel import InstrumentRep
from loqs.backends.reps import RepTuple
from loqs.backends.state import BaseQuantumState, OutcomeDict
from loqs.types import Int

try:
    from quantumsim.sparsedm import SparseDM as _SparseDM
    from quantumsim.dm_np import DensityNP as _DensityNP
except ImportError as e:
    raise ImportError("Failed import, cannot use QuantumSim as backend") from e


T = TypeVar("T", bound="QSimQuantumState")

# Type aliases for static type checking
QSimStateCastableTypes: TypeAlias = "QSimQuantumState | int | _SparseDM"
"""Types that this backend can cast to an underlying state object."""

QubitTypes: TypeAlias = str | int
"""Types this backend can use for qubit labels.

Note that this is technically not a true restriction of SparseDM,
but keeping it simple as other types are unlikely.
"""


class QSimQuantumState(BaseQuantumState):
    """Base class for an object that holds a QuantumSim SparseDM state."""

    name: ClassVar[str] = "QuantumSim"

    _SERIALIZE_ATTRS = [
        "qubit_labels",
        "_qsim_classical",
        "_qsim_idx_in_full_dm",
        "_qsim_dm_class",
        "_qsim_dm_no_qubits",
        "_qsim_dm_data",
        "_qsim_max_bits_in_full_dm",
        "_qsim_classical_probability",
        "_qsim_single_ptms_to_do",
        "_qsim_maj_vot_mask",
        "_qsim_maj_vot_array",
    ]

    _state: _SparseDM
    """Underlying state object."""

    @property
    def state(self) -> _SparseDM:
        """Get the underlying QuantumSim state object.

        Returns
        -------
        _SparseDM
            The internal QuantumSim state object that represents the quantum state.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.

        This property provides access to the raw QuantumSim state object, which
        contains the actual quantum state representation and related metadata.
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

        The QSim backend supports QSIM superoperators, Z-basis projections,
        Z-basis pre/post operations, and Z-basis outcome operation dictionaries
        as input representations.
        """
        return [
            GateRep.QSIM_SUPEROPERATOR,
            InstrumentRep.ZBASIS_PROJECTION,
            InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
            InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT,
        ]

    def __init__(
        self,
        state: QSimStateCastableTypes,
        qubit_labels: Collection[QubitTypes] | None = None,
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
        if isinstance(state, QSimQuantumState):
            self._state = state._state.copy()
        elif isinstance(state, _SparseDM):
            self._state = state
        elif isinstance(state, Int):
            self._state = _SparseDM(int(state))
        else:
            raise ValueError(f"Cannot initialize SparseDM from {state}")

        # Optionally override qubit labels
        if qubit_labels is not None:
            assert len(qubit_labels) == self.state.no_qubits, (
                f"Wrong number of qubit labels ({len(qubit_labels)}) ",
                f"provided (expected {self.state.no_qubits})",
            )

            # The names field is not well typed in SparseDM
            name_map = {k: v for k, v in zip(self.state.names, qubit_labels)}  # type: ignore

            self.state.names = [name_map[n] for n in self.state.names]  # type: ignore
            self.state.classical = {
                name_map[k]: v for k, v in self.state.classical.items()
            }

        self.reset_seed(seed)

    def __str__(self) -> str:
        s = f"Physical {self.name} state:\n"
        s += f"  SparseDM state on {self.state.no_qubits} qubits"
        s += f" ([{self.state.names[0]},...,{self.state.names[-1]}])\n"  # type: ignore
        return s

    def apply_reps_inplace(self, reps: Sequence) -> OutcomeDict:
        """Apply operation representations to the quantum state in-place.

        This method applies a sequence of operation representations (RepTuples)
        directly to the current quantum state, modifying it in-place, and returns
        any measurement outcomes.

        Parameters
        ----------
        reps : Sequence
            Sequence of operation representations to apply to the state.

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
        """
        outcomes: OutcomeDict = defaultdict(list)

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
    ) -> tuple[QSimQuantumState, OutcomeDict]:
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
        tuple[QSimQuantumState, OutcomeDict]
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

    def copy(self) -> QSimQuantumState:
        """Create a deep copy of the quantum state.

        Returns
        -------
        QSimQuantumState
            A new quantum state object that is a deep copy of the current state,
            including the state vector, random number generator state, and all
            other attributes.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.

        This method creates a complete independent copy of the quantum state,
        ensuring that modifications to the copy do not affect the original state.
        """
        new_state = QSimQuantumState(deepcopy(self.state), seed=self.seed)
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
        """
        self.seed = new_seed
        self._rng = np.random.default_rng(new_seed)

    def _apply_gate_rep(self, reptuple: RepTuple):
        rep = reptuple.rep
        # TODO: Can probably check this is an ndarray of the right shape

        qubits = reptuple.qubits
        assert isinstance(qubits, (tuple, list)) and len(qubits) > 0

        reptype = reptuple.reptype
        assert isinstance(reptype, GateRep)

        if reptype == GateRep.QSIM_SUPEROPERATOR:
            if len(qubits) == 1:
                self.state.apply_ptm(qubits[0], rep)
            elif len(qubits) == 2:
                # The qubits are flipped here, and this is a known QuantumSim quirk
                self.state.apply_two_ptm(qubits[1], qubits[0], rep)
            else:
                raise ValueError("Cannot apply more than a 2 qubit operation")
        else:
            raise NotImplementedError(f"Cannot apply GateRep {reptype}")

    def _apply_instrument_rep(self, reptuple: RepTuple) -> OutcomeDict:
        rep = reptuple.rep
        assert isinstance(rep, (tuple, list)) and len(rep) > 1
        reset = rep[0]
        include_outcomes = rep[1]

        qubits = reptuple.qubits
        assert isinstance(qubits, (tuple, list)) and len(qubits) > 0

        reptype = reptuple.reptype
        assert isinstance(reptype, InstrumentRep)

        outcomes: OutcomeDict = defaultdict(list)

        if reptype == InstrumentRep.ZBASIS_PROJECTION:
            # TODO: Could do it all at once probably
            # but currently just copying measureRenormalizeQubit behavior
            for qbit in qubits:
                cbit = self._apply_projective_z_measure(qbit, reset)
                if include_outcomes:
                    outcomes[qbit].append(cbit)
        elif reptype == InstrumentRep.ZBASIS_PRE_POST_OPERATIONS:
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
                cbit = self._apply_projective_z_measure(qbit, reset)
                if include_outcomes:
                    outcomes[qbit].append(cbit)

            # Apply the post-op
            self.apply_reps_inplace([postop])
        elif reptype == InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT:
            if len(qubits) > 1:
                raise NotImplementedError(
                    "More than 1-qubit instruments not yet implemented"
                )
            instrument_dict = rep[0]
            assert set(instrument_dict.keys()) == set((0, 1))

            # Compute the probability of measuring 0
            prob_0 = self._apply_instrument_element_ptm_for_prob(
                instrument_dict[0].rep, qubits[0]
            )

            # Use RNG to see if we measure 0 or 1
            m = self._rng.random()
            cbit = 0 if m < prob_0 else 1
            if include_outcomes:
                outcomes[qubits[0]].append(cbit)

            # Apply the correct PTM based on the classical output we see
            rep_to_apply = instrument_dict[cbit]
            assert rep_to_apply.reptype in self.input_reps
            assert isinstance(rep_to_apply.reptype, GateRep)
            self.apply_reps_inplace([rep_to_apply])

            # Propogate and renormalize (maybe not needed, but safer to do it now)
            self.state.combine_and_apply_single_ptm(qubits[0])
            self.state.renormalize()

        return outcomes

    def _apply_projective_z_measure(
        self, qbit: int | str, reset: int | None
    ) -> int:
        results = self.state.peak_multiple_measurements([qbit])
        # Results has following structure
        # results: [({'A1': 0}, p0), ({'A1': 1}, p1)}
        # results[a][0][qubit] = measurement value
        # results[a][1]        = corresponding probability
        # where a is in {0,1}
        # (unless the bit is classical, in which case,
        # there is only one entry in results)

        # TODO: At this point, we also have probabilities
        # We could do also save that data, maybe helpful later

        m = self._rng.random()
        if m < results[0][1]:
            cbit = results[0][0][qbit]
        else:
            cbit = results[1][0][qbit]
        if qbit in self.state.idx_in_full_dm:
            self.state.project_measurement(qbit, cbit)
        if reset is not None:
            assert reset in [0, 1]
            self.state.set_bit(qbit, reset)
        self.state.renormalize()
        return cbit

    def _apply_instrument_element_ptm_for_prob(
        self, inst_elem_ptm: np.ndarray, inst_bit: int | str
    ) -> float:
        if not isinstance(self._state.full_dm, _DensityNP):
            raise ValueError("Expected a quantumsim.dm_np.DensityNP object")

        # Ensure state is propogated and up to date
        self.state.combine_and_apply_single_ptm(inst_bit)

        # Compute the reduced density matrix
        trace_tensor = np.array([1, 0, 0, 1])
        prob_compute_tensor = inst_elem_ptm[0] + inst_elem_ptm[3]
        bit0 = self.state.idx_in_full_dm[inst_bit]
        trace_argument = []
        for i in range(self.state.full_dm.no_qubits):
            if i == bit0:
                trace_argument.append(prob_compute_tensor)
            else:
                trace_argument.append(trace_tensor)
            trace_argument.append([i])
        indices = list(reversed(range(self.state.full_dm.no_qubits)))
        prob = np.einsum(self.state.full_dm.dm, indices, *trace_argument, optimize=True)  # type: ignore

        # we are doing mat-vec product on only first and last row for our target bit to get probability
        # prob = inst_elem_ptm[0] @ reduced_dm + inst_elem_ptm[3] @ reduced_dm
        # TODO: There is an implicit normalization cancelling out above
        # Be careful of this when moving to more qubits
        # prob *= inst_elem_ptm.shape[0] ** 0.25

        return prob

    def _get_encoding_attr(self, attr, ignore_no_serialize_flags=False):
        """Get an attribute for encoding/serialization purposes.

        This method retrieves specific attributes from the quantum state that are
        needed for serialization, including both standard attributes and internal
        QuantumSim state information.

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

        This method handles retrieval of various internal QuantumSim attributes
        that are needed for proper serialization and deserialization of the
        quantum state.
        """
        # Get any needed internal state from SparseDM
        if attr == "qubit_labels":
            return self.state.names
        if attr == "_qsim_classical":
            return self.state.classical
        if attr == "_qsim_idx_in_full_dm":
            return self.state.idx_in_full_dm
        if attr == "_qsim_dm_class":
            return type(self.state.full_dm)
        if attr == "_qsim_dm_no_qubits":
            return self.state.full_dm.no_qubits
        if attr == "_qsim_dm_data":
            return self.state.full_dm.to_array()
        if attr == "_qsim_max_bits_in_full_dm":
            return self.state.max_bits_in_full_dm
        if attr == "_qsim_classical_probability":
            return self.state.classical_probability
        if attr == "_qsim_single_ptms_to_do":
            return self.state.single_ptms_to_do
        if attr == "_qsim_maj_vot_mask":
            return self.state._last_majority_vote_mask
        if attr == "_qsim_maj_vot_array":
            return self.state._last_majority_vote_array

        # Otherwise fallback
        return super()._get_encoding_attr(attr, ignore_no_serialize_flags)

    @classmethod
    def _from_decoded_attrs(cls: type[T], attr_dict: Mapping) -> T:
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
        quantum state from stored data, including qubit labels and internal
        QuantumSim state information.
        """
        qubit_labels = attr_dict["qubit_labels"]
        obj = cls(len(qubit_labels), qubit_labels)

        # Restore internal QuantumSim state
        obj.state.classical = attr_dict["_qsim_classical"]
        obj.state.idx_in_full_dm = attr_dict["_qsim_idx_in_full_dm"]

        dm_class = attr_dict["_qsim_dm_class"]
        dm_no_qubits = attr_dict["_qsim_dm_no_qubits"]
        dm_data = attr_dict["_qsim_dm_data"]
        obj.state.full_dm = dm_class(dm_no_qubits, data=dm_data)

        obj.state.max_bits_in_full_dm = attr_dict["_qsim_max_bits_in_full_dm"]
        obj.state.classical_probability = attr_dict[
            "_qsim_classical_probability"
        ]
        ptm_data = attr_dict["_qsim_single_ptms_to_do"]

        assert isinstance(ptm_data, dict)
        single_ptms_to_do = defaultdict(list)
        for k, v in ptm_data.items():
            single_ptms_to_do[k] = v
        obj.state.single_ptms_to_do = single_ptms_to_do

        obj.state._last_majority_vote_mask = attr_dict["_qsim_maj_vot_mask"]
        obj.state._last_majority_vote_array = attr_dict["_qsim_maj_vot_array"]

        return obj
