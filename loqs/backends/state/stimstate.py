#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import singledispatchmethod
import h5py
import numpy as np
from typing import ClassVar, TypeAlias, TypeVar, TYPE_CHECKING, Any

from loqs.backends import is_backend_available
from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    ProbabilisticStimGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
    is_rep_compatible,
)
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
STIMStateLike: TypeAlias = (
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

    _SERIALIZE_ATTRS = ["qubit_labels", "_stim_state_vector"]
    """`seed` is deliberately not here to avoid triggering re-caching.
    See #118 for more details."""

    _state: _TableauSimulator
    """Underlying state object."""

    qubit_labels: list[QubitTypes]
    """Qubit labels.

    These are used to map local ints
    to global ints in
    [](api:StimCircuitGateRep) reps.
    """

    @property
    def state(self) -> _TableauSimulator:
        """Get the underlying STIM TableauSimulator state object.

        Returns
        -------
        _TableauSimulator
            The internal STIM TableauSimulator object that represents the quantum state.
        """
        return self._state

    @property
    def input_reps(self) -> list[type[GateRep | InstrumentRep]]:
        return [
            StimCircuitGateRep,
            ProbabilisticStimGateRep,
            ZBasisProjectionInstrumentRep,
            ZBasisPrePostInstrumentRep,
            StimCircuitInstrumentRep,
        ]

    def __init__(
        self,
        state: STIMStateLike,
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

        seed:
            Optional RNG seed. If not provided, default NumPy RNG behavior applies.

        Notes
        -----
        STIM's [](api:stim.TableauSimulator) has its' own internal RNG. We try to prime
        it as much as possible for consistency, but we cannot guarantee completely
        identical RNG when copying/deserializing these objects.
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
        outcomes: OutcomeDict = defaultdict(list)

        if reset_latest_circ:
            self.latest_applied_circuit = _Circuit()

        for rep in reps:
            if isinstance(rep, GateRep):
                self._apply_gate_rep(rep)
            elif isinstance(rep, InstrumentRep):
                rep_outcomes = self._apply_instrument_rep(rep)

                # Merge outcomes with already observed outcomes
                for k, v in rep_outcomes.items():
                    outcomes[k].extend(v)
            else:
                raise NotImplementedError(
                    f"Cannot apply unknown rep type {type(rep).__name__}"
                )

        return outcomes

    def apply_reps(
        self, reps: Sequence
    ) -> tuple[STIMQuantumState, OutcomeDict]:
        return super().apply_reps(reps)

    def copy(self) -> STIMQuantumState:
        new_state = STIMQuantumState(self.state, self.qubit_labels, self.seed)
        new_state._rng = deepcopy(self._rng)
        return new_state

    def reset_seed(self, new_seed: int | None) -> None:
        """Reset the random seed for the quantum state.

        Unlike some other implementations, this method explicitly forces a fresh
        RNG initialization with the new seed in the underlying
        [](api:stim.TableauSimulator) object.

        Parameters
        ----------
        new_seed : int | None
            The new random seed to use. If None, the random number generator
            will use its default seeding behavior.
        """
        # We explicitly don't want to copy RNG here, force a new RNG seed
        self._state = self._state.copy(copy_rng=False, seed=new_seed)
        self.seed = new_seed
        self._rng = np.random.default_rng(new_seed)

    @singledispatchmethod
    def _apply_gate_rep(self, rep: GateRep) -> None:
        raise NotImplementedError(
            f"Cannot apply {type(rep).__name__} to {self.name}"
        )

    @_apply_gate_rep.register
    def _(self, rep: StimCircuitGateRep) -> None:
        qubits = rep.qubit_labels
        # String qubit labels are this backend's own requirement, not a
        # general OperationRep invariant, so still checked here.
        assert all(isinstance(q, str) for q in qubits)
        circuit_str = rep.circuit_str

        if len(qubits) == 0:
            # This is a STIM annotation or comment, pass it on to applied circuit directly
            self.latest_applied_circuit += _Circuit(circuit_str)
            return

        # We have three types of indices here
        # Local: The placeholder/template qubit used in the rep
        # Global: The qubit label
        # Internal: The qubit label index
        local_to_global = {}
        local_to_internal = {}
        for i, q in enumerate(qubits):
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
        for line in circuit_str.split("\n"):
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

    @_apply_gate_rep.register
    def _(self, rep: ProbabilisticStimGateRep) -> None:
        qubits = rep.qubit_labels
        # String qubit labels are this backend's own requirement (see
        # the StimCircuitGateRep overload above).
        assert all(isinstance(q, str) for q in qubits)
        operations = rep.operations
        probs = [r[1] for r in operations]

        # Pick an op to apply
        idx_to_apply = self._rng.choice(list(range(len(operations))), p=probs)

        rep_to_apply = StimCircuitGateRep(operations[idx_to_apply][0], qubits)

        # Apply chosen op
        self.apply_reps_inplace([rep_to_apply], reset_latest_circ=False)

    @singledispatchmethod
    def _apply_instrument_rep(self, rep: InstrumentRep) -> OutcomeDict:
        raise NotImplementedError(
            f"Cannot apply {type(rep).__name__} to {self.name}"
        )

    @_apply_instrument_rep.register
    def _(self, rep: ZBasisProjectionInstrumentRep) -> OutcomeDict:
        qubits = rep.qubit_labels
        assert len(qubits) > 0

        outcomes: OutcomeDict = defaultdict(list)
        reset = rep.reset
        include_outcomes = rep.include_outcome

        for qbit in qubits:
            cbit = self._measure_and_reset(qbit, reset)
            if include_outcomes:
                outcomes[qbit].append(cbit)
        return outcomes

    @_apply_instrument_rep.register
    def _(self, rep: ZBasisPrePostInstrumentRep) -> OutcomeDict:
        qubits = rep.qubit_labels
        assert len(qubits) > 0

        outcomes: OutcomeDict = defaultdict(list)
        reset = rep.reset
        include_outcomes = rep.include_outcome

        # is_rep_compatible is backend-specific, so still checked here.
        preop = rep.pre_op
        postop = rep.post_op
        assert is_rep_compatible(type(preop), self.input_reps)
        assert is_rep_compatible(type(postop), self.input_reps)

        # Apply the pre-op
        self.apply_reps_inplace([preop])

        # Do perfect measurement
        for qbit in qubits:
            cbit = self._measure_and_reset(qbit, reset)
            if include_outcomes:
                outcomes[qbit].append(cbit)

        # Apply the post-op
        self.apply_reps_inplace([postop])
        return outcomes

    @_apply_instrument_rep.register
    def _(self, rep: StimCircuitInstrumentRep) -> OutcomeDict:
        qubits = rep.qubit_labels
        assert len(qubits) > 0

        outcomes: OutcomeDict = defaultdict(list)
        circuit_str = rep.circuit_str

        self.latest_measurement_labels = []

        # We'll reuse the gate apply code...
        self.apply_reps_inplace(
            [StimCircuitGateRep(circuit_str, qubits)],
            reset_latest_circ=False,
        )

        # but then post-process to grab the outcomes from the measurement record
        current_mr = self.state.current_measurement_record()
        mr_entries = [
            int(mre)
            for mre in current_mr[-len(self.latest_measurement_labels) :]
        ]

        for qbit, cbit in zip(self.latest_measurement_labels, mr_entries):
            outcomes[qbit].append(cbit)

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

    def _get_encoding_attr(self, attr, ignore_no_serialize_flags=False):
        if attr == "_stim_state_vector":
            # The tableau itself (six small bit-packed arrays, O(n^2) bits)
            # rather than its exponentially-sized dense state vector -- the
            # same information `state_vector()` would derive, at a fraction
            # of the size and with no exponential reconstruction cost.
            return self.state.current_inverse_tableau().to_numpy(bit_packed=True)

        # Otherwise fallback
        return super()._get_encoding_attr(attr, ignore_no_serialize_flags)

    @classmethod
    def _from_decoded_attrs(cls: type[T], attr_dict: Mapping) -> T:
        qubit_labels = attr_dict["qubit_labels"]
        # "seed" is no longer written, but an older file may still have
        # it -- restoring it there is harmless (if not genuinely
        # meaningful; see _SERIALIZE_ATTRS's own note on why it's dropped).
        seed = attr_dict.get("seed")
        encoded_tableau = attr_dict["_stim_state_vector"]

        if isinstance(encoded_tableau, np.ndarray):
            # A dense state vector, from a file written before this class
            # stored the tableau's own compact bit-packed form directly.
            tableau = _Tableau.from_state_vector(encoded_tableau, endian="little")
        else:
            x2x, x2z, z2x, z2z, x_signs, z_signs = encoded_tableau
            tableau = _Tableau.from_numpy(
                x2x=x2x, x2z=x2z, z2x=z2x, z2z=z2z, x_signs=x_signs, z_signs=z_signs
            )

        obj = cls(tableau, qubit_labels=qubit_labels)
        obj.reset_seed(seed)
        return obj
