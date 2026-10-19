"""Tester for loqs.backends.state.stimstate.

Behaviour shared with the other state backends is in test_states.py.
"""

import pytest
import numpy as np

stim = pytest.importorskip("stim")

from loqs.backends.reps import (
    ProbabilisticStimGateRep,
    RepConstructionError,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
)
from loqs.backends import STIMQuantumState as STIMState


class TestSTIMQuantumState:

    def _check(self, state, expected_state):
        assert (
            state.state.canonical_stabilizers()
            == expected_state.state.canonical_stabilizers()
        )

    def test_init(self):
        qubit_labels = [f"Q{i}" for i in range(5)]
        s = STIMState(5, qubit_labels)

        # Make some nontrivial state to check with
        s.state.x(0)
        s.state.cx(0, 1)

        s3 = STIMState(s.state, qubit_labels)
        self._check(s3, s)

        s4 = STIMState(s.state.current_inverse_tableau(), qubit_labels)
        self._check(s4, s)

        # Initialize first qubit in 1 already
        s5 = STIMState([1, 0, 0, 0, 0], qubit_labels)
        s5.state.cx(0, 1)
        self._check(s5, s)

        # No labels should default to int list
        s7 = STIMState(s.state)
        self._check(s7, s)

        s8 = STIMState(s.state.current_inverse_tableau())
        self._check(s8, s)

        # Qubit label length mismatch should error a problem
        with pytest.raises(AssertionError):
            STIMState(s, ["Q0"])

    def test_gate_circuit_string_forms(self):
        """A rep may hold a multi-line circuit string, and a rep with an
        empty `qubits` tuple (used for annotations such as TICK) is passed
        straight through to the applied-circuit log."""
        state10 = STIMState([1, 0], ["Q0", "Q1"])
        state11 = STIMState([1, 1], ["Q0", "Q1"])

        # A CNOT via H CZ H, all in one rep
        cx_rep = StimCircuitGateRep("H 1\nCZ 0 1\nH 1", ["Q0", "Q1"])
        test, _ = state10.apply_reps([cx_rep])
        self._check(test, state11)

        test = STIMState([0], ["Q0"])
        test.apply_reps_inplace([StimCircuitGateRep("TICK", [])])
        assert "TICK" in str(test.latest_applied_circuit)

    def test_probabilistic_stim_operations_multibranch_multiqubit(self):
        """PROBABILISTIC_STIM_OPERATIONS with more than 2 branches and a
        multi-qubit branch operation must sample according to the given
        distribution. Each trial starts fresh from |00>, so the branch
        taken can be identified unambiguously from the qubits' Z-parities
        afterward: unchanged (branch 0), only Q0 flipped (branch 1), or
        both flipped (branch 2)."""
        probs = [0.5, 0.25, 0.25]
        two_qubit_rep = ProbabilisticStimGateRep(
            [("", probs[0]), ("X 0", probs[1]), ("X 0 1", probs[2])],
            ["Q0", "Q1"],
        )

        n_trials = 300
        test = STIMState([0, 0], ["Q0", "Q1"], seed=20260712)
        counts = [0, 0, 0]
        for _ in range(n_trials):
            test.apply_reps_inplace([two_qubit_rep])
            z0, z1 = test.state.peek_z(0), test.state.peek_z(1)
            if (z0, z1) == (1, 1):
                branch = 0
            elif (z0, z1) == (-1, 1):
                branch = 1
            else:
                assert (z0, z1) == (-1, -1)
                branch = 2
            counts[branch] += 1
            # Reset back to |00> for the next trial
            test.state.do_circuit(stim.Circuit("R 0\nR 1"))  # type: ignore

        for count, p in zip(counts, probs):
            sigma = np.sqrt(p * (1 - p) / n_trials)
            assert abs(count / n_trials - p) < 5 * sigma

    def test_probabilistic_stim_operations_bad_sum_raises(self):
        with pytest.raises(RepConstructionError):
            ProbabilisticStimGateRep([("", 0.5), ("X 0", 0.6)], ["Q0"])

    def test_apply_instrument_stim_circuit_str(self):
        """StimCircuitInstrumentRep reuses the gate-apply code path
        and then extracts outcomes from STIM's measurement record, using
        STIM's own reset instructions to prepare deterministic eigenstates
        so outcomes are exact rather than probabilistic."""
        # Z basis: M
        state0 = STIMState([0], ["Q0"])
        m_rep = StimCircuitInstrumentRep("M 0", ["Q0"])
        outs = state0.apply_reps_inplace([m_rep])
        assert outs["Q0"] == [0]

        # Measure-and-reset: outcome reported, but qubit ends in |0>
        state1b = STIMState([1], ["Q0"])
        mr_rep = StimCircuitInstrumentRep("MR 0", ["Q0"])
        outs = state1b.apply_reps_inplace([mr_rep])
        assert outs["Q0"] == [1]
        self._check(state1b, state0)

        # X basis: reset into the +X eigenstate, MX must read 0 exactly
        state_rx = STIMState([0], ["Q0"])
        rx_mx_rep = StimCircuitInstrumentRep("RX 0\nMX 0", ["Q0"])
        outs = state_rx.apply_reps_inplace([rx_mx_rep])
        assert outs["Q0"] == [0]

        # Multi-qubit measurement in a single instrument rep must map
        # each outcome back to its correct global qubit label.
        state01 = STIMState([0, 1], ["Q0", "Q1"])
        mm_rep = StimCircuitInstrumentRep("M 0 1", ["Q0", "Q1"])
        outs = state01.apply_reps_inplace([mm_rep])
        assert outs == {"Q0": [0], "Q1": [1]}

        # Negated target: physical qubit is |1>, negated target flips the
        # recorded outcome to 0.
        state1c = STIMState([1], ["Q0"])
        neg_m_rep = StimCircuitInstrumentRep("M 0", ["!Q0"])
        outs = state1c.apply_reps_inplace([neg_m_rep])
        assert outs["Q0"] == [0]

    def test_pickle_round_trip(self):
        """A plain pickle must carry the tableau across a process boundary."""
        import pickle

        from loqs.internal.serializable import _rebuild_from_encoding_attrs

        state = STIMState(3, ["Q0", "Q1", "Q2"])
        state.state.h(0)
        state.state.cx(0, 1)
        state.state.s(2)
        state.state.cx(1, 2)

        reduced = state.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
        assert reduced[0] is _rebuild_from_encoding_attrs

        state2 = pickle.loads(pickle.dumps(state))
        assert state2.qubit_labels == state.qubit_labels
        assert (
            state2.state.current_inverse_tableau()
            == state.state.current_inverse_tableau()
        )
