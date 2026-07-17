"""Tester for loqs.backends.state.stimstate"""

import pytest
import numpy as np

stim = pytest.importorskip("stim")

from loqs.backends.reps import (
    KrausGateRep,
    ProbabilisticStimGateRep,
    PTMGateRep,
    QSimSuperoperatorGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.backends import STIMQuantumState as STIMState


class TestSTIMQuantumState:

    def _check(self, state, expected_state):
        assert state.state.canonical_stabilizers() == expected_state.state.canonical_stabilizers()

    def test_init(self):
        # Base initializer
        qubit_labels = [f"Q{i}" for i in range(5)]
        s = STIMState(5, qubit_labels)

        # Make some nontrivial state to check with
        s.state.x(0)
        s.state.cx(0, 1)

        s2 = STIMState(s, qubit_labels)
        self._check(s2, s)

        s3 = STIMState(s.state, qubit_labels)
        self._check(s3, s)

        s4 = STIMState(s.state.current_inverse_tableau(), qubit_labels)
        self._check(s4, s)

        # Initialize first qubit in 1 already
        s5 = STIMState([1, 0, 0, 0, 0], qubit_labels)
        s5.state.cx(0,1)
        self._check(s5, s)

        # Cast checks
        s6 = STIMState.cast(s)
        self._check(s6, s)

        s_int_labels = STIMState(s, qubit_labels=None) # No labels should default to int list
        s7 = STIMState.cast(s.state)
        self._check(s7, s_int_labels)

        s8 = STIMState.cast(s.state.current_inverse_tableau())
        self._check(s8, s_int_labels)

        # Copy check
        s9 = s.copy()
        self._check(s9, s)
        
        # Qubit label length mismatch should error a problem
        with pytest.raises(AssertionError):
            STIMState(s, ["Q0"])
    
    def test_apply_gates(self):
        # Let's apply a X gate
        X_reps = [StimCircuitGateRep("X 0", ["Q0"])]

        # Start in the 0 state
        state0 = STIMState([0], ["Q0"])

        # Also prepare a 1 state as expected
        state1 = STIMState([1], ["Q0"])

        # Test both in-place and not
        test = state0.copy()
        test.apply_reps_inplace(X_reps)
        self._check(test, state1)
        
        test2, outcomes = state0.apply_reps(X_reps)
        self._check(test2, state1)
        assert len(outcomes) == 0

        # Let's try a CNOT via H CZ H
        CX_reps = [
            StimCircuitGateRep("H 0", ["Q1"]),
            StimCircuitGateRep("CZ 0 1", ["Q0", "Q1"]),
            # Not a normal way to specify H, but this should work
            StimCircuitGateRep("H 1", ["Q0", "Q1"])
        ]

        # Start in the 10 state
        state10 = STIMState([1, 0], ["Q0", "Q1"])
        
        # The expected 11 state
        state11 = STIMState([1, 1], ["Q0", "Q1"])

        test3, _ = state10.apply_reps(CX_reps)
        self._check(test3, state11)

        # We should also be able to have all three commands in one rep
        CX_reps2 = [StimCircuitGateRep("H 1\nCZ 0 1\nH 1", ["Q0", "Q1"])]
        test4, _ = state10.apply_reps(CX_reps2)
        self._check(test4, state11)

        # Let's try the probabilistic operations
        prob_of_reset = 0.4
        prob_reset_rep = ProbabilisticStimGateRep([("", 1-prob_of_reset), ("R 0", prob_of_reset)], ["Q0"])
        
        # Let's compute what the expected sampling should be
        rng = np.random.default_rng(20241026)
        # Note 0 and 1 are flipped here. If we pick first element, then we will stay in 1 state,
        # while if we pick second element, we will reset to 0 state
        expected_outcomes = [rng.choice([1, 0], p=[1-prob_of_reset, prob_of_reset]) for _ in range(10)]

        # Start in 1 state, but set RNG to be the same as above so we know expected choices
        state1_rng = STIMState([1], ["Q0"], seed=20241026)
        outcomes = []
        for _ in range(10):
            state1_rng.apply_reps_inplace([prob_reset_rep])
            # peek_z gives 1 if in state 0, -1 if in state 1
            # This prevents a measurement which would use the RNG and mess up our expected samples
            outcome = int(state1_rng.state.peek_z(0) == -1)
            outcomes.append(outcome)
            
            # Go back to 1 state for next test
            state1_rng.state.do_circuit(stim.Circuit("R 0\nX 0")) # type: ignore
        assert outcomes == expected_outcomes

        # Let's try to pass in some unsupported reps
        with pytest.raises(NotImplementedError):
            test.apply_reps([
                UnitaryGateRep(None, "Q0")
            ])
        
        with pytest.raises(NotImplementedError):
            test.apply_reps([
                PTMGateRep(None, "Q0")
            ])
        
        with pytest.raises(NotImplementedError):
            test.apply_reps([
                QSimSuperoperatorGateRep(None, "Q0")
            ])

        with pytest.raises(NotImplementedError):
            test.apply_reps([
                KrausGateRep([(np.eye(2), 1.0)], "Q0")
            ])

    def test_input_reps(self):
        state = STIMState(1, ["Q0"])
        assert set(state.input_reps) == {
            StimCircuitGateRep,
            ProbabilisticStimGateRep,
            ZBasisProjectionInstrumentRep,
            ZBasisPrePostInstrumentRep,
            StimCircuitInstrumentRep,
        }

    def test_gate_zero_qubit_passthrough(self):
        """A StimCircuitGateRep with an empty `qubits` tuple
        (used for annotations that take no qubit targets, such as TICK)
        is passed straight through to the applied-circuit log rather than
        going through qubit-index translation."""
        test = STIMState([0], ["Q0"])
        tick_rep = StimCircuitGateRep("TICK", [])
        test.apply_reps_inplace([tick_rep])
        assert "TICK" in str(test.latest_applied_circuit)

    def test_apply_gate_negated_qubit(self):
        """A `!`-prefixed qubit label in `qubits` marks that
        target as negated when translated into the underlying STIM
        circuit string. STIM records the logical NOT of the actual
        measurement result for a negated measurement target."""
        state1 = STIMState([1], ["Q0"])
        m_rep = StimCircuitGateRep("M 0", ["!Q0"])
        state1.apply_reps_inplace([m_rep])
        # Physical qubit is |1>, but the negated target flips the
        # recorded bit.
        assert state1.state.current_measurement_record() == [False]

    def test_probabilistic_stim_operations_multibranch_multiqubit(self):
        """PROBABILISTIC_STIM_OPERATIONS with more than 2 branches and a
        multi-qubit branch operation must sample according to the given
        distribution. Each trial starts fresh from |00>, so the branch
        taken can be identified unambiguously from the qubits' Z-parities
        afterward: unchanged (branch 0), only Q0 flipped (branch 1), or
        both flipped (branch 2)."""
        probs = [0.5, 0.25, 0.25]
        two_qubit_rep = ProbabilisticStimGateRep([("", probs[0]), ("X 0", probs[1]), ("X 0 1", probs[2])], ["Q0", "Q1"])

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

    def test_probabilistic_stim_operations_negative_probability_raises(self):
        rep = ProbabilisticStimGateRep([("", 1.2), ("X 0", -0.2)], ["Q0"])
        test = STIMState([0], ["Q0"])
        with pytest.raises(AssertionError, match="positive"):
            test.apply_reps_inplace([rep])

    def test_probabilistic_stim_operations_bad_sum_raises(self):
        rep = ProbabilisticStimGateRep([("", 0.5), ("X 0", 0.6)], ["Q0"])
        test = STIMState([0], ["Q0"])
        with pytest.raises(AssertionError, match="sum to 1"):
            test.apply_reps_inplace([rep])

    def test_apply_instrument_stim_circuit_str(self):
        """StimCircuitInstrumentRep reuses the gate-apply code path
        and then extracts outcomes from STIM's measurement record. This
        covers the single-qubit M/MX/MY/MZ/MR/MRX/MRY/MRZ family, using
        STIM's own RX/RY/R reset instructions to prepare deterministic
        eigenstates so outcomes are exact rather than probabilistic."""
        # Z basis: M / MZ
        state0 = STIMState([0], ["Q0"])
        m_rep = StimCircuitInstrumentRep("M 0", ["Q0"])
        outs = state0.apply_reps_inplace([m_rep])
        assert outs["Q0"] == [0]

        state1 = STIMState([1], ["Q0"])
        mz_rep = StimCircuitInstrumentRep("MZ 0", ["Q0"])
        outs = state1.apply_reps_inplace([mz_rep])
        assert outs["Q0"] == [1]

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

        # Y basis: reset into the +Y eigenstate, MY must read 0 exactly
        state_ry = STIMState([0], ["Q0"])
        ry_my_rep = StimCircuitInstrumentRep("RY 0\nMY 0", ["Q0"])
        outs = state_ry.apply_reps_inplace([ry_my_rep])
        assert outs["Q0"] == [0]

        # MRX/MRY/MRZ: measure-and-reset in a non-Z basis
        state_mrx = STIMState([0], ["Q0"])
        state_mrx.apply_reps_inplace(
            [StimCircuitGateRep("RX 0", ["Q0"])]
        )
        mrx_rep = StimCircuitInstrumentRep("MRX 0", ["Q0"])
        outs = state_mrx.apply_reps_inplace([mrx_rep])
        assert outs["Q0"] == [0]
        # After MRX, the qubit is reset back to the +X eigenstate
        assert state_mrx.state.peek_x(0) == 1

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

    def test_apply_instrument_stim_circuit_str_two_qubit_measurement_gap(self):
        """Documents current behavior for two-qubit Pauli-product
        measurements (e.g. MZZ) applied via StimCircuitInstrumentRep:
        `_apply_gate_rep`'s `include_outcomes` list only recognizes the
        single-qubit M/MX/MY/MZ/MR/MRX/MRY/MRZ family, so MZZ's measurement
        bit is recorded in STIM's own measurement record but is not added
        to `latest_measurement_labels` and therefore produces no entry in
        the returned outcome dict at all, even though the measurement
        itself is applied to the state correctly."""
        bell = STIMState([0, 0], ["Q0", "Q1"])
        bell.apply_reps_inplace(
            [StimCircuitGateRep("H 0\nCX 0 1", ["Q0", "Q1"])]
        )
        mzz_rep = StimCircuitInstrumentRep("MZZ 0 1", ["Q0", "Q1"])
        outs = bell.apply_reps_inplace([mzz_rep])
        assert outs == {}
        # The measurement itself was still applied to STIM's own record.
        assert bell.state.current_measurement_record() == [False]

    def test_unsupported_instrument_rep_raises(self):
        test = STIMState([0], ["Q0"])
        with pytest.raises(NotImplementedError):
            test.apply_reps_inplace([
                ZBasisOutcomeOperationDictInstrumentRep({}, True, ["Q0"])
            ])

    def test_apply_instruments(self):
        # Start state
        state0 = STIMState([0], ["Q0"], seed=20241016)

        state1 = STIMState([1], ["Q0"], seed=20241016)

        # Use a Hadamard to put us in the + state
        h_rep = StimCircuitGateRep("H 0", ["Q0"])

        # In-place 10 times
        # Also test no reset
        proj_rep = ZBasisProjectionInstrumentRep(None, True, ["Q0"])
        test = state0.copy()
        outcomes1 = []
        for _ in range(10):
            outs = test.apply_reps_inplace([h_rep, proj_rep])
            out = outs["Q0"][0]
            outcomes1.append(out)

            # Check measurement without reset
            if out == 1:
                self._check(test, state1)
                # Reset manually
                test.state.reset(0)
            else:
                self._check(test, state0)
        
        # Also test no outcomes
        proj2_rep = ZBasisProjectionInstrumentRep(None, False, ["Q0"])
        test1 = state0.copy()
        outs = test1.apply_reps_inplace([h_rep, proj2_rep]*10)
        assert len(outs) == 0
        
        # Now another copy ten times at once with reset
        reset_rep = ZBasisProjectionInstrumentRep(0, True, ["Q0"])
        test2 = state0.copy()
        outs = test2.apply_reps_inplace([h_rep, reset_rep]*10)
        outcomes2 = outs["Q0"]
        
        # Should be same outcomes because of RNG seeding
        assert outcomes1 == outcomes2

        # Now lets test pre/post op
        idle_rep = StimCircuitGateRep("", ["Q0"])

        # Lets do X(pi/2) error before and nothing after
        pre_xpi2_rep = ZBasisPrePostInstrumentRep(0, True, h_rep, idle_rep, ["Q0"])

        test3 = state0.copy()
        outs = test3.apply_reps_inplace([pre_xpi2_rep]*10)
        outcomes3 = outs["Q0"]
        assert outcomes3 == outcomes1

        # Now let's do X(pi/2) after and no nothing before
        # Very first one we have to do X(pi/2) to get same outcomes
        post_xpi2_rep = ZBasisPrePostInstrumentRep(0, True, idle_rep, h_rep, ["Q0"])

        test4 = state0.copy()
        outs = test4.apply_reps_inplace([h_rep] + [post_xpi2_rep]*10)
        outcomes4 = outs["Q0"]
        assert outcomes4 == outcomes1

    def test_zbasis_projection_reset_to_1(self):
        state1 = STIMState([1], ["Q0"], seed=20260712)
        h_rep = StimCircuitGateRep("H 0", ["Q0"])
        reset1_rep = ZBasisProjectionInstrumentRep(1, True, ["Q0"])

        test = STIMState([0], ["Q0"], seed=20260712)
        for _ in range(10):
            test.apply_reps_inplace([h_rep, reset1_rep])
            self._check(test, state1)

    def test_zbasis_pre_post_operations_reset_to_1(self):
        """`reset=1` for ZBASIS_PRE_POST_OPERATIONS must always leave the
        qubit reset to |1>, mirroring `test_zbasis_projection_reset_to_1`
        (both route through the same underlying `_measure_and_reset`)."""
        state1 = STIMState([1], ["Q0"], seed=20260712)
        h_rep = StimCircuitGateRep("H 0", ["Q0"])
        idle_rep = StimCircuitGateRep("", ["Q0"])
        pre_h_reset1_rep = ZBasisPrePostInstrumentRep(1, True, h_rep, idle_rep, ["Q0"])

        test = STIMState([0], ["Q0"], seed=20260712)
        for _ in range(10):
            test.apply_reps_inplace([pre_h_reset1_rep])
            self._check(test, state1)

    def test_zbasis_projection_multiqubit(self):
        """A single ZBASIS_PROJECTION rep applied to multiple qubits at
        once must measure/reset every qubit in `qubits`."""
        h_reps = [
            StimCircuitGateRep("H 0", ["Q0"]),
            StimCircuitGateRep("H 0", ["Q1"]),
        ]
        reset0_rep = ZBasisProjectionInstrumentRep(0, True, ["Q0", "Q1"])

        state00 = STIMState([0, 0], ["Q0", "Q1"])
        test = STIMState([0, 0], ["Q0", "Q1"], seed=20260712)
        for _ in range(10):
            outs = test.apply_reps_inplace(h_reps + [reset0_rep])
            assert set(outs.keys()) == {"Q0", "Q1"}
            assert len(outs["Q0"]) == 1 and len(outs["Q1"]) == 1
            self._check(test, state00)

    def test_serialization(self, make_temp_path):
        # Test bell state
        test = STIMState([1, 0], ["Q0", "Q1"])
        test.state.cx(0, 1)

        with make_temp_path(suffix='.json') as tmp_path:
            test.write(tmp_path)
            test2 = STIMState.read(tmp_path)
            self._check(test, test2)

    def test_serialization_after_instrument(self, make_temp_path):
        """Serialization must round-trip correctly after applying an
        InstrumentRep, not just a plain gate-only circuit."""
        h_rep = StimCircuitGateRep("H 0", ["Q0"])
        reset_rep = ZBasisProjectionInstrumentRep(0, True, ["Q0"])

        test = STIMState([0], ["Q0"], seed=20260712)
        outs_before = test.apply_reps_inplace([h_rep, reset_rep])

        with make_temp_path(suffix=".json") as tmp_path:
            test.write(tmp_path)
            test2 = STIMState.read(tmp_path)

        self._check(test, test2)
        assert outs_before["Q0"] == [0]
        # reset=0 must always leave the qubit in |0> both before and
        # after the round trip.
        self._check(test2, STIMState([0], ["Q0"]))

# class TestSTIMQuantumStateFailedImport:
#         # Mock not having stim available
#         def test_failed_import(self):
#             with mock.patch.dict('sys.modules', {
#                     'stim': None,
#                 }):

#                 with pytest.raises(ImportError):
#                     import importlib
#                     import sys

#                     mod = sys.modules['loqs.backends.state.stimstate']
#                     importlib.reload(mod)
                    
