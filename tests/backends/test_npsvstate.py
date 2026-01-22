"""Tester for loqs.backends.state.qsimstate"""

import mock
import numpy as np
import pytest
from tempfile import NamedTemporaryFile

from loqs.backends.reps import GateRep, RepTuple, InstrumentRep
from loqs.backends.state import NumpyStatevectorQuantumState as SVState


class TestNumPyStatevectorQuantumState:

    def _check(self, state, expected_state):
        assert state.qubit_labels == expected_state.qubit_labels
        assert state.seed == expected_state.seed
        assert np.allclose(state.state, expected_state.state)

    def test_init(self):
        # Base initializer
        qubit_labels = [f"Q{i}" for i in range(5)]
        s = SVState(5, qubit_labels)

        s2 = SVState(s)
        self._check(s2, s)

        # Bitstring initializer
        s3 = SVState([0, 0, 0, 0, 0], qubit_labels)
        self._check(s3, s)

        # Cast checks
        s4 = SVState.cast(s)
        self._check(s4, s)

        # (Flat) Numpy array check also
        all0_state = np.zeros(2**5)
        all0_state[0] = 1
        s5 = SVState.cast((all0_state, qubit_labels))
        self._check(s5, s)

        # This one won't have same labels
        s6 = SVState.cast(5)
        self._check(SVState(5), s6)

        # Copy check
        s7 = s.copy()
        self._check(s7, s)
    
    def test_apply_gates(self):
        # Let's apply a X gate
        U_X = np.array([[0, 1], [1, 0]])
        X_reps = [RepTuple(U_X, ["Q0"], GateRep.UNITARY)]

        # Start in the 0 state
        state0 = SVState([0], ["Q0"])

        # Also prepare a 1 state as expected
        state1 = SVState([1], ["Q0"])

        # Test both in-place and not
        test = state0.copy()
        test.apply_reps_inplace(X_reps)
        self._check(test, state1)
        
        test2, outcomes = state0.apply_reps(X_reps)
        self._check(test2, state1)
        assert len(outcomes) == 0

        # Let's try a CNOT via H CZ H
        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        U_CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        CX_reps = [
            RepTuple(U_H, ["Q1"], GateRep.UNITARY),
            RepTuple(U_CZ, ["Q0", "Q1"], GateRep.UNITARY),
            RepTuple(U_H, ["Q1"], GateRep.UNITARY)
        ]

        # Start in the |10> (big-endian) state
        state10 = SVState([1, 0], ["Q0", "Q1"])
        
        # The expected |11> state
        state11 = SVState([1, 1], ["Q0", "Q1"])

        test3, _ = state10.apply_reps(CX_reps)
        self._check(test3, state11)

        # TODO: Test Kraus
        # Test Kraus operator where applying X with prob 1, and I with prob 0
        X_kraus_rep_w_prob = RepTuple([(U_X, 1.0), (np.eye(2), 0.0)], ["Q0"], GateRep.KRAUS_OPERATORS)
        for _ in range(10):
            test4 = state0.copy()
            test4.apply_reps_inplace([X_kraus_rep_w_prob])
            self._check(test4, state1)
        
        # Test Kraus operator where bitflip happens with half the time
        outcomes1 = []
        half_bitflip_w_prob = RepTuple([(1/np.sqrt(2)*U_X, 0.5), (1/np.sqrt(2)*np.eye(2), 0.5)], ["Q0"], GateRep.KRAUS_OPERATORS)
        test5 = SVState([0], ["Q0"], seed=20260122)
        for _ in range(10):
            test5.apply_reps_inplace([half_bitflip_w_prob])
            # Store if we are in 1 state or 0 state
            outcomes1.append(test5.state[1])
            # Manual reset
            test5._state[0] = 1
            test5._state[1] = 0
        # outcomes should not be all 0 or 1
        assert any([np.isclose(o, 0) for o in outcomes1]) and any([np.isclose(o, 1) for o in outcomes1])

        # Lets do the same half bitflip, but force probability computation
        # Outcomes should be the same if we seed the same
        outcomes2 = []
        half_bitflip_wout_prob = RepTuple([(1/np.sqrt(2)*U_X, None), (1/np.sqrt(2)*np.eye(2), None)], ["Q0"], GateRep.KRAUS_OPERATORS)
        test6 = SVState([0], ["Q0"], seed=20260122)
        for _ in range(10):
            test6.apply_reps_inplace([half_bitflip_wout_prob])
            # Store if we are in 1 state or 0 state
            outcomes2.append(test6.state[1])
            # Manual reset
            test6._state[0] = 1
            test6._state[1] = 0
        # Outcomes should be the same since we seeded the same
        assert np.allclose(outcomes1, outcomes2)

        # Let's try to pass in some unsupported reps
        with pytest.raises(ValueError):
            test.apply_reps([
                RepTuple(None, "Q0", GateRep.PTM)
            ])
        
        with pytest.raises(ValueError):
            test.apply_reps([
                RepTuple(None, "Q0", GateRep.STIM_CIRCUIT_STR)
            ])

        with pytest.raises(ValueError):
            test.apply_reps([
                RepTuple(None, "Q0", GateRep.QSIM_SUPEROPERATOR)
            ])

    def test_apply_instruments(self):
        # H gate to get + state for testing
        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_rep = RepTuple(U_H, ["Q0"], GateRep.UNITARY)

        state0 = SVState([0], ["Q0"], seed=20241016)

        state1 = SVState([1], ["Q0"], seed=20241016)

        # In-place 10 times
        # Also test no reset
        proj_rep = RepTuple((None, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test = state0.copy()
        outcomes1 = []
        for _ in range(10):
            outs = test.apply_reps_inplace([H_rep, proj_rep])
            out = outs["Q0"][0]
            outcomes1.append(out)

            # Check measurement without reset
            if out == 1:
                self._check(test, state1)
                # Reset manually
                test.state[1] = 0
                test.state[0] = 1
            else:
                self._check(test, state0)
        
        # Also test no outcomes
        proj2_rep = RepTuple((None, False), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test1 = state0.copy()
        outs = test1.apply_reps_inplace([H_rep, proj2_rep]*10)
        assert len(outs) == 0
        
        # Now another copy ten times at once with reset
        reset_rep = RepTuple((0, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test2 = state0.copy()
        outs = test2.apply_reps_inplace([H_rep, reset_rep]*10)
        outcomes2 = outs["Q0"]
        
        # Should be same outcomes because of RNG seeding
        assert outcomes1 == outcomes2

        # Now lets test pre/post op
        U_I = np.eye(2)
        idle_rep = RepTuple(U_I, ["Q0"], GateRep.UNITARY)

        # Lets do X(pi/2) error before and nothing after
        pre_H_rep = RepTuple(
            [0, True, H_rep, idle_rep], ["Q0"], InstrumentRep.ZBASIS_PRE_POST_OPERATIONS
        )

        test3 = state0.copy()
        outs = test3.apply_reps_inplace([pre_H_rep]*10)
        outcomes3 = outs["Q0"]
        assert outcomes3 == outcomes1

        # Now let's do X(pi/2) after and no nothing before
        # Very first one we have to do X(pi/2) to get same outcomes
        post_H_rep = RepTuple(
            [0, True, idle_rep, H_rep], ["Q0"], InstrumentRep.ZBASIS_PRE_POST_OPERATIONS
        )

        test4 = state0.copy()
        outs = test4.apply_reps_inplace([H_rep] + [post_H_rep]*10)
        outcomes4 = outs["Q0"]
        assert outcomes4 == outcomes1

        # Finally let's do the outcome/operation dict
        effect0 = np.array([[1, 0]])
        effect1 = np.array([[0, 1]])

        ideal_maps = {
            0: RepTuple(effect0.T @ effect0, ["Q0"], GateRep.UNITARY),
            1: RepTuple(effect1.T @ effect1, ["Q0"], GateRep.UNITARY)
        }
        ideal_map_rep = RepTuple((ideal_maps, True), ["Q0"], InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT)

        test5 = state0.copy()
        outs = test5.apply_reps_inplace([H_rep, ideal_map_rep]*10)
        outcomes5 = outs["Q0"]
        assert outcomes5 == outcomes1

        # Let's use the instrument to also do reset
        reset_maps = {
            0: RepTuple(effect0.T @ effect0, ["Q0"], GateRep.UNITARY),
            1: RepTuple(effect0.T @ effect1, ["Q0"], GateRep.UNITARY)
        }
        reset_map_rep = RepTuple((reset_maps, True), ["Q0"], InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT)

        test6 = state0.copy()
        outs = test6.apply_reps_inplace([H_rep, reset_map_rep]*10)
        outcomes6 = outs["Q0"]
        assert outcomes6 == outcomes1

        noisy_reset_maps = {
            0: RepTuple(U_H @ effect0.T @ effect0, ["Q0"], GateRep.UNITARY),
            1: RepTuple(U_H @ effect0.T @ effect1, ["Q0"], GateRep.UNITARY)
        }
        noisy_reset_map_rep = RepTuple((noisy_reset_maps, True), ["Q0"], InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT)

        test7 = state0.copy()
        outs = test7.apply_reps_inplace([H_rep] + [noisy_reset_map_rep]*10)
        outcomes7 = outs["Q0"]
        assert outcomes7 == outcomes1

    def test_serialization(self):
        # Start in the 10 state
        state10 = SVState([1, 0], ["Q0", "Q1"])
        
        # Let's try a CNOT via H CZ H
        # But let's split the H CZ before serialization
        # and final H after serialization
        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        U_CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

        test, _ = state10.apply_reps([RepTuple(U_H, ["Q1"], GateRep.UNITARY)])
        test.apply_reps_inplace([RepTuple(U_CZ, ["Q0", "Q1"], GateRep.UNITARY)])

        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            test.write(tempf.name)
            
            test2: SVState = SVState.read(tempf.name)
        
        # And finish applying
        test2.apply_reps_inplace([RepTuple(U_H, ["Q1"], GateRep.UNITARY)])
        
        # The expected 11 state
        state11 = SVState([1, 1], ["Q0", "Q1"])

        self._check(test2, state11)
                    
