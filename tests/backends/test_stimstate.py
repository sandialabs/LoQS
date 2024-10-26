"""Tester for loqs.backends.state.stimstate"""

import mock
import pytest
from tempfile import NamedTemporaryFile

try:
    import stim

    NO_STIM = False
except ImportError:
    NO_STIM = True

from loqs.backends.reps import GateRep, RepTuple, InstrumentRep
from loqs.backends.state import STIMQuantumState as STIMState


@pytest.mark.skipif(
    NO_STIM,
    reason="Skipping stim backend tests due to failed import"
)
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
        X_reps = [RepTuple("X 0", ["Q0"], GateRep.STIM_CIRCUIT_STR)]

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
            RepTuple("H 0", ["Q1"], GateRep.STIM_CIRCUIT_STR),
            RepTuple("CZ 0 1", ["Q0", "Q1"], GateRep.STIM_CIRCUIT_STR),
            # Not a normal way to specify H, but this should work
            RepTuple("H 1", ["Q0", "Q1"], GateRep.STIM_CIRCUIT_STR)
        ]

        # Start in the 10 state
        state10 = STIMState([1, 0], ["Q0", "Q1"])
        
        # The expected 11 state
        state11 = STIMState([1, 1], ["Q0", "Q1"])

        test3, _ = state10.apply_reps(CX_reps)
        self._check(test3, state11)

        # We should also be able to have all three commands in one rep
        CX_reps2 = [RepTuple("H 1\nCZ 0 1\nH 1", ["Q0", "Q1"], GateRep.STIM_CIRCUIT_STR)]
        test4, _ = state10.apply_reps(CX_reps2)
        self._check(test4, state11)

        # Let's try to pass in some unsupported reps
        with pytest.raises(NotImplementedError):
            test.apply_reps([
                RepTuple(None, "Q0", GateRep.UNITARY)
            ])
        
        with pytest.raises(NotImplementedError):
            test.apply_reps([
                RepTuple(None, "Q0", GateRep.PTM)
            ])
        
        with pytest.raises(NotImplementedError):
            test.apply_reps([
                RepTuple(None, "Q0", GateRep.QSIM_SUPEROPERATOR)
            ])

    def test_apply_instruments(self):
        # Start state
        state0 = STIMState([0], ["Q0"], seed=20241016)

        state1 = STIMState([1], ["Q0"], seed=20241016)

        # Use a Hadamard to put us in the + state
        h_rep = RepTuple("H 0", ["Q0"], GateRep.STIM_CIRCUIT_STR)

        # In-place 10 times
        # Also test no reset
        proj_rep = RepTuple((None, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
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
        proj2_rep = RepTuple((None, False), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test1 = state0.copy()
        outs = test1.apply_reps_inplace([h_rep, proj2_rep]*10)
        assert len(outs) == 0
        
        # Now another copy ten times at once with reset
        reset_rep = RepTuple((0, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test2 = state0.copy()
        outs = test2.apply_reps_inplace([h_rep, reset_rep]*10)
        outcomes2 = outs["Q0"]
        
        # Should be same outcomes because of RNG seeding
        assert outcomes1 == outcomes2

        # Now lets test pre/post op
        idle_rep = RepTuple("", ["Q0"], GateRep.STIM_CIRCUIT_STR)

        # Lets do X(pi/2) error before and nothing after
        pre_xpi2_rep = RepTuple(
            [0, True, h_rep, idle_rep], ["Q0"], InstrumentRep.ZBASIS_PRE_POST_OPERATIONS
        )

        test3 = state0.copy()
        outs = test3.apply_reps_inplace([pre_xpi2_rep]*10)
        outcomes3 = outs["Q0"]
        assert outcomes3 == outcomes1

        # Now let's do X(pi/2) after and no nothing before
        # Very first one we have to do X(pi/2) to get same outcomes
        post_xpi2_rep = RepTuple(
            [0, True, idle_rep, h_rep], ["Q0"], InstrumentRep.ZBASIS_PRE_POST_OPERATIONS
        )

        test4 = state0.copy()
        outs = test4.apply_reps_inplace([h_rep] + [post_xpi2_rep]*10)
        outcomes4 = outs["Q0"]
        assert outcomes4 == outcomes1

    def test_serialization(self):
        # Test bell state
        test = STIMState([1, 0], ["Q0", "Q1"])
        test.state.cx(0, 1)

        with NamedTemporaryFile("w+", suffix='.json') as tempf:
            test.write(tempf.name)
            
            test2 = STIMState.read(tempf.name)
            self._check(test, test2)

class TestSTIMQuantumStateFailedImport:
        # Mock not having stim available
        def test_failed_import(self):
            with mock.patch.dict('sys.modules', {
                    'stim': None,
                }):

                with pytest.raises(ImportError):
                    import importlib
                    import sys

                    mod = sys.modules['loqs.backends.state.stimstate']
                    importlib.reload(mod)
                    
