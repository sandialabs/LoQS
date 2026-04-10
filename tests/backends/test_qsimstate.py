"""Tester for loqs.backends.state.qsimstate"""

import os
import tempfile
import json

import mock
import numpy as np
import pytest

try:
    from quantumsim.sparsedm import SparseDM as _SparseDM
    from quantumsim import ptm as _ptm
    
    NO_QSIM = False
except ImportError:
    NO_QSIM = True

from loqs.backends.reps import GateRep, RepTuple, InstrumentRep
from loqs.backends import QSimQuantumState as QSimState


@pytest.mark.skipif(
    NO_QSIM,
    reason="Skipping quantumsim backend tests due to failed import"
)
class TestQSimQuantumState:

    def _check(self, state, expected_state):
        assert state.state.names == expected_state.state.names
        assert state.seed == expected_state.seed
        assert np.allclose(state.state.full_dm.dm, expected_state.state.full_dm.dm)

    def test_init(self):
        # Base initializer
        qubit_labels = [f"Q{i}" for i in range(5)]
        s = QSimState(5, qubit_labels)

        s2 = QSimState(s)
        self._check(s2, s)

        qsim_dm = _SparseDM(qubit_labels)
        s3 = QSimState(qsim_dm)
        self._check(s3, s)

        # Cast checks
        s4 = QSimState.cast(s)
        self._check(s4, s)

        s5 = QSimState.cast(qsim_dm)
        self._check(s5, s)

        # This one won't have same labels
        s6 = QSimState.cast(5)
        self._check(QSimState(5), s6)

        # Copy check
        s7 = s.copy()
        self._check(s7, s)
    
    def test_apply_gates(self):
        # Let's apply a X gate
        x_ptm = _ptm.rotate_x_ptm(np.pi)
        X_reps = [RepTuple(x_ptm, ["Q0"], GateRep.QSIM_SUPEROPERATOR)]

        # Start in the 0 state
        state0 = QSimState(1, ["Q0"])
        state0.state.ensure_dense("Q0")

        # Also prepare a 1 state as expected
        state1 = QSimState(1, ["Q0"])
        state1.state.set_bit("Q0", 1)
        state1.state.ensure_dense("Q0")

        # Test both in-place and not
        test = state0.copy()
        test.apply_reps_inplace(X_reps)
        test.state.combine_and_apply_single_ptm("Q0") # Actually force propogation
        self._check(test, state1)
        
        test2, outcomes = state0.apply_reps(X_reps)
        test2.state.combine_and_apply_single_ptm("Q0") # Actually force propogation
        self._check(test2, state1)
        assert len(outcomes) == 0

        # Let's try a CNOT via H CZ H
        h_ptm = _ptm.hadamard_ptm()
        cz_ptm = state0.state._cphase_ptm
        CX_reps = [
            RepTuple(h_ptm, ["Q1"], GateRep.QSIM_SUPEROPERATOR),
            RepTuple(cz_ptm, ["Q0", "Q1"], GateRep.QSIM_SUPEROPERATOR),
            RepTuple(h_ptm, ["Q1"], GateRep.QSIM_SUPEROPERATOR)
        ]

        # Start in the 10 state
        state10 = QSimState(2, ["Q0", "Q1"])
        state10.state.set_bit("Q0", 1)
        state10.state.ensure_dense("Q0")
        state10.state.ensure_dense("Q1")
        
        # The expected 11 state
        state11 = state10.copy()
        state11.state.set_bit("Q1", 1)
        state11.state.ensure_dense("Q0")
        state11.state.ensure_dense("Q1")

        test3, _ = state10.apply_reps(CX_reps)
        test3.state.combine_and_apply_single_ptm("Q0") # Actually force propogation
        test3.state.combine_and_apply_single_ptm("Q1") # Actually force propogation
        self._check(test3, state11)

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
                RepTuple(None, "Q0", GateRep.STIM_CIRCUIT_STR)
            ])

        # Try to pass in too many qubits
        with pytest.raises(ValueError):
            test.apply_reps([
                RepTuple(None, ["Q0", "Q1", "Q2"], GateRep.QSIM_SUPEROPERATOR)
            ])

    def test_apply_instruments(self):
        # H gate to get + state for testing
        xpi2_ptm = _ptm.rotate_x_ptm(np.pi/2)
        xpi2_rep = RepTuple(xpi2_ptm, ["Q0"], GateRep.QSIM_SUPEROPERATOR)

        state0 = QSimState(1, ["Q0"], seed=20241016)

        state1 = QSimState(1, ["Q0"], seed=20241016)
        state1.state.set_bit("Q0", 1)
        # This time, keep it classical since we are going to measure

        # In-place 10 times
        # Also test no reset
        proj_rep = RepTuple((None, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test = state0.copy()
        outcomes1 = []
        for _ in range(10):
            outs = test.apply_reps_inplace([xpi2_rep, proj_rep])
            out = outs["Q0"][0]
            outcomes1.append(out)

            # Check measurement without reset
            if out == 1:
                self._check(test, state1)
                # Reset manually
                test.state.set_bit("Q0", 0)
            else:
                self._check(test, state0)

        
        # Also test no outcomes
        proj2_rep = RepTuple((None, False), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test1 = state0.copy()
        outs = test1.apply_reps_inplace([xpi2_rep, proj2_rep]*10)
        assert len(outs) == 0
        
        # Now another copy ten times at once with reset
        reset_rep = RepTuple((0, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test2 = state0.copy()
        outs = test2.apply_reps_inplace([xpi2_rep, reset_rep]*10)
        outcomes2 = outs["Q0"]
        
        # Should be same outcomes because of RNG seeding
        assert outcomes1 == outcomes2

        # Now lets test pre/post op
        idle_ptm = _ptm.rotate_x_ptm(0)
        idle_rep = RepTuple(idle_ptm, ["Q0"], GateRep.QSIM_SUPEROPERATOR)

        # Lets do X(pi/2) error before and nothing after
        pre_xpi2_rep = RepTuple(
            [0, True, xpi2_rep, idle_rep], ["Q0"], InstrumentRep.ZBASIS_PRE_POST_OPERATIONS
        )

        test3 = state0.copy()
        outs = test3.apply_reps_inplace([pre_xpi2_rep]*10)
        outcomes3 = outs["Q0"]
        assert outcomes3 == outcomes1

        # Now let's do X(pi/2) after and no nothing before
        # Very first one we have to do X(pi/2) to get same outcomes
        post_xpi2_rep = RepTuple(
            [0, True, idle_rep, xpi2_rep], ["Q0"], InstrumentRep.ZBASIS_PRE_POST_OPERATIONS
        )

        test4 = state0.copy()
        outs = test4.apply_reps_inplace([xpi2_rep] + [post_xpi2_rep]*10)
        outcomes4 = outs["Q0"]
        assert outcomes4 == outcomes1

        # Finally let's do the outcome/operation dict
        effect0 = np.array([[1, 0, 0, 0]])
        effect1 = np.array([[0, 0, 0, 1]])

        ideal_maps = {
            0: RepTuple(effect0.T @ effect0, ["Q0"], GateRep.QSIM_SUPEROPERATOR),
            1: RepTuple(effect1.T @ effect1, ["Q0"], GateRep.QSIM_SUPEROPERATOR)
        }
        ideal_map_rep = RepTuple((ideal_maps, True), ["Q0"], InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT)

        test5 = state0.copy()
        outs = test5.apply_reps_inplace([xpi2_rep, ideal_map_rep]*10)
        outcomes5 = outs["Q0"]
        assert outcomes5 == outcomes1

        # Let's use the instrument to also do reset
        reset_maps = {
            0: RepTuple(effect0.T @ effect0, ["Q0"], GateRep.QSIM_SUPEROPERATOR),
            1: RepTuple(effect0.T @ effect1, ["Q0"], GateRep.QSIM_SUPEROPERATOR)
        }
        reset_map_rep = RepTuple((reset_maps, True), ["Q0"], InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT)

        test6 = state0.copy()
        outs = test6.apply_reps_inplace([xpi2_rep, reset_map_rep]*10)
        outcomes6 = outs["Q0"]
        assert outcomes6 == outcomes1

        noisy_reset_maps = {
            0: RepTuple(xpi2_ptm @ effect0.T @ effect0, ["Q0"], GateRep.QSIM_SUPEROPERATOR),
            1: RepTuple(xpi2_ptm @ effect0.T @ effect1, ["Q0"], GateRep.QSIM_SUPEROPERATOR)
        }
        noisy_reset_map_rep = RepTuple((noisy_reset_maps, True), ["Q0"], InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT)

        test7 = state0.copy()
        outs = test7.apply_reps_inplace([xpi2_rep] + [noisy_reset_map_rep]*10)
        outcomes7 = outs["Q0"]
        assert outcomes7 == outcomes1

    def test_serialization(self, make_temp_path):
        # Start in the 10 state
        state10 = QSimState(2, ["Q0", "Q1"])
        state10.state.set_bit("Q0", 1)
        state10.state.ensure_dense("Q0")
        state10.state.ensure_dense("Q1")
        
        # Let's try a CNOT via H CZ H
        # But let's split the H CZ before serialization
        # and final H after serialization
        h_ptm = _ptm.hadamard_ptm()
        cz_ptm = state10.state._cphase_ptm

        test, _ = state10.apply_reps([RepTuple(h_ptm, ["Q1"], GateRep.QSIM_SUPEROPERATOR)])
        test.state.combine_and_apply_single_ptm("Q0") # Actually force propogation
        test.state.combine_and_apply_single_ptm("Q1") # Actually force propogation

        test.apply_reps_inplace([RepTuple(cz_ptm, ["Q0", "Q1"], GateRep.QSIM_SUPEROPERATOR)])
        # Don't force propagation here
        # So serialization should both serialize DM and operations to be applied

        with make_temp_path(suffix='.json') as tmp_path:
            test.write(tmp_path)
            test2 = QSimState.read(tmp_path)
        
        # And finish applying
        assert isinstance(test2, QSimState)
        test2.apply_reps_inplace([RepTuple(h_ptm, ["Q1"], GateRep.QSIM_SUPEROPERATOR)])
        test2.state.combine_and_apply_single_ptm("Q0") # Actually force propogation
        test2.state.combine_and_apply_single_ptm("Q1") # Actually force propogation
        
        # The expected 11 state
        state11 = state10.copy()
        state11.state.set_bit("Q1", 1)
        state11.state.ensure_dense("Q0")
        state11.state.ensure_dense("Q1")

        self._check(test2, state11)

class TestQSimQuantumStateFailedImport:
        # Mock not having the pygsti available
        def test_failed_import(self):
            with mock.patch.dict('sys.modules', {
                    'quantumsim.sparsedm': None,
                    'quantumsim.dm_np': None,
                }):

                with pytest.raises(ImportError):
                    import importlib
                    import sys

                    mod = sys.modules['loqs.backends.state.qsimstate']
                    importlib.reload(mod)
                    
