"""Tester for loqs.backends.state.qsimstate"""

import os
import tempfile
import json

import mock
import numpy as np
import pytest

quantumsim = pytest.importorskip("quantumsim")
from quantumsim.sparsedm import SparseDM as _SparseDM
from quantumsim import ptm as _ptm

from loqs.backends.reps import (
    KrausGateRep,
    ProbabilisticStimGateRep,
    PTMGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    OutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.backends import QSimQuantumState as QSimState

class TestQSimQuantumState:

    def _check(self, state, expected_state):
        assert state.state.names == expected_state.state.names
        # assert state.seed == expected_state.seed  # seed is no longer serialized
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

        # This one won't have same labels
        s6 = QSimState(5)
        self._check(QSimState(5), s6)

        # Copy check
        s7 = s.copy()
        self._check(s7, s)
    
    def test_apply_gates(self):
        # Let's apply a X gate
        x_ptm = _ptm.rotate_x_ptm(np.pi)
        X_reps = [QSimSuperopGateRep(x_ptm, ["Q0"])]

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
            QSimSuperopGateRep(h_ptm, ["Q1"]),
            QSimSuperopGateRep(cz_ptm, ["Q0", "Q1"]),
            QSimSuperopGateRep(h_ptm, ["Q1"])
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
                UnitaryGateRep(np.eye(2), "Q0")
            ])
        
        with pytest.raises(NotImplementedError):
            test.apply_reps([
                PTMGateRep(np.eye(4), "Q0")
            ])
        
        with pytest.raises(NotImplementedError):
            test.apply_reps([
                StimCircuitGateRep("I 0", "Q0")
            ])

        # Try to pass in too many qubits
        with pytest.raises(ValueError):
            test.apply_reps([
                QSimSuperopGateRep(np.eye(64), ["Q0", "Q1", "Q2"])
            ])

        with pytest.raises(NotImplementedError):
            test.apply_reps([
                KrausGateRep([(np.eye(2), 1.0)], "Q0")
            ])

        with pytest.raises(NotImplementedError):
            test.apply_reps([
                ProbabilisticStimGateRep([("X 0", 1.0)], "Q0")
            ])

    def test_input_reps(self):
        state = QSimState(1, ["Q0"])
        assert set(state.input_reps) == {
            QSimSuperopGateRep,
            ZBasisProjectionInstrumentRep,
            ZBasisPrePostInstrumentRep,
            OutcomeOperationDictInstrumentRep,
        }

    def test_unsupported_instrument_rep_raises(self):
        test = QSimState(1, ["Q0"])
        with pytest.raises(NotImplementedError):
            test.apply_reps_inplace([
                StimCircuitInstrumentRep("M 0", ["Q0"])
            ])

    def test_apply_instruments(self):
        # H gate to get + state for testing
        xpi2_ptm = _ptm.rotate_x_ptm(np.pi/2)
        xpi2_rep = QSimSuperopGateRep(xpi2_ptm, ["Q0"])

        state0 = QSimState(1, ["Q0"], seed=20241016)

        state1 = QSimState(1, ["Q0"], seed=20241016)
        state1.state.set_bit("Q0", 1)
        # This time, keep it classical since we are going to measure

        # In-place 10 times
        # Also test no reset
        proj_rep = ZBasisProjectionInstrumentRep(None, True, ["Q0"])
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
        proj2_rep = ZBasisProjectionInstrumentRep(None, False, ["Q0"])
        test1 = state0.copy()
        outs = test1.apply_reps_inplace([xpi2_rep, proj2_rep]*10)
        assert len(outs) == 0
        
        # Now another copy ten times at once with reset
        reset_rep = ZBasisProjectionInstrumentRep(0, True, ["Q0"])
        test2 = state0.copy()
        outs = test2.apply_reps_inplace([xpi2_rep, reset_rep]*10)
        outcomes2 = outs["Q0"]
        
        # Should be same outcomes because of RNG seeding
        assert outcomes1 == outcomes2

        # Now lets test pre/post op
        idle_ptm = _ptm.rotate_x_ptm(0)
        idle_rep = QSimSuperopGateRep(idle_ptm, ["Q0"])

        # Lets do X(pi/2) error before and nothing after
        pre_xpi2_rep = ZBasisPrePostInstrumentRep(0, True, xpi2_rep, idle_rep, ["Q0"])

        test3 = state0.copy()
        outs = test3.apply_reps_inplace([pre_xpi2_rep]*10)
        outcomes3 = outs["Q0"]
        assert outcomes3 == outcomes1

        # Now let's do X(pi/2) after and no nothing before
        # Very first one we have to do X(pi/2) to get same outcomes
        post_xpi2_rep = ZBasisPrePostInstrumentRep(0, True, idle_rep, xpi2_rep, ["Q0"])

        test4 = state0.copy()
        outs = test4.apply_reps_inplace([xpi2_rep] + [post_xpi2_rep]*10)
        outcomes4 = outs["Q0"]
        assert outcomes4 == outcomes1

        # Finally let's do the outcome/operation dict
        effect0 = np.array([[1, 0, 0, 0]])
        effect1 = np.array([[0, 0, 0, 1]])

        ideal_maps = {
            0: QSimSuperopGateRep(effect0.T @ effect0, ["Q0"]),
            1: QSimSuperopGateRep(effect1.T @ effect1, ["Q0"])
        }
        ideal_map_rep = OutcomeOperationDictInstrumentRep(ideal_maps, True, ["Q0"])

        test5 = state0.copy()
        outs = test5.apply_reps_inplace([xpi2_rep, ideal_map_rep]*10)
        outcomes5 = outs["Q0"]
        assert outcomes5 == outcomes1

        # Let's use the instrument to also do reset
        reset_maps = {
            0: QSimSuperopGateRep(effect0.T @ effect0, ["Q0"]),
            1: QSimSuperopGateRep(effect0.T @ effect1, ["Q0"])
        }
        reset_map_rep = OutcomeOperationDictInstrumentRep(reset_maps, True, ["Q0"])

        test6 = state0.copy()
        outs = test6.apply_reps_inplace([xpi2_rep, reset_map_rep]*10)
        outcomes6 = outs["Q0"]
        assert outcomes6 == outcomes1

        noisy_reset_maps = {
            0: QSimSuperopGateRep(xpi2_ptm @ effect0.T @ effect0, ["Q0"]),
            1: QSimSuperopGateRep(xpi2_ptm @ effect0.T @ effect1, ["Q0"])
        }
        noisy_reset_map_rep = OutcomeOperationDictInstrumentRep(noisy_reset_maps, True, ["Q0"])

        test7 = state0.copy()
        outs = test7.apply_reps_inplace([xpi2_rep] + [noisy_reset_map_rep]*10)
        outcomes7 = outs["Q0"]
        assert outcomes7 == outcomes1

    def test_zbasis_projection_reset_to_1(self):
        xpi2_ptm = _ptm.rotate_x_ptm(np.pi / 2)
        xpi2_rep = QSimSuperopGateRep(xpi2_ptm, ["Q0"])
        reset1_rep = ZBasisProjectionInstrumentRep(1, True, ["Q0"])

        state1 = QSimState(1, ["Q0"], seed=20260712)
        state1.state.set_bit("Q0", 1)
        state1.state.ensure_dense("Q0")

        test = QSimState(1, ["Q0"], seed=20260712)
        for _ in range(10):
            test.apply_reps_inplace([xpi2_rep, reset1_rep])
            test.state.combine_and_apply_single_ptm("Q0")
            test.state.ensure_dense("Q0")
            self._check(test, state1)

    def test_zbasis_projection_multiqubit(self):
        """A single ZBASIS_PROJECTION rep applied to multiple qubits at
        once must measure/reset every qubit in `qubits`."""
        xpi2_ptm = _ptm.rotate_x_ptm(np.pi / 2)
        xpi2_reps = [
            QSimSuperopGateRep(xpi2_ptm, ["Q0"]),
            QSimSuperopGateRep(xpi2_ptm, ["Q1"]),
        ]
        reset0_rep = ZBasisProjectionInstrumentRep(0, True, ["Q0", "Q1"])

        state00 = QSimState(2, ["Q0", "Q1"])
        state00.state.ensure_dense("Q0")
        state00.state.ensure_dense("Q1")

        test = QSimState(2, ["Q0", "Q1"], seed=20260712)
        for _ in range(10):
            outs = test.apply_reps_inplace(xpi2_reps + [reset0_rep])
            assert set(outs.keys()) == {"Q0", "Q1"}
            assert len(outs["Q0"]) == 1 and len(outs["Q1"]) == 1
            test.state.combine_and_apply_single_ptm("Q0")
            test.state.combine_and_apply_single_ptm("Q1")
            test.state.ensure_dense("Q0")
            test.state.ensure_dense("Q1")
            assert np.allclose(test.state.full_dm.dm, state00.state.full_dm.dm)

    def test_zbasis_outcome_operation_dict_multiqubit_raises(self):
        """ZBASIS_OUTCOME_OPERATION_DICT explicitly does not support more
        than one qubit in this backend. `outcome_qubits` is given here as a
        single joint classical label (matching a genuine parity-check
        instrument's shape) purely so construction itself succeeds --
        `qsimstate.py`'s own 1-qubit restriction is what's under test."""
        dummy_maps = {
            "even": QSimSuperopGateRep(np.eye(4), ["Q0"]),
            "odd": QSimSuperopGateRep(np.eye(4), ["Q0"]),
        }
        rep = OutcomeOperationDictInstrumentRep(
            dummy_maps, True, ["Q0", "Q1"], outcome_qubits="synd"
        )
        test = QSimState(2, ["Q0", "Q1"])
        with pytest.raises(NotImplementedError):
            test.apply_reps_inplace([rep])

    def test_zbasis_outcome_operation_dict_no_outcomes(self):
        """`include_outcomes=False` for ZBASIS_OUTCOME_OPERATION_DICT must
        suppress the outcome dict entry, while the ideal projector map
        must still always collapse to an exact computational basis state."""
        xpi2_ptm = _ptm.rotate_x_ptm(np.pi / 2)
        xpi2_rep = QSimSuperopGateRep(xpi2_ptm, ["Q0"])

        effect0 = np.array([[1, 0, 0, 0]])
        effect1 = np.array([[0, 0, 0, 1]])
        ideal_maps = {
            0: QSimSuperopGateRep(effect0.T @ effect0, ["Q0"]),
            1: QSimSuperopGateRep(effect1.T @ effect1, ["Q0"]),
        }
        ideal_map_rep_no_outcomes = OutcomeOperationDictInstrumentRep(ideal_maps, False, ["Q0"])

        state0 = QSimState(1, ["Q0"])
        state0.state.ensure_dense("Q0")
        state1 = QSimState(1, ["Q0"])
        state1.state.set_bit("Q0", 1)
        state1.state.ensure_dense("Q0")

        test = QSimState(1, ["Q0"], seed=20260712)
        for _ in range(10):
            outs = test.apply_reps_inplace([xpi2_rep, ideal_map_rep_no_outcomes])
            assert len(outs) == 0
            test.state.combine_and_apply_single_ptm("Q0")
            matches0 = np.allclose(test.state.full_dm.dm, state0.state.full_dm.dm)
            matches1 = np.allclose(test.state.full_dm.dm, state1.state.full_dm.dm)
            assert matches0 or matches1

    def test_zbasis_pre_post_operations_reset_and_no_outcomes(self):
        """ZBASIS_PRE_POST_OPERATIONS with `reset=1` and
        `include_outcomes=False` must suppress the outcome dict entry
        while still always leaving the qubit reset to |1>."""
        xpi2_ptm = _ptm.rotate_x_ptm(np.pi / 2)
        xpi2_rep = QSimSuperopGateRep(xpi2_ptm, ["Q0"])
        idle_ptm = _ptm.rotate_x_ptm(0)
        idle_rep = QSimSuperopGateRep(idle_ptm, ["Q0"])

        pre_reset1_no_outcomes = ZBasisPrePostInstrumentRep(1, False, xpi2_rep, idle_rep, ["Q0"])

        state1 = QSimState(1, ["Q0"], seed=20260712)
        state1.state.set_bit("Q0", 1)
        state1.state.ensure_dense("Q0")

        test = QSimState(1, ["Q0"], seed=20260712)
        for _ in range(10):
            outs = test.apply_reps_inplace([pre_reset1_no_outcomes])
            assert len(outs) == 0
            test.state.combine_and_apply_single_ptm("Q0")
            test.state.ensure_dense("Q0")
            self._check(test, state1)

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

        test, _ = state10.apply_reps([QSimSuperopGateRep(h_ptm, ["Q1"])])
        test.state.combine_and_apply_single_ptm("Q0") # Actually force propogation
        test.state.combine_and_apply_single_ptm("Q1") # Actually force propogation

        test.apply_reps_inplace([QSimSuperopGateRep(cz_ptm, ["Q0", "Q1"])])
        # Don't force propagation here
        # So serialization should both serialize DM and operations to be applied

        with make_temp_path(suffix='.json') as tmp_path:
            test.write(tmp_path)
            test2 = QSimState.read(tmp_path)
        
        # And finish applying
        assert isinstance(test2, QSimState)
        test2.apply_reps_inplace([QSimSuperopGateRep(h_ptm, ["Q1"])])
        test2.state.combine_and_apply_single_ptm("Q0") # Actually force propogation
        test2.state.combine_and_apply_single_ptm("Q1") # Actually force propogation
        
        # The expected 11 state
        state11 = state10.copy()
        state11.state.set_bit("Q1", 1)
        state11.state.ensure_dense("Q0")
        state11.state.ensure_dense("Q1")

        self._check(test2, state11)

    def test_serialization_after_instrument(self, make_temp_path):
        """Serialization must round-trip correctly after applying an
        InstrumentRep, not just a plain gate-only circuit."""
        xpi2_ptm = _ptm.rotate_x_ptm(np.pi / 2)
        xpi2_rep = QSimSuperopGateRep(xpi2_ptm, ["Q0"])
        reset_rep = ZBasisProjectionInstrumentRep(0, True, ["Q0"])

        test = QSimState(1, ["Q0"], seed=20260712)
        outs_before = test.apply_reps_inplace([xpi2_rep, reset_rep])
        test.state.combine_and_apply_single_ptm("Q0")
        test.state.ensure_dense("Q0")

        with make_temp_path(suffix=".json") as tmp_path:
            test.write(tmp_path)
            test2 = QSimState.read(tmp_path)

        assert isinstance(test2, QSimState)
        # assert test2.seed == 20260712  # seed is no longer serialized
        # X(pi/2) gives a random measurement outcome, but reset=0 must
        # always leave the qubit in |0> regardless -- both before and
        # after the round trip.
        assert len(outs_before["Q0"]) == 1
        state0 = QSimState(1, ["Q0"], seed=20260712)
        state0.state.ensure_dense("Q0")
        self._check(test2, state0)

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
                    
