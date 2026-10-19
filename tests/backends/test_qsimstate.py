"""Tester for loqs.backends.state.qsimstate.

Behaviour shared with the other state backends is in test_states.py.
"""

import numpy as np
import pytest

quantumsim = pytest.importorskip("quantumsim")
from quantumsim.sparsedm import SparseDM as _SparseDM
from quantumsim import ptm as _ptm

from loqs.backends.reps import (
    OutcomeOperationDictInstrumentRep,
    QSimSuperopGateRep,
)
from loqs.backends import QSimQuantumState as QSimState


class TestQSimQuantumState:

    def _check(self, state, expected_state):
        assert state.state.names == expected_state.state.names
        assert np.allclose(
            state.state.full_dm.dm, expected_state.state.full_dm.dm
        )

    def test_init(self):
        qubit_labels = [f"Q{i}" for i in range(5)]
        s = QSimState(5, qubit_labels)

        # A SparseDM can be passed directly
        qsim_dm = _SparseDM(qubit_labels)
        s3 = QSimState(qsim_dm)
        self._check(s3, s)

    def test_apply_gates_too_many_qubits_raises(self):
        test = QSimState(3, ["Q0", "Q1", "Q2"])
        with pytest.raises(ValueError):
            test.apply_reps(
                [QSimSuperopGateRep(np.eye(64), ["Q0", "Q1", "Q2"])]
            )

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

    def test_pickle_round_trip(self):
        """The state opts into the encoding-attribute pickle bridge, which
        its GPU density class needs to cross a process boundary."""
        import pickle

        from loqs.internal.serializable import _rebuild_from_encoding_attrs

        state = QSimState(2, ["Q0", "Q1"])
        state.apply_reps_inplace(
            [
                QSimSuperopGateRep(_ptm.rotate_x_ptm(np.pi / 2), ["Q0"]),
                QSimSuperopGateRep(_ptm.rotate_y_ptm(np.pi / 3), ["Q1"]),
            ]
        )

        reduced = state.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
        assert reduced[0] is _rebuild_from_encoding_attrs

        state2 = pickle.loads(pickle.dumps(state))
        assert state2.state.names == state.state.names
        assert np.allclose(
            state2.state.full_dm.to_array(), state.state.full_dm.to_array()
        )
        assert state2.state.classical == state.state.classical
