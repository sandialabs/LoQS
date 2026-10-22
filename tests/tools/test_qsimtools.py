"""Tester for loqs.tools.qsimtools"""

import pytest

# loqs.backends.state.qsimstate imports quantumsim eagerly, so this whole
# module is skipped at collection time when quantumsim is missing.
quantumsim = pytest.importorskip("quantumsim")
import numpy as np
from quantumsim import ptm as _ptm

from loqs.backends.state.qsimstate import QSimQuantumState as QSimState
from loqs.tools.qsimtools import get_state_probs_phases


class TestQSimTools:

    def test_get_state_probs_phases_superposition(self):
        """Test get_state_probs_phases with a |+> state."""
        state = QSimState(1, ["Q0"])
        h_ptm = _ptm.hadamard_ptm()
        state.state.apply_ptm("Q0", h_ptm)

        qubits, probs, phases = get_state_probs_phases(state)

        assert qubits == ["Q0"]
        assert "0" in probs
        assert "1" in probs
        assert abs(probs["0"] - 0.5) < 0.01
        assert abs(probs["1"] - 0.5) < 0.01

    def test_get_state_probs_phases_entangled(self):
        """Test get_state_probs_phases with a Bell state."""
        # Bell state |Φ⁺> = (|01> + |10>)/√2, from the ground state:
        #   1. X90 on both qubits
        #   2. C-Phase
        #   3. X90 on Q0
        state = QSimState(2, ["Q0", "Q1"])

        xpi2_ptm = _ptm.rotate_x_ptm(angle=np.pi / 2)
        cphase_ptm = _ptm.double_kraus_to_ptm(np.diag([1, 1, 1, -1]))
        state.state.apply_ptm("Q0", xpi2_ptm)
        state.state.apply_ptm("Q1", xpi2_ptm)
        state.state.apply_two_ptm("Q0", "Q1", cphase_ptm)
        state.state.apply_ptm("Q0", xpi2_ptm)

        qubits, probs, phases = get_state_probs_phases(state)
        assert len(qubits) == 2
        assert "Q0" in qubits
        assert "Q1" in qubits
        assert "01" in probs
        assert "10" in probs
        assert abs(probs["01"] - 0.5) < 1e-4
        assert abs(probs["10"] - 0.5) < 1e-4
