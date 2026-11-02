"""Tester for loqs.tools.qsimtools"""

import pytest

# loqs.backends.state.qsimstate imports quantumsim eagerly, so this whole
# module is skipped at collection time when quantumsim is missing.
quantumsim = pytest.importorskip("quantumsim")
import numpy as np
from quantumsim import ptm as _ptm

from loqs.backends.state.qsimstate import QSimQuantumState as QSimState
from loqs.tools.qsimtools import (
    get_state_probs_phases,
    print_state_probs_phases,
)


class TestQSimTools:

    def test_get_state_probs_phases_signs_linked_state(self):
        """Unequal-magnitude, single-flip-linked 2-qubit state is signed."""
        state = QSimState(2, ["Q0", "Q1"])
        state.state.apply_ptm("Q0", _ptm.rotate_y_ptm(np.pi / 3))
        state.state.apply_ptm("Q1", _ptm.rotate_y_ptm(-2 * np.pi / 3))

        qubits, probs, phases = get_state_probs_phases(state)

        assert qubits == ["Q0", "Q1"]
        expected = {"00": 0.1875, "01": 0.5625, "10": 0.0625, "11": 0.1875}
        assert probs == pytest.approx(expected, abs=1e-6)
        assert phases == {"00": 1, "01": -1, "10": 1, "11": -1}
        print_state_probs_phases(state)

    def test_get_state_probs_phases_starts_at_smallest_nonzero_string(self):
        """No |00> amplitude: the walk starts at 01 and steps 11 -> 10."""
        # 0.6|10> - 0.48|01> + 0.64|11>, by a Householder reflection of
        # |00> (QuantumSim's two-qubit index order is 00, 10, 01, 11)
        amps = np.array([0, 0.6, -0.48, 0.64])
        w = np.eye(4)[0] - amps
        u = np.eye(4) - 2 * np.outer(w, w) / (w @ w)
        state = QSimState(2, ["Q0", "Q1"])
        state.state.apply_two_ptm("Q0", "Q1", _ptm.double_kraus_to_ptm(u))

        _, _, phases = get_state_probs_phases(state)

        assert phases == {"01": 1, "10": -1, "11": -1}

    @pytest.mark.parametrize("kind", ["disconnected", "complex"])
    def test_get_state_probs_phases_refuses_unsignable_state(self, kind):
        """States the bit-flip walk can't sign raise rather than mis-sign."""
        if kind == "disconnected":
            # (|00> - |11>)/sqrt(2)
            state = QSimState(2, ["Q0", "Q1"])
            h_ptm = _ptm.hadamard_ptm()
            cz_ptm = _ptm.double_kraus_to_ptm(np.diag([1, 1, 1, -1]))
            state.state.apply_ptm("Q1", h_ptm)
            state.state.apply_ptm("Q0", h_ptm)
            state.state.apply_two_ptm("Q0", "Q1", cz_ptm)
            state.state.apply_ptm("Q1", h_ptm)
            state.state.apply_ptm("Q0", _ptm.rotate_z_ptm(np.pi))
        else:
            # (|0> - i|1>)/sqrt(2)
            state = QSimState(1, ["Q0"])
            state.state.apply_ptm("Q0", _ptm.rotate_x_ptm(np.pi / 2))

        with pytest.raises(ValueError):
            get_state_probs_phases(state)
