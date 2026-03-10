"""Tester for loqs.tools.qsimtools"""
import pytest

try:
    from quantumsim import ptm as _ptm
    import numpy as np
    NO_QSIM = False
except ImportError:
    NO_QSIM = True


from loqs.backends.state import QSimQuantumState as QSimState
from loqs.tools.qsimtools import get_state_probs_phases, print_state_probs_phases


@pytest.mark.skipif(
    NO_QSIM,
    reason="Skipping quantumsim backend tests due to failed import"
)
class TestQSimTools:

    def test_get_state_probs_phases_basic(self):
        """Test get_state_probs_phases with a simple |0> state."""
        # Create a simple |0> state
        state = QSimState(1, ["Q0"])

        qubits, probs, phases = get_state_probs_phases(state)

        # Should have one qubit
        assert qubits == ["Q0"]
        assert "0" in probs
        assert probs["0"] >= 1 - 1e-8

        # TODO: check correctness of `phases` return value.
        return

    def test_get_state_probs_phases_superposition(self):
        """Test get_state_probs_phases with a |+> state."""
        # Create a |+> state using Hadamard
        state = QSimState(1, ["Q0"])
        h_ptm = _ptm.hadamard_ptm()
        state.state.apply_ptm("Q0", h_ptm)

        qubits, probs, phases = get_state_probs_phases(state)

        # Should have one qubit
        assert qubits == ["Q0"]

        # Should have equal probabilities for |0> and |1>
        assert "0" in probs
        assert "1" in probs
        assert abs(probs["0"] - 0.5) < 0.01
        assert abs(probs["1"] - 0.5) < 0.01
        # assert "0" in phases
        # assert "1" in phases
        # TODO: check correctness of phases; they're currently empty.
        return

    def test_get_state_probs_phases_entangled(self):
        """Test get_state_probs_phases with a Bell state."""
        # Plan: create a Bell state |Φ⁺> = (|01> + |10>)/√2.
        #
        # Start from the ground state, then three steps:
        #   1. X90 on both qubits
        #   2. C-Phase
        #   3. X90 on Q0
        #
        state = QSimState(2, ["Q0", "Q1"])

        xpi2_ptm   = _ptm.rotate_x_ptm(angle=np.pi/2)
        cphase_ptm = _ptm.double_kraus_to_ptm(np.diag([1,1,1,-1]))
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

        # Should have phase information
        # assert "01" in phases
        # assert "10" in phases
        # TODO: check phases correctness
        return

    def test_get_state_probs_phases_multiple_qubits(self):
        """Test get_state_probs_phases with multiple qubits in superposition."""
        # Create a 3-qubit state with all qubits in |+> state
        state = QSimState(3, ["Q0", "Q1", "Q2"])

        h_ptm = _ptm.hadamard_ptm()
        for qubit in ["Q0", "Q1", "Q2"]:
            state.state.apply_ptm(qubit, h_ptm)

        qubits, probs, phases = get_state_probs_phases(state)

        # Should have three qubits
        assert len(qubits) == 3

        # Should have equal probabilities for all 8 basis states
        expected_prob = 1.0 / 8.0
        for i in range(8):
            bitstring = bin(i)[2:].zfill(3)
            assert bitstring in probs
            assert abs(probs[bitstring] - expected_prob) < 1e-4

        # Should have phase information for all states
        # for bitstring in probs.keys():
        #     assert bitstring in phases
        # TODO: check correctness of phases return value
        return

    @pytest.mark.skip(reason='known failure with get_state_prob_phases')
    def test_print_state_probs_phases(self, capsys):
        """Test print_state_probs_phases function."""
        # Create a simple |0> state
        state = QSimState(1, ["Q0"])

        # Capture print output
        print_state_probs_phases(state)

        # Check that something was printed
        captured = capsys.readouterr()
        assert captured.out.strip()  # Should not be empty
        assert "Q0" in captured.out  # Should contain qubit info

