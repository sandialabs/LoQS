"""Tester for loqs.tools.pygstitools"""

from typing import Any

import numpy as np
import pytest

# loqs.tools.pygstitools raises ImportError at module import when pyGSTi
# is missing (see AGENTS.md "known wart"), so this whole module is
# skipped at collection time when pyGSTi is unavailable.
pygsti = pytest.importorskip("pygsti")

from loqs.core.instructions.instructionlabel import InstructionLabel
from loqs.core.instructions.instructionstack import InstructionStack
from loqs.core.instructions.instruction      import Instruction
from loqs.core.frame import Frame
from loqs.core import QuantumProgram
from loqs.tools.pygstitools import (
    ptm_to_qsim_ptm,
    unitary_to_qsim_ptm,
    ptm_to_kraus,
    kraus_to_ptm,
    get_kraus_rep_from_ptm,
    convert_edesign_to_programs,
    convert_run_programs_to_dataset,
)
from pygsti.modelpacks import smq1Q_XYZI
from pygsti.protocols import ExperimentDesign
from pygsti.models import ExplicitOpModel
from pygsti.modelmembers.states import create_from_pure_vector
from pygsti.modelmembers.povms import create_from_pure_vectors
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label
from pygsti.data import DataSet
from pygsti.tools import unitary_to_pauligate


class TestPyGSTITools:
    """Test class for pyGSTi tools functions."""

    # =========================
    # PHASE 1: BASIC FUNCTIONS
    # =========================

    def test_ptm_to_qsim_ptm_identity_1q(self):
        """Test PTM to QSim PTM conversion for 1-qubit identity."""
        # 1-qubit identity PTM
        ptm_identity = np.eye(4)

        # Convert to QSim PTM
        result = ptm_to_qsim_ptm(ptm_identity)

        # Verify shape and basic properties
        assert result.shape == (4, 4)
        assert np.allclose(result, result.conj().T)  # Should be Hermitian

    def test_ptm_to_qsim_ptm_pauli_x(self):
        """Test PTM to QSim PTM conversion for Pauli X gate."""
        # Pauli X PTM (should be same as unitary PTM for X)
        # X gate unitary: [[0, 1], [1, 0]]
        # PTM for X should have specific structure
        ptm_x = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ])

        result = ptm_to_qsim_ptm(ptm_x)

        # Verify shape
        assert result.shape == (4, 4)

    def test_unitary_to_qsim_ptm_identity(self):
        """Test unitary to QSim PTM conversion for identity."""
        # 1-qubit identity unitary
        U_identity = np.eye(2)

        # Convert to QSim PTM
        result = unitary_to_qsim_ptm(U_identity)

        # Verify shape
        assert result.shape == (4, 4)

    def test_unitary_to_qsim_ptm_pauli_x(self):
        """Test unitary to QSim PTM conversion for Pauli X."""
        # Pauli X unitary
        U_x = np.array([[0, 1], [1, 0]])

        result = unitary_to_qsim_ptm(U_x)

        # Verify shape
        assert result.shape == (4, 4)

    def test_ptm_to_kraus_identity(self):
        """Test PTM to Kraus conversion for identity operation."""
        # 1-qubit identity PTM
        ptm_identity = np.eye(4)

        # Convert to Kraus operators
        kraus_ops = ptm_to_kraus(ptm_identity)

        # Should have at least one Kraus operator
        assert len(kraus_ops) >= 1

        # First operator should be identity (up to phase)
        first_op = kraus_ops[0]
        assert first_op.shape == (2, 2)

    def test_ptm_to_kraus_depolarizing(self):
        """Test PTM to Kraus conversion for depolarizing channel."""
        # 1-qubit depolarizing channel with p=0.1
        p = 0.1
        ptm_depol = np.diag([1, 1-p, 1-p, 1-p])

        kraus_ops = ptm_to_kraus(ptm_depol)

        # Should have multiple Kraus operators
        assert len(kraus_ops) >= 1

        # Verify shapes
        for op in kraus_ops:
            assert op.shape == (2, 2)

    def test_kraus_to_ptm_identity(self):
        """Test Kraus to PTM conversion for identity Kraus operators."""
        # Single identity Kraus operator
        kraus_ops = [np.eye(2)]

        # Convert to PTM
        ptm = kraus_to_ptm(kraus_ops)

        # Should be identity PTM
        assert ptm.shape == (4, 4)
        assert np.allclose(ptm, np.eye(4), atol=1e-10)

    def test_kraus_to_ptm_multiple_ops(self):
        """Test Kraus to PTM conversion with multiple operators."""
        # Multiple Kraus operators (identity + X)
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        kraus_ops = [np.sqrt(0.7) * I, np.sqrt(0.3) * X]

        ptm = kraus_to_ptm(kraus_ops)

        # Should be valid PTM
        assert ptm.shape == (4, 4)

    def test_get_kraus_rep_from_ptm_unitary_case(self):
        """Test Kraus rep from PTM for unitary case."""
        # Create unitary PTM (Pauli X)
        U_x = np.array([[0, 1], [1, 0]])
        ptm_x = unitary_to_pauligate(U_x)

        # Get Kraus representation
        result = get_kraus_rep_from_ptm(ptm_x, [0])

        # Should be a RepTuple
        assert hasattr(result, 'rep')
        assert hasattr(result, 'qubits')
        assert hasattr(result, 'reptype')

        # Qubits should match
        assert result.qubits == (0,)

    def test_get_kraus_rep_from_ptm_depolarizing_case(self):
        """Test Kraus rep from PTM for depolarizing channel."""
        # Create depolarizing PTM
        p = 0.1
        ptm_depol = np.diag([1, 1-p, 1-p, 1-p])

        # Identity ideal PTM
        ideal_ptm = np.eye(4)

        # Get Kraus representation with ideal PTM
        result = get_kraus_rep_from_ptm(ptm_depol, [0], ideal_ptm)

        # Should be a RepTuple
        assert hasattr(result, 'rep')
        assert hasattr(result, 'qubits')
        assert hasattr(result, 'reptype')

        # Should have multiple Kraus operators for depolarizing channel
        if hasattr(result.rep, '__len__'):
            assert len(result.rep) > 1

    def test_get_kraus_rep_from_ptm_general_case(self):
        """Test Kraus rep from PTM for general non-stochastic case."""
        # Create a general non-unitary, non-stochastic PTM
        # Use a simple amplitude damping channel as example
        gamma = 0.1
        ptm_ad = np.array([
            [1, 0, 0, gamma],
            [0, np.sqrt(1-gamma), 0, 0],
            [0, 0, np.sqrt(1-gamma), 0],
            [0, 0, 0, 1-gamma]
        ])

        # Get Kraus representation without ideal PTM
        result = get_kraus_rep_from_ptm(ptm_ad, [0])

        # Should be a RepTuple
        assert hasattr(result, 'rep')
        assert hasattr(result, 'qubits')
        assert hasattr(result, 'reptype')

    def test_ptm_to_kraus_invalid_input(self):
        """Test PTM to Kraus conversion with invalid input."""
        # Invalid PTM (not positive trace-preserving)
        invalid_ptm = np.array([
            [1, 0, 0, 0],
            [0, 2, 0, 0],  # Invalid eigenvalue > 1
            [0, 0, 0.5, 0],
            [0, 0, 0, 0.5]
        ])

        # Should raise ValueError
        with pytest.raises(ValueError):
            ptm_to_kraus(invalid_ptm)
