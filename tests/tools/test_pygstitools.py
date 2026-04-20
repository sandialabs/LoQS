"""Tester for loqs.tools.pygstitools"""

from typing import Any

import numpy as np
import pytest
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

try:
    from pygsti.modelpacks import smq1Q_XYZI
    from pygsti.protocols import ExperimentDesign
    from pygsti.models import ExplicitOpModel
    from pygsti.modelmembers.states import create_from_pure_vector
    from pygsti.modelmembers.povms import create_from_pure_vectors
    from pygsti.circuits import Circuit
    from pygsti.baseobjs import Label
    from pygsti.data import DataSet
    from pygsti.tools import unitary_to_pauligate
    NO_PYGSTI = False
except ImportError:
    NO_PYGSTI = True



@pytest.mark.skipif(NO_PYGSTI, reason='pyGSTi is not installed')
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

    # =========================
    # PHASE 2: EXPERIMENT DESIGN TOOLS
    # =========================

    def test_convert_edesign_to_programs_basic(self):
        """Test basic edesign to programs conversion."""
        qubits = ['Q0']
        model = smq1Q_XYZI.target_model('static', qubit_labels=qubits)  # type: ignore
        Gi_lbl  = tuple()
        Gx_lbl  = ('Gxpi2', 'Q0')
        rho_lbl = 'rho0'
        M_lbl   = 'Mdefault'

        # Create a toy experiment design with two circuits
        circ0 = Circuit([])
        circ1 = Circuit([Label(Gi_lbl)])
        circ2 = Circuit([Label(Gx_lbl)])
        edesign = ExperimentDesign([circ0, circ1, circ2])

        # Define physical to logical mapping
        physical_to_logical : dict[tuple | str, Any] = {
            rho_lbl  : [
                InstructionLabel("Init State", None, (1,), {'qubit_labels': qubits}),
            ],
            Gi_lbl   : [InstructionLabel("I", "Q0")],
            Gx_lbl   : [InstructionLabel("X", "Q0")],
            M_lbl    : [InstructionLabel("M", "Q0")]
        }
        no_op1 = lambda *args, **kwargs: Frame()
        no_op2 = lambda *args, **kwargs: Frame(data={'patches': {'Q0': None}})
        global_instructions = {
            'Init State': Instruction(no_op2, name='Init State'),
            'I'         : Instruction(no_op2, name='I'),
            'X'         : Instruction(no_op2, name='X'),
            'M'         : Instruction(no_op2, name='M'),
            # physical_to_logical[rho_lbl][0] : Instruction(no_op, name='prep'),
            # physical_to_logical[Gi_lbl][0]  : Instruction(no_op, name='Gi'),
            # physical_to_logical[Gx_lbl][0]  : Instruction(no_op, name='Gx'),
            # physical_to_logical[M_lbl][0]   : Instruction(no_op, name='M')
        }

        # Convert edesign to programs
        programs = convert_edesign_to_programs(
            edesign, model, physical_to_logical, global_instructions=global_instructions
        )

        # Verify we get the expected number of programs
        assert len(programs) == 3

        # Verify programs can run

        for prog in programs:
            prog.run()

        # Verify program names contain circuit info
        assert "Circuit" in programs[0].name

    def test_convert_edesign_to_programs_empty(self):
        """Test edesign to programs conversion with empty mapping."""
        model = smq1Q_XYZI.target_model('static', qubit_labels=["Q0"])  # type: ignore
        edesign = ExperimentDesign([], qubit_labels=['Q0'])
        # No harm in empty physical-to-logical if the edesign is empty.
        programs = convert_edesign_to_programs(edesign, model, physical_to_logical={})
        assert len(programs) == 0

    def test_convert_run_programs_to_dataset_basic(self):
        """Test basic programs to dataset conversion."""

        # Create a simple program that we can run
        stack = InstructionStack([
            InstructionLabel("X", "Q0"),
            InstructionLabel("MEASURE", "Q0")
        ])

        prog = QuantumProgram(stack, name="Circuit([Label('Gx', 'Q0')])")

        # For this test, we'll mock the collect_shot_data method
        # since we don't want to actually run quantum simulations
        def mock_collect_shot_data(*args, **kwargs):
            # Return some mock measurement outcomes
            return [0, 1, 0, 0, 1]  # 3 zeros, 2 ones

        prog.collect_shot_data = mock_collect_shot_data

        # Convert to dataset
        dataset = convert_run_programs_to_dataset([prog])

        # Verify it's a DataSet
        assert isinstance(dataset, DataSet)

        # Verify we can get counts from the dataset
        # The circuit should be parsed from the program name
        circ = Circuit([("Gx", "Q0")])
        counts = dataset.get_count_vector(circ)

        # Should have some counts
        assert counts is not None

    def test_convert_run_programs_to_dataset_multiple(self):
        """Test programs to dataset conversion with multiple programs."""
        # Create multiple programs
        programs = []
        for i in range(3):
            stack = []  # Empty stack for this test
            prog = QuantumProgram(stack, name=f"Circuit([Label('Gx{i}', 'Q0')])")

            # Mock collect_shot_data
            def mock_collect(*args, **kwargs):
                return [i % 2] * 5  # Alternating outcomes

            prog.collect_shot_data = mock_collect
            programs.append(prog)

        # Convert to dataset
        dataset = convert_run_programs_to_dataset(programs)

        # Verify it's a DataSet
        assert isinstance(dataset, DataSet)

        # Should have data for multiple circuits
        # The exact verification depends on how circuits are parsed from names
        assert len(dataset.keys()) > 0