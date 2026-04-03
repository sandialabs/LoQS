"""Integrated tests for common types of noise.

These are numerical tests intended to confirm various state backend
representations match under the correct parameters.
"""

from collections import Counter
from scipy.stats import chisquare
import numpy as np
import pytest

try:
    from quantumsim import ptm as _ptm
    
    NO_QSIM = False
except ImportError:
    NO_QSIM = True

try:
    import pygsti
    
    NO_PYGSTI = False
except ImportError:
    NO_PYGSTI = True

try:
    import stim

    NO_STIM = False
except ImportError:
    NO_STIM = True

from loqs.backends.reps import GateRep, InstrumentRep
from loqs.backends import ListPhysicalCircuit, DictNoiseModel
from loqs.backends import QSimQuantumState as QSimState
from loqs.backends import STIMQuantumState as STIMState
from loqs.core import QuantumProgram, QECCode
from loqs.core.instructions import builders
from loqs.tools import pygstitools as pt



@pytest.mark.skipif(
    NO_QSIM,
    reason="Skipping integrated noise backend tests due to failed QuantumSim import"
)
@pytest.mark.skipif(
    NO_PYGSTI,
    reason="Skipping integrated noise backend tests due to failed pyGSTi import"
)
@pytest.mark.skipif(
    NO_STIM,
    reason="Skipping integrated noise backend tests due to failed STIM import"
)
class TestIntegratedNoise:

    def test_1q_depolarizing(self):
        p_depol = 0.1 # The chance we replace state with maximally mixed state

        # Greenbaum 1509.02921, Section 2.1.3 Example 1
        ptm = np.array([
            [1, 0, 0, 0],
            [0, 1-p_depol, 0, 0],
            [0, 0, 1-p_depol, 0],
            [0, 0, 0, 1-p_depol]
        ])

        # Get it in the QuantumSim basis
        qsim_ptm = pt.ptm_to_qsim_ptm(ptm)

        # Also create one from QuantumSim tools, and check they are the same
        qsim_native_ptm = _ptm.dephasing_ptm(p_depol, p_depol, p_depol) # type: ignore
        assert np.allclose(qsim_native_ptm, qsim_ptm) # type: ignore

        qubits = ["Q0"]
        code_1Q = self._create_code(qubits)

        ## QUANTUMSIM
        gate_dict, inst_dict = self._create_model_dicts(qubits, GateRep.QSIM_SUPEROPERATOR)
        gate_dict[("Gi", ("Q0",))] = qsim_native_ptm # Depolarizing idle
        depol_noise_model = DictNoiseModel(
            (gate_dict, inst_dict),
            gatereps=[GateRep.QSIM_SUPEROPERATOR],
            instreps=[InstrumentRep.ZBASIS_PROJECTION]
        )

        stack_Zbasis = [
            ("Init State", None, (1,), {"qubit_labels": ["Q0"]}),
            ("Init Patch 1Q", None, ("L0", ["Q0"])),
            ("I0", "L0"),
            ("Mz0", "L0")
        ]

        program_qsim = QuantumProgram(
            stack_Zbasis,
            default_noise_model=depol_noise_model,
            default_base_seed=20241104,
            state_type=QSimState,
            patch_types={"1Q": code_1Q},
            name="1Q depolarizing test"
        )

        program_results_qsim = program_qsim.run(num_shots=1000)
        outs = [mo["Q0"][0] for mo in program_results_qsim.collect_shot_data("measurement_outcomes", -1)]
        # Because we set the seed, we should exactly match output from test creation
        # Also, because our depol rate is 10%, we expected a flip 5% of the time (the X and Y errors)
        # For 1000 shots, this is ~50 shots should flip
        num_shots = 1_000
        expected_outs  = num_shots * np.array([1 - p_depol/2, p_depol/2])

        program_qsim.run(num_shots=num_shots)
        outs = [mo["Q0"][0] for mo in program_qsim.collect_shot_data("measurement_outcomes", -1)]
        assert abs(Counter(outs)[0] - 950) < 10
        assert abs(Counter(outs)[1] - 50) < 10

        # Also test QuantumSim in the X basis
        stack_Xbasis = [
            ("Init State", None, (1,), {"qubit_labels": ["Q0"]}),
            ("Init Patch 1Q", None, ("L0", ["Q0"])),
            ("H0", "L0"),
            ("I0", "L0"),
            ("H0", "L0"),
            ("Mz0", "L0")
        ]
        program_qsim_Xbasis = QuantumProgram.from_quantum_program(program_qsim, stack_Xbasis)

        program_results_qsim_Xbasis = program_qsim_Xbasis.run(num_shots=1000)
        outs = [mo["Q0"][0] for mo in program_results_qsim_Xbasis.collect_shot_data("measurement_outcomes", -1)]
        # Now Y and Z should flip. Because rate X == rate Z and we have RNG, results should be unchanged
        assert abs(Counter(outs)[0] - 950) < 10
        assert abs(Counter(outs)[1] - 50) < 10

        ## STIM
        gate_dict, inst_dict = self._create_model_dicts(qubits, GateRep.STIM_CIRCUIT_STR)
        # Note that for STIM, it expects the rate of each Pauli flipping,
        # which has a conversion factor of 3/4 (i.e. non-I Paulis/all Paulis)
        gate_dict[("Gi", ("Q0",))] = f"DEPOLARIZE1({3/4 * p_depol}) 0" # Depolarizing idle
        depol_noise_model_stim = DictNoiseModel(
            (gate_dict, inst_dict),
            gatereps=[GateRep.STIM_CIRCUIT_STR],
            instreps=[InstrumentRep.ZBASIS_PROJECTION]
        )

        # STIM handles its own RNG, so this could in principle can differ from QuantumSim results
        program_stim = QuantumProgram.from_quantum_program(
            program_qsim,
            state_type=STIMState,
            default_noise_model=depol_noise_model_stim
        )

        program_results_stim = program_stim.run(num_shots=1000)
        outs = [mo["Q0"][0] for mo in program_results_stim.collect_shot_data("measurement_outcomes", -1)]
        # STIM handles its own RNG, so this could in principle differ from QuantumSim results
        # In practice, for this simple circuit, I've found that this coincidentally matches for Z basis
        assert abs(Counter(outs)[0] - 950) < 10
        assert abs(Counter(outs)[1] - 50) < 10

        # Also test STIM in the X basis
        program_stim_Xbasis = QuantumProgram.from_quantum_program(program_stim, stack_Xbasis)

        program_results_stim_Xbasis = program_stim_Xbasis.run(num_shots=1000)
        outs = [mo["Q0"][0] for mo in program_results_stim_Xbasis.collect_shot_data("measurement_outcomes", -1)]
        # But the RNG is handled differently for X basis, so we don't get a match (although it is still ~50)
        assert abs(Counter(outs)[0] - 950) < 10
        assert abs(Counter(outs)[1] - 50) < 10
    
    # Amp damp tests, given as damping rate, dephasing rate, seed, num shots, and then expected 1 counts in order of:
    # QuantumSim X,I,Mz; QuantumSim H,I,Mz; QuantumSim H,I,Mx; STIM X,I,Mz; STIM H,I,Mz; STIM H,I,Mx
    # Recall that e^{-t/T1} = 1-damping_rate and
    # e^{-t/T2} = sqrt([1-damping_rate][1-dephase_rate])
    # Physical rates often have T1 > T2, so damping_rate < dephase_rate
    # If T1 = T2, then damping rate = dephase rate
    # If dephase rate = 0, this is technically not a quantum channel (I think)
    # For full details on this, see Stefan's 2024-11-14 Mathematica notebook
    amp_damp_dephase_tests = [
        # damping_rate=0.15,dephasing_rate=0. This does not leave stabilizer states as stabilizer states, so not STIM compatible
        # prep |1>, measure Z: expect 15/85
        # prep |+>, measure Z: expect 57.5/42.5
        # prep |+>, measure X: expect 96.1/3.9
        # Skip STIM for these
        (0.15, 0.0, 20241104, 100, [15, 58, 96], False),
        # damping_rate=0.15,dephasing_rate=0.15. This IS compatible with STIM
        # Note that STIM uses more RNG so counts won't be the same
        # Annoyingly, I've found this also differs on machines, so STIM tests must just be within 10
        # It is also fast so I'm running 10x the shots to boost our confidence on those
        # Only +/X changes
        # prep |+>, measure X: expect 92.5/7.5
        (0.15, 0.15, 20241104, 100, [15, 58, 93], True),
        # damping_rate=0.15,dephasing_rate=0.2. This is also compatible with STIM
        # Only +/X changes
        # prep |+>, measure X: expect 91.2/8.8
        (0.15, 0.2, 20241104, 100, [15, 58, 91], True)
    ]
    @pytest.mark.parametrize("p_damp,p_dephase,seed,shots,expected0s,run_stim",amp_damp_dephase_tests)
    @pytest.mark.parametrize("stim_rep",[GateRep.PROBABILISTIC_STIM_OPERATIONS])
    def test_amp_damp_dephase(self, p_damp, p_dephase, seed, shots, expected0s, run_stim, stim_rep):
        # I will test on 3 qubits, with qubit 0 in |1> and qubit 2 in |+>
        # and qubit 1 being the one being damped/dephased
        # We should leave qubits 0 and 2 untouched
        # This is mostly to test the STIM Kraus reduced statevector code
        qubits = ["Q0", "Q1", "Q2"]
        code_3Q = self._create_code(qubits)

        ## QUANTUMSIM 
        qsim_native_ptm = _ptm.amp_ph_damping_ptm(p_damp, p_dephase) # type: ignore
        gate_dict, inst_dict = self._create_model_dicts(qubits, GateRep.QSIM_SUPEROPERATOR)
        gate_dict[("Gi", ("Q1",))] = qsim_native_ptm # Amplitude damping/dephasing idle
        noise_model = DictNoiseModel(
            (gate_dict, inst_dict),
            gatereps=[GateRep.QSIM_SUPEROPERATOR],
            instreps=[InstrumentRep.ZBASIS_PROJECTION]
        )

        stack_Zbasis = [
            ("Init State", None, (len(qubits),), {"qubit_labels": qubits}),
            ("Init Patch 3Q", None, ("L0", qubits)),
            ("X0", "L0"), # Start qubit 0 in 1
            ("H2", "L0"), # Start qubit 2 in +
            ("X1", "L0"),
            ("I1", "L0"),
            ("Mz1", "L0"),
            ("Mz0", "L0"), # Verify qubit 0 still in 1 in Z basis
            ("Mx2", "L0"), # Verify qubit 1 still in 0 in X basis
        ]

        ## QUANTUMSIM
        program_qsim = QuantumProgram(
            stack_Zbasis,
            default_noise_model=noise_model,
            default_base_seed=seed,
            state_type=QSimState,
            patch_types={"3Q": code_3Q},
            name="Amp damp/dephasing test"
        )

        program_results_qsim = program_qsim.run(num_shots=shots)

        def check(program_results, expected):
            outs = [mo["Q1"][0] for mo in program_results.collect_shot_data("measurement_outcomes", -3)]
            assert abs(Counter(outs)[0]-expected) < 10
            # Verify side qubits unaffected
            outs = [mo["Q0"][0] for mo in program_results.collect_shot_data("measurement_outcomes", -2)]
            assert Counter(outs)[0] == 0
            outs = [mo["Q2"][0] for mo in program_results.collect_shot_data("measurement_outcomes", -1)]
            assert Counter(outs)[0] == shots
        
        check(program_results_qsim, expected0s[0])

        # We can test in prep X, meas Z basis also
        stack_Zprep_Xbasis = [
            ("Init State", None, (len(qubits),), {"qubit_labels": qubits}),
            ("Init Patch 3Q", None, ("L0", qubits)),
            ("X0", "L0"), # Start qubit 0 in 1
            ("H2", "L0"), # Start qubit 2 in +
            ("H1", "L0"),
            ("I1", "L0"),
            ("Mz1", "L0"),
            ("Mz0", "L0"), # Verify qubit 0 still in 1 in Z basis
            ("Mx2", "L0"), # Verify qubit 1 still in 0 in X basis
        ]
        program_qsim_Zprep_Xbasis = QuantumProgram.from_quantum_program(program_qsim, stack_Zprep_Xbasis)

        program_results_qsim_Zprep_Xbasis = program_qsim_Zprep_Xbasis.run(num_shots=shots)
        check(program_results_qsim_Zprep_Xbasis, expected0s[1])

        # We can test in X basis also
        stack_Xbasis = [
            ("Init State", None, (len(qubits),), {"qubit_labels": qubits}),
            ("Init Patch 3Q", None, ("L0", qubits)),
            ("X0", "L0"), # Start qubit 0 in 1
            ("H2", "L0"), # Start qubit 2 in +
            ("H1", "L0"),
            ("I1", "L0"),
            ("Mx1", "L0"),
            ("Mz0", "L0"), # Verify qubit 0 still in 1 in Z basis
            ("Mx2", "L0"), # Verify qubit 1 still in 0 in X basis
        ]
        program_qsim_Xbasis = QuantumProgram.from_quantum_program(program_qsim, stack_Xbasis)

        program_results_qsim_Xbasis = program_qsim_Xbasis.run(num_shots=shots)
        check(program_results_qsim_Xbasis, expected0s[2])

        if not run_stim:
            # Don't run STIM tests if outputs not provided
            return

        ## STIM
        kappa = 1 - (1-p_dephase)/(1-p_damp)
        pz = 0.5*(1-np.sqrt(1-kappa))
        if stim_rep == GateRep.PROBABILISTIC_STIM_OPERATIONS:
            damp_rep = [
                (f"Z_ERROR({pz}) 0", (1-p_damp)),
                ("R 0", p_damp),
            ]
        elif stim_rep == GateRep.KRAUS_OPERATORS:
            # The Kraus operators for the asym dephasing/damping channel
            # See operators M in Stefan's Mathematica notebook
            damp_rep = [
                # Identity
                np.sqrt((1-pz)*(1-p_damp))*np.array([[1, 0], [0, 1]]),
                # Reset
                np.sqrt((1-pz)*p_damp)*np.array([[0, 1], [0, 0]]),
                np.sqrt((1-pz)*p_damp)*np.array([[1, 0], [0, 0]]),
                # Dephasing
                np.sqrt(pz)*np.array([[1, 0], [0, -1]]),
            ]
        else:
            raise ValueError(f"Invalid stim rep {stim_rep} for testing")

        gate_dict, inst_dict = self._create_model_dicts(qubits, GateRep.STIM_CIRCUIT_STR)
        gate_dict[("Gi", ("Q1",))] = damp_rep
        noise_model_stim = DictNoiseModel(
            (gate_dict, inst_dict),
            gatereps=[stim_rep, GateRep.STIM_CIRCUIT_STR],
            instreps=[InstrumentRep.ZBASIS_PROJECTION]
        )
        program_stim = QuantumProgram.from_quantum_program(
            program_qsim,
            state_type=STIMState,
            default_noise_model=noise_model_stim
        )

        program_results_stim = program_stim.run(num_shots=shots)
        check(program_results_stim, expected0s[0])

        program_stim_Zprep_Xbasis = QuantumProgram.from_quantum_program(program_stim, stack_Zprep_Xbasis)
        program_results_stim_Zprep_Xbasis = program_stim_Zprep_Xbasis.run(num_shots=shots)
        check(program_results_stim_Zprep_Xbasis, expected0s[1])
        
        program_stim_Xbasis = QuantumProgram.from_quantum_program(program_stim, stack_Xbasis)
        program_results_stim_Xbasis = program_stim_Xbasis.run(num_shots=shots)
        check(program_results_stim_Xbasis, expected0s[2])
    
    @staticmethod
    def _create_code(qubits: list[str]):
        # Make a dummy QECCode with the circuits we want to test
        instructions = {}
        for q in qubits:
            instructions[f"X{q[1]}"] = builders.build_physical_circuit_instruction(
                ListPhysicalCircuit([[("Gx", (q,))]]),
                name=f"X on {q}"
            )
            instructions[f"H{q[1]}"] = builders.build_physical_circuit_instruction(
                ListPhysicalCircuit([[("Gh", (q,))]]),
                name=f"H on {q}"
            )
            instructions[f"I{q[1]}"] = builders.build_physical_circuit_instruction(
                ListPhysicalCircuit([[("Gi", (q,))]]),
                name=f"Noisy I on {q}"
            )
            instructions[f"Mz{q[1]}"] = builders.build_physical_circuit_instruction(
                ListPhysicalCircuit([[("Iz", (q,))]]),
                name=f"Mz on {q}"
            )
            instructions[f"Mx{q[1]}"] = builders.build_physical_circuit_instruction(
                ListPhysicalCircuit([[("Gh", (q,))],[("Iz", (q,))]]),
                name=f"Mx on {q}"
            )

        return QECCode(instructions, qubits, qubits)

    @staticmethod
    def _create_model_dicts(qubits: list[str], gaterep: GateRep):
        assert gaterep in [GateRep.QSIM_SUPEROPERATOR, GateRep.STIM_CIRCUIT_STR]

        gate_dict = {}
        if gaterep == GateRep.QSIM_SUPEROPERATOR:
            for q in qubits:
                gate_dict[('Gi', (q,))] = np.eye(4)
                gate_dict[('Gh', (q,))] = _ptm.hadamard_ptm() # type: ignore
                gate_dict[("Gx", (q,))] = _ptm.rotate_x_ptm(np.pi) # type: ignore
        else:
            for q in qubits:
                gate_dict[('Gi', (q,))] = "I 0"
                gate_dict[('Gh', (q,))] = "H 0"
                gate_dict[('Gx', (q,))] = "X 0"

        inst_dict = {("Iz", (q,)): (0, True) for q in qubits}

        return gate_dict, inst_dict
