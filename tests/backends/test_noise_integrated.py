"""Integrated tests for common types of noise.

These are numerical tests intended to confirm various state backend
representations match under the correct parameters.
"""

from collections import Counter
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
from loqs.backends.circuit import ListPhysicalCircuit
from loqs.backends.model import DictNoiseModel
from loqs.backends.state import QSimQuantumState as QSimState
from loqs.backends.state import STIMQuantumState as STIMState
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

    @classmethod
    def setup_class(cls):
        # Make a dummy QECCode with the circuits we want to test
        X = builders.build_physical_circuit_instruction(
            ListPhysicalCircuit([[("Gx", ("Q0",))]]),
            name="X"
        )
        H = builders.build_physical_circuit_instruction(
            ListPhysicalCircuit([[("Gh", ("Q0",))]]),
            name="H"
        )
        I = builders.build_physical_circuit_instruction(
            ListPhysicalCircuit([[("Gi", ("Q0",))]]),
            name="Idle + amp damp"
        )
        Mz = builders.build_physical_circuit_instruction(
            ListPhysicalCircuit([[("Iz", ("Q0",))]]),
            name="Mz"
        )

        Mx = builders.build_physical_circuit_instruction(
            ListPhysicalCircuit([[("Gh", ("Q0",))], [("Iz", ("Q0",))]]),
            name="Mx"
        )

        cls.code_1Q = QECCode({"X": X, "H": H, "I": I, "Mz": Mz, "Mx": Mx}, ["Q0"], ["Q0"])

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

        ## QUANTUMSIM 
        depol_gate_dict = {
            ("Gi", ("Q0",)): qsim_native_ptm, # Depolarizing idle
            ("Gh", ("Q0",)): _ptm.hadamard_ptm() # type: ignore
        }
        inst_dict = {("Iz", ("Q0",)): (0, True)} # Measure and reset
        depol_noise_model = DictNoiseModel(
            (depol_gate_dict, inst_dict),
            gatereps=[GateRep.QSIM_SUPEROPERATOR],
            instreps=[InstrumentRep.ZBASIS_PROJECTION]
        )

        stack_Zbasis = [
            ("Init State", None, (1,), {"qubit_labels": ["Q0"]}),
            ("Init Patch 1Q", None, ("L0", ["Q0"])),
            ("I", "L0"),
            ("Mz", "L0")
        ]

        program_qsim = QuantumProgram(
            stack_Zbasis,
            default_noise_model=depol_noise_model,
            default_base_seed=20241104,
            state_type=QSimState,
            patch_types={"1Q": self.code_1Q},
            name="1Q depolarizing test"
        )

        program_qsim.run(num_shots=1000)
        outs = [mo["Q0"][0] for mo in program_qsim.collect_shot_data("measurement_outcomes", -1)]
        # Because we set the seed, we should exactly match output from test creation
        # Also, because our depol rate is 10%, we expected a flip 5% of the time (the X and Y errors)
        # For 1000 shots, this is ~50 shots should flip
        expected_outs = {0: 952, 1: 48}
        assert Counter(outs) == expected_outs

        # Also test QuantumSim in the X basis
        stack_Xbasis = [
            ("Init State", None, (1,), {"qubit_labels": ["Q0"]}),
            ("Init Patch 1Q", None, ("L0", ["Q0"])),
            ("H", "L0"),
            ("I", "L0"),
            ("H", "L0"),
            ("Mz", "L0")
        ]
        program_qsim_Xbasis = QuantumProgram.from_quantum_program(program_qsim, stack_Xbasis)

        program_qsim_Xbasis.run(num_shots=1000, reset_shot_histories=True)
        outs = [mo["Q0"][0] for mo in program_qsim_Xbasis.collect_shot_data("measurement_outcomes", -1)]
        # Now Y and Z should flip. Because rate X == rate Z and we have RNG, results should be unchanged
        assert Counter(outs) == expected_outs

        ## STIM
        # Note that for STIM, it expects the rate of each Pauli flipping,
        # which has a conversion factor of 3/4 (i.e. non-I Paulis/all Paulis)
        depol_gate_dict_stim = {
            ("Gi", ("Q0",)): f"DEPOLARIZE1({3/4 * p_depol}) 0", # Depolarizing idle
            ("Gh", ("Q0",)): "H 0"
        }
        depol_noise_model_stim = DictNoiseModel(
            (depol_gate_dict_stim, inst_dict),
            gatereps=[GateRep.STIM_CIRCUIT_STR],
            instreps=[InstrumentRep.ZBASIS_PROJECTION]
        )

        program_stim = QuantumProgram.from_quantum_program(
            program_qsim,
            state_type=STIMState,
            default_noise_model=depol_noise_model_stim
        )

        program_stim.run(num_shots=1000)
        outs = [mo["Q0"][0] for mo in program_stim.collect_shot_data("measurement_outcomes", -1)]
        # STIM handles its own RNG, so this could in principle differ from QuantumSim results
        # In practice, for this simple circuit, I've found that this coincidentally matches for Z basis
        assert Counter(outs) == expected_outs

        # Also test STIM in the X basis
        program_stim_Xbasis = QuantumProgram.from_quantum_program(program_stim, stack_Xbasis)

        program_stim_Xbasis.run(num_shots=1000)
        outs = [mo["Q0"][0] for mo in program_stim_Xbasis.collect_shot_data("measurement_outcomes", -1)]
        # But the RNG is handled differently for X basis, so we don't get a match (although it is still ~50)
        assert Counter(outs) == {0: 944, 1: 56}
    
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
        (0.15, 0.0, 20241104, 1000, [143, 591, 962]),
        # damping_rate=0.15,dephasing_rate=0.15. This IS compatible with STIM
        # Note that STIM uses more RNG so counts won't be the same
        # It is also fast so I'm running 10x the shots to boost our confidence on those
        # Only +/X changes
        # prep |+>, measure X: expect 92.5/7.5
        (0.15, 0.15, 20241104, 1000, [143, 591, 929, 1488, 5717, 9238]),
        # damping_rate=0.15,dephasing_rate=0.2. This is also compatible with STIM
        # Only +/X changes
        # prep |+>, measure X: expect 91.2/8.8
        (0.15, 0.2, 20241104, 1000, [143, 591, 917, 1488, 5747, 9116])
    ]
    @pytest.mark.parametrize("p_damp,p_dephase,seed,shots,expected0s",amp_damp_dephase_tests)
    @pytest.mark.parametrize("stim_rep",["probabilistic"])
    def test_1q_amp_damp_dephase(self, p_damp, p_dephase, seed, shots, expected0s, stim_rep):
        qsim_native_ptm = _ptm.amp_ph_damping_ptm(p_damp, p_dephase) # type: ignore

        ## QUANTUMSIM 
        damp_gate_dict = {
            ("Gi", ("Q0",)): qsim_native_ptm, # Amplitude damping idle
            ("Gh", ("Q0",)): _ptm.hadamard_ptm(), # type: ignore
            ("Gx", ("Q0",)): _ptm.rotate_x_ptm(np.pi) # type: ignore
        }
        inst_dict = {("Iz", ("Q0",)): (0, True)} # Measure and reset
        damp_noise_model = DictNoiseModel(
            (damp_gate_dict, inst_dict),
            gatereps=[GateRep.QSIM_SUPEROPERATOR],
            instreps=[InstrumentRep.ZBASIS_PROJECTION]
        )

        stack_Zbasis = [
            ("Init State", None, (1,), {"qubit_labels": ["Q0"]}),
            ("Init Patch 1Q", None, ("L0", ["Q0"])),
            ("X", "L0"),
            ("I", "L0"),
            ("Mz", "L0"),
        ]

        ## QUANTUMSIM
        program_qsim = QuantumProgram(
            stack_Zbasis,
            default_noise_model=damp_noise_model,
            default_base_seed=seed,
            state_type=QSimState,
            patch_types={"1Q": self.code_1Q},
            name="1Q damparizing test"
        )

        program_qsim.run(num_shots=shots)

        outs = [mo["Q0"][0] for mo in program_qsim.collect_shot_data("measurement_outcomes", -1)]
        assert Counter(outs)[0] == expected0s[0]

        # We can test in prep X, meas Z basis also
        stack_Zprep_Xbasis = [
            ("Init State", None, (1,), {"qubit_labels": ["Q0"]}),
            ("Init Patch 1Q", None, ("L0", ["Q0"])),
            ("H", "L0"),
            ("I", "L0"),
            ("Mz", "L0"),
        ]
        program_qsim_Xbasis = QuantumProgram.from_quantum_program(program_qsim, stack_Zprep_Xbasis)

        program_qsim_Xbasis.run(num_shots=shots)
        outs = [mo["Q0"][0] for mo in program_qsim_Xbasis.collect_shot_data("measurement_outcomes", -1)]
        assert Counter(outs)[0] == expected0s[1]

        # We can test in X basis also
        stack_Xbasis = [
            ("Init State", None, (1,), {"qubit_labels": ["Q0"]}),
            ("Init Patch 1Q", None, ("L0", ["Q0"])),
            ("H", "L0"),
            ("I", "L0"),
            ("Mx", "L0"),
        ]
        program_qsim_Xbasis = QuantumProgram.from_quantum_program(program_qsim, stack_Xbasis)

        program_qsim_Xbasis.run(num_shots=shots)
        outs = [mo["Q0"][0] for mo in program_qsim_Xbasis.collect_shot_data("measurement_outcomes", -1)]
        assert Counter(outs)[0] == expected0s[2]

        if len(expected0s) < 4:
            # Don't run STIM tests if outputs not provided
            return

        ## STIM
        kappa = 1 - (1-p_dephase)/(1-p_damp)
        pz = 0.5*(1-np.sqrt(1-kappa))
        if stim_rep == "probabilistic":
            damp_gate_dict_stim = {
                ("Gi", ("Q0",)): [
                    (f"Z_ERROR({pz}) 0", (1-p_damp)),
                    ("R 0", p_damp),
                ],
                ("Gh", ("Q0",)): "H 0",
                ("Gx", ("Q0",)): "X 0",
            }
            damp_noise_model_stim = DictNoiseModel(
                (damp_gate_dict_stim, inst_dict),
                gatereps=[GateRep.PROBABILISTIC_STIM_OPERATIONS, GateRep.STIM_CIRCUIT_STR],
                instreps=[InstrumentRep.ZBASIS_PROJECTION]
            )
        elif stim_rep == "kraus":
            # The Kraus operators for the asym dephasing/damping channel
            # See operators M in Stefan's Mathematica notebook
            Ks = [
                # Identity
                np.sqrt((1-kappa)*(1-p_damp))*np.array([[1, 0], [0, 1]]),
                # Reset
                np.sqrt((1-kappa)*p_damp)*np.array([[0, 1], [0, 0]]),
                np.sqrt((1-kappa)*p_damp)*np.array([[1, 0], [0, 0]]),
                # Dephasing
                np.sqrt(kappa)*np.array([[1, 0], [0, -1]]),
            ]


            damp_gate_dict_stim = {
                ("Gi", ("Q0",)): Ks,
                ("Gh", ("Q0",)): "H 0",
                ("Gx", ("Q0",)): "X 0",
            }
            damp_noise_model_stim = DictNoiseModel(
                (damp_gate_dict_stim, inst_dict),
                gatereps=[GateRep.KRAUS_OPERATORS, GateRep.STIM_CIRCUIT_STR],
                instreps=[InstrumentRep.ZBASIS_PROJECTION]
            )
        else:
            raise ValueError("Invalid stim rep")

        program_stim = QuantumProgram.from_quantum_program(
            program_qsim,
            state_type=STIMState,
            default_noise_model=damp_noise_model_stim
        )

        program_stim.run(num_shots=10*shots)
        outs = [mo["Q0"][0] for mo in program_stim.collect_shot_data("measurement_outcomes", -1)]
        assert Counter(outs)[0] == expected0s[3]

        program_stim_prepZ_Xbasis = QuantumProgram.from_quantum_program(program_stim, stack_Zprep_Xbasis)        
        program_stim_prepZ_Xbasis.run(num_shots=10*shots)
        outs = [mo["Q0"][0] for mo in program_stim_prepZ_Xbasis.collect_shot_data("measurement_outcomes", -1)]
        assert Counter(outs)[0] == expected0s[4]
        
        program_stim_Xbasis = QuantumProgram.from_quantum_program(program_stim, stack_Xbasis)
        program_stim_Xbasis.run(num_shots=10*shots)
        outs = [mo["Q0"][0] for mo in program_stim_Xbasis.collect_shot_data("measurement_outcomes", -1)]
        assert Counter(outs)[0] == expected0s[5]

        






                    
