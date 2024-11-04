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
        qsim_native_ptm = _ptm.dephasing_ptm(p_depol, p_depol, p_depol)
        assert np.allclose(qsim_native_ptm, qsim_ptm)

        # Make a dummy QECCode with the circuits we want to test
        Gi_Iz_instruction = builders.build_physical_circuit_instruction(
            ListPhysicalCircuit([[("Gi", ("Q0",))], [("Iz", ("Q0",))]]),
            name="Idle and measure circuit (Z basis)"
        )
        H_Gi_H_Iz_instruction = builders.build_physical_circuit_instruction(
            ListPhysicalCircuit([[("Gh", ("Q0",))], [("Gi", ("Q0",))],[("Gh", ("Q0",))], [("Iz", ("Q0",))]]),
            name="Idle and measure circuit (X basis)"
        )

        code_1Q = QECCode({"Gi + Iz": Gi_Iz_instruction, "H + Gi + H + Iz": H_Gi_H_Iz_instruction}, ["Q0"], ["Q0"])

        ## QUANTUMSIM 
        depol_gate_dict = {
            ("Gi", ("Q0",)): qsim_native_ptm, # Depolarizing idle
            ("Gh", ("Q0",)): _ptm.hadamard_ptm()
        }
        inst_dict = {("Iz", ("Q0",)): (0, True)} # Measure and reset
        depol_noise_model = DictNoiseModel(
            (depol_gate_dict, inst_dict),
            gaterep=GateRep.QSIM_SUPEROPERATOR,
            instreps=[InstrumentRep.ZBASIS_PROJECTION]
        )

        stack_Zbasis = [
            ("Init State", None, (1,), {"qubit_labels": ["Q0"]}),
            ("Init Patch 1Q", None, ("L0", ["Q0"])),
            ("Gi + Iz", "L0")
        ]

        program_qsim = QuantumProgram(
            stack_Zbasis,
            default_noise_model=depol_noise_model,
            default_base_seed=20241104,
            state_type=QSimState,
            patch_types={"1Q": code_1Q},
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
            ("H + Gi + H + Iz", "L0")
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
            gaterep=GateRep.STIM_CIRCUIT_STR,
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




                    
