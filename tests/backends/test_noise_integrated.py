"""Integrated tests for common types of noise.

These are numerical tests intended to confirm various state backend
representations match under the correct parameters.
"""

from collections import Counter
import numpy as np
import pytest

quantumsim = pytest.importorskip("quantumsim")
from quantumsim import ptm as _ptm

pygsti = pytest.importorskip("pygsti")

stim = pytest.importorskip("stim")

from loqs.backends.reps import (
    GateRep,
    PTMGateRep,
    ProbabilisticStimGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    UnitaryGateRep,
    ZBasisProjectionInstrumentRep,
    convert as convert_rep,
)
from loqs.backends import ListPhysicalCircuit, DictNoiseModel
from loqs.backends import NumpyStatevectorQuantumState
from loqs.backends import QSimQuantumState as QSimState
from loqs.backends import STIMQuantumState as STIMState
from loqs.core import Frame, QuantumProgram, QECCode, InstructionStack
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import Instruction
from loqs.core.instructions.instructionlabel import InstructionLabel


class TestIntegratedNoise:

    def test_1q_depolarizing(self):
        p_depol = 0.4  # The chance we replace state with maximally mixed state

        # Greenbaum 1509.02921, Section 2.1.3 Example 1
        ptm = np.array(
            [
                [1, 0, 0, 0],
                [0, 1 - p_depol, 0, 0],
                [0, 0, 1 - p_depol, 0],
                [0, 0, 0, 1 - p_depol],
            ]
        )

        # Get it in the QuantumSim basis
        qsim_ptm = convert_rep(
            PTMGateRep(ptm, ["Q0"]), QSimSuperopGateRep
        ).superop

        # Also create one from QuantumSim tools, and check they are the same
        qsim_native_ptm = _ptm.dephasing_ptm(p_depol, p_depol, p_depol)  # type: ignore
        assert np.allclose(qsim_native_ptm, qsim_ptm)  # type: ignore

        qubits = ["Q0"]
        code_1Q = self._create_code(qubits)
        shots = 100
        # X and Y errors flip a Z readout in p_depol / 2 of the shots (sigma
        # 4); the tolerance is 3 sigma, and no noise at all is 5 sigma away.
        expected_flips = 20
        tolerance = 12

        def stack(basis_gates):
            return InstructionStack(
                [
                    {
                        "instruction": "Init State",
                        "state": 1,
                        "qubit_labels": ["Q0"],
                    },
                    {
                        "instruction": "Init Patch 1Q",
                        "new_patch_label": "L0",
                        "qubits": ["Q0"],
                    },
                    *[(g, "L0") for g in basis_gates],
                    ("I0", "L0"),
                    *[(g, "L0") for g in basis_gates],
                    ("Mz0", "L0"),
                ]
            )

        def flips(results):
            outs = [
                mo["Q0"][0]
                for mo in results.collect_shot_data("measurement_outcomes", -1)
            ]
            return Counter(outs)[1]

        ## QUANTUMSIM, Z basis
        gate_dict, inst_dict = self._create_model_dicts(
            qubits, QSimSuperopGateRep
        )
        gate_dict[("Gi", ("Q0",))] = qsim_native_ptm  # Depolarizing idle
        depol_noise_model = DictNoiseModel(
            gate_dict,
            inst_dict,
            gatereps=[QSimSuperopGateRep],
            instreps=[ZBasisProjectionInstrumentRep],
        )
        program_qsim = QuantumProgram(
            stack([]),
            default_noise_model=depol_noise_model,
            default_base_seed=20241104,
            state_type=QSimState,
            patch_types={"1Q": code_1Q},
            name="1Q depolarizing test",
        )
        results = program_qsim.run(num_shots=shots)
        assert abs(flips(results) - expected_flips) < tolerance

        ## STIM, X basis
        gate_dict, inst_dict = self._create_model_dicts(
            qubits, StimCircuitGateRep
        )
        # STIM expects the rate of each Pauli flipping, a factor of 3/4
        # (non-I Paulis over all Paulis) of the depolarizing rate
        gate_dict[("Gi", ("Q0",))] = f"DEPOLARIZE1({3/4 * p_depol}) 0"
        depol_noise_model_stim = DictNoiseModel(
            gate_dict,
            inst_dict,
            gatereps=[StimCircuitGateRep],
            instreps=[ZBasisProjectionInstrumentRep],
        )
        program_stim_Xbasis = QuantumProgram.from_quantum_program(
            program_qsim,
            stack(["H0"]),
            state_type=STIMState,
            default_noise_model=depol_noise_model_stim,
        )
        results = program_stim_Xbasis.run(num_shots=shots)
        assert abs(flips(results) - expected_flips) < tolerance

    def test_amp_damp_dephase(self):
        # Qubit 1, prepared in |1>, is damped/dephased; qubits 0 (|1>) and
        # 2 (|+>) must come out untouched.
        p_damp, p_dephase = 0.15, 0.2
        # Damping resets |1> to 0 in p_damp of the shots, with a 3 sigma
        # tolerance; without damping no shot reads 0.
        shots = 100
        expected_zeros = shots * p_damp
        tolerance = 3 * np.sqrt(shots * p_damp * (1 - p_damp))

        qubits = ["Q0", "Q1", "Q2"]
        code_3Q = self._create_code(qubits)
        stack = InstructionStack(
            [
                {
                    "instruction": "Init State",
                    "state": len(qubits),
                    "qubit_labels": qubits,
                },
                {
                    "instruction": "Init Patch 3Q",
                    "new_patch_label": "L0",
                    "qubits": qubits,
                },
                ("X0", "L0"),  # Start qubit 0 in 1
                ("H2", "L0"),  # Start qubit 2 in +
                ("X1", "L0"),
                ("I1", "L0"),
                ("Mz1", "L0"),
                ("Mz0", "L0"),  # Verify qubit 0 still in 1 in Z basis
                ("Mx2", "L0"),  # Verify qubit 2 still in + in X basis
            ]
        )

        def check(results):
            outs = [
                mo["Q1"][0]
                for mo in results.collect_shot_data("measurement_outcomes", -3)
            ]
            assert abs(Counter(outs)[0] - expected_zeros) < tolerance
            outs = [
                mo["Q0"][0]
                for mo in results.collect_shot_data("measurement_outcomes", -2)
            ]
            assert Counter(outs)[0] == 0
            outs = [
                mo["Q2"][0]
                for mo in results.collect_shot_data("measurement_outcomes", -1)
            ]
            assert Counter(outs)[0] == shots

        ## QUANTUMSIM
        gate_dict, inst_dict = self._create_model_dicts(
            qubits, QSimSuperopGateRep
        )
        gate_dict[("Gi", ("Q1",))] = _ptm.amp_ph_damping_ptm(p_damp, p_dephase)  # type: ignore
        noise_model = DictNoiseModel(
            gate_dict,
            inst_dict,
            gatereps=[QSimSuperopGateRep],
            instreps=[ZBasisProjectionInstrumentRep],
        )
        program_qsim = QuantumProgram(
            stack,
            default_noise_model=noise_model,
            default_base_seed=20241104,
            state_type=QSimState,
            patch_types={"3Q": code_3Q},
            name="Amp damp/dephasing test",
        )
        check(program_qsim.run(num_shots=shots))

        ## STIM
        # Recall that e^{-t/T1} = 1-damping_rate and
        # e^{-t/T2} = sqrt([1-damping_rate][1-dephase_rate])
        kappa = 1 - (1 - p_dephase) / (1 - p_damp)
        pz = 0.5 * (1 - np.sqrt(1 - kappa))
        damp_rep = [
            (f"Z_ERROR({pz}) 0", (1 - p_damp)),
            ("R 0", p_damp),
        ]
        gate_dict, inst_dict = self._create_model_dicts(
            qubits, StimCircuitGateRep
        )
        gate_dict[("Gi", ("Q1",))] = damp_rep
        noise_model_stim = DictNoiseModel(
            gate_dict,
            inst_dict,
            gatereps=[ProbabilisticStimGateRep, StimCircuitGateRep],
            instreps=[ZBasisProjectionInstrumentRep],
        )
        program_stim = QuantumProgram.from_quantum_program(
            program_qsim,
            state_type=STIMState,
            default_noise_model=noise_model_stim,
        )
        check(program_stim.run(num_shots=shots))

    @staticmethod
    def _create_code(qubits: list[str]):
        # Make a dummy QECCode with the circuits we want to test
        instructions = {}
        for q in qubits:
            instructions[f"X{q[1]}"] = (
                builders.build_physical_circuit_instruction(
                    ListPhysicalCircuit([[("Gx", (q,))]]), name=f"X on {q}"
                )
            )
            instructions[f"H{q[1]}"] = (
                builders.build_physical_circuit_instruction(
                    ListPhysicalCircuit([[("Gh", (q,))]]), name=f"H on {q}"
                )
            )
            instructions[f"I{q[1]}"] = (
                builders.build_physical_circuit_instruction(
                    ListPhysicalCircuit([[("Gi", (q,))]]),
                    name=f"Noisy I on {q}",
                )
            )
            instructions[f"Mz{q[1]}"] = (
                builders.build_physical_circuit_instruction(
                    ListPhysicalCircuit([[("Imrz", (q,))]]), name=f"Mz on {q}"
                )
            )
            instructions[f"Mx{q[1]}"] = (
                builders.build_physical_circuit_instruction(
                    ListPhysicalCircuit([[("Gh", (q,))], [("Imrz", (q,))]]),
                    name=f"Mx on {q}",
                )
            )

        return QECCode(instructions, qubits, qubits)

    @staticmethod
    def _create_model_dicts(qubits: list[str], gaterep: type[GateRep]):
        assert gaterep in [QSimSuperopGateRep, StimCircuitGateRep]

        gate_dict: dict[tuple[str, tuple[str, ...]], np.ndarray | str] = {}
        if gaterep is QSimSuperopGateRep:
            for q in qubits:
                gate_dict[("Gi", (q,))] = np.eye(4)
                gate_dict[("Gh", (q,))] = _ptm.hadamard_ptm()  # type: ignore
                gate_dict[("Gx", (q,))] = _ptm.rotate_x_ptm(np.pi)  # type: ignore
        else:
            for q in qubits:
                gate_dict[("Gi", (q,))] = "I 0"
                gate_dict[("Gh", (q,))] = "H 0"
                gate_dict[("Gx", (q,))] = "X 0"

        inst_dict = {("Imrz", (q,)): (0, True) for q in qubits}

        return gate_dict, inst_dict


class TestBackendAgreement:
    """A noiseless entangled program with mid-circuit measurement and
    feed-forward gives the same outcomes on every state backend."""

    DATA = ["D0", "D1", "D2"]
    QUBITS = DATA + ["A0"]

    _H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    _X = np.array([[0, 1], [1, 0]])
    _CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    _CNOT_PAIRS = [
        ("D0", "D1"),
        ("D1", "D2"),
        ("D0", "A0"),
        ("D1", "A0"),
        ("D2", "A0"),
    ]

    @staticmethod
    def _flip_if_parities_equal(measurement_outcomes, stack, patch_label):
        parities = [mo["A0"][0] for mo in measurement_outcomes]
        if parities[0] == parities[1]:
            stack = stack.insert_instruction(
                0, InstructionLabel("Flip D0", patch_label=patch_label)
            )
        return Frame({"stack": stack, "parities": parities})

    @classmethod
    def _create_code(cls):
        def circ(layers):
            return builders.build_physical_circuit_instruction(
                ListPhysicalCircuit(layers)
            )

        instructions = {
            "GHZ": circ(
                [
                    [("Gh", ("D0",))],
                    [("Gcnot", ("D0", "D1"))],
                    [("Gcnot", ("D1", "D2"))],
                ]
            ),
            "Parity 1": circ(
                [
                    [("Gcnot", ("D0", "A0"))],
                    [("Gcnot", ("D1", "A0"))],
                    [("Imrz", ("A0",))],
                ]
            ),
            "Parity 2": circ(
                [
                    [("Gcnot", ("D1", "A0"))],
                    [("Gcnot", ("D2", "A0"))],
                    [("Imrz", ("A0",))],
                ]
            ),
            "Flip D0": circ([[("Gx", ("D0",))]]),
            "Measure Data": circ([[("Imrz", (q,)) for q in cls.DATA]]),
            "Feed-Forward": Instruction(
                cls._flip_if_parities_equal,
                param_priorities={"measurement_outcomes": ["history[-2:]"]},
                name="Feed-Forward",
            ),
        }
        return QECCode(instructions, cls.QUBITS, cls.QUBITS)

    @classmethod
    def _create_model(cls, rep):
        units = {("Gh", ("D0",)): cls._H, ("Gx", ("D0",)): cls._X}
        units.update({("Gcnot", pair): cls._CX for pair in cls._CNOT_PAIRS})
        gate_dict = {
            key: convert_rep(UnitaryGateRep(u, key[1]), rep)
            for key, u in units.items()
        }
        inst_dict = {("Imrz", (q,)): (0, True) for q in cls.QUBITS}
        return DictNoiseModel(
            gate_dict,
            inst_dict,
            gatereps=[rep],
            instreps=[ZBasisProjectionInstrumentRep],
        )

    @pytest.mark.parametrize(
        "state_type, gaterep",
        [
            (QSimState, QSimSuperopGateRep),
            (STIMState, StimCircuitGateRep),
            (NumpyStatevectorQuantumState, UnitaryGateRep),
        ],
        ids=["qsim", "stim", "numpy-statevector"],
    )
    def test_ghz_parity_feed_forward(self, state_type, gaterep):
        stack = [
            {
                "instruction": "Init State",
                "state": 4,
                "qubit_labels": self.QUBITS,
            },
            {
                "instruction": "Init Patch 4Q",
                "new_patch_label": "L0",
                "qubits": self.QUBITS,
            },
        ] + [
            (name, "L0")
            for name in (
                "GHZ",
                "Parity 1",
                "Parity 2",
                "Feed-Forward",
                "Measure Data",
            )
        ]
        program = QuantumProgram(
            stack,
            default_noise_model=self._create_model(gaterep),
            default_base_seed=20260101,
            state_type=state_type,
            patch_types={"4Q": self._create_code()},
            name="backend agreement test",
        )
        results = program.run(num_shots=20)

        all_parities = results.collect_shot_data("parities", "all")
        all_outcomes = results.collect_shot_data("measurement_outcomes", -1)
        assert len(all_outcomes) == 20
        for shot_parities, mo in zip(all_parities, all_outcomes):
            # GHZ stabilizers: both parities are 0, so the flip always fires
            assert [p for p in shot_parities if p is not None] == [[0, 0]]
            data = tuple(mo[q][0] for q in self.DATA)
            assert data in [(1, 0, 0), (0, 1, 1)]
