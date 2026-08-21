"""Tester for loqs.codepacks.codepack_surf17_tomita2014"""

import pytest
from typing import Literal

pygsti = pytest.importorskip("pygsti")

from loqs.backends import (
    PyGSTiPhysicalCircuit,
    DictNoiseModel,
    NumpyStatevectorQuantumState,
    STIMQuantumState,
)
from loqs.backends.reps import StimCircuitGateRep, UnitaryGateRep
from loqs.core import QuantumProgram
from loqs.codepacks import codepack_surf17_tomita2014 as codepack_surf17
from loqs.tools import fttools


class TestSurf17Codepack:

    @staticmethod
    def _create_program(circuit_backend, model_backend, state_backend, layout: Literal["surf17", "surf13", "surf10"] = "surf17", basis="Z"):
        code_surf = codepack_surf17.create_qec_code(
            layout=layout,
            circuit_backend=circuit_backend,
        )

        if layout == "surf10":
            qubits = [f"D{i}" for i in range(9)] + ["A9"]
        elif layout == "surf13":
            qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 13)]
        else:
            qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 17)]

        gaterep = UnitaryGateRep
        ideal_model = codepack_surf17.create_ideal_model(
            qubits,
            gaterep=gaterep,
            model_backend=model_backend,
        )

        state_type = state_backend
        patch_types = {"SURF": code_surf}

        prep_inst = "Zero Prep" if basis == "Z" else "Plus Prep"
        meas_inst = "FT Logical Z Measure" if basis == "Z" else "FT Logical X Measure"

        stack = [
            {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
            {"instruction": "Init Patch SURF", "new_patch_label": "L0", "qubits": qubits},
            (prep_inst, "L0"),
            (meas_inst, "L0"),
        ]

        return QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            state_type=state_type,
            patch_types=patch_types,
            name=f"Prep {basis}, measure {basis}",
        )

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
    @pytest.mark.parametrize("workflow", [
        # Z-basis tests
        ("Z", [], 0),                                       # Prep 0, measure Z -> 0
        ("Z", [("X", "L0")], 1),                             # Prep 0, apply X, measure Z -> 1
        ("Z", [("QEC", "L0")], 0),                           # Prep 0, QEC, measure Z -> 0
        # X-basis tests
        ("X", [], 0),                                       # Prep +, measure X -> 0
        ("X", [("Z", "L0")], 1),                             # Prep +, apply Z, measure X -> 1
        ("X", [("QEC", "L0")], 0),                           # Prep +, QEC, measure X -> 0
        # H gate tests
        ("Z", [("H", "L0")], 0),                             # Prep 0, apply H (becomes +), measure X -> 0
    ])
    def test_workflow(self, layout, workflow):
        circuit_backend = PyGSTiPhysicalCircuit
        model_backend = DictNoiseModel
        state_backend = NumpyStatevectorQuantumState

        prep_basis, gates, expected = workflow

        # For the H gate test, we prep Z (0), apply H, and measure X
        meas_basis = "X" if "H" in [g[0] for g in gates] else prep_basis

        ref_program = self._create_program(
            circuit_backend,
            model_backend,
            state_backend,
            layout=layout,
            basis=prep_basis,
        )

        if layout == "surf10":
            qubits = [f"D{i}" for i in range(9)] + ["A9"]
        elif layout == "surf13":
            qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 13)]
        else:
            qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 17)]

        prep_inst = "Zero Prep" if prep_basis == "Z" else "Plus Prep"
        meas_inst = "FT Logical Z Measure" if meas_basis == "Z" else "FT Logical X Measure"

        stack = [
            {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
            {"instruction": "Init Patch SURF", "new_patch_label": "L0", "qubits": qubits},
            (prep_inst, "L0"),
        ]
        for g in gates:
            stack.append(g)
        stack.append((meas_inst, "L0"))

        program = QuantumProgram.from_quantum_program(ref_program, stack)
        program_results = program.run()
        assert program_results.collect_shot_data("logical_measurement", -1)[0] == expected

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_qec_fault_tolerance(self, layout, basis):
        """Weight-1 SE faults never flip the logical, in either basis.

        The X basis is the sensitive one for Z-ancilla hook errors (a Z
        fault on an interior Z-check ancilla mid-extraction propagates onto
        two data qubits); the Z-template CNOT order b,d,a,c keeps that hook
        perpendicular to Z_L.
        """
        circuit_backend = PyGSTiPhysicalCircuit
        model_backend = DictNoiseModel
        state_backend = STIMQuantumState

        code_surf = codepack_surf17.create_qec_code(
            layout=layout,
            circuit_backend=circuit_backend,
            num_qec_rounds=3,
        )

        if layout == "surf10":
            qubits = [f"D{i}" for i in range(9)] + ["A9"]
        elif layout == "surf13":
            qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 13)]
        else:
            qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 17)]

        ideal_model = codepack_surf17.create_ideal_model(
            qubits,
            gaterep=StimCircuitGateRep,
            model_backend=model_backend,
        )

        stack = [
            {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
            {"instruction": "Init Patch SURF", "new_patch_label": "L0", "qubits": qubits},
            ("Zero Prep" if basis == "Z" else "Plus Prep", "L0"),
            ("Syndrome Extraction", "L0"),  # index 3 (inject here)
            ("Syndrome Extraction", "L0"),  # index 4
            ("Syndrome Extraction", "L0"),  # index 5
            ("Decoder", "L0"),
            (f"FT Logical {basis} Measure", "L0"),
        ]

        base_program = QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            state_type=state_backend,
            patch_types={"SURF": code_surf},
            name="QEC FT Test",
        )

        noise_injected_programs = fttools.build_discrete_error_injection_programs(
            base_program=base_program,
            instruction_to_analyze=code_surf.instructions["Syndrome Extraction"],
            stack_idx_to_modify=3,
            error_circuit_labels=["Gxpi", "Gypi", "Gzpi"],
        )

        failed = fttools.run_discrete_error_injected_programs(
            noise_injected_programs,
            [("logical_measurement", -1)],
            [0],
        )
        assert len(failed) == 0

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
    def test_measurement_fault_tolerance(self, layout):
        circuit_backend = PyGSTiPhysicalCircuit
        model_backend = DictNoiseModel
        state_backend = STIMQuantumState

        code_surf = codepack_surf17.create_qec_code(
            layout=layout,
            circuit_backend=circuit_backend,
        )

        if layout == "surf10":
            qubits = [f"D{i}" for i in range(9)] + ["A9"]
        elif layout == "surf13":
            qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 13)]
        else:
            qubits = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 17)]

        ideal_model = codepack_surf17.create_ideal_model(
            qubits,
            gaterep=StimCircuitGateRep,
            model_backend=model_backend,
        )

        stack = [
            {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
            {"instruction": "Init Patch SURF", "new_patch_label": "L0", "qubits": qubits},
            ("Zero Prep", "L0"),
            ("Raw Z Data Measure", "L0"),  # index 3
            ("FT Z logical parity calculation", "L0"),
        ]

        base_program = QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            state_type=state_backend,
            patch_types={"SURF": code_surf},
            name="Measurement FT Test",
        )

        noise_injected_programs = fttools.build_discrete_error_injection_programs(
            base_program=base_program,
            instruction_to_analyze=code_surf.instructions["Raw Z Data Measure"],
            stack_idx_to_modify=3,
            error_circuit_labels=["Gxpi", "Gypi", "Gzpi"],
        )

        failed = fttools.run_discrete_error_injected_programs(
            noise_injected_programs,
            [("logical_measurement", -1)],
            [0],
        )
        assert len(failed) == 0


_IDLE_NAMES = {"Gi1Q", "Gi2Q", "GiMCM", "Gi"}
_DATA_QUBITS = [f"D{i}" for i in range(9)]


def _per_qubit_idle_counts(code) -> dict[str, int]:
    """Count idle-gate instances per data qubit in a codepack's Syndrome
    Extraction circuit."""
    circuit = code.instructions["Syndrome Extraction"].data["circuit"].circuit
    counts = {q: 0 for q in _DATA_QUBITS}
    for lidx in range(circuit.depth):
        for comp in circuit._layer_components(lidx):
            if comp.qubits and comp.name in _IDLE_NAMES:
                for q in comp.qubits:
                    if q in counts:
                        counts[q] += 1
    return counts


class TestSurf17IdleLayout:
    """Tests for the idle_layout parameter of create_qec_code, covering
    self-padding (idle_layout == layout) and borrowing a more-parallel
    layout's idle schedule onto a more-serialized one."""

    # Hand-derived from surf17's 7-layer merged Syndrome Extraction round:
    # each data qubit's idle count is 7 minus how many of the 8 checks
    # touch it with a real gate.
    SURF17_IDLE_COUNTS = {
        "D0": 5, "D1": 4, "D2": 5, "D3": 4, "D4": 3,
        "D5": 4, "D6": 5, "D7": 4, "D8": 5,
    }

    # surf13's own self-padded schedule runs X and Z checks in 2 sequential
    # 7-layer passes (14 layers total) rather than surf17's single merged
    # round, so it accumulates roughly double the idle time -- this is
    # exactly the discrepancy idle_layout="surf17" (below) is meant to fix.
    SURF13_SELF_IDLE_COUNTS = {
        "D0": 10, "D1": 9, "D2": 10, "D3": 9, "D4": 8,
        "D5": 9, "D6": 10, "D7": 9, "D8": 10,
    }

    def test_no_idle_layout_omits_idles(self):
        code = codepack_surf17.create_qec_code(
            layout="surf10", circuit_backend=PyGSTiPhysicalCircuit
        )
        counts = _per_qubit_idle_counts(code)
        assert all(c == 0 for c in counts.values())

    def test_self_idle_layout_surf17(self):
        code = codepack_surf17.create_qec_code(
            layout="surf17", idle_layout="surf17", circuit_backend=PyGSTiPhysicalCircuit
        )
        counts = _per_qubit_idle_counts(code)
        assert counts == self.SURF17_IDLE_COUNTS

    def test_self_idle_layout_surf13(self):
        code = codepack_surf17.create_qec_code(
            layout="surf13", idle_layout="surf13", circuit_backend=PyGSTiPhysicalCircuit
        )
        counts = _per_qubit_idle_counts(code)
        assert counts == self.SURF13_SELF_IDLE_COUNTS

    @pytest.mark.parametrize("layout", ["surf10", "surf13"])
    def test_borrow_idle_layout_from_surf17(self, layout):
        code = codepack_surf17.create_qec_code(
            layout=layout, idle_layout="surf17", circuit_backend=PyGSTiPhysicalCircuit
        )
        counts = _per_qubit_idle_counts(code)
        assert counts == self.SURF17_IDLE_COUNTS

    def test_surf10_cannot_self_supply_idle_layout(self):
        with pytest.raises(ValueError):
            codepack_surf17.create_qec_code(
                layout="surf10", idle_layout="surf10",
                circuit_backend=PyGSTiPhysicalCircuit,
            )

    def test_cannot_borrow_from_more_serialized_layout(self):
        with pytest.raises(ValueError):
            codepack_surf17.create_qec_code(
                layout="surf17", idle_layout="surf10",
                circuit_backend=PyGSTiPhysicalCircuit,
            )
        with pytest.raises(ValueError):
            codepack_surf17.create_qec_code(
                layout="surf13", idle_layout="surf10",
                circuit_backend=PyGSTiPhysicalCircuit,
            )

    def test_unknown_idle_layout_raises(self):
        with pytest.raises(ValueError):
            codepack_surf17.create_qec_code(
                layout="surf17", idle_layout="surf99",  # type: ignore
                circuit_backend=PyGSTiPhysicalCircuit,
            )
