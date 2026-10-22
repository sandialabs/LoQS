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
from loqs.core import InstructionStack, QuantumProgram
from loqs.codepacks import codepack_surf17_tomita2014 as codepack_surf17


class TestSurf17Codepack:

    @staticmethod
    def _create_program(
        circuit_backend,
        model_backend,
        state_backend,
        layout: Literal["surf17", "surf13", "surf10"] = "surf17",
        basis="Z",
    ):
        code_surf = codepack_surf17.create_qec_code(
            layout=layout,
            circuit_backend=circuit_backend,
        )

        if layout == "surf10":
            qubits = [f"D{i}" for i in range(9)] + ["A9"]
        elif layout == "surf13":
            qubits = [f"D{i}" for i in range(9)] + [
                f"A{i}" for i in range(9, 13)
            ]
        else:
            qubits = [f"D{i}" for i in range(9)] + [
                f"A{i}" for i in range(9, 17)
            ]

        gaterep = UnitaryGateRep
        ideal_model = codepack_surf17.create_ideal_model(
            qubits,
            gaterep=gaterep,
            model_backend=model_backend,
        )

        state_type = state_backend
        patch_types = {"SURF": code_surf}

        prep_inst = "Zero Prep" if basis == "Z" else "Plus Prep"
        meas_inst = (
            "FT Logical Z Measure" if basis == "Z" else "FT Logical X Measure"
        )

        stack = InstructionStack(
            [
                {
                    "instruction": "Init State",
                    "state": len(qubits),
                    "qubit_labels": qubits,
                },
                {
                    "instruction": "Init Patch SURF",
                    "new_patch_label": "L0",
                    "qubits": qubits,
                },
                (prep_inst, "L0"),
                (meas_inst, "L0"),
            ]
        )

        return QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            state_type=state_type,
            patch_types=patch_types,
            name=f"Prep {basis}, measure {basis}",
        )

    @pytest.mark.parametrize(
        "layout, workflow",
        [
            # Every workflow at surf17
            ("surf17", ("Z", [], 0)),  # Prep 0, measure Z -> 0
            ("surf17", ("Z", [("X", "L0")], 1)),  # Prep 0, X, measure Z -> 1
            ("surf17", ("Z", [("QEC", "L0")], 0)),  # Prep 0, QEC, measure Z
            ("surf17", ("X", [], 0)),  # Prep +, measure X -> 0
            ("surf17", ("X", [("Z", "L0")], 1)),  # Prep +, Z, measure X -> 1
            ("surf17", ("X", [("QEC", "L0")], 0)),  # Prep +, QEC, measure X
            # Prep 0, apply H (becomes +), measure X -> 0
            ("surf17", ("Z", [("H", "L0")], 0)),
            # The QEC workflows at the other layouts
            ("surf13", ("Z", [("QEC", "L0")], 0)),
            ("surf13", ("X", [("QEC", "L0")], 0)),
            ("surf10", ("Z", [("QEC", "L0")], 0)),
            ("surf10", ("X", [("QEC", "L0")], 0)),
        ],
    )
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
            qubits = [f"D{i}" for i in range(9)] + [
                f"A{i}" for i in range(9, 13)
            ]
        else:
            qubits = [f"D{i}" for i in range(9)] + [
                f"A{i}" for i in range(9, 17)
            ]

        prep_inst = "Zero Prep" if prep_basis == "Z" else "Plus Prep"
        meas_inst = (
            "FT Logical Z Measure"
            if meas_basis == "Z"
            else "FT Logical X Measure"
        )

        stack = [
            {
                "instruction": "Init State",
                "state": len(qubits),
                "qubit_labels": qubits,
            },
            {
                "instruction": "Init Patch SURF",
                "new_patch_label": "L0",
                "qubits": qubits,
            },
            (prep_inst, "L0"),
        ]
        for g in gates:
            stack.append(g)
        stack.append((meas_inst, "L0"))

        program = QuantumProgram.from_quantum_program(
            ref_program, InstructionStack(stack)
        )
        program_results = program.run()
        assert (
            program_results.collect_shot_data("logical_measurement", -1)[0]
            == expected
        )

    # One fixed fault per basis that reaches the global-measurement decoder's
    # correction. Its outcome depends on the schedule, so only runs are checked.
    @pytest.mark.parametrize(
        "basis, fault",
        [("Z", (0, "Gxpi", 0)), ("X", (0, "Gypi", 2))],
    )
    def test_measurement_fault_branch_smoke(self, basis, fault):
        qubits = [f"D{i}" for i in range(9)] + ["A9"]
        code_surf = codepack_surf17.create_qec_code(
            layout="surf10",
            circuit_backend=PyGSTiPhysicalCircuit,
        )
        ideal_model = codepack_surf17.create_ideal_model(
            qubits,
            gaterep=StimCircuitGateRep,
            model_backend=DictNoiseModel,
        )

        stack = InstructionStack(
            [
                {
                    "instruction": "Init State",
                    "state": len(qubits),
                    "qubit_labels": qubits,
                },
                {
                    "instruction": "Init Patch SURF",
                    "new_patch_label": "L0",
                    "qubits": qubits,
                },
                ("Zero Prep" if basis == "Z" else "Plus Prep", "L0"),
                {
                    "instruction": f"Raw {basis} Data Measure",
                    "patch_label": "L0",
                    "error_injections": [fault],
                },
                (f"FT {basis} logical parity calculation", "L0"),
            ]
        )
        program = QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            state_type=STIMQuantumState,
            patch_types={"SURF": code_surf},
            name="Measurement fault smoke",
        )

        results = program.run(num_shots=10, verbose=False)
        outcomes = results.collect_shot_data("logical_measurement", -1)
        assert len(outcomes) == 10
        assert all(o in (0, 1) for o in outcomes)


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
        "D0": 5,
        "D1": 4,
        "D2": 5,
        "D3": 4,
        "D4": 3,
        "D5": 4,
        "D6": 5,
        "D7": 4,
        "D8": 5,
    }

    # surf13's own self-padded schedule runs X and Z checks in 2 sequential
    # 7-layer passes (14 layers total) rather than surf17's single merged
    # round, so it accumulates roughly double the idle time -- this is
    # exactly the discrepancy idle_layout="surf17" (below) is meant to fix.
    SURF13_SELF_IDLE_COUNTS = {
        "D0": 10,
        "D1": 9,
        "D2": 10,
        "D3": 9,
        "D4": 8,
        "D5": 9,
        "D6": 10,
        "D7": 9,
        "D8": 10,
    }

    def test_no_idle_layout_omits_idles(self):
        code = codepack_surf17.create_qec_code(
            layout="surf10", circuit_backend=PyGSTiPhysicalCircuit
        )
        counts = _per_qubit_idle_counts(code)
        assert all(c == 0 for c in counts.values())

    def test_self_idle_layout_surf17(self):
        code = codepack_surf17.create_qec_code(
            layout="surf17",
            idle_layout="surf17",
            circuit_backend=PyGSTiPhysicalCircuit,
        )
        counts = _per_qubit_idle_counts(code)
        assert counts == self.SURF17_IDLE_COUNTS

    def test_self_idle_layout_surf13(self):
        code = codepack_surf17.create_qec_code(
            layout="surf13",
            idle_layout="surf13",
            circuit_backend=PyGSTiPhysicalCircuit,
        )
        counts = _per_qubit_idle_counts(code)
        assert counts == self.SURF13_SELF_IDLE_COUNTS

    @pytest.mark.parametrize("layout", ["surf10", "surf13"])
    def test_borrow_idle_layout_from_surf17(self, layout):
        code = codepack_surf17.create_qec_code(
            layout=layout,
            idle_layout="surf17",
            circuit_backend=PyGSTiPhysicalCircuit,
        )
        counts = _per_qubit_idle_counts(code)
        assert counts == self.SURF17_IDLE_COUNTS

    def test_unknown_idle_layout_raises(self):
        with pytest.raises(ValueError):
            codepack_surf17.create_qec_code(
                layout="surf17",
                idle_layout="surf99",  # type: ignore
                circuit_backend=PyGSTiPhysicalCircuit,
            )
