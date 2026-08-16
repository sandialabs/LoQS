"""Tester for loqs.tools.fttools"""

import pytest

pygsti = pytest.importorskip("pygsti")
stim = pytest.importorskip("stim")

from loqs.backends import PyGSTiPhysicalCircuit
from loqs.core import QuantumProgram
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import Instruction
from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.tools import fttools


def _build_circuit_program():
    """A minimal, real (but not necessarily run-able) QuantumProgram
    whose single stack entry holds a PyGSTi-backed physical circuit
    instruction: H on Q0, then CNOT(Q0, Q1). Exercises both entries of
    PAULI_PROPAGATION_GATE_MAP.
    """
    circ = PyGSTiPhysicalCircuit([("Gh", "Q0")], qubit_labels=["Q0", "Q1"])
    circ = circ.append([("Gcnot", "Q0", "Q1")])
    inst = builders.build_physical_circuit_instruction(circuit=circ, name="Circuit")
    program = QuantumProgram(
        instruction_stack=[{"instruction": "Circuit"}],
        global_instructions={"Circuit": inst},
        name="ft base program",
    )
    return program, inst, circ


def _build_counter_program():
    """A minimal, genuinely run-able QuantumProgram (no physical circuit
    at all) for exercising the run/collect-and-compare machinery."""
    trivial_code = trivial_codepack.create_qec_code()
    qubits = ["Q0"]
    ideal_model = trivial_codepack.create_ideal_model(qubits)
    stack = [
        {"instruction": "Init Patch Trivial", "new_patch_label": "L0", "qubits": qubits},
        {"instruction": "Init Counter", "patch_label": "L0", "initial_value": 0},
        {"instruction": "Increment", "patch_label": "L0", "increment_by": 1},
    ]
    return QuantumProgram(
        stack,
        default_noise_model=ideal_model,
        patch_types={"Trivial": trivial_code},
        name="ft counter test",
    )


class TestBuildDiscreteErrorInjectionProgramForCombo:

    def test_injects_error_and_preserves_rest_of_stack(self):
        program, _, _ = _build_circuit_program()
        new_program = fttools.build_discrete_error_injection_program_for_combo(
            program, 0, [(0, "Gxpi", 0)]
        )
        assert new_program is not program
        assert len(new_program.instruction_stack) == 1
        new_label = new_program.instruction_stack[0]
        assert new_label["instruction"] == "Circuit"
        assert new_label["error_injections"] == [(0, "Gxpi", 0)]
        # The original program/label must be untouched (deepcopy, not alias)
        assert "error_injections" not in program.instruction_stack[0]

    def test_weight_2_combo_and_name_includes_both_labels(self):
        program, _, _ = _build_circuit_program()
        new_program = fttools.build_discrete_error_injection_program_for_combo(
            program, 0, [(2, "Gxpi", 0), (2, "Gzpi", 1)]
        )
        new_label = new_program.instruction_stack[0]
        assert new_label["error_injections"] == [(2, "Gxpi", 0), (2, "Gzpi", 1)]
        assert "Gxpi" in new_program.name and "Gzpi" in new_program.name

    def test_empty_error_injections_uses_placeholder_layer_in_name(self):
        program, _, _ = _build_circuit_program()
        new_program = fttools.build_discrete_error_injection_program_for_combo(
            program, 0, []
        )
        assert "layer ?" in new_program.name


class TestPauliPropagation:

    def test_is_stim_pauli_propagation_available(self):
        assert fttools.is_stim_pauli_propagation_available() is True

    def test_propagate_x_through_hadamard_becomes_z(self):
        _, _, circ = _build_circuit_program()
        # H maps X->Z; a lone Z on the CNOT control commutes through unchanged.
        signature = fttools.propagate_pauli_signature(circ, 0, {0: "X"})
        assert signature == ((0, "Z"),)

    def test_propagate_x_through_cnot_control_spreads_to_target(self):
        _, _, circ = _build_circuit_program()
        # Starting after the H (layer 1, just the CNOT): X on the control
        # propagates to X on both control and target.
        signature = fttools.propagate_pauli_signature(circ, 1, {0: "X"})
        assert signature == ((0, "X"), (1, "X"))

    def test_propagate_skips_idle_gates(self):
        circ = PyGSTiPhysicalCircuit([("Imrz", "Q0")], qubit_labels=["Q0"])
        signature = fttools.propagate_pauli_signature(circ, 0, {0: "X"})
        assert signature == ((0, "X"),)

    def test_propagate_unsupported_gate_raises(self):
        circ = PyGSTiPhysicalCircuit([("Gzpi", "Q0")], qubit_labels=["Q0"])
        with pytest.raises(ValueError, match="No Pauli-propagation rule"):
            fttools.propagate_pauli_signature(circ, 0, {0: "X"})

    def test_prune_error_combos_weight_1(self):
        _, _, circ = _build_circuit_program()
        representatives, total = fttools.prune_error_combos_by_propagation(
            circ, ["Gxpi", "Gzpi"], post_twoq_gates=False
        )
        # 3 locations x 2 labels = 6 combos total; some propagate to the
        # same final signature and get pruned to fewer representatives.
        assert total == 6
        assert 0 < len(representatives) <= total

    def test_prune_error_combos_weight_2(self):
        _, _, circ = _build_circuit_program()
        representatives, total = fttools.prune_error_combos_by_propagation(
            circ, ["Gxpi", "Gzpi"], post_twoq_gates=True
        )
        # 1 two-qubit-gate location x 2x2 label combos = 4
        assert total == 4
        assert len(representatives) <= total

    def test_prune_error_combos_falls_back_to_unpruned_without_stim(self, monkeypatch):
        _, _, circ = _build_circuit_program()
        monkeypatch.setattr(
            fttools, "is_stim_pauli_propagation_available", lambda: False
        )
        representatives, total = fttools.prune_error_combos_by_propagation(
            circ, ["Gxpi", "Gzpi"], post_twoq_gates=False
        )
        assert len(representatives) == total == 6


class TestBuildPrunedDiscreteErrorInjectionPrograms:

    def test_returns_fewer_or_equal_programs_than_total(self):
        program, inst, _ = _build_circuit_program()
        programs, total = fttools.build_pruned_discrete_error_injection_programs(
            program, inst, 0, ["Gxpi", "Gzpi"], post_twoq_gates=False
        )
        assert total == 6
        assert 0 < len(programs) <= total
        for p in programs:
            assert "error_injections" in p.instruction_stack[0]


class TestBuildDiscreteErrorInjectionPrograms:

    def test_missing_circuit_key_raises(self):
        program, _, _ = _build_circuit_program()

        def apply_fn():
            pass
        bad_inst = Instruction(apply_fn, data={}, name="bad")

        with pytest.raises(ValueError, match="Key 'circuit' not available"):
            fttools.build_discrete_error_injection_programs(
                program, bad_inst, 0, ["Gxpi"]
            )

    def test_weight_1_sweep_covers_every_location_and_label(self):
        program, inst, _ = _build_circuit_program()
        programs = fttools.build_discrete_error_injection_programs(
            program, inst, 0, ["Gxpi", "Gzpi"], post_twoq_gates=False
        )
        # 3 locations x 2 labels
        assert len(programs) == 6
        all_injections = {
            tuple(p.instruction_stack[0]["error_injections"]) for p in programs
        }
        assert len(all_injections) == 6

    def test_weight_2_sweep_covers_post_twoq_gate_locations(self):
        program, inst, _ = _build_circuit_program()
        programs = fttools.build_discrete_error_injection_programs(
            program, inst, 0, ["Gxpi", "Gzpi"], post_twoq_gates=True
        )
        # 1 two-qubit-gate location x 2x2 label combos
        assert len(programs) == 4
        for p in programs:
            injections = p.instruction_stack[0]["error_injections"]
            assert len(injections) == 2


class TestRunDiscreteErrorInjectedPrograms:

    def test_all_succeed_returns_empty_failed_list(self, capsys):
        program = _build_counter_program()
        failed = fttools.run_discrete_error_injected_programs(
            [program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
        )
        assert failed == []
        assert "All programs succeeded!" in capsys.readouterr().out

    def test_some_fail_are_collected_and_reported(self, capsys):
        program = _build_counter_program()
        failed = fttools.run_discrete_error_injected_programs(
            [program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[999],
            num_shots=1,
        )
        assert failed == [program, program]
        assert "Failed 2 programs!" in capsys.readouterr().out


class TestProgramOutput:

    def test_matching_output_returns_true(self):
        program = _build_counter_program()
        assert fttools.test_program_output(
            program, [("counter", -1)], [1], num_shots=1
        )

    def test_mismatched_output_returns_false(self):
        program = _build_counter_program()
        assert not fttools.test_program_output(
            program, [("counter", -1)], [999], num_shots=1
        )

    def test_verbose_mismatch_prints_output_and_expected(self, capsys):
        program = _build_counter_program()
        fttools.test_program_output(
            program, [("counter", -1)], [999], num_shots=1, verbose=True
        )
        out = capsys.readouterr().out
        assert "Output:" in out and "Expected:" in out

    def test_skip_run_without_previous_results_raises(self):
        program = _build_counter_program()
        with pytest.raises(ValueError, match="Cannot skip run"):
            fttools.test_program_output(
                program, [("counter", -1)], [1], skip_run=True
            )

    def test_skip_run_with_previous_results_reuses_them(self):
        program = _build_counter_program()
        program._last_results = program.run(num_shots=1)
        assert fttools.test_program_output(
            program, [("counter", -1)], [1], skip_run=True
        )
