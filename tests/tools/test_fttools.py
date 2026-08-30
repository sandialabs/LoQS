"""Tester for loqs.tools.fttools"""

import sys

import pytest

pygsti = pytest.importorskip("pygsti")
stim = pytest.importorskip("stim")

from loqs.backends import PyGSTiPhysicalCircuit
from loqs.core import QuantumProgram
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import Instruction
from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.tools import fttools
from loqs.tools.paralleltools import ParallelStrategy


def _build_shot_executor():
    """Module-level factory (not a closure) building a fresh loky
    executor -- a picklable `shot_executor` factory for hybrid
    shot-/program-level parallelism tests."""
    import loky

    return loky.get_reusable_executor(max_workers=1)


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
        runner = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
        )
        failed = runner.run()
        assert failed == []
        assert "All programs succeeded!" in capsys.readouterr().out

    def test_some_fail_are_collected_and_reported(self, capsys):
        program = _build_counter_program()
        runner = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[999],
            num_shots=1,
        )
        failed = runner.run()
        assert failed == [program, program]
        assert "Failed 2 programs!" in capsys.readouterr().out


class TestRunDiscreteErrorInjectedProgramsParallel:
    """[](api:FaultInjectionRunner)'s `parallel_strategy` (a
    [](api:ParallelStrategy)) path, against real `loky` and `submitit`
    executors -- both must return the driver's own original program
    objects in the failed list (per the runner's own contract), not
    copies that crossed a process boundary. `ParallelStrategy`'s own
    construction-time validation (mutual exclusion, `n_program_chunks`
    requirements) is covered directly in test_paralleltools.py, not
    duplicated here."""

    def test_loky_program_executor_all_succeed_returns_empty_failed_list(
        self, capsys
    ):
        loky = pytest.importorskip("loky")
        program = _build_counter_program()
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )

        runner = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            parallel_strategy=strategy,
        )
        failed = runner.run()

        assert failed == []
        assert "All programs succeeded!" in capsys.readouterr().out

    def test_loky_program_executor_failures_are_the_driver_s_own_objects(
        self,
    ):
        loky = pytest.importorskip("loky")
        program = _build_counter_program()
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )

        runner = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[999],
            num_shots=1,
            parallel_strategy=strategy,
        )
        failed = runner.run()

        assert failed == [program, program]
        assert all(p is program for p in failed)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason=(
            "submitit unconditionally registers a SIGCONT handler for "
            "every job it runs (submitit/core/job_environment.py), a "
            "POSIX-only signal that doesn't exist in Windows's `signal` "
            "module at all -- a real, unconditional upstream limitation "
            "(submitit targets SLURM, a Linux-only scheduler), not "
            "something fixable from LoQS's side."
        ),
    )
    def test_submitit_program_executor_matches_serial_result(
        self, tmp_path
    ):
        submitit = pytest.importorskip("submitit")
        program = _build_counter_program()
        strategy = ParallelStrategy(
            program_executor=submitit.AutoExecutor(
                folder=tmp_path, cluster="local"
            ),
            n_program_chunks=2,
        )

        runner = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            parallel_strategy=strategy,
        )
        failed = runner.run()

        assert failed == []

    def test_hybrid_program_and_shot_executor_matches_serial_result(self):
        """program_executor (across programs) and shot_executor (within
        each program's own shots) nested together -- the real hybrid
        parallelism this stage adds."""
        loky = pytest.importorskip("loky")
        program = _build_counter_program()
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
            shot_executor=_build_shot_executor,
        )

        runner = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            parallel_strategy=strategy,
        )
        failed = runner.run()

        assert failed == []

    def test_hybrid_with_live_loky_shot_executor_needs_no_hand_written_factory(
        self,
    ):
        """A plain live loky executor works as shot_executor here too --
        ParallelStrategy auto-converts it to a picklable factory, so a
        caller never needs to write one by hand (see
        test_paralleltools.py for coverage of the conversion itself)."""
        loky = pytest.importorskip("loky")
        program = _build_counter_program()
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
            shot_executor=loky.get_reusable_executor(max_workers=2),
        )

        runner = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            parallel_strategy=strategy,
        )
        failed = runner.run()

        assert failed == []


class TestFaultInjectionRunnerCheckpointing:
    """Checkpoint/resume and crash-recovery tests for FaultInjectionRunner."""

    def _read_completed_indices(self, ckpt):
        """Return set of indices completed in checkpoint dir's worker files."""
        from loqs.tools.multiprogramrunner import _read_worker_files

        return set(_read_worker_files(ckpt).keys())

    def test_checkpoint_run_matches_non_checkpointed_result(self, tmp_path):
        """Checkpointing doesn't change the result; leaves checkpoint dir."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        runner = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        failed = runner.run()

        assert failed == []
        assert ckpt.exists()
        assert ckpt.is_dir()
        assert (ckpt / "runner.h5").exists()
        assert self._read_completed_indices(ckpt) == {0, 1}

    def test_existing_checkpoint_with_matching_config_auto_resumes(self, tmp_path):
        """Resumed call with matching config continues from checkpoint."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        # First run
        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        failed1 = runner1.run()
        assert failed1 == []

        # Second run with same config: should auto-resume
        runner2 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        failed2 = runner2.run()
        assert failed2 == []

    def test_existing_content_without_runner_h5_raises(self, tmp_path):
        """Content that isn't a recognized checkpoint (no runner.h5) raises."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        (ckpt / "unrelated.txt").write_text("not a checkpoint")

        runner = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        with pytest.raises(FileExistsError):
            runner.run()

    def test_resume_skips_already_checkpointed_programs(self, tmp_path):
        """Resuming skips programs recorded in journal; only re-runs missing."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        # First: checkpoint only the first program
        partial_programs = [program]
        runner1 = fttools.FaultInjectionRunner(
            errored_programs=partial_programs,
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        failed1 = runner1.run()
        assert failed1 == []
        assert self._read_completed_indices(ckpt) == {0}

        # Second: resume with all programs, verify only index 1 runs
        built_indices = []
        original_run_one_program = fttools._run_one_program

        def spy_run_one_program(program, index, **kwargs):
            built_indices.append(index)
            return original_run_one_program(program, index, **kwargs)

        fttools._run_one_program = spy_run_one_program
        try:
            runner2 = fttools.FaultInjectionRunner(
                errored_programs=[program, program],
                collect_shot_data_args=[("counter", -1)],
                expected_outcomes=[1],
                num_shots=1,
                item_checkpoint_dir=ckpt,
            )
            failed2 = runner2.run()
            assert failed2 == []
            # Only index 1 should have been built (index 0 already done)
            assert built_indices == [1]
            assert self._read_completed_indices(ckpt) == {0, 1}
        finally:
            fttools._run_one_program = original_run_one_program

    def test_crash_truncated_last_program_is_redone_on_resume(self, tmp_path):
        """A program incomplete in journal after crash is redone on resume."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        # First run: complete both programs
        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        failed1 = runner1.run()
        assert failed1 == []
        assert self._read_completed_indices(ckpt) == {0, 1}

        # Simulate a crash by rewriting each worker file to keep only
        # index 0 (as if the second program's entry was never durably
        # checkpointed).
        import h5py
        from loqs.internal.streamingmerge import (
            iter_dict_attr_entries,
            merge_dict_attr,
        )

        for wfile in ckpt.glob("worker_*_runner.h5"):
            with h5py.File(wfile, "r") as f:
                kept = [
                    (k, v)
                    for k, v in iter_dict_attr_entries(f, "results")
                    if k == 0
                ]
            wfile.unlink()
            with h5py.File(wfile, "a") as f:
                merge_dict_attr(
                    f,
                    "results",
                    kept,
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

        assert self._read_completed_indices(ckpt) == {0}

        # Resume should re-run the missing program
        runner2 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        failed2 = runner2.run()
        assert failed2 == []
        assert self._read_completed_indices(ckpt) == {0, 1}

    def test_genuine_crash_recovery_via_read_and_run(self, tmp_path):
        """Simulate crash partway, recover via FaultInjectionRunner.read().run()."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        # Set up a runner that will crash partway
        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program, program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )

        # Patch _run_one_program to crash at index 1
        original_run_one_program = fttools._run_one_program
        crash_triggered = []

        def crashing_run_one_program(program, index, **kwargs):
            if index == 1 and not crash_triggered:
                crash_triggered.append(True)
                raise RuntimeError("simulated crash")
            return original_run_one_program(program, index, **kwargs)

        fttools._run_one_program = crashing_run_one_program
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                runner1.run()
        finally:
            fttools._run_one_program = original_run_one_program

        # Verify partial completion: only index 0 journaled
        assert self._read_completed_indices(ckpt) == {0}

        # Recover via .read().run()
        runner2 = fttools.FaultInjectionRunner.read(ckpt / "runner.h5")
        failed2 = runner2.run()
        assert failed2 == []

        # All items should now be journaled
        assert self._read_completed_indices(ckpt) == {0, 1, 2}

    def test_resume_mismatched_num_shots_raises(self, tmp_path):
        """Mismatched num_shots on resume raises ValueError naming the field."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        runner1.run()

        runner2 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=2,  # Different!
            item_checkpoint_dir=ckpt,
        )
        with pytest.raises(ValueError, match="num_shots"):
            runner2.run()

    def test_resume_mismatched_collect_shot_data_args_raises(self, tmp_path):
        """Mismatched collect_shot_data_args on resume raises ValueError."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        runner1.run()

        runner2 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", 0)],  # Different!
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        with pytest.raises(ValueError, match="collect_shot_data_args"):
            runner2.run()

    def test_resume_mismatched_expected_outcomes_raises(self, tmp_path):
        """Mismatched expected_outcomes on resume raises ValueError."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        runner1.run()

        runner2 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[999],  # Different!
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        with pytest.raises(ValueError, match="expected_outcomes"):
            runner2.run()

    def test_force_resume_bypasses_config_mismatch(self, tmp_path):
        """force_resume=True proceeds despite config mismatch."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            item_checkpoint_dir=ckpt,
        )
        failed1 = runner1.run()
        assert failed1 == []

        # Resume with different num_shots but force_resume=True
        runner2 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=2,  # Different, but forced
            item_checkpoint_dir=ckpt,
            force_resume=True,
        )
        failed2 = runner2.run()
        assert failed2 == []


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
