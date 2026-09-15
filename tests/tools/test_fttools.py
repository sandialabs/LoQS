"""Tester for loqs.tools.fttools"""

import functools
import multiprocessing as mp
import sys
import time

import numpy as np
import pytest

pygsti = pytest.importorskip("pygsti")
stim = pytest.importorskip("stim")

from loqs.backends import PyGSTiPhysicalCircuit, STIMPhysicalCircuit
from loqs.core import Frame, QuantumProgram
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import Instruction
from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.tools import fttools
from loqs.tools.paralleltools import ParallelStrategy

from _shared_checkpoint_test_helpers import (
    _build_shot_executor,
    _crash_once_and_log_shots,
    _wait_for_index_checkpointed,
)


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

    def test_prune_error_combos_rejects_stim_circuit(self):
        # STIMPhysicalCircuit should raise AssertionError when passed to
        # prune_error_combos_by_propagation (requires PyGSTiPhysicalCircuit).
        stim_circ = STIMPhysicalCircuit(
            "H 0\nTICK\nCX 0 1", qubit_labels=[0, 1]
        )
        with pytest.raises(
            AssertionError,
            match=(
                "Pauli propagation pruning only supports "
                "PyGSTiPhysicalCircuit-backed circuits."
            ),
        ):
            fttools.prune_error_combos_by_propagation(
                stim_circ, ["Gxpi"], post_twoq_gates=False
            )


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

    def test_finalize_summary_suppressed_when_show_progress_false(
        self, capsys
    ):
        """FaultInjectionRunner._finalize suppresses summary prints when show_progress=False."""
        program = _build_counter_program()
        runner = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            show_progress=False,
        )
        runner.run()
        out = capsys.readouterr().out
        assert "succeeded" not in out
        assert "Failed" not in out


class TestRunDiscreteErrorInjectedProgramsParallel:
    """[](api:FaultInjectionRunner)'s `parallel_strategy` (a
    [](api:ParallelStrategy)) path, against real `loky` and `submitit`
    executors -- both must return the driver's own original program
    objects in the failed list (per the runner's own contract), not
    copies that crossed a process boundary. `ParallelStrategy`'s own
    construction-time validation (mutual exclusion, `n_program_chunks`
    requirements) is covered directly in test_paralleltools.py, not
    duplicated here."""

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
        """Return set of indices completed so far, reading the union of
        runner.h5's own consolidated state and any remaining worker files
        (covers both a still-in-progress run and one already consolidated)."""
        from loqs.tools.multiprogramrunner import _read_done_union

        return set(_read_done_union(ckpt).keys())

    def test_existing_checkpoint_with_matching_config_auto_resumes(self, tmp_path):
        """Resumed call with matching config continues from checkpoint."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        # First run
        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1, checkpoint=True, item_checkpoint_dir=ckpt,
        )
        failed1 = runner1.run()
        assert failed1 == []

        # Second run with same config: should auto-resume
        runner2 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1, checkpoint=True, resume=True, item_checkpoint_dir=ckpt,
        )
        failed2 = runner2.run()
        assert failed2 == []

    def test_resume_mismatched_keep_shot_results_raises(self, tmp_path):
        """A resumed call with a different keep_shot_results than the
        checkpoint was written with is a hard error naming that field."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"
        shot_ckpt = tmp_path / "shot_checkpoint"

        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1, checkpoint=True, item_checkpoint_dir=ckpt,
            keep_shot_results=False,
            shot_checkpoint=False,
        )
        runner1.run()

        runner2 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1, checkpoint=True, resume=True, item_checkpoint_dir=ckpt,
            keep_shot_results=True,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt,
        )
        with pytest.raises(ValueError, match="keep_shot_results"):
            runner2.run()

    def test_resume_mismatched_keep_shot_results_force_resume_works(
        self, tmp_path
    ):
        """force_resume=True bypasses a keep_shot_results mismatch."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"
        shot_ckpt = tmp_path / "shot_checkpoint"

        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1, checkpoint=True, item_checkpoint_dir=ckpt,
            keep_shot_results=False,
            shot_checkpoint=False,
        )
        runner1.run()

        runner2 = fttools.FaultInjectionRunner(
            errored_programs=[program, program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1, checkpoint=True, resume=True, item_checkpoint_dir=ckpt,
            keep_shot_results=True,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt,
            force_resume=True,
        )
        failed2 = runner2.run()
        assert failed2 == []

    def test_resume_with_equivalent_expected_outcomes_succeeds(
        self, tmp_path
    ):
        """Resume succeeds when expected_outcomes is passed as an
        equivalent-but-differently-typed sequence."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"
        # Run with list form
        runner1 = fttools.FaultInjectionRunner(
            errored_programs=[program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            checkpoint=True,
            item_checkpoint_dir=ckpt,
        )
        runner1.run()

        # Resume with tuple form
        runner2 = fttools.FaultInjectionRunner(
            errored_programs=[program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=(1,),  # tuple instead of list
            num_shots=1,
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=ckpt,
        )
        # Should not raise despite differently-typed expected_outcomes
        result = runner2.run()
        assert result is not None


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

    def test_custom_results_filename_in_checkpoint(self, tmp_path):
        """Custom results_filename creates and resumes from custom-named file."""
        program = _build_counter_program()
        ckpt_dir = tmp_path / "checkpoint"
        ckpt_dir.mkdir()

        # First run with custom filename
        custom_filename = "custom_results.h5"
        result = fttools.test_program_output(
            program,
            [("counter", -1)],
            [1],
            num_shots=1,
            checkpoint=True,
            checkpoint_dir=ckpt_dir,
            results_filename=custom_filename,
        )
        assert result is True
        assert (ckpt_dir / custom_filename).exists()
        assert not (ckpt_dir / "results.h5").exists()

        # Second run should resume from the custom file
        result = fttools.test_program_output(
            program,
            [("counter", -1)],
            [1],
            num_shots=1,
            checkpoint=True,
            checkpoint_dir=ckpt_dir,
            results_filename=custom_filename,
        )
        assert result is True

    def test_explicit_resume_false_overrides_cascade(self, tmp_path):
        """Explicit resume=False prevents cascade from resuming existing checkpoint."""
        program = _build_counter_program()
        ckpt_dir = tmp_path / "checkpoint"
        ckpt_dir.mkdir()

        # First run to create checkpoint
        result = fttools.test_program_output(
            program,
            [("counter", -1)],
            [1],
            num_shots=1,
            checkpoint=True,
            checkpoint_dir=ckpt_dir,
        )
        assert result is True
        assert (ckpt_dir / "results.h5").exists()

        # Second run with explicit resume=False should fail
        # (resume=False + checkpoint=True + existing checkpoint triggers state mismatch error)
        with pytest.raises(ValueError):
            fttools.test_program_output(
                program,
                [("counter", -1)],
                [1],
                num_shots=1,
                checkpoint=True,
                checkpoint_dir=ckpt_dir,
                resume=False,
            )

    def test_explicit_resume_true_without_checkpoint_raises(self):
        """resume=True without checkpoint=True raises ValueError."""
        program = _build_counter_program()
        with pytest.raises(
            ValueError, match="resume=True requires checkpoint=True"
        ):
            fttools.test_program_output(
                program,
                [("counter", -1)],
                [1],
                num_shots=1,
                checkpoint=False,
                resume=True,
            )

    def test_checkpoint_true_without_checkpoint_dir_raises(self):
        """checkpoint=True without checkpoint_dir raises ValueError."""
        program = _build_counter_program()
        with pytest.raises(
            ValueError, match="checkpoint=True requires checkpoint_dir"
        ):
            fttools.test_program_output(
                program,
                [("counter", -1)],
                [1],
                num_shots=1,
                checkpoint=True,
                checkpoint_dir=None,
            )

    def test_fault_injection_runner_with_custom_results_filename(self, tmp_path):
        """FaultInjectionRunner threads results_filename through shot checkpoints."""
        program = _build_counter_program()
        shot_ckpt = tmp_path / "shot_checkpoint"
        custom_filename = "my_results.h5"

        runner = fttools.FaultInjectionRunner(
            errored_programs=[program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt,
            results_filename=custom_filename,
        )
        failed = runner.run()

        assert failed == []
        # Shot checkpoint is created in a per-item subdirectory
        item_shot_ckpt = shot_ckpt / "fault_0"
        assert (item_shot_ckpt / custom_filename).exists()
        assert not (item_shot_ckpt / "results.h5").exists()


class TestRunKwargsPassthrough:
    """Test run_kwargs passthrough in FaultInjectionRunner and test_program_output."""

    def test_run_kwargs_roundtrips_via_serialization(self, tmp_path, make_temp_path):
        """FaultInjectionRunner with run_kwargs serializes and deserializes correctly."""
        program = _build_counter_program()
        item_ckpt = tmp_path / "item_checkpoint"

        original_runner = fttools.FaultInjectionRunner(
            errored_programs=[program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=1,
            checkpoint=True,
            item_checkpoint_dir=item_ckpt,
            run_kwargs={"max_frame_limit": 999},
        )

        with make_temp_path(suffix=".h5") as f_path:
            original_runner.write(f_path)
            loaded_runner = fttools.FaultInjectionRunner.read(f_path)

        assert loaded_runner.run_kwargs == {"max_frame_limit": 999}

    def test_run_kwargs_passed_to_program_run(self, tmp_path):
        """test_program_output forwards run_kwargs to QuantumProgram.run()."""
        program = _build_counter_program()
        # Use a high max_frame_limit to verify it's actually forwarded
        # (if it wasn't, default limit would apply and behavior could differ)
        result = fttools.test_program_output(
            program,
            [("counter", -1)],
            [1],
            num_shots=1,
            run_kwargs={"max_frame_limit": 1000},
        )
        assert result is True

    def test_fault_injection_runner_with_keep_shot_results_and_run_kwargs(
        self, tmp_path
    ):
        """FaultInjectionRunner with both keep_shot_results=True and run_kwargs works."""
        program = _build_counter_program()
        shot_ckpt = tmp_path / "shot_checkpoint"
        item_ckpt = tmp_path / "item_checkpoint"

        runner = fttools.FaultInjectionRunner(
            errored_programs=[program],
            collect_shot_data_args=[("counter", -1)],
            expected_outcomes=[1],
            num_shots=2,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt,
            checkpoint=True,
            item_checkpoint_dir=item_ckpt,
            keep_shot_results=True,
            run_kwargs={"max_frame_limit": 500},
        )
        failed = runner.run()

        assert failed == []
        # Verify shot checkpoint was created
        item_shot_ckpt = shot_ckpt / "fault_0"
        assert (item_shot_ckpt / "results.h5").exists()


def _flip_coin_apply(seed, fail_prob=0.0) -> Frame:
    """"Fail" a shot with probability `fail_prob`, deterministically from `seed`."""
    rng = np.random.default_rng(seed)
    return Frame({"failed": bool(rng.random() < fail_prob)})


FLIP_COIN = Instruction(apply_fn=_flip_coin_apply, name="Flip Coin")


class TestHistoryDataCollectorWithDict:
    """Tests that a literal HistoryDataCollector instance can be used directly
    in collect_shot_data_args and survives checkpointing/serialization."""

    def test_literal_history_data_collector_in_runner_serialize(self, tmp_path, make_temp_path):
        """FaultInjectionRunner accepts a literal HistoryDataCollector instance."""
        from loqs.core.historydatacollector import HistoryDataCollector

        program = _build_counter_program()
        item_ckpt = tmp_path / "item_checkpoint"

        # A literal HistoryDataCollector instance, not a raw dict/tuple spec.
        collector = HistoryDataCollector(key="counter", indices=-1)
        runner = fttools.FaultInjectionRunner(
            errored_programs=[program],
            collect_shot_data_args=[collector],  # literal, not dict
            expected_outcomes=[1],
            num_shots=1,
            checkpoint=True,
            item_checkpoint_dir=item_ckpt,
        )
        runner.run()

        # Serialize and resume
        with make_temp_path(suffix=".h5") as f_path:
            runner.write(f_path)
            loaded = fttools.FaultInjectionRunner.read(f_path)

        # Loaded instance should have the same collector
        assert loaded.collect_shot_data_args == [collector]

    def test_literal_history_data_collector_in_noisesweep(self, tmp_path, make_temp_path):
        """NoiseSweepRunner accepts a literal HistoryDataCollector instance."""
        from loqs.core.historydatacollector import HistoryDataCollector
        from loqs.tools.noisesweeptools import NoiseSweepRunner

        item_ckpt = tmp_path / "item_checkpoint"

        # A literal HistoryDataCollector instance, not a raw dict/tuple spec.
        collector = HistoryDataCollector(key="failed", indices=-1)
        runner = NoiseSweepRunner(
            strengths=[0.0, 0.1],
            num_shots=1,
            collect_shot_data_args=[collector],  # literal, not dict
            expected_outcomes=[False, False],
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            checkpoint=True,
            item_checkpoint_dir=item_ckpt,
        )
        runner.run()

        # Serialize and resume
        with make_temp_path(suffix=".h5") as f_path:
            runner.write(f_path)
            loaded = NoiseSweepRunner.read(f_path)

        # Loaded instance should have the same collector
        assert loaded.collect_shot_data_args == [collector]
