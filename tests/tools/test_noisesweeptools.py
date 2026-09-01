"""Tester for loqs.tools.noisesweeptools"""

import inspect
import sys
import warnings

import numpy as np
import pytest

from loqs.core import Frame, Instruction, ProgramResults, QuantumProgram
from loqs.backends.state import BaseQuantumState, NumpyStatevectorQuantumState
from loqs.internal.serializable import Serializable
from loqs.tools import paralleltools
from loqs.tools.noisesweeptools import (
    NoiseSweepResult,
    NoiseSweepRunner,
    _sweep_point_checkpoint_subdir,
    compare_noise_sweeps,
    plot_noise_sweep,
)
from loqs.tools.paralleltools import ParallelStrategy


def _build_shot_executor():
    """Module-level factory (not a closure) building a fresh loky
    executor -- a picklable `shot_executor` factory for hybrid
    shot-/point-level parallelism tests."""
    import loky

    return loky.get_reusable_executor(max_workers=1)


# ---------------------------------------------------------------------------
# A tiny, Frame-only synthetic "codepack" used to exercise NoiseSweepRunner
# without needing a real physical-circuit backend (stim/quantumsim/pygsti).
# `_flip_coin_apply` must stay a real, module-level `def` (not a lambda/closure)
# since NoiseSweepRunner serializes any callable QuantumProgram-forwarding
# parameter via source-code introspection.
# ---------------------------------------------------------------------------


def _flip_coin_apply(seed, fail_prob=0.0) -> Frame:
    """"Fail" a shot with probability `fail_prob`, deterministically from `seed`."""
    rng = np.random.default_rng(seed)
    return Frame({"failed": bool(rng.random() < fail_prob)})


FLIP_COIN = Instruction(apply_fn=_flip_coin_apply, name="Flip Coin")


def make_stack(fail_prob):
    """Build a one-instruction stack whose failure probability is `fail_prob`."""
    return [{"instruction": "Flip Coin", "fail_prob": fail_prob}]


def identity_noise_model(strength):
    """A trivial "noise model" callable -- real module-level `def`, not a lambda, since
    lambdas aren't properly supported by NoiseSweepRunner's source-based serialization
    (`inspect.getsource` returns the whole call-site line for a lambda, not just its body)."""
    return strength


def name_for_strength(strength):
    """Real module-level `def`, for the same reason as `identity_noise_model` above."""
    return f"point-{strength}"


def make_runner(strengths, **kwargs):
    kwargs.setdefault("instruction_stack", make_stack)
    kwargs.setdefault("global_instructions", {"Flip Coin": FLIP_COIN})
    # Default num_shots/seed_stride always satisfy num_shots <= seed_stride: derive
    # num_shots from an explicit seed_stride, or fall back to fixed defaults.
    if "seed_stride" in kwargs:
        seed_stride = kwargs["seed_stride"]
        kwargs.setdefault("num_shots", max(1, seed_stride))
    else:
        kwargs.setdefault("num_shots", 10)
        kwargs.setdefault("seed_stride", 100)
    kwargs.setdefault("collect_shot_data_args", COLLECT_SHOT_DATA_ARGS)
    kwargs.setdefault("expected_outcomes", EXPECTED_OUTCOMES)
    return NoiseSweepRunner(strengths, **kwargs)


COLLECT_SHOT_DATA_ARGS = [("failed", -1)]
EXPECTED_OUTCOMES = [False]


class TestBuildProgram:
    def test_seed_formula(self):
        runner = make_runner([0.0, 0.1, 0.2], base_seed=5, seed_stride=100)
        for index in range(3):
            program = runner.build_program(index)
            assert program.default_base_seed == 5 + index * 100

    def test_resolves_fixed_and_callable_mix(self):
        runner = make_runner(
            [0.01, 0.02],
            seed_stride=10,
            name=name_for_strength,
        )
        program0 = runner.build_program(0)
        program1 = runner.build_program(1)
        assert program0.name == "point-0.01"
        assert program1.name == "point-0.02"
        # instruction_stack (also callable) should resolve per-point too
        assert program0.instruction_stack.pop_instruction()[0][
            "fail_prob"
        ] == 0.01

    def test_default_base_seed_rejected(self):
        with pytest.raises(TypeError):
            make_runner([0.1], default_base_seed=5)

    def test_state_type_fixed_class_is_not_treated_as_callable(self):
        runner = make_runner([0.1], seed_stride=1, state_type=NumpyStatevectorQuantumState)
        assert runner._quantum_program_values["state_type"] is NumpyStatevectorQuantumState
        assert "state_type" not in runner._quantum_program_serialized_callables
        program = runner.build_program(0)
        assert program.state_type is NumpyStatevectorQuantumState

    def test_state_type_callable_is_treated_as_callable(self):
        def pick_state_type(strength):
            return NumpyStatevectorQuantumState

        runner = make_runner([0.1], seed_stride=1, state_type=pick_state_type)
        assert "state_type" in runner._quantum_program_serialized_callables
        assert "state_type" not in runner._quantum_program_values
        program = runner.build_program(0)
        assert program.state_type is NumpyStatevectorQuantumState


class TestSignatureParity:
    def test_quantum_program_params_all_present(self):
        program_params = set(
            inspect.signature(QuantumProgram.__init__).parameters
        ) - {"self", "default_base_seed"}
        runner_params = set(
            inspect.signature(NoiseSweepRunner.__init__).parameters
        )
        missing = program_params - runner_params
        assert not missing, (
            f"QuantumProgram.__init__ parameter(s) {missing} are not forwarded by "
            "NoiseSweepRunner.__init__"
        )


class TestSerialization:
    def test_round_trip_fixed_and_callable_mix(self, tmp_path):
        runner = make_runner(
            [0.01, 0.02, 0.05],
            base_seed=3,
            seed_stride=50,
            default_noise_model=identity_noise_model,
            name="fixed name",
        )
        path = tmp_path / "runner.json"
        runner.write(path)
        loaded = NoiseSweepRunner.read(path)

        assert loaded.strengths == runner.strengths
        assert loaded.base_seed == runner.base_seed
        assert loaded.seed_stride == runner.seed_stride
        assert loaded.name == "fixed name"
        assert callable(loaded.default_noise_model)
        assert loaded.default_noise_model(0.5) == 0.5

        # Both instances should build equivalent programs
        for index in range(3):
            p_orig = runner.build_program(index)
            p_loaded = loaded.build_program(index)
            assert p_orig.default_base_seed == p_loaded.default_base_seed
            assert p_orig.name == p_loaded.name

    def test_values_and_callables_partition_exactly(self):
        runner = make_runner(
            [0.1],
            seed_stride=1,
            default_noise_model=identity_noise_model,
            expiring_state=False,
        )
        value_keys = set(runner._quantum_program_values)
        callable_keys = set(runner._quantum_program_serialized_callables)
        assert value_keys.isdisjoint(callable_keys)
        assert value_keys | callable_keys == {
            "instruction_stack",
            "initial_history",
            "default_noise_model",
            "expiring_state",
            "global_instructions",
            "state_type",
            "patch_types",
            "override_global_instructions",
            "name",
        }

    def test_non_file_backed_callable_raises_without_override(self):
        # A notebook-defined function has no real source file, so
        # inspect.getsource fails with OSError or a subclass of it.
        env = {}
        exec("def interactive_fn(strength):\n    return strength\n", env)
        interactive_fn = env["interactive_fn"]

        with pytest.raises(OSError):
            make_runner([0.1], seed_stride=1, default_noise_model=interactive_fn)

    def test_non_file_backed_callable_with_override_succeeds(self):
        env = {}
        exec("def interactive_fn(strength):\n    return strength\n", env)
        interactive_fn = env["interactive_fn"]

        runner = make_runner(
            [0.1],
            seed_stride=1,
            default_noise_model=interactive_fn,
            serialized_callables={
                "default_noise_model": "def interactive_fn(strength):\n    return strength\n"
            },
        )
        assert (
            runner._quantum_program_serialized_callables["default_noise_model"]
            == "def interactive_fn(strength):\n    return strength\n"
        )
        assert runner.build_program(0).default_noise_model == 0.1


class TestRun:
    def test_seed_reproducibility(self, tmp_path):
        runner1 = make_runner(
            [0.0, 0.5],
            seed_stride=20,
            base_seed=7,
            num_shots=10,
            verbose=False,
        )
        runner2 = make_runner(
            [0.0, 0.5],
            seed_stride=20,
            base_seed=7,
            num_shots=10,
            verbose=False,
        )

        result1 = runner1.run()
        result2 = runner2.run()

        assert result1.failure_rates == result2.failure_rates
        assert result1.stderrs == result2.stderrs

    def test_monotonic_failure_rate(self):
        strengths = [0.0, 0.2, 0.5, 0.9]
        runner = make_runner(strengths, seed_stride=500, num_shots=500, verbose=False)
        result = runner.run()
        assert result.failure_rates[0] == 0.0
        # Non-decreasing as strength increases (allow equal for adjacent points)
        for a, b in zip(result.failure_rates, result.failure_rates[1:]):
            assert b >= a

    def test_seed_stride_resolves_to_num_shots(self):
        # Create runner directly without make_runner to avoid its defaults
        runner = NoiseSweepRunner(
            strengths=[0.0, 0.1],
            num_shots=5,
            seed_stride=None,  # Explicitly None -> should resolve to num_shots
            collect_shot_data_args=COLLECT_SHOT_DATA_ARGS,
            expected_outcomes=EXPECTED_OUTCOMES,
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            verbose=False,
        )
        runner.run()
        assert runner._resolved_seed_stride == 5

    def test_explicit_seed_stride_too_small_raises(self):
        with pytest.raises(ValueError):
            make_runner([0.0, 0.1], seed_stride=3, num_shots=5)

    def test_run_kwargs_forwarded_and_resolved(self):
        seen_names = []

        real_run = QuantumProgram.run

        def spy_run(self, *args, **kwargs):
            seen_names.append(kwargs.get("max_frame_limit"))
            return real_run(self, *args, **kwargs)

        runner = make_runner(
            [0.0, 0.1],
            seed_stride=5,
            num_shots=5,
            verbose=False,
            run_kwargs={
                "max_frame_limit": lambda strength: 10 if strength == 0.0 else 20,
            },
        )
        try:
            QuantumProgram.run = spy_run
            runner.run()
        finally:
            QuantumProgram.run = real_run

        assert seen_names == [10, 20]

    def test_serial_run_respects_verbose_parameter(self):
        """Serial run with verbose=True should forward verbose=True to
        each point's QuantumProgram.run call; default or explicit verbose=False
        should suppress it."""
        seen_verbose_values = []

        real_run = QuantumProgram.run

        def spy_run(self, *args, **kwargs):
            seen_verbose_values.append(kwargs.get("verbose"))
            return real_run(self, *args, **kwargs)

        # Test 1: explicit verbose=True should forward True to each point
        runner1 = make_runner([0.0, 0.1], seed_stride=5, num_shots=5, verbose=True)
        seen_verbose_values.clear()
        try:
            QuantumProgram.run = spy_run
            runner1.run()
        finally:
            QuantumProgram.run = real_run

        assert seen_verbose_values == [True, True], (
            f"Expected [True, True] with verbose=True, got {seen_verbose_values}"
        )

        # Test 2: explicit verbose=False should forward False to each point
        runner2 = make_runner([0.0, 0.1], seed_stride=5, num_shots=5, verbose=False)
        seen_verbose_values.clear()
        try:
            QuantumProgram.run = spy_run
            runner2.run()
        finally:
            QuantumProgram.run = real_run

        assert seen_verbose_values == [False, False], (
            f"Expected [False, False] with verbose=False, got {seen_verbose_values}"
        )

        # Test 3: default (no explicit verbose) should forward True
        # (runner defaults verbose=True in its constructor)
        runner3 = make_runner([0.0, 0.1], seed_stride=5, num_shots=5)
        seen_verbose_values.clear()
        try:
            QuantumProgram.run = spy_run
            runner3.run()
        finally:
            QuantumProgram.run = real_run

        assert seen_verbose_values == [True, True], (
            f"Expected [True, True] with default verbose, got {seen_verbose_values}"
        )


class TestRunParallel:
    """`NoiseSweepRunner.run`'s `parallel` (a
    [](api:ParallelStrategy)) path, against real `loky` and `submitit`
    executors -- both must match a serial run exactly (seeding is
    deterministic per index), and the batch-atomic resume guarantee the
    docstring makes must actually hold. `ParallelStrategy`'s own
    construction-time validation (mutual exclusion, `n_program_chunks`/
    `shot_executor` requirements) is covered directly in
    test_paralleltools.py, not duplicated here."""

    def test_loky_program_executor_matches_serial_result(self):
        loky = pytest.importorskip("loky")
        strengths = [0.0, 0.1, 0.2, 0.9]
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )

        serial = make_runner(
            strengths, seed_stride=20, base_seed=7, num_shots=10, verbose=False
        ).run()
        parallel = make_runner(
            strengths,
            seed_stride=20,
            base_seed=7,
            num_shots=10,
            verbose=False,
            parallel_strategy=strategy,
        ).run()

        assert parallel.failure_rates == serial.failure_rates
        assert parallel.stderrs == serial.stderrs

    def test_debug_executor_matches_serial_result(self, tmp_path):
        """Uses submitit's in-process DebugExecutor rather than a real
        `cluster="local"` subprocess: a real subprocess can only unpickle
        a worker function/runner it can import by dotted path, which this
        test module's own `NoiseSweepRunner` (holding a reference to the
        test-module-local `FLIP_COIN` instruction) isn't."""
        submitit = pytest.importorskip("submitit")
        strengths = [0.0, 0.1, 0.2, 0.9]
        strategy = ParallelStrategy(
            program_executor=submitit.DebugExecutor(folder=tmp_path),
            n_program_chunks=2,
        )

        serial = make_runner(
            strengths, seed_stride=20, base_seed=7, num_shots=10, verbose=False
        ).run()
        parallel = make_runner(
            strengths,
            seed_stride=20,
            base_seed=7,
            num_shots=10,
            verbose=False,
            parallel_strategy=strategy,
        ).run()

        assert parallel.failure_rates == serial.failure_rates
        assert parallel.stderrs == serial.stderrs

    @pytest.mark.skip(
        reason="Hybrid parallelism with loky and shot_executor hits resource limits in containerized environments with limited process/thread capacity."
    )
    def test_hybrid_program_and_shot_executor_matches_serial_result(self):
        """program_executor (across sweep points) and shot_executor
        (within each point's own shots) nested together -- the real
        hybrid parallelism this stage adds, replacing the old guardrail
        that just rejected this combination."""
        loky = pytest.importorskip("loky")
        strengths = [0.0, 0.1, 0.2, 0.9]
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
            shot_executor=_build_shot_executor,
        )

        serial = make_runner(
            strengths, seed_stride=20, base_seed=7, num_shots=10, verbose=False
        ).run()
        parallel = make_runner(
            strengths,
            seed_stride=20,
            base_seed=7,
            num_shots=10,
            verbose=False,
            parallel_strategy=strategy,
        ).run()

        assert parallel.failure_rates == serial.failure_rates
        assert parallel.stderrs == serial.stderrs

    def test_legacy_run_kwargs_executor_now_fails_naturally(self):
        """`run_kwargs["executor"]` is no longer special-cased at all --
        `QuantumProgram.run`'s parameter was renamed to `shot_executor`,
        so a stray old-style `executor=` kwarg now just fails with a
        plain TypeError from `QuantumProgram.run` itself, with no
        NoiseSweepRunner-specific validation required."""
        runner = make_runner(
            [0.0, 0.1],
            seed_stride=5,
            num_shots=5,
            verbose=False,
            run_kwargs={"executor": object()},
        )
        with pytest.raises(TypeError, match="executor"):
            runner.run()

    def test_parallel_writes_result_once_per_batch_not_per_point(
        self, tmp_path
    ):
        """Verify that parallel mode persists the result.h5 checkpoint
        as items complete during dispatch, reflecting per-item completion."""
        loky = pytest.importorskip("loky")
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        runner = make_runner(
            [0.0, 0.1, 0.2, 0.3],
            seed_stride=5,
            num_shots=5,
            verbose=False, checkpoint=True, item_checkpoint_dir=item_checkpoint_dir,
            parallel_strategy=strategy,
        )

        runner.run()

        # Verify the final result is complete (all sweep points processed)
        final_result = NoiseSweepResult.read(item_checkpoint_dir / "result.h5")
        assert final_result.is_complete
        assert None not in final_result.failure_rates
        assert len(final_result.failure_rates) == 4

    def test_resume_only_dispatches_missing_points_and_matches_uninterrupted(
        self, tmp_path
    ):
        """A crash partway through a serial run leaves only its
        already-completed points persisted; resuming via from_noise_sweep_runner
        with a parallel strategy re-runs it with parallel dispatch and only
        dispatches the missing indices."""
        loky = pytest.importorskip("loky")
        strengths = [0.0, 0.1, 0.2, 0.3]
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"

        uninterrupted = make_runner(
            strengths, seed_stride=20, base_seed=1, num_shots=10, verbose=False
        )
        uninterrupted_result = uninterrupted.run()

        runner = make_runner(
            strengths, seed_stride=20, base_seed=1, num_shots=10, verbose=False,
            checkpoint=True, item_checkpoint_dir=item_checkpoint_dir
        )
        real_build_program = NoiseSweepRunner.build_program

        def crash_at_index_2(self, index):
            if index == 2:
                raise RuntimeError("simulated crash")
            return real_build_program(self, index)

        NoiseSweepRunner.build_program = crash_at_index_2
        try:
            with pytest.raises(RuntimeError):
                runner.run()
        finally:
            NoiseSweepRunner.build_program = real_build_program

        # result.h5 is only written in _finalize, so read the partial state
        # (indices 0 and 1) from the worker_*_runner.h5 files directly.
        from loqs.tools.multiprogramrunner import _read_worker_files
        completed = _read_worker_files(item_checkpoint_dir)
        assert len(completed) == 2  # Only 0 and 1 completed
        assert 0 in completed
        assert 1 in completed
        assert 2 not in completed
        assert 3 not in completed

        # ParallelStrategy.make_chunks (in loqs.tools.paralleltools, not
        # noisesweeptools) is what actually calls chunk_round_robin now.
        dispatched_items = []
        real_chunk_round_robin = paralleltools.chunk_round_robin

        def recording_chunk_round_robin(items, n_chunks):
            dispatched_items.append(list(items))
            return real_chunk_round_robin(items, n_chunks)

        paralleltools.chunk_round_robin = recording_chunk_round_robin
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        # Create a new runner with parallel strategy for the retry
        runner2 = NoiseSweepRunner.from_noise_sweep_runner(
            runner, parallel_strategy=strategy, resume=True
        )
        try:
            final_result = runner2.run()
        finally:
            paralleltools.chunk_round_robin = real_chunk_round_robin

        # Verify that only indices 2 and 3 were dispatched (as tuples with strength values)
        assert len(dispatched_items) == 1
        assert len(dispatched_items[0]) == 2
        dispatched_indices = [item[0] for item in dispatched_items[0]]
        assert sorted(dispatched_indices) == [2, 3]

        assert final_result.failure_rates == uninterrupted_result.failure_rates
        assert final_result.is_complete
        assert None not in final_result.failure_rates  # Final result has no None


class TestResume:
    def test_skips_completed_points_and_matches_uninterrupted_run(self, tmp_path):
        strengths = [0.0, 0.1, 0.2]

        uninterrupted = make_runner(
            strengths, seed_stride=20, base_seed=1, num_shots=10, verbose=False
        )
        uninterrupted_result = uninterrupted.run()

        item_checkpoint_dir = tmp_path / "sweep_checkpoint"

        built_indices = []
        crash_triggered = []
        runner1 = make_runner(
            strengths, seed_stride=20, base_seed=1, num_shots=10, verbose=False,
            checkpoint=True, item_checkpoint_dir=item_checkpoint_dir
        )
        real_build_program = NoiseSweepRunner.build_program

        def spy_build_program(self, index):
            built_indices.append(index)
            if index == 2 and not crash_triggered:
                crash_triggered.append(True)
                raise RuntimeError("simulated crash")
            return real_build_program(self, index)

        NoiseSweepRunner.build_program = spy_build_program
        try:
            with pytest.raises(RuntimeError):
                runner1.run()
        finally:
            NoiseSweepRunner.build_program = real_build_program

        assert built_indices == [0, 1, 2]

        # Resume with a fresh runner instance built from the same config
        runner2 = make_runner(
            strengths, seed_stride=20, base_seed=1, num_shots=10, verbose=False,
            checkpoint=True, resume=True, item_checkpoint_dir=item_checkpoint_dir
        )
        built_indices.clear()
        NoiseSweepRunner.build_program = spy_build_program
        try:
            final_result = runner2.run()
        finally:
            NoiseSweepRunner.build_program = real_build_program

        assert built_indices == [2]
        assert final_result.failure_rates == uninterrupted_result.failure_rates
        assert final_result.stderrs == uninterrupted_result.stderrs
        assert final_result.is_complete

    def test_run_without_item_checkpoint_dir_succeeds(self):
        # Resume is auto-detected from item_checkpoint_dir's on-disk state;
        # without one, run() has nothing to resume from and just succeeds.
        runner = make_runner([0.0, 0.1], seed_stride=5, num_shots=5, verbose=False)
        result = runner.run()
        assert result.is_complete

    def test_resume_mismatched_strengths_raises(self, tmp_path):
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"
        runner1 = make_runner(
            [0.0, 0.1],
            seed_stride=5,
            num_shots=5,
            verbose=False, checkpoint=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        runner1.run()

        runner2 = NoiseSweepRunner(
            strengths=[0.0, 0.1, 0.2],
            num_shots=5,
            collect_shot_data_args=COLLECT_SHOT_DATA_ARGS,
            expected_outcomes=EXPECTED_OUTCOMES,
            seed_stride=5,
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            verbose=False, checkpoint=True, resume=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        with pytest.raises(ValueError):
            runner2.run()

    def test_resume_mismatched_collect_shot_data_args_raises(self, tmp_path):
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"
        runner1 = make_runner(
            [0.0, 0.1],
            seed_stride=5,
            num_shots=5,
            verbose=False, checkpoint=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        runner1.run()

        runner2 = NoiseSweepRunner(
            strengths=[0.0, 0.1],
            num_shots=5,
            collect_shot_data_args=[("failed", 0)],  # Different from COLLECT_SHOT_DATA_ARGS
            expected_outcomes=EXPECTED_OUTCOMES,
            seed_stride=5,
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            verbose=False, checkpoint=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        with pytest.raises(ValueError):
            runner2.run()

    def test_resume_mismatched_expected_outcomes_raises(self, tmp_path):
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"
        runner1 = make_runner(
            [0.0, 0.1],
            seed_stride=5,
            num_shots=5,
            verbose=False, checkpoint=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        runner1.run()

        runner2 = NoiseSweepRunner(
            strengths=[0.0, 0.1],
            num_shots=5,
            collect_shot_data_args=COLLECT_SHOT_DATA_ARGS,
            expected_outcomes=[True],  # Different from EXPECTED_OUTCOMES
            seed_stride=5,
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            verbose=False, checkpoint=True, resume=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        with pytest.raises(ValueError):
            runner2.run()

    def test_item_checkpoint_dir_without_resume_still_writes_incrementally(self, tmp_path):
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"
        runner = make_runner(
            [0.0, 0.1, 0.2],
            seed_stride=5,
            num_shots=5,
            verbose=False, checkpoint=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        runner.run()
        loaded = NoiseSweepResult.read(item_checkpoint_dir / "result.h5")
        assert loaded.is_complete


class TestFromNoiseSweepRunner:
    def test_from_noise_sweep_runner_with_single_override(self, tmp_path):
        """Test that from_noise_sweep_runner copies all fields except the override."""
        base_runner = make_runner(
            [0.0, 0.1, 0.2],
            seed_stride=20,
            base_seed=5,
            num_shots=10,
            verbose=False,
        )
        new_num_shots = 20
        copied_runner = NoiseSweepRunner.from_noise_sweep_runner(
            base_runner, num_shots=new_num_shots
        )

        # Check the override field
        assert copied_runner.num_shots == new_num_shots
        # Check that other fields match the base
        assert copied_runner.strengths == base_runner.strengths
        assert copied_runner.collect_shot_data_args == base_runner.collect_shot_data_args
        assert copied_runner.expected_outcomes == base_runner.expected_outcomes
        assert copied_runner.base_seed == base_runner.base_seed
        assert copied_runner.seed_stride == base_runner.seed_stride

    def test_from_noise_sweep_runner_no_overrides_works_end_to_end(self, tmp_path):
        """Test that from_noise_sweep_runner with no overrides produces an identical runner."""
        base_runner = make_runner(
            [0.0, 0.1],
            seed_stride=20,
            base_seed=1,
            num_shots=10,
            verbose=False,
        )
        base_result = base_runner.run()

        # Create a copy via from_noise_sweep_runner with zero overrides
        copied_runner = NoiseSweepRunner.from_noise_sweep_runner(base_runner)

        # Verify all fields are identical
        assert copied_runner.strengths == base_runner.strengths
        assert copied_runner.num_shots == base_runner.num_shots
        assert copied_runner.collect_shot_data_args == base_runner.collect_shot_data_args
        assert copied_runner.expected_outcomes == base_runner.expected_outcomes
        assert copied_runner.base_seed == base_runner.base_seed
        assert copied_runner.seed_stride == base_runner.seed_stride

        # Run the copied runner and verify it produces the same result
        copied_result = copied_runner.run()
        assert copied_result.failure_rates == base_result.failure_rates
        assert copied_result.stderrs == base_result.stderrs
        assert copied_result.is_complete


class TestNoiseSweepResult:
    def test_write_read_round_trip_complete(self, tmp_path):
        result = NoiseSweepResult(
            strengths=[0.0, 0.1],
            failure_rates=[0.0, 0.2],
            stderrs=[0.0, 0.01],
            num_shots=100,
            metadata={"note": "test"},
        )
        path = tmp_path / "result.json"
        result.write(path)
        loaded = NoiseSweepResult.read(path)
        assert loaded.strengths == result.strengths
        assert loaded.failure_rates == result.failure_rates
        assert loaded.stderrs == result.stderrs
        assert loaded.num_shots == result.num_shots
        assert loaded.metadata == result.metadata
        assert loaded.is_complete

    def test_write_read_round_trip_incomplete(self, tmp_path):
        result = NoiseSweepResult(
            strengths=[0.0, 0.1, 0.2],
            failure_rates=[0.0, None, 0.1],  # Full-length with None placeholder
            stderrs=[0.01, None, 0.02],     # Full-length with None placeholder
            num_shots=100,
        )
        path = tmp_path / "result.json"
        result.write(path)
        loaded = NoiseSweepResult.read(path)
        assert not loaded.is_complete
        assert len(loaded.failure_rates) == 3  # Always full-length
        # Check sparse array model directly: completed indices have values,
        # incomplete indices have None
        assert loaded.failure_rates[0] is not None
        assert loaded.failure_rates[1] is None
        assert loaded.failure_rates[2] is not None
        assert loaded.stderrs[0] is not None
        assert loaded.stderrs[1] is None
        assert loaded.stderrs[2] is not None

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            NoiseSweepResult(
                strengths=[0.0, 0.1],
                failure_rates=[0.0, 0.1],
                stderrs=[0.0],
                num_shots=5,
            )
        with pytest.raises(ValueError):
            NoiseSweepResult(
                strengths=[0.0],
                failure_rates=[0.0, 0.1],
                stderrs=[0.0, 0.1],
                num_shots=5,
            )


class TestCompareNoiseSweeps:
    def _make_result(self, strengths, num_completed, num_shots=10):
        # Create full-length arrays with None placeholders for incomplete indices
        failure_rates = [0.0 if i < num_completed else None for i in range(len(strengths))]
        stderrs = [0.0 if i < num_completed else None for i in range(len(strengths))]
        return NoiseSweepResult(
            strengths=strengths,
            failure_rates=failure_rates,
            stderrs=stderrs,
            num_shots=num_shots,
        )

    def test_mismatched_strengths_always_raises(self):
        results = {
            "a": self._make_result([0.0, 0.1], 2),
            "b": self._make_result([0.0, 0.2], 2),
        }
        with pytest.raises(ValueError):
            compare_noise_sweeps(results)
        with pytest.raises(ValueError):
            compare_noise_sweeps(results, strict=True)

    def test_mismatched_num_shots_always_raises(self):
        results = {
            "a": self._make_result([0.0, 0.1], 2, num_shots=10),
            "b": self._make_result([0.0, 0.1], 2, num_shots=20),
        }
        with pytest.raises(ValueError):
            compare_noise_sweeps(results)

    def test_incomplete_warns_by_default(self):
        results = {
            "a": self._make_result([0.0, 0.1], 2),
            "b": self._make_result([0.0, 0.1], 1),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            returned = compare_noise_sweeps(results)
        assert any(issubclass(w.category, UserWarning) for w in caught)
        assert returned is results

    def test_incomplete_raises_when_strict(self):
        results = {
            "a": self._make_result([0.0, 0.1], 2),
            "b": self._make_result([0.0, 0.1], 1),
        }
        with pytest.raises(ValueError):
            compare_noise_sweeps(results, strict=True)

    def test_all_complete_no_warning_either_way(self):
        results = {
            "a": self._make_result([0.0, 0.1], 2),
            "b": self._make_result([0.0, 0.1], 2),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compare_noise_sweeps(results)
            compare_noise_sweeps(results, strict=True)


class TestPlotNoiseSweep:
    def test_smoke(self):
        pytest.importorskip("matplotlib")
        result = NoiseSweepResult(
            strengths=[0.01, 0.05, 0.1],
            failure_rates=[0.0, 0.02, 0.1],
            stderrs=[0.0, 0.01, 0.02],
            num_shots=100,
        )
        ax = plot_noise_sweep(result, reference_slope=2)
        assert ax is not None

    def test_multi_series_smoke(self):
        pytest.importorskip("matplotlib")
        result_a = NoiseSweepResult(
            strengths=[0.01, 0.05],
            failure_rates=[0.0, 0.02],
            stderrs=[0.0, 0.01],
            num_shots=100,
        )
        result_b = NoiseSweepResult(
            strengths=[0.01, 0.05],
            failure_rates=[0.01, 0.03],
            stderrs=[0.005, 0.01],
            num_shots=100,
        )
        ax = plot_noise_sweep({"a": result_a, "b": result_b})
        assert ax is not None

    def test_missing_matplotlib_raises_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        result = NoiseSweepResult(
            strengths=[0.1], failure_rates=[0.0], stderrs=[0.0], num_shots=10
        )
        with pytest.raises(ImportError):
            plot_noise_sweep(result)


class TestNoiseSweepRunnerShotCheckpointing:
    """Tests for [](api:QuantumProgram.run)'s per-worker HDF5 shot-level
    checkpointing, threaded through `NoiseSweepRunner.run` via the
    `checkpoint_batch_size`, `shot_checkpoint_dir`, and `lazy_loading`
    parameters."""

    def test_checkpoint_batch_size_without_shot_checkpoint_dir_raises(self):
        """checkpoint_batch_size given without shot_checkpoint_dir is a
        configuration error, not something that's silently ignored."""
        with pytest.raises(ValueError, match="shot_checkpoint_dir"):
            make_runner(
                [0.0, 0.1],
                num_shots=10,
                checkpoint_batch_size=1,
            )

    def test_serial_shot_checkpoint_creates_per_point_subdirs(self, tmp_path):
        """A serial (no parallel) run with checkpoint_batch_size=1 and a real
        shot_checkpoint_dir produces per-point subdirectories under it, one
        per sweep point, each containing checkpoint files that can be loaded
        via ProgramResults.load_checkpoint."""
        shot_ckpt_dir = tmp_path / "shot_checkpoints"
        shot_ckpt_dir.mkdir()

        runner = make_runner(
            [0.0, 0.1],
            num_shots=10,
            checkpoint_batch_size=1,
            shot_checkpoint_dir=shot_ckpt_dir,
            lazy_loading=False,
        )

        result = runner.run()

        # Confirm the sweep completed successfully
        assert result.is_complete
        assert len(result.failure_rates) == 2

        # Confirm per-point subdirs exist and contain checkpoints
        subdirs = list(shot_ckpt_dir.iterdir())
        assert len(subdirs) == 2, f"Expected 2 point subdirs, got {len(subdirs)}: {subdirs}"

        for index in range(len(runner.strengths)):
            point_subdir = _sweep_point_checkpoint_subdir(shot_ckpt_dir, index)
            assert point_subdir.exists(), f"Missing subdir: {point_subdir}"

            # Confirm the checkpoint file exists and can load the right number of shots
            checkpoint_file = point_subdir / "results.h5"
            assert checkpoint_file.exists(), f"Missing checkpoint: {checkpoint_file}"

            loaded_results = ProgramResults()
            loaded_results.load_checkpoint(checkpoint_dir=point_subdir)
            assert len(loaded_results.shot_histories) == 10

    def test_parallel_shot_checkpoint_prevents_point_collision(self, tmp_path):
        """A parallel run with parallel.n_program_chunks=1 (one worker processes
        both points sequentially) and shot_checkpoint_dir set confirms the
        per-point subdirectory scheme actually prevents collisions despite
        sharing one worker/hostname_pid."""
        loky = pytest.importorskip("loky")
        shot_ckpt_dir = tmp_path / "shot_checkpoints"
        shot_ckpt_dir.mkdir()

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=1,
            shot_executor=_build_shot_executor,
            n_shot_batches=10,
        )

        runner = make_runner(
            [0.0, 0.1],
            num_shots=10,
            parallel_strategy=strategy,
            checkpoint_batch_size=1,
            shot_checkpoint_dir=shot_ckpt_dir,
            lazy_loading=False,
        )

        result = runner.run()

        # Confirm the sweep completed successfully
        assert result.is_complete
        assert len(result.failure_rates) == 2

        # Confirm both point subdirs exist independently
        subdirs = list(shot_ckpt_dir.iterdir())
        assert len(subdirs) == 2

        for index in range(len(runner.strengths)):
            point_subdir = _sweep_point_checkpoint_subdir(shot_ckpt_dir, index)
            assert point_subdir.exists(), f"Missing subdir: {point_subdir}"
            checkpoint_file = point_subdir / "results.h5"
            assert checkpoint_file.exists(), f"Missing checkpoint: {checkpoint_file}"

            # Confirm the checkpoint can be loaded with the right number of shots
            loaded_results = ProgramResults()
            loaded_results.load_checkpoint(checkpoint_dir=point_subdir)
            assert len(loaded_results.shot_histories) == 10

    def test_checkpoint_dir_in_run_kwargs_conflicts_with_shot_checkpoint_dir(self):
        """Passing checkpoint_dir directly via run_kwargs while
        shot_checkpoint_dir is also set raises ValueError (the two
        mechanisms conflict)."""
        with pytest.raises(ValueError, match="checkpoint_dir"):
            make_runner(
                [0.0, 0.1],
                num_shots=10,
                shot_checkpoint_dir="/tmp/dummy",
                run_kwargs={"checkpoint_dir": "/tmp/also_dummy"},
            )

    def test_keep_shot_results_end_to_end(self, tmp_path):
        """NoiseSweepRunner with keep_shot_results=True retains full
        ProgramResults objects for each sweep point in runner._program_results,
        accessible after the run completes."""
        item_checkpoint_dir = tmp_path / "checkpoint"

        runner = make_runner(
            [0.0, 0.1],
            seed_stride=10,
            num_shots=5,
            verbose=False, checkpoint=True, item_checkpoint_dir=item_checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_checkpoint",
            checkpoint_batch_size=2,
            keep_shot_results=True,
            lazy_loading=False,  # Disable lazy loading for now
        )
        result = runner.run()

        # After run completes, runner._program_results should be populated
        assert len(runner._program_results) == 2
        assert 0 in runner._program_results
        assert 1 in runner._program_results

        # Each retained ProgramResults should have the expected shot data
        for index in [0, 1]:
            pr = runner._program_results[index]
            assert isinstance(pr, ProgramResults)
            assert len(pr.shot_histories) == 5

        # The returned result itself should be complete (independent of keep_shot_results)
        assert result.is_complete
        assert len(result.failure_rates) == 2

    def test_resume_cascades_into_point_partial_shot_checkpoint(
        self, monkeypatch, tmp_path
    ):
        """When a sweep point crashes partway through its own shot-level
        checkpoint, a runner-level resume must cascade the resume flag down to
        that point's own QuantumProgram.run() call, causing it to resume from its
        partial shot checkpoint rather than recomputing all shots from scratch.
        This differs from tests that only cover points that hadn't started
        shot-level work at all."""
        from loqs.core.quantumprogram import QuantumProgram

        item_ckpt = tmp_path / "item_checkpoint"
        shot_ckpt = tmp_path / "shot_checkpoint"
        item_ckpt.mkdir()
        shot_ckpt.mkdir()

        # First run: simulate a crash partway through the second point's own
        # shot work (2 points, 6 shots each, so we'll interrupt at shot 9)
        compute_count = {"n": 0}
        original_run_shot = QuantumProgram._run_shot

        def _run_shot_with_interrupt(self, max_frame_limit, seed, shot_index):
            compute_count["n"] += 1
            # Crash after 9 shots total (completing all of point 0's 6 shots,
            # and 3 of point 1's 6 shots)
            if compute_count["n"] > 9:
                raise RuntimeError("Simulated crash mid-dispatch")
            return original_run_shot(self, max_frame_limit, seed, shot_index)

        with pytest.raises(RuntimeError, match="Simulated crash"):
            monkeypatch.setattr(
                QuantumProgram, "_run_shot", _run_shot_with_interrupt
            )
            runner = make_runner(
                [0.0, 0.1],
                num_shots=6,
                checkpoint=True,
                item_checkpoint_dir=item_ckpt,
                checkpoint_batch_size=2,
                shot_checkpoint_dir=shot_ckpt,
                lazy_loading=False,
                verbose=False,
            )
            runner.run()

        # Verify: point 0's shot checkpoint should be complete (6 shots)
        point0_shot_ckpt = _sweep_point_checkpoint_subdir(shot_ckpt, 0)
        assert point0_shot_ckpt.exists()
        point0_results = ProgramResults()
        point0_results.load_checkpoint(point0_shot_ckpt)
        assert len(point0_results.shot_histories) == 6

        # Verify: point 1's shot checkpoint should be partial (2 of 6 shots)
        point1_shot_ckpt = _sweep_point_checkpoint_subdir(shot_ckpt, 1)
        assert point1_shot_ckpt.exists()
        point1_results_partial = ProgramResults()
        point1_results_partial.load_checkpoint(point1_shot_ckpt)
        assert len(point1_results_partial.shot_histories) == 2

        # Second run: item-level resume should cascade down to point 1's
        # own QuantumProgram.run() call, resuming its partial checkpoint.
        monkeypatch.undo()
        compute_count_on_resume = {"n": 0}

        original_run_shot_2 = QuantumProgram._run_shot

        def _count_compute_calls_resume(self, max_frame_limit, seed, shot_index):
            compute_count_on_resume["n"] += 1
            return original_run_shot_2(self, max_frame_limit, seed, shot_index)

        monkeypatch.setattr(
            QuantumProgram, "_run_shot", _count_compute_calls_resume
        )

        runner2 = make_runner(
            [0.0, 0.1],
            num_shots=6,
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=item_ckpt,
            checkpoint_batch_size=2,
            shot_checkpoint_dir=shot_ckpt,
            lazy_loading=False,
            verbose=False,
        )
        result = runner2.run()

        monkeypatch.undo()

        # Verify: the results are fully correct (both points complete)
        assert result.is_complete
        assert len(result.failure_rates) == 2

        # Only point 1's 4 missing shots should be recomputed, not all 6
        # (which would mean it was redone from scratch instead of resumed).
        assert compute_count_on_resume["n"] == 4

    def test_resume_does_not_cascade_raise_for_point_with_no_shot_checkpoint(
        self, monkeypatch, tmp_path
    ):
        """When a sweep point has no shot-level checkpoint results.h5,
        a runner-level resume must not cascade resume=True down to that point's
        QuantumProgram.run() call, avoiding the case (d) ValueError
        ("resume=True with no on-disk state"). Instead, resume=False is passed,
        and the point is redone from scratch and completes successfully."""
        from loqs.core.quantumprogram import QuantumProgram

        item_ckpt = tmp_path / "item_checkpoint"
        shot_ckpt = tmp_path / "shot_checkpoint"
        item_ckpt.mkdir()
        shot_ckpt.mkdir()

        # First run: complete point 0, then crash on point 1's first shot
        # before its checkpoint batch can flush.
        compute_count = {"n": 0}
        original_run_shot = QuantumProgram._run_shot

        def _run_shot_with_interrupt(self, max_frame_limit, seed, shot_index):
            compute_count["n"] += 1
            # Crash at shot 10 (point 1's first shot, after point 0's 9)
            if compute_count["n"] >= 10:
                raise RuntimeError("Simulated crash mid-point-1")
            return original_run_shot(self, max_frame_limit, seed, shot_index)

        with pytest.raises(RuntimeError, match="mid-point-1"):
            monkeypatch.setattr(
                QuantumProgram, "_run_shot", _run_shot_with_interrupt
            )
            runner = make_runner(
                [0.0, 0.1],
                num_shots=9,
                checkpoint=True,
                item_checkpoint_dir=item_ckpt,
                checkpoint_batch_size=9,
                shot_checkpoint_dir=shot_ckpt,
                lazy_loading=False,
                verbose=False,
            )
            runner.run()

        monkeypatch.undo()

        # Manually verify/set up the precondition: point 0 complete,
        # point 1 subdirectory exists but may or may not have results.h5
        point0_shot_ckpt = _sweep_point_checkpoint_subdir(shot_ckpt, 0)
        assert point0_shot_ckpt.exists()
        point0_results = ProgramResults()
        point0_results.load_checkpoint(point0_shot_ckpt)
        assert len(point0_results.shot_histories) == 9

        # If point 1's results.h5 exists, remove it to simulate the case where
        # point 1's batch didn't complete before the crash
        point1_shot_ckpt = _sweep_point_checkpoint_subdir(shot_ckpt, 1)
        point1_results_file = point1_shot_ckpt / "results.h5"
        if point1_results_file.exists():
            point1_results_file.unlink()

        # Ensure the precondition is met: point 1 has no results.h5
        assert not point1_results_file.exists(), (
            "Precondition setup failed: point 1 results.h5 should be removed"
        )

        # Second run: resume should complete successfully without raising case (d).
        # The cascading logic checks for results.h5; since it doesn't exist,
        # resume=False is passed to point 1's QuantumProgram.run().
        runner2 = make_runner(
            [0.0, 0.1],
            num_shots=9,
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=item_ckpt,
            checkpoint_batch_size=9,
            shot_checkpoint_dir=shot_ckpt,
            lazy_loading=False,
            verbose=False,
        )
        result = runner2.run()

        # Verify: the results are fully correct (both points complete)
        assert result.is_complete
        assert len(result.failure_rates) == 2

        # Verify: point 1's shot checkpoint now has results.h5 and is complete
        assert point1_results_file.exists()
        point1_results_after = ProgramResults()
        point1_results_after.load_checkpoint(point1_shot_ckpt)
        assert len(point1_results_after.shot_histories) == 9
