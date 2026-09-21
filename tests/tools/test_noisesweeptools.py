"""Tester for loqs.tools.noisesweeptools"""

import inspect
import sys
import warnings

import numpy as np
import pytest

from loqs.core import Frame, Instruction, QuantumProgram
from loqs.backends.state import NumpyStatevectorQuantumState
from loqs.tools import paralleltools
from loqs.tools.noisesweeptools import (
    NoiseSweepResult,
    NoiseSweepRunner,
    compare_noise_sweeps,
    plot_noise_sweep,
)
from loqs.tools.paralleltools import ParallelStrategy


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

    def test_wall_clock_timing_attributes(self):
        """item_wall_clock_times and shot_wall_clock_times are populated after run()."""
        strengths = [0.0, 0.1, 0.2]
        runner = make_runner(
            strengths, seed_stride=20, base_seed=1, num_shots=10, verbose=False
        )
        runner.run()

        # item_wall_clock_times: dict[int, float] with one entry per strength
        assert set(runner.item_wall_clock_times.keys()) == {0, 1, 2}
        for item_time in runner.item_wall_clock_times.values():
            assert isinstance(item_time, float)
            assert item_time > 0

        # shot_wall_clock_times: dict[int, dict[int, float]]
        # same item indices, each with per-shot timing dict
        assert set(runner.shot_wall_clock_times.keys()) == {0, 1, 2}
        for shot_times_dict in runner.shot_wall_clock_times.values():
            assert isinstance(shot_times_dict, dict)
            assert len(shot_times_dict) == 10  # num_shots=10
            for shot_idx, shot_time in shot_times_dict.items():
                assert isinstance(shot_idx, int)
                assert isinstance(shot_time, float)
                assert shot_time > 0


class TestRunParallel:
    """`NoiseSweepRunner.run`'s `parallel` (a
    [](api:ParallelStrategy)) path, against real `loky` and `submitit`
    executors -- both must match a serial run exactly (seeding is
    deterministic per index), and the batch-atomic resume guarantee the
    docstring makes must actually hold. `ParallelStrategy`'s own
    construction-time validation (mutual exclusion, `n_program_chunks`/
    `shot_executor` requirements) is covered directly in
    test_paralleltools.py, not duplicated here."""

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

        # Read the partial state (indices 0 and 1) from the worker_*_runner.h5
        # files directly.
        from loqs.internal.streamingmerge import read_checkpoint_dict_attr_union
        completed = read_checkpoint_dict_attr_union(
            item_checkpoint_dir, None, "worker_*_runner.h5", "results"
        )
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

    def test_resume_with_default_seed_stride_resolving_to_num_shots(self, tmp_path):
        """Resume should succeed when effective seeding is identical: one runner
        uses seed_stride=None (resolves to num_shots), another uses explicit seed_stride
        equal to num_shots."""
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"

        # First runner: seed_stride=None, which resolves to num_shots=5
        runner1 = NoiseSweepRunner(
            strengths=[0.0, 0.1],
            num_shots=5,
            collect_shot_data_args=COLLECT_SHOT_DATA_ARGS,
            expected_outcomes=EXPECTED_OUTCOMES,
            seed_stride=None,  # Explicitly None -> will resolve to num_shots=5
            base_seed=0,
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            verbose=False, checkpoint=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        runner1.run()

        # Second runner: explicit seed_stride=5 (which equals num_shots)
        # This has the same effective seeding as runner1, so resume should succeed
        runner2 = NoiseSweepRunner(
            strengths=[0.0, 0.1],
            num_shots=5,
            collect_shot_data_args=COLLECT_SHOT_DATA_ARGS,
            expected_outcomes=EXPECTED_OUTCOMES,
            seed_stride=5,  # Explicitly set to match resolved value
            base_seed=0,
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            verbose=False, checkpoint=True, resume=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        # This should NOT raise ValueError about seed_stride mismatch
        result = runner2.run()
        assert result.is_complete

    def test_resume_with_different_resolved_seed_stride_still_raises(self, tmp_path):
        """Verify that a genuine mismatch in effective seeding (different _resolved_seed_stride)
        is still caught even though we now check _resolved_seed_stride instead of seed_stride."""
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"

        # First runner: seed_stride=None, which resolves to num_shots=5
        runner1 = NoiseSweepRunner(
            strengths=[0.0, 0.1],
            num_shots=5,
            collect_shot_data_args=COLLECT_SHOT_DATA_ARGS,
            expected_outcomes=EXPECTED_OUTCOMES,
            seed_stride=None,  # Explicitly None -> will resolve to num_shots=5
            base_seed=0,
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            verbose=False, checkpoint=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        runner1.run()

        # Second runner: seed_stride=None but different num_shots=10
        # This has different effective seeding (_resolved_seed_stride would be 10, not 5)
        runner2 = NoiseSweepRunner(
            strengths=[0.0, 0.1],
            num_shots=10,  # Different num_shots -> different _resolved_seed_stride
            collect_shot_data_args=COLLECT_SHOT_DATA_ARGS,
            expected_outcomes=EXPECTED_OUTCOMES,
            seed_stride=None,
            base_seed=0,
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            verbose=False, checkpoint=True, resume=True, item_checkpoint_dir=item_checkpoint_dir,
        )
        # This SHOULD raise ValueError about seed_stride mismatch
        with pytest.raises(ValueError, match="seed_stride"):
            runner2.run()

    def test_resume_with_equivalent_expected_outcomes_succeeds(
        self, tmp_path
    ):
        """Resume succeeds when expected_outcomes is passed as an
        equivalent-but-differently-typed sequence (e.g. list vs tuple)."""
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"
        # Run with list form
        runner1 = make_runner(
            [0.0, 0.1],
            seed_stride=5,
            num_shots=5,
            verbose=False,
            checkpoint=True,
            item_checkpoint_dir=item_checkpoint_dir,
            expected_outcomes=[False],
        )
        runner1.run()

        # Resume with tuple form
        runner2 = NoiseSweepRunner(
            strengths=[0.0, 0.1],
            num_shots=5,
            collect_shot_data_args=COLLECT_SHOT_DATA_ARGS,
            expected_outcomes=(False,),  # tuple instead of list
            seed_stride=5,
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            verbose=False,
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=item_checkpoint_dir,
        )
        # Should not raise despite differently-typed expected_outcomes
        result = runner2.run()
        assert result is not None


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

    def test_from_noise_sweep_runner_preserves_and_overrides_runner_filename(
        self, tmp_path
    ):
        """from_noise_sweep_runner preserves runner_filename by default and
        correctly applies an explicit override."""
        custom_file = "my_custom_runner.h5"
        base_runner = make_runner(
            [0.0, 0.1],
            seed_stride=5,
            num_shots=5,
            verbose=False,
            runner_filename=custom_file,
        )
        assert base_runner.runner_filename == custom_file

        # Copy with no override should preserve the custom filename
        copied_runner = NoiseSweepRunner.from_noise_sweep_runner(base_runner)
        assert copied_runner.runner_filename == custom_file

        # Copy with an explicit override should use the new filename
        new_file = "another_runner.h5"
        overridden_runner = NoiseSweepRunner.from_noise_sweep_runner(
            base_runner, runner_filename=new_file
        )
        assert overridden_runner.runner_filename == new_file


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

    def test_incomplete_missing_point_count_correct(self):
        """Regression test for bug where missing_point_count was always 0.

        Verifies that the warning message correctly reports the count of None
        entries in failure_rates, not len(strengths) - len(failure_rates).
        """
        results = {
            "partial": self._make_result([0.0, 0.1, 0.2, 0.3, 0.4], 3),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compare_noise_sweeps(results)
        assert len(caught) == 1
        warning_msg = str(caught[0].message)
        # Should report 2 missing points, not 0
        assert (
            "2 point(s) missing" in warning_msg
        ), f"Expected '2 point(s) missing' in warning, got: {warning_msg}"


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

    def test_incomplete_points_not_plotted(self):
        """Regression test for bug where None/nan values leak into plotted data.

        Verifies that incomplete points (None values in failure_rates/stderrs)
        are excluded from both the main line/errorbar plot and the zero-rate
        open-marker plot, and do not appear in the guide line fit.
        """
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        # Create a result with a mix of valid points and None (incomplete) points.
        # Pattern: valid zero, valid nonzero, None (incomplete), valid nonzero
        result = NoiseSweepResult(
            strengths=[0.01, 0.05, 0.1, 0.2],
            failure_rates=[0.0, 0.05, None, 0.2],
            stderrs=[0.0, 0.01, None, 0.05],
            num_shots=100,
        )
        fig, ax = plt.subplots()
        ax = plot_noise_sweep(result, ax=ax, reference_slope=2)

        # Check that plotted data contains no nan values.
        # The line is drawn only for nonzero points (index 1, 3).
        assert len(ax.lines) > 0, "Expected at least one line to be plotted"
        main_line = ax.lines[0]
        x_data = main_line.get_xdata()
        y_data = main_line.get_ydata()

        # No nan should appear in the plotted data
        assert not np.any(np.isnan(x_data)), "Found nan in line x_data"
        assert not np.any(np.isnan(y_data)), "Found nan in line y_data"

        # The main line should contain the two nonzero valid points (indices 1, 3)
        assert len(x_data) == 2, "Expected 2 points in main line"

        # Verify guide line exists and contains no nan
        assert any(
            "slope=2" in str(getattr(line, "_label", "")) for line in ax.lines
        ), (
            "Expected guide line with slope=2 label"
        )

        plt.close(fig)

    def test_incomplete_points_guide_line_excludes_nan(self):
        """Verify the guide line's x-range is bounded by valid points only, not by an
        incomplete point's strength value even when that strength is the series' own extreme."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        # The last, most extreme strength (0.5) is incomplete -- a guide line that still uses
        # it as its own x-bound reveals the incomplete point's strength leaking into the fit.
        result = NoiseSweepResult(
            strengths=[0.01, 0.05, 0.1, 0.5],
            failure_rates=[0.01, 0.02, 0.03, None],
            stderrs=[0.005, 0.005, 0.005, None],
            num_shots=100,
        )
        fig, ax = plt.subplots()
        ax = plot_noise_sweep(result, ax=ax, reference_slope=2)

        guide_lines = [
            line
            for line in ax.lines
            if getattr(line, "_label", "").startswith("slope")
        ]
        assert len(guide_lines) > 0, "Expected guide line to be plotted despite incomplete points"

        guide_line = guide_lines[0]
        guide_x = guide_line.get_xdata()
        guide_y = guide_line.get_ydata()

        assert len(guide_x) == 2, "Expected 2-point guide line"
        assert not np.any(np.isnan(guide_x)), "Found nan in guide line x_data"
        assert not np.any(np.isnan(guide_y)), "Found nan in guide line y_data"
        # The guide line's upper bound must be the largest *valid* strength (0.1),
        # not the incomplete point's own strength (0.5).
        assert guide_x.max() == pytest.approx(0.1)

        plt.close(fig)


class TestNoiseSweepRunnerShotCheckpointing:
    """Tests for [](api:QuantumProgram.run)'s per-worker HDF5 shot-level
    checkpointing, threaded through `NoiseSweepRunner.run` via the
    `shot_checkpoint`, `shot_checkpoint_dir`, and `lazy_loading`
    parameters."""

    def test_from_noise_sweep_runner_with_serialized_callables(self):
        """from_noise_sweep_runner should preserve serialized_callables."""
        env = {}
        exec("def interactive_fn(strength):\n    return strength\n", env)
        interactive_fn = env["interactive_fn"]

        runner1 = make_runner(
            [0.1],
            seed_stride=1,
            default_noise_model=interactive_fn,
            serialized_callables={
                "default_noise_model": "def interactive_fn(strength):\n    return strength\n"
            },
        )

        # from_noise_sweep_runner without explicit serialized_callables should
        # preserve the original runner's serialized_callables
        runner2 = NoiseSweepRunner.from_noise_sweep_runner(
            runner1, strengths=[0.2]
        )

        # Should NOT raise OSError when trying to serialize/deserialize
        # (which would happen if serialized_callables was lost)
        assert (
            runner2._quantum_program_serialized_callables[
                "default_noise_model"
            ]
            == "def interactive_fn(strength):\n    return strength\n"
        )

        # Verify build_program works (would fail if callables weren't preserved)
        program = runner2.build_program(0)
        assert program.default_noise_model == 0.2


class TestNoiseSweepRunnerHooks:
    """Direct unit tests of NoiseSweepRunner's own derived reduce_program_outcomes/
    _build_output/CHKPT_SUBDIR_PREFIX/mismatch-field hook implementations, in
    isolation from MultiProgramRunner's generic dispatch/checkpoint/resume/parallel
    machinery -- that generic behavior is covered once, generically, in
    test_multiprogramrunner.py."""

    def test_reduce_program_outcomes_computes_failure_rate(self):
        """reduce_program_outcomes turns one program's shot outcomes into
        a (failure_rate, stderr) tuple via _compute_failure_rate, independent
        of any dispatch/checkpoint machinery."""
        runner = make_runner(
            [0.1],
            seed_stride=20,
            num_shots=20,
        )
        program = runner.build_program(0)
        program_results = program.run(num_shots=20, verbose=False)
        failure_rate, stderr = runner.reduce_program_outcomes(program_results)
        # Both should be numeric values in [0, 1]
        assert isinstance(failure_rate, float)
        assert isinstance(stderr, float)
        assert 0 <= failure_rate <= 1
        assert 0 <= stderr <= 1

    def test_build_output_assembles_noise_sweep_result_without_disk_write(
        self, tmp_path
    ):
        """_build_output builds a NoiseSweepResult from (strength, (failure_rate,
        stderr)) pairs without writing to disk, even when item_checkpoint_dir is set."""
        runner = make_runner(
            [0.1, 0.2, 0.3],
            seed_stride=10,
            num_shots=10,
            item_checkpoint_dir=tmp_path / "ckpt",
            checkpoint=True,
        )
        # Build hand-crafted ordered_results: (strength, (failure_rate, stderr)) pairs
        ordered_results = [
            (0.1, (0.1, 0.05)),
            (0.2, (0.2, 0.06)),
            (0.3, None),  # Incomplete point
        ]
        result = runner._build_output(ordered_results)
        assert isinstance(result, NoiseSweepResult)
        assert result.failure_rates == [0.1, 0.2, None]
        assert result.stderrs == [0.05, 0.06, None]
        assert result.strengths == [0.1, 0.2, 0.3]
        # Verify no result.h5 file was created
        assert not (tmp_path.exists() and any(tmp_path.glob("**/result.h5")))

    def test_chkpt_subdir_prefix_is_point(self):
        """NoiseSweepRunner.CHKPT_SUBDIR_PREFIX should be "point"."""
        assert NoiseSweepRunner.CHKPT_SUBDIR_PREFIX == "point"

    def test_mismatch_check_fields_lists_expected_fields(self):
        """_mismatch_check_fields returns the expected set of fields for resume
        consistency checking."""
        runner = make_runner(
            [0.1],
            seed_stride=10,
            num_shots=10,
        )
        fields = runner._mismatch_check_fields()
        assert fields == [
            "strengths",
            "base_seed",
            "_resolved_seed_stride",
            "num_shots",
            "_normalized_collect_shot_data_args",
            "expected_outcomes",
            "keep_shot_results",
        ]

    def test_mismatch_field_display_name_maps_resolved_seed_stride(self):
        """_mismatch_field_display_name maps internal field names to public names."""
        runner = make_runner(
            [0.1],
            seed_stride=10,
            num_shots=10,
        )
        assert runner._mismatch_field_display_name("_resolved_seed_stride") == "seed_stride"
