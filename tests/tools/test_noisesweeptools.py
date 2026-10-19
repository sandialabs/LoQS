"""Tester for loqs.tools.noisesweeptools"""

import contextlib
import inspect
import sys
import warnings

import numpy as np
import pytest

from loqs.core import Frame, Instruction, QuantumProgram
from loqs.backends.state import NumpyStatevectorQuantumState
from loqs.internal.checkpointstore import ITEM_LAYOUT, CheckpointStore
from loqs.tools import paralleltools
from loqs.tools.multiprogramrunner import CheckpointConfig
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
    """ "Fail" a shot with probability `fail_prob`, deterministically from `seed`."""
    rng = np.random.default_rng(seed)
    return Frame({"failed": bool(rng.random() < fail_prob)})


FLIP_COIN = Instruction(apply_fn=_flip_coin_apply, name="Flip Coin")


def make_stack(fail_prob):
    """Build a one-instruction stack whose failure probability is `fail_prob`."""
    return [{"instruction": "Flip Coin", "fail_prob": fail_prob}]


def identity_noise_model(strength):
    """A trivial "noise model" callable -- real module-level `def`, not a lambda, since
    lambdas aren't properly supported by NoiseSweepRunner's source-based serialization
    (`inspect.getsource` returns the whole call-site line for a lambda, not just its body).
    """
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

INTERACTIVE_FN_SOURCE = "def interactive_fn(strength):\n    return strength\n"


@pytest.fixture
def interactive_fn():
    """A notebook-style function with no real source file, so
    inspect.getsource fails with OSError or a subclass of it."""
    env = {}
    exec(INTERACTIVE_FN_SOURCE, env)
    return env["interactive_fn"]


@contextlib.contextmanager
def _spy_program_run(seen, kwarg):
    """Append each QuantumProgram.run call's `kwarg` kwarg to `seen`."""
    real_run = QuantumProgram.run

    def spy_run(self, *args, **kwargs):
        seen.append(kwargs.get(kwarg))
        return real_run(self, *args, **kwargs)

    QuantumProgram.run = spy_run
    try:
        yield
    finally:
        QuantumProgram.run = real_run


@contextlib.contextmanager
def _hooked_build_program(before):
    """Call `before(index)` ahead of every NoiseSweepRunner.build_program."""
    real_build_program = NoiseSweepRunner.build_program

    def hooked_build_program(self, index):
        before(index)
        return real_build_program(self, index)

    NoiseSweepRunner.build_program = hooked_build_program
    try:
        yield
    finally:
        NoiseSweepRunner.build_program = real_build_program


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
        assert (
            program0.instruction_stack.pop_instruction()[0]["fail_prob"]
            == 0.01
        )

    def test_default_base_seed_rejected(self):
        with pytest.raises(TypeError):
            make_runner([0.1], default_base_seed=5)

    def test_state_type_fixed_class_is_not_treated_as_callable(self):
        runner = make_runner(
            [0.1], seed_stride=1, state_type=NumpyStatevectorQuantumState
        )
        assert (
            runner._quantum_program_values["state_type"]
            is NumpyStatevectorQuantumState
        )
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
        assert not missing


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

    def test_non_file_backed_callable_raises_without_override(
        self, interactive_fn
    ):
        with pytest.raises(OSError):
            make_runner(
                [0.1], seed_stride=1, default_noise_model=interactive_fn
            )

    def test_non_file_backed_callable_with_override_succeeds(
        self, interactive_fn
    ):
        runner = make_runner(
            [0.1],
            seed_stride=1,
            default_noise_model=interactive_fn,
            serialized_callables={
                "default_noise_model": INTERACTIVE_FN_SOURCE
            },
        )
        assert (
            runner._quantum_program_serialized_callables["default_noise_model"]
            == INTERACTIVE_FN_SOURCE
        )
        assert runner.build_program(0).default_noise_model == 0.1


class TestRun:
    def test_seed_reproducibility(self):
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
        runner = make_runner(
            strengths, seed_stride=500, num_shots=500, verbose=False
        )
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

    @pytest.mark.parametrize(
        "verbose_kwargs, expected",
        [({"verbose": True}, True), ({"verbose": False}, False), ({}, True)],
        ids=["explicit_true", "explicit_false", "default_true"],
    )
    def test_serial_run_respects_verbose_parameter(
        self, verbose_kwargs, expected
    ):
        """Serial run forwards the runner's `verbose` (default True) to each
        point's QuantumProgram.run call."""
        seen_verbose_values = []
        runner = make_runner(
            [0.0, 0.1], seed_stride=5, num_shots=5, **verbose_kwargs
        )
        with _spy_program_run(seen_verbose_values, "verbose"):
            runner.run()

        assert seen_verbose_values == [expected, expected]


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
        already-completed points persisted; resuming with a parallel strategy
        re-runs it with parallel dispatch and only dispatches the missing
        indices."""
        loky = pytest.importorskip("loky")
        strengths = [0.0, 0.1, 0.2, 0.3]
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"

        uninterrupted = make_runner(
            strengths, seed_stride=20, base_seed=1, num_shots=10, verbose=False
        )
        uninterrupted_result = uninterrupted.run()

        runner = make_runner(
            strengths,
            seed_stride=20,
            base_seed=1,
            num_shots=10,
            verbose=False,
            config=CheckpointConfig(item_checkpoint_dir=item_checkpoint_dir),
        )

        def crash_at_index_2(index):
            if index == 2:
                raise RuntimeError("simulated crash")

        with _hooked_build_program(crash_at_index_2):
            with pytest.raises(RuntimeError):
                runner.run()

        # The partial state (indices 0 and 1) comes from the per-item
        # payload files, read through the item store.
        completed = CheckpointStore(item_checkpoint_dir, ITEM_LAYOUT).keys(
            strict=False
        )
        assert completed == [0, 1]  # Only 0 and 1 completed

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
        runner2 = make_runner(
            strengths,
            seed_stride=20,
            base_seed=1,
            num_shots=10,
            verbose=False,
            config=CheckpointConfig(
                item_checkpoint_dir=item_checkpoint_dir, resume=True
            ),
            parallel_strategy=strategy,
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
        assert (
            None not in final_result.failure_rates
        )  # Final result has no None


class TestResume:
    def test_skips_completed_points_and_matches_uninterrupted_run(
        self, tmp_path
    ):
        strengths = [0.0, 0.1, 0.2]

        uninterrupted = make_runner(
            strengths, seed_stride=20, base_seed=1, num_shots=10, verbose=False
        )
        uninterrupted_result = uninterrupted.run()

        item_checkpoint_dir = tmp_path / "sweep_checkpoint"

        built_indices = []
        crash_triggered = []
        runner1 = make_runner(
            strengths,
            seed_stride=20,
            base_seed=1,
            num_shots=10,
            verbose=False,
            config=CheckpointConfig(item_checkpoint_dir=item_checkpoint_dir),
        )

        def spy_build_program(index):
            built_indices.append(index)
            if index == 2 and not crash_triggered:
                crash_triggered.append(True)
                raise RuntimeError("simulated crash")

        with _hooked_build_program(spy_build_program):
            with pytest.raises(RuntimeError):
                runner1.run()

        assert built_indices == [0, 1, 2]

        # Resume with a fresh runner instance built from the same config
        runner2 = make_runner(
            strengths,
            seed_stride=20,
            base_seed=1,
            num_shots=10,
            verbose=False,
            config=CheckpointConfig(
                item_checkpoint_dir=item_checkpoint_dir, resume=True
            ),
        )
        built_indices.clear()
        with _hooked_build_program(spy_build_program):
            final_result = runner2.run()

        assert built_indices == [2]
        assert final_result.failure_rates == uninterrupted_result.failure_rates
        assert final_result.stderrs == uninterrupted_result.stderrs
        assert final_result.is_complete

    @pytest.mark.parametrize(
        "second_overrides, expectation",
        [
            # seed_stride=None resolves to num_shots=5, so seed_stride=5 is
            # the same effective seeding.
            ({"seed_stride": 5}, contextlib.nullcontext()),
            # Different num_shots -> different _resolved_seed_stride.
            (
                {"num_shots": 10},
                pytest.raises(ValueError, match="seed_stride"),
            ),
            # Equivalent-but-differently-typed expected_outcomes.
            ({"expected_outcomes": (False,)}, contextlib.nullcontext()),
        ],
        ids=[
            "default_seed_stride_resolving_to_num_shots",
            "different_resolved_seed_stride_raises",
            "equivalent_expected_outcomes",
        ],
    )
    def test_resume_checks_effective_config(
        self, tmp_path, second_overrides, expectation
    ):
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"
        runner_kwargs = dict(
            strengths=[0.0, 0.1],
            num_shots=5,
            collect_shot_data_args=COLLECT_SHOT_DATA_ARGS,
            expected_outcomes=[False],
            seed_stride=None,
            base_seed=0,
            instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
            global_instructions={"Flip Coin": FLIP_COIN},
            verbose=False,
        )
        NoiseSweepRunner(
            **runner_kwargs,
            config=CheckpointConfig(item_checkpoint_dir=item_checkpoint_dir),
        ).run()

        runner2 = NoiseSweepRunner(
            **{**runner_kwargs, **second_overrides},
            config=CheckpointConfig(
                item_checkpoint_dir=item_checkpoint_dir, resume=True
            ),
        )
        with expectation:
            assert runner2.run().is_complete


class TestNoiseSweepResult:
    @pytest.mark.parametrize(
        "result, expected_complete",
        [
            (
                NoiseSweepResult(
                    strengths=[0.0, 0.1],
                    failure_rates=[0.0, 0.2],
                    stderrs=[0.0, 0.01],
                    num_shots=100,
                    metadata={"note": "test"},
                ),
                True,
            ),
            # Full-length with None placeholders for incomplete points
            (
                NoiseSweepResult(
                    strengths=[0.0, 0.1, 0.2],
                    failure_rates=[0.0, None, 0.1],
                    stderrs=[0.01, None, 0.02],
                    num_shots=100,
                ),
                False,
            ),
        ],
        ids=["complete", "incomplete"],
    )
    def test_write_read_round_trip(self, tmp_path, result, expected_complete):
        path = tmp_path / "result.json"
        result.write(path)
        loaded = NoiseSweepResult.read(path)
        assert loaded.strengths == result.strengths
        assert loaded.failure_rates == result.failure_rates
        assert loaded.stderrs == result.stderrs
        assert loaded.num_shots == result.num_shots
        assert loaded.metadata == result.metadata
        assert loaded.is_complete is expected_complete
        assert len(loaded.failure_rates) == len(result.strengths)

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
        failure_rates = [
            0.0 if i < num_completed else None for i in range(len(strengths))
        ]
        stderrs = [
            0.0 if i < num_completed else None for i in range(len(strengths))
        ]
        return NoiseSweepResult(
            strengths=strengths,
            failure_rates=failure_rates,
            stderrs=stderrs,
            num_shots=num_shots,
        )

    @pytest.mark.parametrize("strict", [False, True])
    @pytest.mark.parametrize(
        "b_strengths, b_num_shots",
        [([0.0, 0.2], 10), ([0.0, 0.1], 20)],
        ids=["mismatched_strengths", "mismatched_num_shots"],
    )
    def test_mismatched_series_always_raises(
        self, b_strengths, b_num_shots, strict
    ):
        results = {
            "a": self._make_result([0.0, 0.1], 2, num_shots=10),
            "b": self._make_result(b_strengths, 2, num_shots=b_num_shots),
        }
        with pytest.raises(ValueError):
            compare_noise_sweeps(results, strict=strict)

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
        assert "2 point(s) missing" in warning_msg


class TestPlotNoiseSweep:
    @pytest.mark.parametrize(
        "results, kwargs",
        [
            (
                NoiseSweepResult(
                    strengths=[0.01, 0.05, 0.1],
                    failure_rates=[0.0, 0.02, 0.1],
                    stderrs=[0.0, 0.01, 0.02],
                    num_shots=100,
                ),
                {"reference_slope": 2},
            ),
            (
                {
                    "a": NoiseSweepResult(
                        strengths=[0.01, 0.05],
                        failure_rates=[0.0, 0.02],
                        stderrs=[0.0, 0.01],
                        num_shots=100,
                    ),
                    "b": NoiseSweepResult(
                        strengths=[0.01, 0.05],
                        failure_rates=[0.01, 0.03],
                        stderrs=[0.005, 0.01],
                        num_shots=100,
                    ),
                },
                {},
            ),
        ],
        ids=["single_series", "multi_series"],
    )
    def test_smoke(self, results, kwargs):
        pytest.importorskip("matplotlib")
        ax = plot_noise_sweep(results, **kwargs)
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
        assert len(ax.lines) > 0
        main_line = ax.lines[0]
        x_data = main_line.get_xdata()
        y_data = main_line.get_ydata()

        # No nan should appear in the plotted data
        assert not np.any(np.isnan(x_data))
        assert not np.any(np.isnan(y_data))

        # The main line should contain the two nonzero valid points (indices 1, 3)
        assert len(x_data) == 2

        # Verify guide line exists and contains no nan
        assert any(
            "slope=2" in str(getattr(line, "_label", "")) for line in ax.lines
        )

        plt.close(fig)

    def test_incomplete_points_guide_line_excludes_nan(self):
        """Verify the guide line's x-range is bounded by valid points only, not by an
        incomplete point's strength value even when that strength is the series' own extreme.
        """
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
        assert len(guide_lines) > 0

        guide_line = guide_lines[0]
        guide_x = guide_line.get_xdata()
        guide_y = guide_line.get_ydata()

        assert len(guide_x) == 2
        assert not np.any(np.isnan(guide_x))
        assert not np.any(np.isnan(guide_y))
        # The guide line's upper bound must be the largest *valid* strength (0.1),
        # not the incomplete point's own strength (0.5).
        assert guide_x.max() == pytest.approx(0.1)

        plt.close(fig)


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

    def test_reduce_program_outcomes_raises_on_missing_shot(self):
        """A results object with fewer shots than `num_shots` raises
        instead of reporting a failure rate biased low."""
        runner = make_runner([0.1], seed_stride=20, num_shots=2)
        program = runner.build_program(0)
        program_results = program.run(num_shots=1, verbose=False)
        with pytest.raises(RuntimeError, match="1.*2"):
            runner.reduce_program_outcomes(program_results)

    def test_build_output_assembles_noise_sweep_result_without_disk_write(
        self, tmp_path
    ):
        """_build_output builds a NoiseSweepResult from (strength, (failure_rate,
        stderr)) pairs without writing to disk, even when item_checkpoint_dir is set.
        """
        runner = make_runner(
            [0.1, 0.2, 0.3],
            seed_stride=10,
            num_shots=10,
            config=CheckpointConfig(item_checkpoint_dir=tmp_path / "ckpt"),
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
        # Verify no file was written
        assert not any(path.is_file() for path in tmp_path.rglob("*"))

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
        assert (
            runner._mismatch_field_display_name("_resolved_seed_stride")
            == "seed_stride"
        )
