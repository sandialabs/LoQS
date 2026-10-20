"""Tester for loqs.tools.noisesweeptools"""

import contextlib
import inspect
import math
import sys
import warnings

import numpy as np
import pytest

from loqs.core import Frame, Instruction, QuantumProgram
from loqs.backends.state import NumpyStatevectorQuantumState
from loqs.tools import paralleltools
from loqs.tools.multiprogramrunner import CheckpointConfig
from loqs.tools.noisesweeptools import (
    NoiseSweepResult,
    NoiseSweepRunner,
    compare_noise_sweeps,
    plot_noise_sweep,
)
from loqs.tools.paralleltools import ParallelStrategy
from tests.tools._shared_checkpoint_test_helpers import (
    InlineExecutor,
    inline_execution_patches,
)

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


def pick_state_type(strength):
    """Real module-level `def` returning a fixed state type per point."""
    return NumpyStatevectorQuantumState


def make_runner(strengths, **kwargs):
    kwargs.setdefault("instruction_stack", make_stack)
    kwargs.setdefault("global_instructions", {"Flip Coin": FLIP_COIN})
    # Default num_shots/seed_stride always satisfy num_shots <= seed_stride: derive
    # num_shots from an explicit seed_stride, or fall back to fixed defaults.
    if kwargs.get("seed_stride") is not None:
        kwargs.setdefault("num_shots", max(1, kwargs["seed_stride"]))
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
    @pytest.mark.parametrize(
        "seed_stride, num_shots, stride",
        [(100, 10, 100), (None, 5, 5)],
        ids=["explicit_stride", "default_stride"],
    )
    def test_seed_formula(self, seed_stride, num_shots, stride):
        runner = make_runner(
            [0.0, 0.1, 0.2],
            base_seed=5,
            seed_stride=seed_stride,
            num_shots=num_shots,
        )
        for index in range(3):
            program = runner.build_program(index)
            assert program.default_base_seed == 5 + index * stride

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

    @pytest.mark.parametrize(
        "state_type",
        [NumpyStatevectorQuantumState, pick_state_type],
        ids=["class", "function"],
    )
    def test_state_type_class_vs_callable(self, state_type):
        runner = make_runner([0.1], seed_stride=1, state_type=state_type)
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
        assert runner.build_program(0).default_noise_model == 0.1


class TestRun:
    def test_failure_rate_and_stderr(self):
        """Strengths 0.0 and 1.0 fail no shot and every shot, so their rates
        need no seeded draw; every point's stderr follows from its own rate."""
        num_shots = 20
        runner = make_runner(
            [0.0, 0.5, 1.0],
            seed_stride=num_shots,
            num_shots=num_shots,
            verbose=False,
        )
        result = runner.run()
        assert result.failure_rates[0] == 0.0
        assert result.failure_rates[2] == 1.0
        for rate, stderr in zip(result.failure_rates, result.stderrs):
            assert stderr == math.sqrt(rate * (1 - rate) / num_shots)

    def test_explicit_seed_stride_too_small_raises(self):
        with pytest.raises(ValueError):
            make_runner([0.0, 0.1], seed_stride=3, num_shots=5)

    def test_serial_run_forwards_verbose_parameter(self):
        """Serial run forwards the runner's `verbose` to each point's
        QuantumProgram.run call (False differs from the default)."""
        seen_verbose_values = []
        runner = make_runner(
            [0.0, 0.1], seed_stride=5, num_shots=5, verbose=False
        )
        with _spy_program_run(seen_verbose_values, "verbose"):
            runner.run()

        assert seen_verbose_values == [False, False]


class TestResume:
    def test_resume_matches_uninterrupted_run(self, tmp_path, monkeypatch):
        """A serial run crashes at point 2; a fresh runner resumes with a
        parallel strategy, dispatches only points 2 and 3, and matches an
        uninterrupted run (seeding is per index)."""
        strengths = [0.0, 0.1, 0.2, 0.3]
        item_checkpoint_dir = tmp_path / "sweep_checkpoint"
        runner_kwargs = dict(
            seed_stride=20, base_seed=1, num_shots=10, verbose=False
        )

        uninterrupted_result = make_runner(strengths, **runner_kwargs).run()

        runner = make_runner(
            strengths,
            config=CheckpointConfig(item_checkpoint_dir=item_checkpoint_dir),
            **runner_kwargs,
        )

        def crash_at_index_2(index):
            if index == 2:
                raise RuntimeError("simulated crash")

        with _hooked_build_program(crash_at_index_2):
            with pytest.raises(RuntimeError):
                runner.run()

        dispatched_items = []
        real_chunk_round_robin = paralleltools.chunk_round_robin

        def recording_chunk_round_robin(items, n_chunks):
            dispatched_items.append(list(items))
            return real_chunk_round_robin(items, n_chunks)

        monkeypatch.setattr(
            paralleltools, "chunk_round_robin", recording_chunk_round_robin
        )
        with inline_execution_patches():
            runner2 = make_runner(
                strengths,
                config=CheckpointConfig(
                    item_checkpoint_dir=item_checkpoint_dir, resume=True
                ),
                parallel_strategy=ParallelStrategy(
                    program_executor=InlineExecutor(), n_program_chunks=2
                ),
                **runner_kwargs,
            )
            final_result = runner2.run()

        assert [[item[0] for item in items] for items in dispatched_items] == [
            [2, 3]
        ]
        assert final_result.failure_rates == uninterrupted_result.failure_rates
        assert final_result.stderrs == uninterrupted_result.stderrs
        assert final_result.is_complete

    @pytest.mark.parametrize(
        "second_overrides",
        [
            # seed_stride=None resolves to num_shots=5, so seed_stride=5 is
            # the same effective seeding.
            {"seed_stride": 5},
            # Equivalent-but-differently-typed expected_outcomes.
            {"expected_outcomes": (False,)},
        ],
        ids=[
            "default_seed_stride_resolving_to_num_shots",
            "equivalent_expected_outcomes",
        ],
    )
    def test_resume_checks_effective_config(self, tmp_path, second_overrides):
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

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            NoiseSweepResult(
                strengths=[0.0, 0.1],
                failure_rates=[0.0, 0.1],
                stderrs=[0.0],
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

    @pytest.mark.parametrize(
        "b_strengths, b_num_shots, strict",
        [([0.0, 0.2], 10, False), ([0.0, 0.1], 20, True)],
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

    def test_incomplete_warns_then_raises_when_strict(self):
        results = {
            "a": self._make_result([0.0, 0.1], 2),
            "b": self._make_result([0.0, 0.1], 1),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            returned = compare_noise_sweeps(results)
        assert any(issubclass(w.category, UserWarning) for w in caught)
        assert returned is results
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
        results = {
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
        }
        ax = plot_noise_sweep(results)
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
        """Incomplete (None) points reach neither the main line nor the
        guide line's x-range, even when one is the series' last strength."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        result = NoiseSweepResult(
            strengths=[0.01, 0.05, 0.1, 0.2, 0.5],
            failure_rates=[0.0, 0.05, None, 0.2, None],
            stderrs=[0.0, 0.01, None, 0.05, None],
            num_shots=100,
        )
        fig, ax = plt.subplots()
        ax = plot_noise_sweep(result, ax=ax, reference_slope=2)

        # The main line holds the two valid nonzero points.
        main_x = ax.lines[0].get_xdata()
        assert not np.any(np.isnan(main_x))
        assert list(main_x) == pytest.approx([0.05, 0.2])

        # The guide line spans the valid strengths only.
        (guide_line,) = [
            line for line in ax.lines if line.get_linestyle() == "--"
        ]
        assert list(guide_line.get_xdata()) == pytest.approx([0.01, 0.2])

        plt.close(fig)


class TestNoiseSweepRunnerHooks:
    """Direct unit tests of NoiseSweepRunner's own derived reduce_program_outcomes/
    _build_output/CHKPT_SUBDIR_PREFIX/mismatch-field hook implementations, in
    isolation from MultiProgramRunner's generic dispatch/checkpoint/resume/parallel
    machinery -- that generic behavior is covered once, generically, in
    test_multiprogramrunner.py."""

    def test_reduce_program_outcomes_raises_on_missing_shot(self):
        """A results object with fewer shots than `num_shots` raises
        instead of reporting a failure rate biased low."""
        runner = make_runner([0.1], seed_stride=20, num_shots=2)
        program = runner.build_program(0)
        program_results = program.run(num_shots=1, verbose=False)
        with pytest.raises(RuntimeError):
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
