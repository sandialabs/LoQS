"""Tester for loqs.tools.paralleltools"""

import os
import sys
import warnings
from unittest.mock import MagicMock

import pytest

from loqs.core.executors import MapArrayExecutor, SubmitExecutor
from loqs.tools.paralleltools import (
    ChunkResourceStats,
    ExecutorSpec,
    ParallelStrategy,
    ProfileResult,
    _assign_chunks_to_workers,
    _canonical_shapes,
    _ResourceSampler,
    _worker_plan,
    chunk_round_robin,
    format_profile_table,
    pin_worker_threads,
    plot_profile_results,
    profile_strategies,
    reused_slurm_allocation,
    resolve_shot_executor,
    run_chunks_with_map_array_executor,
    run_chunks_with_submit_executor,
)


def _double_chunk(chunk: list[int]) -> list[int]:
    """Module-level worker (not a closure) so plain pickle can resolve it
    by dotted import path, matching what a real MPIPoolExecutor worker
    would require."""
    return [x * 2 for x in chunk]


def _sleep_and_return_chunk(chunk: list[int]) -> list[int]:
    """Sleeps longer for a chunk holding a smaller value, then returns it
    unchanged, so completion order is the reverse of submission order --
    used to confirm run_chunks_with_submit_executor reorders by submission
    index rather than trusting as_completed's own yield order."""
    import time

    time.sleep((3 - chunk[0]) * 0.2)
    return chunk


def _report_blas_thread_counts():
    """Force a BLAS op (registering numpy's thread pool if not already
    registered) and report every detected pool's thread count. Defined at
    module level so `loky` can pickle a plain reference to it."""
    import numpy as np
    import threadpoolctl

    x = np.random.rand(200, 200)
    _ = x @ x
    return [pool["num_threads"] for pool in threadpoolctl.threadpool_info()]


def _pin_worker_threads_task() -> None:
    """Module-level wrapper around pin_worker_threads, submitted to a real
    executor rather than calling it directly in the test process -- it's a
    sticky, non-context-manager call, so running it in-process would leak
    the one-thread limit into every later test sharing that process."""
    pin_worker_threads()


def _build_shot_executor():
    """Module-level factory (not a closure) building a fresh loky executor
    -- used as a picklable `shot_executor` factory in ParallelStrategy
    tests."""
    import loky

    return loky.get_reusable_executor(max_workers=1)


class _FakeMapArrayExecutor:
    """A minimal stand-in satisfying both SubmitExecutor and
    MapArrayExecutor structurally (like a real submitit.Executor), used to
    confirm ParallelStrategy.dispatch prefers map_array without needing a
    real submitit subprocess."""

    def submit(self, fn, /, *args, **kwargs):
        raise AssertionError(
            "dispatch() should have used map_array, not submit, for an "
            "object satisfying MapArrayExecutor"
        )

    def map_array(self, fn, *iterables):
        return [_FakeJob(fn(*args)) for args in zip(*iterables)]


class _FakeJob:
    def __init__(self, result):
        self._result = result

    def done(self, force_check: bool = False) -> bool:
        return True

    def result(self):
        return self._result


class TestChunkRoundRobin:

    def test_round_robin_assignment(self):
        assert chunk_round_robin(list(range(7)), 3) == [
            [0, 3, 6],
            [1, 4],
            [2, 5],
        ]

    def test_n_chunks_less_than_one_raises(self):
        with pytest.raises(ValueError, match="n_chunks"):
            chunk_round_robin([1, 2], 0)

    def test_more_chunks_than_items_leaves_some_empty(self):
        assert chunk_round_robin([1, 2], 5) == [[1], [2], [], [], []]

    def test_empty_items_returns_empty_chunks(self):
        assert chunk_round_robin([], 3) == [[], [], []]


class TestPinWorkerThreads:

    def test_pins_thread_pools_to_one_inside_a_worker(self):
        """Run inside a real, single-use loky worker (not the test process
        itself), mirroring test_quantumprogram.py's identical pattern for
        QuantumProgram._run_shot_worker."""
        loky = pytest.importorskip("loky")
        pytest.importorskip("threadpoolctl")

        executor = loky.get_reusable_executor(max_workers=1, reuse=False)
        try:
            # Force this worker's BLAS thread pool to register before
            # pinning, matching a real worker that already imported numpy.
            before = executor.submit(_report_blas_thread_counts).result()
            if not before:
                pytest.skip(
                    "No threadpoolctl-visible BLAS backend in this worker "
                    "(e.g. numpy built against Apple's Accelerate on "
                    "macOS, which threadpoolctl cannot introspect or "
                    "control at all) -- nothing to verify pinning against."
                )
            executor.submit(_pin_worker_threads_task).result()
            after = executor.submit(_report_blas_thread_counts).result()
        finally:
            executor.shutdown(wait=True)

        assert after, "expected at least one detected thread pool"
        assert all(n == 1 for n in after)

    def test_warns_when_threadpoolctl_not_installed(self, monkeypatch):
        monkeypatch.setattr(
            "loqs.tools.paralleltools.threadpool_limits", None
        )
        with pytest.warns(UserWarning, match="threadpoolctl"):
            pin_worker_threads()


class TestResolveShotExecutor:

    def test_none_returns_none(self):
        assert resolve_shot_executor(None) is None

    def test_live_executor_is_returned_as_is(self):
        loky = pytest.importorskip("loky")
        executor = loky.get_reusable_executor(max_workers=1)
        assert resolve_shot_executor(executor) is executor

    def test_factory_callable_is_called_once(self):
        calls = []

        def factory():
            calls.append(1)
            return "an executor"

        result = resolve_shot_executor(factory)

        assert result == "an executor"
        assert len(calls) == 1


class TestRunChunksWithSubmitExecutor:

    def test_loky_satisfies_submit_executor_protocol(self):
        loky = pytest.importorskip("loky")
        executor = loky.get_reusable_executor(max_workers=2)
        assert isinstance(executor, SubmitExecutor)

    def test_loky_computes_every_chunk_correctly(self):
        loky = pytest.importorskip("loky")
        executor = loky.get_reusable_executor(max_workers=2)

        chunks = [[3, 3, 3], [1], [2, 2]]
        results = run_chunks_with_submit_executor(
            executor, _double_chunk, chunks
        )

        assert results == [[6, 6, 6], [2], [4, 4]]

    def test_loky_preserves_submission_order_despite_reversed_completion(
        self,
    ):
        loky = pytest.importorskip("loky")
        executor = loky.get_reusable_executor(max_workers=3)

        chunks = [[0], [1], [2]]
        results = run_chunks_with_submit_executor(
            executor, _sleep_and_return_chunk, chunks
        )

        assert results == [[0], [1], [2]]


class TestRunChunksWithMapArrayExecutor:

    def test_debug_executor_preserves_chunk_order(self, tmp_path):
        """Uses submitit's in-process DebugExecutor rather than a real
        `cluster="local"` subprocess: a real subprocess can only unpickle
        a worker function it can import by dotted path, which a pytest
        test-module-local function like `_double_chunk` isn't -- unlike
        the real, installed package-level worker functions this dispatch
        machinery actually ships with (see test_pygstitools.py/
        test_fttools.py/test_noisesweeptools.py for coverage against a
        real subprocess-based `cluster="local"` executor)."""
        submitit = pytest.importorskip("submitit")
        executor = submitit.DebugExecutor(folder=tmp_path)
        assert isinstance(executor, MapArrayExecutor)

        chunks = [[3, 3, 3], [1], [2, 2]]
        results = run_chunks_with_map_array_executor(
            executor, _double_chunk, chunks, poll_interval=0.05
        )

        assert results == [[6, 6, 6], [2], [4, 4]]


class TestParallelStrategy:

    def test_defaults_to_not_chunked(self):
        strategy = ParallelStrategy()
        assert strategy.is_chunked is False

    def test_program_executor_makes_it_chunked(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1)
        )
        assert strategy.is_chunked is True

    def test_map_array_executor_without_n_program_chunks_raises(self):
        with pytest.raises(ValueError, match="n_program_chunks"):
            ParallelStrategy(program_executor=_FakeMapArrayExecutor())

    def test_submit_executor_without_n_program_chunks_is_fine(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1)
        )
        assert strategy.n_program_chunks is None

    def test_live_loky_shot_executor_with_program_executor_is_auto_converted(
        self,
    ):
        """A recognized backend (currently just loky) is transparently
        replaced with a picklable ExecutorSpec built from its own
        construction parameters, instead of requiring a caller to write
        a factory function by hand."""
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=2,
            shot_executor=loky.get_reusable_executor(max_workers=3),
        )
        assert isinstance(strategy.shot_executor, ExecutorSpec)
        assert strategy.shot_executor.exec_backend == "loky"
        assert strategy.shot_executor.kwargs == {"max_workers": 3}

    def test_live_unrecognized_shot_executor_with_program_executor_raises(
        self,
    ):
        loky = pytest.importorskip("loky")
        with pytest.raises(ValueError, match="shot_executor"):
            ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=1),
                n_program_chunks=2,
                shot_executor=_FakeMapArrayExecutor(),
            )

    def test_factory_shot_executor_with_program_executor_is_fine(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=2,
            shot_executor=_build_shot_executor,
        )
        assert strategy.shot_executor is _build_shot_executor

    def test_live_shot_executor_without_program_executor_is_fine(self):
        loky = pytest.importorskip("loky")
        executor = loky.get_reusable_executor(max_workers=1)
        strategy = ParallelStrategy(shot_executor=executor)
        assert strategy.shot_executor is executor

    def test_make_chunks_defaults_to_one_chunk_per_item(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1)
        )
        assert strategy.make_chunks([1, 2, 3]) == [[1], [2], [3]]

    def test_make_chunks_honors_explicit_n_program_chunks(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=2,
        )
        assert strategy.make_chunks([1, 2, 3]) == [[1, 3], [2]]

    def test_dispatch_via_submit_executor(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2)
        )
        chunks = [[1], [2, 2]]
        results = strategy.dispatch(_double_chunk, chunks)
        assert results == [[2], [4, 4]]

    def test_dispatch_prefers_map_array_over_submit(self):
        """A single object satisfying both SubmitExecutor and
        MapArrayExecutor (like a real submitit.Executor) must be
        dispatched via map_array, not submit -- _FakeMapArrayExecutor
        raises if .submit() is ever called, confirming the priority
        ordering ParallelStrategy.dispatch documents."""
        strategy = ParallelStrategy(
            program_executor=_FakeMapArrayExecutor(), n_program_chunks=2
        )
        chunks = [[1], [2, 2]]
        results = strategy.dispatch(_double_chunk, chunks)
        assert results == [[2], [4, 4]]

    def test_program_executor_accepts_executor_spec_unresolved(self):
        """An ExecutorSpec is accepted directly as program_executor and
        stays unresolved (no live pool built) until dispatch() actually
        needs one."""
        spec = ExecutorSpec("loky", {"max_workers": 2})
        strategy = ParallelStrategy(program_executor=spec, n_program_chunks=2)
        assert strategy.is_chunked is True
        assert strategy.program_executor is spec
        assert strategy._resolved_program_executor is None

    def test_dispatch_resolves_and_caches_executor_spec_program_executor(self):
        pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=ExecutorSpec("loky", {"max_workers": 1}),
            n_program_chunks=1,
        )
        strategy.dispatch(_double_chunk, [[1, 2]])
        resolved = strategy._resolved_program_executor
        assert resolved is not None
        strategy.dispatch(_double_chunk, [[3, 4]])
        assert strategy._resolved_program_executor is resolved

    def test_multiple_differently_sized_executor_specs_coexist_safely(self):
        """loky.get_reusable_executor() is a process-wide singleton --
        building several live, differently-sized instances up front (one
        per strategy) would silently invalidate the earlier ones.
        ExecutorSpec sidesteps this by staying unresolved until each
        strategy's own dispatch() call actually needs it, one strategy
        at a time."""
        pytest.importorskip("loky")
        strategies = {
            label: ParallelStrategy(
                program_executor=ExecutorSpec("loky", {"max_workers": n}),
                n_program_chunks=n,
            )
            for label, n in [("4x", 4), ("2x", 2), ("1x", 1)]
        }
        for strategy in strategies.values():
            chunks = strategy.make_chunks([1, 2, 3, 4])
            results = strategy.dispatch(_double_chunk, chunks)
            assert sorted(x for chunk in results for x in chunk) == [
                2,
                4,
                6,
                8,
            ]


class TestExecutorSpecAndParallelStrategySerialization:
    """`ExecutorSpec`/`ParallelStrategy` are `Serializable`, needed so a
    `MultiProgramRunner` holding either can survive a `.write()`/`.read()`
    round-trip (e.g. as part of its own crash-recovery snapshot)."""

    def test_executor_spec_round_trips(self, tmp_path):
        spec = ExecutorSpec(exec_backend="loky", kwargs={"max_workers": 3})
        path = tmp_path / "spec.json"
        spec.write(path)
        loaded = ExecutorSpec.read(path)
        assert loaded.exec_backend == "loky"
        assert loaded.kwargs == {"max_workers": 3}

    def test_parallel_strategy_round_trips_with_none_executors(
        self, tmp_path
    ):
        strategy = ParallelStrategy(n_program_chunks=2)
        path = tmp_path / "strategy.json"
        strategy.write(path)
        loaded = ParallelStrategy.read(path)
        assert loaded.program_executor is None
        assert loaded.shot_executor is None
        assert loaded.n_program_chunks == 2

    def test_parallel_strategy_round_trips_with_executor_spec(
        self, tmp_path
    ):
        strategy = ParallelStrategy(
            program_executor=ExecutorSpec(
                exec_backend="loky", kwargs={"max_workers": 2}
            ),
            n_program_chunks=2,
            shot_executor=ExecutorSpec(
                exec_backend="loky", kwargs={"max_workers": 1}
            ),
        )
        path = tmp_path / "strategy.json"
        strategy.write(path)
        loaded = ParallelStrategy.read(path)
        assert isinstance(loaded.program_executor, ExecutorSpec)
        assert loaded.program_executor.exec_backend == "loky"
        assert loaded.program_executor.kwargs == {"max_workers": 2}
        assert isinstance(loaded.shot_executor, ExecutorSpec)
        assert loaded.shot_executor.kwargs == {"max_workers": 1}

    def test_parallel_strategy_round_trips_live_recognized_executor(
        self, tmp_path
    ):
        """A live, loky-backed program_executor is transparently converted
        to an ExecutorSpec on encode, the same conversion already applied
        to a live shot_executor at construction time."""
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        path = tmp_path / "strategy.json"
        strategy.write(path)
        loaded = ParallelStrategy.read(path)
        assert isinstance(loaded.program_executor, ExecutorSpec)
        assert loaded.program_executor.exec_backend == "loky"

    def test_encoding_live_unrecognized_program_executor_raises(
        self, tmp_path
    ):
        strategy = ParallelStrategy(
            program_executor=_FakeMapArrayExecutor(), n_program_chunks=1
        )
        with pytest.raises(ValueError, match="program_executor"):
            strategy.write(tmp_path / "strategy.json")


class TestParallelStrategyDescribe:

    def test_all_serial_reports_both_axes_serial(self):
        strategy = ParallelStrategy()
        description = strategy.describe([1, 2, 3])
        assert description == "program axis: serial\nshot axis: serial"

    def test_program_executor_reports_full_breakdown(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=4),
            n_program_chunks=2,
        )
        description = strategy.describe([1, 2, 3, 4, 5, 6])
        assert "program axis: loky(max_workers=4)" in description
        assert "# of program chunks: 2" in description
        assert "# of programs/chunk: 3" in description
        assert "# of chunks/worker:  1" in description

    def test_map_array_executor_reports_plain_type_name(self):
        """`_FakeMapArrayExecutor` isn't a recognized backend (unlike a
        real `submitit.Executor`, which also isn't recognized for
        automatic factory construction -- see _introspect_executor_spec),
        so its tag falls back to its plain type name. Its worker count
        also isn't exposed, so "# of chunks/worker" is simply omitted
        rather than shown as "unknown"."""
        strategy = ParallelStrategy(
            program_executor=_FakeMapArrayExecutor(), n_program_chunks=2
        )
        description = strategy.describe()
        assert "program axis: _FakeMapArrayExecutor" in description
        assert "# of program chunks: 2" in description
        assert "# of chunks/worker" not in description

    def test_factory_shot_executor_reports_factory_name(self):
        strategy = ParallelStrategy(shot_executor=_build_shot_executor)
        description = strategy.describe()
        assert description.endswith(
            "shot axis: factory `_build_shot_executor`"
        )

    def test_auto_converted_shot_executor_reports_backend_and_kwargs(self):
        """A live loky executor given alongside program_executor is
        auto-converted to an ExecutorSpec (see TestParallelStrategy) --
        describe() should report its actual backend/kwargs, not just a
        generic "factory"."""
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=2,
            shot_executor=loky.get_reusable_executor(max_workers=3),
        )
        description = strategy.describe([1, 2, 3, 4], num_shots=30)
        assert "shot axis: loky(max_workers=3)" in description
        assert "# of shots:        30" in description
        assert "# of shots/worker: 10" in description

    def test_live_shot_executor_without_program_executor_reports_tag(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            shot_executor=loky.get_reusable_executor(max_workers=3)
        )
        description = strategy.describe(num_shots=10)
        assert "shot axis: loky(max_workers=3)" in description
        assert "# of shots:        10" in description
        assert "# of shots/worker: 4" in description

    def test_no_num_shots_omits_shot_rows(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            shot_executor=loky.get_reusable_executor(max_workers=3)
        )
        description = strategy.describe()
        assert description == "program axis: serial\nshot axis: loky(max_workers=3)"

    def test_no_items_omits_program_chunk_rows_needing_items(self):
        """"# of program chunks" only needs n_program_chunks (or
        program_executor being chunked at all), not items -- but
        "# of programs/chunk" genuinely needs a real item count, so only
        that row is omitted."""
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        description = strategy.describe()
        assert "# of program chunks: 2" in description
        assert "# of programs/chunk" not in description
        assert "# of chunks/worker:  1" in description

    def test_executor_spec_program_executor_describes_without_resolving(self):
        """describe() reads an unresolved ExecutorSpec's own kwargs
        directly (like it already does for shot_executor) rather than
        building a live pool just to introspect it."""
        strategy = ParallelStrategy(
            program_executor=ExecutorSpec("loky", {"max_workers": 4}),
            n_program_chunks=2,
        )
        description = strategy.describe([1, 2, 3, 4])
        assert "program axis: loky(max_workers=4)" in description
        assert "# of chunks/worker:  1" in description
        assert strategy._resolved_program_executor is None


class TestWorkerPlan:
    """Direct unit coverage of the pure grouping/padding logic behind
    ParallelStrategy.plot's collapsing -- more precise than parsing
    rendered text, for an algorithm with real edge cases worth pinning
    down exactly."""

    def test_canonical_shapes_takes_elementwise_max_per_chunk_count(self):
        # Two chunk-count-1 workers ([2], [1]) and one chunk-count-2
        # worker ([3, 1]) -- each chunk-count group gets its own
        # independent canonical shape.
        assigned = [[2], [1], [3, 1]]
        assert _canonical_shapes(assigned) == {1: (2,), 2: (3, 1)}

    def test_worker_plan_matches_the_4_1_worked_example(self):
        """6 items into 4 chunks split 2/2/1/1 -- W0/W1 are exact
        duplicates (2 real items each), W2/W3 are exact duplicates of
        each other (1 real item each), padded up to W0's shape."""
        assigned, idle = _assign_chunks_to_workers([2, 2, 1, 1], 4)
        assert idle == []
        plan = _worker_plan(assigned, idle, 4)

        assert plan[0].kind == "exemplar"
        assert plan[0].real_sizes == (2,)
        assert plan[0].padded_sizes == (2,)

        assert plan[1].kind == "pointer"
        assert plan[1].points_to == 0

        assert plan[2].kind == "exemplar"
        assert plan[2].real_sizes == (1,)
        assert plan[2].padded_sizes == (2,)

        assert plan[3].kind == "pointer"
        assert plan[3].points_to == 2

    def test_worker_plan_collapses_idle_workers_too(self):
        assigned, idle = _assign_chunks_to_workers([3, 3], 4)
        assert idle == [2, 3]
        plan = _worker_plan(assigned, idle, 4)

        assert plan[0].kind == "exemplar"
        assert plan[1].kind == "pointer"
        assert plan[1].points_to == 0
        assert plan[2].kind == "idle_exemplar"
        assert plan[3].kind == "idle_pointer"
        assert plan[3].points_to == 2

    def test_worker_plan_does_not_collapse_genuinely_different_shapes(self):
        """Only *exact* duplicates collapse. Two workers with chunk-count
        2 but different real per-chunk sizes ([3, 1] vs. [2, 2]) are each
        their own exemplar, even though they share a canonical shape
        ([3, 2])."""
        assigned = [[3, 1], [2, 2]]
        plan = _worker_plan(assigned, [], 2)
        assert plan[0].kind == "exemplar"
        assert plan[0].real_sizes == (3, 1)
        assert plan[0].padded_sizes == (3, 2)
        assert plan[1].kind == "exemplar"
        assert plan[1].real_sizes == (2, 2)
        assert plan[1].padded_sizes == (3, 2)


class TestParallelStrategyPlot:

    @staticmethod
    def _texts(ax) -> list[str]:
        return [t.get_text() for t in ax.texts]

    @staticmethod
    def _hatched_patch_count(ax) -> int:
        """Number of hatched (idle-marked) Rectangle patches currently
        drawn -- a more precise, rename-proof stand-in for counting
        program labels now that individual programs aren't labeled at
        all (see ParallelStrategy.plot's own docstring for why)."""
        return sum(1 for p in ax.patches if p.get_hatch() == "///")

    def test_all_serial_smoke(self):
        pytest.importorskip("matplotlib")
        ax = ParallelStrategy().plot([1, 2, 3])
        assert ax is not None
        assert "shots: serial" in ax.get_title()

    def test_all_serial_labels_the_lone_box_and_shot_axis_serial(self):
        """No program_executor means everything runs sequentially in the
        driver process, not a real worker pool, and no shot_executor
        means the same for shots -- the diagram should say "Serial" for
        both axes rather than implying a "PW0" worker exists."""
        pytest.importorskip("matplotlib")
        ax = ParallelStrategy().plot([1, 2, 3])
        texts = self._texts(ax)
        assert texts.count("Serial") == 2
        assert "PW0" not in texts

    def test_program_only_reports_active_workers_and_chunks(self):
        pytest.importorskip("matplotlib")
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        ax = strategy.plot([1, 2, 3, 4, 5, 6])
        assert "2/2 program worker(s) active" in ax.get_title()
        assert "up to 1 chunk(s)/worker" in ax.get_title()
        assert "shots: serial" in ax.get_title()

    def test_more_workers_than_chunks_leaves_some_idle(self):
        """Requesting more workers than there are chunks to hand out
        leaves some workers with nothing to do -- the diagram must draw
        that shortfall as explicitly idle: one fully-hatched exemplar
        (the first idle worker), with any further idle workers collapsed
        to a pointer at it rather than redrawn."""
        pytest.importorskip("matplotlib")
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=4),
            n_program_chunks=2,
        )
        ax = strategy.plot([1, 2, 3, 4, 5, 6])
        assert "2/4 program worker(s) active" in ax.get_title()
        texts = self._texts(ax)
        assert "PW2 (idle)" in texts
        assert "PW3 (idle)" not in texts
        assert "PW3 = PW2" in texts

    def test_duplicate_active_workers_collapse_to_pointer_labels(self):
        """Under round-robin dispatch, workers with the exact same real
        chunk-size signature are exact duplicates of each other -- the
        first is drawn in full, later ones collapse to a small pointer
        label rather than a redundant redraw."""
        pytest.importorskip("matplotlib")
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=4),
            n_program_chunks=4,
        )
        ax = strategy.plot([1, 2, 3, 4, 5, 6])
        texts = self._texts(ax)
        assert "PW0" in texts
        assert "PW1" not in texts
        assert "PW1 = PW0" in texts

    def test_ragged_worker_padded_with_nested_idle_program(self):
        """A worker whose own chunk fell short of its chunk-count
        group's largest ("canonical") shape -- an uneven round-robin
        remainder, not a scheduling gap -- is drawn at that larger
        shape, with the shortfall as a hatched idle *program* nested
        inside an otherwise fully-drawn worker, not the whole worker
        hatched: it did get real work, just less of it. 6 items into 4
        chunks split 2/2/1/1: PW0/PW1 are exact duplicates (2 real items
        each, 0 idle padding), and PW2/PW3 are exact duplicates of each
        other (1 real item each) -- PW3 collapses to a pointer, so only
        PW2's own chunk is actually drawn, padded to PW0's 2-slot shape
        with exactly one hatched idle placeholder."""
        pytest.importorskip("matplotlib")
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=4),
            n_program_chunks=4,
        )
        ax = strategy.plot([1, 2, 3, 4, 5, 6])
        texts = self._texts(ax)
        assert "PW2" in texts
        assert self._hatched_patch_count(ax) == 1

    def test_more_chunks_than_workers_repeats_the_chunk_unit(self):
        """A worker assigned more than one chunk processes them
        sequentially, one at a time -- the title's own chunks-per-worker
        count reports every assigned chunk (not just the first), since
        individual chunks aren't labeled."""
        pytest.importorskip("matplotlib")
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=4,
        )
        ax = strategy.plot([1, 2, 3, 4, 5, 6])
        assert "2/2 program worker(s) active" in ax.get_title()
        assert "up to 2 chunk(s)/worker" in ax.get_title()

    def test_real_uneven_chunk_sizes_from_make_chunks(self):
        """7 items in 2 round-robin chunks split 4/3, not evenly -- the
        diagram should draw that real split (via strategy.make_chunks),
        not a rounded/averaged approximation."""
        pytest.importorskip("matplotlib")
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        assert [len(c) for c in strategy.make_chunks(list(range(7)))] == [
            4,
            3,
        ]
        ax = strategy.plot(list(range(7)))
        assert ax is not None

    def test_shot_executor_set_shows_lanes_not_serial(self):
        """No program_executor here means the *program* axis still shows
        "Serial" (correctly), but the *shot* axis has a real executor and
        should show real SW{i} lanes instead of a "Serial" shot lane."""
        pytest.importorskip("matplotlib")
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            shot_executor=loky.get_reusable_executor(max_workers=3)
        )
        ax = strategy.plot([1, 2, 3])
        texts = self._texts(ax)
        assert texts.count("Serial") == 1
        assert {"SW0", "SW1", "SW2"} <= set(texts)
        assert "shots: 3 worker(s)" in ax.get_title()

    def test_explicit_overrides_take_precedence_over_introspection(self):
        """Mirrors the real submitit/MPI tutorial usage: program_executor
        is real (just doesn't expose a worker count), and program_workers
        is given explicitly to illustrate one anyway; an explicit
        shot_workers also forces a drawn pool instead of "Serial", even
        though this strategy has no real shot_executor."""
        pytest.importorskip("matplotlib")
        strategy = ParallelStrategy(
            program_executor=_FakeMapArrayExecutor(), n_program_chunks=4
        )
        ax = strategy.plot(
            [1, 2, 3, 4, 5, 6],
            program_workers=4,
            shot_workers=2,
        )
        assert "4/4 program worker(s) active" in ax.get_title()
        assert "shots: 2 worker(s)" in ax.get_title()
        assert "Serial" not in self._texts(ax)

    def test_unresolvable_worker_counts_default_to_one(self):
        pytest.importorskip("matplotlib")
        strategy = ParallelStrategy(
            program_executor=_FakeMapArrayExecutor(), n_program_chunks=2
        )
        ax = strategy.plot([1, 2, 3])
        assert "1/1 program worker(s) active" in ax.get_title()

    def test_legend_false_omits_legend(self):
        pytest.importorskip("matplotlib")
        ax = ParallelStrategy().plot([1, 2, 3], legend=False)
        assert ax.get_legend() is None

    def test_legend_true_by_default(self):
        pytest.importorskip("matplotlib")
        ax = ParallelStrategy().plot([1, 2, 3])
        assert ax.get_legend() is not None

    def test_missing_matplotlib_raises_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        with pytest.raises(ImportError):
            ParallelStrategy().plot([1, 2, 3])


def _double_chunk_slow(chunk: list[int]) -> list[int]:
    """Module-level worker (not a closure) that does a tiny bit of real
    work, for profile_strategies tests -- long enough for a handful of
    real resource samples at a short sample_interval, without making the
    suite slow."""
    import time

    time.sleep(0.15)
    return [x * 2 for x in chunk]


def _profiling_work_fn(strategy: ParallelStrategy) -> list[int]:
    """Module-level work_fn (not a closure) for profile_strategies
    tests -- real end-to-end dispatch through a caller-built
    ParallelStrategy, matching how a real call site would be used."""
    import time

    items = list(range(4))
    if not strategy.is_chunked:
        time.sleep(0.15)
        return [x * 2 for x in items]
    chunks = strategy.make_chunks(items)
    return [x for chunk in strategy.dispatch(_double_chunk_slow, chunks) for x in chunk]


class _FakeMemInfo:
    def __init__(self, rss: int) -> None:
        self.rss = rss


class _FakeProcess:
    """Minimal psutil.Process-like double for TestResourceSampler --
    exposes just what _ResourceSampler._sample() touches, with a
    call-counted cpu_percent() so priming behavior is directly
    observable without needing a real process tree."""

    def __init__(
        self, pid: int, rss: int = 100 * 1024 * 1024, cpu_value: float = 50.0
    ) -> None:
        self.pid = pid
        self.cpu_percent_calls = 0
        self._rss = rss
        self._cpu_value = cpu_value

    def memory_info(self) -> _FakeMemInfo:
        return _FakeMemInfo(self._rss)

    def cpu_percent(self, interval=None) -> float:
        self.cpu_percent_calls += 1
        return self._cpu_value

    def children(self, recursive: bool = True) -> list:
        return []


class TestResourceSampler:
    """_ResourceSampler._sample()'s per-PID caching/priming logic --
    psutil.Process.children() constructs a brand-new object for every
    child on every call, so cpu_percent() needs its own cached object
    per PID to correctly measure "since the last sample" rather than
    "since the process started" (see _sample()'s own docstring)."""

    @staticmethod
    def _bare_sampler() -> _ResourceSampler:
        """A _ResourceSampler with its real __init__ (which constructs a
        genuine psutil.Process(pid)) bypassed, so its internal state can
        be set directly against fake process doubles instead."""
        sampler = _ResourceSampler.__new__(_ResourceSampler)
        sampler._known_procs = {}
        sampler._cpu_samples = []
        sampler._peak_memory_mb = 0.0
        return sampler

    def test_newly_seen_process_is_primed_and_excluded_from_cpu(self):
        pytest.importorskip("psutil")
        sampler = self._bare_sampler()
        proc = _FakeProcess(pid=1, rss=100 * 1024 * 1024, cpu_value=50.0)
        sampler._process = proc

        sampler._sample()

        assert sampler._cpu_samples == []
        assert sampler._peak_memory_mb == pytest.approx(100.0)
        assert proc.cpu_percent_calls == 1
        assert 1 in sampler._known_procs

    def test_already_known_process_contributes_cpu_without_double_priming(
        self,
    ):
        pytest.importorskip("psutil")
        sampler = self._bare_sampler()
        proc = _FakeProcess(pid=1, cpu_value=75.0)
        sampler._process = proc
        sampler._known_procs = {1: proc}

        sampler._sample()

        assert sampler._cpu_samples == [75.0]
        assert proc.cpu_percent_calls == 1

    def test_child_seen_partway_through_is_primed_separately_from_self(self):
        """A child appearing after self is already known gets primed and
        excluded on its own first sample, while the already-known self
        process still contributes normally in that same round -- the
        child then contributes starting the next sample. The fake children()
        returns a different _FakeProcess object (same pid, same cpu_value)
        on each call, mimicking real psutil.Process.children() behavior,
        to verify the cached object is reused rather than the fresh one."""
        pytest.importorskip("psutil")
        sampler = self._bare_sampler()
        self_proc = _FakeProcess(pid=1, cpu_value=10.0)
        # Two distinct _FakeProcess objects, both pid=2, both cpu_value=99.0
        first_child = _FakeProcess(pid=2, cpu_value=99.0)
        second_child = _FakeProcess(pid=2, cpu_value=99.0)
        # Alternate between the two on successive calls to children()
        children_list = [first_child, second_child]
        children_iter = iter(children_list)
        self_proc.children = lambda recursive=True: [next(children_iter)]
        sampler._process = self_proc
        sampler._known_procs = {1: self_proc}

        sampler._sample()
        assert sampler._cpu_samples == [10.0]
        assert first_child.cpu_percent_calls == 1
        assert 2 in sampler._known_procs
        # Verify the cached object is the first child
        assert sampler._known_procs[2] is first_child

        sampler._sample()
        assert sampler._cpu_samples == [10.0, 10.0 + 99.0]
        # The first child object should have been reused and queried again
        assert first_child.cpu_percent_calls == 2
        # The second child object should never have been used
        assert second_child.cpu_percent_calls == 0

    def test_real_child_process_is_measured_across_multiple_samples(self):
        """End-to-end check against a real OS process tree (not a
        mock/double): a real, sustained-CPU-use child process should show
        up as multiple real, non-zero samples, confirming the priming
        mechanism doesn't silently exclude it forever. Deliberately
        doesn't assert a tight percentage range -- real system load (e.g.
        other tests running concurrently under pytest-xdist) can
        legitimately reduce how much of a core this child actually
        gets."""
        pytest.importorskip("psutil")
        import subprocess

        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time\nt=time.time()\n"
                "while time.time() - t < 1.0:\n    pass",
            ]
        )
        try:
            sampler = _ResourceSampler(os.getpid(), 0.1)
            sampler.start()
            child.wait()
        finally:
            if child.poll() is None:
                child.kill()
            stats = sampler.stop()

        assert stats.num_samples >= 3
        assert stats.mean_cpu_percent > 0.0
        assert stats.peak_memory_mb > 0.0


class TestParallelStrategyResourceStats:
    """dispatch()'s transparent self-reporting wrapping -- see
    TestProfileStrategies for the full profile_strategies-level
    behavior this exists to support."""

    def test_collect_resource_stats_defaults_false_and_is_a_noop(self):
        loky = pytest.importorskip("loky")
        pytest.importorskip("psutil")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1)
        )
        results = strategy.dispatch(_double_chunk, [[1], [2]])
        assert results == [[2], [4]]
        assert strategy.pop_resource_stats() == []

    def test_collect_resource_stats_true_reports_one_entry_per_chunk(self):
        loky = pytest.importorskip("loky")
        pytest.importorskip("psutil")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            collect_resource_stats=True,
            resource_sample_interval=0.05,
        )
        chunks = [[1], [2, 2]]
        results = strategy.dispatch(_double_chunk_slow, chunks)
        assert results == [[2], [4, 4]]
        stats = strategy.pop_resource_stats()
        assert len(stats) == len(chunks)
        assert all(isinstance(s, ChunkResourceStats) for s in stats)
        assert all(s.peak_memory_mb > 0 for s in stats)

    def test_pop_resource_stats_clears_between_calls(self):
        loky = pytest.importorskip("loky")
        pytest.importorskip("psutil")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            collect_resource_stats=True,
            resource_sample_interval=0.05,
        )
        strategy.dispatch(_double_chunk_slow, [[1]])
        first = strategy.pop_resource_stats()
        assert len(first) == 1
        assert strategy.pop_resource_stats() == []


class TestProfileStrategies:

    def test_serial_strategy_reports_wall_time_and_one_chunk_stat_per_repeat(
        self,
    ):
        pytest.importorskip("psutil")
        results = profile_strategies(
            _profiling_work_fn,
            {"serial": ParallelStrategy()},
            repeats=2,
            sample_interval=0.02,
        )
        result = results["serial"]
        assert isinstance(result, ProfileResult)
        assert result.wall_time_mean > 0
        assert len(result.chunk_stats) == 2
        assert result.peak_memory_mb is not None
        assert result.peak_memory_mb > 0

    def test_chunked_strategy_reports_one_chunk_stat_per_dispatched_chunk(
        self,
    ):
        loky = pytest.importorskip("loky")
        pytest.importorskip("psutil")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        results = profile_strategies(
            _profiling_work_fn,
            {"loky-2x": strategy},
            repeats=1,
            sample_interval=0.02,
        )
        result = results["loky-2x"]
        assert len(result.chunk_stats) == 2
        assert result.peak_memory_mb is not None
        assert result.mean_cpu_percent is not None
        # dispatch()'s own return value is unaffected by profiling --
        # confirms the strategy object isn't left mutated in a way that
        # would leak stats-collection into a later, real (non-profiling)
        # use of the same object.
        assert strategy.collect_resource_stats is False
        assert strategy.pop_resource_stats() == []

    def test_results_keyed_by_the_same_labels_given(self):
        pytest.importorskip("psutil")
        strategies = {"a": ParallelStrategy(), "b": ParallelStrategy()}
        results = profile_strategies(
            _profiling_work_fn, strategies, sample_interval=0.02
        )
        assert set(results.keys()) == {"a", "b"}

    def test_repeats_defaults_to_one(self):
        pytest.importorskip("psutil")
        results = profile_strategies(
            _profiling_work_fn, {"serial": ParallelStrategy()}, sample_interval=0.02
        )
        assert len(results["serial"].chunk_stats) == 1
        assert results["serial"].wall_time_std == 0.0

    def test_small_chunk_regime_warns(self):
        """A large sample_interval relative to how long each chunk
        actually takes means too few samples land inside it -- should
        warn rather than silently reporting unreliable numbers."""
        loky = pytest.importorskip("loky")
        pytest.importorskip("psutil")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        with pytest.warns(UserWarning, match="resource-sampling intervals"):
            profile_strategies(
                _profiling_work_fn,
                {"fast": strategy},
                sample_interval=5.0,
            )

    def test_reuse_slurm_allocation_requires_a_real_allocation(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        with pytest.raises(RuntimeError, match="SLURM_JOB_ID"):
            profile_strategies(
                _profiling_work_fn,
                {"serial": ParallelStrategy()},
                reuse_slurm_allocation=True,
            )

    def test_warmup_false_by_default_calls_work_fn_exactly_repeats_times(self):
        pytest.importorskip("psutil")
        calls = []
        results = profile_strategies(
            lambda strategy: calls.append(1),
            {"serial": ParallelStrategy()},
            repeats=3,
        )
        assert len(calls) == 3
        assert len(results["serial"].chunk_stats) == 3

    def test_warmup_true_adds_one_call_excluded_from_results(self):
        pytest.importorskip("psutil")
        calls = []
        results = profile_strategies(
            lambda strategy: calls.append(1),
            {"serial": ParallelStrategy()},
            repeats=3,
            warmup=True,
        )
        assert len(calls) == 4  # 1 warmup + 3 real, timed repeats
        assert len(results["serial"].chunk_stats) == 3

    def test_warmup_true_for_chunked_strategy_excludes_warmup_chunk_stats(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=2,
        )
        results = profile_strategies(
            _profiling_work_fn,
            {"loky": strategy},
            repeats=2,
            sample_interval=0.02,
            warmup=True,
        )
        # 2 chunks x 2 real repeats -- the warmup pass's own 2 chunks of
        # stats are discarded, not folded in as a third repeat's worth.
        assert len(results["loky"].chunk_stats) == 4

    def test_speedup_computed_against_fully_serial_baseline(self, monkeypatch):
        """Checks the speedup arithmetic itself, not real parallel
        speedup (that's covered empirically elsewhere) -- so
        time.perf_counter is mocked to a fixed sequence instead of
        relying on real time.sleep() calls, which flake on a slow/loaded
        CI runner where a real 10ms sleep can occasionally exceed a real
        50ms one. perf_counter is called exactly twice per strategy
        (start/end) in dict order: serial, then fast."""
        pytest.importorskip("psutil")
        times = iter([0.0, 0.5, 0.5, 0.6])
        monkeypatch.setattr(
            "loqs.tools.paralleltools.time.perf_counter",
            lambda: next(times),
        )
        results = profile_strategies(
            lambda strategy: None,
            {
                "serial": ParallelStrategy(),
                "fast": ParallelStrategy(program_executor=object()),
            },
            sample_interval=0.02,
        )
        assert results["serial"].speedup is None
        assert results["fast"].speedup is not None
        assert results["fast"].speedup == pytest.approx(5.0)

    def test_speedup_is_none_when_no_fully_serial_strategy_given(self):
        pytest.importorskip("psutil")
        results = profile_strategies(
            lambda strategy: None,
            {
                "a": ParallelStrategy(program_executor=object()),
                "b": ParallelStrategy(shot_executor=object()),
            },
            sample_interval=0.02,
        )
        assert results["a"].speedup is None
        assert results["b"].speedup is None

    def test_executor_spec_program_executor_is_shut_down_and_reset(
        self, monkeypatch
    ):
        pytest.importorskip("loky")
        fake_executor = MagicMock()
        monkeypatch.setattr(
            "loky.get_reusable_executor", lambda **kwargs: fake_executor
        )
        spec = ExecutorSpec("loky", {"max_workers": 2})
        strategy = ParallelStrategy(program_executor=spec, n_program_chunks=2)

        def work_fn(s):
            s._resolve_program_executor()
            return None

        profile_strategies(work_fn, {"spec": strategy})
        fake_executor.shutdown.assert_called_once_with(
            wait=True, kill_workers=True
        )
        assert strategy._resolved_program_executor is None

    def test_executor_spec_program_executor_shut_down_on_work_fn_exception(
        self, monkeypatch
    ):
        pytest.importorskip("loky")
        fake_executor = MagicMock()
        monkeypatch.setattr(
            "loky.get_reusable_executor", lambda **kwargs: fake_executor
        )
        spec = ExecutorSpec("loky", {"max_workers": 2})
        strategy = ParallelStrategy(program_executor=spec, n_program_chunks=2)

        def failing_work_fn(s):
            s._resolve_program_executor()
            raise RuntimeError("work_fn failed")

        with pytest.raises(RuntimeError, match="work_fn failed"):
            profile_strategies(failing_work_fn, {"spec": strategy})

        fake_executor.shutdown.assert_called_once_with(
            wait=True, kill_workers=True
        )
        assert strategy._resolved_program_executor is None

    def test_live_program_executor_is_not_shut_down(self):
        fake_executor = MagicMock()
        strategy = ParallelStrategy(
            program_executor=fake_executor, n_program_chunks=2
        )
        profile_strategies(lambda s: None, {"live": strategy})
        fake_executor.shutdown.assert_not_called()

    def test_recycled_worker_pool_warns_when_repeat_slower_than_warmup(
        self, monkeypatch
    ):
        times = iter([0.0, 0.1, 1.0, 1.1, 2.0, 2.5])
        monkeypatch.setattr(
            "loqs.tools.paralleltools.time.perf_counter",
            lambda: next(times),
        )
        fake_executor = MagicMock()
        strategy = ParallelStrategy(
            program_executor=fake_executor, n_program_chunks=2
        )
        with pytest.warns(UserWarning, match="timeout"):
            profile_strategies(
                lambda s: None,
                {"loky": strategy},
                repeats=2,
                warmup=True,
            )

    def test_recycled_worker_pool_warns_when_repeat_slower_without_warmup(
        self, monkeypatch
    ):
        times = iter([0.0, 0.5, 1.0, 1.1])
        monkeypatch.setattr(
            "loqs.tools.paralleltools.time.perf_counter",
            lambda: next(times),
        )
        fake_executor = MagicMock()
        strategy = ParallelStrategy(
            program_executor=fake_executor, n_program_chunks=2
        )
        with pytest.warns(UserWarning, match="timeout"):
            profile_strategies(
                lambda s: None,
                {"loky": strategy},
                repeats=2,
                warmup=False,
            )

    def test_recycled_worker_pool_no_warning_when_timings_consistent(
        self, monkeypatch
    ):
        times = iter([0.0, 0.1, 1.0, 1.1, 2.0, 2.1])
        monkeypatch.setattr(
            "loqs.tools.paralleltools.time.perf_counter",
            lambda: next(times),
        )
        fake_executor = MagicMock()
        strategy = ParallelStrategy(
            program_executor=fake_executor, n_program_chunks=2
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            profile_strategies(
                lambda s: None,
                {"loky": strategy},
                repeats=2,
                warmup=True,
            )

    def test_serial_strategy_does_not_warn_on_slowdown(self, monkeypatch):
        pytest.importorskip("psutil")
        times = iter([0.0, 0.1, 1.0, 1.1, 2.0, 2.5])
        monkeypatch.setattr(
            "loqs.tools.paralleltools.time.perf_counter",
            lambda: next(times),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings(
                "ignore", message=".*resource-sampling intervals.*"
            )
            profile_strategies(
                lambda s: None,
                {"serial": ParallelStrategy()},
                repeats=2,
                warmup=True,
            )


class TestFormatProfileTable:

    def test_formats_wall_time_and_dashes_for_missing_resource_stats(self):
        results = {
            "serial": ProfileResult(wall_time_mean=1.5, wall_time_std=0.1),
        }
        table = format_profile_table(results)
        assert "serial" in table
        assert "1.5 +/- 0.1" in table
        assert "--" in table

    def test_formats_resource_stats_when_present(self):
        results = {
            "loky": ProfileResult(
                wall_time_mean=2.0,
                wall_time_std=0.0,
                chunk_stats=[
                    ChunkResourceStats(
                        peak_memory_mb=100.0, mean_cpu_percent=50.0, num_samples=5
                    ),
                    ChunkResourceStats(
                        peak_memory_mb=200.0, mean_cpu_percent=150.0, num_samples=5
                    ),
                ],
            ),
        }
        table = format_profile_table(results)
        assert "200.0" in table  # max peak memory
        assert "100.0" in table  # mean CPU% of 50 and 150

    def test_speedup_column_dashes_when_none(self):
        results = {"serial": ProfileResult(wall_time_mean=1.0, wall_time_std=0.0)}
        table = format_profile_table(results)
        assert "speedup" in table
        assert "--" in table

    def test_speedup_column_formats_value(self):
        results = {
            "fast": ProfileResult(
                wall_time_mean=1.0, wall_time_std=0.0, speedup=2.5
            ),
        }
        table = format_profile_table(results)
        assert "2.50x" in table


class TestPlotProfileResults:

    def test_wall_time_only_panel_when_no_resource_stats(self):
        pytest.importorskip("matplotlib")
        results = {
            "serial": ProfileResult(wall_time_mean=1.0, wall_time_std=0.0),
        }
        axes = plot_profile_results(results)
        assert len(axes) == 1

    def test_speedup_annotated_on_wall_time_bars(self):
        pytest.importorskip("matplotlib")
        results = {
            "serial": ProfileResult(wall_time_mean=2.0, wall_time_std=0.0),
            "fast": ProfileResult(
                wall_time_mean=1.0, wall_time_std=0.0, speedup=2.0
            ),
        }
        axes = plot_profile_results(results)
        texts = [t.get_text() for t in axes[0].texts]
        assert "2.00x" in texts

    def test_three_panels_when_resource_stats_present(self):
        pytest.importorskip("matplotlib")
        results = {
            "loky": ProfileResult(
                wall_time_mean=1.0,
                wall_time_std=0.1,
                chunk_stats=[
                    ChunkResourceStats(
                        peak_memory_mb=100.0, mean_cpu_percent=50.0, num_samples=5
                    )
                ],
            ),
        }
        axes = plot_profile_results(results)
        assert len(axes) == 3


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "the fake sbatch mechanism this class exercises writes a POSIX "
        "(#!/bin/bash) shell script and invokes it directly via "
        "subprocess.run(['sbatch', ...]) -- Windows has no bash "
        "interpreter and won't resolve an extension-less file as "
        "executable via PATH, so this is a real, unconditional platform "
        "limitation of the mechanism itself (which only ever targets a "
        "real SLURM cluster, a Linux-only scheduler), not something "
        "fixable from LoQS's side."
    ),
)
class TestReusedSlurmAllocation:

    def test_raises_without_slurm_job_id(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        with pytest.raises(RuntimeError, match="SLURM_JOB_ID"):
            with reused_slurm_allocation():
                pass

    def test_installs_and_restores_path(self, monkeypatch):
        import os

        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        original_path = os.environ.get("PATH", "")
        with reused_slurm_allocation():
            new_path = os.environ["PATH"]
            assert new_path != original_path
            fake_dir = new_path.split(os.pathsep)[0]
            fake_sbatch = os.path.join(fake_dir, "sbatch")
            assert os.path.exists(fake_sbatch)
            assert os.access(fake_sbatch, os.X_OK)
        assert os.environ["PATH"] == original_path
        assert not os.path.exists(fake_dir)

    def test_fake_sbatch_runs_script_directly_and_prints_job_id(
        self, monkeypatch, tmp_path
    ):
        """Real end-to-end check of the trick itself, not just that the
        wrapper file exists: invoking the fake `sbatch` on a plain
        script (with `#SBATCH` header lines it should just ignore as
        bash comments) actually runs that script's real payload (here,
        writing a marker file) and prints output matching submitit's own
        job-ID-parsing regex (`"job (?P<id>[0-9]+)"`)."""
        import re
        import subprocess
        import time

        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        marker = tmp_path / "marker.txt"
        script = tmp_path / "job.sh"
        script.write_text(
            "#!/bin/bash\n#SBATCH --time=10\necho ran > " + str(marker) + "\n"
        )
        script.chmod(0o755)
        with reused_slurm_allocation():
            output = subprocess.run(
                ["sbatch", str(script)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        assert re.search(r"job \d+", output)
        for _ in range(50):
            if marker.exists():
                break
            time.sleep(0.1)
        assert marker.read_text() == "ran\n"


class TestParallelStrategyNShotBatches:
    """Test the new n_shot_batches field on ParallelStrategy."""

    def test_n_shot_batches_field_in_serialize_attrs(self):
        """Verify n_shot_batches is included in _SERIALIZE_ATTRS."""
        from loqs.tools.paralleltools import ParallelStrategy
        assert "n_shot_batches" in ParallelStrategy._SERIALIZE_ATTRS

    def test_n_shot_batches_field_default_none(self):
        """Verify n_shot_batches defaults to None."""
        from loqs.tools.paralleltools import ParallelStrategy
        strategy = ParallelStrategy()
        assert strategy.n_shot_batches is None

    def test_n_shot_batches_field_explicit_value(self):
        """Verify n_shot_batches can be set explicitly."""
        from loqs.tools.paralleltools import ParallelStrategy
        strategy = ParallelStrategy(n_shot_batches=5)
        assert strategy.n_shot_batches == 5

    def test_n_shot_batches_serialization(self):
        """Verify n_shot_batches round-trips through serialization."""
        import tempfile
        from pathlib import Path
        from loqs.tools.paralleltools import ParallelStrategy

        strategy_orig = ParallelStrategy(n_shot_batches=7)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "strategy.json"
            strategy_orig.write(path)
            strategy_decoded = ParallelStrategy.read(path)
            assert strategy_decoded.n_shot_batches == 7
