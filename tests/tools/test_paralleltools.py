"""Tester for loqs.tools.paralleltools"""

import sys

import pytest

from loqs.core.executors import MapArrayExecutor, SubmitExecutor
from loqs.tools.paralleltools import (
    ExecutorSpec,
    ParallelStrategy,
    _assign_chunks_to_workers,
    _canonical_shapes,
    _worker_plan,
    chunk_round_robin,
    pin_worker_threads,
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
        assert strategy.shot_executor.backend == "loky"
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
