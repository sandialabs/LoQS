"""Tester for loqs.tools.paralleltools"""

import pytest

from loqs.core.executors import MapArrayExecutor, SubmitExecutor
from loqs.tools.paralleltools import (
    ParallelStrategy,
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
            executor.submit(_report_blas_thread_counts).result()
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

    def test_live_shot_executor_with_program_executor_raises(self):
        loky = pytest.importorskip("loky")
        with pytest.raises(ValueError, match="shot_executor"):
            ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=1),
                n_program_chunks=2,
                shot_executor=loky.get_reusable_executor(max_workers=1),
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
