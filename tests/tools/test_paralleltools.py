"""Tester for loqs.tools.paralleltools"""

import pytest

from loqs.tools.paralleltools import (
    ChunkExecutor,
    chunk_round_robin,
    pin_worker_threads,
    run_chunks_with_executor,
    run_chunks_with_submitit,
)


def _double_chunk(chunk: list[int]) -> list[int]:
    """Module-level worker (not a closure) so plain pickle can resolve it
    by dotted import path, matching what a real MPIPoolExecutor worker
    would require."""
    return [x * 2 for x in chunk]


def _sleep_and_return_chunk(chunk: list[int]) -> list[int]:
    """Sleeps longer for a chunk holding a smaller value, then returns it
    unchanged, so completion order is the reverse of submission order --
    used to confirm run_chunks_with_executor reorders by submission
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


class TestRunChunksWithExecutor:

    def test_loky_satisfies_chunk_executor_protocol(self):
        loky = pytest.importorskip("loky")
        executor = loky.get_reusable_executor(max_workers=2)
        assert isinstance(executor, ChunkExecutor)

    def test_loky_computes_every_chunk_correctly(self):
        loky = pytest.importorskip("loky")
        executor = loky.get_reusable_executor(max_workers=2)

        chunks = [[3, 3, 3], [1], [2, 2]]
        results = run_chunks_with_executor(executor, _double_chunk, chunks)

        assert results == [[6, 6, 6], [2], [4, 4]]

    def test_loky_preserves_submission_order_despite_reversed_completion(
        self,
    ):
        loky = pytest.importorskip("loky")
        executor = loky.get_reusable_executor(max_workers=3)

        chunks = [[0], [1], [2]]
        results = run_chunks_with_executor(
            executor, _sleep_and_return_chunk, chunks
        )

        assert results == [[0], [1], [2]]


class TestRunChunksWithSubmitit:

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

        chunks = [[3, 3, 3], [1], [2, 2]]
        results = run_chunks_with_submitit(
            executor, _double_chunk, chunks, poll_interval=0.05
        )

        assert results == [[6, 6, 6], [2], [4, 4]]
