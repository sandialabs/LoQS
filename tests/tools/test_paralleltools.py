"""Tester for loqs.tools.paralleltools"""

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from loqs.core.executors import MapArrayExecutor
from loqs.internal import pin_worker_threads
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

    time.sleep((3 - chunk[0]) * 0.05)
    return chunk


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
        with pytest.raises(ValueError):
            chunk_round_robin([1, 2], 0)


class TestPinWorkerThreads:

    @pytest.fixture(autouse=True)
    def _unpinned_process(self, monkeypatch):
        monkeypatch.setattr("loqs.internal._threads_pinned_pid", None)

    def test_pins_thread_pools_to_one(self):
        """Runs in this process: `threadpool_limits(limits=None)` restores
        the original limits on exit, so the pin does not leak."""
        threadpoolctl = pytest.importorskip("threadpoolctl")
        import numpy as np

        with threadpoolctl.threadpool_limits(limits=None):
            # Force numpy's BLAS pool to register before pinning.
            x = np.random.rand(200, 200)
            _ = x @ x
            if not threadpoolctl.threadpool_info():
                pytest.skip("No threadpoolctl-visible BLAS backend")
            pin_worker_threads()
            after = [
                pool["num_threads"] for pool in threadpoolctl.threadpool_info()
            ]
        assert all(n == 1 for n in after)

    def test_pins_once_per_process(self, monkeypatch):
        calls = []
        monkeypatch.setattr("loqs.internal.threadpool_limits", calls.append)
        pin_worker_threads()
        pin_worker_threads()
        assert calls == [1]

    def test_warns_when_threadpoolctl_not_installed(self, monkeypatch):
        monkeypatch.setattr("loqs.internal.threadpool_limits", None)
        with pytest.warns(UserWarning):
            pin_worker_threads()


class TestResolveShotExecutor:

    def test_resolve_shot_executor(self):
        assert resolve_shot_executor(None) is None

        with ThreadPoolExecutor(max_workers=1) as executor:
            assert resolve_shot_executor(executor) is executor

        assert resolve_shot_executor(lambda: "an executor") == "an executor"


class TestRunChunksWithSubmitExecutor:

    def test_preserves_submission_order_despite_reversed_completion(self):
        chunks = [[0], [1], [2]]
        with ThreadPoolExecutor(max_workers=3) as executor:
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

    def test_map_array_executor_without_n_program_chunks_raises(self):
        with pytest.raises(ValueError):
            ParallelStrategy(program_executor=_FakeMapArrayExecutor())

    def test_shot_executor_with_program_executor(self):
        """A live loky shot executor beside a program executor becomes an
        ExecutorSpec; a factory, or a live executor with no program
        executor, stays as given."""
        loky = pytest.importorskip("loky")
        program_executor = loky.get_reusable_executor(max_workers=1)
        strategy = ParallelStrategy(
            program_executor=program_executor,
            n_program_chunks=2,
            shot_executor=loky.get_reusable_executor(max_workers=3),
        )
        assert isinstance(strategy.shot_executor, ExecutorSpec)
        assert strategy.shot_executor.exec_backend == "loky"
        assert strategy.shot_executor.kwargs == {"max_workers": 3}

        strategy = ParallelStrategy(
            program_executor=program_executor,
            n_program_chunks=2,
            shot_executor=_build_shot_executor,
        )
        assert strategy.shot_executor is _build_shot_executor

        executor = loky.get_reusable_executor(max_workers=1)
        strategy = ParallelStrategy(shot_executor=executor)
        assert strategy.shot_executor is executor

    def test_live_unrecognized_shot_executor_with_program_executor_raises(
        self,
    ):
        loky = pytest.importorskip("loky")
        with pytest.raises(ValueError):
            ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=1),
                n_program_chunks=2,
                shot_executor=_FakeMapArrayExecutor(),
            )

    @pytest.mark.parametrize(
        "n_program_chunks", [None, 2], ids=["default", "explicit"]
    )
    def test_make_chunks(self, n_program_chunks):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=n_program_chunks,
        )
        expected = (
            [[1], [2], [3]] if n_program_chunks is None else [[1, 3], [2]]
        )
        assert strategy.make_chunks([1, 2, 3]) == expected

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

    def test_executor_spec_program_executor_resolves_once_on_dispatch(
        self, monkeypatch
    ):
        """An ExecutorSpec program executor stays unresolved until the
        first dispatch, which builds the pool once and reuses it."""
        pytest.importorskip("loky")
        pool = ThreadPoolExecutor(max_workers=2)
        builds = []

        def fake_get_reusable_executor(**kwargs):
            builds.append(kwargs)
            return pool

        monkeypatch.setattr(
            "loky.get_reusable_executor", fake_get_reusable_executor
        )
        spec = ExecutorSpec("loky", {"max_workers": 2})
        strategy = ParallelStrategy(program_executor=spec, n_program_chunks=2)
        try:
            assert strategy.is_chunked is True
            assert strategy.program_executor is spec
            assert strategy._resolved_program_executor is None

            results = strategy.dispatch(_double_chunk, [[1], [2, 2]])
            assert results == [[2], [4, 4]]
            resolved = strategy._resolved_program_executor
            assert resolved is not None

            strategy.dispatch(_double_chunk, [[1], [2, 2]])
            assert strategy._resolved_program_executor is resolved
            assert len(builds) == 1
        finally:
            pool.shutdown()


class TestExecutorSpecAndParallelStrategySerialization:
    """`ExecutorSpec`/`ParallelStrategy` are `Serializable`, needed so a
    `MultiProgramRunner` holding either can survive a `.write()`/`.read()`
    round-trip (e.g. as part of its own crash-recovery snapshot)."""

    @pytest.mark.parametrize("case", ["serial", "executors"])
    def test_parallel_strategy_round_trip(self, tmp_path, case):
        if case == "serial":
            strategy = ParallelStrategy(n_program_chunks=2, n_shot_batches=7)
        else:
            loky = pytest.importorskip("loky")
            strategy = ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=2),
                n_program_chunks=2,
                shot_executor=ExecutorSpec("loky", {"max_workers": 1}),
            )
        path = tmp_path / "strategy.json"
        strategy.write(path)
        loaded = ParallelStrategy.read(path)
        if case == "serial":
            assert loaded.program_executor is None
            assert loaded.shot_executor is None
            assert loaded.n_program_chunks == 2
            assert loaded.n_shot_batches == 7
        else:
            assert isinstance(loaded.program_executor, ExecutorSpec)
            assert loaded.program_executor.exec_backend == "loky"
            assert loaded.program_executor.kwargs == {"max_workers": 2}
            assert isinstance(loaded.shot_executor, ExecutorSpec)
            assert loaded.shot_executor.kwargs == {"max_workers": 1}

    def test_encoding_live_unrecognized_program_executor_raises(
        self, tmp_path
    ):
        strategy = ParallelStrategy(
            program_executor=_FakeMapArrayExecutor(), n_program_chunks=1
        )
        with pytest.raises(ValueError):
            strategy.write(tmp_path / "strategy.json")


class TestParallelStrategyDescribe:

    def test_describe_reports_program_and_shot_breakdown(self):
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=4),
            n_program_chunks=2,
            shot_executor=ExecutorSpec("loky", {"max_workers": 3}),
        )
        description = strategy.describe([1, 2, 3, 4, 5, 6], num_shots=30)
        assert "program axis: loky(max_workers=4)" in description
        assert "# of program chunks: 2" in description
        assert "# of programs/chunk: 3" in description
        assert "# of chunks/worker:  1" in description
        assert "shot axis: loky(max_workers=3)" in description
        assert "# of shots:        30" in description
        assert "# of shots/worker: 10" in description


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
        texts = self._texts(ax)
        assert "Serial" not in texts
        assert "PW3 = PW2" in texts
        assert "SW1" in texts


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
    return [
        x
        for chunk in strategy.dispatch(_double_chunk_slow, chunks)
        for x in chunk
    ]


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


class TestParallelStrategyResourceStats:
    """dispatch()'s transparent self-reporting wrapping -- see
    TestProfileStrategies for the full profile_strategies-level
    behavior this exists to support."""

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
        pytest.importorskip("psutil")
        with ThreadPoolExecutor(max_workers=2) as executor:
            strategy = ParallelStrategy(
                program_executor=executor, n_program_chunks=2
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

    def test_small_chunk_regime_warns(self):
        """A large sample_interval relative to how long each chunk
        actually takes means too few samples land inside it -- should
        warn rather than silently reporting unreliable numbers."""
        pytest.importorskip("psutil")
        with ThreadPoolExecutor(max_workers=2) as executor:
            strategy = ParallelStrategy(
                program_executor=executor, n_program_chunks=2
            )
            with pytest.warns(UserWarning):
                profile_strategies(
                    _profiling_work_fn,
                    {"fast": strategy},
                    sample_interval=5.0,
                )

    def test_reuse_slurm_allocation_requires_a_real_allocation(
        self, monkeypatch
    ):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        with pytest.raises(RuntimeError):
            profile_strategies(
                _profiling_work_fn,
                {"serial": ParallelStrategy()},
                reuse_slurm_allocation=True,
            )

    @pytest.mark.parametrize(
        "warmup", [False, True], ids=["no_warmup", "warmup"]
    )
    def test_warmup_adds_one_excluded_call(self, warmup):
        pytest.importorskip("psutil")
        calls = []
        results = profile_strategies(
            lambda strategy: calls.append(1),
            {"serial": ParallelStrategy()},
            repeats=3,
            warmup=warmup,
        )
        assert len(calls) == (4 if warmup else 3)
        assert len(results["serial"].chunk_stats) == 3

    def test_warmup_true_for_chunked_strategy_excludes_warmup_chunk_stats(
        self,
    ):
        with ThreadPoolExecutor(max_workers=1) as executor:
            strategy = ParallelStrategy(
                program_executor=executor, n_program_chunks=2
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

    @pytest.mark.parametrize("case", ["returns", "raises"])
    def test_executor_spec_program_executor_is_shut_down(
        self, monkeypatch, case
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
            if case == "raises":
                raise RuntimeError("work_fn failed")

        if case == "raises":
            with pytest.raises(RuntimeError):
                profile_strategies(work_fn, {"spec": strategy})
        else:
            profile_strategies(work_fn, {"spec": strategy})

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
        with pytest.warns(UserWarning):
            profile_strategies(
                lambda s: None,
                {"loky": strategy},
                repeats=2,
                warmup=True,
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


class TestFormatProfileTable:

    def test_formats_resource_stats_when_present(self):
        results = {
            "loky": ProfileResult(
                wall_time_mean=2.0,
                wall_time_std=0.0,
                chunk_stats=[
                    ChunkResourceStats(
                        peak_memory_mb=100.0,
                        mean_cpu_percent=50.0,
                        num_samples=5,
                    ),
                    ChunkResourceStats(
                        peak_memory_mb=200.0,
                        mean_cpu_percent=150.0,
                        num_samples=5,
                    ),
                ],
            ),
        }
        table = format_profile_table(results)
        assert "200.0" in table  # max peak memory
        assert "100.0" in table  # mean CPU% of 50 and 150


class TestPlotProfileResults:

    def test_three_panels_when_resource_stats_present(self):
        pytest.importorskip("matplotlib")
        results = {
            "loky": ProfileResult(
                wall_time_mean=1.0,
                wall_time_std=0.1,
                chunk_stats=[
                    ChunkResourceStats(
                        peak_memory_mb=100.0,
                        mean_cpu_percent=50.0,
                        num_samples=5,
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
