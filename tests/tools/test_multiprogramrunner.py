"""Tester for loqs.tools.multiprogramrunner"""

import contextlib
import h5py
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import pytest

from loqs.internal import worker_id
from loqs.internal.streamingmerge import iter_dict_attr_entries
from loqs.tools.paralleltools import ParallelStrategy
from loqs.tools.multiprogramrunner import MultiProgramRunner


# Module-level worker functions for parallel/multiprocessing tests


def _double_item(item, index, *, shot_executor, **kwargs):
    """Double an integer item."""
    return item * 2


def _count_and_double(item, index, *, shot_executor, **kwargs):
    """Count how many times this is called, and double the item.

    Uses a counter in static_kwargs.
    """
    call_count_list = kwargs.get("call_count")
    if call_count_list is None:
        call_count_list = [0]
    call_count_list[0] += 1
    return item * 2


def _sleep_and_double(item, index, *, shot_executor, **kwargs):
    """Sleep briefly then double the item (for parallel timing tests)."""
    sleep_time = kwargs.get("sleep_time", 0.01)
    time.sleep(sleep_time)
    return item * 2


def _raise_after_n(item, index, *, shot_executor, **kwargs):
    """Raise an exception after processing a certain number of items.

    Uses counter in static_kwargs.
    """
    max_count = kwargs.get("max_count", 999)
    call_count_list = kwargs.get("call_count")
    if call_count_list is None:
        call_count_list = [0]
    call_count_list[0] += 1
    if call_count_list[0] > max_count:
        raise RuntimeError(f"Simulated crash after {max_count} items")
    return item * 2


def _write_worker_file(args):
    """Helper for concurrent write test: each process writes entries to its worker file."""
    checkpoint_dir, worker_id_base, num_entries = args
    from loqs.internal.streamingmerge import merge_dict_attr

    items_to_write = []
    for i in range(num_entries):
        items_to_write.append((worker_id_base * 100 + i, worker_id_base * 1000 + i))

    # Simulate parallel worker file writes (like a real parallel run)
    worker_file_path = (
        Path(checkpoint_dir) / f"worker_{worker_id()}_runner.h5"
    )
    with h5py.File(worker_file_path, "a") as f:
        for index, result in items_to_write:
            merge_dict_attr(
                f,
                "results",
                [(index, result)],
                key_use_dataset=True,
                value_use_dataset=False,
            )


# Test runner helpers for checkpoint/resume/parallel tests

def _track_shot_executor(item, index, *, shot_executor, **kwargs):
    """Helper function to track shot_executor values."""
    _track_shot_executor.calls.append(shot_executor)
    return item * 2


_track_shot_executor.calls = []


class _SimpleDoubleRunner(MultiProgramRunner):
    """Simple runner that doubles items, for checkpoint tests."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items"]

    def __init__(self, items, **kwargs):
        super().__init__(**kwargs)
        self.items = items

    def _get_items(self):
        return self.items

    def _process_item_fn(self):
        return _double_item

    def _static_kwargs(self):
        return {}

    def _make_on_item_done(self):
        return None

    def _finalize(self, result_list):
        return result_list


class _RaisingRunner(MultiProgramRunner):
    """Runner that raises after N items (crash simulation)."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items"]

    def __init__(self, items, process_fn=_raise_after_n, max_count=999, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self.process_fn = process_fn
        self.call_count = [0]
        self.max_count = max_count

    def _get_items(self):
        return self.items

    def _process_item_fn(self):
        return self.process_fn

    def _static_kwargs(self):
        return {"call_count": self.call_count, "max_count": self.max_count}

    def _make_on_item_done(self):
        return None

    def _finalize(self, result_list):
        return result_list


class _TrackingRunner(MultiProgramRunner):
    """Runner that tracks on_item_done calls."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "items",
        "max_count",
    ]

    def __init__(self, items, process_fn=_double_item, max_count=999, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self.process_fn = process_fn
        self.call_count = [0]
        self.max_count = max_count
        self.on_item_done_calls = []

    def _get_items(self):
        return self.items

    def _process_item_fn(self):
        return self.process_fn

    def _static_kwargs(self):
        return {"call_count": self.call_count, "max_count": self.max_count}

    def _make_on_item_done(self):
        def track(index, item, result):
            self.on_item_done_calls.append((index, item, result))
        return track

    def _finalize(self, result_list):
        return result_list


class _SleepingRunner(MultiProgramRunner):
    """Runner that sleeps before returning results, for timing tests."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items", "sleep_time"]

    def __init__(self, items, sleep_time=0.01, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self.sleep_time = sleep_time
        self.timestamps = []

    def _get_items(self):
        return self.items

    def _process_item_fn(self):
        return _sleep_and_double

    def _static_kwargs(self):
        return {"sleep_time": self.sleep_time}

    def _make_on_item_done(self):
        def track(index, item, result):
            self.timestamps.append(time.time())
        return track

    def _finalize(self, result_list):
        return result_list


class TestMultiProgramRunnerSerialWithCheckpoint:
    """Tests for serial execution with checkpointing."""

    def test_serial_with_checkpoint_full_run(self, tmp_path):
        """Full serial run with checkpointing."""
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(5))

        runner = _TrackingRunner(
            items,
            process_fn=_double_item, checkpoint=True, item_checkpoint_dir=checkpoint_dir,
        )
        results = runner.run()

        assert results == [0, 2, 4, 6, 8]
        # Verify on_item_done was called for each item
        assert len(runner.on_item_done_calls) == 5
        for i, (index, item, result) in enumerate(runner.on_item_done_calls):
            assert index == i
            assert item == i
            assert result == i * 2

        # Verify worker file exists with correct entries
        worker_files = list(checkpoint_dir.glob("worker_*_runner.h5"))
        assert len(worker_files) == 1
        with h5py.File(worker_files[0], "r") as f:
            entries = list(iter_dict_attr_entries(f, "results"))
        assert len(entries) == 5

    def test_serial_crash_simulation_and_resume(self, tmp_path):
        """Simulate a crash and verify resume capability."""
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(10))

        # First run: crash after 3 items
        runner1 = _RaisingRunner(
            items,
            process_fn=_raise_after_n, checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            max_count=3,
        )

        with pytest.raises(RuntimeError, match="Simulated crash"):
            runner1.run()

        # Verify worker file has 3 entries
        worker_files = list(checkpoint_dir.glob("worker_*_runner.h5"))
        assert len(worker_files) == 1
        with h5py.File(worker_files[0], "r") as f:
            entries = list(iter_dict_attr_entries(f, "results"))
        assert len(entries) == 3

        # Second run: resume with normal function on same checkpoint dir
        runner2 = _TrackingRunner(
            items,
            process_fn=_count_and_double, checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir,
        )
        results = runner2.run()

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Only 7 items should have been processed (10 - 3 already done)
        assert runner2.call_count[0] == 7
        # on_item_done should have been called for all 10 items (3 replayed + 7 new)
        assert len(runner2.on_item_done_calls) == 10


class TestMultiProgramRunnerParallel:
    """Tests for parallel execution with checkpointing."""

    def test_parallel_with_checkpoint_full_run(self, tmp_path):
        """Full parallel run with checkpointing."""
        loky = pytest.importorskip("loky")
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(10))

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=4,
        )

        runner = _TrackingRunner(
            items,
            process_fn=_double_item, checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            parallel_strategy=strategy,
        )
        results = runner.run()

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Should have multiple worker files (one per worker)
        worker_files = list(checkpoint_dir.glob("worker_*_runner.h5"))
        assert len(worker_files) >= 1
        # Verify all 10 items were written to worker files
        total_entries = 0
        for worker_file in worker_files:
            with h5py.File(worker_file, "r") as f:
                total_entries += len(
                    list(iter_dict_attr_entries(f, "results"))
                )
        assert total_entries == 10
        # on_item_done should have been called for all items
        assert len(runner.on_item_done_calls) == 10

    def test_parallel_with_polling_updates_during_dispatch(self, tmp_path):
        """Verify on_item_done is called during dispatch, not just after."""
        loky = pytest.importorskip("loky")
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(8))

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=4,
        )

        runner = _SleepingRunner(
            items,
            sleep_time=0.05, checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            parallel_strategy=strategy,
            poll_interval=0.1,
        )
        results = runner.run()

        assert results == [0, 2, 4, 6, 8, 10, 12, 14]
        # Verify all callbacks were made
        assert len(runner.timestamps) == 8
        # Verify timestamps are spread out (not all clustered at end)
        # This is a weak test but good enough to verify polling happened
        if len(runner.timestamps) > 1:
            time_span = runner.timestamps[-1] - runner.timestamps[0]
            # Allow some tolerance but polling should give spread > just a few ms
            assert time_span > 0.01  # At least spread across updates

    def test_parallel_resume_from_partial_run(self, tmp_path):
        """Resume parallel execution from a partial checkpoint."""
        loky = pytest.importorskip("loky")
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(10))

        # First run: do a partial run that completes some items
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=3,
        )

        runner1 = _SimpleDoubleRunner(
            items, checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            parallel_strategy=strategy,
        )

        # Manually create a partial completion scenario by manually seeding
        # the worker files after creating runner.h5
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        runner1_path = checkpoint_dir / "runner.h5"
        runner1.write(runner1_path)
        # Now add partial results
        _seed_partial_worker_file(checkpoint_dir, done_indices=[0, 2, 4])

        # Second run: continue from checkpoint
        runner2 = _SimpleDoubleRunner(
            items, checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir,
            parallel_strategy=strategy,
        )
        results = runner2.run()

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Verify that all 10 items are now done
        from loqs.tools.multiprogramrunner import _read_worker_files

        done = _read_worker_files(checkpoint_dir)
        assert len(done) == 10  # All 10 should be done now
        # The important test is that the final results are correct
        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


class TestConcurrentWorkerWrites:
    """Tests for concurrent writing to worker files."""

    def test_concurrent_workers_writing_simultaneously_lose_no_entries(
        self, tmp_path
    ):
        """Several processes writing worker files concurrently must not corrupt."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        num_workers = 4
        entries_per_worker = 25

        # Spawn multiple processes to write to the same checkpoint_dir
        with mp.Pool(num_workers) as pool:
            pool.map(
                _write_worker_file,
                [
                    (str(checkpoint_dir), worker_id_base, entries_per_worker)
                    for worker_id_base in range(num_workers)
                ],
            )

        # Verify every entry was written
        from loqs.tools.multiprogramrunner import _read_worker_files

        done = _read_worker_files(checkpoint_dir)
        assert len(done) == num_workers * entries_per_worker
        for worker_id_base in range(num_workers):
            for i in range(entries_per_worker):
                index = worker_id_base * 100 + i
                assert index in done
                assert done[index] == worker_id_base * 1000 + i


class TestParallelToolsOnPollCallback:
    """Tests for on_poll callback in paralleltools."""

    def test_submit_executor_on_poll_called_multiple_times(self, tmp_path):
        """on_poll callback is invoked multiple times during dispatch."""
        loky = pytest.importorskip("loky")
        from loqs.tools.paralleltools import run_chunks_with_submit_executor

        call_count = [0]

        def on_poll():
            call_count[0] += 1

        executor = loky.get_reusable_executor(max_workers=2)
        chunks = [[1, 2], [3, 4], [5, 6]]

        def worker(chunk):
            time.sleep(0.05)
            return [x * 2 for x in chunk]

        results = run_chunks_with_submit_executor(
            executor,
            worker,
            chunks,
            on_poll=on_poll,
            poll_interval=0.02,
        )

        assert results == [[2, 4], [6, 8], [10, 12]]
        # on_poll should have been called at least once
        assert call_count[0] >= 1


# Helper functions


def _seed_partial_worker_file(checkpoint_dir: Path, done_indices: list[int]):
    """Seed a checkpoint directory with partial worker file results."""
    worker_file_path = (
        checkpoint_dir / f"worker_{worker_id()}_runner.h5"
    )
    from loqs.internal.streamingmerge import merge_dict_attr

    with h5py.File(worker_file_path, "a") as f:
        for index in done_indices:
            merge_dict_attr(
                f,
                "results",
                [(index, index * 2)],
                key_use_dataset=True,
                value_use_dataset=False,
            )


# Regression tests for bugs fixed


class TestBugRegressions:
    """Regression tests for bugs that were fixed during implementation."""

    def test_bug1_parallel_without_checkpoint_dir_returns_correct_results(self):
        """Bug 1: Parallel execution with item_checkpoint_dir=None was crashing.

        Root cause: _run_parallel didn't capture/return its dispatch results,
        and final assembly had no source of truth when checkpointing was disabled.
        """
        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        runner = _SimpleDoubleRunner(
            list(range(5)),
            parallel_strategy=strategy,
            item_checkpoint_dir=None,
        )
        results = runner.run()
        assert results == [0, 2, 4, 6, 8]

    def test_bug2_serial_respects_parallel_shot_executor(self):
        """Bug 2: Serial execution ignored parallel.shot_executor.

        Root cause: _run_serial hardcoded shot_executor=None instead of
        resolving from the ParallelStrategy.
        """
        class _ShotExecutorTracker(MultiProgramRunner):
            _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items"]

            def __init__(self, items, **kwargs):
                super().__init__(**kwargs)
                self.items = items

            def _get_items(self):
                return self.items

            def _process_item_fn(self):
                return _track_shot_executor

            def _static_kwargs(self):
                return {}

            def _make_on_item_done(self):
                return None

            def _finalize(self, result_list):
                return result_list

        _track_shot_executor.calls = []
        strategy = ParallelStrategy(shot_executor="SENTINEL_EXECUTOR")
        runner = _ShotExecutorTracker(
            [1, 2, 3],
            parallel_strategy=strategy,
            item_checkpoint_dir=None,
        )
        runner.run()
        # All calls should receive the sentinel value, not None
        assert _track_shot_executor.calls == ["SENTINEL_EXECUTOR", "SENTINEL_EXECUTOR", "SENTINEL_EXECUTOR"]

    def test_bug3_parallel_resume_no_double_on_item_done(self, tmp_path):
        """Bug 3: Parallel resume with on_item_done double-invoked for replayed items.

        Root cause: on_poll's observed_indices set wasn't seeded with already-done
        indices, so it re-fired on_item_done during polling for items that were
        already replayed during the initial "replay already-done items" loop.
        """
        loky = pytest.importorskip("loky")
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Set up runner.h5 first, then seed with partial results
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        runner_init = _TrackingRunner(
            list(range(6)),
            process_fn=_double_item, checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            parallel_strategy=strategy,
        )
        runner_init.write(checkpoint_dir / "runner.h5")

        # Seed with 2 already-done items
        _seed_partial_worker_file(checkpoint_dir, done_indices=[0, 1])

        # Now resume from the checkpoint
        runner = _TrackingRunner(
            list(range(6)),
            process_fn=_double_item, checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir,
            parallel_strategy=strategy,
        )
        runner.run()

        # Count invocations per index
        index_counts = {}
        for idx, item, result in runner.on_item_done_calls:
            index_counts[idx] = index_counts.get(idx, 0) + 1

        # Each index should appear exactly once, not twice
        for idx, count in index_counts.items():
            assert count == 1, f"Index {idx} was called {count} times (expected 1)"

    def test_parallel_without_checkpoint_dir_still_fires_on_item_done(self):
        """A parallel run with item_checkpoint_dir=None still invokes on_item_done
        once per item, via a final catch-up pass over any item not already
        observed through checkpoint-directory polling."""
        loky = pytest.importorskip("loky")

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )

        runner = _TrackingRunner(
            list(range(4)),
            process_fn=_double_item,
            parallel_strategy=strategy,
            item_checkpoint_dir=None,
        )
        runner.run()

        # on_item_done should have been called once for each item
        assert len(runner.on_item_done_calls) == 4, f"Expected 4 calls to on_item_done, got {len(runner.on_item_done_calls)}"

        # Verify all expected indices were called (order not guaranteed in parallel)
        called_indices = {call_index for call_index, _, _ in runner.on_item_done_calls}
        assert called_indices == {0, 1, 2, 3}, f"Not all indices called: {called_indices}"

        # Verify results are correct for each item
        for call_index, call_item, call_result in runner.on_item_done_calls:
            assert call_item == call_index, f"Item mismatch for index {call_index}"
            assert call_result == call_index * 2, f"Result mismatch for index {call_index}"


def _multiply_item(item, index, *, shot_executor, multiplier, **kwargs):
    return item * multiplier


class _CountingRunner(MultiProgramRunner):
    """Minimal concrete `MultiProgramRunner` for testing the base class's own
    `run()`/mismatch-check/`force_resume`/crash-recovery behavior in
    isolation from any real tool's domain logic."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "items",
        "multiplier",
    ]

    def __init__(self, items, multiplier=2, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self.multiplier = multiplier

    def _get_items(self):
        return self.items

    def _process_item_fn(self):
        return _multiply_item

    def _static_kwargs(self):
        return {"multiplier": self.multiplier}

    def _make_on_item_done(self):
        return None

    def _finalize(self, result_list):
        return result_list

    def _mismatch_check_fields(self):
        return ["multiplier"]


# Module-level (not test-local) state/function/class for
# test_crash_recovery_via_read_and_run: Serializable.read() resolves a
# decoded object's class by dotted import path, which a test-function-local
# class definition doesn't have.
_FLAKY_CALL_COUNT = {"n": 0}


def _flaky_multiply(item, index, *, shot_executor, multiplier, **kwargs):
    _FLAKY_CALL_COUNT["n"] += 1
    if _FLAKY_CALL_COUNT["n"] == 2:
        raise RuntimeError("simulated crash mid-dispatch")
    return item * multiplier


class _FlakyRunner(_CountingRunner):
    def _process_item_fn(self):
        return _flaky_multiply


class _KeyedRunner(_CountingRunner):
    """_CountingRunner that uses item-based keys for index_map persistence."""

    def _item_key_fn(self):
        # Key items by their string representation (like EdesignRunner does)
        return lambda item: f"item_{item}"


class _KeyedRunnerFixedSignature(MultiProgramRunner):
    """A `MultiProgramRunner` subclass with an explicit, fixed `__init__`
    parameter list -- no `**kwargs` passthrough to `super().__init__`,
    matching real tools like `EdesignRunner`. Proves `index_map`/
    `_reduced_results` survive deserialization even when the subclass's own
    constructor can't accept them directly; a `**kwargs`-forwarding
    subclass like `_CountingRunner` would pass both straight through its
    constructor regardless of whether `_from_decoded_attrs` actually
    restores them, hiding a real regression."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "items",
        "multiplier",
    ]

    def __init__(
        self,
        items,
        multiplier=2,
        checkpoint=False,
        resume=False,
        item_checkpoint_dir=None,
        force_resume=False,
        parallel_strategy=None,
        checkpoint_batch_size=None,
        shot_checkpoint_dir=None,
        lazy_loading=True,
        keep_shot_results=False,
        poll_interval=1.0,
        show_progress=True,
    ):
        super().__init__(
            checkpoint=checkpoint,
            resume=resume,
            parallel_strategy=parallel_strategy,
            item_checkpoint_dir=item_checkpoint_dir,
            force_resume=force_resume,
            checkpoint_batch_size=checkpoint_batch_size,
            shot_checkpoint_dir=shot_checkpoint_dir,
            lazy_loading=lazy_loading,
            keep_shot_results=keep_shot_results,
            poll_interval=poll_interval,
            show_progress=show_progress,
        )
        self.items = items
        self.multiplier = multiplier

    def _get_items(self):
        return self.items

    def _item_key_fn(self):
        return lambda item: f"item_{item}"

    def _process_item_fn(self):
        return _multiply_item

    def _static_kwargs(self):
        return {"multiplier": self.multiplier}

    def _make_on_item_done(self):
        return None

    def _finalize(self, result_list):
        return result_list


class TestProgramRunnerRunAndCrashRecovery:
    """Tests for `MultiProgramRunner.run()`'s own generic checkpoint/resume/
    mismatch-check/crash-recovery behavior, via `_CountingRunner`."""

    def test_run_without_checkpoint_dir(self):
        runner = _CountingRunner([1, 2, 3], multiplier=2)
        assert runner.run() == [2, 4, 6]

    def test_run_writes_runner_h5_before_dispatch_completes(
        self, tmp_path
    ):
        """The runner.h5 snapshot must exist as soon as run() starts
        dispatching, not only after it successfully finishes -- otherwise
        a crash mid-dispatch would leave nothing to recover from."""
        checkpoint_dir = tmp_path / "ckpt"

        def _process_and_check(item, index, *, shot_executor, multiplier, **kwargs):
            assert (checkpoint_dir / "runner.h5").exists()
            return item * multiplier

        class _CheckingRunner(_CountingRunner):
            def _process_item_fn(self):
                return _process_and_check

        runner = _CheckingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        assert runner.run() == [2, 4, 6]

    def test_existing_content_without_runner_h5_raises(self, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "unrelated.txt").write_text("not a runner.h5")

        with pytest.raises(FileExistsError):
            _CountingRunner(
                [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
            ).run()

    def test_matching_config_auto_resumes(self, tmp_path):
        """A matching config with existing checkpoint allows resume."""
        checkpoint_dir = tmp_path / "ckpt"
        first = _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        assert first.run() == [2, 4, 6]

        resumed = _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir
        )
        assert resumed.run() == [2, 4, 6]

    def test_mismatched_config_raises(self, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        ).run()

        mismatched = _CountingRunner(
            [1, 2, 3], multiplier=3, checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir
        )
        with pytest.raises(ValueError, match="multiplier"):
            mismatched.run()

    def test_force_resume_bypasses_mismatch(self, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        ).run()

        mismatched = _CountingRunner(
            [1, 2, 3],
            multiplier=3, checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir,
            force_resume=True,
        )
        # Already-done items are trusted as-is (their original,
        # multiplier=2 results), not recomputed under the new multiplier.
        assert mismatched.run() == [2, 4, 6]

    def test_crash_recovery_via_read_and_run(self, tmp_path):
        """A process interrupted partway through dispatch can be fully
        recovered from just the on-disk runner.h5 -- no need for the
        original script's own in-memory object."""
        checkpoint_dir = tmp_path / "ckpt"
        _FLAKY_CALL_COUNT["n"] = 0

        interrupted = _FlakyRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        with pytest.raises(RuntimeError, match="simulated crash"):
            interrupted.run()

        assert (checkpoint_dir / "runner.h5").exists()

        # Recover using nothing but the on-disk snapshot -- no reference
        # to `interrupted` itself.
        recovered = MultiProgramRunner.read(checkpoint_dir / "runner.h5")
        assert recovered.run() == [2, 4, 6]


class TestMergeReducedResult:
    """Tests for MultiProgramRunner._merge_reduced_result method."""

    def test_merge_reduced_result_persists_to_runner_h5(self, tmp_path):
        """_merge_reduced_result appends to _reduced_results in runner.h5."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        runner.run()

        # Now merge in some reduced results
        runner._merge_reduced_result(0, "reduced_0")
        runner._merge_reduced_result(1, "reduced_1")

        # Verify they were written to runner.h5
        runner_path = checkpoint_dir / "runner.h5"
        from loqs.internal.streamingmerge import iter_dict_attr_entries

        with h5py.File(runner_path, "r") as f:
            reduced = dict(iter_dict_attr_entries(f, "_reduced_results"))

        assert reduced == {0: "reduced_0", 1: "reduced_1"}


class TestIndexMapPersistence:
    """Tests for index_map persistence through deserialization."""

    def test_index_map_stable_across_reordered_resume(self, tmp_path):
        """Items keep their originally-assigned index across a resumed
        `MultiProgramRunner.run()` call even when passed in a different order
        or as a subset."""
        checkpoint_dir = tmp_path / "ckpt"

        runner1 = _KeyedRunner(
            [10, 20, 30], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        assert runner1.run() == [20, 40, 60]
        assert runner1.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

        # Resume with a different order and only a subset -- both items
        # are already done, so this only exercises index stability.
        runner2 = _KeyedRunner(
            [30, 10], multiplier=2, checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir
        )
        assert runner2.run() == [60, 20]
        assert runner2.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

    def test_index_map_survives_deserialization_via_read(self, tmp_path):
        """index_map is correctly restored when deserializing via .read()."""
        checkpoint_dir = tmp_path / "ckpt"
        items = [10, 20, 30]

        # First run: populate index_map with real data
        runner1 = _KeyedRunner(
            items, multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        result1 = runner1.run()
        assert result1 == [20, 40, 60]
        assert runner1.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

        # Deserialize via .read() (the critical test: does index_map survive?)
        runner_path = checkpoint_dir / "runner.h5"
        runner2 = MultiProgramRunner.read(runner_path)

        # The deserialized runner must have the exact same index_map
        # (this is the core assertion that proves the fix works)
        assert runner2.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

    def test_index_map_survives_deserialization_with_fixed_signature_subclass(
        self, tmp_path
    ):
        """index_map/_reduced_results survive `.read()` even for a subclass
        whose own `__init__` has a fixed parameter list and never forwards
        arbitrary `**kwargs` to `super().__init__` -- matching real tools
        like `EdesignRunner`. This is the actual shape the original bug
        occurred against: a `**kwargs`-forwarding test double (like
        `_KeyedRunner` above) would pass `index_map` straight through its
        own constructor regardless of whether `_from_decoded_attrs` pops/
        restores it correctly, so it can't catch this on its own."""
        checkpoint_dir = tmp_path / "ckpt"
        items = [10, 20, 30]

        runner1 = _KeyedRunnerFixedSignature(
            items, multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        result1 = runner1.run()
        assert result1 == [20, 40, 60]
        assert runner1.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

        runner_path = checkpoint_dir / "runner.h5"
        runner2 = _KeyedRunnerFixedSignature.read(runner_path)

        assert runner2.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}


# Test doubles and utilities for keep_shot_results tests


def _make_synthetic_program_results(index, shot_count=5):
    """Create a synthetic ProgramResults for testing."""
    from loqs.core.programresults import ProgramResults
    from loqs.core.history import History
    from loqs.core import Frame

    pr = ProgramResults(lazy_loading=False, name=f"Results_{index}")
    for i in range(shot_count):
        history = History()
        history.append(Frame({"item": index, "shot": i}))
        pr.add_shot(i, history)
    return pr


def _process_item_with_kept_shots(
    item, index, *, shot_executor, keep_shot_results=False, **kwargs
):
    """Test double process_item following the new return contract.

    When keep_shot_results=False, returns bare result (int).
    When keep_shot_results=True, returns (result, pr) tuple where pr is
    the in-memory ProgramResults or None if checkpoint reading will handle it.
    """
    result = item * 2
    if keep_shot_results:
        # Create synthetic results for this item
        pr = _make_synthetic_program_results(index)
        return (result, pr)
    else:
        return result


def _process_item_with_checkpoint(
    item, index, *, shot_executor, keep_shot_results=False, shot_checkpoint_dir=None, **kwargs
):
    """Test double that creates real checkpoint files for each item.

    When keep_shot_results=True, returns (result, pr) where pr is loaded from
    the checkpoint (or None to let the dispatch layer load it).
    When keep_shot_results=False, returns bare result.
    """
    from loqs.core.programresults import ProgramResults

    result = item * 2

    if keep_shot_results and shot_checkpoint_dir is not None:
        # Create a per-item checkpoint directory
        item_dir = Path(shot_checkpoint_dir) / f"item_{index}"
        item_dir.mkdir(parents=True, exist_ok=True)

        # Write the synthetic ProgramResults to checkpoint
        pr = _make_synthetic_program_results(index)
        pr.checkpoint(checkpoint_dir=item_dir)

        # Return None so the dispatch layer will load from checkpoint
        return (result, None)
    elif keep_shot_results:
        # No checkpoint dir, use in-memory
        pr = _make_synthetic_program_results(index)
        return (result, pr)
    else:
        return result


class _KeepShotResultsRunner(MultiProgramRunner):
    """Test runner that supports keep_shot_results."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items"]

    def __init__(self, items, **kwargs):
        super().__init__(**kwargs)
        self.items = items

    def _get_items(self):
        return self.items

    def _process_item_fn(self):
        return _process_item_with_kept_shots

    def _static_kwargs(self):
        return {}

    def _make_on_item_done(self):
        return None

    def _finalize(self, result_list):
        return result_list

    def _shot_checkpoint_subdir(self, index: int) -> Path | None:
        """Override to provide per-item checkpoint directory when set."""
        if self.shot_checkpoint_dir is not None:
            return Path(self.shot_checkpoint_dir) / f"item_{index}"
        return None


class _CheckpointedKeepShotResultsRunner(_KeepShotResultsRunner):
    """Test runner that uses checkpoint_batch_size with keep_shot_results."""

    def _process_item_fn(self):
        return _process_item_with_checkpoint

    def _static_kwargs(self):
        return {"shot_checkpoint_dir": self.shot_checkpoint_dir}


class TestKeepShotResults:
    """Tests for MultiProgramRunner.keep_shot_results mechanism."""

    def test_keep_shot_results_false_default(self, tmp_path):
        """keep_shot_results defaults to False."""
        runner = _KeepShotResultsRunner(
            [1, 2, 3], checkpoint=True, item_checkpoint_dir=tmp_path / "ckpt"
        )
        assert runner.keep_shot_results is False

    def test_keep_shot_results_enabled_on_construction(self, tmp_path):
        """keep_shot_results can be set during construction."""
        runner = _KeepShotResultsRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=tmp_path / "ckpt",
            checkpoint_batch_size=2,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            keep_shot_results=True,
        )
        assert runner.keep_shot_results is True

    def test_keep_shot_results_without_checkpoint_batch_raises(self, tmp_path):
        """keep_shot_results requires checkpoint_batch_size, so kept results are
        always read back from an item's own on-disk shot checkpoint rather than
        held fully in memory for every item at once."""
        with pytest.raises(ValueError, match="checkpoint_batch_size"):
            _KeepShotResultsRunner(
                [1, 2, 3],
                checkpoint=True,
                item_checkpoint_dir=tmp_path / "ckpt",
                keep_shot_results=True,
            )

    def test_keep_shot_results_false_leaves_empty(self, tmp_path):
        """When keep_shot_results=False (the default), _program_results stays empty."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _KeepShotResultsRunner(
            [1, 2, 3], checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            keep_shot_results=False,
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # _program_results should remain empty
        assert len(runner._program_results) == 0

    def test_keep_shot_results_lazy_loading(self, tmp_path):
        """With lazy_loading=True, _program_results contains lazy handles."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _CheckpointedKeepShotResultsRunner(
            [1, 2, 3], checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            checkpoint_batch_size=2,
            keep_shot_results=True,
            lazy_loading=True,
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify _program_results was populated with lazy ProgramResults
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            assert index in runner._program_results
            # Check that it's a lazy ProgramResults (has nested source set)
            pr = runner._program_results[index]
            assert pr._nested_source_file is not None
            assert pr._nested_source_index == index

    def test_keep_shot_results_lazy_loading_disabled(self, tmp_path):
        """With lazy_loading=False, _program_results contains eager results."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _CheckpointedKeepShotResultsRunner(
            [1, 2, 3], checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            checkpoint_batch_size=2,
            keep_shot_results=True,
            lazy_loading=False,
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify _program_results was populated with eager ProgramResults
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            assert index in runner._program_results
            # Check that it has shot_histories eagerly loaded
            pr = runner._program_results[index]
            assert len(pr.shot_histories) == 5  # _make_synthetic_program_results makes 5 shots

    def test_keep_shot_results_resume_preserves(self, tmp_path):
        """Resuming a run with keep_shot_results persists correctly."""
        checkpoint_dir = tmp_path / "ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        # First run completes all items
        runner1 = _CheckpointedKeepShotResultsRunner(
            [1, 2, 3], checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            checkpoint_batch_size=2,
            keep_shot_results=True,
        )
        result1 = runner1.run()
        assert result1 == [2, 4, 6]
        assert len(runner1._program_results) == 3

        # Resume (all items already done)
        runner2 = _CheckpointedKeepShotResultsRunner(
            [1, 2, 3], checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            checkpoint_batch_size=2,
            keep_shot_results=True,
        )
        result2 = runner2.run()
        assert result2 == [2, 4, 6]
        # Program results should still be populated on resume
        assert len(runner2._program_results) == 3

    def test_keep_shot_results_with_checkpoint_batch_serial(self, tmp_path):
        """keep_shot_results with checkpoint_batch_size works in serial dispatch."""
        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        runner = _CheckpointedKeepShotResultsRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=item_checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            checkpoint_batch_size=2,
            keep_shot_results=True,
            lazy_loading=True,
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify _program_results was populated
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            assert index in runner._program_results
            pr = runner._program_results[index]
            # Verify structure shows it's configured for nested loading
            # (the actual shots are in per-item checkpoint dirs, not runner.h5)
            assert pr is not None

    def test_keep_shot_results_with_checkpoint_batch_parallel(self, tmp_path):
        """keep_shot_results with checkpoint_batch_size works in parallel dispatch."""
        import loky

        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        runner = _CheckpointedKeepShotResultsRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=item_checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            checkpoint_batch_size=2,
            keep_shot_results=True,
            lazy_loading=False,
            parallel_strategy=ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=2),
                n_program_chunks=2,
            ),
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify _program_results was populated with eager ProgramResults
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            assert index in runner._program_results
            pr = runner._program_results[index]
            # Should have shot_histories eagerly loaded
            assert len(pr.shot_histories) == 5

    def test_keep_shot_results_lazy_shot_content_verification(self, tmp_path):
        """Verify lazy-loaded ProgramResults can access shot data."""
        item_checkpoint_dir = tmp_path / "item_ckpt"

        runner = _CheckpointedKeepShotResultsRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=item_checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            checkpoint_batch_size=2,
            keep_shot_results=True,
            lazy_loading=True,
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify structure: lazy loading should have created ProgramResults objects
        pr = runner._program_results[1]
        assert pr is not None
        # Verify that it's set up for lazy loading (has nested source configured)
        assert pr._nested_source_file is not None
        assert pr._nested_source_index == 1
        # Verify shots can be retrieved and collected lazily from runner.h5
        shot = pr.get_shot_history(0)
        assert shot is not None
        data = pr.collect_shot_data("item", "all")
        assert len(data) == 5
        assert all(1 in frame_data for frame_data in data)

    def test_keep_shot_results_eager_shot_content_verification(self, tmp_path):
        """Verify eagerly-loaded shots contain correct data end-to-end."""
        item_checkpoint_dir = tmp_path / "item_ckpt"

        runner = _CheckpointedKeepShotResultsRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=item_checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            checkpoint_batch_size=2,
            keep_shot_results=True,
            lazy_loading=False,
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify shot content through eager loading
        pr = runner._program_results[1]
        for shot_idx in range(5):
            shot = pr.shot_histories[shot_idx]
            assert shot is not None
            # Verify frame data
            frame_data = shot.collect_data("item", "all")
            assert 1 in frame_data  # Item index should be 1

    def test_keep_shot_results_write_read_round_trip(self, tmp_path):
        """Writing and reading back a runner with keep_shot_results=True preserves the setting."""
        checkpoint_dir = tmp_path / "checkpoint"
        runner_file = checkpoint_dir / "runner.h5"

        # Create a runner with keep_shot_results=True
        runner1 = _CheckpointedKeepShotResultsRunner(
            [1, 2], checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            checkpoint_batch_size=2,
            keep_shot_results=True,
        )

        # Verify the setting is True before we write
        assert runner1.keep_shot_results is True

        # Run and write to disk
        runner1.run()
        runner1.write(runner_file)

        # Read it back WITHOUT re-passing keep_shot_results
        runner2 = _KeepShotResultsRunner.read(runner_file)

        # Verify that the setting was restored from disk
        assert runner2.keep_shot_results is True


def _shot_progress_item_processor(
    item, index, *, shot_executor, num_shots=5, keep_shot_results=False, **kwargs
):
    """Process an item for shot progress testing."""
    from loqs.core.programresults import ProgramResults
    from loqs.core.history import History
    from loqs.core import Frame

    # Create a ProgramResults with some shots
    pr = ProgramResults()
    for i in range(num_shots):
        history = History()
        history.append(Frame({"item": item, "shot": i}))
        pr.add_shot(i, history)

    # Checkpoint if enabled
    checkpoint_dir = kwargs.get("shot_checkpoint_dir")
    if checkpoint_dir is not None:
        pr.checkpoint(checkpoint_dir=checkpoint_dir)

    if keep_shot_results:
        return item * 2, pr
    else:
        return item * 2


class TestShotProgressBar:
    """Tests for shot-level progress bar (Stage 17.7)."""

    class _ShotProgressTestRunner(MultiProgramRunner):
        """Runner that supports shot-level progress testing."""

        _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items"]

        def __init__(self, items, num_shots=5, **kwargs):
            super().__init__(**kwargs)
            self.items = items
            self.num_shots = num_shots

        def _get_items(self):
            return self.items

        def _process_item_fn(self):
            import functools

            return functools.partial(
                _shot_progress_item_processor, num_shots=self.num_shots
            )

        def _static_kwargs(self):
            return {}

        def _make_on_item_done(self):
            return None

        def _finalize(self, result_list):
            return result_list

        def _shot_checkpoint_subdir(self, index: int) -> Path | None:
            """Return per-item shot checkpoint subdirectory."""
            if (
                self.shot_checkpoint_dir is not None
                and self.checkpoint_batch_size is not None
            ):
                return self.shot_checkpoint_dir / f"item_{index}"
            return None

    def test_num_shots_for_progress_hook_returns_num_shots(self):
        """Verify _num_shots_for_progress returns self.num_shots."""
        runner = self._ShotProgressTestRunner(
            [1, 2, 3], num_shots=10, show_progress=False
        )
        assert runner._num_shots_for_progress() == 10

    def test_current_item_index_round_trip(self, tmp_path):
        """Verify current_item_index attribute round-trips correctly."""
        from loqs.tools.multiprogramrunner import (
            _write_current_item_index_with_retry,
        )

        worker_file = tmp_path / "worker_test_runner.h5"

        # Write current_item_index
        _write_current_item_index_with_retry(worker_file, 42)

        # Read it back
        with h5py.File(worker_file, "r") as f:
            assert f.attrs["current_item_index"] == 42

        # Overwrite with new value
        _write_current_item_index_with_retry(worker_file, 99)

        # Verify it was overwritten
        with h5py.File(worker_file, "r") as f:
            assert f.attrs["current_item_index"] == 99

    def test_read_worker_current_indices_tolerates_missing_files(self, tmp_path):
        """Verify _read_worker_current_indices handles missing/unreadable files."""
        from loqs.tools.multiprogramrunner import _read_worker_current_indices

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        # No workers yet, should return empty set
        indices = _read_worker_current_indices(checkpoint_dir)
        assert indices == set()

        # Create a worker file with current_item_index
        worker_file = checkpoint_dir / "worker_test_runner.h5"
        with h5py.File(worker_file, "a") as f:
            f.attrs["current_item_index"] = 5

        indices = _read_worker_current_indices(checkpoint_dir)
        assert indices == {5}

        # Create another worker file
        worker_file2 = checkpoint_dir / "worker_test2_runner.h5"
        with h5py.File(worker_file2, "a") as f:
            f.attrs["current_item_index"] = 7

        indices = _read_worker_current_indices(checkpoint_dir)
        assert indices == {5, 7}

    def test_shot_progress_prints_once_per_run_parallel(
        self, tmp_path, capsys
    ):
        """Verify shot progress message is printed once when hook is non-None
        but checkpointing isn't configured (parallel dispatch)."""
        loky = pytest.importorskip("loky")

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=2,
        )

        # Parallel dispatch WITHOUT shot checkpointing
        runner = self._ShotProgressTestRunner(
            [1, 2, 3],
            num_shots=5,
            parallel_strategy=strategy,
            show_progress=True,
            # checkpoint_batch_size and shot_checkpoint_dir are NOT set
        )
        runner.run()

        captured = capsys.readouterr()
        assert "Shot-level progress reporting requires" in captured.out
        assert "checkpoint_batch_size" in captured.out
        assert "shot_checkpoint_dir" in captured.out

    def test_shot_progress_silent_when_show_progress_false(
        self, tmp_path, capsys
    ):
        """Verify no message when show_progress=False, even though every
        other condition for it would otherwise be met."""
        loky = pytest.importorskip("loky")

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=2,
        )

        runner = self._ShotProgressTestRunner(
            [1, 2, 3],
            num_shots=5,
            parallel_strategy=strategy,
            show_progress=False,
            # checkpoint_batch_size and shot_checkpoint_dir are NOT set
        )
        runner.run()

        captured = capsys.readouterr()
        assert "Shot-level progress reporting requires" not in captured.out

    def test_shot_progress_silent_for_serial_dispatch(
        self, tmp_path, capsys
    ):
        """Verify no message is printed for serial dispatch even if hook is non-None."""
        # Serial dispatch (no parallel_strategy)
        runner = self._ShotProgressTestRunner(
            [1, 2, 3],
            num_shots=5,
            parallel_strategy=None,
            show_progress=True,
        )
        runner.run()

        captured = capsys.readouterr()
        # Should NOT print the message for serial dispatch
        assert "Shot-level progress reporting requires" not in captured.out

    def test_shot_progress_silent_when_hook_returns_none(
        self, tmp_path, capsys
    ):
        """Verify no message when _num_shots_for_progress returns None."""

        class _NoShotsRunner(self._ShotProgressTestRunner):
            def _num_shots_for_progress(self):
                return None

        loky = pytest.importorskip("loky")
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=2,
        )

        runner = _NoShotsRunner(
            [1, 2, 3],
            num_shots=5,
            parallel_strategy=strategy,
            show_progress=True,
            # checkpointing not configured
        )
        runner.run()

        captured = capsys.readouterr()
        # Should NOT print the message when hook returns None
        assert "Shot-level progress reporting requires" not in captured.out

    def test_shots_bar_suppressed_when_show_progress_false(
        self, tmp_path, capsys
    ):
        """No shots bar (and no misconfiguration print) when show_progress=False,
        even with parallel dispatch and checkpointing fully configured -- the
        shots bar must respect the same opt-out as the plain items bar."""
        loky = pytest.importorskip("loky")

        from unittest.mock import patch
        from tqdm import tqdm as orig_tqdm

        tqdm_calls = []

        def tqdm_spy(*args, **kwargs):
            tqdm_calls.append(kwargs.get("desc", ""))
            return orig_tqdm(*args, **kwargs)

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=2,
        )
        runner = self._ShotProgressTestRunner(
            [1, 2, 3],
            num_shots=5,
            parallel_strategy=strategy,
            checkpoint=True,
            item_checkpoint_dir=tmp_path / "item_ckpt",
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            checkpoint_batch_size=2,
            show_progress=False,
        )
        with patch(
            "loqs.tools.multiprogramrunner.tqdm", side_effect=tqdm_spy
        ):
            runner.run()

        assert "Shots" not in tqdm_calls
        captured = capsys.readouterr()
        assert "Shot-level progress reporting requires" not in captured.out

    def test_shots_bar_total_correct_on_resumed_run(self, tmp_path):
        """Regression test: shots bar total is sized correctly even on resumed run.

        Bug: if shots bar was sized as total=len(remaining) * num_shots instead of
        len(items) * num_shots, then on a resumed run (where remaining < items),
        the initial value would be too small and .n could exceed total.

        This test directly verifies the bar initialization parameters when some
        items are pre-marked as done (simulating a prior interrupted run).
        """
        loky = pytest.importorskip("loky")

        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"
        item_checkpoint_dir.mkdir()
        shot_checkpoint_dir.mkdir()

        from unittest.mock import patch
        from tqdm import tqdm as orig_tqdm

        tqdm_events = []

        class TqdmSpy:
            def __init__(self, *args, **kwargs):
                self.tqdm_obj = orig_tqdm(*args, **kwargs)
                tqdm_events.append(
                    {
                        "event": "init",
                        "total": self.tqdm_obj.total,
                        "initial": self.tqdm_obj.n,
                        "desc": kwargs.get("desc", ""),
                    }
                )

            def __getattr__(self, name):
                return getattr(self.tqdm_obj, name)

            def __setattr__(self, name, value):
                if name == "tqdm_obj":
                    super().__setattr__(name, value)
                else:
                    setattr(self.tqdm_obj, name, value)
                    if name == "n":
                        tqdm_events.append(
                            {
                                "event": "set_n",
                                "n": value,
                                "total": self.tqdm_obj.total,
                                "desc": getattr(self.tqdm_obj, "desc", ""),
                            }
                        )

            def refresh(self):
                return self.tqdm_obj.refresh()

            def close(self):
                return self.tqdm_obj.close()

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=2,
        )

        from loqs.tools import multiprogramrunner as mpr_module

        # Simulates 2 already-done items (a prior interrupted run). Only the
        # tqdm init params computed before dispatch matter here, so a raise
        # from the mismatched later dispatch/assembly logic is swallowed below.
        def read_with_preseeded_done(checkpoint_dir, *args, **kwargs):
            return {0: 0, 2: 4}

        with patch("loqs.tools.multiprogramrunner.tqdm", side_effect=TqdmSpy):
            with patch.object(
                mpr_module,
                "_read_worker_files",
                side_effect=read_with_preseeded_done,
            ):
                runner = self._ShotProgressTestRunner(
                    [1, 2, 3],
                    num_shots=5,
                    checkpoint=True,
                    item_checkpoint_dir=item_checkpoint_dir,
                    parallel_strategy=strategy,
                    shot_checkpoint_dir=shot_checkpoint_dir,
                    checkpoint_batch_size=2,
                    show_progress=True,
                )
                with contextlib.suppress(Exception):
                    runner.run()

        # Find shots bar initialization event
        shots_inits = [
            e
            for e in tqdm_events
            if e["event"] == "init" and e["desc"] == "Shots"
        ]
        assert shots_inits, "Shots bar was never created"
        shots_init = shots_inits[0]

        # total must be len(items)*num_shots=15, not len(remaining)*num_shots=5
        # (items=[1,2,3], done=[0,2], remaining=[1,3]) -- it has to stay fixed
        # across a resume rather than shrinking as items complete.
        assert shots_init["total"] == 15, (
            f"Expected shots_pbar.total=15 (len(items)=3 * num_shots=5), "
            f"got {shots_init['total']}"
        )

        # initial should be len(done) * num_shots = 2 * 5 = 10
        # (indices 0 and 2 were pre-done)
        assert shots_init["initial"] == 10, (
            f"Expected shots_pbar.initial=10 (len(done)=2 * num_shots=5), "
            f"got {shots_init['initial']}"
        )
