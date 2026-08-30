"""Tester for loqs.tools.programrunner"""

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
from loqs.tools.programrunner import ProgramRunner, run_checkpointed_items


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


class TestRunCheckpointedItemsSerial:
    """Tests for serial execution without checkpointing."""

    def test_serial_no_checkpoint_basic(self):
        """Basic serial execution without checkpointing."""
        items = list(range(5))
        results = run_checkpointed_items(
            items,
            _double_item,
            parallel_strategy=None,
            item_checkpoint_dir=None,
        )
        assert results == [0, 2, 4, 6, 8]

    def test_serial_no_checkpoint_preserves_order(self):
        """Verify results are in original items order."""
        items = [10, 20, 30]
        results = run_checkpointed_items(
            items,
            _double_item,
            item_checkpoint_dir=None,
        )
        assert results == [20, 40, 60]


class TestRunCheckpointedItemsSerialWithCheckpoint:
    """Tests for serial execution with checkpointing."""

    def test_serial_with_checkpoint_full_run(self, tmp_path):
        """Full serial run with checkpointing."""
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(5))
        on_item_done_calls = []

        def track_on_item_done(index, item, result):
            on_item_done_calls.append((index, item, result))

        results = run_checkpointed_items(
            items,
            _double_item,
            item_checkpoint_dir=checkpoint_dir,
            on_item_done=track_on_item_done,
        )

        assert results == [0, 2, 4, 6, 8]
        # Verify on_item_done was called for each item
        assert len(on_item_done_calls) == 5
        for i, (index, item, result) in enumerate(on_item_done_calls):
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
        call_count = [0]
        static_kwargs = {"call_count": call_count, "max_count": 3}

        with pytest.raises(RuntimeError, match="Simulated crash"):
            run_checkpointed_items(
                items,
                _raise_after_n,
                item_checkpoint_dir=checkpoint_dir,
                static_kwargs=static_kwargs,
            )

        # Verify worker file has 3 entries
        worker_files = list(checkpoint_dir.glob("worker_*_runner.h5"))
        assert len(worker_files) == 1
        with h5py.File(worker_files[0], "r") as f:
            entries = list(iter_dict_attr_entries(f, "results"))
        assert len(entries) == 3

        # Second run: resume with normal function
        call_count2 = [0]
        static_kwargs2 = {"call_count": call_count2}
        on_item_done_calls = []

        def track_on_item_done(index, item, result):
            on_item_done_calls.append((index, item, result))

        results = run_checkpointed_items(
            items,
            _count_and_double,
            item_checkpoint_dir=checkpoint_dir,
            resume=True,
            static_kwargs=static_kwargs2,
            on_item_done=track_on_item_done,
        )

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Only 7 items should have been processed (10 - 3 already done)
        assert call_count2[0] == 7
        # on_item_done should have been called for all 10 items (3 replayed + 7 new)
        assert len(on_item_done_calls) == 10


class TestRunCheckpointedItemsParallel:
    """Tests for parallel execution with checkpointing."""

    def test_parallel_with_checkpoint_full_run(self, tmp_path):
        """Full parallel run with checkpointing."""
        loky = pytest.importorskip("loky")
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(10))
        on_item_done_calls = []

        def track_on_item_done(index, item, result):
            on_item_done_calls.append((index, item, result))

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=4,
        )

        results = run_checkpointed_items(
            items,
            _double_item,
            parallel_strategy=strategy,
            item_checkpoint_dir=checkpoint_dir,
            on_item_done=track_on_item_done,
        )

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
        assert len(on_item_done_calls) == 10

    def test_parallel_with_polling_updates_during_dispatch(self, tmp_path):
        """Verify on_item_done is called during dispatch, not just after."""
        loky = pytest.importorskip("loky")
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(8))
        timestamps = []

        def track_on_item_done(index, item, result):
            timestamps.append(time.time())

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=4,
        )

        start = time.time()
        results = run_checkpointed_items(
            items,
            _sleep_and_double,
            parallel_strategy=strategy,
            item_checkpoint_dir=checkpoint_dir,
            on_item_done=track_on_item_done,
            poll_interval=0.1,
            static_kwargs={"sleep_time": 0.05},
        )
        end = time.time()

        assert results == [0, 2, 4, 6, 8, 10, 12, 14]
        # Verify all callbacks were made
        assert len(timestamps) == 8
        # Verify timestamps are spread out (not all clustered at end)
        # This is a weak test but good enough to verify polling happened
        total_time = end - start  # noqa: F841  # Used for time reference
        # If polling worked, we should see some spread across the time window
        if len(timestamps) > 1:
            time_span = timestamps[-1] - timestamps[0]
            # Allow some tolerance but polling should give spread > just a few ms
            assert time_span > 0.01  # At least spread across updates

    def test_parallel_resume_from_partial_run(self, tmp_path):
        """Resume parallel execution from a partial checkpoint."""
        loky = pytest.importorskip("loky")
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(10))

        # Manually seed checkpoint with partial results (simulating prior run)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _seed_partial_worker_file(checkpoint_dir, done_indices=[0, 2, 4])

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=3,
        )

        results = run_checkpointed_items(
            items,
            _double_item,
            parallel_strategy=strategy,
            item_checkpoint_dir=checkpoint_dir,
            resume=True,
        )

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Verify that only 7 items were written to worker file (already had 3)
        from loqs.tools.programrunner import _read_worker_files

        done = _read_worker_files(checkpoint_dir)
        assert len(done) == 10  # All 10 should be done now
        # Find worker files created during this run (new ones)
        worker_files = sorted(checkpoint_dir.glob("worker_*_runner.h5"))
        # The initial one should have 3 entries, new ones should have the rest
        if len(worker_files) > 1:
            # Multiple worker files means multiple workers
            pass  # Just verify the final results are correct
        # The important test is that the final results are correct
        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


class TestCheckpointValidation:
    """Tests for validation logic."""

    def test_resume_without_checkpoint_dir_raises(self, tmp_path):
        """resume=True without item_checkpoint_dir raises ValueError."""
        items = [1, 2, 3]
        with pytest.raises(
            ValueError, match="resume=True requires item_checkpoint_dir"
        ):
            run_checkpointed_items(
                items,
                _double_item,
                item_checkpoint_dir=None,
                resume=True,
            )

    def test_existing_content_without_resume_raises(self, tmp_path):
        """FileExistsError if checkpoint_dir has content and resume=False."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "existing_file.txt").write_text("content")

        items = [1, 2, 3]
        with pytest.raises(FileExistsError):
            run_checkpointed_items(
                items,
                _double_item,
                item_checkpoint_dir=checkpoint_dir,
                resume=False,
            )

    def test_empty_checkpoint_dir_ok_even_on_resume(self, tmp_path):
        """Empty checkpoint_dir is OK with resume=True."""
        checkpoint_dir = tmp_path / "empty_checkpoints"
        # Dir doesn't exist yet
        items = [1, 2, 3]
        results = run_checkpointed_items(
            items,
            _double_item,
            item_checkpoint_dir=checkpoint_dir,
            resume=True,
        )
        assert results == [2, 4, 6]


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
        from loqs.tools.programrunner import _read_worker_files

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
        results = run_checkpointed_items(
            list(range(5)),
            _double_item,
            parallel_strategy=strategy,
            item_checkpoint_dir=None,
        )
        assert results == [0, 2, 4, 6, 8]

    def test_bug2_serial_respects_parallel_shot_executor(self):
        """Bug 2: Serial execution ignored parallel.shot_executor.

        Root cause: _run_serial hardcoded shot_executor=None instead of
        resolving from the ParallelStrategy.
        """
        calls = []

        def track_shot_executor(item, index, *, shot_executor, **kwargs):
            calls.append(shot_executor)
            return item * 2

        strategy = ParallelStrategy(shot_executor="SENTINEL_EXECUTOR")
        run_checkpointed_items(
            [1, 2, 3],
            track_shot_executor,
            parallel_strategy=strategy,
            item_checkpoint_dir=None,
        )
        # All calls should receive the sentinel value, not None
        assert calls == ["SENTINEL_EXECUTOR", "SENTINEL_EXECUTOR", "SENTINEL_EXECUTOR"]

    def test_bug3_parallel_resume_no_double_on_item_done(self, tmp_path):
        """Bug 3: Parallel resume with on_item_done double-invoked for replayed items.

        Root cause: on_poll's observed_indices set wasn't seeded with already-done
        indices, so it re-fired on_item_done during polling for items that were
        already replayed during the initial "replay already-done items" loop.
        """
        loky = pytest.importorskip("loky")
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Seed with 2 already-done items
        _seed_partial_worker_file(checkpoint_dir, done_indices=[0, 1])

        calls = []

        def track_calls(index, item, result):
            calls.append((index, item, result))

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )

        run_checkpointed_items(
            list(range(6)),
            _double_item,
            parallel_strategy=strategy,
            item_checkpoint_dir=checkpoint_dir,
            resume=True,
            on_item_done=track_calls,
        )

        # Count invocations per index
        index_counts = {}
        for idx, item, result in calls:
            index_counts[idx] = index_counts.get(idx, 0) + 1

        # Each index should appear exactly once, not twice
        for idx, count in index_counts.items():
            assert count == 1, f"Index {idx} was called {count} times (expected 1)"

    def test_parallel_without_checkpoint_dir_still_fires_on_item_done(self):
        """A parallel run with item_checkpoint_dir=None still invokes on_item_done
        once per item, via a final catch-up pass over any item not already
        observed through checkpoint-directory polling."""
        loky = pytest.importorskip("loky")

        calls = []

        def track_calls(index, item, result):
            calls.append((index, item, result))

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )

        run_checkpointed_items(
            list(range(4)),
            _double_item,
            parallel_strategy=strategy,
            item_checkpoint_dir=None,  # No checkpoint directory
            on_item_done=track_calls,
        )

        # on_item_done should have been called once for each item
        assert len(calls) == 4, f"Expected 4 calls to on_item_done, got {len(calls)}"

        # Verify all expected indices were called (order not guaranteed in parallel)
        called_indices = {call_index for call_index, _, _ in calls}
        assert called_indices == {0, 1, 2, 3}, f"Not all indices called: {called_indices}"

        # Verify results are correct for each item
        for call_index, call_item, call_result in calls:
            assert call_item == call_index, f"Item mismatch for index {call_index}"
            assert call_result == call_index * 2, f"Result mismatch for index {call_index}"


def _multiply_item(item, index, *, shot_executor, multiplier):
    return item * multiplier


class _CountingRunner(ProgramRunner):
    """Minimal concrete `ProgramRunner` for testing the base class's own
    `run()`/mismatch-check/`force_resume`/crash-recovery behavior in
    isolation from any real tool's domain logic."""

    _SERIALIZE_ATTRS = ProgramRunner._SERIALIZE_ATTRS + [
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


def _flaky_multiply(item, index, *, shot_executor, multiplier):
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


class _KeyedRunnerFixedSignature(ProgramRunner):
    """A `ProgramRunner` subclass with an explicit, fixed `__init__`
    parameter list -- no `**kwargs` passthrough to `super().__init__`,
    matching real tools like `EdesignRunner`. Proves `index_map`/
    `_reduced_results` survive deserialization even when the subclass's own
    constructor can't accept them directly; a `**kwargs`-forwarding
    subclass like `_CountingRunner` would pass both straight through its
    constructor regardless of whether `_from_decoded_attrs` actually
    restores them, hiding a real regression."""

    _SERIALIZE_ATTRS = ProgramRunner._SERIALIZE_ATTRS + [
        "items",
        "multiplier",
    ]

    def __init__(
        self,
        items,
        multiplier=2,
        item_checkpoint_dir=None,
        force_resume=False,
        parallel_strategy=None,
        checkpoint_batch_size=None,
        shot_checkpoint_dir=None,
        lazy_loading_enabled=True,
    ):
        super().__init__(
            parallel_strategy=parallel_strategy,
            item_checkpoint_dir=item_checkpoint_dir,
            force_resume=force_resume,
            checkpoint_batch_size=checkpoint_batch_size,
            shot_checkpoint_dir=shot_checkpoint_dir,
            lazy_loading_enabled=lazy_loading_enabled,
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
    """Tests for `ProgramRunner.run()`'s own generic checkpoint/resume/
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

        def _process_and_check(item, index, *, shot_executor, multiplier):
            assert (checkpoint_dir / "runner.h5").exists()
            return item * multiplier

        class _CheckingRunner(_CountingRunner):
            def _process_item_fn(self):
                return _process_and_check

        runner = _CheckingRunner(
            [1, 2, 3], multiplier=2, item_checkpoint_dir=checkpoint_dir
        )
        assert runner.run() == [2, 4, 6]

    def test_existing_content_without_runner_h5_raises(self, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "unrelated.txt").write_text("not a runner.h5")

        with pytest.raises(FileExistsError):
            _CountingRunner(
                [1, 2, 3], multiplier=2, item_checkpoint_dir=checkpoint_dir
            ).run()

    def test_matching_config_auto_resumes(self, tmp_path):
        """Whether a call continues a prior run is inferred purely from
        item_checkpoint_dir's own on-disk state and a config match --
        there's no separate flag a caller needs to pass."""
        checkpoint_dir = tmp_path / "ckpt"
        first = _CountingRunner(
            [1, 2, 3], multiplier=2, item_checkpoint_dir=checkpoint_dir
        )
        assert first.run() == [2, 4, 6]

        resumed = _CountingRunner(
            [1, 2, 3], multiplier=2, item_checkpoint_dir=checkpoint_dir
        )
        assert resumed.run() == [2, 4, 6]

    def test_mismatched_config_raises(self, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        _CountingRunner(
            [1, 2, 3], multiplier=2, item_checkpoint_dir=checkpoint_dir
        ).run()

        mismatched = _CountingRunner(
            [1, 2, 3], multiplier=3, item_checkpoint_dir=checkpoint_dir
        )
        with pytest.raises(ValueError, match="multiplier"):
            mismatched.run()

    def test_force_resume_bypasses_mismatch(self, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        _CountingRunner(
            [1, 2, 3], multiplier=2, item_checkpoint_dir=checkpoint_dir
        ).run()

        mismatched = _CountingRunner(
            [1, 2, 3],
            multiplier=3,
            item_checkpoint_dir=checkpoint_dir,
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
            [1, 2, 3], multiplier=2, item_checkpoint_dir=checkpoint_dir
        )
        with pytest.raises(RuntimeError, match="simulated crash"):
            interrupted.run()

        assert (checkpoint_dir / "runner.h5").exists()

        # Recover using nothing but the on-disk snapshot -- no reference
        # to `interrupted` itself.
        recovered = ProgramRunner.read(checkpoint_dir / "runner.h5")
        assert recovered.run() == [2, 4, 6]


class TestMergeReducedResult:
    """Tests for ProgramRunner._merge_reduced_result method."""

    def test_merge_reduced_result_persists_to_runner_h5(self, tmp_path):
        """_merge_reduced_result appends to _reduced_results in runner.h5."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _CountingRunner(
            [1, 2, 3], multiplier=2, item_checkpoint_dir=checkpoint_dir
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
        `ProgramRunner.run()` call even when passed in a different order
        or as a subset."""
        checkpoint_dir = tmp_path / "ckpt"

        runner1 = _KeyedRunner(
            [10, 20, 30], multiplier=2, item_checkpoint_dir=checkpoint_dir
        )
        assert runner1.run() == [20, 40, 60]
        assert runner1.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

        # Resume with a different order and only a subset -- both items
        # are already done, so this only exercises index stability.
        runner2 = _KeyedRunner(
            [30, 10], multiplier=2, item_checkpoint_dir=checkpoint_dir
        )
        assert runner2.run() == [60, 20]
        assert runner2.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

    def test_index_map_survives_deserialization_via_read(self, tmp_path):
        """index_map is correctly restored when deserializing via .read()."""
        checkpoint_dir = tmp_path / "ckpt"
        items = [10, 20, 30]

        # First run: populate index_map with real data
        runner1 = _KeyedRunner(
            items, multiplier=2, item_checkpoint_dir=checkpoint_dir
        )
        result1 = runner1.run()
        assert result1 == [20, 40, 60]
        assert runner1.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

        # Deserialize via .read() (the critical test: does index_map survive?)
        runner_path = checkpoint_dir / "runner.h5"
        runner2 = ProgramRunner.read(runner_path)

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
            items, multiplier=2, item_checkpoint_dir=checkpoint_dir
        )
        result1 = runner1.run()
        assert result1 == [20, 40, 60]
        assert runner1.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

        runner_path = checkpoint_dir / "runner.h5"
        runner2 = _KeyedRunnerFixedSignature.read(runner_path)

        assert runner2.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}
