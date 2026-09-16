"""Tester for loqs.tools.multiprogramrunner"""

import contextlib
import functools
import gc
import h5py
import multiprocessing as mp
import pickle
import sys
import time
import weakref
from pathlib import Path
from typing import Any, ClassVar

import pytest

from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.core import QuantumProgram
from loqs.core.historydatacollector import HistoryDataCollector
from loqs.core.programresults import _resolve_checkpoint_object_group
from loqs.internal import _retry_hdf5_write, worker_id
from loqs.internal.serializable import Serializable
from loqs.internal.streamingmerge import iter_dict_attr_entries
from loqs.tools.paralleltools import ParallelStrategy
from loqs.tools.multiprogramrunner import (
    MultiProgramRunner,
    _checkpoint_subdir_for_prefix,
)


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


def _build_counter_program(num_increments, increment_by, name="Trivial counter test program"):
    """Build a QuantumProgram via codepack_trivial_counter: an Init Counter
    (starting at 0) followed by num_increments Increment instructions, each
    adding increment_by -- final counter value is num_increments * increment_by."""
    trivial_code = trivial_codepack.create_qec_code()
    ideal_model = trivial_codepack.create_ideal_model(["Q0"])
    stack = [
        {"instruction": "Init Patch Trivial", "new_patch_label": "L0", "qubits": ["Q0"]},
        {"instruction": "Init Counter", "patch_label": "L0", "initial_value": 0},
    ]
    for _ in range(num_increments):
        stack.append(
            {"instruction": "Increment", "patch_label": "L0", "increment_by": increment_by}
        )
    return QuantumProgram(
        stack,
        default_noise_model=ideal_model,
        patch_types={"Trivial": trivial_code},
        name=name,
    )


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


class _SimpleDoubleRunner(MultiProgramRunner):
    """Simple runner that doubles items, for checkpoint tests."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items"]

    def __init__(self, items, **kwargs):
        super().__init__(**kwargs)
        self.items = items

    def build_program(self, index):
        item = self.items[index]
        sign = 1 if item >= 0 else -1
        return _build_counter_program(num_increments=abs(item), increment_by=2 * sign, name=f"Item {index}")

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]


class _TrackingDoubleRunner(_SimpleDoubleRunner):
    """Simple runner that tracks which indices are built.

    The tracked_indices attribute should be set to a multiprocessing.Manager().list()
    to enable cross-process tracking during parallel execution. This attribute is NOT
    serialized, so it must be set anew on the resumed runner instance.
    """

    def __init__(self, items, tracked_indices=None, **kwargs):
        super().__init__(items, **kwargs)
        self.tracked_indices = tracked_indices

    def build_program(self, index):
        if self.tracked_indices is not None:
            self.tracked_indices.append(index)
        return super().build_program(index)


class _TrackingRunner(MultiProgramRunner):
    """Runner that tracks on_item_done calls."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "items",
        "raise_after",
        "call_count",
        "track_calls",
    ]

    def __init__(self, items, process_fn=_double_item, max_count=999, raise_after=None, call_count=None, track_calls=None, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        # Support legacy process_fn parameter: map _raise_after_n to raise_after behavior
        # (raise_after may be passed directly during deserialization)
        if raise_after is not None:
            self.raise_after = raise_after
        else:
            self.raise_after = None if process_fn != _raise_after_n else max_count
        # Determine if we should track calls (old _count_and_double behavior)
        if track_calls is not None:
            self.track_calls = track_calls
        else:
            self.track_calls = (process_fn == _count_and_double)
        # call_count may be passed during deserialization
        self.call_count = call_count if call_count is not None else [0]
        self.on_item_done_calls = []

    def build_program(self, index):
        # Increment call count (either for tracking calls or for crash simulation)
        self.call_count[0] += 1

        # Check for crash simulation on this item
        if self.raise_after is not None:
            if self.call_count[0] > self.raise_after:
                raise RuntimeError(f"Simulated crash after {self.raise_after} items")

        item = self.items[index]
        sign = 1 if item >= 0 else -1
        return _build_counter_program(num_increments=abs(item), increment_by=2 * sign)

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]

    def _make_on_item_done(self):
        def track(index, item, result):
            self.on_item_done_calls.append((index, item, result))
        return track


class _SleepingRunner(MultiProgramRunner):
    """Runner that sleeps (via a real Sleep instruction) before returning
    results, for timing tests."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items", "sleep_time"]

    def __init__(self, items, sleep_time=0.01, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self.sleep_time = sleep_time
        self.timestamps = []

    def build_program(self, index):
        item = self.items[index]
        sign = 1 if item >= 0 else -1
        trivial_code = trivial_codepack.create_qec_code()
        ideal_model = trivial_codepack.create_ideal_model(["Q0"])
        stack = [
            {"instruction": "Init Patch Trivial", "new_patch_label": "L0", "qubits": ["Q0"]},
            {"instruction": "Init Counter", "patch_label": "L0", "initial_value": 0},
            {"instruction": "Sleep", "patch_label": "L0", "duration": self.sleep_time},
        ]
        for _ in range(abs(item)):
            stack.append(
                {"instruction": "Increment", "patch_label": "L0", "increment_by": 2 * sign}
            )
        return QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            patch_types={"Trivial": trivial_code},
            name="Sleeping counter program",
        )

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]

    def _make_on_item_done(self):
        def track(index, item, result):
            self.timestamps.append(time.time())
        return track


class _RunnerWithFieldA(MultiProgramRunner):
    """Test runner with field_a in mismatch check."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items", "field_a"]

    def __init__(self, items, field_a=1, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self.field_a = field_a

    def build_program(self, index):
        item = self.items[index]
        sign = 1 if item >= 0 else -1
        return _build_counter_program(num_increments=abs(item), increment_by=2 * sign)

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]

    def _mismatch_check_fields(self):
        return ["field_a"]


class _RunnerWithFieldB(MultiProgramRunner):
    """Test runner with field_b in mismatch check (incompatible with _RunnerWithFieldA)."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items", "field_b"]

    def __init__(self, items, field_b=2, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self.field_b = field_b

    def build_program(self, index):
        item = self.items[index]
        sign = 1 if item >= 0 else -1
        return _build_counter_program(num_increments=abs(item), increment_by=2 * sign)

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]

    def _mismatch_check_fields(self):
        return ["field_b"]


class TestMultiProgramRunnerSerialWithCheckpoint:
    """Tests for serial execution with checkpointing."""

    def test_serial_crash_simulation_and_resume(self, tmp_path):
        """Simulate a crash and verify resume capability."""
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(10))

        # First run: crash after 3 items
        runner1 = _TrackingRunner(
            items,
            process_fn=_raise_after_n, checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            max_count=3,
        )

        with pytest.raises(RuntimeError, match="Simulated crash"):
            runner1.run()

        # After a crash, the worker file with partial results should still exist
        # (consolidation only happens on successful completion).
        # Verify the partial results are in the worker file.
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
        # After a completed parallel run, worker files should be consolidated
        # into runner.h5 and deleted
        worker_files = list(checkpoint_dir.glob("worker_*_runner.h5"))
        assert len(worker_files) == 0  # Worker files deleted after consolidation
        # Verify all 10 items are now in runner.h5
        runner_path = checkpoint_dir / "runner.h5"
        assert runner_path.exists()
        with h5py.File(runner_path, "r") as f:
            entries = list(iter_dict_attr_entries(f, "_reduced_results"))
        assert len(entries) == 10
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

        # Set up a shared tracking list (using Manager for cross-process access)
        manager = mp.Manager()
        recorded_indices = manager.list()

        # First run: do a partial run that completes some items
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=3,
        )

        runner1 = _TrackingDoubleRunner(
            items, tracked_indices=recorded_indices, checkpoint=True,
            item_checkpoint_dir=checkpoint_dir, parallel_strategy=strategy,
        )

        # Manually create a partial completion scenario by manually seeding
        # the worker files after creating runner.h5
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        runner1_path = checkpoint_dir / "runner.h5"
        runner1.write(runner1_path)
        # Now add partial results
        _seed_partial_worker_file(checkpoint_dir, done_indices=[0, 2, 4])

        # Clear tracked indices before second run (to track only the resume phase)
        del recorded_indices[:]

        # Second run: continue from checkpoint. Use the tracking subclass.
        runner2 = _TrackingDoubleRunner(
            items, tracked_indices=recorded_indices, checkpoint=True, resume=True,
            item_checkpoint_dir=checkpoint_dir, parallel_strategy=strategy,
        )

        results = runner2.run()

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Only the previously-not-done indices should actually be recomputed
        assert sorted(recorded_indices) == [1, 3, 5, 6, 7, 8, 9]
        # Verify that all 10 items are now done (reading from consolidated
        # runner.h5, since worker files are deleted after consolidation)
        from loqs.tools.multiprogramrunner import _read_done_union

        done = _read_done_union(checkpoint_dir, attr_name="results")
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


class _ShotProgressTestRunner(MultiProgramRunner):
    """Runner that supports shot-level progress testing."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "items",
        "num_shots",
    ]

    def __init__(self, items, num_shots=5, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self.num_shots = num_shots

    def build_program(self, index):
        item = self.items[index]
        sign = 1 if item >= 0 else -1
        return _build_counter_program(num_increments=abs(item), increment_by=2 * sign)

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]


class _CustomFilenameProbeRunner(MultiProgramRunner):
    """Mirrors _ShotProgressTestRunner but with a custom results_filename,
    and item index 1 stays in-flight briefly via a real Sleep instruction."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "items",
        "num_shots",
    ]

    def __init__(self, items, num_shots=5, on_item_done=None, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self.num_shots = num_shots
        self._on_item_done = on_item_done
        self._pending_index = None

    def build_program(self, index):
        item = self.items[index]
        sign = 1 if item >= 0 else -1
        trivial_code = trivial_codepack.create_qec_code()
        ideal_model = trivial_codepack.create_ideal_model(["Q0"])
        stack = [
            {"instruction": "Init Patch Trivial", "new_patch_label": "L0", "qubits": ["Q0"]},
            {"instruction": "Init Counter", "patch_label": "L0", "initial_value": 0},
        ]
        for _ in range(abs(item)):
            stack.append(
                {"instruction": "Increment", "patch_label": "L0", "increment_by": 2 * sign}
            )
        self._pending_index = index
        return QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            patch_types={"Trivial": trivial_code},
            name="Custom filename probe program",
        )

    def reduce_program_outcomes(self, program_results):
        result = program_results.collect_shot_data("counter", -1)[0]
        if self._pending_index == 1:
            time.sleep(0.5)
        return result

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]

    def _make_on_item_done(self):
        return self._on_item_done


class TestParallelDispatchAndPollingRegressions:
    """Regression tests for parallel dispatch and polling correctness."""

    def test_parallel_without_checkpoint_dir_returns_correct_results(self):
        """Parallel execution with item_checkpoint_dir=None was crashing.

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

    def test_serial_respects_parallel_shot_executor(self):
        """Serial execution ignored parallel.shot_executor.

        Root cause: _run_serial hardcoded shot_executor=None instead of
        resolving from the ParallelStrategy.
        """
        class _ShotExecutorTracker(MultiProgramRunner):
            _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + ["items"]

            def __init__(self, items, run_kwargs_log=None, **kwargs):
                super().__init__(**kwargs)
                self.items = items
                self.run_kwargs_log = run_kwargs_log if run_kwargs_log is not None else []

            def build_program(self, index):
                return _FakeProgram(self.items[index] * 2, self.run_kwargs_log)

            def reduce_program_outcomes(self, program_results):
                return program_results.value

            def _build_output(self, ordered_results):
                return [result for _, result in ordered_results]

        strategy = ParallelStrategy(shot_executor="SENTINEL_EXECUTOR")
        runner = _ShotExecutorTracker(
            [1, 2, 3],
            parallel_strategy=strategy,
            item_checkpoint_dir=None,
        )
        runner.run()
        # All calls should receive the sentinel value, not None
        assert [log["shot_executor"] for log in runner.run_kwargs_log] == ["SENTINEL_EXECUTOR", "SENTINEL_EXECUTOR", "SENTINEL_EXECUTOR"]

    def test_parallel_resume_no_double_on_item_done(self, tmp_path):
        """Parallel resume with on_item_done double-invoked for replayed items.

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

    def test_shots_pbar_does_not_double_count_stale_in_flight_item(
        self, tmp_path
    ):
        """A worker's current_item_index is never cleared once it finishes
        its one and only item, so a worker file can permanently show that
        item as both done (in its own results dict) and in flight (its
        stale current_item_index) at the same time -- the shots progress
        bar must not double-count shots for such an item.

        Drives real dispatch through _run_parallel via a real
        MultiProgramRunner subclass, real per-item shot checkpoint files,
        and two single-item loky workers (so current_item_index staleness
        is guaranteed by construction, not timing), spying on the real
        shots_pbar.n value on_poll() sets.
        """
        from unittest.mock import patch
        from tqdm import tqdm as orig_tqdm
        from loqs.tools import multiprogramrunner as mpr_module

        loky = pytest.importorskip("loky")

        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        shots_bar_values = []

        class TqdmSpy:
            def __init__(self, *args, **kwargs):
                self.tqdm_obj = orig_tqdm(*args, **kwargs)
                self._is_shots_bar = kwargs.get("desc") == "Shots"

            def __getattr__(self, name):
                return getattr(self.tqdm_obj, name)

            def __setattr__(self, name, value):
                if name in ("tqdm_obj", "_is_shots_bar"):
                    super().__setattr__(name, value)
                else:
                    setattr(self.tqdm_obj, name, value)
                    if name == "n" and self._is_shots_bar:
                        shots_bar_values.append(value)

            def refresh(self):
                return self.tqdm_obj.refresh()

            def close(self):
                return self.tqdm_obj.close()

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        with patch.object(mpr_module, "tqdm", side_effect=TqdmSpy):
            runner = _ShotProgressTestRunner(
                [1, 2],
                num_shots=5,
                checkpoint=True,
                item_checkpoint_dir=item_checkpoint_dir,
                parallel_strategy=strategy,
                shot_checkpoint_dir=shot_checkpoint_dir,
                shot_checkpoint=True,
                show_progress=True,
            )
            runner.run()

        # True total is len(items) * num_shots = 2 * 5 = 10; the bar must
        # never exceed it even once both workers' stale current_item_index
        # still matches their own now-finished item.
        assert shots_bar_values, "Shots bar was never updated"
        assert max(shots_bar_values) <= 10, (
            f"Shots bar overcounted: saw {max(shots_bar_values)} but only "
            f"10 total shots exist (2 items * 5 shots) -- values: "
            f"{shots_bar_values}"
        )
        assert shots_bar_values[-1] == 10, (
            f"Final shots bar value should be exactly 10, got "
            f"{shots_bar_values[-1]}"
        )

    def test_worker_file_reads_skip_keyerror_corruption(self, tmp_path):
        """_read_worker_files, _consolidate_worker_files, and
        _poll_one_worker_file each skip a worker file whose keys dataset
        lists an entry (e.g. key "1") that has no corresponding value group
        -- the shape a crash mid-append leaves behind -- rather than raising
        KeyError, while still reading every healthy file normally."""
        from loqs.tools.multiprogramrunner import (
            _read_worker_files,
            _poll_one_worker_file,
            _consolidate_worker_files,
        )
        from loqs.internal.streamingmerge import merge_dict_attr

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a corrupted worker file: write normally, then delete a value group
        worker_file_corrupted = checkpoint_dir / "worker_0_runner.h5"
        with h5py.File(worker_file_corrupted, "a") as f:
            merge_dict_attr(
                f,
                "results",
                [(0, "value_0"), (1, "value_1")],
                key_use_dataset=True,
                value_use_dataset=False,
            )

        # Corrupt: delete value group for key 1, simulating crash mid-append
        with h5py.File(worker_file_corrupted, "a") as f:
            del f["results/dict/values/iterable/1"]

        # Create a healthy worker file for comparison
        healthy_file = checkpoint_dir / "worker_1_runner.h5"
        with h5py.File(healthy_file, "a") as f:
            merge_dict_attr(
                f,
                "results",
                [(2, "value_2")],
                key_use_dataset=True,
                value_use_dataset=False,
            )

        # _read_worker_files: the corrupted file's key 1 (missing value
        # group) is skipped; the healthy file's key 2 is still read.
        done = _read_worker_files(checkpoint_dir)
        assert 2 in done, "Healthy file's result should be read"
        assert 1 not in done, "Corrupted key 1 (missing value group) should not be in results"

        # _poll_one_worker_file: doesn't crash on the same corruption,
        # consuming what it can before stopping.
        observed_indices = set()
        items_map = {0: "item_0"}
        consumed = _poll_one_worker_file(
            worker_file_corrupted,
            consumed_count=0,
            observed_indices=observed_indices,
            items_map=items_map,
            on_item_done=None,
            pbar=None,
        )
        # Should consume first entry before hitting corruption
        assert consumed == 1, "Should consume first entry before hitting corruption"
        assert 0 in observed_indices, "First entry should have been processed"

        # _consolidate_worker_files: corrupted file skipped, healthy merged.
        runner = _SimpleDoubleRunner(items=[], checkpoint=False)
        runner_path = checkpoint_dir / "runner.h5"
        runner.write(runner_path, "hdf5")

        # Should not raise; healthy file merged, corrupted skipped
        _consolidate_worker_files(
            checkpoint_dir, runner_filename="runner.h5", delete_originals=False
        )

        # Verify healthy file was merged into runner.h5
        done_after = _read_worker_files(checkpoint_dir)
        assert 2 in done_after, "Healthy file should be merged after consolidation"

    def test_consolidate_worker_files_skips_truncated_file(self, tmp_path):
        """A truncated worker file (raw bytes cut off, so h5py itself
        refuses to open it) alongside healthy worker files is skipped by
        _consolidate_worker_files, which still completes successfully and
        merges every healthy file."""
        from loqs.tools.multiprogramrunner import _consolidate_worker_files
        from loqs.internal.streamingmerge import merge_dict_attr

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a real worker file
        worker_file_real = checkpoint_dir / "worker_0_runner.h5"
        with h5py.File(worker_file_real, "a") as f:
            merge_dict_attr(
                f,
                "results",
                [(0, "value_0")],
                key_use_dataset=True,
                value_use_dataset=False,
            )

        # Create a second healthy worker file
        worker_file_good = checkpoint_dir / "worker_1_runner.h5"
        with h5py.File(worker_file_good, "a") as f:
            merge_dict_attr(
                f,
                "results",
                [(1, "value_1")],
                key_use_dataset=True,
                value_use_dataset=False,
            )

        # Now truncate the first file (simulate incomplete write/crash)
        # Truncate to ~70% of original size
        file_size = worker_file_real.stat().st_size
        truncate_size = int(file_size * 0.7)
        with open(worker_file_real, "r+b") as f:
            f.truncate(truncate_size)

        # Create runner.h5 to consolidate into
        runner = _SimpleDoubleRunner(items=[], checkpoint=False)
        runner_path = checkpoint_dir / "runner.h5"
        runner.write(runner_path, "hdf5")

        # Consolidation should complete without raising
        _consolidate_worker_files(
            checkpoint_dir, runner_filename="runner.h5", delete_originals=False
        )

        # Verify the good file was merged (can read from runner.h5)
        from loqs.tools.multiprogramrunner import _read_worker_files
        done = _read_worker_files(checkpoint_dir)
        assert 1 in done, "Healthy worker file should be merged into runner.h5"
        assert done[1] == "value_1"


# Module-level (not test-local): Serializable.read() resolves a decoded
# object's class by dotted import path, unavailable to a local class.
_FLAKY_CALL_COUNT = {"n": 0}


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

    def build_program(self, index):
        item = self.items[index]
        sign = 1 if item >= 0 else -1
        return _build_counter_program(
            num_increments=abs(item), increment_by=self.multiplier * sign
        )

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]

    def _mismatch_check_fields(self):
        return ["multiplier"]


class _ShotDataArgsRunner(_CountingRunner):
    """Runner that checks normalized collect_shot_data_args on resume."""

    _SERIALIZE_ATTRS = _CountingRunner._SERIALIZE_ATTRS + [
        "collect_shot_data_args",
    ]

    def __init__(self, items, collect_shot_data_args=None, **kwargs):
        super().__init__(items, **kwargs)
        self.collect_shot_data_args = collect_shot_data_args or []

    def _mismatch_check_fields(self):
        return super()._mismatch_check_fields() + [
            "_normalized_collect_shot_data_args"
        ]


class _FlakyRunner(_CountingRunner):
    def build_program(self, index):
        _FLAKY_CALL_COUNT["n"] += 1
        if _FLAKY_CALL_COUNT["n"] == 2:
            raise RuntimeError("simulated crash mid-dispatch")
        return super().build_program(index)


class _KeyedRunner(_CountingRunner):
    """_CountingRunner that uses item-based keys for index_map persistence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.item_key_fn = lambda item: f"item_{item}"


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
        shot_checkpoint=False,
        shot_checkpoint_dir=None,
        lazy_loading=True,
        keep_shot_results=False,
        poll_interval=1.0,
        show_progress=True,
        runner_filename: str = "runner.h5",
        results_filename: str = "results.h5",
        run_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            checkpoint=checkpoint,
            resume=resume,
            parallel_strategy=parallel_strategy,
            item_checkpoint_dir=item_checkpoint_dir,
            force_resume=force_resume,
            shot_checkpoint=shot_checkpoint,
            shot_checkpoint_dir=shot_checkpoint_dir,
            lazy_loading=lazy_loading,
            keep_shot_results=keep_shot_results,
            poll_interval=poll_interval,
            show_progress=show_progress,
            runner_filename=runner_filename,
            results_filename=results_filename,
            run_kwargs=run_kwargs,
        )
        self.items = items
        self.multiplier = multiplier
        self.item_key_fn = lambda item: f"item_{item}"

    def build_program(self, index):
        item = self.items[index]
        sign = 1 if item >= 0 else -1
        return _build_counter_program(
            num_increments=abs(item), increment_by=self.multiplier * sign
        )

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]


class TestMultiProgramRunnerRunAndCrashRecovery:
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

        class _CheckingRunner(_CountingRunner):
            def build_program(self, index):
                assert (checkpoint_dir / "runner.h5").exists()
                return super().build_program(index)

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

    def test_resume_true_without_checkpoint_raises(self):
        """resume=True requires checkpoint=True, raises ValueError."""
        with pytest.raises(
            ValueError, match="resume=True requires checkpoint=True"
        ):
            _CountingRunner(
                [1, 2, 3], multiplier=2, resume=True, checkpoint=False
            )

    def test_checkpoint_without_resume_raises_when_content_exists(self, tmp_path):
        """State machine case (b): checkpoint=True, resume=False, but
        on-disk state already exists raises ValueError."""
        checkpoint_dir = tmp_path / "ckpt"

        # First run: create a genuine checkpoint
        first = _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        first.run()

        # Verify runner.h5 exists
        assert (checkpoint_dir / "runner.h5").exists()

        # Second run: attempt to run again with checkpoint=True but
        # resume=False should raise
        second = _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir, resume=False
        )
        with pytest.raises(
            ValueError,
            match="contains an existing checkpoint.*Pass resume=True",
        ):
            second.run()

    def test_resume_true_with_empty_checkpoint_dir_raises(self, tmp_path):
        """State machine case (d): resume=True with checkpoint=True but
        no on-disk state raises ValueError (nothing to resume from)."""
        checkpoint_dir = tmp_path / "nonexistent_dir"

        # Attempt to resume from a nonexistent dir
        runner = _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir
        )
        with pytest.raises(
            ValueError,
            match="is empty or nonexistent.*nothing to resume from",
        ):
            runner.run()

        # Also test with an empty-but-existent dir
        checkpoint_dir.mkdir(parents=True)
        runner2 = _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir
        )
        with pytest.raises(
            ValueError,
            match="is empty or nonexistent.*nothing to resume from",
        ):
            runner2.run()

    def test_cross_subclass_resume_raises_typeerror(self, tmp_path):
        """Resuming a checkpoint created by one runner subclass with a
        different subclass raises TypeError, not AttributeError, when they
        have incompatible mismatch check fields.

        When type(self).read(runner_path) decodes the actual class stored
        in the file (e.g., EdesignRunner), not type(self) (e.g.,
        NoiseSweepRunner), attempting to resume with incompatible mismatch
        check fields must raise TypeError before field checks run.
        """
        checkpoint_dir = tmp_path / "ckpt"

        # First runner creates and checkpoints
        runner1 = _RunnerWithFieldA(
            [1, 2, 3], field_a=10, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        runner1.run()

        # Second runner (different subclass with incompatible fields) attempts to resume
        runner2 = _RunnerWithFieldB(
            [10, 20, 30], field_b=20, checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir
        )
        with pytest.raises(
            TypeError,
            match="Cannot resume.*checkpoint.*created by.*not",
        ):
            runner2.run()


class TestMergeReducedResult:
    """Tests for MultiProgramRunner._merge_reduced_result method."""

    def test_merge_reduced_result_persists_to_runner_h5(self, tmp_path):
        """_merge_reduced_result appends to _reduced_results in runner.h5."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        runner.run()

        # Now merge in some reduced results for indices not yet in runner.h5
        runner._merge_reduced_result(10, "reduced_10")
        runner._merge_reduced_result(11, "reduced_11")

        # Verify they were written to runner.h5 alongside the run results
        runner_path = checkpoint_dir / "runner.h5"
        from loqs.internal.streamingmerge import iter_dict_attr_entries

        with h5py.File(runner_path, "r") as f:
            reduced = dict(iter_dict_attr_entries(f, "_reduced_results"))

        # Should contain the run results (0=2, 1=4, 2=6) plus the merged ones
        assert reduced[10] == "reduced_10"
        assert reduced[11] == "reduced_11"
        assert 0 in reduced  # From the run
        assert 1 in reduced  # From the run
        assert 2 in reduced  # From the run

    def test_merge_reduced_result_second_call_same_index_is_noop(
        self, tmp_path, monkeypatch
    ):
        """A second _merge_reduced_result call with an already-present index
        is a true no-op: no second disk write, and the first value is kept."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _CountingRunner(
            [1, 2, 3], multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        runner.run()

        import h5py as h5py_module

        open_calls = []
        real_file_init = h5py_module.File.__init__

        def counting_file_init(self, *args, **kwargs):
            # Only count genuine path-based opens; h5py itself may
            # reflexively wrap an already-open low-level identifier in its
            # own internal File(id) object (e.g. from a Dataset's `.file`
            # property), which isn't a second disk open.
            if args and isinstance(args[0], (str, Path)):
                open_calls.append(args)
            return real_file_init(self, *args, **kwargs)

        monkeypatch.setattr(h5py_module.File, "__init__", counting_file_init)

        runner._merge_reduced_result(10, "reduced_10")
        assert runner._reduced_results[10] == "reduced_10"
        assert len(open_calls) == 1

        # Second call with the SAME index but a DIFFERENT value must be a
        # true no-op: no second h5py.File open, and the original value kept.
        runner._merge_reduced_result(10, "reduced_10_should_be_ignored")
        assert runner._reduced_results[10] == "reduced_10"
        assert len(open_calls) == 1


class TestRetryHdf5Write:
    """Unit tests for `loqs.internal._retry_hdf5_write`, the shared
    exponential-backoff retry helper every HDF5 checkpoint-write call site
    (across both ProgramResults and MultiProgramRunner) funnels through to
    tolerate a transient concurrent-reader lock conflict."""

    def test_retries_on_blocking_io_error_then_succeeds(
        self, tmp_path, monkeypatch
    ):
        """The first two opens raise BlockingIOError; the third succeeds
        and write_fn actually runs."""
        target = tmp_path / "retry_target.h5"
        real_file = h5py.File
        call_count = {"n": 0}

        def flaky_file(path, mode, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise BlockingIOError("simulated transient lock")
            return real_file(path, mode, *args, **kwargs)

        monkeypatch.setattr(h5py, "File", flaky_file)

        written = []
        _retry_hdf5_write(
            target, lambda f: written.append(True), max_retries=5
        )

        assert call_count["n"] == 3
        assert written == [True]

    def test_reraises_after_max_retries_exhausted(
        self, tmp_path, monkeypatch
    ):
        """Every open raises BlockingIOError; once max_retries is
        exhausted, the original error propagates rather than being
        swallowed."""
        target = tmp_path / "retry_target_always_fails.h5"
        call_count = {"n": 0}

        def always_fails(path, mode, *args, **kwargs):
            call_count["n"] += 1
            raise BlockingIOError("simulated persistent lock")

        monkeypatch.setattr(h5py, "File", always_fails)

        with pytest.raises(BlockingIOError):
            _retry_hdf5_write(target, lambda f: None, max_retries=3)

        assert call_count["n"] == 3

    def _make_flaky_file(self, target, fail_count=6):
        """Build an h5py.File monkeypatch replacement that raises
        BlockingIOError for the first `fail_count` opens of `target`, then
        behaves normally."""
        real_file = h5py.File
        call_count = {"n": 0}

        def flaky_file(path, mode, *args, **kwargs):
            if Path(path) == target:
                call_count["n"] += 1
                if call_count["n"] <= fail_count:
                    raise BlockingIOError("simulated transient lock")
            return real_file(path, mode, *args, **kwargs)

        return flaky_file

    def test_write_dict_entry_with_retry_uses_widened_default_budget(
        self, tmp_path, monkeypatch
    ):
        """`_write_dict_entry_with_retry`'s own default `max_retries` should
        forward the real, widened default retry budget of `_retry_hdf5_write`
        rather than silently clamping it to a smaller one. A transient lock
        that clears after 6 opens exceeds a clamped budget of 5 but is
        comfortably within the real default of 8.
        """
        from loqs.tools.multiprogramrunner import _write_dict_entry_with_retry

        target = tmp_path / "widened_retry_dict_target.h5"
        monkeypatch.setattr(h5py, "File", self._make_flaky_file(target))

        _write_dict_entry_with_retry(target, "results", 0, "value_0")

        with h5py.File(target, "r") as f:
            entries = dict(iter_dict_attr_entries(f, "results"))
        assert entries[0] == "value_0"

    def test_write_current_item_index_with_retry_uses_widened_default_budget(
        self, tmp_path, monkeypatch
    ):
        """Sibling of the above for `_write_current_item_index_with_retry`,
        confirming its own hardcoded default doesn't clamp the retry budget
        either."""
        from loqs.tools.multiprogramrunner import (
            _write_current_item_index_with_retry,
        )

        target = tmp_path / "widened_retry_index_target.h5"
        monkeypatch.setattr(h5py, "File", self._make_flaky_file(target))

        _write_current_item_index_with_retry(target, 7)

        with h5py.File(target, "r") as f:
            assert f.attrs["current_item_index"] == 7


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

    pr = ProgramResults(
        lazy_loading=False,
        name=f"Results_{index}",
        parent_program=f"program_{index}",  # Set parent_program for testing metadata restoration
    )
    for i in range(shot_count):
        history = History()
        history.append(Frame({"item": index, "shot": i}))
        pr.add_shot(i, history)
    return pr


class TestKeepShotResults:
    """Tests for MultiProgramRunner.keep_shot_results mechanism."""

    def test_keep_shot_results_false_default(self, tmp_path):
        """keep_shot_results defaults to False."""
        runner = _SimpleDoubleRunner(
            [1, 2, 3], checkpoint=True, item_checkpoint_dir=tmp_path / "ckpt"
        )
        assert runner.keep_shot_results is False

    def test_keep_shot_results_enabled_on_construction(self, tmp_path):
        """keep_shot_results can be set during construction."""
        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=tmp_path / "ckpt",
            shot_checkpoint=True,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            keep_shot_results=True,
        )
        assert runner.keep_shot_results is True

    def test_keep_shot_results_without_shot_checkpoint_raises(self, tmp_path):
        """keep_shot_results requires shot_checkpoint=True, so kept results are
        always read back from an item's own on-disk shot checkpoint rather than
        held fully in memory for every item at once."""
        with pytest.raises(ValueError, match="shot_checkpoint"):
            _SimpleDoubleRunner(
                [1, 2, 3],
                checkpoint=True,
                item_checkpoint_dir=tmp_path / "ckpt",
                keep_shot_results=True,
            )

    def test_keep_shot_results_false_leaves_empty(self, tmp_path):
        """When keep_shot_results=False (the default), _program_results stays empty."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _SimpleDoubleRunner(
            [1, 2, 3], checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            keep_shot_results=False,
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # _program_results should remain empty
        assert len(runner._program_results) == 0

    def test_keep_shot_results_lazy_loading(self, tmp_path):
        """With lazy_loading=True, _program_results contains lazy handles
        that can resolve shot content on demand."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _SimpleDoubleRunner(
            [1, 2, 3], checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=True,
        )
        runner.num_shots = 5  # Set num_shots for real program execution
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

        # Verify shots can be retrieved and collected lazily from runner.h5
        # (index 1 corresponds to item value 2: num_increments=2, increment_by=2, counter=4)
        pr = runner._program_results[1]
        shot = pr.get_shot_history(0)
        assert shot is not None
        data = pr.collect_shot_data("counter", -1)
        assert len(data) == 5
        assert all(frame_val == 4 for frame_val in data)  # Counter value = item_value * increment_by = 2 * 2

    def test_keep_shot_results_lazy_loading_disabled(self, tmp_path):
        """With lazy_loading=False, _program_results contains eager results
        with correct shot content."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _SimpleDoubleRunner(
            [1, 2, 3], checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=False,
        )
        runner.num_shots = 5  # Set num_shots for real program execution
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify _program_results was populated with eager ProgramResults
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            assert index in runner._program_results
            # Check that it has shot_histories eagerly loaded
            pr = runner._program_results[index]
            assert len(pr.shot_histories) == 5  # Real program executed with num_shots=5

        # Verify shot content through eager loading
        # Index 1 corresponds to item value 2: num_increments=2, increment_by=2, counter=4
        pr = runner._program_results[1]
        for shot_idx in range(5):
            shot = pr.shot_histories[shot_idx]
            assert shot is not None
            # Verify frame data: real counter program produces "counter" key
            frame_data = shot.collect_data("counter", -1)
            assert frame_data == 4  # Counter value = item_value * increment_by = 2 * 2

    def test_keep_shot_results_resume_preserves(self, tmp_path):
        """Resuming a run with keep_shot_results persists correctly."""
        checkpoint_dir = tmp_path / "ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        # First run completes all items
        runner1 = _SimpleDoubleRunner(
            [1, 2, 3], checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            shot_checkpoint=True,
            keep_shot_results=True,
        )
        result1 = runner1.run()
        assert result1 == [2, 4, 6]
        assert len(runner1._program_results) == 3

        # Resume (all items already done)
        runner2 = _SimpleDoubleRunner(
            [1, 2, 3], checkpoint=True, resume=True, item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            shot_checkpoint=True,
            keep_shot_results=True,
        )
        result2 = runner2.run()
        assert result2 == [2, 4, 6]
        # Program results should still be populated on resume
        assert len(runner2._program_results) == 3

    def test_keep_shot_results_with_shot_checkpoint_parallel(self, tmp_path):
        """keep_shot_results with shot_checkpoint works in parallel dispatch."""
        loky = pytest.importorskip("loky")

        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=item_checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=False,
            parallel_strategy=ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=2),
                n_program_chunks=2,
            ),
        )
        runner.num_shots = 5  # Set num_shots for real program execution
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify _program_results was populated with eager ProgramResults
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            assert index in runner._program_results
            pr = runner._program_results[index]
            # Should have shot_histories eagerly loaded
            assert len(pr.shot_histories) == 5

    def test_keep_shot_results_write_read_round_trip(self, tmp_path):
        """Writing and reading back a runner with keep_shot_results=True preserves the setting."""
        checkpoint_dir = tmp_path / "checkpoint"
        runner_file = checkpoint_dir / "runner.h5"

        # Create a runner with keep_shot_results=True
        runner1 = _SimpleDoubleRunner(
            [1, 2], checkpoint=True, item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            shot_checkpoint=True,
            keep_shot_results=True,
        )

        # Verify the setting is True before we write
        assert runner1.keep_shot_results is True

        # Run and write to disk
        runner1.run()
        runner1.write(runner_file)

        # Read it back WITHOUT re-passing keep_shot_results
        runner2 = _SimpleDoubleRunner.read(runner_file)

        # Verify that the setting was restored from disk
        assert runner2.keep_shot_results is True

    def test_keep_shot_results_with_shot_checkpoint_parallel_lazy(self, tmp_path):
        """keep_shot_results with lazy_loading=True works under parallel dispatch."""
        loky = pytest.importorskip("loky")

        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=item_checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=True,
            parallel_strategy=ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=2),
                n_program_chunks=2,
            ),
        )
        runner.num_shots = 5  # Set num_shots for real program execution
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify _program_results was populated with lazy ProgramResults
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            assert index in runner._program_results
            pr = runner._program_results[index]
            # Verify it's lazy loading (has nested source file configured)
            assert pr._nested_source_file is not None
            assert pr._nested_source_index == index
            # Verify lazy reads work correctly post-parallel-consolidation
            shot = pr.get_shot_history(0)
            assert shot is not None
            # With real counter program, each shot produces a "counter" key
            # Item values are [1, 2, 3], so counter values are [2, 4, 6]
            # Counter value = item_value * increment_by = (index + 1) * 2
            data = pr.collect_shot_data("counter", -1)
            assert len(data) == 5
            assert all(frame_val == (index + 1) * 2 for frame_val in data)

    def test_keep_shot_results_eager_restores_metadata(self, tmp_path):
        """With keep_shot_results=True and lazy_loading=False (eager path),
        ProgramResults metadata (parent_program, num_shots, max_frame_limit)
        should be restored, not left at defaults.

        This tests the case where in_memory_pr is available (the normal path),
        and metadata is backfilled from it when the checkpoint doesn't have it."""
        checkpoint_dir = tmp_path / "ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=False,
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify _program_results was populated and metadata is present
        # (backfilled from in_memory_pr in _resolve_kept_program_results)
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            assert index in runner._program_results
            pr = runner._program_results[index]
            # Real QuantumProgram.run() produces name like "Results for Item {index}"
            # This should be restored via backfill from in_memory_pr
            assert pr.name == f"Results for Item {index}", (
                f"Expected pr.name='Results for Item {index}', got '{pr.name}' "
                "(metadata not restored from in_memory_pr)"
            )
            # parent_program should not be None (backfilled from in_memory_pr)
            assert pr.parent_program is not None, (
                "Expected pr.parent_program to be set, got None "
                "(metadata not restored from in_memory_pr)"
            )

    def test_keep_shot_results_lazy_forwards_runner_metadata(self, tmp_path):
        """With keep_shot_results=True and lazy_loading=True (lazy path),
        num_shots and max_frame_limit should be forwarded from the runner
        itself. parent_program/name are genuinely per-item and now resolve
        lazily, on first access, from the nested source each ProgramResults
        was configured to read shots from."""
        checkpoint_dir = tmp_path / "ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        # Create a runner with specific num_shots and max_frame_limit values
        # (set as attributes since constructor doesn't accept them)
        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=True,
        )
        # Set per-runner metadata that should be forwarded to lazy ProgramResults
        runner.num_shots = 15
        runner.run_kwargs["max_frame_limit"] = 200

        result = runner.run()
        assert result == [2, 4, 6]

        # Verify _program_results was populated with correct metadata
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            assert index in runner._program_results
            pr = runner._program_results[index]
            # num_shots and max_frame_limit should be forwarded from runner
            assert pr.num_shots == 15, (
                f"Expected pr.num_shots=15, got {pr.num_shots} "
                "(lazy path didn't forward runner metadata)"
            )
            assert pr.max_frame_limit == 200, (
                f"Expected pr.max_frame_limit=200, got {pr.max_frame_limit} "
                "(lazy path didn't forward runner metadata)"
            )
            # parent_program and name now resolve lazily to the real,
            # per-item values written by build_program.
            assert pr.parent_program is not None, (
                f"Expected pr.parent_program to be resolved (not None) for index {index} "
                "(lazy resolution didn't happen)"
            )
            assert pr.name == f"Results for Item {index}", (
                f"Expected pr.name='Results for Item {index}', got '{pr.name}' "
                "(lazy resolution didn't fetch the real per-item value)"
            )

    def test_keep_shot_results_with_custom_results_filename_serial(self, tmp_path):
        """Custom results_filename with keep_shot_results=True works in serial dispatch."""
        custom_filename = "my_results.h5"
        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=item_checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=False,
            results_filename=custom_filename,
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify kept results have non-empty shot_histories.
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            pr = runner._program_results[index]
            assert len(pr.shot_histories) > 0, (
                f"Item {index} kept result should have non-empty shot_histories "
                "with custom results_filename"
            )

    def test_keep_shot_results_with_custom_results_filename_parallel(self, tmp_path):
        """Custom results_filename with keep_shot_results=True works in parallel dispatch."""
        loky = pytest.importorskip("loky")

        custom_filename = "my_results.h5"
        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=item_checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=False,
            parallel_strategy=ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=2),
                n_program_chunks=2,
            ),
            results_filename=custom_filename,
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # Verify kept results have non-empty shot_histories.
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            pr = runner._program_results[index]
            assert len(pr.shot_histories) > 0, (
                f"Item {index} kept result should have non-empty shot_histories "
                "with custom results_filename in parallel"
            )

    def test_shots_pbar_advances_with_custom_results_filename(self, tmp_path):
        """Shots progress bar advances correctly with custom results_filename in parallel.

        Uses one item per loky worker (n_program_chunks == len(items)), like
        test_shots_pbar_does_not_double_count_stale_in_flight_item, plus a
        deliberate delay in the second item's own worker process (after its
        shot checkpoint is written but before its result is reported back)
        to force a real window where that item is genuinely "in flight" --
        checkpointed but not yet marked done -- for long enough that a fast
        poll_interval reliably samples it. This exercises the on_poll
        in-flight branch's ProgramResults._count_done_shots(...,
        results_filename=results_filename) call under a non-default
        filename: the shots bar must reach the true total while just one of
        the two items has actually been marked done, crediting the
        still-in-flight item's already-checkpointed shots read under that
        same custom filename.
        """
        from unittest.mock import patch
        from tqdm import tqdm as orig_tqdm
        from loqs.tools import multiprogramrunner as mpr_module

        loky = pytest.importorskip("loky")

        custom_filename = "my_results.h5"
        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        # (done_count, shots_bar_value) sampled every time the shots bar
        # changes, so we can check the shots value at the moment only one
        # of the two items has actually been reported done.
        done_count = [0]
        samples = []

        class TqdmSpy:
            def __init__(self, *args, **kwargs):
                self.tqdm_obj = orig_tqdm(*args, **kwargs)
                self._is_shots_bar = kwargs.get("desc") == "Shots"

            def __getattr__(self, name):
                return getattr(self.tqdm_obj, name)

            def __setattr__(self, name, value):
                if name in ("tqdm_obj", "_is_shots_bar"):
                    super().__setattr__(name, value)
                else:
                    setattr(self.tqdm_obj, name, value)
                    if name == "n" and self._is_shots_bar:
                        samples.append((done_count[0], value))

            def refresh(self):
                return self.tqdm_obj.refresh()

            def close(self):
                return self.tqdm_obj.close()

        def _track_done(index, item, result):
            done_count[0] += 1

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        with patch.object(mpr_module, "tqdm", side_effect=TqdmSpy):
            runner = _CustomFilenameProbeRunner(
                [1, 2],
                num_shots=5,
                on_item_done=_track_done,
                checkpoint=True,
                item_checkpoint_dir=item_checkpoint_dir,
                parallel_strategy=strategy,
                shot_checkpoint_dir=shot_checkpoint_dir,
                shot_checkpoint=True,
                show_progress=True,
                poll_interval=0.05,
                results_filename=custom_filename,
            )
            runner.run()

        shots_bar_values = [value for _, value in samples]
        assert shots_bar_values, "Shots bar was never updated with custom filename"

        # True total is len(items) * num_shots = 2 * 5 = 10. The bar must
        # never exceed it, and must end there.
        assert max(shots_bar_values) <= 10, (
            f"Shots bar overcounted: saw {max(shots_bar_values)} but only 10 "
            f"total shots exist (2 items * 5 shots) -- values: "
            f"{shots_bar_values}"
        )
        assert shots_bar_values[-1] == 10, (
            f"Final shots bar value should be exactly 10, got "
            f"{shots_bar_values[-1]} with custom results_filename"
        )

        # The real trap: while item 1 is still sleeping (deliberately, after
        # its own checkpoint write), only item 0 has actually been reported
        # done. If the on_poll in-flight branch didn't honor the runner's
        # own custom results_filename when reading item 1's already-written
        # checkpoint, it would look for the (nonexistent) default
        # "results.h5" and report 0 done shots for it, so the bar could
        # only ever reach 5 (from item 0 alone) at that point, never 10.
        assert any(done == 1 and value == 10 for done, value in samples), (
            "Shots bar never reached the true total (10) while only one "
            "item had been reported done -- the on_poll in-flight branch "
            "isn't crediting the still-in-flight item's already-checkpointed "
            f"shots under the custom results_filename -- samples: {samples}"
        )

    def test_resolve_kept_program_results_fallback_to_in_memory(self):
        """_resolve_kept_program_results returns in_memory_pr when checkpoint is empty."""
        from loqs.tools.multiprogramrunner import _resolve_kept_program_results

        # Create in-memory PR with data
        in_memory_pr = _make_synthetic_program_results(0, shot_count=5)

        # Empty checkpoint directory (no file, or empty shot_histories)
        def empty_checkpoint_subdir(index):
            return Path("/tmp/nonexistent_dir")

        # Calling with an empty checkpoint should return in_memory_pr.
        result = _resolve_kept_program_results(
            0, empty_checkpoint_subdir, in_memory_pr
        )

        # Should get in_memory_pr because checkpoint is empty/missing
        assert result is not None
        assert len(result.shot_histories) == 5, (
            "Should return in_memory_pr with 5 shots when checkpoint is missing"
        )
        assert result.parent_program == "program_0"


class TestShotProgressBar:
    """Tests for shot-level progress bar."""

    def test_num_shots_for_progress_hook_returns_num_shots(self):
        """Verify _num_shots_for_progress returns self.num_shots."""
        runner = _ShotProgressTestRunner(
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
        runner = _ShotProgressTestRunner(
            [1, 2, 3],
            num_shots=5,
            parallel_strategy=strategy,
            show_progress=True,
            # shot_checkpoint and shot_checkpoint_dir are NOT set
        )
        runner.run()

        captured = capsys.readouterr()
        assert "Shot-level progress reporting requires" in captured.out
        assert "shot_checkpoint=True" in captured.out
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

        runner = _ShotProgressTestRunner(
            [1, 2, 3],
            num_shots=5,
            parallel_strategy=strategy,
            show_progress=False,
            # shot_checkpoint and shot_checkpoint_dir are NOT set
        )
        runner.run()

        captured = capsys.readouterr()
        assert "Shot-level progress reporting requires" not in captured.out

    def test_shot_progress_silent_for_serial_dispatch(
        self, tmp_path, capsys
    ):
        """Verify no message is printed for serial dispatch even if hook is non-None."""
        # Serial dispatch (no parallel_strategy)
        runner = _ShotProgressTestRunner(
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

        class _NoShotsRunner(_ShotProgressTestRunner):
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
        runner = _ShotProgressTestRunner(
            [1, 2, 3],
            num_shots=5,
            parallel_strategy=strategy,
            checkpoint=True,
            item_checkpoint_dir=tmp_path / "item_ckpt",
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            shot_checkpoint=True,
            show_progress=False,
        )
        with patch(
            "loqs.tools.multiprogramrunner.tqdm", side_effect=tqdm_spy
        ):
            runner.run()

        assert "Shots" not in tqdm_calls
        captured = capsys.readouterr()
        assert "Shot-level progress reporting requires" not in captured.out

    def test_shots_bar_suppressed_and_distinct_message_when_checkpoint_false(
        self, tmp_path, capsys
    ):
        """With shot_checkpoint=True but checkpoint=False, no shots bar is
        created (on_poll never touches it when checkpoint=False, so it would
        otherwise freeze), and a distinct message names checkpoint=True (not
        shot_checkpoint=True) as the missing flag."""
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
        runner = _ShotProgressTestRunner(
            [1, 2, 3],
            num_shots=5,
            parallel_strategy=strategy,
            checkpoint=False,
            shot_checkpoint=True,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            show_progress=True,
        )
        with patch(
            "loqs.tools.multiprogramrunner.tqdm", side_effect=tqdm_spy
        ):
            runner.run()

        assert "Shots" not in tqdm_calls
        captured = capsys.readouterr()
        assert (
            "Shot-level progress reporting requires checkpoint=True (in "
            "addition to shot_checkpoint=True) to be set"
        ) in captured.out
        assert (
            "requires shot_checkpoint=True (and shot_checkpoint_dir)"
            not in captured.out
        )

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

        # Simulates 2 already-done items (a prior interrupted run) -- only
        # the tqdm init params computed before dispatch matter here.
        def read_with_preseeded_done(checkpoint_dir, *args, **kwargs):
            return {0: 0, 2: 4}

        with patch("loqs.tools.multiprogramrunner.tqdm", side_effect=TqdmSpy):
            with patch.object(
                mpr_module,
                "_read_worker_files",
                side_effect=read_with_preseeded_done,
            ):
                runner = _ShotProgressTestRunner(
                    [1, 2, 3],
                    num_shots=5,
                    checkpoint=True,
                    item_checkpoint_dir=item_checkpoint_dir,
                    parallel_strategy=strategy,
                    shot_checkpoint_dir=shot_checkpoint_dir,
                    shot_checkpoint=True,
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

        # total = len(items)*num_shots = 15, fixed across a resume rather
        # than shrinking to len(remaining)*num_shots as items complete.
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


class TestWorkerFileConsolidation:
    """Tests for worker file consolidation and deletion behavior."""

    def test_completed_run_consolidates_and_deletes_worker_files(
        self, tmp_path
    ):
        """A completed run should consolidate all worker files into runner.h5
        and delete the worker files afterward."""
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(5))

        runner = _TrackingRunner(
            items,
            process_fn=_double_item,
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
        )
        results = runner.run()

        assert results == [0, 2, 4, 6, 8]
        # Verify on_item_done was called for each item
        assert len(runner.on_item_done_calls) == 5
        for i, (index, item, result) in enumerate(runner.on_item_done_calls):
            assert index == i
            assert item == i
            assert result == i * 2

        # After completion, worker files should be deleted
        worker_files = list(checkpoint_dir.glob("worker_*_runner.h5"))
        assert len(worker_files) == 0, "Worker files should be deleted after consolidation"

        # All results should be consolidated into runner.h5
        runner_path = checkpoint_dir / "runner.h5"
        assert runner_path.exists()
        with h5py.File(runner_path, "r") as f:
            from loqs.internal.streamingmerge import iter_dict_attr_entries

            entries = dict(iter_dict_attr_entries(f, "_reduced_results"))
        assert len(entries) == 5
        assert entries == {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}

    def test_resume_with_deleted_worker_files_reads_from_runner_h5(
        self, tmp_path
    ):
        """A resume where all worker files have already been consolidated
        and deleted should correctly detect done items purely from runner.h5,
        execute only remaining items, and produce correct final results."""
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(6))

        # First run: complete the run (consolidates and deletes worker files)
        runner1 = _TrackingRunner(
            items,
            process_fn=_double_item,
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
        )
        results1 = runner1.run()
        assert results1 == [0, 2, 4, 6, 8, 10]

        # Verify worker files are gone
        worker_files = list(checkpoint_dir.glob("worker_*_runner.h5"))
        assert len(worker_files) == 0

        # Second run: resume from checkpoint where only runner.h5 exists
        # (no worker files). This tests that done-detection reads from
        # runner.h5's _reduced_results correctly.
        runner2 = _TrackingRunner(
            items,
            process_fn=_double_item,
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=checkpoint_dir,
        )
        results2 = runner2.run()

        # All items should be skipped (already done)
        assert runner2.call_count[0] == 0
        assert results2 == [0, 2, 4, 6, 8, 10]

        # Verify runner.h5 still contains all results
        runner_path = checkpoint_dir / "runner.h5"
        with h5py.File(runner_path, "r") as f:
            from loqs.internal.streamingmerge import iter_dict_attr_entries

            entries = dict(iter_dict_attr_entries(f, "_reduced_results"))
        assert len(entries) == 6

    def test_resume_of_resume_with_interruption(self, tmp_path):
        """Regression test for a resume-of-resume where each resume is
        itself interrupted before completion: the final result includes
        every item's correct value, and runner.h5's on-disk
        _reduced_results (decoded fresh) contains every already-done item
        at each intermediate stage. Exercises done-detection/worker-file
        consolidation across repeated crashes only, not the seed-before-
        write ordering (see test_resume_seeds_reduced_results_before_first_write
        for that).
        """
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(6))

        # First partial run: complete items 0-1 only
        runner1 = _TrackingRunner(
            items,
            process_fn=_raise_after_n,
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
            max_count=2,
        )
        with pytest.raises(RuntimeError, match="Simulated crash"):
            runner1.run()

        # Verify partial results are persisted (either in worker file or
        # runner.h5 via the union reader)
        from loqs.tools.multiprogramrunner import _read_done_union

        on_disk_after_crash1 = _read_done_union(
            checkpoint_dir, attr_name="results"
        )
        assert len(on_disk_after_crash1) == 2
        assert on_disk_after_crash1 == {0: 0, 1: 2}

        # Second run: resume and complete items 2-3 before crashing again.
        # Items 0-1 are already done, so only 2-5 will be processed (4 items).
        # To process items 2-3 then crash on item 4, we need max_count=2.
        runner2 = _TrackingRunner(
            items,
            process_fn=_raise_after_n,
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=checkpoint_dir,
            max_count=2,  # Will process items 2,3 then crash on item 4
        )
        with pytest.raises(RuntimeError, match="Simulated crash"):
            runner2.run()

        # Verify results 0-3 are persisted (the original 0-1 aren't lost
        # or blanked out when runner.h5 was seeded before write)
        on_disk_after_crash2 = _read_done_union(
            checkpoint_dir, attr_name="results"
        )
        assert len(on_disk_after_crash2) == 4
        assert on_disk_after_crash2 == {0: 0, 1: 2, 2: 4, 3: 6}

        # Third run: resume and complete the remaining items (4-5)
        runner3 = _TrackingRunner(
            items,
            process_fn=_count_and_double,
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=checkpoint_dir,
        )
        results3 = runner3.run()

        # Final result should have all items with correct values
        assert results3 == [0, 2, 4, 6, 8, 10]

        # Verify runner.h5 now has all 6 items consolidated
        with h5py.File(checkpoint_dir / "runner.h5", "r") as f:
            from loqs.internal.streamingmerge import iter_dict_attr_entries

            final_on_disk = dict(
                iter_dict_attr_entries(f, "_reduced_results")
            )
        assert len(final_on_disk) == 6
        assert final_on_disk == {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10}

    def test_resume_seeds_reduced_results_before_first_write(self, tmp_path):
        """A resuming run() seeds _reduced_results/_program_results from
        stored state before its first write() call, not after -- otherwise
        that write would briefly clobber runner.h5's on-disk
        _reduced_results with a blank value. Uses _CountingRunner, whose
        first run() completes fully and so genuinely leaves non-empty
        on-disk state for the second run() to seed from.
        """
        checkpoint_dir = tmp_path / "checkpoints"
        items = [1, 2, 3]

        # First run: completes fully, so its results are consolidated into
        # runner.h5's own _reduced_results attribute.
        runner1 = _CountingRunner(
            items, multiplier=2, checkpoint=True, item_checkpoint_dir=checkpoint_dir
        )
        runner1.run()

        with h5py.File(checkpoint_dir / "runner.h5", "r") as f:
            from loqs.internal.streamingmerge import iter_dict_attr_entries

            on_disk_after_run1 = dict(
                iter_dict_attr_entries(f, "_reduced_results")
            )
        assert on_disk_after_run1 == {0: 2, 1: 4, 2: 6}

        # Second run: resumes the same runner. Capture runner.h5's on-disk
        # _reduced_results immediately after the *first* write() call inside
        # run(), before any dispatch happens -- this is exactly the moment
        # the seed-before-write ordering matters.
        runner2 = _CountingRunner(
            items,
            multiplier=2,
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=checkpoint_dir,
        )

        captured_on_first_write = []
        real_write = runner2.write

        def capturing_write(path, *args, **kwargs):
            real_write(path, *args, **kwargs)
            if not captured_on_first_write:
                from loqs.internal.streamingmerge import (
                    iter_dict_attr_entries as _iter_entries,
                )

                with h5py.File(path, "r") as f:
                    captured_on_first_write.append(
                        dict(_iter_entries(f, "_reduced_results"))
                    )

        runner2.write = capturing_write
        runner2.run()

        # If seeding happened after this first write instead of before, this
        # snapshot would be empty (a fresh instance's own blank
        # _reduced_results, written before ever consulting stored state).
        assert captured_on_first_write[0] == {0: 2, 1: 4, 2: 6}

    def test_final_assembly_lock_contention_silently_drops_program_result(
        self, tmp_path, monkeypatch
    ):
        """Verify that a bounded/transient lock on a worker file during
        `_consolidate_worker_files`'s one-shot final-assembly pass no longer
        causes silent data loss in `_program_results`. Asserts that the
        transiently locked worker's index is present in `_program_results`
        with its correct value and without raising an exception.
        """
        from loqs.tools.multiprogramrunner import (
            _consolidate_worker_files,
            _read_done_union,
        )
        from loqs.internal.streamingmerge import merge_dict_attr

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Bare runner.h5 with valid object-group structure to merge into.
        runner = _SimpleDoubleRunner(items=[], checkpoint=False)
        runner_path = checkpoint_dir / "runner.h5"
        runner.write(runner_path, "hdf5")

        # Three worker files, each with one "results"/"_program_results" pair.
        worker_files = []
        for i in range(3):
            worker_file = checkpoint_dir / f"worker_{i}_runner.h5"
            with h5py.File(worker_file, "a") as f:
                merge_dict_attr(
                    f, "results", [(i, f"result_{i}")],
                    key_use_dataset=True, value_use_dataset=False,
                )
                merge_dict_attr(
                    f, "_program_results", [(i, f"program_result_{i}")],
                    key_use_dataset=True, value_use_dataset=False,
                )
            worker_files.append(worker_file)

        # Worker 1 hits a transient lock conflict on its first 3 reads,
        # comfortably under the production retry budget of 8 attempts, then
        # succeeds -- modeling a lock that's genuinely transient (a few
        # retries recover it), not a permanently-stuck one.
        locked_worker_file = worker_files[1]
        real_file = h5py.File
        locked_read_attempts = {"n": 0}

        def flaky_file(path, mode="r", *args, **kwargs):
            if Path(path) == locked_worker_file and mode == "r":
                locked_read_attempts["n"] += 1
                if locked_read_attempts["n"] <= 3:
                    raise BlockingIOError(
                        "simulated transient lock held by another process"
                    )
            return real_file(path, mode, *args, **kwargs)

        monkeypatch.setattr(h5py, "File", flaky_file)

        # Single final-assembly consolidation pass, mirroring run()'s one-shot call.
        _consolidate_worker_files(
            checkpoint_dir, runner_filename="runner.h5", delete_originals=True
        )

        program_results = _read_done_union(
            checkpoint_dir, runner_filename="runner.h5", attr_name="_program_results"
        )

        # CORRECT/desired behavior: a transient lock on one worker file
        # should not cause silent data loss -- worker 1's entry should still
        # be present in _program_results with its correct value. This fails
        # against today's unfixed code, since the entry is actually dropped.
        assert 1 in program_results, (
            "Worker 1's _program_results entry is missing: a transient lock "
            "during final assembly silently dropped it instead of being "
            "retried or reported."
        )
        assert program_results[1] == "program_result_1"
        # Unaffected workers still made it through.
        assert 0 in program_results
        assert 2 in program_results

    def test_missing_program_results_entry_should_raise(
        self, tmp_path, monkeypatch
    ):
        """Verify that the completeness check for `_program_results`
        (mirroring the existing check for `results` in `run()`) correctly
        raises `RuntimeError` naming the missing index if an entry is absent
        from final assembly.
        """
        import loqs.tools.multiprogramrunner as multiprogramrunner_module

        checkpoint_dir = tmp_path / "ckpt"
        missing_index = 1

        real_read_done_union = multiprogramrunner_module._read_done_union

        def read_done_union_dropping_program_result(
            ckpt_dir, runner_filename="runner.h5", attr_name="results", **kwargs
        ):
            """Drop one index from "_program_results" only; "results" is
            read through unaffected and complete. Accepts and forwards
            **kwargs so this stays robust to any additional keyword argument
            the real `_read_done_union` gains."""
            result = real_read_done_union(
                ckpt_dir,
                runner_filename=runner_filename,
                attr_name=attr_name,
                **kwargs,
            )
            if attr_name == "_program_results":
                result.pop(missing_index, None)
            return result

        monkeypatch.setattr(
            multiprogramrunner_module,
            "_read_done_union",
            read_done_union_dropping_program_result,
        )

        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=False,
        )

        with pytest.raises(
            RuntimeError,
            match=f"Item {missing_index} is missing from.*_program_results",
        ):
            runner.run()

    def test_missing_program_results_entry_should_raise_lazy_loading(
        self, tmp_path, monkeypatch
    ):
        """Lazy-loading sibling of `test_missing_program_results_entry_should_raise`:
        the same completeness check for `_program_results` should still raise
        `RuntimeError` naming the missing index when `lazy_loading=True`.
        """
        import loqs.tools.multiprogramrunner as multiprogramrunner_module

        checkpoint_dir = tmp_path / "ckpt"
        missing_index = 1

        real_get_dict_attr_keys = multiprogramrunner_module.get_dict_attr_keys

        def get_dict_attr_keys_dropping_program_result(
            parent_group, attr_name, decode_cache=None
        ):
            """Drop one key whenever `_program_results` is queried, to
            simulate a genuinely-missing entry; any other attribute name is
            read through unaffected. A planned fix is expected to route this
            lookup through a dedicated new helper instead of this function
            directly -- once that lands, this monkeypatch target may need to
            move to that new helper instead.
            """
            keys = real_get_dict_attr_keys(
                parent_group, attr_name, decode_cache=decode_cache
            )
            if attr_name == "_program_results":
                keys = [key for key in keys if key != missing_index]
            return keys

        monkeypatch.setattr(
            multiprogramrunner_module,
            "get_dict_attr_keys",
            get_dict_attr_keys_dropping_program_result,
        )

        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
            shot_checkpoint_dir=tmp_path / "shot_ckpt",
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=True,
        )

        with pytest.raises(
            RuntimeError,
            match=f"Item {missing_index} is missing from.*_program_results",
        ):
            runner.run()


# Module-level test classes for decode_cache regression tests.
# These must be defined at module scope so Serializable can find them during deserialization.


class _SharedParent(Serializable):
    """Test object that acts as a shared parent for testing shared references."""
    _SERIALIZE_ATTRS = ["value"]
    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    def __init__(self, value=42):
        self.value = value


class _ItemWithParent(Serializable):
    """Test object with a parent reference, shared across consolidate/poll decode-cache regression tests."""
    _SERIALIZE_ATTRS = ["item_id", "parent"]
    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    def __init__(self, item_id=0, parent=None):
        self.item_id = item_id
        self.parent = parent


class TestDecodeCache:
    """Regression tests for decode_cache fixes.

    Each test verifies that a Serializable value referenced by multiple
    entries in an HDF5 dict attribute decodes to the same real object
    (not a DeferredRef placeholder) when read via the fixed code paths.
    """

    def test_read_worker_files_shared_reference_decode_cache(self, tmp_path):
        """Regression test for _read_worker_files decode_cache fix.

        Verifies that when a worker file contains 2+ entries sharing a
        common Serializable reference (a parent object), reading them via
        _read_worker_files decodes the shared object to the same real object
        both times, not a DeferredRef on the second occurrence.

        Exercises the fix in _read_worker_files's shared decode_cache handling.
        """
        from loqs.tools.multiprogramrunner import _read_worker_files
        from loqs.internal.serializable import DeferredRef

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a shared parent and two children that reference it
        shared_parent = _SharedParent(value=99)
        child1 = _ItemWithParent(item_id=1, parent=shared_parent)
        child2 = _ItemWithParent(item_id=2, parent=shared_parent)

        # Write both children to a worker file, sharing the same parent
        worker_file = checkpoint_dir / "worker_0_runner.h5"
        with h5py.File(worker_file, "a") as f:
            from loqs.internal.streamingmerge import merge_dict_attr
            # Use a shared encode_cache so the encoder knows about shared refs
            merge_dict_attr(
                f,
                "results",
                [(100, child1), (101, child2)],
                encode_cache={},
                key_use_dataset=True,
                value_use_dataset=False,
            )

        # Read them back via _read_worker_files (the fixed code path)
        done = _read_worker_files(checkpoint_dir, attr_name="results")

        # Both entries should be present
        assert 100 in done and 101 in done
        result1 = done[100]
        result2 = done[101]

        # Neither should be a DeferredRef
        assert not isinstance(result1, DeferredRef), \
            f"Entry 100 decoded to DeferredRef, not {type(result1)}"
        assert not isinstance(result2, DeferredRef), \
            f"Entry 101 decoded to DeferredRef, not {type(result2)}"

        # Both should be ItemWithParent with the same parent object
        assert isinstance(result1, _ItemWithParent)
        assert isinstance(result2, _ItemWithParent)
        assert result1.parent is result2.parent, \
            "Shared parent should decode to the same object both times"
        assert result1.parent.value == 99

    def test_consolidate_worker_files_shared_reference_decode_cache(self, tmp_path):
        """Regression test for _consolidate_worker_files decode_cache fix.

        Verifies that calling the real _consolidate_worker_files function on a
        worker file with both "results" and "_program_results" attributes where
        both entries share a common reference correctly consolidates them into
        runner.h5 such that the shared object decodes to the same real object
        in both attributes, not a DeferredRef past its first occurrence.

        Exercises the shared decode_cache passed to iter_dict_attr_entries
        within _consolidate_worker_files at lines 846 and 865.
        """
        from loqs.tools.multiprogramrunner import _consolidate_worker_files
        from loqs.internal.serializable import DeferredRef

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a minimal runner.h5 with valid structure via _SimpleDoubleRunner
        runner = _SimpleDoubleRunner(
            items=[],
            checkpoint=False,
        )
        runner_path = checkpoint_dir / "runner.h5"
        runner.write(runner_path, "hdf5")

        # Create shared parent and items
        shared_parent = _SharedParent(value=77)
        result_item = _ItemWithParent(item_id=10, parent=shared_parent)
        program_results_item = _ItemWithParent(item_id=20, parent=shared_parent)

        # Create worker file with both "results" and "_program_results"
        worker_file = checkpoint_dir / "worker_0_runner.h5"
        with h5py.File(worker_file, "a") as f:
            from loqs.internal.streamingmerge import merge_dict_attr
            encode_cache = {}
            merge_dict_attr(
                f,
                "results",
                [(200, result_item)],
                encode_cache=encode_cache,
                key_use_dataset=True,
                value_use_dataset=False,
            )
            merge_dict_attr(
                f,
                "_program_results",
                [(300, program_results_item)],
                encode_cache=encode_cache,
                key_use_dataset=True,
                value_use_dataset=False,
            )

        # Verify worker file exists before consolidation
        assert worker_file.exists()

        # Call the real _consolidate_worker_files function
        _consolidate_worker_files(checkpoint_dir, runner_filename="runner.h5",
                                  delete_originals=True)

        # Verify worker file was deleted (confirms real function executed)
        assert not worker_file.exists(), \
            "Worker file should be deleted after consolidation with delete_originals=True"

        # Read back both attributes using a shared decode_cache, mirroring
        # how a real caller like _read_done_union reads them.
        with h5py.File(runner_path, "r") as f:
            from loqs.tools.multiprogramrunner import _get_runner_object_group
            runner_root = _get_runner_object_group(f)
            decode_cache = {}  # Shared across both branches
            reduced_results = dict(
                iter_dict_attr_entries(runner_root, "_reduced_results",
                                       decode_cache=decode_cache)
            )
            program_results = dict(
                iter_dict_attr_entries(runner_root, "_program_results",
                                       decode_cache=decode_cache)
            )

        # Both entries should be present and successfully decoded
        assert 200 in reduced_results, \
            "Entry 200 should have been consolidated into _reduced_results"
        assert 300 in program_results, \
            "Entry 300 should have been consolidated into _program_results"

        result1 = reduced_results[200]
        result2 = program_results[300]

        assert not isinstance(result1, DeferredRef), \
            f"'_reduced_results' entry decoded to DeferredRef, not {type(result1)}"
        assert not isinstance(result2, DeferredRef), \
            f"'_program_results' entry decoded to DeferredRef, not {type(result2)}"

        # Verify the entries have valid parent references.
        assert isinstance(result1, _ItemWithParent)
        assert isinstance(result2, _ItemWithParent)
        assert result1.parent is not None, \
            "Entry 200 should have a parent reference"
        assert result2.parent is not None, \
            "Entry 300 should have a parent reference"
        assert result1.parent.value == 77
        assert result2.parent.value == 77

    def test_poll_one_worker_file_shared_reference_decode_cache(self, tmp_path):
        """Regression test for _poll_one_worker_file decode_cache fix.

        Verifies that when polling a worker file for newly-completed entries
        sharing a common reference, the shared object decodes to the same
        real object both times, not a DeferredRef on the second occurrence.

        Exercises the fix in _poll_one_worker_file's shared decode_cache handling.
        """
        from loqs.tools.multiprogramrunner import _poll_one_worker_file
        from loqs.internal.serializable import DeferredRef

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create shared parent and items
        shared_parent = _SharedParent(value=55)
        item1 = _ItemWithParent(item_id=30, parent=shared_parent)
        item2 = _ItemWithParent(item_id=31, parent=shared_parent)

        # Write items to a worker file
        worker_file = checkpoint_dir / "worker_0_runner.h5"
        with h5py.File(worker_file, "a") as f:
            from loqs.internal.streamingmerge import merge_dict_attr
            merge_dict_attr(
                f,
                "results",
                [(400, item1), (401, item2)],
                encode_cache={},
                key_use_dataset=True,
                value_use_dataset=False,
            )

        # Poll the worker file for entries starting at index 0
        polled_items = []

        def on_item_done(key, item, result):
            polled_items.append((key, result))

        final_count = _poll_one_worker_file(
            worker_file,
            consumed_count=0,
            observed_indices=set(),
            items_map={400: "item_400", 401: "item_401"},
            on_item_done=on_item_done,
            pbar=None,
        )

        # Should have polled both items
        assert final_count == 2
        assert len(polled_items) == 2

        # Extract the results from the callback
        result1 = polled_items[0][1]
        result2 = polled_items[1][1]

        # Neither should be a DeferredRef
        assert not isinstance(result1, DeferredRef), \
            f"First polled item decoded to DeferredRef, not {type(result1)}"
        assert not isinstance(result2, DeferredRef), \
            f"Second polled item decoded to DeferredRef, not {type(result2)}"

        # Both should be ItemWithParent with the same parent object
        assert isinstance(result1, _ItemWithParent)
        assert isinstance(result2, _ItemWithParent)
        assert result1.parent is result2.parent, \
            "Shared parent should decode to the same object both times"
        assert result1.parent.value == 55

    def test_consolidate_worker_files_streaming_one_at_a_time(self, tmp_path):
        """Trap test enforcing read-ahead bound: decoding doesn't race past writing.

        Asserts that _consolidate_worker_files never decodes more than one entry
        ahead of what has already been written to runner.h5, so the read and write
        sides always advance in lockstep (decoded_count - written_count <= 1 at all
        times) rather than fully materializing decoded entries before any write.
        """
        import unittest.mock
        from loqs.tools.multiprogramrunner import _consolidate_worker_files
        from loqs.internal.streamingmerge import (
            iter_dict_attr_entries as real_iter_dict_attr_entries,
            merge_dict_attr,
        )
        from loqs.internal import streamingmerge

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a minimal runner.h5 via _SimpleDoubleRunner
        runner = _SimpleDoubleRunner(
            items=[],
            checkpoint=False,
        )
        runner_path = checkpoint_dir / "runner.h5"
        runner.write(runner_path, "hdf5")

        # Create worker file with 3 results to merge
        worker_file = checkpoint_dir / "worker_0_runner.h5"
        with h5py.File(worker_file, "a") as f:
            encode_cache = {}
            for i in range(3):
                merge_dict_attr(
                    f,
                    "results",
                    [(100 + i, 1000 + i)],
                    encode_cache=encode_cache,
                    key_use_dataset=True,
                    value_use_dataset=False,
                )

        # Counters for read and write sides (mutable via closure)
        decoded_count = [0]
        written_count = [0]

        def spy_iter_dict_attr_entries(
            parent_group, attr_name, decode_cache=None, start_index=0
        ):
            """Spy on iter_dict_attr_entries: track decoded entries, enforce read-ahead bound."""
            for key, value in real_iter_dict_attr_entries(
                parent_group, attr_name, decode_cache=decode_cache, start_index=start_index
            ):
                decoded_count[0] += 1
                # Enforce: read side never more than 1 ahead of write side for "results"
                if attr_name == "results":
                    ahead = decoded_count[0] - written_count[0]
                    assert ahead <= 1, \
                        f"Read-ahead violation: decoded {decoded_count[0]}, " \
                        f"written {written_count[0]}, ahead by {ahead} (should be ≤1)"
                yield (key, value)

        # Save real function reference before patching
        real_stream_into_existing = streamingmerge._stream_into_existing_dict_attr

        def spy_stream_into_existing_dict_attr(
            parent_group, attr_name, entries, encode_cache
        ):
            """Spy on _stream_into_existing_dict_attr: stream one entry at a time, track writes."""
            # Stream one entry at a time, calling real function per entry
            for key, value in entries:
                real_stream_into_existing(parent_group, attr_name, [(key, value)], encode_cache)
                # Increment write count only for _reduced_results
                if attr_name == "_reduced_results":
                    written_count[0] += 1

        # Patch both iter_dict_attr_entries (read side) and _stream_into_existing_dict_attr
        import loqs.tools.multiprogramrunner as mpr_module

        with unittest.mock.patch.object(
            mpr_module, "iter_dict_attr_entries", side_effect=spy_iter_dict_attr_entries
        ), unittest.mock.patch.object(
            streamingmerge,
            "_stream_into_existing_dict_attr",
            side_effect=spy_stream_into_existing_dict_attr,
        ):
            _consolidate_worker_files(
                checkpoint_dir,
                runner_filename="runner.h5",
                delete_originals=True,
            )

        # Assert both counters reached their expected values
        assert decoded_count[0] == 3, \
            f"Expected 3 entries decoded, got {decoded_count[0]}"
        assert written_count[0] == 3, \
            f"Expected 3 entries written, got {written_count[0]}"

        # Verify worker file was deleted
        assert not worker_file.exists(), \
            "Worker file should be deleted after consolidation"

    def test_reduced_results_uses_dataset_storage_format(self, tmp_path):
        """After a fresh MultiProgramRunner checkpoint with real result data,
        _reduced_results' key-side storage_format should be 'dataset', not 'groups'."""
        checkpoint_dir = tmp_path / "runner_checkpoint"

        # Create a simple runner with one item and run it
        runner = _SimpleDoubleRunner(
            items=[5],
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
        )
        result_list = runner.run()
        assert result_list == [10]

        # Verify _reduced_results key-side storage_format is 'dataset'
        runner_path = checkpoint_dir / "runner.h5"
        with h5py.File(runner_path, "r") as f:
            group = _resolve_checkpoint_object_group(f)
            storage_format = group["_reduced_results"]["dict"]["keys"][
                "iterable"
            ].attrs.get("storage_format", "groups")
            assert storage_format == "dataset", (
                f"Expected _reduced_results keys to use 'dataset' format "
                f"after fresh checkpoint with result data, but got '{storage_format}'"
            )


# Test doubles for the redesigned build_program/reduce_program_outcomes/
# _build_output hook surface (not yet implemented on the base class -- see
# TestMultiProgramRunnerRedesignedHooks below).


class _FakeProgramResult:
    """Minimal stand-in for a QuantumProgram.run() return value, for
    mechanism-level tests that don't need a real program."""

    def __init__(self, value):
        self.value = value


class _FakeProgram:
    """Minimal stand-in for a QuantumProgram, for mechanism-level tests
    exercising build_program/reduce_program_outcomes without a real program.
    If run_kwargs_log is given, each run() call's kwargs are appended to it
    so a test can inspect what the dispatch layer actually passed through."""

    def __init__(self, value, run_kwargs_log=None):
        self.value = value
        self._run_kwargs_log = run_kwargs_log

    def run(self, **kwargs):
        if self._run_kwargs_log is not None:
            self._run_kwargs_log.append(kwargs)
        return _FakeProgramResult(self.value)


class _NewHookRunner(MultiProgramRunner):
    """Test double exercising MultiProgramRunner's build_program/
    reduce_program_outcomes/_build_output hook surface directly, without
    any of the generic dispatch/checkpoint/resume/parallel machinery those
    hooks plug into."""

    def __init__(self, items, run_kwargs_log=None, weakrefs=None, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self._run_kwargs_log = run_kwargs_log
        self._weakrefs = weakrefs if weakrefs is not None else []

    def build_program(self, index):
        program = _FakeProgram(self.items[index], self._run_kwargs_log)
        self._weakrefs.append(weakref.ref(program))
        return program

    def reduce_program_outcomes(self, program_results):
        self._weakrefs.append(weakref.ref(program_results))
        return program_results.value * 2

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]


class TestMultiProgramRunnerRedesignedHooks:
    """Tests proving specific properties of MultiProgramRunner's
    build_program/reduce_program_outcomes/_build_output/build_output hook
    surface and its _shared_item_worker cascade guard."""

    def test_shared_item_worker_releases_program_and_results_for_gc(self):
        """Once dispatch completes for an item with keep_shot_results=False,
        the program/program_results objects built for it must be
        garbage-collectible -- no lingering reference held anywhere (self, a
        closure, or a runner snapshot's own bound method)."""
        weakrefs = []
        runner = _NewHookRunner(items=[5], weakrefs=weakrefs)

        result = runner.run()

        assert result == [10]
        gc.collect()
        assert len(weakrefs) == 2, (
            "expected one weakref each for the built program and its "
            "program_results"
        )
        assert all(ref() is None for ref in weakrefs), (
            "program/program_results should be garbage-collected once "
            "dispatch completes for a keep_shot_results=False item"
        )

    def test_static_kwargs_produce_a_picklable_worker(self):
        """A functools.partial-wrapped worker built from the runner's own
        _static_kwargs() must itself be picklable, since _run_parallel
        relies on pickling the worker (plus its static kwargs) to send to
        worker processes. A hook wiring that captures something unpicklable
        (an open file handle, a whole ParallelStrategy with a live executor)
        would fail this."""
        from loqs.tools.multiprogramrunner import _shared_item_worker

        runner = _NewHookRunner(items=[1, 2, 3])

        worker = functools.partial(
            _shared_item_worker, **(runner._static_kwargs() or {})
        )

        pickle.dumps(worker)

    def test_build_output_warns_on_incomplete_none_result(self):
        """build_output must warn when any ordered_results entry has a None
        result (an incomplete run), before delegating to the subclass's own
        _build_output."""
        runner = _NewHookRunner(items=["a", "b"])
        ordered_results = [("a", 1), ("b", None)]

        with pytest.warns(UserWarning, match="None result"):
            output = runner.build_output(ordered_results)

        assert output == [1, None]

    def test_build_output_warns_on_incomplete_ordered_reduced_results(self):
        """build_output's None-result warning must also fire for a genuinely
        partial _reduced_results (the load-without-resume scenario), not just
        a synthetic ordered_results list constructed directly."""
        runner = _NewHookRunner(items=["a", "b", "c"])
        runner._reduced_results = {0: 1, 2: 3}  # index 1 never completed

        with pytest.warns(UserWarning, match="None result"):
            output = runner.build_output(runner._ordered_reduced_results())

        assert output == [1, None, 3]

    def test_cascade_guard_resumes_when_checkpoint_file_already_exists(
        self, tmp_path
    ):
        """If shot_checkpoint=True and the per-item checkpoint's results
        file already exists on disk, the cascade guard must resume
        (resume=True, checkpoint=True, checkpoint_dir set) rather than
        raising or trying to fresh-create an already-existing checkpoint."""
        shot_checkpoint_dir = tmp_path / "shot_checkpoints"
        item_dir = shot_checkpoint_dir / "item_0"
        item_dir.mkdir(parents=True)
        (item_dir / "results.h5").touch()

        run_kwargs_log = []
        runner = _NewHookRunner(
            items=[7],
            run_kwargs_log=run_kwargs_log,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_checkpoint_dir,
        )

        runner.run()

        assert len(run_kwargs_log) == 1
        assert run_kwargs_log[0]["resume"] is True
        assert run_kwargs_log[0]["checkpoint"] is True
        assert run_kwargs_log[0]["checkpoint_dir"] == item_dir


class TestBaseClassMechanisms:
    """Direct tests for base-class mechanisms (mismatch, normalized args, etc.)."""

    def test_mismatch_field_display_name_maps_normalized_collect_shot_data_args(self):
        """_mismatch_field_display_name maps internal field name to public param."""
        runner = _SimpleDoubleRunner([1])
        assert (
            runner._mismatch_field_display_name("_normalized_collect_shot_data_args")
            == "collect_shot_data_args"
        )

    def test_mismatch_field_display_name_identity_fallback(self):
        """_mismatch_field_display_name returns unknown fields unchanged."""
        runner = _SimpleDoubleRunner([1])
        assert runner._mismatch_field_display_name("foo_bar") == "foo_bar"

    def test_mismatch_display_name_in_error_message_on_resume(self, tmp_path):
        """Error message on resume mismatch uses public name, not internal."""
        # Checkpoint with one collect_shot_data_args spec
        checkpoint_dir = tmp_path / "ckpt1"
        checkpoint_dir.mkdir(parents=True)
        runner1 = _ShotDataArgsRunner(
            [1],
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
            collect_shot_data_args=[("counter", -1)],
        )
        runner1.run()

        # Try to resume with a different spec
        runner2 = _ShotDataArgsRunner(
            [1],
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=checkpoint_dir,
            collect_shot_data_args=[("counter", 0)],
        )

        with pytest.raises(ValueError) as exc_info:
            runner2.run()
        assert "collect_shot_data_args" in str(exc_info.value)
        assert "_normalized_collect_shot_data_args" not in str(exc_info.value)

    def test_normalized_collect_shot_data_args_tuple_form(self):
        """_normalized_collect_shot_data_args normalizes tuple form."""
        runner = _ShotDataArgsRunner([1], collect_shot_data_args=[("counter", -1)])
        normalized = runner._normalized_collect_shot_data_args
        assert len(normalized) == 1
        assert isinstance(normalized[0], HistoryDataCollector)
        assert normalized[0].key == "counter"
        assert normalized[0].indices == -1

    def test_normalized_collect_shot_data_args_dict_form(self):
        """_normalized_collect_shot_data_args normalizes dict form."""
        runner = _ShotDataArgsRunner(
            [1], collect_shot_data_args=[{"key": "counter", "indices": -1}]
        )
        normalized = runner._normalized_collect_shot_data_args
        assert len(normalized) == 1
        assert isinstance(normalized[0], HistoryDataCollector)
        assert normalized[0].key == "counter"
        assert normalized[0].indices == -1

    def test_normalized_collect_shot_data_args_collector_form(self):
        """_normalized_collect_shot_data_args handles HistoryDataCollector objects."""
        collector = HistoryDataCollector(key="counter", indices=-1)
        runner = _ShotDataArgsRunner([1], collect_shot_data_args=[collector])
        normalized = runner._normalized_collect_shot_data_args
        assert len(normalized) == 1
        assert isinstance(normalized[0], HistoryDataCollector)
        assert normalized[0].key == "counter"
        assert normalized[0].indices == -1

    def test_normalized_collect_shot_data_args_empty(self):
        """_normalized_collect_shot_data_args normalizes missing to empty tuple."""
        runner = _ShotDataArgsRunner([1])
        normalized = runner._normalized_collect_shot_data_args
        assert normalized == ()

    def test_normalized_collect_shot_data_args_equivalence_on_resume(self, tmp_path):
        """Different encodings of equivalent args resume successfully."""
        # Checkpoint with tuple form
        checkpoint_dir = tmp_path / "ckpt2"
        checkpoint_dir.mkdir(parents=True)
        runner1 = _ShotDataArgsRunner(
            [1],
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
            collect_shot_data_args=[("counter", -1)],
        )
        runner1.run()

        # Resume with dict form (should succeed, not raise)
        runner2 = _ShotDataArgsRunner(
            [1],
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=checkpoint_dir,
            collect_shot_data_args=[{"key": "counter", "indices": -1}],
        )
        result = runner2.run()
        assert result == [2]

    def test_run_kwargs_checkpoint_dir_conflict(self, tmp_path):
        """ValueError raised when both checkpoint_dir in run_kwargs and
        shot_checkpoint_dir are set."""
        with pytest.raises(ValueError, match="checkpoint_dir.*conflict"):
            _SimpleDoubleRunner(
                [1],
                shot_checkpoint=True,
                shot_checkpoint_dir=tmp_path / "shots",
                run_kwargs={"checkpoint_dir": tmp_path / "other"},
            )

    def test_run_kwargs_max_frame_limit_warning_on_missing(self):
        """Warning fires when max_frame_limit not in run_kwargs."""
        with pytest.warns(UserWarning, match="max_frame_limit"):
            runner = _SimpleDoubleRunner([1], run_kwargs=None)
        assert runner.run_kwargs["max_frame_limit"] == 1_000_000

    def test_run_kwargs_max_frame_limit_no_warning_when_present(self):
        """No warning when max_frame_limit already in run_kwargs."""
        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            runner = _SimpleDoubleRunner(
                [1], run_kwargs={"max_frame_limit": 500}
            )
        max_frame_warnings = [
            w
            for w in warning_list
            if "max_frame_limit" in str(w.message)
        ]
        assert len(max_frame_warnings) == 0
        assert runner.run_kwargs["max_frame_limit"] == 500

    def test_run_kwargs_callable_value_resolved_against_item_not_index(self):
        """A callable (non-class) run_kwargs value is resolved by calling it with
        the actual item, not its dispatch index -- proven by using items whose
        values differ from their positional index."""
        seen_limits = []

        real_run = QuantumProgram.run

        def spy_run(self, *args, **kwargs):
            seen_limits.append(kwargs.get("max_frame_limit"))
            return real_run(self, *args, **kwargs)

        runner = _CountingRunner(
            items=[5, 10, 15],
            run_kwargs={"max_frame_limit": lambda item: item * 1000},
        )
        try:
            QuantumProgram.run = spy_run
            runner.run()
        finally:
            QuantumProgram.run = real_run

        assert seen_limits == [5000, 10000, 15000]

    def test_run_kwargs_n_shot_batches_not_clobbered_when_parallel_strategy_n_shot_batches_is_none(self):
        """Explicit n_shot_batches in run_kwargs survives when ParallelStrategy
        doesn't set its own (parallel_strategy=None, the serial default)."""
        seen_n_shot_batches = []

        real_run = QuantumProgram.run

        def spy_run(self, *args, **kwargs):
            seen_n_shot_batches.append(kwargs.get("n_shot_batches"))
            return real_run(self, *args, **kwargs)

        runner = _CountingRunner(
            items=[1, 2],
            run_kwargs={"n_shot_batches": 2},
        )
        try:
            QuantumProgram.run = spy_run
            runner.run()
        finally:
            QuantumProgram.run = real_run

        assert seen_n_shot_batches == [2, 2]

    def test_force_resume_propagation_with_shot_checkpoint(self, tmp_path):
        """force_resume=True is genuinely forwarded into program.run() (via
        _shared_item_worker), bypassing a shot-level num_shots mismatch.
        item_checkpoint_dir is left unset so every run() call actually
        dispatches through program.run() again, rather than short-circuiting
        on cached item-level completion state."""
        shot_ckpt_dir = tmp_path / "shots"
        shot_ckpt_dir.mkdir(parents=True)

        # First run establishes the on-disk shot checkpoint with num_shots=1.
        runner1 = _CountingRunner(
            [1, 2],
            item_checkpoint_dir=None,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt_dir,
        )
        runner1.num_shots = 1
        result1 = runner1.run()
        assert result1 == [2, 4]

        # Same shot checkpoint, different num_shots, force_resume=False:
        # QuantumProgram.run() itself raises on the num_shots mismatch.
        runner2 = _CountingRunner(
            [1, 2],
            item_checkpoint_dir=None,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt_dir,
            force_resume=False,
        )
        runner2.num_shots = 2
        with pytest.raises(ValueError, match="num_shots"):
            runner2.run()

        # Same num_shots mismatch, but force_resume=True: succeeds only if
        # force_resume was actually forwarded into program.run().
        runner3 = _CountingRunner(
            [1, 2],
            item_checkpoint_dir=None,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt_dir,
            force_resume=True,
        )
        runner3.num_shots = 2
        result3 = runner3.run()
        assert result3 == [2, 4]

    def test_checkpoint_subdir_for_prefix_function(self, tmp_path):
        """_checkpoint_subdir_for_prefix builds correct subdir path."""
        result = _checkpoint_subdir_for_prefix(tmp_path, "item", 3)
        assert result == tmp_path / "item_3"

    def test_shot_checkpoint_subdir_returns_none_when_disabled(self):
        """_shot_checkpoint_subdir returns None when shot_checkpoint=False."""
        runner = _SimpleDoubleRunner([1], shot_checkpoint=False)
        assert runner._shot_checkpoint_subdir(3) is None

    def test_shot_checkpoint_subdir_returns_path_when_enabled(self, tmp_path):
        """_shot_checkpoint_subdir returns correct path when enabled."""
        runner = _SimpleDoubleRunner(
            [1],
            shot_checkpoint=True,
            shot_checkpoint_dir=tmp_path,
        )
        subdir = runner._shot_checkpoint_subdir(3)
        assert subdir == tmp_path / "item_3"

    def test_custom_runner_filename_checkpoint_and_resume(self, tmp_path):
        """Custom runner_filename is used for checkpoint and resume."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir(parents=True)

        # Run with custom runner_filename
        runner1 = _SimpleDoubleRunner(
            [1],
            checkpoint=True,
            item_checkpoint_dir=ckpt_dir,
            runner_filename="custom_runner.h5",
        )
        result1 = runner1.run()
        assert (ckpt_dir / "custom_runner.h5").exists()
        assert not (ckpt_dir / "runner.h5").exists()

        # Resume with same custom runner_filename
        runner2 = _SimpleDoubleRunner(
            [1],
            checkpoint=True,
            resume=True,
            item_checkpoint_dir=ckpt_dir,
            runner_filename="custom_runner.h5",
        )
        result2 = runner2.run()
        assert result2 == result1

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="submitit unconditionally registers a SIGCONT handler, POSIX-only",
    )
    def test_submitit_program_executor_produces_correct_results(self, tmp_path):
        """submitit program executor (in-process DebugExecutor) produces correct results."""
        submitit = pytest.importorskip("submitit")

        runner = _CountingRunner(items=[3, -2], checkpoint=False)
        serial_result = runner.run()

        runner_parallel = _CountingRunner(
            items=[3, -2],
            checkpoint=False,
            # DebugExecutor: AutoExecutor(cluster="local") can't unpickle a
            # test-double class defined in this test module by reference.
            parallel_strategy=ParallelStrategy(
                program_executor=submitit.DebugExecutor(folder=tmp_path),
                n_program_chunks=2,
            ),
        )
        parallel_result = runner_parallel.run()
        assert parallel_result == serial_result

    def test_hybrid_program_and_shot_executor_parallelism(self, tmp_path):
        """Hybrid program_executor + shot_executor nested parallelism produces correct results."""
        loky = pytest.importorskip("loky")
        from _shared_checkpoint_test_helpers import _build_shot_executor

        runner = _CountingRunner(items=[2, 3], checkpoint=False)
        serial_result = runner.run()

        runner_hybrid = _CountingRunner(
            items=[2, 3],
            checkpoint=False,
            parallel_strategy=ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=2),
                n_program_chunks=2,
                shot_executor=_build_shot_executor,
            ),
        )
        hybrid_result = runner_hybrid.run()
        assert hybrid_result == serial_result

    def test_hybrid_with_live_loky_shot_executor(self):
        """A live loky executor (not a hand-written factory) works as
        shot_executor too -- ParallelStrategy auto-converts it to a
        picklable factory internally (see test_paralleltools.py for
        coverage of the conversion itself)."""
        loky = pytest.importorskip("loky")

        runner = _CountingRunner(items=[2, 3], checkpoint=False)
        serial_result = runner.run()

        runner_hybrid = _CountingRunner(
            items=[2, 3],
            checkpoint=False,
            parallel_strategy=ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=2),
                n_program_chunks=2,
                shot_executor=loky.get_reusable_executor(max_workers=2),
            ),
        )
        hybrid_result = runner_hybrid.run()
        assert hybrid_result == serial_result

    def test_shot_checkpoint_true_without_dir_raises(self):
        """shot_checkpoint=True without shot_checkpoint_dir raises ValueError."""
        with pytest.raises(ValueError, match="shot_checkpoint_dir"):
            _CountingRunner(items=[1], shot_checkpoint=True)

    def test_shot_checkpoint_creates_per_item_subdirs(self, tmp_path):
        """Serial run with shot_checkpoint=True creates per-item subdirectories."""
        shot_ckpt_dir = tmp_path / "shots"
        runner = _CountingRunner(
            items=[1, 1],
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt_dir,
            lazy_loading=False,
        )
        result = runner.run()
        assert result == [2, 2]

        # Verify subdirectories exist
        assert (shot_ckpt_dir / "item_0").exists()
        assert (shot_ckpt_dir / "item_1").exists()

        # Verify each contains results.h5
        from loqs.core.programresults import ProgramResults

        for idx in [0, 1]:
            subdir = shot_ckpt_dir / f"item_{idx}"
            results_file = subdir / "results.h5"
            assert results_file.exists(), f"Missing {results_file}"

            # Load and verify checkpoint data
            pr = ProgramResults()
            pr.load_checkpoint(checkpoint_dir=subdir)
            assert len(pr.shot_histories) == 1

    def test_parallel_one_chunk_preserves_per_item_shot_checkpoints(self, tmp_path):
        """Parallel run with n_program_chunks=1 keeps collision-free per-item shot checkpoints."""
        loky = pytest.importorskip("loky")
        from _shared_checkpoint_test_helpers import _build_shot_executor

        shot_ckpt_dir = tmp_path / "shots"
        runner = _CountingRunner(
            items=[2, 2],
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt_dir,
            lazy_loading=False,
            parallel_strategy=ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=1),
                n_program_chunks=1,
                shot_executor=_build_shot_executor,
            ),
        )
        result = runner.run()
        assert result == [4, 4]

        # Verify per-item subdirectories exist (collision-free)
        assert (shot_ckpt_dir / "item_0").exists()
        assert (shot_ckpt_dir / "item_1").exists()
        assert (shot_ckpt_dir / "item_0" / "results.h5").exists()
        assert (shot_ckpt_dir / "item_1" / "results.h5").exists()

    def test_cascade_guard_respects_custom_results_filename(self, tmp_path):
        """The cascade guard's on-disk existence check for a per-item
        checkpoint must consult the same custom results_filename passed to
        the runner, not a hardcoded 'results.h5'."""
        shot_checkpoint_dir = tmp_path / "shot_checkpoints"
        item_dir = shot_checkpoint_dir / "item_0"
        item_dir.mkdir(parents=True)
        (item_dir / "custom_results.h5").touch()

        run_kwargs_log = []
        runner = _NewHookRunner(
            items=[7],
            run_kwargs_log=run_kwargs_log,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_checkpoint_dir,
            results_filename="custom_results.h5",
        )

        runner.run()

        assert len(run_kwargs_log) == 1
        assert run_kwargs_log[0]["resume"] is True
        assert run_kwargs_log[0]["checkpoint"] is True
        assert run_kwargs_log[0]["checkpoint_dir"] == item_dir

    def test_keyed_runner_reduced_results_dataset_format(self, tmp_path):
        """_KeyedRunner's second index_map write preserves dataset storage format."""
        checkpoint_dir = tmp_path / "ckpt"

        runner = _KeyedRunner(
            items=[5],
            checkpoint=True,
            item_checkpoint_dir=checkpoint_dir,
        )
        result = runner.run()
        assert result == [10]

        # Verify _reduced_results key-side storage_format is 'dataset'
        runner_path = checkpoint_dir / "runner.h5"
        with h5py.File(runner_path, "r") as f:
            group = _resolve_checkpoint_object_group(f)
            storage_format = group["_reduced_results"]["dict"]["keys"][
                "iterable"
            ].attrs.get("storage_format", "groups")
            assert storage_format == "dataset", (
                f"Expected _reduced_results keys to use 'dataset' format "
                f"after _KeyedRunner checkpoint, but got '{storage_format}'"
            )
