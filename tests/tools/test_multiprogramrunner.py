"""Tester for loqs.tools.multiprogramrunner"""

import contextlib
import functools
import gc
import h5py
import pickle
import sys
import time
import unittest.mock
import weakref
from pathlib import Path
from typing import Any, ClassVar

import pytest
from _shared_checkpoint_test_helpers import (
    InlineExecutor,
    build_program_interrupts_at,
    build_sweep_runner,
    inline_execution_patches,
    open_hdf5_file_paths,
    record_h5_writes,
    sessions_in,
    spy_on_renames,
)
from tqdm import tqdm as orig_tqdm

from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.core import QuantumProgram
from loqs.core.programresults import ProgramResults
from loqs.internal import worker_id
from loqs.internal.checkpointstore import (
    ITEM_LAYOUT,
    CheckpointReadError,
    CheckpointStore,
)
from loqs.internal.serializable import Serializable
from loqs.tools.noisesweeptools import NoiseSweepRunner
from loqs.tools.paralleltools import ExecutorSpec, ParallelStrategy
from loqs.tools.multiprogramrunner import (
    CheckpointConfig,
    MultiProgramRunner,
)


@pytest.fixture
def inline():
    """Run in-process parallel dispatch cheaply: see `inline_execution_patches`."""
    with inline_execution_patches():
        yield


def _loky_strategy(max_workers=2, n_program_chunks=2, **kwargs):
    """A `ParallelStrategy` dispatching programs over a loky pool."""
    loky = pytest.importorskip("loky")
    return ParallelStrategy(
        program_executor=loky.get_reusable_executor(max_workers=max_workers),
        n_program_chunks=n_program_chunks,
        **kwargs,
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


def _build_counter_program(
    num_increments, increment_by, name="Trivial counter test program"
):
    """Build a QuantumProgram via codepack_trivial_counter: an Init Counter
    (starting at 0) followed by num_increments Increment instructions, each
    adding increment_by -- final counter value is num_increments * increment_by.
    """
    trivial_code = trivial_codepack.create_qec_code()
    ideal_model = trivial_codepack.create_ideal_model(["Q0"])
    stack = [
        {
            "instruction": "Init Patch Trivial",
            "new_patch_label": "L0",
            "qubits": ["Q0"],
        },
        {
            "instruction": "Init Counter",
            "patch_label": "L0",
            "initial_value": 0,
        },
    ]
    for _ in range(num_increments):
        stack.append(
            {
                "instruction": "Increment",
                "patch_label": "L0",
                "increment_by": increment_by,
            }
        )
    return QuantumProgram(
        stack,
        default_noise_model=ideal_model,
        patch_types={"Trivial": trivial_code},
        name=name,
    )


def _assert_wall_clock_times(runner, num_items, num_shots=None):
    """Assert `runner` holds one positive float item wall-clock time and one
    non-empty per-shot dict of positive floats for each of `num_items`
    items, each keyed by every shot index if `num_shots` is given."""
    assert set(runner.item_wall_clock_times.keys()) == set(range(num_items))
    for value in runner.item_wall_clock_times.values():
        assert isinstance(value, float)
        assert value > 0

    assert set(runner.shot_wall_clock_times.keys()) == set(range(num_items))
    for shot_times in runner.shot_wall_clock_times.values():
        assert isinstance(shot_times, dict)
        assert len(shot_times) > 0
        if num_shots is not None:
            assert set(shot_times.keys()) == set(range(num_shots))
        for value in shot_times.values():
            assert isinstance(value, float)
            assert value > 0


def _make_tqdm_spy(on_init=None, on_set_n=None):
    """Build a `tqdm` stand-in wrapping the real one, reporting each
    construction as `on_init(tqdm_obj, desc)` and each assignment to `.n`
    as `on_set_n(desc, value)`."""

    class TqdmSpy:
        def __init__(self, *args, **kwargs):
            self.tqdm_obj = orig_tqdm(*args, **kwargs)
            self._desc = kwargs.get("desc", "")
            if on_init is not None:
                on_init(self.tqdm_obj, self._desc)

        def __getattr__(self, name):
            return getattr(self.tqdm_obj, name)

        def __setattr__(self, name, value):
            if name in ("tqdm_obj", "_desc"):
                super().__setattr__(name, value)
            else:
                setattr(self.tqdm_obj, name, value)
                if name == "n" and on_set_n is not None:
                    on_set_n(self._desc, value)

        def refresh(self):
            return self.tqdm_obj.refresh()

        def close(self):
            return self.tqdm_obj.close()

    return TqdmSpy


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
        return _build_counter_program(
            num_increments=abs(item),
            increment_by=2 * sign,
            name=f"Item {index}",
        )

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]


class _TrackingDoubleRunner(_SimpleDoubleRunner):
    """Simple runner that tracks which indices are built.

    The tracked_indices attribute is a plain list, shared by in-process
    (Inline) runs. This attribute is NOT serialized, so it must be set anew
    on the resumed runner instance.
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

    def __init__(
        self,
        items,
        process_fn=_double_item,
        max_count=999,
        raise_after=None,
        call_count=None,
        track_calls=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.items = items
        # Support legacy process_fn parameter: map _raise_after_n to raise_after behavior
        # (raise_after may be passed directly during deserialization)
        if raise_after is not None:
            self.raise_after = raise_after
        else:
            self.raise_after = (
                None if process_fn != _raise_after_n else max_count
            )
        # Determine if we should track calls (old _count_and_double behavior)
        if track_calls is not None:
            self.track_calls = track_calls
        else:
            self.track_calls = process_fn == _count_and_double
        # call_count may be passed during deserialization
        self.call_count = call_count if call_count is not None else [0]
        self.on_item_done_calls = []

    def build_program(self, index):
        # Increment call count (either for tracking calls or for crash simulation)
        self.call_count[0] += 1

        # Check for crash simulation on this item
        if self.raise_after is not None:
            if self.call_count[0] > self.raise_after:
                raise RuntimeError(
                    f"Simulated crash after {self.raise_after} items"
                )

        item = self.items[index]
        sign = 1 if item >= 0 else -1
        return _build_counter_program(
            num_increments=abs(item), increment_by=2 * sign
        )

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

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "items",
        "sleep_time",
    ]

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
            {
                "instruction": "Init Patch Trivial",
                "new_patch_label": "L0",
                "qubits": ["Q0"],
            },
            {
                "instruction": "Init Counter",
                "patch_label": "L0",
                "initial_value": 0,
            },
            {
                "instruction": "Sleep",
                "patch_label": "L0",
                "duration": self.sleep_time,
            },
        ]
        for _ in range(abs(item)):
            stack.append(
                {
                    "instruction": "Increment",
                    "patch_label": "L0",
                    "increment_by": 2 * sign,
                }
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


def _crash_after_three_items(checkpoint_dir, items):
    """Run a serial `_TrackingRunner` that crashes after 3 items."""
    runner = _TrackingRunner(
        items,
        process_fn=_raise_after_n,
        config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        max_count=3,
    )
    with pytest.raises(RuntimeError):
        runner.run()


class TestMultiProgramRunnerSerialWithCheckpoint:
    """Tests for serial execution with checkpointing."""

    def test_serial_crash_simulation_and_resume(self, tmp_path):
        """Simulate a crash and verify resume capability."""
        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(10))

        _crash_after_three_items(checkpoint_dir, items)

        # After a crash, the finished items' payloads stay in the store
        # (consolidation only happens on successful completion).
        store = CheckpointStore(checkpoint_dir, ITEM_LAYOUT)
        assert store.keys() == [0, 1, 2]

        # Second run: resume with normal function on same checkpoint dir
        runner2 = _TrackingRunner(
            items,
            process_fn=_count_and_double,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
        )
        results = runner2.run()

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        # Only 7 items should have been processed (10 - 3 already done)
        assert runner2.call_count[0] == 7
        # on_item_done should have been called for all 10 items (3 replayed + 7 new)
        assert len(runner2.on_item_done_calls) == 10
        # Every item, including the 3 finished before the crash, has timings.
        _assert_wall_clock_times(runner2, len(items))


class TestMultiProgramRunnerParallel:
    """Tests for parallel execution with checkpointing."""

    @pytest.mark.parametrize("ledgers", ["normal", "blank_ledgers"])
    def test_parallel_resume_from_seeded_partial_run(
        self, tmp_path, inline, ledgers
    ):
        """A resumed parallel run rebuilds only the items not already done.
        With `blank_ledgers` the two readers of ledger contents report
        nothing done, as lost or reordered ledger writes on a share without
        POSIX write ordering would; results still come from the payloads."""
        import dataclasses

        import loqs.tools.multiprogramrunner as mpr

        checkpoint_dir = tmp_path / "checkpoints"
        items = list(range(10))
        tracked_indices = []

        runner1 = _TrackingDoubleRunner(
            items,
            tracked_indices=tracked_indices,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        _seed_partial_run(checkpoint_dir, runner1, done_indices=[0, 2, 4])

        runner2 = _TrackingDoubleRunner(
            items,
            tracked_indices=tracked_indices,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
            parallel_strategy=ParallelStrategy(
                program_executor=InlineExecutor(), n_program_chunks=3
            ),
        )

        real_status = mpr.read_swmr_ledger_status
        status_calls = []

        def blank_status(group):
            status_calls.append(1)
            snap = real_status(group)
            done = None if snap.done is None else snap.done & False
            return dataclasses.replace(
                snap, done=done, current_item_index=0, item_shots_done=0
            )

        with contextlib.ExitStack() as stack:
            if ledgers == "blank_ledgers":
                stack.enter_context(
                    unittest.mock.patch.object(
                        mpr, "read_swmr_ledger_status", blank_status
                    )
                )
                stack.enter_context(
                    unittest.mock.patch(
                        "loqs.core.programresults.read_swmr_ledger_done_union",
                        lambda *args, **kwargs: set(),
                    )
                )
            results = runner2.run()

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        assert sorted(tracked_indices) == [1, 3, 5, 6, 7, 8, 9]
        _assert_wall_clock_times(runner2, 10)
        if ledgers == "normal":
            store = CheckpointStore(checkpoint_dir, ITEM_LAYOUT)
            assert store.keys() == list(range(10))
        else:
            assert len(status_calls) >= 1


class TestItemAndShotWallClockTimes:
    """Tests for `MultiProgramRunner.item_wall_clock_times` and
    `.shot_wall_clock_times`, populated during `.run()` from each item's
    own per-item worker-body duration and its `ProgramResults.shot_wall_clock_times`,
    respectively."""

    @pytest.mark.parametrize(
        "checkpoint, parallel",
        [(True, False), (False, True)],
        ids=["serial-checkpoint", "inline_parallel-no_checkpoint"],
    )
    def test_populates_item_and_shot_wall_clock_times(
        self, tmp_path, inline, checkpoint, parallel
    ):
        """A run populates `item_wall_clock_times` with exactly one positive
        float per item index, and `shot_wall_clock_times` with exactly one
        per-item dict per item index, each keyed by every shot index that
        item actually ran -- for the payload store and the in-memory store."""
        items = list(range(4))
        num_shots = 3

        run_kwargs = {"num_shots": num_shots, "max_frame_limit": 1_000_000}
        checkpoint_dir = tmp_path / "checkpoints" if checkpoint else None

        strategy = None
        if parallel:
            strategy = ParallelStrategy(
                program_executor=InlineExecutor(), n_program_chunks=2
            )

        runner = _SleepingRunner(
            items,
            sleep_time=0.02,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
            parallel_strategy=strategy,
            run_kwargs=run_kwargs,
        )
        runner.run()

        _assert_wall_clock_times(runner, len(items), num_shots=num_shots)


# Helper functions


def _seed_partial_run(
    checkpoint_dir: Path, runner: MultiProgramRunner, done_indices: list[int]
):
    """Seed a checkpoint directory as an interrupted run of `runner` leaves
    it: its `runner.h5` plus one store payload per done index, holding a
    result and synthetic `item_wall_clock_times`/`shot_wall_clock_times`
    entries, so a resume test can confirm both timing attributes cover a
    manually-seeded "already done" item, not just the ones the resuming run
    actually processes."""
    store = CheckpointStore(checkpoint_dir, ITEM_LAYOUT)
    store.write_snapshot(runner)
    for index in done_indices:
        store.put_batch(
            {
                index: {
                    "results": index * 2,
                    "item_wall_clock_times": 0.5 + index,
                    "shot_wall_clock_times": {0: 0.1 + index},
                }
            },
            worker_id=worker_id(),
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
        return _build_counter_program(
            num_increments=abs(item), increment_by=2 * sign
        )

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]


class _CustomFilenameProbeRunner(MultiProgramRunner):
    """Mirrors _ShotProgressTestRunner, but item index 1's shots each run a
    real Sleep instruction, so that item stays in flight shot by shot."""

    _SERIALIZE_ATTRS = MultiProgramRunner._SERIALIZE_ATTRS + [
        "items",
        "num_shots",
    ]

    SLOW_SHOT_SECONDS = 0.3

    def __init__(self, items, num_shots=5, on_item_done=None, **kwargs):
        super().__init__(**kwargs)
        self.items = items
        self.num_shots = num_shots
        self._on_item_done = on_item_done

    def build_program(self, index):
        item = self.items[index]
        sign = 1 if item >= 0 else -1
        trivial_code = trivial_codepack.create_qec_code()
        ideal_model = trivial_codepack.create_ideal_model(["Q0"])
        stack = [
            {
                "instruction": "Init Patch Trivial",
                "new_patch_label": "L0",
                "qubits": ["Q0"],
            },
            {
                "instruction": "Init Counter",
                "patch_label": "L0",
                "initial_value": 0,
            },
        ]
        if index == 1:
            stack.append(
                {
                    "instruction": "Sleep",
                    "patch_label": "L0",
                    "duration": self.SLOW_SHOT_SECONDS,
                }
            )
        for _ in range(abs(item)):
            stack.append(
                {
                    "instruction": "Increment",
                    "patch_label": "L0",
                    "increment_by": 2 * sign,
                }
            )
        return QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            patch_types={"Trivial": trivial_code},
            name="Custom filename probe program",
        )

    def reduce_program_outcomes(self, program_results):
        return program_results.collect_shot_data("counter", -1)[0]

    def _build_output(self, ordered_results):
        return [result for _, result in ordered_results]

    def _make_on_item_done(self):
        return self._on_item_done


class TestParallelDispatchAndPollingRegressions:
    """Regression tests for parallel dispatch and polling correctness."""

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
                self.run_kwargs_log = (
                    run_kwargs_log if run_kwargs_log is not None else []
                )

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
            config=CheckpointConfig(item_checkpoint_dir=None),
        )
        runner.run()
        # All calls should receive the sentinel value, not None
        assert [log["shot_executor"] for log in runner.run_kwargs_log] == [
            "SENTINEL_EXECUTOR",
            "SENTINEL_EXECUTOR",
            "SENTINEL_EXECUTOR",
        ]

    def test_parallel_without_checkpoint_dir_returns_results_and_fires_on_item_done(
        self, inline
    ):
        """A parallel run with item_checkpoint_dir=None returns the right
        results (it once crashed, as _run_parallel didn't return its
        dispatch results and final assembly had no source of truth without
        checkpointing) and still invokes on_item_done once per item, via a
        final catch-up pass over any item not already observed through
        checkpoint-directory polling."""
        strategy = ParallelStrategy(
            program_executor=InlineExecutor(), n_program_chunks=2
        )

        runner = _TrackingRunner(
            list(range(4)),
            process_fn=_double_item,
            parallel_strategy=strategy,
            config=CheckpointConfig(item_checkpoint_dir=None),
        )
        results = runner.run()

        assert results == [0, 2, 4, 6]

        # on_item_done should have been called once for each item
        assert len(runner.on_item_done_calls) == 4

        # Verify all expected indices were called (order not guaranteed in parallel)
        called_indices = {
            call_index for call_index, _, _ in runner.on_item_done_calls
        }
        assert called_indices == {0, 1, 2, 3}

        # Verify results are correct for each item
        for call_index, call_item, call_result in runner.on_item_done_calls:
            assert call_item == call_index
            assert call_result == call_index * 2

    def test_poll_skips_item_whose_payload_is_gone(self, tmp_path):
        """A done index whose payload was consolidated away is left for a
        later poll rather than raising."""
        from loqs.internal.swmrledger import (
            mark_ledger_item_done,
            open_swmr_writer,
        )
        from loqs.tools.multiprogramrunner import _poll_one_worker_file

        store = CheckpointStore(tmp_path / "ckpt", ITEM_LAYOUT)
        store.directory.mkdir()
        ledger_file, ledger_group = open_swmr_writer(
            store.ledger_path("w"), fields=["done", "wall_clock_times"]
        )
        try:
            mark_ledger_item_done(ledger_group, 0, 0.5)
            calls: list[Any] = []
            consumed = _poll_one_worker_file(
                store,
                store.ledger_path("w"),
                0,
                set(),
                {0: "item_0"},
                lambda *args: calls.append(args),
                None,
            )
            assert consumed == 0
            assert calls == []
        finally:
            ledger_file.close()


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


def _submitit_debug_executor_factory():
    # Module-level, with its import in the body: the factory is stored as
    # source, so it must not depend on anything but its own body.
    import submitit

    return submitit.DebugExecutor(folder="submitit-logs")


def _thread_executor_factory():
    # Module-level and self-contained, since a factory is stored as source.
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=1)


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
        parallel_strategy=None,
        config=None,
        run_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            parallel_strategy=parallel_strategy,
            config=config,
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

    @pytest.mark.parametrize(
        "checkpoint_field",
        ["item_checkpoint_dir", "shot_checkpoint_dir"],
        ids=["item", "shot"],
    )
    def test_thread_program_executor_refused_with_checkpointing(
        self, tmp_path, checkpoint_field
    ):
        checkpoint_dir = tmp_path / "ckpt"
        runner = _CountingRunner(
            [1, 2, 3],
            multiplier=2,
            config=CheckpointConfig(**{checkpoint_field: checkpoint_dir}),
            parallel_strategy=ParallelStrategy(
                program_executor=_thread_executor_factory,
                n_program_chunks=2,
            ),
        )
        with pytest.raises(ValueError):
            runner.run()
        assert not checkpoint_dir.exists()

    def test_existing_content_without_runner_h5_raises(self, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "unrelated.txt").write_text("not a runner.h5")

        with pytest.raises(FileExistsError):
            _CountingRunner(
                [1, 2, 3],
                multiplier=2,
                config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
            ).run()

    def test_mismatched_config_raises(self, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        _CountingRunner(
            [1, 2, 3],
            multiplier=2,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        ).run()

        mismatched = _CountingRunner(
            [1, 2, 3],
            multiplier=3,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
        )
        with pytest.raises(ValueError):
            mismatched.run()

    def test_force_resume_bypasses_mismatch(self, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        _CountingRunner(
            [1, 2, 3],
            multiplier=2,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        ).run()

        mismatched = _CountingRunner(
            [1, 2, 3],
            multiplier=3,
            config=CheckpointConfig(
                resume=True,
                item_checkpoint_dir=checkpoint_dir,
                force_resume=True,
            ),
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
            [1, 2, 3],
            multiplier=2,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        with pytest.raises(RuntimeError):
            interrupted.run()

        assert (checkpoint_dir / "runner.h5").exists()

        # Recover using nothing but the on-disk snapshot -- no reference
        # to `interrupted` itself.
        recovered = MultiProgramRunner.read(checkpoint_dir / "runner.h5")
        assert recovered.run() == [2, 4, 6]

    def test_resume_renames_touch_no_open_file(self, tmp_path, monkeypatch):
        """A resume rewrites runner.h5 without any rename involving a file
        the resume still holds open, which Windows can't do."""
        checkpoint_dir = tmp_path / "ckpt"
        _FLAKY_CALL_COUNT["n"] = 0
        with pytest.raises(RuntimeError):
            _FlakyRunner(
                [1, 2, 3],
                multiplier=2,
                config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
            ).run()

        calls = spy_on_renames(monkeypatch)
        resumed = _FlakyRunner(
            [1, 2, 3],
            multiplier=2,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
        )
        assert resumed.run() == [2, 4, 6]

        assert (checkpoint_dir / "runner.h5").resolve() in [
            dst for dst, _ in calls
        ]
        assert not any(touched for _, touched in calls)

    def test_checkpoint_without_resume_raises_when_content_exists(
        self, tmp_path
    ):
        """State machine case (b): checkpoint=True, resume=False, but
        on-disk state already exists raises ValueError."""
        checkpoint_dir = tmp_path / "ckpt"

        # First run: create a genuine checkpoint
        first = _CountingRunner(
            [1, 2, 3],
            multiplier=2,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        first.run()

        # Verify runner.h5 exists
        assert (checkpoint_dir / "runner.h5").exists()

        # Second run: attempt to run again with checkpoint=True but
        # resume=False should raise
        second = _CountingRunner(
            [1, 2, 3],
            multiplier=2,
            config=CheckpointConfig(
                item_checkpoint_dir=checkpoint_dir, resume=False
            ),
        )
        with pytest.raises(ValueError):
            second.run()

    def test_resume_true_with_empty_checkpoint_dir_raises(self, tmp_path):
        """State machine case (d): resume=True with checkpoint=True but
        no on-disk state raises ValueError (nothing to resume from)."""
        checkpoint_dir = tmp_path / "nonexistent_dir"

        runner = _CountingRunner(
            [1, 2, 3],
            multiplier=2,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
        )
        with pytest.raises(ValueError):
            runner.run()

    def test_cross_subclass_resume_raises_typeerror(self, tmp_path):
        """Resuming a checkpoint created by one runner subclass with a
        different subclass raises TypeError, not AttributeError, when they
        have incompatible mismatch check fields.

        `type(self).read(runner_path)` decodes the class stored in the file
        (`_SimpleDoubleRunner`), not `type(self)` (`_CountingRunner`, whose
        mismatch field `multiplier` the stored class lacks), so the resume
        must raise TypeError before field checks run.
        """
        checkpoint_dir = tmp_path / "ckpt"

        _SimpleDoubleRunner(
            [1, 2, 3],
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        ).run()

        resuming = _CountingRunner(
            [1, 2, 3],
            multiplier=2,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
        )
        with pytest.raises(TypeError):
            resuming.run()


class TestResumeRewrite:
    """`run()` writes `runner.h5` once before dispatch and reads only the
    config of a stored checkpoint."""

    def test_resume_with_item_key_fn_rewrites_runner_h5_once(self, tmp_path):
        """A resume of a runner with `item_key_fn` makes one `runner.h5`
        write session before its first item runs, and keeps the finished
        items' results. Catches a second rewrite for the index map, and a
        rewrite that loses entries."""
        ckpt = tmp_path / "ckpt"
        config = CheckpointConfig(
            item_checkpoint_dir=ckpt, show_progress=False
        )
        _KeyedRunner([1, 2], config=config).run()

        resume_config = CheckpointConfig(
            item_checkpoint_dir=ckpt, resume=True, show_progress=False
        )
        runner = _KeyedRunner([1, 2, 3, 4], config=resume_config)
        real_build_program = runner.build_program
        sessions_before_item = []

        with record_h5_writes() as tap:

            def build_program(index):
                sessions_before_item.append(
                    len(sessions_in(tap.sessions, "runner"))
                )
                return real_build_program(index)

            runner.build_program = build_program
            results = runner.run()

        assert sessions_before_item == [1, 1]
        assert results == [2, 4, 6, 8]
        assert runner._reduced_results == {0: 2, 1: 4, 2: 6, 3: 8}

    def test_mismatch_check_decodes_no_kept_results(
        self, tmp_path, monkeypatch
    ):
        """A resume whose stored config differs still raises the "Cannot
        resume" error while decoding any kept `ProgramResults` would raise.
        Catches a mismatch check that decodes every item."""
        item_dir, shot_dir = tmp_path / "items", tmp_path / "shots"
        build_sweep_runner(item_dir, shot_dir).run()

        def refuse(cls, attrs):
            raise AssertionError("a kept ProgramResults was decoded")

        monkeypatch.setattr(
            ProgramResults, "_from_decoded_attrs", classmethod(refuse)
        )
        runner = build_sweep_runner(item_dir, shot_dir, resume=True)
        runner.num_shots += 1

        with pytest.raises(ValueError):
            runner.run()


class TestStrictItemAssembly:
    """Resume and final assembly read the item store strictly and check
    completeness in both the checkpointed and uncheckpointed branches."""

    def test_resume_with_truncated_payload_raises_before_any_item(
        self, tmp_path, monkeypatch
    ):
        """A corrupt payload raises `CheckpointReadError` naming it, instead
        of the item being silently redone."""
        item_dir, shot_dir = tmp_path / "items", tmp_path / "shots"
        with build_program_interrupts_at(1):
            with pytest.raises(KeyboardInterrupt):
                build_sweep_runner(item_dir, shot_dir).run()

        (payload,) = CheckpointStore(item_dir, ITEM_LAYOUT).payload_paths()
        payload.write_bytes(payload.read_bytes()[:100])

        built = []
        real_build_program = NoiseSweepRunner.build_program
        monkeypatch.setattr(
            NoiseSweepRunner,
            "build_program",
            lambda self, i: built.append(i) or real_build_program(self, i),
        )
        runner = build_sweep_runner(item_dir, shot_dir, resume=True)
        with pytest.raises(CheckpointReadError):
            runner.run()
        assert built == []


class TestMergeReducedResult:
    """Tests for MultiProgramRunner._merge_reduced_result method."""

    def test_merge_reduced_result_updates_memory_only_and_keeps_first(
        self, tmp_path, monkeypatch
    ):
        """_merge_reduced_result touches no file (an item's payload is its
        durable record), and a repeated index keeps the first value."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _CountingRunner(
            [1, 2, 3],
            multiplier=2,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        runner.run()

        open_calls = []
        real_file_init = h5py.File.__init__

        def counting_file_init(self, *args, **kwargs):
            if args and isinstance(args[0], (str, Path)):
                open_calls.append(args)
            return real_file_init(self, *args, **kwargs)

        monkeypatch.setattr(h5py.File, "__init__", counting_file_init)

        runner._merge_reduced_result(10, "reduced_10")
        runner._merge_reduced_result(10, "reduced_10_should_be_ignored")
        assert runner._reduced_results[10] == "reduced_10"
        assert open_calls == []


class TestProcessAndCheckpointItemAtomicity:
    """`_process_and_checkpoint_item` must checkpoint one item's results via
    a single payload-file open, not one separate open per attribute -- a
    crash between separate opens could otherwise leave `results` durably
    written while the timing attributes are permanently lost after resume."""

    def test_process_and_checkpoint_item_writes_checkpoint_entries_in_one_transaction(
        self, tmp_path, monkeypatch
    ):
        """Count real payload-file opens (still delegating to the genuine
        `h5py.File`, so the actual write still happens) while
        `_process_and_checkpoint_item` processes one item with
        `keep_shot_results=True` and a real resolved `ProgramResults`, then
        confirm the payload file holds all 4 attributes (`results`,
        `item_wall_clock_times`, `shot_wall_clock_times`, `_program_results`).
        """
        from loqs.internal.swmrledger import open_swmr_writer
        from loqs.tools.multiprogramrunner import (
            _ITEM_LEDGER_FIELDS,
            _process_and_checkpoint_item,
        )

        store = CheckpointStore(tmp_path / "ckpt", ITEM_LAYOUT)
        store.directory.mkdir()
        _, ledger_group = open_swmr_writer(
            store.ledger_path("test"),
            fields=_ITEM_LEDGER_FIELDS,
        )

        def stub_process_item(
            item, index, *, shot_executor=None, n_shot_batches=None, **kwargs
        ):
            return {
                "_reduced_results": item * 2,
                "item_wall_clock_times": 0.001,
                "shot_wall_clock_times": {0: 0.0005},
            }

        # The kept results come from the item's own shot checkpoint.
        shot_dir = tmp_path / "shots"
        kept = _make_synthetic_program_results(0, shot_count=1)
        kept.num_shots = 1
        kept.checkpoint(shot_dir)

        real_h5py_file = h5py.File
        open_count = {"n": 0}

        def counting_h5py_file(path, mode, *args, **kwargs):
            if Path(path).name.endswith("_payload.h5.tmp"):
                open_count["n"] += 1
            return real_h5py_file(path, mode, *args, **kwargs)

        monkeypatch.setattr(h5py, "File", counting_h5py_file)

        _process_and_checkpoint_item(
            stub_process_item,
            item=5,
            index=0,
            static_kwargs={},
            shot_executor=None,
            n_shot_batches=None,
            keep_shot_results=True,
            shot_checkpoint_subdir=lambda index: shot_dir,
            ledger_group=ledger_group,
            item_store=store,
        )

        assert open_count["n"] == 1

        (payload_path,) = store.payload_paths()
        view = store.payload_view(payload_path)
        assert dict(view.iter_items("results")) == {0: 10}
        assert dict(view.iter_items("item_wall_clock_times")) == {
            0: pytest.approx(0.001)
        }
        assert dict(view.iter_items("shot_wall_clock_times")) == {
            0: {0: 0.0005}
        }
        assert view.keys("_program_results") == [0]

    @pytest.mark.parametrize(
        "shot_state", ["no_dir", "no_num_shots", "missing_shot"]
    )
    def test_unloadable_kept_results_raise_and_write_nothing(
        self, tmp_path, shot_state
    ):
        """Kept results come only from the item's shot checkpoint. When it
        can't supply every shot, the item raises and writes no payload,
        rather than keeping partial results."""
        from loqs.internal.swmrledger import open_swmr_writer
        from loqs.tools.multiprogramrunner import (
            _ITEM_LEDGER_FIELDS,
            _process_and_checkpoint_item,
        )

        store = CheckpointStore(tmp_path / "ckpt", ITEM_LAYOUT)
        store.directory.mkdir()
        ledger_file, ledger_group = open_swmr_writer(
            store.ledger_path("test"),
            fields=_ITEM_LEDGER_FIELDS,
        )
        shot_dir = tmp_path / "shots"
        shot_dir.mkdir()
        if shot_state in ("no_num_shots", "missing_shot"):
            kept = _make_synthetic_program_results(0, shot_count=1)
            kept.num_shots = None if shot_state == "no_num_shots" else 2
            kept.checkpoint(shot_dir)

        def stub_process_item(item, index, **kwargs):
            return {
                "_reduced_results": item,
                "item_wall_clock_times": 0.001,
                "shot_wall_clock_times": {0: 0.0005},
            }

        try:
            with pytest.raises(RuntimeError):
                _process_and_checkpoint_item(
                    stub_process_item,
                    item=5,
                    index=0,
                    static_kwargs={},
                    shot_executor=None,
                    n_shot_batches=None,
                    keep_shot_results=True,
                    shot_checkpoint_subdir=lambda index: (
                        None if shot_state == "no_dir" else shot_dir
                    ),
                    ledger_group=ledger_group,
                    item_store=store,
                )
        finally:
            ledger_file.close()
        assert store.payload_paths() == []


class TestItemCheckpointWithLedger:
    """`_write_item_checkpoint_with_ledger` writes one item's checkpoint
    entries to a dedicated payload file and marks the item done in the SWMR
    ledger only once that payload write has fully completed; `_get_worker_
    ledger` caches its opened ledger group per process per directory."""

    def test_write_item_checkpoint_with_ledger_does_not_mark_done_on_failure(
        self, tmp_path, monkeypatch
    ):
        """If `merge_dict_attr` raises partway through the payload write,
        the exception propagates and the item stays not done. A retry then
        writes the payload and marks the item done, only once the payload
        is renamed into place and closed, with the right entries and
        wall-clock time."""
        from loqs.internal import checkpointstore as store_module
        from loqs.internal.swmrledger import (
            open_swmr_writer,
            read_swmr_ledger_status,
        )
        from loqs.tools import multiprogramrunner as mpr_module

        store = CheckpointStore(tmp_path / "ckpt", ITEM_LAYOUT)
        store.directory.mkdir()
        _, ledger_group = open_swmr_writer(
            store.ledger_path("test"),
            fields=["done", "wall_clock_times"],
        )

        real_merge_dict_attr = store_module.merge_dict_attr
        call_count = {"n": 0}

        def flaky_merge_dict_attr(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated failure partway through")
            return real_merge_dict_attr(*args, **kwargs)

        entry = {
            "results": 1,
            "item_wall_clock_times": 0.1,
            "shot_wall_clock_times": {0: 0.05},
        }

        with monkeypatch.context() as patch:
            patch.setattr(
                store_module, "merge_dict_attr", flaky_merge_dict_attr
            )
            with pytest.raises(RuntimeError):
                mpr_module._write_item_checkpoint_with_ledger(
                    store, ledger_group, 5, entry, wall_clock_time=0.1
                )
        assert store.payload_paths() == []

        assert call_count["n"] == 2

        status = read_swmr_ledger_status(ledger_group)
        done_at_5 = (
            bool(status.done[5])
            if status.done is not None and len(status.done) > 5
            else False
        )
        assert done_at_5 is False

        real_mark = mpr_module.mark_ledger_item_done
        marked: list[int] = []

        def spy_mark(group, index, *args, **kwargs):
            path = store.payload_path(worker_id(), 5)
            assert path.exists()
            assert path.resolve() not in open_hdf5_file_paths()
            assert not list(store.directory.glob("*.tmp"))
            assert 5 in store.payload_view(path).keys()
            marked.append(index)
            return real_mark(group, index, *args, **kwargs)

        monkeypatch.setattr(mpr_module, "mark_ledger_item_done", spy_mark)
        mpr_module._write_item_checkpoint_with_ledger(
            store, ledger_group, 5, entry, wall_clock_time=0.1
        )

        assert marked == [5]
        view = store.payload_view(store.payload_path(worker_id(), 5))
        assert dict(view.iter_items("results")) == {5: 1}
        assert dict(view.iter_items("item_wall_clock_times")) == {
            5: pytest.approx(0.1)
        }
        status = read_swmr_ledger_status(ledger_group)
        assert bool(status.done[5]) is True
        assert status.wall_clock_times[5] == pytest.approx(0.1)

    @pytest.mark.parametrize("program_executor", ["serial", "inline_pool"])
    def test_no_item_ledger_open_after_run(
        self, tmp_path, inline, program_executor
    ):
        """Neither the driver nor a chunk worker keeps its item ledger open
        after its task: an open ledger fails consolidation's delete on
        Windows and leaves a live ledger final assembly can't read. An
        inline chunk worker runs in this process, so the check covers it."""
        strategy = None
        if program_executor == "inline_pool":
            strategy = ParallelStrategy(
                program_executor=InlineExecutor(), n_program_chunks=2
            )
        checkpoint_dir = tmp_path / "checkpoints"

        runner = _TrackingRunner(
            list(range(4)),
            process_fn=_double_item,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
            parallel_strategy=strategy,
        )
        results = runner.run()

        assert results == [0, 2, 4, 6]
        store = CheckpointStore(checkpoint_dir, ITEM_LAYOUT)
        assert [p.name for p in checkpoint_dir.iterdir()] == [
            store.canonical_path.name
        ]
        resolved_dir = checkpoint_dir.resolve()
        assert not any(
            path.is_relative_to(resolved_dir)
            for path in open_hdf5_file_paths()
        )


class TestIndexMapPersistence:
    """Tests for index_map persistence through deserialization."""

    def test_index_map_stable_across_reordered_resume(self, tmp_path):
        """Items keep their originally-assigned index across a resumed
        `MultiProgramRunner.run()` call even when passed in a different order
        or as a subset."""
        checkpoint_dir = tmp_path / "ckpt"

        runner1 = _KeyedRunner(
            [10, 20, 30],
            multiplier=2,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        assert runner1.run() == [20, 40, 60]
        assert runner1.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

        # Resume with a different order and only a subset -- both items
        # are already done, so this only exercises index stability.
        runner2 = _KeyedRunner(
            [30, 10],
            multiplier=2,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
        )
        assert runner2.run() == [60, 20]
        assert runner2.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

    def test_index_map_survives_deserialization_via_read(self, tmp_path):
        """index_map/_reduced_results survive `.read()` for a subclass with
        a fixed `__init__` signature that doesn't forward `**kwargs` (like
        `EdesignRunner`). A `**kwargs`-forwarding double would pass
        `index_map` through regardless of `_from_decoded_attrs`, so it
        can't catch this."""
        checkpoint_dir = tmp_path / "ckpt"
        items = [10, 20, 30]

        # First run: populate index_map with real data
        runner1 = _KeyedRunnerFixedSignature(
            items,
            multiplier=2,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        result1 = runner1.run()
        assert result1 == [20, 40, 60]
        assert runner1.index_map == {"item_10": 0, "item_20": 1, "item_30": 2}

        # Deserialize via .read() (the critical test: does index_map survive?)
        runner_path = checkpoint_dir / "runner.h5"
        runner2 = _KeyedRunnerFixedSignature.read(runner_path)

        # The deserialized runner must have the exact same index_map
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

    def test_keep_shot_results_false_leaves_empty(self, tmp_path):
        """When keep_shot_results is False, _program_results stays empty."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            config=CheckpointConfig(
                item_checkpoint_dir=checkpoint_dir, keep_shot_results=False
            ),
        )
        result = runner.run()
        assert result == [2, 4, 6]

        # _program_results should remain empty
        assert len(runner._program_results) == 0

    def test_keep_shot_results_lazy_loading(self, tmp_path):
        """With lazy_loading=True, _program_results contains lazy handles
        that can resolve shot content on demand, with the runner's
        `num_shots` and `max_frame_limit` forwarded and the parent program
        and name resolved from the nested source."""
        checkpoint_dir = tmp_path / "ckpt"
        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            config=CheckpointConfig(
                item_checkpoint_dir=checkpoint_dir,
                shot_checkpoint_dir=tmp_path / "shot_ckpt",
                keep_shot_results=True,
                lazy_loading=True,
            ),
        )
        # Per-runner metadata forwarded to lazy ProgramResults (set as
        # attributes; the constructor doesn't accept them)
        runner.num_shots = 5
        runner.run_kwargs["max_frame_limit"] = 200
        result = runner.run()
        assert result == [2, 4, 6]

        # `_run_dispatch` populates these two attrs directly from disk once
        # a checkpointed run completes; confirmed here alongside the
        # lazy-proxy check below.
        assert len(runner.item_wall_clock_times) == 3
        assert len(runner.shot_wall_clock_times) == 3

        # Verify _program_results was populated with lazy ProgramResults
        assert len(runner._program_results) == 3
        for index in [0, 1, 2]:
            assert index in runner._program_results
            # Check that it's a lazy ProgramResults (has nested source set)
            pr = runner._program_results[index]
            assert pr._nested_store is not None
            # The lazy proxy's shot_wall_clock_times must resolve from the
            # runner's own per-item timing data, not its constructor default.
            assert pr.shot_wall_clock_times
            assert (
                pr.shot_wall_clock_times == runner.shot_wall_clock_times[index]
            )

            # Shots load lazily from runner.h5; item `index`'s counter is
            # item_value * increment_by = (index + 1) * 2.
            shot = pr.get_shot_history(0)
            assert shot is not None
            data = pr.collect_shot_data("counter", -1)
            assert len(data) == 5
            assert all(frame_val == (index + 1) * 2 for frame_val in data)
            assert pr.num_shots == 5
            assert pr.max_frame_limit == 200
            assert pr.parent_program is not None
            assert pr.name == f"Results for Item {index}"

    @pytest.mark.parametrize(
        "parallel, results_filename",
        [(False, "my_results.h5"), (True, None)],
        ids=["serial_custom", "parallel_loky"],
    )
    def test_keep_shot_results_eager_loading(
        self, tmp_path, parallel, results_filename
    ):
        """With lazy_loading=False, _program_results contains eager results
        with correct shot content and metadata. The custom filename catches
        a hard-coded one; the real loky run has another process write the
        kept results, which the driver consolidates and decodes."""
        config_kwargs = {}
        if results_filename is not None:
            config_kwargs["results_filename"] = results_filename
        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            config=CheckpointConfig(
                item_checkpoint_dir=tmp_path / "item_ckpt",
                shot_checkpoint_dir=tmp_path / "shot_ckpt",
                keep_shot_results=True,
                lazy_loading=False,
                **config_kwargs,
            ),
            parallel_strategy=_loky_strategy() if parallel else None,
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
            assert len(pr.shot_histories) == 5

            # Eager load: "counter" equals item_value * increment_by
            # = (index + 1) * 2.
            for shot_idx in range(5):
                shot = pr.shot_histories[shot_idx]
                assert shot is not None
                assert shot.collect_data("counter", -1) == (index + 1) * 2
            assert pr.parent_program is not None
            assert pr.name == f"Results for Item {index}"

    def test_keep_shot_results_resume_preserves(self, tmp_path):
        """Resuming a run with keep_shot_results persists correctly."""
        checkpoint_dir = tmp_path / "ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        # First run completes all items
        runner1 = _SimpleDoubleRunner(
            [1, 2, 3],
            config=CheckpointConfig(
                item_checkpoint_dir=checkpoint_dir,
                shot_checkpoint_dir=shot_checkpoint_dir,
                keep_shot_results=True,
            ),
        )
        result1 = runner1.run()
        assert result1 == [2, 4, 6]
        assert len(runner1._program_results) == 3

        # Resume (all items already done)
        runner2 = _SimpleDoubleRunner(
            [1, 2, 3],
            config=CheckpointConfig(
                resume=True,
                item_checkpoint_dir=checkpoint_dir,
                shot_checkpoint_dir=shot_checkpoint_dir,
                keep_shot_results=True,
            ),
        )
        result2 = runner2.run()
        assert result2 == [2, 4, 6]
        # Program results should still be populated on resume
        assert len(runner2._program_results) == 3

    def test_keep_shot_results_write_read_round_trip(self, tmp_path):
        """Writing and reading back a runner with keep_shot_results=True preserves
        the setting and the ability to lazily read shot data for every kept item.
        """
        checkpoint_dir = tmp_path / "checkpoint"
        output_file = tmp_path / "output.h5"

        # Create a runner with keep_shot_results=True
        runner1 = _SimpleDoubleRunner(
            [1, 2],
            config=CheckpointConfig(
                item_checkpoint_dir=checkpoint_dir,
                shot_checkpoint_dir=tmp_path / "shot_ckpt",
                keep_shot_results=True,
            ),
        )
        runner1.num_shots = 3  # Set num_shots for real program execution

        # Verify the setting is True before we write
        assert runner1.keep_shot_results is True

        # Run, then write to a distinct output file (never the checkpoint
        # file itself, which write()'s own truncating open would destroy).
        runner1.run()
        runner1.write(output_file)

        # Read it back WITHOUT re-passing keep_shot_results
        runner2 = _SimpleDoubleRunner.read(output_file)

        # Verify that the setting was restored from disk
        assert runner2.keep_shot_results is True

        # Lazy shot access must survive the round trip: each kept entry's
        # nested view must relink to the on-disk checkpoint store.
        assert len(runner2._program_results) == 2
        for index in [0, 1]:
            pr = runner2._program_results[index]
            assert pr._nested_store is not None
            assert pr._nested_store._parent.canonical_path == (
                runner2.item_checkpoint_dir / runner2.runner_filename
            )
            assert pr._nested_store._key == index
            assert pr.get_shot_history(0) is not None
            data = pr.collect_shot_data("counter", -1)
            assert len(data) == 3

    def test_shots_pbar_advances_with_custom_results_filename(self, tmp_path):
        """Shots progress bar advances correctly with custom results_filename in parallel.

        Uses one item per loky worker (n_program_chunks == len(items)). The
        second item's shots each run a real Sleep instruction and, at 5
        shots, each is its own checkpoint batch, so that item stays in
        flight while its ledger records its shots one by one. The bar must
        never exceed the true total and must end at it under the custom
        filename.

        The live count must read only SWMR shot ledgers: every driver-side
        open of a file under the shot checkpoint directory has to be a
        `worker_*_checkpoint.h5` opened with `swmr=True`, never
        `results_filename` or a payload file, which workers write without
        SWMR.
        """
        import fnmatch
        from unittest.mock import patch
        from loqs.internal.checkpointstore import SHOT_LAYOUT, CheckpointStore
        from loqs.tools import multiprogramrunner as mpr_module

        loky = pytest.importorskip("loky")

        custom_filename = "my_results.h5"
        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"
        resolved_shot_dir = shot_checkpoint_dir.resolve()
        ledger_glob = (
            CheckpointStore(shot_checkpoint_dir, SHOT_LAYOUT)
            .ledger_path("*")
            .name
        )

        # Every driver-side open under the shot checkpoint directory, as
        # (file name, swmr flag).
        shot_dir_opens = []
        real_h5_file = h5py.File

        class RecordingFile(real_h5_file):
            def __init__(self, name, *args, **kwargs):
                path = Path(name).resolve()
                if resolved_shot_dir in path.parents:
                    shot_dir_opens.append(
                        (path.name, kwargs.get("swmr", False))
                    )
                super().__init__(name, *args, **kwargs)

        shots_bar_values = []

        def record_shots_bar_value(desc, value):
            if desc == "Shots":
                shots_bar_values.append(value)

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )
        with (
            patch.object(
                mpr_module,
                "tqdm",
                side_effect=_make_tqdm_spy(on_set_n=record_shots_bar_value),
            ),
            patch.object(h5py, "File", RecordingFile),
        ):
            runner = _CustomFilenameProbeRunner(
                [1, 2],
                num_shots=5,
                parallel_strategy=strategy,
                config=CheckpointConfig(
                    item_checkpoint_dir=item_checkpoint_dir,
                    shot_checkpoint_dir=shot_checkpoint_dir,
                    show_progress=True,
                    poll_interval=0.05,
                    results_filename=custom_filename,
                ),
            )
            runner.run()

        assert shot_dir_opens
        for name, swmr in shot_dir_opens:
            assert fnmatch.fnmatch(name, ledger_glob)
            assert swmr

        assert shots_bar_values

        # True total is len(items) * num_shots = 2 * 5 = 10. The bar must
        # never exceed it, and must end there.
        assert max(shots_bar_values) <= 10
        assert shots_bar_values[-1] == 10

    @pytest.mark.slow
    def test_keep_shot_results_parallel_forces_checkpoint_read_write_race(
        self, tmp_path
    ):
        """With `poll_interval=0.0`, the driver polls shot progress as fast
        as it can while workers flush shot checkpoints. Checkpoint opens are
        never retried, so any lock collision would surface as
        `BlockingIOError`. The driver reads only SWMR shot ledgers while
        workers write, so the run completes with every shot kept."""
        loky = pytest.importorskip("loky")

        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        executor = loky.ProcessPoolExecutor(max_workers=2)
        try:
            strategy = ParallelStrategy(
                program_executor=executor,
                n_program_chunks=2,
            )
            runner = _SimpleDoubleRunner(
                [1, 2, 3, 4],
                config=CheckpointConfig(
                    item_checkpoint_dir=item_checkpoint_dir,
                    shot_checkpoint_dir=shot_checkpoint_dir,
                    keep_shot_results=True,
                    lazy_loading=False,
                    poll_interval=0.0,
                ),
                parallel_strategy=strategy,
            )
            runner.num_shots = 30
            result = runner.run()

            assert result == [2, 4, 6, 8]
            assert len(runner._program_results) == 4
            for index in range(4):
                assert len(runner._program_results[index].shot_histories) == 30
        finally:
            executor.shutdown(wait=True)


class TestShotProgressBar:
    """Tests for shot-level progress bar."""

    def test_read_worker_current_indices_tolerates_missing_files(
        self, tmp_path
    ):
        """Verify _read_worker_current_indices handles missing/unreadable files."""
        from loqs.internal.swmrledger import (
            open_swmr_writer,
            update_ledger_in_flight,
        )
        from loqs.tools.multiprogramrunner import (
            _ITEM_LEDGER_FIELDS,
            _read_worker_current_indices,
        )

        store = CheckpointStore(tmp_path / "checkpoint", ITEM_LAYOUT)
        store.directory.mkdir()

        # No workers yet, should return empty set
        assert _read_worker_current_indices(store) == set()

        handles = []
        try:
            for worker, index in (("test", 5), ("test2", 7)):
                handle = open_swmr_writer(
                    store.ledger_path(worker), fields=_ITEM_LEDGER_FIELDS
                )
                handles.append(handle)
                update_ledger_in_flight(handle[1], index)
                expected = {5} if index == 5 else {5, 7}
                assert _read_worker_current_indices(store) == expected

            # A non-HDF5 file at a ledger name is skipped.
            store.ledger_path("junk").write_bytes(b"not hdf5")
            assert _read_worker_current_indices(store) == {5, 7}
        finally:
            for handle in handles:
                handle[0].close()

    @pytest.mark.parametrize(
        "checkpoint_dirs, show_progress, prints_notice",
        [
            ("none", True, True),
            ("both", False, False),
            ("none", False, False),
        ],
        ids=[
            "prints_once_parallel_without_shot_checkpoint",
            "silent_when_show_progress_false_fully_configured",
            "silent_when_show_progress_false",
        ],
    )
    def test_shot_progress_message_and_shots_bar(
        self,
        tmp_path,
        inline,
        capsys,
        checkpoint_dirs,
        show_progress,
        prints_notice,
    ):
        """Parallel dispatch with a missing checkpoint setting prints the
        shot-level progress notice only with show_progress=True, and no
        shots bar is ever created in these configurations: the shots bar
        must respect the same show_progress opt-out as the plain items bar,
        and on_poll never touches it unless checkpointing is fully set up,
        so it would otherwise freeze."""
        from unittest.mock import patch

        config_kwargs = {"show_progress": show_progress}
        if checkpoint_dirs == "both":
            config_kwargs["item_checkpoint_dir"] = tmp_path / "item_ckpt"
            config_kwargs["shot_checkpoint_dir"] = tmp_path / "shot_ckpt"

        tqdm_descs = []
        runner = _ShotProgressTestRunner(
            [1, 2, 3],
            num_shots=5,
            parallel_strategy=ParallelStrategy(
                program_executor=InlineExecutor(), n_program_chunks=2
            ),
            config=CheckpointConfig(**config_kwargs),
        )
        with patch(
            "loqs.tools.multiprogramrunner.tqdm",
            side_effect=_make_tqdm_spy(
                on_init=lambda tqdm_obj, desc: tqdm_descs.append(desc)
            ),
        ):
            runner.run()

        assert "Shots" not in tqdm_descs
        assert (capsys.readouterr().out != "") is prints_notice

    def test_shots_bar_total_correct_on_resumed_run(self, tmp_path, inline):
        """Regression test: shots bar total is sized correctly even on resumed run.

        Bug: if shots bar was sized as total=len(remaining) * num_shots instead of
        len(items) * num_shots, then on a resumed run (where remaining < items),
        the initial value would be too small and .n could exceed total.

        This test directly verifies the bar initialization parameters when some
        items are pre-marked as done (simulating a prior interrupted run).
        """
        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"
        item_checkpoint_dir.mkdir()
        shot_checkpoint_dir.mkdir()

        from unittest.mock import patch

        tqdm_inits = []

        def record_init(tqdm_obj, desc):
            tqdm_inits.append(
                {
                    "total": tqdm_obj.total,
                    "initial": tqdm_obj.n,
                    "desc": desc,
                }
            )

        strategy = ParallelStrategy(
            program_executor=InlineExecutor(), n_program_chunks=2
        )

        # Simulates 2 already-done items (a prior interrupted run) -- only
        # the tqdm init params computed before dispatch matter here.
        def read_with_preseeded_done(*args, **kwargs):
            return iter([(0, 0), (2, 4)])

        with patch(
            "loqs.tools.multiprogramrunner.tqdm",
            side_effect=_make_tqdm_spy(on_init=record_init),
        ):
            with patch.object(
                CheckpointStore,
                "iter_items",
                side_effect=read_with_preseeded_done,
            ):
                runner = _ShotProgressTestRunner(
                    [1, 2, 3],
                    num_shots=5,
                    parallel_strategy=strategy,
                    config=CheckpointConfig(
                        item_checkpoint_dir=item_checkpoint_dir,
                        shot_checkpoint_dir=shot_checkpoint_dir,
                        show_progress=True,
                    ),
                )
                with contextlib.suppress(Exception):
                    runner.run()

        # Find shots bar initialization event
        shots_inits = [e for e in tqdm_inits if e["desc"] == "Shots"]
        assert shots_inits
        shots_init = shots_inits[0]

        # total = len(items)*num_shots = 15, fixed across a resume rather
        # than shrinking to len(remaining)*num_shots as items complete.
        assert shots_init["total"] == 15

        # initial should be len(done) * num_shots = 2 * 5 = 10
        # (indices 0 and 2 were pre-done)
        assert shots_init["initial"] == 10


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
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
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
        store = CheckpointStore(checkpoint_dir, ITEM_LAYOUT)
        assert store.ledger_paths() == []
        assert store.payload_paths() == []

        # All results should be consolidated into runner.h5
        runner_path = checkpoint_dir / "runner.h5"
        assert runner_path.exists()
        with h5py.File(runner_path, "r") as f:
            from loqs.internal.streamingmerge import iter_dict_attr_entries

            entries = dict(iter_dict_attr_entries(f, "_reduced_results"))
        assert len(entries) == 5
        assert entries == {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}

    def test_consolidate_warns_when_delete_fails(self, tmp_path):
        """A worker file whose delete fails after its merge raises a
        `RuntimeWarning` and is left in place, instead of being silently
        ignored; the run still returns the right results."""
        checkpoint_dir = tmp_path / "checkpoints"
        real_unlink = Path.unlink

        def failing_unlink(self, *args, **kwargs):
            if self.name.startswith("worker_"):
                raise PermissionError(13, "file in use", str(self))
            return real_unlink(self, *args, **kwargs)

        runner = _TrackingRunner(
            list(range(3)),
            process_fn=_double_item,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        with unittest.mock.patch.object(Path, "unlink", failing_unlink):
            with pytest.warns(RuntimeWarning):
                results = runner.run()

        assert results == [0, 2, 4]
        store = CheckpointStore(checkpoint_dir, ITEM_LAYOUT)
        assert store.ledger_paths()
        assert store.payload_paths()

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
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        results1 = runner1.run()
        assert results1 == [0, 2, 4, 6, 8, 10]
        _assert_wall_clock_times(runner1, 6)

        # Verify worker files are gone
        store = CheckpointStore(checkpoint_dir, ITEM_LAYOUT)
        assert store.ledger_paths() == []

        # Second run: resume from checkpoint where only runner.h5 exists
        # (no worker files). This tests that done-detection reads from
        # runner.h5's _reduced_results correctly.
        runner2 = _TrackingRunner(
            items,
            process_fn=_double_item,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
        )
        results2 = runner2.run()

        # All items should be skipped (already done)
        assert runner2.call_count[0] == 0
        assert results2 == [0, 2, 4, 6, 8, 10]
        _assert_wall_clock_times(runner2, 6)

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
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
            max_count=2,
        )
        with pytest.raises(RuntimeError):
            runner1.run()

        # Verify partial results are persisted (in payloads or runner.h5)
        store = CheckpointStore(checkpoint_dir, ITEM_LAYOUT)
        on_disk_after_crash1 = dict(store.iter_items())
        assert len(on_disk_after_crash1) == 2
        assert on_disk_after_crash1 == {0: 0, 1: 2}

        # Second run: resume and complete items 2-3 before crashing again.
        # Items 0-1 are already done, so only 2-5 will be processed (4 items).
        # To process items 2-3 then crash on item 4, we need max_count=2.
        runner2 = _TrackingRunner(
            items,
            process_fn=_raise_after_n,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
            max_count=2,  # Will process items 2,3 then crash on item 4
        )
        with pytest.raises(RuntimeError):
            runner2.run()

        # Verify results 0-3 are persisted (the original 0-1 aren't lost
        # or blanked out when runner.h5 was seeded before write)
        on_disk_after_crash2 = dict(
            CheckpointStore(checkpoint_dir, ITEM_LAYOUT).iter_items()
        )
        assert len(on_disk_after_crash2) == 4
        assert on_disk_after_crash2 == {0: 0, 1: 2, 2: 4, 3: 6}

        # Third run: resume and complete the remaining items (4-5)
        runner3 = _TrackingRunner(
            items,
            process_fn=_count_and_double,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
        )
        results3 = runner3.run()

        # Final result should have all items with correct values
        assert results3 == [0, 2, 4, 6, 8, 10]

        # Verify runner.h5 now has all 6 items consolidated
        with h5py.File(checkpoint_dir / "runner.h5", "r") as f:
            from loqs.internal.streamingmerge import iter_dict_attr_entries

            final_on_disk = dict(iter_dict_attr_entries(f, "_reduced_results"))
        assert len(final_on_disk) == 6
        assert final_on_disk == {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10}

    def test_resume_rewrite_keeps_committed_results(
        self, tmp_path, monkeypatch
    ):
        """A resuming run() rewrites runner.h5 with the finished items'
        results still in it, so the rewrite never leaves a window in which
        they are blank. Uses _CountingRunner, whose first run() completes
        fully and so genuinely leaves non-empty on-disk state.
        """
        checkpoint_dir = tmp_path / "checkpoints"
        items = [1, 2, 3]

        # First run: completes fully, so its results are consolidated into
        # runner.h5's own _reduced_results attribute.
        runner1 = _CountingRunner(
            items,
            multiplier=2,
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        runner1.run()

        with h5py.File(checkpoint_dir / "runner.h5", "r") as f:
            from loqs.internal.streamingmerge import iter_dict_attr_entries

            on_disk_after_run1 = dict(
                iter_dict_attr_entries(f, "_reduced_results")
            )
        assert on_disk_after_run1 == {0: 2, 1: 4, 2: 6}

        # Second run: resumes the same runner. Capture runner.h5's on-disk
        # results right after the rewrite inside run(), before any dispatch
        # happens.
        runner2 = _CountingRunner(
            items,
            multiplier=2,
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
        )

        captured_after_rewrite = []
        real_write_snapshot = CheckpointStore.write_snapshot

        def capturing_write_snapshot(self, obj, **kwargs):
            real_write_snapshot(self, obj, **kwargs)
            captured_after_rewrite.append(dict(self.iter_items("results")))

        monkeypatch.setattr(
            CheckpointStore, "write_snapshot", capturing_write_snapshot
        )
        runner2.run()

        # The runner's own blank _reduced_results must not have replaced
        # the finished items' results.
        assert captured_after_rewrite == [{0: 2, 1: 4, 2: 6}]

    def test_missing_program_results_entry_should_raise(
        self, tmp_path, monkeypatch
    ):
        """The completeness check for `_program_results` raises
        `RuntimeError` naming the missing index if an entry is absent from
        the item store at final assembly."""
        checkpoint_dir = tmp_path / "ckpt"
        missing_index = 1
        real_keys = CheckpointStore.keys

        def keys_dropping_program_result(self, attr=None, **kwargs):
            """Drop one key from `_program_results` only; `results` is
            read through unaffected and complete."""
            keys = real_keys(self, attr, **kwargs)
            if attr == "_program_results":
                keys = [key for key in keys if key != missing_index]
            return keys

        monkeypatch.setattr(
            CheckpointStore, "keys", keys_dropping_program_result
        )

        runner = _SimpleDoubleRunner(
            [1, 2, 3],
            config=CheckpointConfig(
                item_checkpoint_dir=checkpoint_dir,
                shot_checkpoint_dir=tmp_path / "shot_ckpt",
                keep_shot_results=True,
                lazy_loading=True,
            ),
        )

        with pytest.raises(RuntimeError):
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

    def test_shared_reference_decode_cache(self, tmp_path):
        """Regression test for the store's consolidation: a Serializable
        value referenced by multiple attributes of one item's payload
        decodes to a real object (not a DeferredRef placeholder on its
        later occurrences) after consolidation."""
        store = CheckpointStore(tmp_path / "checkpoints", ITEM_LAYOUT)
        store.write_snapshot(_SimpleDoubleRunner(items=[]))

        shared_parent = _SharedParent(value=77)
        result_item = _ItemWithParent(item_id=10, parent=shared_parent)
        program_results_item = _ItemWithParent(
            item_id=20, parent=shared_parent
        )
        store.put_batch(
            {
                200: {
                    "results": result_item,
                    "item_wall_clock_times": 0.5,
                    "shot_wall_clock_times": {},
                    "_program_results": program_results_item,
                }
            },
            worker_id="0",
        )
        store.consolidate()
        assert store.payload_paths() == []

        for result in (store.get(200), store.get(200, "_program_results")):
            assert isinstance(result, _ItemWithParent)
            assert isinstance(result.parent, _SharedParent)
            assert result.parent.value == 77


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
        assert len(weakrefs) == 2
        assert all(ref() is None for ref in weakrefs)

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

    def test_build_output_warns_on_incomplete_ordered_reduced_results(self):
        """build_output's None-result warning must also fire for a genuinely
        partial _reduced_results (the load-without-resume scenario), not just
        a synthetic ordered_results list constructed directly."""
        runner = _NewHookRunner(items=["a", "b", "c"])
        runner._reduced_results = {0: 1, 2: 3}  # index 1 never completed

        with pytest.warns(UserWarning):
            output = runner.build_output(runner._ordered_reduced_results())

        assert output == [1, None, 3]

    def test_cascade_guard_resumes_when_checkpoint_file_already_exists(
        self, tmp_path
    ):
        """If shot_checkpoint=True and the per-item checkpoint's results
        file already exists on disk, the cascade guard must resume
        (resume=True, checkpoint=True, checkpoint_dir set) rather than
        raising or trying to fresh-create an already-existing checkpoint.
        Its on-disk existence check must consult the same custom
        results_filename passed to the runner, not a hardcoded
        'results.h5'."""
        shot_checkpoint_dir = tmp_path / "shot_checkpoints"
        item_dir = shot_checkpoint_dir / "item_0"
        item_dir.mkdir(parents=True)
        results_filename = "custom_results.h5"
        (item_dir / results_filename).touch()

        run_kwargs_log = []
        runner = _NewHookRunner(
            items=[7],
            run_kwargs_log=run_kwargs_log,
            config=CheckpointConfig(
                shot_checkpoint_dir=shot_checkpoint_dir,
                results_filename=results_filename,
            ),
        )

        runner.run()

        assert len(run_kwargs_log) == 1
        assert run_kwargs_log[0]["resume"] is True
        assert run_kwargs_log[0]["checkpoint"] is True
        assert run_kwargs_log[0]["checkpoint_dir"] == item_dir


class TestBaseClassMechanisms:
    """Direct tests for base-class mechanisms (mismatch, normalized args, etc.)."""

    @pytest.mark.parametrize("spec", ["equivalent", "different"])
    def test_resume_compares_normalized_collect_shot_data_args(
        self, tmp_path, spec
    ):
        """A resume with an equivalent encoding of the stored
        `collect_shot_data_args` succeeds; a different spec raises."""
        checkpoint_dir = tmp_path / "ckpt"
        checkpoint_dir.mkdir(parents=True)
        _ShotDataArgsRunner(
            [1],
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
            collect_shot_data_args=[("counter", -1)],
        ).run()

        if spec == "equivalent":
            resumed_args = [{"key": "counter", "indices": -1}]
        else:
            resumed_args = [("counter", 0)]
        runner2 = _ShotDataArgsRunner(
            [1],
            config=CheckpointConfig(
                resume=True, item_checkpoint_dir=checkpoint_dir
            ),
            collect_shot_data_args=resumed_args,
        )

        if spec == "equivalent":
            assert runner2.run() == [2]
        else:
            with pytest.raises(ValueError):
                runner2.run()

    def test_run_kwargs_checkpoint_dir_conflict(self, tmp_path):
        """ValueError raised when both checkpoint_dir in run_kwargs and
        shot_checkpoint_dir are set."""
        with pytest.raises(ValueError):
            _SimpleDoubleRunner(
                [1],
                config=CheckpointConfig(
                    shot_checkpoint_dir=tmp_path / "shots"
                ),
                run_kwargs={"checkpoint_dir": tmp_path / "other"},
            )

    def test_run_kwargs_max_frame_limit_warning_on_missing(self):
        """Warning fires when max_frame_limit not in run_kwargs."""
        with pytest.warns(UserWarning):
            runner = _SimpleDoubleRunner([1], run_kwargs=None)
        assert "max_frame_limit" in runner.run_kwargs

    def test_run_kwargs_reach_program_run(self):
        """`run_kwargs` reach `program.run`: a callable value is resolved
        against the item (not its index), `n_shot_batches` is kept, and a
        user `progress_callback` passes through as the same object."""

        def user_callback(shots_done, shots_total):
            pass

        log = []
        _NewHookRunner(
            items=[5, 10, 15],
            run_kwargs_log=log,
            run_kwargs={
                "max_frame_limit": lambda item: item * 1000,
                "n_shot_batches": 2,
                "progress_callback": user_callback,
            },
            config=CheckpointConfig(item_checkpoint_dir=None),
        ).run()

        assert len(log) == 3
        for kwargs, item in zip(log, [5, 10, 15]):
            assert kwargs["max_frame_limit"] == item * 1000
            assert kwargs["n_shot_batches"] == 2
            assert kwargs["progress_callback"] is user_callback

        plain_log = []
        _NewHookRunner(
            items=[5],
            run_kwargs_log=plain_log,
            config=CheckpointConfig(item_checkpoint_dir=None),
        ).run()
        assert "progress_callback" not in plain_log[0]

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
            config=CheckpointConfig(
                item_checkpoint_dir=None, shot_checkpoint_dir=shot_ckpt_dir
            ),
        )
        runner1.num_shots = 1
        result1 = runner1.run()
        assert result1 == [2, 4]

        # Same shot checkpoint, different num_shots, force_resume=False:
        # QuantumProgram.run() itself raises on the num_shots mismatch.
        runner2 = _CountingRunner(
            [1, 2],
            config=CheckpointConfig(
                item_checkpoint_dir=None,
                shot_checkpoint_dir=shot_ckpt_dir,
                force_resume=False,
            ),
        )
        runner2.num_shots = 2
        with pytest.raises(ValueError):
            runner2.run()

        # Same num_shots mismatch, but force_resume=True: succeeds only if
        # force_resume was actually forwarded into program.run().
        runner3 = _CountingRunner(
            [1, 2],
            config=CheckpointConfig(
                item_checkpoint_dir=None,
                shot_checkpoint_dir=shot_ckpt_dir,
                force_resume=True,
            ),
        )
        runner3.num_shots = 2
        result3 = runner3.run()
        assert result3 == [2, 4]

    def test_custom_runner_filename_checkpoint_and_resume(self, tmp_path):
        """Custom runner_filename is used for checkpoint and resume."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir(parents=True)

        # Run with custom runner_filename
        runner1 = _SimpleDoubleRunner(
            [1],
            config=CheckpointConfig(
                item_checkpoint_dir=ckpt_dir,
                runner_filename="custom_runner.h5",
            ),
        )
        result1 = runner1.run()
        assert (ckpt_dir / "custom_runner.h5").exists()
        assert not (ckpt_dir / "runner.h5").exists()

        # Resume with same custom runner_filename
        runner2 = _SimpleDoubleRunner(
            [1],
            config=CheckpointConfig(
                resume=True,
                item_checkpoint_dir=ckpt_dir,
                runner_filename="custom_runner.h5",
            ),
        )
        result2 = runner2.run()
        assert result2 == result1

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="submitit unconditionally registers a SIGCONT handler, POSIX-only",
    )
    def test_submitit_debug_executor_factory_with_checkpointing(
        self, tmp_path, monkeypatch
    ):
        """A zero-argument program_executor factory is called, so a
        checkpointed submitit run works and resumes from its runner file."""
        pytest.importorskip("submitit")
        monkeypatch.chdir(tmp_path)

        items = [3, -2, 1, 4]
        serial_result = _CountingRunner(items=items).run()

        checkpoint_dir = tmp_path / "ckpt"
        runner = _CountingRunner(
            items=items,
            parallel_strategy=ParallelStrategy(
                program_executor=_submitit_debug_executor_factory,
                n_program_chunks=2,
            ),
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        assert runner.run() == serial_result

        resumed = _CountingRunner.read(checkpoint_dir / "runner.h5")
        assert resumed.run() == serial_result

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="submitit unconditionally registers a SIGCONT handler, POSIX-only",
    )
    def test_submitit_executor_spec_with_checkpointing(self, tmp_path):
        """A submitit `ExecutorSpec` is stored as data, and a runner read
        from its file rebuilds an executor with the same settings."""
        pytest.importorskip("submitit")
        spec = ExecutorSpec.submitit(
            tmp_path / "submitit-logs", cluster="debug", timeout_min=5
        )
        items = [3, -2, 1, 4]
        serial_result = _CountingRunner(items=items).run()

        checkpoint_dir = tmp_path / "ckpt"
        runner = _CountingRunner(
            items=items,
            parallel_strategy=ParallelStrategy(
                program_executor=spec, n_program_chunks=2
            ),
            config=CheckpointConfig(item_checkpoint_dir=checkpoint_dir),
        )
        assert runner.run() == serial_result

        resumed = _CountingRunner.read(checkpoint_dir / "runner.h5")
        loaded_spec = resumed.parallel_strategy.program_executor
        assert loaded_spec == spec
        executor = loaded_spec()
        assert executor.folder == tmp_path / "submitit-logs"
        assert executor.cluster == "debug"
        assert executor.parameters == {"timeout_min": 5}

    def test_hybrid_program_and_shot_executor_parallelism(self, tmp_path):
        """Nested program and shot pools return the serial results, and
        each item gets its own shot-checkpoint directory even when two
        items share a chunk."""
        loky = pytest.importorskip("loky")
        from _shared_checkpoint_test_helpers import _build_shot_executor

        items = [2, 2, 2, 2]
        serial_result = _CountingRunner(items=items).run()

        shot_ckpt_dir = tmp_path / "shots"
        runner = _CountingRunner(
            items=items,
            config=CheckpointConfig(
                shot_checkpoint_dir=shot_ckpt_dir, lazy_loading=False
            ),
            parallel_strategy=ParallelStrategy(
                program_executor=loky.get_reusable_executor(max_workers=2),
                n_program_chunks=2,
                shot_executor=_build_shot_executor,
            ),
        )
        assert runner.run() == serial_result

        for index in range(len(items)):
            assert (shot_ckpt_dir / f"item_{index}" / "results.h5").exists()


class _ProgressCallingFakeProgram(_FakeProgram):
    """`_FakeProgram` whose `run()` reports `(0, num_shots)` then
    `(num_shots, num_shots)` to its `progress_callback`, as a real
    `QuantumProgram.run()` does at its first and last shot."""

    def __init__(self, value, num_shots, run_kwargs_log=None):
        super().__init__(value, run_kwargs_log)
        self.num_shots = num_shots

    def run(self, **kwargs):
        callback = kwargs["progress_callback"]
        callback(0, self.num_shots)
        callback(self.num_shots, self.num_shots)
        return super().run(**kwargs)


def _read_item_ledger_snapshot(item_checkpoint_dir):
    """Read this process's (closed) item ledger in `item_checkpoint_dir`."""
    from loqs.internal.swmrledger import (
        open_swmr_reader,
        read_swmr_ledger_status,
    )
    from loqs.internal import worker_id

    store = CheckpointStore(item_checkpoint_dir, ITEM_LAYOUT)
    f, ledger_group = open_swmr_reader(store.ledger_path(worker_id()))
    try:
        return read_swmr_ledger_status(ledger_group)
    finally:
        f.close()


class TestLedgerShotProgress:
    """Item workers record the in-flight item, a heartbeat and live shot
    progress in their SWMR item ledger, rate-limited, without swallowing a
    user `progress_callback` given in `run_kwargs`."""

    def test_rate_limited_callback_writes_first_interval_and_final_calls(
        self, tmp_path
    ):
        """With a fake clock, only the first call, a call at least
        `_LEDGER_PROGRESS_MIN_INTERVAL` (1 s) after the last write, and the
        final `(N, N)` call write shot counts and a heartbeat. A missing limit would flush HDF5
        on every shot; a dropped final write would leave the item looking
        unfinished."""
        from loqs.internal.swmrledger import (
            open_swmr_writer,
            update_ledger_heartbeat,
        )
        from loqs.tools.multiprogramrunner import (
            _ITEM_LEDGER_FIELDS,
            _make_ledger_progress_callback,
        )

        ledger_file, ledger_group = open_swmr_writer(
            tmp_path / "ledger.h5", fields=_ITEM_LEDGER_FIELDS
        )
        try:
            now = [0.0]
            callback = _make_ledger_progress_callback(
                ledger_group, 3, clock=lambda: now[0]
            )
            num_shots = 10
            # (clock time, shots_done, expect a write)
            steps = [
                (0.0, 0, True),  # first call
                (0.5, 3, False),
                (0.9, 5, False),
                (1.0, 6, True),  # interval reached since t=0.0
                (1.5, 8, False),
                (1.7, num_shots, True),  # final shot, inside the interval
            ]
            last_written = None
            for clock_time, shots_done, expect_write in steps:
                update_ledger_heartbeat(ledger_group, timestamp=-1.0)
                now[0] = clock_time
                callback(shots_done, num_shots)

                heartbeat = float(ledger_group["last_heartbeat"][0])
                ledger_done = int(ledger_group["item_shots_done"][0])
                ledger_total = int(ledger_group["item_shots_total"][0])
                if expect_write:
                    last_written = shots_done
                    assert heartbeat > 0, (clock_time, heartbeat)
                    assert int(ledger_group["current_item_index"][0]) == 3
                else:
                    assert heartbeat == -1.0, (clock_time, heartbeat)
                assert ledger_done == last_written, (clock_time, ledger_done)
                assert ledger_total == num_shots
        finally:
            ledger_file.close()

    def test_user_progress_callback_combined_with_ledger_callback(
        self, tmp_path
    ):
        """With item checkpointing on, both a user `progress_callback` in
        `run_kwargs` and the item ledger see `run()`'s progress calls.
        Catches the ledger callback silently replacing the user's."""
        from loqs.tools import multiprogramrunner as mpr

        item_checkpoint_dir = tmp_path / "ckpt"
        item_checkpoint_dir.mkdir()
        num_shots = 3
        user_calls = []

        def user_callback(shots_done, shots_total):
            user_calls.append((shots_done, shots_total))

        process_item = functools.partial(
            mpr._shared_item_worker,
            build_program=lambda index: _ProgressCallingFakeProgram(
                7, num_shots
            ),
            reduce_program_outcomes=lambda program_results: program_results.value,
            run_kwargs={"progress_callback": user_callback},
        )
        results = mpr._run_serial(
            [(4, "x")],
            process_item,
            {},
            CheckpointStore(item_checkpoint_dir, ITEM_LAYOUT),
            None,
            None,
            None,
        )

        assert results[4]["_reduced_results"] == 7
        assert user_calls == [(0, num_shots), (num_shots, num_shots)]
        snapshot = _read_item_ledger_snapshot(item_checkpoint_dir)
        assert snapshot.current_item_index == 4
        assert snapshot.item_shots_done == num_shots
        assert snapshot.item_shots_total == num_shots


class TestDriverLiveItemReads:
    """The driver's live reads of new-format worker ledgers, over
    synthetic ledgers from the shared builder."""

    def test_live_poll_opens_only_done_payloads(self, tmp_path, monkeypatch):
        """Only done items' payloads are opened and each notifies once.
        Catches the live path opening unfinished items' payloads,
        double-notifying, or not notifying (the end-of-dispatch pass
        would otherwise hide it)."""
        import shutil

        from _shared_checkpoint_test_helpers import write_item_ledger
        from loqs.tools import multiprogramrunner as mpr

        worker_file = write_item_ledger(
            tmp_path,
            payloads={0: "value0", 2: "value2"},
            current_item=3,
            shots_done=1,
            shots_total=4,
        )
        prefix = "worker_nodeA_1000_0000abcd"
        for decoy in (1, 3):
            shutil.copy(
                tmp_path / f"{prefix}_item_0_payload.h5",
                tmp_path / f"{prefix}_item_{decoy}_payload.h5",
            )

        opened = []
        real_payload_view = CheckpointStore.payload_view

        def recording_payload_view(self, path):
            opened.append(Path(path).name)
            return real_payload_view(self, path)

        monkeypatch.setattr(
            CheckpointStore, "payload_view", recording_payload_view
        )

        items_map = {i: f"item{i}" for i in range(4)}
        notified = []
        observed: set[int] = set()

        def on_item_done(index, item, value):
            notified.append((index, item, value))

        store = CheckpointStore(tmp_path, ITEM_LAYOUT)
        count = mpr._poll_one_new_format_worker_file(
            store, worker_file, 0, observed, items_map, on_item_done, None
        )
        assert sorted(opened) == [
            f"{prefix}_item_0_payload.h5",
            f"{prefix}_item_2_payload.h5",
        ]
        assert sorted(notified) == [
            (0, "item0", "value0"),
            (2, "item2", "value2"),
        ]
        assert count == 2
        assert observed == {0, 2}

        opened.clear()
        notified.clear()
        again = mpr._poll_one_new_format_worker_file(
            store, worker_file, count, observed, items_map, on_item_done, None
        )
        assert opened == []
        assert notified == []
        assert again == count
