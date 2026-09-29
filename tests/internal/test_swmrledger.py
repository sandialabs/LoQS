"""Tests for loqs/internal/swmrledger.py's SWMR progress ledger primitives,
and for the `worker_id()` hardening in loqs/internal/__init__.py."""

import multiprocessing
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from loqs.internal import worker_id
from loqs.internal.swmrledger import (
    SwmrLedgerSnapshot,
    init_swmr_ledger,
    mark_ledger_item_done,
    open_swmr_reader,
    open_swmr_writer,
    read_swmr_ledger_status,
    refresh_swmr_ledger,
    update_ledger_heartbeat,
    update_ledger_in_flight,
)

ALL_FIELDS = [
    "done",
    "wall_clock_times",
    "current_item_index",
    "item_shots_done",
    "item_shots_total",
    "last_heartbeat",
]

PROCESS_TIMEOUT = 10


# --------------------------------------------------------------------------
# Module-level process targets (must be picklable for multiprocessing.Process)
# --------------------------------------------------------------------------


def _writer_full_cycle(path_str, ready_event, writes_done_event):
    """Write every field once, then keep the file open briefly so a reader
    definitely has a chance to observe it live."""
    f, ledger_group = open_swmr_writer(Path(path_str), ALL_FIELDS, capacity=4)
    ready_event.set()
    mark_ledger_item_done(ledger_group, 0, 1.5)
    mark_ledger_item_done(ledger_group, 2, 2.5)
    update_ledger_in_flight(
        ledger_group, item_index=1, shots_done=3, shots_total=10
    )
    update_ledger_heartbeat(ledger_group, timestamp=12345.0)
    writes_done_event.set()
    time.sleep(0.2)
    f.close()


def _reader_collect_status(path_str, ready_event, writes_done_event, queue):
    """Wait for the writer to finish, then relay a full snapshot back."""
    ready_event.wait(timeout=PROCESS_TIMEOUT)
    f, ledger_group = open_swmr_reader(Path(path_str))
    writes_done_event.wait(timeout=PROCESS_TIMEOUT)
    snapshot = read_swmr_ledger_status(ledger_group)
    queue.put(
        {
            "done": snapshot.done.tolist(),
            "wall_clock_times": snapshot.wall_clock_times.tolist(),
            "current_item_index": snapshot.current_item_index,
            "item_shots_done": snapshot.item_shots_done,
            "item_shots_total": snapshot.item_shots_total,
            "last_heartbeat": snapshot.last_heartbeat,
        }
    )
    f.close()


def _writer_extend_sequence(
    path_str, ready_event, phase1_event, go_extend_event, phase2_event
):
    """Write two items within the initial small capacity, then (once told
    to go) write a far-out index that forces auto-extension."""
    f, ledger_group = open_swmr_writer(
        Path(path_str), ["done", "wall_clock_times"], capacity=2, chunk_size=2
    )
    ready_event.set()
    mark_ledger_item_done(ledger_group, 0, 0.1)
    mark_ledger_item_done(ledger_group, 1, 0.2)
    phase1_event.set()
    go_extend_event.wait(timeout=PROCESS_TIMEOUT)
    mark_ledger_item_done(ledger_group, 5, 0.5)
    phase2_event.set()
    time.sleep(0.2)
    f.close()


def _reader_extend_sequence(
    path_str, ready_event, phase1_event, captured_event, phase2_event, queue
):
    """Capture a pre-extension snapshot, signal that it was captured, then
    capture a post-extension snapshot once the writer has grown the ledger."""
    ready_event.wait(timeout=PROCESS_TIMEOUT)
    f, ledger_group = open_swmr_reader(Path(path_str))
    phase1_event.wait(timeout=PROCESS_TIMEOUT)
    refresh_swmr_ledger(ledger_group)
    shape1 = ledger_group["done"].shape[0]
    done1 = ledger_group["done"][()].tolist()
    captured_event.set()
    phase2_event.wait(timeout=PROCESS_TIMEOUT)
    refresh_swmr_ledger(ledger_group)
    queue.put(
        {
            "shape1": shape1,
            "done1": done1,
            "shape2": ledger_group["done"].shape[0],
            "done2": ledger_group["done"][()].tolist(),
            "time5": ledger_group["wall_clock_times"][5],
        }
    )
    f.close()


def _writer_hammer(path_str, ready_event, iterations, error_queue):
    """Repeatedly write every in-flight field in a tight loop, to stress
    concurrent access alongside a hammering reader."""
    try:
        f, ledger_group = open_swmr_writer(Path(path_str), ALL_FIELDS)
        ready_event.set()
        for i in range(iterations):
            mark_ledger_item_done(ledger_group, i, float(i))
            update_ledger_in_flight(
                ledger_group,
                item_index=i,
                shots_done=i,
                shots_total=iterations,
            )
            update_ledger_heartbeat(ledger_group)
        f.close()
    except Exception as exc:  # noqa: BLE001 -- relay any failure to the parent
        error_queue.put(repr(exc))


def _reader_hammer(path_str, ready_event, iterations, error_queue):
    """Repeatedly refresh and read the full status in a tight loop,
    concurrently with a hammering writer."""
    try:
        ready_event.wait(timeout=PROCESS_TIMEOUT)
        f, ledger_group = open_swmr_reader(Path(path_str))
        for _ in range(iterations):
            read_swmr_ledger_status(ledger_group)
        f.close()
    except Exception as exc:  # noqa: BLE001 -- relay any failure to the parent
        error_queue.put(repr(exc))


def _report_worker_id_once_and_twice(queue):
    """Report one call's value alongside a same-process repeat, so a caller
    can check both cross-process uniqueness and within-process stability."""
    queue.put((worker_id(), worker_id()))


# --------------------------------------------------------------------------
# Single-process unit tests
# --------------------------------------------------------------------------


class TestSwmrLedgerUnit:
    def test_init_creates_only_requested_fields(self, tmp_path):
        with h5py.File(tmp_path / "ledger.h5", "w") as f:
            ledger_group = init_swmr_ledger(
                f, ["done", "wall_clock_times"], capacity=3
            )
            assert set(ledger_group.keys()) == {"done", "wall_clock_times"}

    def test_init_dataset_shapes_and_dtypes(self, tmp_path):
        with h5py.File(tmp_path / "ledger.h5", "w") as f:
            ledger_group = init_swmr_ledger(f, ALL_FIELDS, capacity=5)

            assert ledger_group["done"].shape == (5,)
            assert ledger_group["done"].maxshape == (None,)
            assert ledger_group["done"].dtype == bool
            assert ledger_group["wall_clock_times"].maxshape == (None,)
            assert np.isnan(ledger_group["wall_clock_times"][0])

            for field in (
                "current_item_index",
                "item_shots_done",
                "item_shots_total",
            ):
                assert ledger_group[field].shape == (1,)
                assert ledger_group[field].chunks == (1,)
                assert ledger_group[field][0] == 0

            assert np.isnan(ledger_group["last_heartbeat"][0])

    def test_init_unknown_field_raises(self, tmp_path):
        with h5py.File(tmp_path / "ledger.h5", "w") as f:
            with pytest.raises(ValueError):
                init_swmr_ledger(f, ["not_a_real_field"])

    def test_mark_item_done_auto_extends_geometric_growth(self, tmp_path):
        with h5py.File(tmp_path / "ledger.h5", "w", libver="latest") as f:
            ledger_group = init_swmr_ledger(
                f, ["done", "wall_clock_times"], capacity=4
            )
            f.swmr_mode = True

            # Within capacity: no growth.
            mark_ledger_item_done(ledger_group, 1, 1.0)
            assert ledger_group["done"].shape == (4,)

            # index=4 >= capacity=4: grows to max(4*2, 5) = 8.
            mark_ledger_item_done(ledger_group, 4, 4.0)
            assert ledger_group["done"].shape == (8,)
            assert ledger_group["wall_clock_times"].shape == (8,)

            # index=20 >= capacity=8: grows to max(8*2, 21) = 21, not just 20.
            mark_ledger_item_done(ledger_group, 20, 20.0)
            assert ledger_group["done"].shape == (21,)

            done = ledger_group["done"][()]
            assert done[1] and done[4] and done[20]
            assert not done[0] and not done[2] and not done[19]
            assert ledger_group["wall_clock_times"][20] == 20.0

    def test_mark_item_done_extends_from_zero_capacity(self, tmp_path):
        with h5py.File(tmp_path / "ledger.h5", "w", libver="latest") as f:
            ledger_group = init_swmr_ledger(
                f, ["done", "wall_clock_times"], capacity=0
            )
            f.swmr_mode = True
            mark_ledger_item_done(ledger_group, 0, 0.5)
            assert ledger_group["done"].shape == (1,)
            assert ledger_group["done"][0]

    def test_update_in_flight_progression(self, tmp_path):
        """Shots update in place while the index is unchanged, then both
        shot fields move together once the index advances."""
        with h5py.File(tmp_path / "ledger.h5", "w", libver="latest") as f:
            ledger_group = init_swmr_ledger(
                f,
                ["current_item_index", "item_shots_done", "item_shots_total"],
            )
            f.swmr_mode = True

            update_ledger_in_flight(
                ledger_group, item_index=0, shots_done=1, shots_total=10
            )
            update_ledger_in_flight(
                ledger_group, item_index=0, shots_done=5, shots_total=10
            )
            assert ledger_group["current_item_index"][0] == 0
            assert ledger_group["item_shots_done"][0] == 5

            update_ledger_in_flight(
                ledger_group, item_index=1, shots_done=2, shots_total=6
            )
            assert ledger_group["current_item_index"][0] == 1
            assert ledger_group["item_shots_done"][0] == 2
            assert ledger_group["item_shots_total"][0] == 6

    def test_update_in_flight_resets_shots_before_index_advances(self):
        """Prove the write ordering directly, via a recording fake ledger
        group standing in for the real h5py datasets: the reset-to-0 write
        (and flush) of item_shots_done must happen strictly before the
        current_item_index write (and flush) -- not just that the final
        values end up correct."""

        class _RecordingDataset:
            def __init__(self, name, value, log):
                self._name = name
                self._value = value
                self._log = log

            def __getitem__(self, idx):
                return self._value

            def __setitem__(self, idx, value):
                self._value = value
                self._log.append((self._name, "set", value))

            def flush(self):
                self._log.append((self._name, "flush", None))

        log = []
        fake_ledger_group = {
            "current_item_index": _RecordingDataset(
                "current_item_index", 0, log
            ),
            "item_shots_done": _RecordingDataset("item_shots_done", 7, log),
            "item_shots_total": _RecordingDataset("item_shots_total", 10, log),
        }

        update_ledger_in_flight(
            fake_ledger_group, item_index=1, shots_done=3, shots_total=9
        )

        reset_set_pos = log.index(("item_shots_done", "set", 0))
        reset_flush_pos = log.index(
            ("item_shots_done", "flush", None), reset_set_pos
        )
        index_set_pos = log.index(("current_item_index", "set", 1))
        assert reset_set_pos < index_set_pos
        assert reset_flush_pos < index_set_pos

    def test_update_heartbeat_explicit_and_default_timestamp(
        self, tmp_path, monkeypatch
    ):
        with h5py.File(tmp_path / "ledger.h5", "w", libver="latest") as f:
            ledger_group = init_swmr_ledger(f, ["last_heartbeat"])
            f.swmr_mode = True

            update_ledger_heartbeat(ledger_group, timestamp=999.5)
            assert ledger_group["last_heartbeat"][0] == 999.5

            monkeypatch.setattr(time, "time", lambda: 42.0)
            update_ledger_heartbeat(ledger_group)
            assert ledger_group["last_heartbeat"][0] == 42.0

    def test_read_status_item_level_snapshot(self, tmp_path):
        with h5py.File(tmp_path / "ledger.h5", "w", libver="latest") as f:
            ledger_group = init_swmr_ledger(f, ALL_FIELDS, capacity=3)
            f.swmr_mode = True
            mark_ledger_item_done(ledger_group, 1, 5.0)
            update_ledger_in_flight(
                ledger_group, item_index=2, shots_done=4, shots_total=8
            )
            update_ledger_heartbeat(ledger_group, timestamp=100.0)

            snapshot = read_swmr_ledger_status(ledger_group, refresh=False)
            assert isinstance(snapshot, SwmrLedgerSnapshot)
            assert snapshot.done.tolist() == [False, True, False]
            assert snapshot.wall_clock_times[1] == 5.0
            assert snapshot.current_item_index == 2
            assert snapshot.item_shots_done == 4
            assert snapshot.item_shots_total == 8
            assert snapshot.last_heartbeat == 100.0

    def test_read_status_shot_level_snapshot_leaves_absent_fields_none(
        self, tmp_path
    ):
        with h5py.File(tmp_path / "ledger.h5", "w", libver="latest") as f:
            ledger_group = init_swmr_ledger(
                f, ["done", "wall_clock_times"], capacity=2
            )
            f.swmr_mode = True
            mark_ledger_item_done(ledger_group, 0, 1.0)

            snapshot = read_swmr_ledger_status(ledger_group, refresh=False)
            assert snapshot.done.tolist() == [True, False]
            assert snapshot.current_item_index is None
            assert snapshot.item_shots_done is None
            assert snapshot.item_shots_total is None
            assert snapshot.last_heartbeat is None

    def test_snapshot_is_frozen(self):
        snapshot = SwmrLedgerSnapshot(current_item_index=1)
        with pytest.raises(Exception):
            snapshot.current_item_index = 2

    def test_worker_id_is_cached_and_pid_gated(self, monkeypatch):
        real_getpid = os.getpid
        monkeypatch.setattr(os, "getpid", lambda: real_getpid())
        first = worker_id()
        assert first == worker_id()
        assert first.split("_")[-2] == str(real_getpid())

        # A "different" PID (simulating a post-fork child that inherited
        # this module's cached state) must never reuse the cached value.
        monkeypatch.setattr(os, "getpid", lambda: real_getpid() + 12345)
        assert worker_id() != first


# --------------------------------------------------------------------------
# Real cross-process concurrency tests
# --------------------------------------------------------------------------


class TestSwmrLedgerConcurrency:
    def test_reader_sees_all_writer_fields_after_refresh(self, tmp_path):
        path = tmp_path / "ledger.h5"
        ready_event = multiprocessing.Event()
        writes_done_event = multiprocessing.Event()
        queue = multiprocessing.Queue()

        writer = multiprocessing.Process(
            target=_writer_full_cycle,
            args=(str(path), ready_event, writes_done_event),
        )
        reader = multiprocessing.Process(
            target=_reader_collect_status,
            args=(str(path), ready_event, writes_done_event, queue),
        )
        try:
            writer.start()
            reader.start()
            result = queue.get(timeout=PROCESS_TIMEOUT)
            writer.join(timeout=PROCESS_TIMEOUT)
            reader.join(timeout=PROCESS_TIMEOUT)
        finally:
            writer.terminate()
            reader.terminate()

        assert writer.exitcode == 0
        assert reader.exitcode == 0
        assert result["done"] == [True, False, True, False]
        assert result["wall_clock_times"][0] == 1.5
        assert result["wall_clock_times"][2] == 2.5
        assert result["current_item_index"] == 1
        assert result["item_shots_done"] == 3
        assert result["item_shots_total"] == 10
        assert result["last_heartbeat"] == 12345.0

    def test_reader_observes_auto_extension_mid_flight(self, tmp_path):
        path = tmp_path / "ledger.h5"
        ready_event = multiprocessing.Event()
        phase1_event = multiprocessing.Event()
        captured_event = multiprocessing.Event()
        go_extend_event = multiprocessing.Event()
        phase2_event = multiprocessing.Event()
        queue = multiprocessing.Queue()

        writer = multiprocessing.Process(
            target=_writer_extend_sequence,
            args=(
                str(path),
                ready_event,
                phase1_event,
                go_extend_event,
                phase2_event,
            ),
        )
        reader = multiprocessing.Process(
            target=_reader_extend_sequence,
            args=(
                str(path),
                ready_event,
                phase1_event,
                captured_event,
                phase2_event,
                queue,
            ),
        )
        try:
            writer.start()
            reader.start()
            # Only let the writer extend once the reader has captured the
            # pre-extension (small-capacity) snapshot, so the two captured
            # states are genuinely distinct rather than a lucky race.
            assert captured_event.wait(timeout=PROCESS_TIMEOUT)
            go_extend_event.set()
            result = queue.get(timeout=PROCESS_TIMEOUT)
            writer.join(timeout=PROCESS_TIMEOUT)
            reader.join(timeout=PROCESS_TIMEOUT)
        finally:
            writer.terminate()
            reader.terminate()

        assert writer.exitcode == 0
        assert reader.exitcode == 0
        assert result["shape1"] == 2
        assert result["done1"] == [True, True]
        # max(2 * 2, 5 + 1) = 6
        assert result["shape2"] == 6
        assert result["done2"] == [True, True, False, False, False, True]
        assert result["time5"] == 0.5

    def test_no_lock_errors_under_concurrent_access(self, tmp_path):
        path = tmp_path / "ledger.h5"
        ready_event = multiprocessing.Event()
        error_queue = multiprocessing.Queue()
        iterations = 50

        writer = multiprocessing.Process(
            target=_writer_hammer,
            args=(str(path), ready_event, iterations, error_queue),
        )
        reader = multiprocessing.Process(
            target=_reader_hammer,
            args=(str(path), ready_event, iterations, error_queue),
        )
        try:
            writer.start()
            reader.start()
            writer.join(timeout=PROCESS_TIMEOUT)
            reader.join(timeout=PROCESS_TIMEOUT)
        finally:
            writer.terminate()
            reader.terminate()

        assert writer.exitcode == 0
        assert reader.exitcode == 0
        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get())
        assert errors == []

    def test_worker_id_unique_across_processes_stable_within_one(self):
        queue = multiprocessing.Queue()
        p1 = multiprocessing.Process(
            target=_report_worker_id_once_and_twice, args=(queue,)
        )
        p2 = multiprocessing.Process(
            target=_report_worker_id_once_and_twice, args=(queue,)
        )
        try:
            p1.start()
            p2.start()
            p1_first, p1_second = queue.get(timeout=PROCESS_TIMEOUT)
            p2_first, _ = queue.get(timeout=PROCESS_TIMEOUT)
            p1.join(timeout=PROCESS_TIMEOUT)
            p2.join(timeout=PROCESS_TIMEOUT)
        finally:
            p1.terminate()
            p2.terminate()

        assert p1.exitcode == 0
        assert p2.exitcode == 0
        assert p1_first == p1_second
        assert p1_first != p2_first
