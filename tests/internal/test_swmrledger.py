"""Tests for loqs/internal/swmrledger.py's SWMR progress ledger primitives,
and for the `worker_id()` hardening in loqs/internal/__init__.py."""

import faulthandler
import multiprocessing
import os
import queue as queue_module
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from loqs.internal import worker_id
from loqs.internal.swmrledger import (
    SwmrLedgerSnapshot,
    held_swmr_writer,
    init_swmr_ledger,
    is_swmr_ledger_file,
    mark_ledger_item_done,
    open_swmr_reader,
    open_swmr_writer,
    read_swmr_ledger_done_union,
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

PROCESS_TIMEOUT = 60


# --------------------------------------------------------------------------
# Module-level process targets (must be picklable for multiprocessing.Process)
# --------------------------------------------------------------------------


def _writer_full_cycle(path_str, ready_event, writes_done_event, error_queue):
    """Write every field once, then keep the file open briefly so a reader
    definitely has a chance to observe it live."""
    faulthandler.enable()
    operation = "open"
    try:
        f, ledger_group = open_swmr_writer(
            Path(path_str), ALL_FIELDS, capacity=4
        )
        ready_event.set()
        operation = "mark"
        mark_ledger_item_done(ledger_group, 0, 1.5)
        mark_ledger_item_done(ledger_group, 2, 2.5)
        operation = "in_flight"
        update_ledger_in_flight(
            ledger_group, item_index=1, shots_done=3, shots_total=10
        )
        operation = "heartbeat"
        update_ledger_heartbeat(ledger_group, timestamp=12345.0)
        writes_done_event.set()
        time.sleep(0.2)
        f.close()
    except Exception as exc:  # noqa: BLE001 -- relay any failure to the parent
        error_queue.put(("writer", operation, repr(exc)))


def _reader_collect_status(
    path_str, ready_event, writes_done_event, queue, error_queue
):
    """Wait for the writer to finish, then relay a full snapshot back."""
    faulthandler.enable()
    operation = "open"
    try:
        ready_event.wait(timeout=PROCESS_TIMEOUT)
        f, ledger_group = open_swmr_reader(Path(path_str))
        writes_done_event.wait(timeout=PROCESS_TIMEOUT)
        operation = "read_status"
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
    except Exception as exc:  # noqa: BLE001 -- relay any failure to the parent
        error_queue.put(("reader", operation, repr(exc)))


def _writer_extend_sequence(
    path_str,
    ready_event,
    phase1_event,
    go_extend_event,
    phase2_event,
    error_queue,
):
    """Write two items within the initial small capacity, then (once told
    to go) write a far-out index that forces auto-extension."""
    faulthandler.enable()
    operation = "open"
    try:
        f, ledger_group = open_swmr_writer(
            Path(path_str),
            ["done", "wall_clock_times"],
            capacity=2,
            chunk_size=2,
        )
        ready_event.set()
        operation = "mark"
        mark_ledger_item_done(ledger_group, 0, 0.1)
        mark_ledger_item_done(ledger_group, 1, 0.2)
        phase1_event.set()
        go_extend_event.wait(timeout=PROCESS_TIMEOUT)
        mark_ledger_item_done(ledger_group, 5, 0.5)
        phase2_event.set()
        time.sleep(0.2)
        f.close()
    except Exception as exc:  # noqa: BLE001 -- relay any failure to the parent
        error_queue.put(("writer", operation, repr(exc)))


def _reader_extend_sequence(
    path_str,
    ready_event,
    phase1_event,
    captured_event,
    phase2_event,
    queue,
    error_queue,
):
    """Capture a pre-extension snapshot, signal that it was captured, then
    capture a post-extension snapshot once the writer has grown the ledger."""
    faulthandler.enable()
    operation = "open"
    try:
        ready_event.wait(timeout=PROCESS_TIMEOUT)
        f, ledger_group = open_swmr_reader(Path(path_str))
        phase1_event.wait(timeout=PROCESS_TIMEOUT)
        operation = "refresh"
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
    except Exception as exc:  # noqa: BLE001 -- relay any failure to the parent
        error_queue.put(("reader", operation, repr(exc)))


def _writer_hammer(path_str, ready_event, iterations, capacity, error_queue):
    """Repeatedly write every in-flight field in a tight loop, to stress
    concurrent access alongside a hammering reader."""
    faulthandler.enable()
    operation = "open"
    try:
        f, ledger_group = open_swmr_writer(
            Path(path_str), ALL_FIELDS, capacity=capacity
        )
        ready_event.set()
        for i in range(iterations):
            operation = "mark"
            mark_ledger_item_done(ledger_group, i, float(i))
            operation = "in_flight"
            update_ledger_in_flight(
                ledger_group,
                item_index=i,
                shots_done=i,
                shots_total=iterations,
            )
            operation = "heartbeat"
            update_ledger_heartbeat(ledger_group)
        f.close()
    except Exception as exc:  # noqa: BLE001 -- relay any failure to the parent
        error_queue.put(("writer", operation, repr(exc)))


def _reader_hammer(path_str, ready_event, iterations, error_queue):
    """Repeatedly refresh and read the full status in a tight loop,
    concurrently with a hammering writer."""
    faulthandler.enable()
    operation = "open"
    try:
        ready_event.wait(timeout=PROCESS_TIMEOUT)
        f, ledger_group = open_swmr_reader(Path(path_str))
        operation = "read_status"
        for _ in range(iterations):
            read_swmr_ledger_status(ledger_group)
        f.close()
    except Exception as exc:  # noqa: BLE001 -- relay any failure to the parent
        error_queue.put(("reader", operation, repr(exc)))


def _reader_poll_until_stopped(
    path_str, role, ready_event, stop_event, error_queue
):
    """Open, read and close the ledger in a loop until told to stop, as a
    progress poller does while a writer repeatedly reopens the ledger.

    A failed round is relayed and polling carries on, so the writer stays
    under contention; only the first few failures are relayed in full, to
    keep the queue's pipe from filling, plus a count of the rest."""
    faulthandler.enable()
    max_relayed = 10
    failures = 0
    ready_event.set()
    while not stop_event.is_set():
        operation = "open"
        try:
            f, ledger_group = open_swmr_reader(Path(path_str))
            try:
                operation = "read_status"
                read_swmr_ledger_status(ledger_group)
            finally:
                f.close()
        except Exception as exc:  # noqa: BLE001 -- relay to the parent
            failures += 1
            if failures <= max_relayed:
                error_queue.put((role, operation, repr(exc)))
    if failures > max_relayed:
        error_queue.put(
            (role, "open/read_status", f"{failures - max_relayed} more")
        )


def _join_and_report(
    processes: dict[str, multiprocessing.Process], error_queue
) -> str:
    """Join every child (terminating any still alive after the timeout) and
    describe each child's exit code and every relayed error. Negative or
    large exit codes are also shown in hex (0xC0000005 is an access
    violation on Windows)."""
    for process in processes.values():
        process.join(timeout=PROCESS_TIMEOUT)
    lines = []
    for role, process in processes.items():
        if process.is_alive():
            process.terminate()
            process.join(timeout=PROCESS_TIMEOUT)
            lines.append(f"{role}: still alive after {PROCESS_TIMEOUT}s")
        code = process.exitcode
        text = f"{role}: exit code {code}"
        if code is not None and (code < 0 or code > 255):
            text += f" (0x{code & 0xFFFFFFFF:08X})"
        lines.append(text)
    while True:
        try:
            role, operation, error = error_queue.get(timeout=0.1)
        except queue_module.Empty:
            break
        lines.append(f"{role} failed in {operation!r}: {error}")
    return "\n".join(lines)


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


def _open_hdf5_file_names() -> set[str]:
    """Return the names of every HDF5 file this process holds open."""
    names = set()
    for fid in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE):
        names.add(os.fsdecode(h5py.h5f.get_name(fid)))
    return names


def test_open_swmr_writer_renames_new_ledger_only_when_closed(
    tmp_path, monkeypatch
):
    """A new ledger's temporary file is renamed into place only once no
    HDF5 handle holds it, since Windows can't rename an open file."""
    path = tmp_path / "worker_ledger.h5"
    real_replace = os.replace
    open_at_replace = []

    def recording_replace(src, dst):
        src_resolved = Path(src).resolve()
        open_at_replace.append(
            any(
                Path(name).resolve() == src_resolved
                for name in _open_hdf5_file_names()
            )
        )
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)

    f, ledger_group = open_swmr_writer(path, ALL_FIELDS)
    try:
        assert open_at_replace == [False]
        assert f.swmr_mode is True
        assert path.exists()
        assert "done" in ledger_group
        assert list(tmp_path.glob("*.tmp")) == []
    finally:
        f.close()


def test_held_ledger_is_readable_in_same_process(tmp_path):
    """While this process holds a ledger's writer, its own reads of that
    ledger still succeed, so it isn't mistaken for a legacy file."""
    path = tmp_path / "worker_ledger.h5"
    with held_swmr_writer(path, ALL_FIELDS) as group:
        mark_ledger_item_done(group, 3, 0.5)
        assert is_swmr_ledger_file(path) is True
        assert read_swmr_ledger_done_union(tmp_path, path.name) == {3}


def test_open_swmr_writer_cleans_up_tmp_on_failure(tmp_path):
    """A failure before a new ledger reaches its final path leaves neither
    the final path nor its temporary file behind."""
    path = tmp_path / "worker_ledger.h5"
    with pytest.raises(ValueError):
        open_swmr_writer(path, ["not_a_real_field"])
    assert not path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_open_swmr_writer_reopens_existing_ledger_in_place(tmp_path):
    """Reopening an existing ledger keeps its datasets and `done` bits."""
    path = tmp_path / "worker_ledger.h5"
    f, ledger_group = open_swmr_writer(path, ALL_FIELDS, capacity=4)
    mark_ledger_item_done(ledger_group, 2, 0.25)
    f.close()

    f, ledger_group = open_swmr_writer(path, ALL_FIELDS)
    try:
        assert f.swmr_mode is True
        assert set(ledger_group) == set(ALL_FIELDS)
        snapshot = read_swmr_ledger_status(ledger_group)
        assert snapshot.done is not None
        assert np.flatnonzero(snapshot.done).tolist() == [2]
        assert snapshot.wall_clock_times is not None
        assert snapshot.wall_clock_times[2] == 0.25
        assert list(tmp_path.glob("*.tmp")) == []
    finally:
        f.close()


# --------------------------------------------------------------------------
# Real cross-process concurrency tests
# --------------------------------------------------------------------------


class TestSwmrLedgerConcurrency:
    def test_reader_sees_all_writer_fields_after_refresh(self, tmp_path):
        path = tmp_path / "ledger.h5"
        ready_event = multiprocessing.Event()
        writes_done_event = multiprocessing.Event()
        queue = multiprocessing.Queue()
        error_queue = multiprocessing.Queue()

        processes = {
            "writer": multiprocessing.Process(
                target=_writer_full_cycle,
                args=(str(path), ready_event, writes_done_event, error_queue),
            ),
            "reader": multiprocessing.Process(
                target=_reader_collect_status,
                args=(
                    str(path),
                    ready_event,
                    writes_done_event,
                    queue,
                    error_queue,
                ),
            ),
        }
        result = None
        try:
            for process in processes.values():
                process.start()
            try:
                result = queue.get(timeout=PROCESS_TIMEOUT)
            except queue_module.Empty:
                pass
        finally:
            report = _join_and_report(processes, error_queue)

        if result is None:
            pytest.fail(report)
        for process in processes.values():
            assert process.exitcode == 0, report
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
        error_queue = multiprocessing.Queue()

        processes = {
            "writer": multiprocessing.Process(
                target=_writer_extend_sequence,
                args=(
                    str(path),
                    ready_event,
                    phase1_event,
                    go_extend_event,
                    phase2_event,
                    error_queue,
                ),
            ),
            "reader": multiprocessing.Process(
                target=_reader_extend_sequence,
                args=(
                    str(path),
                    ready_event,
                    phase1_event,
                    captured_event,
                    phase2_event,
                    queue,
                    error_queue,
                ),
            ),
        }
        result = None
        captured = False
        try:
            for process in processes.values():
                process.start()
            # Only let the writer extend once the reader has captured the
            # pre-extension (small-capacity) snapshot, so the two captured
            # states are genuinely distinct rather than a lucky race.
            captured = captured_event.wait(timeout=PROCESS_TIMEOUT)
            go_extend_event.set()
            if captured:
                try:
                    result = queue.get(timeout=PROCESS_TIMEOUT)
                except queue_module.Empty:
                    pass
        finally:
            report = _join_and_report(processes, error_queue)

        assert captured, report
        if result is None:
            pytest.fail(report)
        for process in processes.values():
            assert process.exitcode == 0, report
        assert result["shape1"] == 2
        assert result["done1"] == [True, True]
        # max(2 * 2, 5 + 1) = 6
        assert result["shape2"] == 6
        assert result["done2"] == [True, True, False, False, False, True]
        assert result["time5"] == 0.5

    @pytest.mark.parametrize(
        "capacity", [0, 50], ids=["auto_extend", "presized"]
    )
    def test_no_lock_errors_under_concurrent_access(self, tmp_path, capacity):
        """Run once with resizing under SWMR and once without, so a failure
        shows whether only the resize path is at fault."""
        path = tmp_path / "ledger.h5"
        ready_event = multiprocessing.Event()
        error_queue = multiprocessing.Queue()
        iterations = 50

        processes = {
            "writer": multiprocessing.Process(
                target=_writer_hammer,
                args=(
                    str(path),
                    ready_event,
                    iterations,
                    capacity,
                    error_queue,
                ),
            ),
            "reader": multiprocessing.Process(
                target=_reader_hammer,
                args=(str(path), ready_event, iterations, error_queue),
            ),
        }
        try:
            for process in processes.values():
                process.start()
        finally:
            report = _join_and_report(processes, error_queue)

        for process in processes.values():
            assert process.exitcode == 0, report
        assert "failed in" not in report, report

    def test_writer_reopens_existing_ledger_under_concurrent_readers(
        self, tmp_path
    ):
        """A writer reopening an existing ledger must never fail because a
        poller has it open, as happens when every task opens and closes its
        own ledger handle."""
        path = tmp_path / "ledger.h5"
        fields = ["done", "wall_clock_times"]
        rounds = 300
        f, _ = open_swmr_writer(path, fields)
        f.close()

        stop_event = multiprocessing.Event()
        error_queue = multiprocessing.Queue()
        ready_events = {
            role: multiprocessing.Event() for role in ("reader1", "reader2")
        }
        processes = {
            role: multiprocessing.Process(
                target=_reader_poll_until_stopped,
                args=(str(path), role, ready, stop_event, error_queue),
            )
            for role, ready in ready_events.items()
        }
        writer_failures = []
        try:
            for process in processes.values():
                process.start()
            for ready in ready_events.values():
                assert ready.wait(timeout=PROCESS_TIMEOUT)
            for i in range(rounds):
                try:
                    f, ledger_group = open_swmr_writer(path, fields)
                    try:
                        mark_ledger_item_done(ledger_group, i, 0.1)
                    finally:
                        f.close()
                except Exception as exc:  # noqa: BLE001 -- collect failures
                    writer_failures.append(f"round {i}: {exc!r}")
        finally:
            stop_event.set()
            report = _join_and_report(processes, error_queue)

        assert writer_failures == [], (
            f"{len(writer_failures)} of {rounds} reopens failed, first: "
            f"{writer_failures[0]}\n{report}"
        )
        assert "failed in" not in report, report
        for process in processes.values():
            assert process.exitcode == 0, report
        f, ledger_group = open_swmr_reader(path)
        try:
            assert int(np.count_nonzero(ledger_group["done"][()])) == rounds
        finally:
            f.close()

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
