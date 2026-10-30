"""Tests for loqs/internal/swmrledger.py's SWMR progress ledger primitives,
and for the `worker_id()` hardening in loqs/internal/__init__.py."""

import faulthandler
import multiprocessing
import os
import queue as queue_module
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from loqs.internal import worker_id
from loqs.internal.swmrledger import (
    _recover_stale_ledger,
    _release_lock,
    _try_lock_exclusive,
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


def _writer_extend_sequence(
    path_str,
    ready_event,
    phase1_event,
    go_extend_event,
    phase2_event,
    error_queue,
):
    """Write every field within the initial capacity, then (once told to go)
    write a far-out index that forces auto-extension."""
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
        phase1_event.set()
        go_extend_event.wait(timeout=PROCESS_TIMEOUT)
        operation = "extend"
        mark_ledger_item_done(ledger_group, 9, 0.9)
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
        operation = "read_status"
        snapshot = read_swmr_ledger_status(ledger_group)
        status = {
            "done": snapshot.done.tolist(),
            "wall_clock_times": snapshot.wall_clock_times.tolist(),
            "current_item_index": snapshot.current_item_index,
            "item_shots_done": snapshot.item_shots_done,
            "item_shots_total": snapshot.item_shots_total,
            "last_heartbeat": snapshot.last_heartbeat,
        }
        shape1 = ledger_group["done"].shape[0]
        captured_event.set()
        operation = "refresh"
        phase2_event.wait(timeout=PROCESS_TIMEOUT)
        refresh_swmr_ledger(ledger_group)
        queue.put(
            {
                "status": status,
                "shape1": shape1,
                "shape2": ledger_group["done"].shape[0],
                "done2": ledger_group["done"][()].tolist(),
                "time9": ledger_group["wall_clock_times"][9],
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


# --------------------------------------------------------------------------
# Single-process unit tests
# --------------------------------------------------------------------------


class TestSwmrLedgerUnit:
    def test_init_dataset_shapes_and_dtypes(self, tmp_path):
        with h5py.File(tmp_path / "ledger.h5", "w") as f:
            ledger_group = init_swmr_ledger(f, ALL_FIELDS, capacity=5)

            assert ledger_group["done"].dtype == bool
            assert ledger_group["done"].maxshape == (None,)
            assert ledger_group["wall_clock_times"].maxshape == (None,)
            assert np.isnan(ledger_group["wall_clock_times"][0])

            for field in (
                "current_item_index",
                "item_shots_done",
                "item_shots_total",
            ):
                assert ledger_group[field][0] == 0

            assert np.isnan(ledger_group["last_heartbeat"][0])

    def test_shot_ledger_marks_and_grows(self, tmp_path):
        """Marks land at the right index from capacity 0, growth is
        geometric, and a shot-level snapshot leaves item fields None."""
        with h5py.File(tmp_path / "zero.h5", "w", libver="latest") as f:
            ledger_group = init_swmr_ledger(
                f, ["done", "wall_clock_times"], capacity=0
            )
            f.swmr_mode = True
            mark_ledger_item_done(ledger_group, 0, 0.5)
            assert ledger_group["done"].shape == (1,)
            assert ledger_group["done"][0]

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

            snapshot = read_swmr_ledger_status(ledger_group, refresh=False)
            assert snapshot.done.tolist() == done.tolist()
            assert snapshot.current_item_index is None
            assert snapshot.item_shots_done is None
            assert snapshot.item_shots_total is None
            assert snapshot.last_heartbeat is None

    def test_item_level_snapshot(self, tmp_path, monkeypatch):
        """The in-flight progression and the default-timestamp heartbeat
        read back through the status snapshot."""
        with h5py.File(tmp_path / "ledger.h5", "w", libver="latest") as f:
            ledger_group = init_swmr_ledger(f, ALL_FIELDS, capacity=3)
            f.swmr_mode = True
            mark_ledger_item_done(ledger_group, 1, 5.0)

            # Shots update in place while the index is unchanged, then
            # both shot fields move together once the index advances.
            update_ledger_in_flight(
                ledger_group, item_index=0, shots_done=1, shots_total=10
            )
            update_ledger_in_flight(
                ledger_group, item_index=0, shots_done=5, shots_total=10
            )
            snapshot = read_swmr_ledger_status(ledger_group, refresh=False)
            assert snapshot.current_item_index == 0
            assert snapshot.item_shots_done == 5

            update_ledger_in_flight(
                ledger_group, item_index=2, shots_done=4, shots_total=8
            )
            monkeypatch.setattr(time, "time", lambda: 42.0)
            update_ledger_heartbeat(ledger_group)

            snapshot = read_swmr_ledger_status(ledger_group, refresh=False)
            assert isinstance(snapshot, SwmrLedgerSnapshot)
            assert snapshot.done.tolist() == [False, True, False]
            assert snapshot.wall_clock_times[1] == 5.0
            assert snapshot.current_item_index == 2
            assert snapshot.item_shots_done == 4
            assert snapshot.item_shots_total == 8
            assert snapshot.last_heartbeat == 42.0

    def test_worker_id_is_cached_and_pid_gated(self, monkeypatch):
        real_getpid = os.getpid
        monkeypatch.setattr(os, "getpid", lambda: real_getpid())
        first = worker_id()
        assert first == worker_id()

        # A "different" PID (simulating a post-fork child that inherited
        # this module's cached state) must never reuse the cached value.
        monkeypatch.setattr(os, "getpid", lambda: real_getpid() + 12345)
        assert worker_id() != first

    def test_reopen_after_killed_writer_clears_stale_flag(self, tmp_path):
        """A ledger whose writer was killed while it held the file reopens
        with its done flags and fill values intact and holds the writer
        lock, and no `.tmp` is left."""
        path = tmp_path / "ledger.h5"
        _stale_ledger_copy(tmp_path, path)

        f, group = open_swmr_writer(path, ["done", "wall_clock_times"])
        try:
            assert _try_lock_exclusive(path) is None
            assert bool(group["done"][3]) is True
            mark_ledger_item_done(group, 20, 2.5)
            done = group["done"][()]
            assert done[20] and done[3]
            assert not done[4:20].any()
            assert np.isnan(group["wall_clock_times"][10])
        finally:
            f.close()

        with h5py.File(path, "r") as plain:
            assert bool(plain["swmr_ledger"]["done"][3]) is True
        assert list(tmp_path.glob("*.tmp")) == []

    def test_held_stale_ledger_is_not_rebuilt(self, tmp_path):
        """Recovery refuses, with the original error, a ledger this process
        still holds open, and leaves the file as it was. Otherwise the open
        handle would keep writing to the replaced file and lose its shots."""
        path = tmp_path / "ledger.h5"
        _stale_ledger_copy(tmp_path, path)
        before = path.read_bytes()
        original = OSError("open for write")

        f, _group = open_swmr_reader(path)
        try:
            with pytest.raises(OSError) as excinfo:
                _recover_stale_ledger(path, original)
        finally:
            f.close()

        assert excinfo.value is original
        assert path.read_bytes() == before
        assert list(tmp_path.glob("*.tmp")) == []

    def test_live_writer_ledger_is_left_alone(self, tmp_path):
        """A second writer on a ledger a live process holds fails, and
        neither replaces the file nor leaves a `.tmp`."""
        path = tmp_path / "ledger.h5"
        _stale_ledger_copy(tmp_path, path)

        # The lock belongs to its descriptor, so this one stands in for a
        # live writer's lock and blocks recovery in this process.
        lock_fd = _try_lock_exclusive(path)
        assert lock_fd is not None
        try:
            inode = os.stat(path).st_ino
            with pytest.raises(OSError) as excinfo:
                open_swmr_writer(path, ["done", "wall_clock_times"])
            assert excinfo.value.__cause__ is None
            assert os.stat(path).st_ino == inode
            assert list(tmp_path.glob("*.tmp")) == []
            f, group = open_swmr_reader(path)
            try:
                assert bool(group["done"][3]) is True
            finally:
                f.close()
        finally:
            _release_lock(lock_fd)


def _stale_ledger_copy(tmp_path, target):
    """Leave `target` as a killed writer's ledger: a byte copy of a
    still-open ledger keeps HDF5's "open for write" flag, with item 3 done."""
    scratch = tmp_path / "scratch_ledger.h5"
    f, group = open_swmr_writer(scratch, ["done", "wall_clock_times"])
    try:
        mark_ledger_item_done(group, 3, 1.5)
        f.flush()
        shutil.copyfile(scratch, target)
    finally:
        f.close()
    with pytest.raises(OSError):
        h5py.File(target, "r+")


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

    f, _ = open_swmr_writer(path, ALL_FIELDS)
    try:
        assert open_at_replace == [False]
    finally:
        f.close()


def test_held_ledger_is_readable_in_same_process(tmp_path):
    """While this process holds a ledger's writer, its own reads of that
    ledger still succeed, so it isn't mistaken for a legacy file."""
    path = tmp_path / "worker_ledger.h5"
    with held_swmr_writer(path, ALL_FIELDS) as group:
        mark_ledger_item_done(group, 3, 0.5)
        assert is_swmr_ledger_file(path) is True
        assert read_swmr_ledger_done_union([path]) == {3}


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
        snapshot = read_swmr_ledger_status(ledger_group)
        assert snapshot.done is not None
        assert np.flatnonzero(snapshot.done).tolist() == [2]
        assert snapshot.wall_clock_times is not None
        assert snapshot.wall_clock_times[2] == 0.25
    finally:
        f.close()


# --------------------------------------------------------------------------
# Real cross-process concurrency tests
# --------------------------------------------------------------------------


class TestSwmrLedgerConcurrency:
    def test_reader_sees_ledger_fields_and_growth_across_processes(
        self, tmp_path
    ):
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
        status = result["status"]
        assert result["shape1"] == 4
        assert status["done"] == [True, False, True, False]
        assert status["wall_clock_times"][0] == 1.5
        assert status["wall_clock_times"][2] == 2.5
        assert status["current_item_index"] == 1
        assert status["item_shots_done"] == 3
        assert status["item_shots_total"] == 10
        assert status["last_heartbeat"] == 12345.0
        # max(2 * 4, 9 + 1) = 10
        assert result["shape2"] == 10
        assert [i for i, d in enumerate(result["done2"]) if d] == [0, 2, 9]
        assert result["time9"] == 0.9

    def test_no_lock_errors_under_concurrent_access(self, tmp_path):
        """A refreshing reader beside a growing writer hits no lock error."""
        capacity = 0
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
