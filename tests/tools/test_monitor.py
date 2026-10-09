"""Tests for `loqs.tools.monitor` state aggregation."""

import os
import shutil
import socket

import h5py
import pytest
from _shared_checkpoint_test_helpers import write_item_ledger

from loqs.internal.swmrledger import open_swmr_writer
from loqs.tools import multiprogramrunner
from loqs.tools.monitor import (
    MonitorTracker,
    WorkerState,
    read_snapshot,
)
from loqs.tools.monitor import state as monitor_state

WALL = 1_000_000.0
STALE = 100.0


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _tracker(directory, clock=None):
    return MonitorTracker(
        directory,
        stale_after=STALE,
        clock=clock or FakeClock(),
        wall_clock=lambda: WALL,
    )


def _by_pid(snapshot):
    return {w.pid: w for w in snapshot.workers}


def test_happy_path_two_running_workers(tmp_path):
    write_item_ledger(
        tmp_path,
        pid=1000,
        current_item=3,
        shots_done=40,
        shots_total=100,
        done=(0, 1),
        heartbeat=WALL - 5,
    )
    write_item_ledger(
        tmp_path,
        pid=1001,
        suffix="1111aaaa",
        current_item=7,
        shots_done=10,
        shots_total=50,
        done=(2,),
        heartbeat=WALL - 20,
    )
    snap = _tracker(tmp_path).poll()
    assert snap.directory_exists
    assert not snap.skipped
    w0, w1 = snap.workers
    assert (w0.host, w0.pid, w0.worker_id) == (
        "nodeA",
        1000,
        "nodeA_1000_0000abcd",
    )
    assert w0.state is WorkerState.RUNNING
    assert (w0.current_item, w0.shots_done, w0.shots_total) == (3, 40, 100)
    assert w0.items_done == 2
    assert w0.heartbeat_age == pytest.approx(5)
    assert w0.since_change is None
    assert (w1.pid, w1.current_item, w1.items_done) == (1001, 7, 1)
    assert w1.heartbeat_age == pytest.approx(20)
    t = snap.totals
    assert t.state_counts[WorkerState.RUNNING] == 2
    assert t.state_counts[WorkerState.STALE] == 0
    assert t.hosts == ("nodeA",)
    assert t.items_done == 3
    assert (t.shots_done, t.shots_total) == (50, 150)


def test_idle_worker_never_stale(tmp_path):
    # Push both staleness rules past STALE: an old heartbeat for the first
    # poll, then one observed change followed by a long monitor-clock gap.
    clock = FakeClock()
    write_item_ledger(
        tmp_path,
        current_item=2,
        done=(0, 1, 2),
        heartbeat=WALL - 10 * STALE,
    )
    tracker = _tracker(tmp_path, clock)
    w = tracker.poll().workers[0]
    assert w.heartbeat_age >= STALE
    assert w.state is WorkerState.IDLE
    write_item_ledger(tmp_path, heartbeat=WALL - 1)
    clock.t += 1
    w = tracker.poll().workers[0]
    assert w.since_change is not None
    assert w.state is WorkerState.IDLE
    clock.t += 10 * STALE
    w = tracker.poll().workers[0]
    assert w.since_change >= STALE
    assert w.state is WorkerState.IDLE
    assert w.state is WorkerState.IDLE
    assert (w.current_item, w.shots_done, w.shots_total) == (None, None, None)
    assert w.items_done == 3


def test_starting_worker_is_not_item_zero(tmp_path):
    write_item_ledger(tmp_path)
    clock = FakeClock()
    tracker = _tracker(tmp_path, clock)
    w = tracker.poll().workers[0]
    assert w.state is WorkerState.STARTING
    assert w.current_item is None
    assert w.heartbeat_age is None
    clock.t += 10 * STALE
    assert tracker.poll().workers[0].state is WorkerState.STARTING


def test_crash_and_respawn_only_crashed_is_stale(tmp_path):
    clock = FakeClock()
    tracker = _tracker(tmp_path, clock)
    crashed = dict(pid=1000, suffix="aaaaaaaa", current_item=1)
    respawn = dict(pid=1001, suffix="bbbbbbbb", current_item=2)
    shots = 0

    def advance(spec, n):
        write_item_ledger(
            tmp_path,
            shots_done=n,
            shots_total=1000,
            heartbeat=WALL - 10,
            **spec,
        )

    advance(crashed, 0)
    advance(respawn, 0)
    tracker.poll()
    # The crashed worker advances once here and never again.
    clock.t = 10.0
    advance(crashed, 5)
    shots += 1
    advance(respawn, shots)
    snap = tracker.poll()
    assert {w.state for w in snap.workers} == {WorkerState.RUNNING}

    def step(to):
        nonlocal shots
        clock.t = to
        shots += 1
        advance(respawn, shots)
        return _by_pid(tracker.poll())

    ws = step(10.0 + STALE - 1)
    assert ws[1000].state is WorkerState.RUNNING
    ws = step(10.0 + STALE)
    assert ws[1000].state is WorkerState.STALE
    assert ws[1000].current_item == 1
    assert ws[1000].since_change == pytest.approx(STALE)
    assert ws[1001].state is WorkerState.RUNNING
    assert ws[1001].since_change == pytest.approx(0)
    ws = step(10.0 + 2 * STALE)
    assert ws[1000].state is WorkerState.STALE
    assert ws[1001].state is WorkerState.RUNNING


def test_heartbeat_fallback(tmp_path):
    clock = FakeClock()
    tracker = _tracker(tmp_path, clock)
    write_item_ledger(
        tmp_path,
        pid=1,
        suffix="aaaaaaaa",
        current_item=0,
        shots_done=1,
        shots_total=10,
        heartbeat=WALL - 5 * STALE,
    )
    write_item_ledger(
        tmp_path,
        pid=2,
        suffix="bbbbbbbb",
        current_item=0,
        shots_done=1,
        shots_total=10,
        heartbeat=WALL - 1,
    )
    ws = _by_pid(tracker.poll())
    assert ws[1].state is WorkerState.STALE
    assert ws[1].heartbeat_age > STALE
    assert ws[2].state is WorkerState.RUNNING

    one_off = _by_pid(
        read_snapshot(tmp_path, stale_after=STALE, wall_clock=lambda: WALL)
    )
    assert one_off[1].state is WorkerState.STALE
    assert one_off[2].state is WorkerState.RUNNING

    clock.t += 1.0
    ws = _by_pid(tracker.poll())
    assert ws[1].state is WorkerState.STALE
    assert ws[1].since_change is None

    clock.t += 1.0
    write_item_ledger(
        tmp_path,
        pid=1,
        suffix="aaaaaaaa",
        current_item=0,
        shots_done=2,
        shots_total=10,
        heartbeat=WALL - 1,
    )
    ws = _by_pid(tracker.poll())
    assert ws[1].state is WorkerState.RUNNING
    assert ws[1].since_change == pytest.approx(0)


def test_bad_files_are_skipped_with_reasons(tmp_path, monkeypatch):
    good = write_item_ledger(
        tmp_path, current_item=0, shots_total=5, heartbeat=WALL - 1
    )
    shutil.copy(good, tmp_path / (good.name + ".tmp"))
    garbage = tmp_path / "worker_nodeB_2_0000abcd_runner.h5"
    garbage.write_bytes(b"not an hdf5 file at all" * 100)
    legacy = tmp_path / "worker_nodeC_3_0000abcd_runner.h5"
    with h5py.File(legacy, "w", libver="latest") as f:
        f.attrs["current_item_index"] = 4
    missing = tmp_path / "worker_nodeD_4_0000abcd_runner.h5"
    real_list = monitor_state._list_worker_files
    monkeypatch.setattr(
        monitor_state,
        "_list_worker_files",
        lambda d: [*real_list(d), missing],
    )
    snap = _tracker(tmp_path).poll()
    assert [w.pid for w in snap.workers] == [1000]
    reasons = {s.path.name: s.reason for s in snap.skipped}
    assert reasons == {
        garbage.name: "unreadable",
        legacy.name: "legacy",
        missing.name: "unreadable",
    }
    assert snap.notes


def test_unrecognized_name_is_skipped_unread(tmp_path):
    bad = tmp_path / "worker_nodeA_notapid_0000abcd_runner.h5"
    bad.write_bytes(b"x")
    snap = _tracker(tmp_path).poll()
    assert [(s.path.name, s.reason) for s in snap.skipped] == [
        (bad.name, "unrecognized")
    ]
    assert snap.notes


def test_partial_visibility_and_empty_directories(tmp_path):
    for host, pid in (("nodeA", 1), ("node_b_2", 2), ("nodeC", 3)):
        write_item_ledger(
            tmp_path,
            host=host,
            pid=pid,
            current_item=pid,
            shots_total=5,
            heartbeat=WALL - 1,
        )
    tracker = _tracker(tmp_path)
    snap = tracker.poll()
    assert snap.totals.hosts == ("nodeA", "nodeC", "node_b_2")
    (tmp_path / "worker_node_b_2_2_0000abcd_runner.h5").unlink()
    snap = tracker.poll()
    assert [w.pid for w in snap.workers] == [1, 3]
    assert snap.totals.hosts == ("nodeA", "nodeC")

    empty = tmp_path / "empty"
    empty.mkdir()
    snap = _tracker(empty).poll()
    assert snap.workers == () and snap.directory_exists and snap.notes
    snap = _tracker(tmp_path / "nope").poll()
    assert snap.workers == () and not snap.directory_exists and snap.notes


def test_worker_name_matches_driver_ledger_path(tmp_path):
    path = multiprogramrunner._item_ledger_path(tmp_path)
    f, _ = open_swmr_writer(path, multiprogramrunner._ITEM_LEDGER_FIELDS)
    f.close()
    snap = read_snapshot(tmp_path)
    assert len(snap.workers) == 1
    assert snap.workers[0].host == socket.gethostname()
    assert snap.workers[0].pid == os.getpid()
