"""Tests for the `loqs-monitor` console script."""

import time

import pytest
from _shared_checkpoint_test_helpers import write_item_ledger

from loqs.tools.monitor import cli


def test_once_smoke_two_workers(tmp_path, capsys):
    now = time.time()
    write_item_ledger(
        tmp_path,
        host="nodeA",
        pid=1000,
        current_item=3,
        shots_done=40,
        shots_total=100,
        done=(0, 1),
        heartbeat=now,
    )
    write_item_ledger(
        tmp_path,
        host="nodeB",
        pid=2000,
        suffix="1111aaaa",
        current_item=2,
        done=(0, 1, 2),
        heartbeat=now,
    )

    assert cli.main(["--once", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "nodeA" in out
    assert "nodeB" in out
    assert "running" in out
    assert "idle" in out
    assert "40/100" in out


def test_once_missing_directory_returns_2(tmp_path, capsys):
    missing = tmp_path / "does_not_exist"

    assert cli.main(["--once", str(missing)]) == 2

    out = capsys.readouterr().out
    assert "does not exist" in out


def test_live_missing_directory_polls_until_interrupt(tmp_path, monkeypatch):
    pytest.importorskip("rich")
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)
        if len(calls) >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", fake_sleep)

    rc = cli.main([str(tmp_path / "missing"), "--interval", "0.01"])

    assert rc == 0
    assert len(calls) >= 2


WALL = 10_000.0
STALE = 60.0


def _by_host_snapshot(directory):
    from loqs.tools.monitor.state import read_snapshot

    def ledger(host, pid, suffix, **kw):
        write_item_ledger(directory, host=host, pid=pid, suffix=suffix, **kw)

    # hostA: one running worker and one stale worker (heartbeat 200 s old).
    ledger(
        "hostA",
        1,
        "0000aaa1",
        current_item=2,
        shots_done=10,
        shots_total=100,
        done=(0,),
        heartbeat=WALL - 5,
    )
    ledger(
        "hostA",
        2,
        "0000aaa2",
        current_item=3,
        shots_done=20,
        shots_total=50,
        done=(0, 1),
        heartbeat=WALL - 200,
    )
    # hostB: an idle worker with nonzero shot fields, and a starting worker.
    ledger(
        "hostB",
        1,
        "0000bbb1",
        current_item=2,
        shots_done=7,
        shots_total=9,
        done=(0, 1, 2),
        heartbeat=WALL - 30,
    )
    ledger("hostB", 2, "0000bbb2", done=(0,))
    # hostC: one running worker.
    ledger(
        "hostC",
        1,
        "0000ccc1",
        current_item=1,
        shots_done=5,
        shots_total=10,
        done=(0,),
        heartbeat=WALL - 10,
    )
    return read_snapshot(directory, stale_after=STALE, wall_clock=lambda: WALL)


def test_by_host_aggregation_and_filter(tmp_path):
    from loqs.tools.monitor.state import WorkerState as S

    snap = _by_host_snapshot(tmp_path)

    rows = {r.host: r for r in cli.aggregate_by_host(snap.workers)}
    assert list(rows) == ["hostA", "hostB", "hostC"]

    a, b, c = rows["hostA"], rows["hostB"], rows["hostC"]
    assert a.workers == 2
    assert (a.state_counts[S.RUNNING], a.state_counts[S.STALE]) == (1, 1)
    assert a.items_done == 3
    assert (a.shots_done, a.shots_total) == (30, 150)
    assert a.no_update_age == pytest.approx(200)
    assert a.has_stale

    assert b.workers == 2
    assert (b.state_counts[S.IDLE], b.state_counts[S.STARTING]) == (1, 1)
    assert b.items_done == 4
    assert (b.shots_done, b.shots_total) == (0, 0)
    assert b.no_update_age is None
    assert not b.has_stale

    assert c.workers == 1
    assert c.items_done == 1
    assert (c.shots_done, c.shots_total) == (5, 10)
    assert c.no_update_age == pytest.approx(10)
    assert not c.has_stale

    filtered, n_before = cli.filter_hosts(snap, ["hostA", "hostC", "nomatch*"])
    assert n_before == 3
    assert {w.host for w in filtered.workers} == {"hostA", "hostC"}
    t = filtered.totals
    assert t.hosts == ("hostA", "hostC")
    assert (t.state_counts[S.RUNNING], t.state_counts[S.STALE]) == (2, 1)
    assert t.items_done == 4
    assert (t.shots_done, t.shots_total) == (35, 160)
    extra = [n for n in filtered.notes if n not in snap.notes]
    assert len(extra) == 1 and "nomatch*" in extra[0]

    # Matching is case-sensitive, and no patterns leaves the snapshot alone.
    empty, _ = cli.filter_hosts(snap, ["HOSTA"])
    assert empty.workers == ()
    same, n = cli.filter_hosts(snap, [])
    assert same is snap and n is None


def test_once_by_host_smoke(tmp_path, capsys):
    _by_host_snapshot(tmp_path)

    rc = cli.main([str(tmp_path), "--once", "--by-host", "--host", "host[AC]"])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    for host, n_rows in (("hostA", 1), ("hostB", 0), ("hostC", 1)):
        assert sum(ln.startswith(host) for ln in lines) == n_rows
    assert any("Showing 2 of 3 hosts" in ln for ln in lines)
