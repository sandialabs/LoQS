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
