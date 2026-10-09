#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""State aggregation for monitoring a running `MultiProgramRunner` dispatch.

Reads every worker's SWMR item ledger in one `item_checkpoint_dir` and
summarizes it as plain data. Nothing here renders or prints anything.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from loqs.internal.swmrledger import (
    SwmrLedgerSnapshot,
    open_swmr_reader,
    read_swmr_ledger_status,
)

# Must match the file names written by `multiprogramrunner._item_ledger_path`
# (guarded by a test). The glob never matches the writer's `.tmp` files.
WORKER_LEDGER_GLOB = "worker_*_runner.h5"

DEFAULT_STALE_AFTER = 120.0

_NAME_PREFIX = "worker_"
_NAME_SUFFIX = "_runner.h5"
_SUFFIX_RE = re.compile(r"[0-9a-fA-F]{8}")


class WorkerState(str, Enum):
    """Progress state of one worker."""

    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    STALE = "stale"


@dataclass(frozen=True)
class WorkerSummary:
    """One worker's progress, as read in one poll.

    `current_item`, `shots_done` and `shots_total` are set only when
    `state` is running or stale. `heartbeat_age` is
    `wall_clock() - last_heartbeat` (unclamped; None with no heartbeat).
    `since_change` is monitor-clock seconds since the last observed change
    (None until a change has been observed).
    """

    worker_id: str
    host: str
    pid: int
    path: Path
    state: WorkerState
    items_done: int
    current_item: int | None
    shots_done: int | None
    shots_total: int | None
    heartbeat_age: float | None
    since_change: float | None


@dataclass(frozen=True)
class SkippedFile:
    """A worker file that was not summarized, and why.

    `reason` is `legacy` (pre-ledger file), `unreadable` (open or read
    error) or `unrecognized` (unparsable name or missing fields).
    """

    path: Path
    reason: str


@dataclass(frozen=True)
class MonitorTotals:
    """Aggregates over the summarized workers.

    `shots_done` and `shots_total` sum over running and stale workers only.
    """

    state_counts: dict[WorkerState, int]
    hosts: tuple[str, ...]
    items_done: int
    shots_done: int
    shots_total: int


@dataclass(frozen=True)
class MonitorSnapshot:
    """The result of one poll of an item checkpoint directory."""

    directory: Path
    directory_exists: bool
    workers: tuple[WorkerSummary, ...]
    skipped: tuple[SkippedFile, ...]
    totals: MonitorTotals
    notes: tuple[str, ...]


def _list_worker_files(directory: Path) -> list[Path]:
    """List candidate worker ledger files in `directory`, sorted."""
    # Keep in step with `multiprogramrunner._item_ledger_path`.
    return sorted(directory.glob(WORKER_LEDGER_GLOB))


def _parse_worker_name(path: Path) -> tuple[str, str, int] | None:
    """Parse `worker_<host>_<pid>_<suffix>_runner.h5` from the right.

    Returns `(worker_id, host, pid)`, or None when the name doesn't match.
    Hostnames may contain `_`, so the split is from the right.
    """
    name = path.name
    if not (name.startswith(_NAME_PREFIX) and name.endswith(_NAME_SUFFIX)):
        return None
    core = name[len(_NAME_PREFIX) : -len(_NAME_SUFFIX)]
    parts = core.rsplit("_", 2)
    if len(parts) != 3:
        return None
    host, pid, suffix = parts
    if not host or not (pid.isascii() and pid.isdigit()):
        return None
    if _SUFFIX_RE.fullmatch(suffix) is None:
        return None
    return core, host, int(pid)


def _read_ledger(path: Path) -> SwmrLedgerSnapshot | str:
    """Read one ledger file, or return a skip reason string."""
    try:
        f, group = open_swmr_reader(path)
    except KeyError:
        return "legacy"
    except OSError:
        return "unreadable"
    try:
        snap = read_swmr_ledger_status(group)
    except (OSError, KeyError, ValueError):
        return "unreadable"
    finally:
        f.close()
    if (
        snap.done is None
        or snap.current_item_index is None
        or snap.last_heartbeat is None
    ):
        return "unrecognized"
    return snap


def _fingerprint(snap: SwmrLedgerSnapshot, items_done: int) -> tuple:
    heartbeat = snap.last_heartbeat
    if heartbeat is None or math.isnan(heartbeat):
        heartbeat = None
    return (
        snap.current_item_index,
        snap.item_shots_done,
        items_done,
        heartbeat,
    )


def _is_in_flight(snap: SwmrLedgerSnapshot) -> bool:
    """Whether the snapshot's current item is not yet marked done."""
    done = snap.done
    index = snap.current_item_index
    assert done is not None and index is not None
    return not (0 <= index < len(done) and bool(done[index]))


def _classify(
    snap: SwmrLedgerSnapshot,
    heartbeat_age: float | None,
    since_change: float | None,
    stale_after: float,
) -> WorkerState:
    """Starting / idle / running / stale for one worker's snapshot."""
    if heartbeat_age is None:
        return WorkerState.STARTING
    if not _is_in_flight(snap):
        return WorkerState.IDLE
    age = since_change if since_change is not None else heartbeat_age
    return WorkerState.STALE if age >= stale_after else WorkerState.RUNNING


def compute_totals(workers: list[WorkerSummary]) -> MonitorTotals:
    counts = {state: 0 for state in WorkerState}
    shots_done = shots_total = 0
    for w in workers:
        counts[w.state] += 1
        if w.state in (WorkerState.RUNNING, WorkerState.STALE):
            shots_done += w.shots_done or 0
            shots_total += w.shots_total or 0
    return MonitorTotals(
        state_counts=counts,
        hosts=tuple(sorted({w.host for w in workers})),
        items_done=sum(w.items_done for w in workers),
        shots_done=shots_done,
        shots_total=shots_total,
    )


def _build_notes(
    directory_exists: bool,
    n_listed: int,
    skipped: list[SkippedFile],
) -> tuple[str, ...]:
    notes: list[str] = []
    if not directory_exists:
        notes.append(
            "The directory does not exist. Check the path, or that the run "
            "was started with item checkpointing enabled."
        )
    elif n_listed == 0:
        notes.append(
            "No worker ledgers found: item checkpointing may be off, the "
            "run may not have started yet, or it finished and its ledgers "
            "were consolidated into runner.h5."
        )
    reasons = [s.reason for s in skipped]
    if "legacy" in reasons:
        notes.append(
            "Legacy worker files (written before progress ledgers) were "
            "skipped and are not monitored."
        )
    n_bad = sum(r in ("unreadable", "unrecognized") for r in reasons)
    if n_bad:
        notes.append(
            f"{n_bad} file(s) could not be read or recognized in this poll."
        )
    return tuple(notes)


class MonitorTracker:
    """Polls one item checkpoint directory, tracking per-worker change.

    Staleness is judged by this tracker's own `clock`: once a change in a
    worker's progress has been observed, an in-flight worker is stale when
    nothing has changed for `stale_after` seconds. Before any change has
    been observed, the worker's heartbeat age (by `wall_clock`) is used.

    Parameters
    ----------
    directory : Path | str
        The `item_checkpoint_dir` to scan.
    stale_after : float, optional
        Seconds without change before an in-flight worker is stale.
    clock : Callable[[], float], optional
        Monotonic clock for change tracking.
    wall_clock : Callable[[], float], optional
        Wall clock compared against workers' heartbeat timestamps.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        stale_after: float = DEFAULT_STALE_AFTER,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.directory = Path(directory)
        self.stale_after = stale_after
        self._clock = clock
        self._wall_clock = wall_clock
        # worker id -> (last fingerprint, clock time of last change or None)
        self._history: dict[str, tuple[tuple, float | None]] = {}

    def _observe(
        self, worker_id: str, fingerprint: tuple, now: float
    ) -> float | None:
        """Record this poll's fingerprint; return time of last change."""
        previous = self._history.get(worker_id)
        if previous is None:
            last_change = None
        elif previous[0] != fingerprint:
            last_change = now
        else:
            last_change = previous[1]
        self._history[worker_id] = (fingerprint, last_change)
        return last_change

    def _summarize(
        self,
        path: Path,
        ident: tuple[str, str, int],
        snap: SwmrLedgerSnapshot,
        now: float,
        wall: float,
    ) -> WorkerSummary:
        worker_id, host, pid = ident
        assert snap.done is not None and snap.last_heartbeat is not None
        items_done = int(snap.done.sum())
        last_change = self._observe(
            worker_id, _fingerprint(snap, items_done), now
        )
        since_change = None if last_change is None else now - last_change
        heartbeat_age = (
            None
            if math.isnan(snap.last_heartbeat)
            else wall - snap.last_heartbeat
        )
        state = _classify(snap, heartbeat_age, since_change, self.stale_after)
        active = state in (WorkerState.RUNNING, WorkerState.STALE)
        return WorkerSummary(
            worker_id=worker_id,
            host=host,
            pid=pid,
            path=path,
            state=state,
            items_done=items_done,
            current_item=snap.current_item_index if active else None,
            shots_done=snap.item_shots_done or 0 if active else None,
            shots_total=snap.item_shots_total or 0 if active else None,
            heartbeat_age=heartbeat_age,
            since_change=since_change,
        )

    def poll(self) -> MonitorSnapshot:
        """Read every worker ledger once and return a `MonitorSnapshot`."""
        directory_exists = self.directory.is_dir()
        files = _list_worker_files(self.directory) if directory_exists else []
        now = self._clock()
        wall = self._wall_clock()

        workers: list[WorkerSummary] = []
        skipped: list[SkippedFile] = []
        listed_ids: set[str] = set()
        for path in files:
            ident = _parse_worker_name(path)
            if ident is None:
                skipped.append(SkippedFile(path, "unrecognized"))
                continue
            listed_ids.add(ident[0])
            result = _read_ledger(path)
            if isinstance(result, str):
                skipped.append(SkippedFile(path, result))
            else:
                workers.append(self._summarize(path, ident, result, now, wall))

        for gone in set(self._history) - listed_ids:
            del self._history[gone]

        workers.sort(key=lambda w: (w.host, w.pid, w.worker_id))
        return MonitorSnapshot(
            directory=self.directory,
            directory_exists=directory_exists,
            workers=tuple(workers),
            skipped=tuple(skipped),
            totals=compute_totals(workers),
            notes=_build_notes(directory_exists, len(files), skipped),
        )


def read_snapshot(
    directory: Path | str,
    *,
    stale_after: float = DEFAULT_STALE_AFTER,
    wall_clock: Callable[[], float] = time.time,
) -> MonitorSnapshot:
    """Poll `directory` once with a fresh `MonitorTracker`."""
    return MonitorTracker(
        directory, stale_after=stale_after, wall_clock=wall_clock
    ).poll()
