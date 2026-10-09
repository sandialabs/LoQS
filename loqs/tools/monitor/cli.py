#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""`loqs-monitor` console script: watch the workers of a running
multi-program dispatch from its item checkpoint directory.

```
loqs-monitor <item_checkpoint_dir> [--interval SECONDS] [--stale-after SECONDS]
             [--by-host] [--host PATTERN ...] [--once]
```

Live mode (the default) shows a `rich` table refreshed every `--interval`
seconds until Ctrl-C; it needs `rich` (`pip install "loqs[parallel]"`).
`--once` prints one plain-text snapshot and needs no `rich`.

Both modes render the same rows, built by one shared formatting helper.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase

from loqs.tools.monitor.state import (
    DEFAULT_STALE_AFTER,
    MonitorSnapshot,
    MonitorTracker,
    WorkerState,
    WorkerSummary,
    compute_totals,
    read_snapshot,
)

COLUMNS = (
    "host",
    "pid",
    "state",
    "current item",
    "items done",
    "shots done/total",
    "no update in",
)

HOST_COLUMNS = (
    "Host",
    "Workers",
    "Starting",
    "Running",
    "Idle",
    "Stale",
    "Items done",
    "Shots",
    "No update",
)


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}")
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loqs-monitor",
        description=(
            "Monitor the workers of a running LoQS multi-program dispatch "
            "from its item checkpoint directory."
        ),
    )
    parser.add_argument(
        "directory", help="the run's item_checkpoint_dir (may not exist yet)"
    )
    parser.add_argument(
        "--interval",
        type=_positive_float,
        default=2.0,
        help="seconds between polls in live mode (default: %(default)s)",
    )
    parser.add_argument(
        "--stale-after",
        type=_positive_float,
        default=DEFAULT_STALE_AFTER,
        help=(
            "seconds without progress before a running worker is shown as "
            "stale (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--by-host",
        action="store_true",
        help="show one aggregated row per host instead of one per worker",
    )
    parser.add_argument(
        "--host",
        action="append",
        metavar="PATTERN",
        help=(
            "only show hosts matching this shell-style glob (case-sensitive); "
            "may be repeated"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="print one plain-text snapshot and exit (no rich needed)",
    )
    return parser


def format_age(seconds: float | None) -> str:
    """Compact duration such as `45s`, `3m12s` or `2h05m`; "-" for None."""
    if seconds is None:
        return "-"
    total = int(max(seconds, 0))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


def _no_update_age(w: WorkerSummary) -> float | None:
    if w.since_change is not None:
        return w.since_change
    if w.heartbeat_age is not None:
        return max(w.heartbeat_age, 0.0)
    return None


def format_rows(snapshot: MonitorSnapshot) -> list[tuple[str, ...]]:
    """One row of strings per worker, sorted by host then pid."""
    rows: list[tuple[str, ...]] = []
    for w in sorted(snapshot.workers, key=lambda w: (w.host, w.pid)):
        active = w.state in (WorkerState.RUNNING, WorkerState.STALE)
        rows.append(
            (
                w.host,
                str(w.pid),
                w.state.value,
                (
                    str(w.current_item)
                    if active and w.current_item is not None
                    else "-"
                ),
                str(w.items_done),
                f"{w.shots_done}/{w.shots_total}" if active else "-",
                format_age(_no_update_age(w)),
            )
        )
    return rows


@dataclass(frozen=True)
class HostRow:
    """One host's aggregate over its workers.

    Shots sum over running and stale workers only; `no_update_age` is the
    largest no-update age among those workers (None when there are none).
    """

    host: str
    workers: int
    state_counts: dict[WorkerState, int]
    items_done: int
    shots_done: int
    shots_total: int
    no_update_age: float | None
    has_stale: bool


def aggregate_by_host(workers: Iterable[WorkerSummary]) -> list[HostRow]:
    """One `HostRow` per host, sorted alphabetically by host."""
    by_host: dict[str, list[WorkerSummary]] = {}
    for w in workers:
        by_host.setdefault(w.host, []).append(w)
    rows: list[HostRow] = []
    for host in sorted(by_host):
        group = by_host[host]
        totals = compute_totals(group)
        ages = [
            age
            for w in group
            if w.state in (WorkerState.RUNNING, WorkerState.STALE)
            and (age := _no_update_age(w)) is not None
        ]
        rows.append(
            HostRow(
                host=host,
                workers=len(group),
                state_counts=totals.state_counts,
                items_done=totals.items_done,
                shots_done=totals.shots_done,
                shots_total=totals.shots_total,
                no_update_age=max(ages) if ages else None,
                has_stale=totals.state_counts[WorkerState.STALE] > 0,
            )
        )
    return rows


def _format_host_row(row: HostRow) -> tuple[str, ...]:
    return (
        row.host,
        str(row.workers),
        *(str(row.state_counts[s]) for s in WorkerState),
        str(row.items_done),
        f"{row.shots_done}/{row.shots_total}",
        format_age(row.no_update_age),
    )


def filter_hosts(
    snapshot: MonitorSnapshot, patterns: Sequence[str]
) -> tuple[MonitorSnapshot, int]:
    """Keep the workers whose host matches any glob in `patterns`.

    Matching is case-sensitive (`fnmatch.fnmatchcase`). Returns the
    filtered snapshot, with totals recomputed and a note for each pattern
    that matched no host, and the number of hosts before filtering. With no
    patterns the snapshot is returned unchanged.
    """
    n_before = len(snapshot.totals.hosts)
    if not patterns:
        return snapshot, n_before
    hosts = snapshot.totals.hosts
    kept = tuple(
        w
        for w in snapshot.workers
        if any(fnmatchcase(w.host, p) for p in patterns)
    )
    unmatched = [
        f"No host matches --host {p!r}."
        for p in patterns
        if not any(fnmatchcase(h, p) for h in hosts)
    ]
    filtered = replace(
        snapshot,
        workers=kept,
        totals=compute_totals(list(kept)),
        notes=(*snapshot.notes, *unmatched),
    )
    return filtered, n_before


def format_summary(
    snapshot: MonitorSnapshot, hosts_total: int | None = None
) -> list[str]:
    """The lines shown above the rows.

    With `hosts_total`, adds a `Showing N of M hosts` line.
    """
    t = snapshot.totals
    counts = ", ".join(f"{t.state_counts[s]} {s.value}" for s in WorkerState)
    lines = [
        f"Directory: {snapshot.directory}",
        f"Workers: {counts}",
        f"Hosts: {len(t.hosts)}",
    ]
    if hosts_total is not None:
        lines.append(f"Showing {len(t.hosts)} of {hosts_total} hosts")
    lines += [
        f"Items done: {t.items_done}",
        f"Shots done/total: {t.shots_done}/{t.shots_total}",
    ]
    return lines


def format_plain(
    snapshot: MonitorSnapshot,
    *,
    by_host: bool = False,
    hosts_total: int | None = None,
) -> str:
    """Plain-text rendering of a snapshot, per worker or per host."""
    lines = format_summary(snapshot, hosts_total)
    columns: tuple[str, ...]
    if by_host:
        columns = HOST_COLUMNS
        rows = [
            _format_host_row(r) for r in aggregate_by_host(snapshot.workers)
        ]
    else:
        columns = COLUMNS
        rows = format_rows(snapshot)
    if rows:
        table = [columns, *rows]
        widths = [max(len(r[i]) for r in table) for i in range(len(columns))]
        lines.append("")
        for r in table:
            lines.append(
                "  ".join(c.ljust(wd) for c, wd in zip(r, widths)).rstrip()
            )
    if snapshot.notes:
        lines.append("")
        lines.extend(snapshot.notes)
    return "\n".join(lines)


def _run_once(
    directory: str,
    stale_after: float,
    by_host: bool = False,
    patterns: Sequence[str] = (),
) -> int:
    snapshot = read_snapshot(directory, stale_after=stale_after)
    snapshot, hosts_total = _view(snapshot, patterns)
    print(format_plain(snapshot, by_host=by_host, hosts_total=hosts_total))
    return 0 if snapshot.directory_exists else 2


def _render_live(
    snapshot: MonitorSnapshot,
    by_host: bool = False,
    hosts_total: int | None = None,
):
    from rich.console import Group, RenderableType
    from rich.table import Table
    from rich.text import Text

    if by_host:
        table = Table(*HOST_COLUMNS)
        for host_row in aggregate_by_host(snapshot.workers):
            table.add_row(
                *_format_host_row(host_row),
                style="bold red" if host_row.has_stale else None,
            )
    else:
        table = Table(*COLUMNS)
        for row in format_rows(snapshot):
            style = "bold red" if row[2] == WorkerState.STALE.value else None
            table.add_row(*row, style=style)
    parts: list[RenderableType] = [
        Text("\n".join(format_summary(snapshot, hosts_total))),
        table,
    ]
    if snapshot.notes:
        parts.append(Text("\n".join(snapshot.notes)))
    return Group(*parts)


def _view(
    snapshot: MonitorSnapshot, patterns: Sequence[str]
) -> tuple[MonitorSnapshot, int | None]:
    """Apply the host filter; `hosts_total` is None when not filtering."""
    filtered, hosts_total = filter_hosts(snapshot, patterns)
    return filtered, (hosts_total if patterns else None)


def _run_live(
    directory: str,
    interval: float,
    stale_after: float,
    by_host: bool = False,
    patterns: Sequence[str] = (),
) -> int:
    try:
        from rich.live import Live
    except ImportError:
        print(
            'loqs-monitor live mode needs rich: pip install "loqs[parallel]" '
            "(or use --once)",
            file=sys.stderr,
        )
        return 1

    tracker = MonitorTracker(directory, stale_after=stale_after)

    def render():
        snapshot, hosts_total = _view(tracker.poll(), patterns)
        return _render_live(snapshot, by_host, hosts_total)

    try:
        with Live(render(), auto_refresh=False) as live:
            while True:
                live.update(render(), refresh=True)
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    patterns = args.host or []
    if args.once:
        return _run_once(
            args.directory, args.stale_after, args.by_host, patterns
        )
    return _run_live(
        args.directory,
        args.interval,
        args.stale_after,
        args.by_host,
        patterns,
    )


if __name__ == "__main__":
    sys.exit(main())
