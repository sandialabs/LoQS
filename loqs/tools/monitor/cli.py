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
loqs-monitor <item_checkpoint_dir> [--interval SECONDS] [--stale-after SECONDS] [--once]
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

from loqs.tools.monitor.state import (
    DEFAULT_STALE_AFTER,
    MonitorSnapshot,
    MonitorTracker,
    WorkerState,
    WorkerSummary,
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


def format_summary(snapshot: MonitorSnapshot) -> list[str]:
    """The lines shown above the rows."""
    t = snapshot.totals
    counts = ", ".join(f"{t.state_counts[s]} {s.value}" for s in WorkerState)
    return [
        f"Directory: {snapshot.directory}",
        f"Workers: {counts}",
        f"Hosts: {len(t.hosts)}",
        f"Items done: {t.items_done}",
        f"Shots done/total: {t.shots_done}/{t.shots_total}",
    ]


def format_plain(snapshot: MonitorSnapshot) -> str:
    """Plain-text rendering of a snapshot."""
    lines = format_summary(snapshot)
    rows = format_rows(snapshot)
    if rows:
        table = [COLUMNS, *rows]
        widths = [max(len(r[i]) for r in table) for i in range(len(COLUMNS))]
        lines.append("")
        for r in table:
            lines.append(
                "  ".join(c.ljust(wd) for c, wd in zip(r, widths)).rstrip()
            )
    if snapshot.notes:
        lines.append("")
        lines.extend(snapshot.notes)
    return "\n".join(lines)


def _run_once(directory: str, stale_after: float) -> int:
    snapshot = read_snapshot(directory, stale_after=stale_after)
    print(format_plain(snapshot))
    return 0 if snapshot.directory_exists else 2


def _render_live(snapshot: MonitorSnapshot):
    from rich.console import Group, RenderableType
    from rich.table import Table
    from rich.text import Text

    table = Table(*COLUMNS)
    for row in format_rows(snapshot):
        style = "bold red" if row[2] == WorkerState.STALE.value else None
        table.add_row(*row, style=style)
    parts: list[RenderableType] = [
        Text("\n".join(format_summary(snapshot))),
        table,
    ]
    if snapshot.notes:
        parts.append(Text("\n".join(snapshot.notes)))
    return Group(*parts)


def _run_live(directory: str, interval: float, stale_after: float) -> int:
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
    try:
        with Live(_render_live(tracker.poll()), auto_refresh=False) as live:
            while True:
                live.update(_render_live(tracker.poll()), refresh=True)
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.once:
        return _run_once(args.directory, args.stale_after)
    return _run_live(args.directory, args.interval, args.stale_after)


if __name__ == "__main__":
    sys.exit(main())
