"""Shared test-only helper functions for checkpoint/resume crash-injection
tests across test_pygstitools.py, test_noisesweeptools.py, and
test_fttools.py -- consolidated from three near-identical copies."""


def write_item_ledger(
    directory,
    *,
    host="nodeA",
    pid=1000,
    suffix="0000abcd",
    current_item=None,
    shots_done=0,
    shots_total=0,
    done=(),
    payloads=None,
    heartbeat=None,
):
    """Create or update a synthetic worker item ledger in `directory`.

    Built with the driver's own ledger fields and payload writer, at the
    driver's file naming. Calling it again on an existing path reopens the
    ledger and applies only the given updates. The file is closed before
    returning, so a same-process reader can open it.

    `payloads` is `{index: value}`; each is written to its own payload
    file and marks the item done. Returns the ledger path.
    """
    from pathlib import Path

    from loqs.internal.swmrledger import (
        mark_ledger_item_done,
        open_swmr_writer,
        update_ledger_heartbeat,
        update_ledger_in_flight,
    )
    from loqs.tools.multiprogramrunner import (
        _ITEM_LEDGER_FIELDS,
        _write_item_checkpoint_with_ledger,
    )

    directory = Path(directory)
    path = directory / f"worker_{host}_{pid}_{suffix}_runner.h5"
    f, group = open_swmr_writer(path, _ITEM_LEDGER_FIELDS)
    try:
        for index in done:
            mark_ledger_item_done(group, index, 1.0)
        for index, value in (payloads or {}).items():
            payload_path = (
                directory
                / f"worker_{host}_{pid}_{suffix}_item_{index}_payload.h5"
            )
            _write_item_checkpoint_with_ledger(
                group, payload_path, index, [("results", value, False)], 1.0
            )
        if current_item is not None:
            update_ledger_in_flight(
                group, current_item, shots_done, shots_total
            )
        if heartbeat is not None:
            update_ledger_heartbeat(group, timestamp=heartbeat)
    finally:
        f.close()
    return path


def _build_shot_executor():
    """Module-level factory (not a closure) building a fresh loky
    executor -- a picklable `shot_executor` factory for hybrid
    shot-/program-level parallelism tests."""
    import loky

    return loky.get_reusable_executor(max_workers=1)
