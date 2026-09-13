"""Shared test-only helper functions for checkpoint/resume crash-injection
tests across test_pygstitools.py, test_noisesweeptools.py, and
test_fttools.py -- consolidated from three near-identical copies."""

import time

from loqs.core import QuantumProgram
from loqs.tools.multiprogramrunner import _read_worker_files


def _build_shot_executor():
    """Module-level factory (not a closure) building a fresh loky
    executor -- a picklable `shot_executor` factory for hybrid
    shot-/program-level parallelism tests."""
    import loky

    return loky.get_reusable_executor(max_workers=1)


def _wait_for_index_checkpointed(
    item_checkpoint_dir, index, timeout=30.0, poll_interval=0.02
):
    """Poll `item_checkpoint_dir`'s on-disk checkpoint state (worker files
    only, never `runner.h5` itself) until `index` is durably recorded as
    done, or raise `TimeoutError`. Used to deterministically order a
    crash-injecting worker's own item against a sibling item dispatched
    concurrently to a different real worker process, since both complete
    in real (unordered) time otherwise."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if index in _read_worker_files(item_checkpoint_dir):
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"Item {index} was not checkpointed within {timeout}s")


def _crash_once_and_log_shots(
    item,
    index,
    *,
    original_fn,
    crash_index,
    shots_before_crash,
    wait_for_index,
    item_checkpoint_dir,
    crash_triggered,
    shot_log,
    call_log,
    **kwargs,
):
    """Module-level wrapper (not a closure) around a `MultiProgramRunner`
    per-item worker function, crashing exactly once: on `crash_index`'s
    first dispatch, after `shots_before_crash` of its own shots have
    completed. `crash_triggered` (a `multiprocessing.Manager` dict) gates
    this across real worker processes, since a resumed run re-dispatches
    the same item to a (possibly different) worker that must not crash
    again. Every real `QuantumProgram._run_shot` call and every wrapper
    invocation are recorded to `shot_log`/`call_log` (`Manager` lists),
    observable from the main test process across process boundaries.
    Waits for `wait_for_index` to be checkpointed before crashing, so the
    sibling item dispatched to the other real worker is guaranteed done
    (and durably checkpointed) once the crash is observed."""
    call_log.append(index)

    should_crash = index == crash_index and not crash_triggered.get(
        "triggered", False
    )
    if should_crash:
        crash_triggered["triggered"] = True
        if wait_for_index is not None:
            _wait_for_index_checkpointed(item_checkpoint_dir, wait_for_index)

    original_run_shot = QuantumProgram._run_shot
    computed = {"n": 0}

    def _run_shot_and_maybe_crash(self, max_frame_limit, seed, shot_index):
        if should_crash and computed["n"] >= shots_before_crash:
            raise RuntimeError("Simulated real-worker crash mid-item")
        result = original_run_shot(self, max_frame_limit, seed, shot_index)
        computed["n"] += 1
        shot_log.append((index, shot_index))
        return result

    QuantumProgram._run_shot = _run_shot_and_maybe_crash
    try:
        return original_fn(item, index, **kwargs)
    finally:
        QuantumProgram._run_shot = original_run_shot
