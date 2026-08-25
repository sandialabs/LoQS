#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Shared program-level parallelism helpers.

Chunks a sequence of work items (e.g. edesign circuits, error-injected
programs) and dispatches one worker call per chunk, either through any
plain `.submit()`-based executor (`loky`, `mpi4py.futures.MPIPoolExecutor`)
or through a `submitit.Executor`'s bulk `map_array` (a single SLURM
array-job submission for every chunk, rather than one per `.submit()`
call). Callers own building the actual per-chunk worker function; this
module only owns chunking, dispatch, and progress reporting, so it can be
reused by every program-level call site rather than reimplemented per tool.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, as_completed
import time
from typing import Any, Protocol, TypeVar, runtime_checkable
import warnings

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None  # type: ignore

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


@runtime_checkable
class ChunkExecutor(Protocol):
    """Structural type accepted by `run_chunks_with_executor`.

    Only a `.submit()` method returning a `concurrent.futures.Future` is
    required, so a `loky.get_reusable_executor()` instance or an
    `mpi4py.futures.MPIPoolExecutor` both satisfy this without this module
    depending on either package directly.
    """

    def submit(self, fn, /, *args, **kwargs) -> Future: ...


def chunk_round_robin(items: Sequence[T], n_chunks: int) -> list[list[T]]:
    """Split `items` into `n_chunks` round-robin/striped chunks (item `i`
    goes to chunk `i % n_chunks`), rather than naive contiguous slicing.

    This is deliberate insurance against workloads whose per-item cost
    drifts with position (e.g. GST circuits growing deeper with index,
    which naive contiguous chunking would concentrate almost entirely into
    the last few chunks). It's a placeholder for a real cost-weighted
    (longest-processing-time-first) chunker, meant to be swappable in
    behind this same call site without touching the surrounding dispatch
    code.
    """
    if n_chunks < 1:
        raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
    chunks: list[list[T]] = [[] for _ in range(n_chunks)]
    for i, item in enumerate(items):
        chunks[i % n_chunks].append(item)
    return chunks


def pin_worker_threads() -> None:
    """Pin this process's numerical-library thread pools to one thread.

    The primary, always-correct layer of the thread-oversubscription
    discipline every chunk-processing worker entry point must apply as
    its first action, regardless of which executor backend runs it:
    environment variables (`OMP_NUM_THREADS`, etc.) only help if set
    before the relevant library first initializes its own thread pool,
    which isn't guaranteed for a worker process that already imported
    `numpy`/`pygsti`-adjacent code before reaching this call. Meant to be
    called directly inside a plain, module-level worker function -- not
    built via a decorator, since a decorator would return a closure that
    plain `pickle` (needed for `mpi4py.futures.MPIPoolExecutor`) can't
    resolve by dotted import path.
    """
    if threadpool_limits is not None:
        threadpool_limits(1)
    else:
        warnings.warn(
            "threadpoolctl is not installed, so worker thread pools "
            "cannot be limited to avoid oversubscription. Install "
            "loqs[parallel] or loqs[mpi]."
        )


def run_chunks_with_executor(
    executor: ChunkExecutor,
    worker_fn: Callable[[list[T]], R],
    chunks: Sequence[list[T]],
    desc: str = "Processing chunks",
) -> list[R]:
    """Dispatch `worker_fn` over each of `chunks` via any `ChunkExecutor`
    (e.g. `loky` or `mpi4py.futures.MPIPoolExecutor`), returning one result
    per chunk in the same order as `chunks` (not completion order).

    Progress is reported per completed chunk via
    `tqdm(as_completed(...))`, matching this codebase's existing shot-level
    executor pattern -- `as_completed` blocks efficiently on the futures'
    own wait primitives rather than polling, so the bar advances as chunks
    actually finish.
    """
    futures_to_index = {
        executor.submit(worker_fn, chunk): i
        for i, chunk in enumerate(chunks)
    }
    results: list[Any] = [None] * len(chunks)
    for future in tqdm(
        as_completed(futures_to_index), desc=desc, total=len(chunks)
    ):
        results[futures_to_index[future]] = future.result()
    return results


def run_chunks_with_submitit(
    submitit_executor: Any,
    worker_fn: Callable[[list[T]], R],
    chunks: Sequence[list[T]],
    desc: str = "Processing chunks",
    poll_interval: float = 1.0,
) -> list[R]:
    """Dispatch `worker_fn` over `chunks` via a single `submitit.Executor.
    map_array` call (one `sbatch` submission covering every chunk, rather
    than one per `.submit()` -- the whole reason to prefer `submitit` for
    scheduler fan-out), returning one result per chunk in the same order
    as `chunks`.

    `submitit.Job`s aren't `concurrent.futures.Future`s, so progress can't
    use `as_completed`; instead this polls each job's cheap `.done()`
    check (which only occasionally falls back to an actual cluster status
    call, throttled internally by `submitit` itself) every `poll_interval`
    seconds.
    """
    jobs = submitit_executor.map_array(worker_fn, chunks)
    pending = set(range(len(jobs)))
    with tqdm(total=len(jobs), desc=desc) as bar:
        while pending:
            newly_done = {i for i in pending if jobs[i].done()}
            pending -= newly_done
            bar.update(len(newly_done))
            if pending:
                time.sleep(poll_interval)
    return [job.result() for job in jobs]
