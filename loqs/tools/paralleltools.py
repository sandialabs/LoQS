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
programs, noise-sweep points) and dispatches one worker call per chunk,
either through any plain `.submit()`-based executor (`loky`,
`mpi4py.futures.MPIPoolExecutor`) or through a `submitit.Executor`'s bulk
`map_array` (a single SLURM array-job submission for every chunk, rather
than one per `.submit()` call). [](api:ParallelStrategy) bundles this
program-level dispatch together with optional nested shot-level
parallelism into one reusable object; the lower-level `chunk_*`/
`run_chunks_with_*` functions it's built on are also exposed directly for
callers that want finer control. Callers own building the actual
per-chunk worker function; this module owns chunking, dispatch, and
progress reporting, so it can be reused by every program-level call site
rather than reimplemented per tool.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import as_completed
from dataclasses import dataclass
import time
from typing import Any, TypeVar
import warnings

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None  # type: ignore

from tqdm import tqdm

from loqs.core.executors import MapArrayExecutor, SubmitExecutor

T = TypeVar("T")
R = TypeVar("R")


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


def resolve_shot_executor(
    shot_executor: SubmitExecutor | Callable[[], SubmitExecutor] | None,
) -> SubmitExecutor | None:
    """Resolve `shot_executor` into an actual executor instance (or
    `None`), accepting either a live executor or a zero-argument factory
    callable that builds one, disambiguated via `callable()` -- executors
    don't implement `__call__` in practice, so this is unambiguous.

    A live executor (e.g. a `loky.get_reusable_executor()` instance) holds
    real OS resources (pipes/locks) that cannot be pickled across a
    process boundary, so it can only be used as-is when nothing needs to
    pickle it (no program-level chunking in play). A *factory* -- a plain,
    picklable, zero-argument callable that builds and returns a fresh
    executor -- sidesteps this: it crosses the process boundary instead of
    the executor itself, and each worker calls it once to get an executor
    native to its own process, nesting shot-level parallelism inside
    program-level parallelism. [](api:ParallelStrategy) is what actually
    enforces that a factory (not a live executor) is used whenever
    program-level chunking is also in play; this function just resolves
    whichever form it's given. Callers should invoke this once per chunk
    (not once per item within a chunk), so a factory that spins up new
    subprocesses doesn't pay that cost more than once per chunk.
    """
    if shot_executor is None:
        return None
    return shot_executor() if callable(shot_executor) else shot_executor


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


def run_chunks_with_submit_executor(
    executor: SubmitExecutor,
    worker_fn: Callable[[list[T]], R],
    chunks: Sequence[list[T]],
    desc: str = "Processing chunks",
) -> list[R]:
    """Dispatch `worker_fn` over each of `chunks` via any `SubmitExecutor`
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


def run_chunks_with_map_array_executor(
    executor: MapArrayExecutor,
    worker_fn: Callable[[list[T]], R],
    chunks: Sequence[list[T]],
    desc: str = "Processing chunks",
    poll_interval: float = 1.0,
) -> list[R]:
    """Dispatch `worker_fn` over `chunks` via a single `MapArrayExecutor.
    map_array` call (e.g. one `sbatch` submission covering every chunk for
    a `submitit.Executor`, rather than one per `.submit()` -- the whole
    reason to prefer a `MapArrayExecutor` for scheduler fan-out), returning
    one result per chunk in the same order as `chunks`.

    A `submitit.Job` isn't a `concurrent.futures.Future`, so progress can't
    use `as_completed`; instead this polls each job's cheap `.done()`
    check (which only occasionally falls back to an actual cluster status
    call, throttled internally by `submitit` itself) every `poll_interval`
    seconds.
    """
    jobs = executor.map_array(worker_fn, chunks)
    pending = set(range(len(jobs)))
    with tqdm(total=len(jobs), desc=desc) as bar:
        while pending:
            newly_done = {i for i in pending if jobs[i].done()}
            pending -= newly_done
            bar.update(len(newly_done))
            if pending:
                time.sleep(poll_interval)
    return [job.result() for job in jobs]


@dataclass
class ParallelStrategy:
    """Bundles program-level chunk dispatch together with optional nested
    shot-level parallelism, reused identically by every
    `loqs.tools` call site that parallelizes "one `QuantumProgram` per
    chunk item" work (edesign circuits, error-injected programs,
    noise-sweep points -- each of which builds and runs exactly one
    `QuantumProgram`).

    Parameters
    ----------
    program_executor : SubmitExecutor | MapArrayExecutor | None
        A `SubmitExecutor` (e.g. `loky.get_reusable_executor()` or
        `mpi4py.futures.MPIPoolExecutor`, dispatched one `.submit()` call
        per chunk) or a `MapArrayExecutor` (e.g. a `submitit.Executor`,
        dispatched via a single bulk `.map_array()` call covering every
        chunk) -- which protocol it satisfies determines the dispatch
        mechanism automatically via `isinstance`, checking
        `MapArrayExecutor` first (a `submitit.Executor` satisfies both
        protocols, but `map_array`'s bulk submission is the efficient
        path). `None` means no program-level parallelism: chunks run
        serially, in the driver process.
    n_program_chunks : int | None
        Number of round-robin chunks to split work into. Required when
        `program_executor` is a `MapArrayExecutor` (submitting one array
        task per item would be dominated by scheduling overhead for
        typical LoQS workloads); defaults to one chunk per item when
        `program_executor` is a plain `SubmitExecutor`; ignored when
        `program_executor` is `None`.
    shot_executor : SubmitExecutor | Callable[[], SubmitExecutor] | None
        Nested shot-level parallelism, forwarded as
        `QuantumProgram.run(shot_executor=...)` for every program built
        inside a chunk. A live `SubmitExecutor` (used as-is) is only valid
        when `program_executor` is `None`, since no process boundary is
        crossed in that case; a zero-argument factory callable is
        required when `program_executor` is also given (resolved once per
        chunk, inside that chunk's own worker process), since a live
        executor holds OS resources (pipes/locks) that cannot be pickled
        across that boundary. `None` means no shot-level parallelism.
    """

    program_executor: SubmitExecutor | MapArrayExecutor | None = None
    n_program_chunks: int | None = None
    shot_executor: SubmitExecutor | Callable[[], SubmitExecutor] | None = (
        None
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.program_executor, MapArrayExecutor)
            and self.n_program_chunks is None
        ):
            raise ValueError(
                "n_program_chunks is required when program_executor is a "
                "MapArrayExecutor -- submitting one array task per item "
                "would be dominated by scheduler overhead for typical "
                "LoQS workloads, so a chunk count must be chosen "
                "deliberately rather than defaulted."
            )
        if (
            self.program_executor is not None
            and self.shot_executor is not None
            and not callable(self.shot_executor)
        ):
            raise ValueError(
                "shot_executor must be a zero-argument factory callable "
                "(not a live executor) when program_executor is also "
                "given: a live executor holds OS resources (pipes/locks) "
                "that cannot be pickled across the process boundary each "
                "dispatched chunk crosses."
            )

    @property
    def is_chunked(self) -> bool:
        """Whether `program_executor` is set, i.e. whether chunks are
        dispatched to a real executor rather than processed serially in
        the driver process."""
        return self.program_executor is not None

    def make_chunks(self, items: Sequence[T]) -> list[list[T]]:
        """Split `items` into round-robin chunks per `n_program_chunks`
        (defaulting to one chunk per item when unset)."""
        n = (
            self.n_program_chunks
            if self.n_program_chunks is not None
            else len(items)
        )
        return chunk_round_robin(items, n)

    def dispatch(
        self,
        worker_fn: Callable[[list[T]], R],
        chunks: Sequence[list[T]],
        desc: str = "Processing chunks",
    ) -> list[R]:
        """Dispatch `worker_fn` over `chunks` via `program_executor`,
        picking the dispatch mechanism (`.submit()`-per-chunk vs. a bulk
        `.map_array()` call) automatically. Only valid when `is_chunked`
        is `True`."""
        assert self.program_executor is not None, (
            "dispatch() requires program_executor to be set; check "
            "is_chunked first."
        )
        # Checked first: a MapArrayExecutor (e.g. submitit.Executor) also
        # satisfies SubmitExecutor, but map_array's bulk submission is the
        # efficient path it's chosen for.
        if isinstance(self.program_executor, MapArrayExecutor):
            return run_chunks_with_map_array_executor(
                self.program_executor, worker_fn, chunks, desc=desc
            )
        return run_chunks_with_submit_executor(
            self.program_executor, worker_fn, chunks, desc=desc
        )
