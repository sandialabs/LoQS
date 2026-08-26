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
import math
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


def _executor_worker_count(executor: Any) -> int | None:
    """Best-effort worker count for an executor, or `None` if it doesn't
    expose one.

    `loky` and stdlib `ProcessPoolExecutor`-family executors track this as
    `_max_workers`; some others (e.g. `mpi4py.futures.MPIPoolExecutor`)
    expose a public `max_workers` instead. A `submitit.Executor` doesn't
    have a fixed worker count at all -- its parallelism depends on
    scheduler availability -- so this returns `None` for it.
    """
    for attr in ("_max_workers", "max_workers"):
        value = getattr(executor, attr, None)
        if isinstance(value, int):
            return value
    return None


@dataclass
class ExecutorSpec:
    """Picklable recipe for building a fresh executor of a given backend,
    extracted from a live executor's own construction parameters.

    An instance is itself a zero-argument factory (`__call__` builds and
    returns a new executor), so it satisfies the factory-callable
    requirement [](api:ParallelStrategy) has for `shot_executor` whenever
    `program_executor` chunking is in play -- this is what lets a caller
    hand `ParallelStrategy` an ordinary live executor directly (see
    `_introspect_executor_spec`) instead of writing a factory function by
    hand. Only `loky` is currently recognized; unrecognized backends still
    require an explicit factory.
    """

    backend: str
    kwargs: dict[str, Any]

    def __call__(self) -> SubmitExecutor:
        if self.backend == "loky":
            import loky

            return loky.get_reusable_executor(**self.kwargs)
        raise ValueError(
            f"ExecutorSpec has no builder for backend {self.backend!r}."
        )

    def describe(self) -> str:
        """Short, human-readable label, e.g. `"loky(max_workers=2)"`."""
        params = ", ".join(
            f"{key}={value}" for key, value in sorted(self.kwargs.items())
        )
        return f"{self.backend}({params})"


def _introspect_executor_spec(executor: SubmitExecutor) -> ExecutorSpec | None:
    """Best-effort extraction of a live executor's backend and
    construction parameters into an `ExecutorSpec`, or `None` if the
    backend isn't recognized.

    Only `loky` (identified by its executor class's top-level module
    name, so this doesn't need to import `loky` itself) is currently
    supported, extracting `max_workers` via the same `_max_workers`
    attribute `_executor_worker_count` reads.
    """
    if type(executor).__module__.split(".")[0] == "loky":
        max_workers = _executor_worker_count(executor)
        kwargs = {} if max_workers is None else {"max_workers": max_workers}
        return ExecutorSpec("loky", kwargs)
    return None


def _indent_rows(rows: Sequence[tuple[str, int]]) -> list[str]:
    """Render `(label, value)` pairs as indented `"label value"` lines,
    right-padding labels to a common width so the values line up in a
    column. Returns an empty list (no lines at all) when `rows` is
    empty, rather than an empty/blank line."""
    if not rows:
        return []
    width = max(len(label) for label, _ in rows)
    return [f"     {label:<{width}} {value}" for label, value in rows]


def _describe_executor_tag(executor: Any) -> str:
    """Short backend tag for a live executor, e.g. `"loky(max_workers=2)"`
    for a recognized backend (via `_introspect_executor_spec`), or its
    plain type name otherwise (e.g. `submitit.AutoExecutor`, which isn't
    introspected for parameters)."""
    spec = _introspect_executor_spec(executor)
    return spec.describe() if spec is not None else type(executor).__name__


def _shot_executor_worker_count(
    shot_executor: (
        SubmitExecutor | ExecutorSpec | Callable[[], SubmitExecutor]
    ),
) -> int | None:
    """Worker count for a resolved `shot_executor` value -- an
    `ExecutorSpec` (reads its own `max_workers` kwarg directly, without
    building an executor just to introspect it), a live executor (via
    `_executor_worker_count`), or `None` if it's an arbitrary factory
    callable, whose internals aren't introspectable at all."""
    if isinstance(shot_executor, ExecutorSpec):
        value = shot_executor.kwargs.get("max_workers")
        return value if isinstance(value, int) else None
    if callable(shot_executor):
        return None
    return _executor_worker_count(shot_executor)


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
        executor.submit(worker_fn, chunk): i for i, chunk in enumerate(chunks)
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
        inside a chunk. `None` means no shot-level parallelism. When
        `program_executor` is `None`, a live executor is used as-is (no
        process boundary is crossed). When `program_executor` is also
        given, a live executor can't cross the process boundary each
        dispatched chunk crosses -- a recognized backend (currently just
        `loky`) is transparently replaced with an `ExecutorSpec` built
        from its own construction parameters, which builds an equivalent
        fresh executor inside each chunk worker; an unrecognized backend
        instead requires an explicit zero-argument factory callable,
        which is used as given.
    """

    program_executor: SubmitExecutor | MapArrayExecutor | None = None
    n_program_chunks: int | None = None
    shot_executor: SubmitExecutor | Callable[[], SubmitExecutor] | None = None

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
            spec = _introspect_executor_spec(self.shot_executor)
            if spec is None:
                raise ValueError(
                    "shot_executor is a live executor of an unrecognized "
                    "backend, given alongside program_executor: a live "
                    "executor holds OS resources (pipes/locks) that "
                    "cannot be pickled across the process boundary each "
                    "dispatched chunk crosses, and this backend isn't "
                    "recognized for automatic factory construction (see "
                    "_introspect_executor_spec). Pass a zero-argument "
                    "factory callable instead."
                )
            self.shot_executor = spec

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

    def describe(
        self,
        items: Sequence[T] | None = None,
        num_shots: int | None = None,
    ) -> str:
        """Human-readable summary of this strategy's configured
        parallelism at each axis, for sanity-checking a configuration
        before running anything.

        Each axis is reported as a short backend tag (e.g.
        `"loky(max_workers=2)"`, via `_describe_executor_tag`/
        `ExecutorSpec.describe`), followed by whichever indented fields
        can actually be computed from what's known -- a field is simply
        omitted (not printed as "unknown") when a required input (`items`,
        `num_shots`, or a worker count the executor doesn't expose) is
        missing. The program axis can report a genuine chunk count
        (`n_program_chunks` is an explicit, independently-chosen setting)
        alongside how many chunks land on each worker; the shot axis has
        no analogous chunking setting -- shots are dispatched one at a
        time -- so it only reports the total and the resulting average
        per worker.
        """
        lines = [self._describe_program_axis(items)]
        lines.extend(self._describe_shot_axis(num_shots))
        return "\n".join(lines)

    def _describe_program_axis(self, items: Sequence[T] | None) -> str:
        """First line of `describe()`'s program-axis block, plus any
        indented `# of ...` rows that can be computed."""
        if self.program_executor is None:
            return "program axis: serial"

        rows: list[tuple[str, int]] = []
        n_chunks = self.n_program_chunks
        if n_chunks is None and items is not None:
            n_chunks = len(items)
        if n_chunks is not None:
            rows.append(("# of program chunks:", n_chunks))
        if items is not None and n_chunks is not None:
            programs_per_chunk = math.ceil(len(items) / n_chunks)
            rows.append(("# of programs/chunk:", programs_per_chunk))
        workers = _executor_worker_count(self.program_executor)
        if n_chunks is not None and workers is not None:
            chunks_per_worker = math.ceil(n_chunks / workers)
            rows.append(("# of chunks/worker:", chunks_per_worker))

        header = (
            f"program axis: {_describe_executor_tag(self.program_executor)}"
        )
        return "\n".join([header, *_indent_rows(rows)])

    def _describe_shot_axis(self, num_shots: int | None) -> list[str]:
        """Lines of `describe()`'s shot-axis block, plus any indented
        `# of ...` rows that can be computed."""
        if self.shot_executor is None:
            return ["shot axis: serial"]
        if isinstance(self.shot_executor, ExecutorSpec):
            header = f"shot axis: {self.shot_executor.describe()}"
        elif callable(self.shot_executor):
            name = getattr(
                self.shot_executor, "__name__", repr(self.shot_executor)
            )
            return [f"shot axis: factory `{name}`"]
        else:
            header = f"shot axis: {_describe_executor_tag(self.shot_executor)}"

        rows: list[tuple[str, int]] = []
        if num_shots is not None:
            rows.append(("# of shots:", num_shots))
        workers = _shot_executor_worker_count(self.shot_executor)
        if num_shots is not None and workers is not None:
            shots_per_worker = math.ceil(num_shots / workers)
            rows.append(("# of shots/worker:", shots_per_worker))

        return [header, *_indent_rows(rows)]


Box = tuple[float, float, float, float]

_PLOT_COLORS = {
    "node": "#4c72b0",
    "worker": "#dd8452",
    "chunk": "#55a868",
    "program": "#c44e52",
    "shot": "#8172b2",
    "idle": "#999999",
}

# Relative horizontal space given to a fully-drawn exemplar worker vs. a
# collapsed "= W0"-style pointer standing in for a duplicate of one --
# see _worker_plan/ParallelStrategy.plot.
_EXEMPLAR_WEIGHT = 1.0
_POINTER_WEIGHT = 0.3


def _split_box(
    box: Box, n: int, horizontal: bool, gap_frac: float = 0.06
) -> list[Box]:
    """Split `box` (`x, y, w, h`) into `n` equal sub-boxes, side by side
    along the width (`horizontal=True`) or stacked along the height, with
    a small gap between siblings so each remains individually visible."""
    return _split_box_weighted(box, [1.0] * n, horizontal, gap_frac)


def _split_box_weighted(
    box: Box,
    weights: Sequence[float],
    horizontal: bool,
    gap_frac: float = 0.06,
) -> list[Box]:
    """Split `box` into `len(weights)` sub-boxes sized proportionally to
    `weights` (an equal split is just every weight being `1.0`), side by
    side along the width (`horizontal=True`) or stacked along the height,
    with a small gap between siblings. Used to give a collapsed
    "= W0"-style pointer (see `_worker_plan`) much less room than the
    exemplar box it points to, rather than an equal share it has no use
    for."""
    x, y, w, h = box
    n = len(weights)
    if n == 0:
        return []
    if n == 1:
        return [box]
    total = sum(weights)
    if horizontal:
        gap = w * gap_frac / n
        available = w - gap * (n - 1)
        boxes = []
        cursor = x
        for weight in weights:
            size = available * weight / total
            boxes.append((cursor, y, size, h))
            cursor += size + gap
        return boxes
    gap = h * gap_frac / n
    available = h - gap * (n - 1)
    boxes = []
    cursor = y
    for weight in weights:
        size = available * weight / total
        boxes.append((x, cursor, w, size))
        cursor += size + gap
    return boxes


def _split_box_frac(
    box: Box, first_frac: float, horizontal: bool, gap_frac: float = 0.08
) -> tuple[Box, Box]:
    """Split `box` into two pieces, the first taking `first_frac` of the
    width (`horizontal=True`) or height, with a small gap between them --
    used wherever a box needs carving into two unequal-sized regions,
    unlike `_split_box`'s same-size splits."""
    x, y, w, h = box
    if horizontal:
        gap = w * gap_frac
        first_size = (w - gap) * first_frac
        second_size = w - gap - first_size
        return (
            (x, y, first_size, h),
            (x + first_size + gap, y, second_size, h),
        )
    gap = h * gap_frac
    first_size = (h - gap) * first_frac
    second_size = h - gap - first_size
    # Stacked boxes read top-to-bottom in the figure, but y grows upward
    # in data coordinates -- the "first" (top) piece is the higher one.
    return (
        (x, y + second_size + gap, w, first_size),
        (x, y, w, second_size),
    )


def _inset_box(box: Box, margin_frac: float = 0.07) -> Box:
    """Shrink `box` by `margin_frac` on every side, leaving a visible
    border around whatever's drawn inside it -- this is what makes each
    level's own rectangle stay visible as a frame around its children,
    rather than being entirely covered by them."""
    x, y, w, h = box
    mx, my = w * margin_frac, h * margin_frac
    return (x + mx, y + my, w - 2 * mx, h - 2 * my)


def _box_center(box: Box) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def _box_top_center(box: Box) -> tuple[float, float]:
    """Top-center point of `box` -- where a curved, "hop over the top"
    sequential arrow starts and lands (see `_curved_arrow`). Deliberately
    the *center*, not the near corner: `arc3`'s curvature scales with the
    distance between its two endpoints, and adjacent boxes' near corners
    are only a small gap apart, which produces a barely-visible sliver of
    a curve -- center-to-center is a full box-width-plus-gap apart, which
    reads as an actual arc."""
    x, y, w, h = box
    return (x + w / 2, y + h)


def _box_bottom_center(box: Box) -> tuple[float, float]:
    """Bottom-center point of `box` -- the "hop below" counterpart to
    `_box_top_center`, used for the sequential program-to-program arrows
    inside a chunk, which arc through the open space below the program
    row (see `_draw_chunk_unit`)."""
    x, y, w, h = box
    return (x + w / 2, y)


def _assign_chunks_to_workers(
    chunk_sizes: Sequence[int], program_workers: int
) -> tuple[list[list[int]], list[int]]:
    """Round-robin assign `chunk_sizes` (one entry per chunk, in dispatch
    order) across `program_workers` slots, mirroring the round-robin
    order chunks are actually submitted in. Returns `(assigned, idle)`:
    `assigned[w]` is the ordered list of chunk sizes worker `w` handles
    (sequentially, one at a time -- a single worker process can only run
    one chunk at once), and `idle` lists worker indices left with no
    chunks at all, whenever there are fewer chunks than workers. A real
    executor's actual runtime assignment depends on scheduling, not a
    fixed formula -- this is a deterministic stand-in good enough for
    illustration, not a claim about exact real-world ordering."""
    n_chunks = len(chunk_sizes)
    active_workers = min(program_workers, n_chunks)
    assigned: list[list[int]] = [[] for _ in range(active_workers)]
    for j, size in enumerate(chunk_sizes):
        assigned[j % active_workers].append(size)
    idle = list(range(active_workers, program_workers))
    return assigned, idle


def _chunk_sizes(
    strategy: "ParallelStrategy",
    items: Sequence[T] | None,
    program_workers: int,
) -> list[int]:
    """Real per-chunk item counts for the diagram, from
    `strategy.make_chunks(items)` when `items` is available (so uneven
    round-robin splits are shown exactly, not just an average) --
    falling back to one placeholder item per chunk when `items` isn't
    given, since there's then no real count to draw at all. A serial
    (unchunked) strategy is treated as a single chunk holding every item,
    matching how it actually runs: one sequential pass in the driver
    process, no dispatch at all."""
    if strategy.program_executor is None:
        return [len(items) if items is not None else 1]
    if items is not None:
        return [len(chunk) for chunk in strategy.make_chunks(items)]
    n_chunks = (
        strategy.n_program_chunks
        if strategy.n_program_chunks is not None
        else program_workers
    )
    return [1] * n_chunks


def _canonical_shapes(
    assigned: list[list[int]],
) -> dict[int, tuple[int, ...]]:
    """For each distinct chunk-count among active workers, the
    elementwise-max ("canonical") per-chunk-index size across every
    worker with that many chunks -- the shape a worker with fewer items
    in one of its chunks (an uneven round-robin remainder, not a
    scheduling gap -- see `_worker_plan`) gets padded up to, so it's
    drawn the same size as its "full" siblings with the shortfall marked
    explicitly instead of just being drawn smaller."""
    by_length: dict[int, list[tuple[int, ...]]] = {}
    for sizes in assigned:
        by_length.setdefault(len(sizes), []).append(tuple(sizes))
    return {
        length: tuple(max(sig[i] for sig in sigs) for i in range(length))
        for length, sigs in by_length.items()
    }


@dataclass
class _WorkerSlot:
    """One worker's rendering plan (see `_worker_plan`): `kind` is
    `"exemplar"` (drawn in full -- the first worker seen with this real
    shape), `"pointer"` (a duplicate of an already-drawn exemplar,
    collapsed to a small label), or the `idle_*` equivalents for workers
    with no chunks at all. `real_sizes`/`padded_sizes` (exemplars only)
    are this worker's true per-chunk item counts and the shape it should
    actually be drawn at (padded up to its chunk-count group's canonical
    shape, per `_canonical_shapes`) -- the difference is rendered as
    hatched "idle program" placeholders, not by shrinking the box.
    `points_to` (pointers only) is the worker index the label refers to.
    """

    kind: str
    real_sizes: tuple[int, ...] = ()
    padded_sizes: tuple[int, ...] = ()
    points_to: int | None = None


def _worker_plan(
    assigned: list[list[int]], idle: list[int], program_workers: int
) -> list[_WorkerSlot]:
    """One `_WorkerSlot` per worker index, `0..program_workers - 1`, in
    original left-to-right order -- collapsing changes what's drawn per
    slot, not the order workers appear in. The first worker with a given
    real chunk-size signature (or the first fully-idle worker) becomes
    that group's exemplar; every later worker sharing the exact same
    signature collapses to a pointer at that exemplar instead of being
    redrawn -- a deliberately conservative rule: only *exact* duplicates
    collapse, never a "close enough" match."""
    canonical = _canonical_shapes(assigned)
    seen_active: dict[tuple[int, ...], int] = {}
    first_idle: int | None = None
    idle_set = set(idle)
    plan: list[_WorkerSlot] = []
    for w in range(program_workers):
        if w in idle_set:
            if first_idle is None:
                first_idle = w
                plan.append(_WorkerSlot(kind="idle_exemplar"))
            else:
                plan.append(
                    _WorkerSlot(kind="idle_pointer", points_to=first_idle)
                )
            continue
        real_sizes = tuple(assigned[w])
        if real_sizes not in seen_active:
            seen_active[real_sizes] = w
            plan.append(
                _WorkerSlot(
                    kind="exemplar",
                    real_sizes=real_sizes,
                    padded_sizes=canonical[len(real_sizes)],
                )
            )
        else:
            plan.append(
                _WorkerSlot(kind="pointer", points_to=seen_active[real_sizes])
            )
    return plan


@dataclass
class _RenderGroup:
    """One thing actually drawn side-by-side in the node, after merging
    consecutive duplicate workers (see `_group_worker_plan`): either a
    single exemplar/idle-exemplar worker (`worker_indices` is one index,
    `slot` is its own `_WorkerSlot`), or a whole run of consecutive
    duplicates of the same exemplar collapsed into one label
    (`worker_indices` is every worker index in that run, `points_to` is
    the exemplar they duplicate)."""

    kind: str
    worker_indices: tuple[int, ...]
    slot: _WorkerSlot | None = None
    points_to: int | None = None


def _group_worker_plan(plan: Sequence[_WorkerSlot]) -> list[_RenderGroup]:
    """Merge consecutive runs of duplicate workers pointing at the same
    exemplar into one `_RenderGroup` each, so the diagram draws one label
    per run (e.g. `"PW1,...,PWk = PW0"`, sandwiched between the real
    program-worker lane dividers on either side of it) instead of one
    redundant label per duplicate worker. Exemplars (real or idle) are
    never merged with each other -- each is always its own group, drawn
    in full."""
    groups: list[_RenderGroup] = []
    i = 0
    n = len(plan)
    while i < n:
        slot = plan[i]
        if slot.kind in ("exemplar", "idle_exemplar"):
            groups.append(
                _RenderGroup(kind=slot.kind, worker_indices=(i,), slot=slot)
            )
            i += 1
            continue
        j = i
        while (
            j < n
            and plan[j].kind == slot.kind
            and plan[j].points_to == slot.points_to
        ):
            j += 1
        groups.append(
            _RenderGroup(
                kind=slot.kind + "_group",
                worker_indices=tuple(range(i, j)),
                points_to=slot.points_to,
            )
        )
        i = j
    return groups


def _curved_arrow(
    ax: "matplotlib.axes.Axes",  # noqa: F821
    start: tuple[float, float],
    end: tuple[float, float],
    rad: float = 0.45,
) -> None:
    """Draw a literal, curved "hop" arrow from `start` to `end`, arcing
    through open space above both -- this diagram's marker for
    "sequential, resource-shared" relationships (each program in a chunk
    handing off to the next; a worker moving on to its next chunk), as
    opposed to a nested/side-by-side box, which marks genuine concurrency
    instead. Curved rather than straight specifically so it never needs
    real gap space between tightly-packed boxes to stay legible -- it
    travels through the room above them instead."""
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="black",
        )
    )


def _draw_caption(
    ax: "matplotlib.axes.Axes",  # noqa: F821
    box: Box,
    text: str,
    *,
    fontsize: float = 6.5,
) -> None:
    """Draw `text` centered in `box`, small and italic -- an explanatory
    caption, as opposed to `_draw_centered_label`'s bold lane labels."""
    x, y, w, h = box
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        fontsize=fontsize,
        ha="center",
        va="center",
        style="italic",
        color="#555555",
    )


def _draw_labeled_box(
    ax: "matplotlib.axes.Axes",  # noqa: F821
    box: Box,
    level: str,
    label: str | None = None,
    *,
    hatched: bool = False,
    fill: bool = True,
    label_fontsize: float = 7,
    label_fontweight: str | None = None,
) -> None:
    """Draw one rectangle of `ParallelStrategy.plot`'s diagram: solid
    fill for a real, concurrently-active box; `hatched=True` for anything
    idle (a whole worker with no chunks, or a single padded-in "idle
    program" slot)."""
    from matplotlib.patches import Rectangle

    x, y, w, h = box
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=_PLOT_COLORS[level] if fill else "none",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.35 if fill else 1.0,
            hatch="///" if hatched else None,
        )
    )
    if label:
        ax.text(
            x + 0.015 * w,
            y + h - 0.03 * h,
            label,
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            va="top",
            ha="left",
        )


def _draw_centered_label(
    ax: "matplotlib.axes.Axes",  # noqa: F821
    box: Box,
    text: str,
    *,
    fontsize: float = 8,
    color: str = "black",
    fontweight: str | None = "bold",
    style: str | None = None,
    rotation: float = 0,
) -> None:
    """Draw `text` centered in `box`, horizontally by default -- the
    common label treatment for anything living in a lane (a
    program-worker lane's own header, or a collapsed duplicate-worker
    label), as opposed to `program`/`chunk`/`node` labels, which caption
    their own box from outside or from a corner instead."""
    cx, cy = _box_center(box)
    ax.text(
        cx,
        cy,
        text,
        fontsize=fontsize,
        color=color,
        fontweight=fontweight,
        style=style,
        ha="center",
        va="center",
        rotation=rotation,
    )


def _duplicate_group_label(indices: Sequence[int], points_to: int) -> str:
    """Label text for a merged run of duplicate workers, e.g. `"PW1 =
    PW0"` for a single duplicate, `"PW1, PW2 = PW0"` for exactly two, or
    `"PW1,...,PW3 = PW0"` for a longer run -- always naming the exemplar
    (`points_to`) the whole run duplicates."""
    if len(indices) == 1:
        heading = f"PW{indices[0]}"
    elif len(indices) == 2:
        heading = f"PW{indices[0]}, PW{indices[1]}"
    else:
        heading = f"PW{indices[0]},...,PW{indices[-1]}"
    return f"{heading} = PW{points_to}"


def _draw_pointer_group(
    ax: "matplotlib.axes.Axes",  # noqa: F821
    box: Box,
    indices: Sequence[int],
    points_to: int,
) -> None:
    """Draw a merged run of duplicate workers (see `_group_worker_plan`)
    as just a label naming the whole range (`_duplicate_group_label`),
    with no box of its own -- it's already visually sandwiched between
    the real program-worker lane dividers drawn on either side of it, so
    a separate enclosure would only be redundant."""
    _draw_centered_label(
        ax, box, _duplicate_group_label(indices, points_to), rotation=90
    )


def _draw_chunk_unit(
    ax: "matplotlib.axes.Axes",  # noqa: F821
    box: Box,
    real_size: int,
    padded_size: int,
    shot_serial: bool,
    shot_workers: int,
) -> None:
    """Draw one chunk's worth of work: a filled background for the whole
    unit, a sequential row of program boxes (connected by curved "hop"
    arrows below, captioned `"(loop over programs)"` when there's more
    than one -- see `_curved_arrow` -- since they run one at a time,
    never concurrently; not individually labeled, since a program's
    position within a chunk is not its overall index among every program
    dispatched -- only true for `PW0`'s very first chunk), and a grid of
    shot boxes nested directly inside each program box -- a single
    "Serial" lane when shots run serially, or one lane per shot worker
    otherwise. When shots aren't serial, dashed lines mark every
    shot-worker lane boundary (both between consecutive shot-box rows and
    framing the topmost/bottommost row), spanning the chunk's full width
    and cutting through every program column, including idle ones (a real
    shot-worker lane still exists there, it just has no real shot box to
    show). `padded_size - real_size` trailing program slots (present when
    this chunk's real item count fell short of its chunk-count group's
    canonical shape -- see `_canonical_shapes`) are drawn as hatched, idle
    placeholders instead of real program boxes, with no arrow connecting
    them, since nothing actually runs there."""
    _draw_labeled_box(ax, box, "chunk", fill=True)
    program_row = _inset_box(box, margin_frac=0.02)
    # Room below the program row for the sequential "hop" arrows to arc
    # through, plus a small caption underneath them.
    program_row, arrow_room = _split_box_frac(
        program_row, 0.87, horizontal=False
    )

    # A slim left-hand gutter, reserved for the SW{i}/"Serial" lane
    # label(s), kept separate from the program columns themselves so a
    # label never overlaps a program/shot box.
    gutter, program_area = _split_box_frac(program_row, 0.06, horizontal=True)

    slot_count = max(padded_size, 1)
    program_boxes = _split_box(
        program_area, slot_count, horizontal=True, gap_frac=0.1
    )

    # Shot-lane boundaries, computed once from the full program row so
    # every program column (real or idle) shares the exact same lane
    # y-extents -- this is what lets the full-width dashed dividers land
    # exactly at the gaps between each program's own shot boxes. A serial
    # shot axis still gets one lane (a single "full shot block"), just
    # labeled "Serial" below instead of "SW0".
    lane_boxes = _split_box(
        program_area,
        1 if shot_serial else shot_workers,
        horizontal=False,
        gap_frac=0.1,
    )

    for i, program_box in enumerate(program_boxes):
        if i < real_size:
            _draw_labeled_box(ax, program_box, "program")
            px, py, pw, ph = program_box
            for lane_box in lane_boxes:
                _, ly, _, lh = lane_box
                shot_box = _inset_box((px, ly, pw, lh), margin_frac=0.06)
                _draw_labeled_box(ax, shot_box, "shot")
        else:
            _draw_labeled_box(ax, program_box, "idle", hatched=True)
    for left, right in zip(
        program_boxes[: real_size - 1], program_boxes[1:real_size]
    ):
        _curved_arrow(
            ax, _box_bottom_center(left), _box_bottom_center(right), rad=0.45
        )
    if real_size > 1:
        _draw_caption(ax, arrow_room, "(loop over programs)")

    # A dashed divider between every consecutive pair of lanes, plus one
    # framing the very top of the topmost lane and one framing the very
    # bottom of the bottommost -- len(lane_boxes) + 1 lines total, each
    # spanning the whole chunk's own outer width (not just the inset
    # program row) so it reaches edge to edge, cutting through every
    # program column. Also draws each lane's "SW{i}"/"Serial" label, in
    # the reserved gutter -- lane_boxes[-1] is the topmost lane (see
    # _split_box_weighted), so SW0 reads top-to-bottom.
    row_x, _, row_w, _ = box

    def _draw_sw_line(y: float) -> None:
        ax.plot(
            [row_x, row_x + row_w],
            [y, y],
            color=_PLOT_COLORS["shot"],
            linewidth=2.2,
            linestyle="--",
        )

    if not shot_serial:
        # A smaller inset than the shot box's own (0.06) leaves a visible
        # gap between the framing line and the box it frames, rather than
        # the line touching the box's edge directly. A serial shot axis
        # has only the one "full shot block" and no real lane boundary to
        # mark, so it gets no lines at all -- just its "Serial" label
        # below.
        top_lane, bottom_lane = lane_boxes[-1], lane_boxes[0]
        _, top_ly, _, top_lh = _inset_box(
            (row_x, top_lane[1], row_w, top_lane[3]), margin_frac=0.03
        )
        _, bottom_ly, _, _ = _inset_box(
            (row_x, bottom_lane[1], row_w, bottom_lane[3]), margin_frac=0.03
        )
        _draw_sw_line(top_ly + top_lh)
        _draw_sw_line(bottom_ly)
        for lower, upper in zip(lane_boxes, lane_boxes[1:]):
            _, ly, _, lh = lower
            _, uy, _, _ = upper
            _draw_sw_line((ly + lh + uy) / 2)
    for k, lane_box in enumerate(lane_boxes):
        sw_index = len(lane_boxes) - 1 - k
        gx, gy, gw, gh = gutter
        lane_label = "Serial" if shot_serial else f"SW{sw_index}"
        ax.text(
            gx + gw * 0.65,
            lane_box[1] + lane_box[3] / 2,
            lane_label,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
            color="black",
        )


def _plot(
    self: "ParallelStrategy",
    items: Sequence[T] | None = None,
    program_workers: int | None = None,
    shot_workers: int | None = None,
    node_count: int = 1,
    legend: bool = True,
) -> "matplotlib.axes.Axes":  # noqa: F821
    """`ParallelStrategy.plot`: diagram of this strategy's real dispatch
    structure. Defined here (as a plain function attached to the class
    below, `ParallelStrategy.plot = _plot`) rather than in the class body
    itself, since it depends on the drawing helpers defined throughout
    the rest of this module. Always builds and returns its own new
    `Axes` on a fresh figure -- deliberately does *not* accept a
    caller-supplied `ax` to draw into, since embedding one of these in a
    caller-managed subplot grid reads as more confusing than helpful once
    a real multi-node (`node_count > 1`) layout is in the picture.

    The node and each program-axis worker (`PW0`, `PW1`, ...; `Serial`
    when `program_executor` is `None`) are labeled at their own top-left
    corner. Workers are drawn as lanes separated by a dashed line running
    the node's full height -- genuinely concurrent, so drawn side by side
    -- but a worker's own chunk(s), and the programs within one chunk, are
    *not* concurrent with each other (a worker runs one chunk at a time;
    one chunk's programs run one at a time, in a `for` loop, inside
    whichever worker picked that chunk up), so they're drawn instead as a
    left-to-right sequence connected by a curved "hop" arrow below each
    pair, with a `"(loop over programs)"` caption underneath -- arcing
    through open space rather than needing real gap space between the
    boxes, so it stays legible even when they're packed tightly. Each
    program's box directly contains its own stack of shot boxes, one per
    shot worker (or a single one labeled `Serial` when `shot_executor` is
    `None`); dashed `"SW0"`/`"SW1"`/... lines then span the chunk's full
    width at the gaps between those shot-box rows, cutting through every
    program column in the chunk (including idle ones) since it really is
    the same resolved shot executor, reused sequentially by every program
    in turn.

    Workers that would otherwise be exact duplicates of each other (the
    common case under round-robin dispatch) are collapsed: the first
    worker with a given real chunk-size signature is drawn in full, and
    every consecutive run of later workers sharing that exact signature
    is merged into one label, e.g. `"PW1,...,PWk = PW0"`, with no box of
    its own -- it's simply sandwiched between the real program-worker
    lane dividers on either side of it (instead of being redrawn, or
    redrawn one small box per duplicate) -- this applies identically to
    fully idle workers (more workers requested than there were chunks to
    hand out; the first is drawn as one hatched box) and to active ones.
    A worker whose own chunk(s) fell short of its chunk-count group's
    largest ("canonical") shape -- an uneven round-robin remainder, e.g.
    one worker's chunk getting 1 item while another's got 2 -- is drawn
    at that larger shape, with the shortfall rendered as hatched, idle
    *program* slots nested inside an otherwise normal worker, rather than
    hatching the entire worker: it did get real work, just less of it.
    See `_worker_plan` for the exact rule.

    `program_workers`/`shot_workers` default to whatever this strategy's
    own executors expose (the same introspection `describe()` uses),
    falling back to `1` when that can't be determined (e.g. a
    `submitit.Executor`, whose real parallelism is scheduler-dependent).
    Pass them explicitly to illustrate a hypothetical worker count
    instead -- e.g. for a backend that isn't actually installed/running
    in the current environment. An explicit `shot_workers` also forces
    real shot lanes to be drawn even when `shot_executor` is `None`,
    rather than a single `Serial` one -- useful for the same
    hypothetical-illustration purpose. Real per-chunk program counts come
    from `make_chunks(items)` when `items` is given (so an uneven
    round-robin split is shown exactly, not just an average).

    `legend=False` omits the level-color legend below the diagram.

    Requires matplotlib (`pip install loqs[visualization]`); the import
    happens inside this function body, not at module level, so the rest
    of this module has no hard plotting dependency.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    strategy = self
    if program_workers is None:
        program_workers = (
            _executor_worker_count(strategy.program_executor)
            if strategy.program_executor is not None
            else None
        ) or 1

    shot_serial = strategy.shot_executor is None and shot_workers is None
    if not shot_serial and shot_workers is None:
        shot_workers = _shot_executor_worker_count(strategy.shot_executor) or 1

    sizes = _chunk_sizes(strategy, items, program_workers)
    assigned, idle = _assign_chunks_to_workers(sizes, program_workers)
    plan = _worker_plan(assigned, idle, program_workers)
    groups = _group_worker_plan(plan)

    _, ax = plt.subplots(figsize=(7, 5.5))

    for i, node_box in enumerate(
        _split_box((0.0, 0.0, 1.0, 1.0), node_count, horizontal=True)
    ):
        _draw_labeled_box(
            ax,
            node_box,
            "node",
            f"Node {i}",
            label_fontsize=8,
            label_fontweight="bold",
        )
        weights = [
            (
                _EXEMPLAR_WEIGHT
                if group.kind in ("exemplar", "idle_exemplar")
                else _POINTER_WEIGHT
            )
            for group in groups
        ]
        inset = _inset_box(node_box, margin_frac=0.02)
        # A blank buffer strip below the node's own top-left label,
        # before the PW gutter row starts, so the two never sit on the
        # same visual row.
        _, below_node_label = _split_box_frac(inset, 0.05, horizontal=False)
        # Every group's own label -- PW{w}/Serial, "PW{w} (idle)", or a
        # collapsed duplicate's "PW1,...,PWk = PW0" -- lives centered in a
        # thin shared gutter row at the top; chunks are laid out in the
        # (larger) row below it. Both rows are split with the same
        # per-group weights, so a group's gutter and content share the
        # same x/width and line up exactly.
        gutter_row, content_row = _split_box_frac(
            below_node_label, 0.05, horizontal=False
        )
        # A larger-than-default gap between adjacent groups gives each
        # lane's own text/boxes (especially an idle worker's hatched
        # box) some breathing room from the dashed divider lines on
        # either side, rather than sitting flush against them.
        full_boxes = _split_box_weighted(
            inset, weights, horizontal=True, gap_frac=0.1
        )
        gutter_boxes = _split_box_weighted(
            gutter_row, weights, horizontal=True, gap_frac=0.1
        )
        content_boxes = _split_box_weighted(
            content_row, weights, horizontal=True, gap_frac=0.1
        )

        # Dashed program-worker lane dividers, one per *interior* boundary
        # between adjacent groups (not at the node's own left/right edges,
        # which already bound the first/last group) -- each spans the
        # node box's full height, edge to edge.
        _, node_y, _, node_h = node_box
        for box in full_boxes[1:]:
            x, _, _, _ = box
            ax.plot(
                [x, x],
                [node_y, node_y + node_h],
                color=_PLOT_COLORS["worker"],
                linewidth=1.5,
                linestyle="--",
            )

        for full_box, gutter_box, content_box, group in zip(
            full_boxes, gutter_boxes, content_boxes, groups
        ):
            if group.kind in ("pointer_group", "idle_pointer_group"):
                _draw_pointer_group(
                    ax, full_box, group.worker_indices, group.points_to
                )
                continue
            w = group.worker_indices[0]
            if group.kind == "idle_exemplar":
                _draw_labeled_box(ax, content_box, "idle", hatched=True)
                _draw_centered_label(ax, gutter_box, f"PW{w} (idle)")
                continue

            slot = group.slot
            worker_label = (
                "Serial" if strategy.program_executor is None else f"PW{w}"
            )
            _draw_centered_label(
                ax,
                gutter_box,
                worker_label,
                fontweight="bold",
            )
            chunk_boxes = _split_box(
                _inset_box(content_box, margin_frac=0.01),
                len(slot.real_sizes),
                horizontal=True,
                gap_frac=0.1,
            )
            for chunk_box, real_size, padded_size in zip(
                chunk_boxes, slot.real_sizes, slot.padded_sizes
            ):
                _draw_chunk_unit(
                    ax,
                    chunk_box,
                    real_size,
                    padded_size,
                    shot_serial,
                    shot_workers,
                )
            for left, right in zip(chunk_boxes, chunk_boxes[1:]):
                _curved_arrow(
                    ax,
                    _box_top_center(left),
                    _box_top_center(right),
                    rad=0.25,
                )

    active_workers = program_workers - len(idle)
    max_chunks_per_worker = max((len(c) for c in assigned), default=0)
    shot_summary = "serial" if shot_serial else f"{shot_workers} worker(s)"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(
        f"{active_workers}/{program_workers} program worker(s) active, "
        f"up to {max_chunks_per_worker} chunk(s)/worker\n"
        f"shots: {shot_summary}",
        fontsize=9,
    )
    if legend:
        boxed_levels = ("chunk", "program", "shot", "idle")
        boxed_labels = ("Chunk", "Program", "Shots", "Idle")
        handles = [
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=_PLOT_COLORS[level],
                alpha=0.35,
                edgecolor="black",
                hatch="///" if level == "idle" else None,
            )
            for level in boxed_levels
        ]
        labels = list(boxed_labels)
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=len(labels),
            frameon=False,
            fontsize=8,
        )
    return ax


# Attached here, rather than in the class body above, since _plot depends
# on drawing helpers defined throughout the rest of this module.
ParallelStrategy.plot = _plot
