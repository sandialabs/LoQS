#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Generic mechanism for running a sequence of items with checkpoint/resume/progress tracking."""

from __future__ import annotations

import base64
import functools
import json
import os
import pickle
import socket
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from tqdm import tqdm

from loqs.internal.serializable import Serializable
from loqs.tools.paralleltools import (
    ParallelStrategy,
    resolve_shot_executor,
    pin_worker_threads,
)

T = TypeVar("T")
R = TypeVar("R")


def run_checkpointed_items(
    items: Sequence[T],
    process_item: Callable[..., R],
    parallel_strategy: ParallelStrategy | None = None,
    desc: str = "Processing items",
    static_kwargs: dict | None = None,
    item_checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    item_key_fn: Callable[[T], str] | None = None,
    on_item_done: Callable[[int, T, R], None] | None = None,
    show_progress: bool = True,
    poll_interval: float = 1.0,
) -> list[R]:
    """Run a list of items with checkpoint/resume/progress tracking.

    Parameters
    ----------
    items : Sequence[T]
        Items to process.
    process_item : Callable[..., R]
        Function to process each item. Signature: (item, index, *, shot_executor, **static_kwargs) -> R
    parallel_strategy : ParallelStrategy | None
        Parallelization strategy. None means serial execution.
    desc : str
        Progress bar description.
    static_kwargs : dict | None
        Static keyword arguments passed to process_item.
    item_checkpoint_dir : str | Path | None
        Directory for checkpointing item results. If set, enables resume capability.
    resume : bool
        If True, resume from prior checkpoint. Raises ValueError if item_checkpoint_dir is None.
    item_key_fn : Callable[[T], str] | None
        Function to get a string key for each item. If provided, maintains an index_map.json.
    on_item_done : Callable[[int, T, R], None] | None
        Callback invoked when an item completes: on_item_done(index, item, result).
    show_progress : bool
        Whether to show a progress bar.
    poll_interval : float
        Polling interval (seconds) for reading journal files during parallel dispatch.

    Returns
    -------
    list[R]
        Results in the original items order.
    """
    # Validation
    if resume and item_checkpoint_dir is None:
        raise ValueError("resume=True requires item_checkpoint_dir to be set")

    if item_checkpoint_dir is not None:
        item_checkpoint_dir = Path(item_checkpoint_dir)
        if not resume and item_checkpoint_dir.exists():
            if any(item_checkpoint_dir.iterdir()):
                raise FileExistsError(
                    f"{item_checkpoint_dir} exists with content and resume=False"
                )
        item_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Item indexing/identity
    if item_key_fn is not None and item_checkpoint_dir is not None:
        index_map = _load_or_init_index_map(item_checkpoint_dir)
        items_with_index = _assign_indices_with_keys(
            items, item_key_fn, index_map
        )
        _save_index_map_atomic(item_checkpoint_dir, index_map)
    else:
        items_with_index = [(i, item) for i, item in enumerate(items)]

    # Read prior progress
    done: dict[int, R] = {}
    if item_checkpoint_dir is not None:
        done = _read_journal_files(item_checkpoint_dir)
        if not resume:
            done = {}

    # Replay already-done items
    for index, item in items_with_index:
        if index in done and on_item_done is not None:
            on_item_done(index, item, done[index])

    # Determine remaining work
    remaining = [
        (index, item) for index, item in items_with_index if index not in done
    ]

    # Progress bar setup
    pbar = None
    if show_progress:
        pbar = tqdm(total=len(items), initial=len(done), desc=desc)

    try:
        # Dispatch
        newly_computed: dict[int, Any] = {}
        if not remaining:
            # Nothing to do, skip dispatch
            pass
        elif parallel_strategy is None or not parallel_strategy.is_chunked:
            # Serial execution
            newly_computed = _run_serial(
                remaining,
                process_item,
                static_kwargs or {},
                item_checkpoint_dir,
                on_item_done,
                parallel_strategy,
                pbar,
            )
        else:
            # Parallel execution
            newly_computed = _run_parallel(
                remaining,
                process_item,
                static_kwargs or {},
                item_checkpoint_dir,
                on_item_done,
                items_with_index,
                parallel_strategy,
                pbar,
                poll_interval,
                done.keys(),
            )

        # Final assembly - read authoritative results from journal if available
        if item_checkpoint_dir is not None:
            final_done = _read_journal_files(item_checkpoint_dir)
        else:
            # No checkpointing, combine prior done + newly computed
            final_done = done.copy()
            final_done.update(newly_computed)

        # Build result list in original order
        result_list = []
        for index, _ in items_with_index:
            if index not in final_done:
                raise RuntimeError(
                    f"Item {index} is missing from final results"
                )
            result_list.append(final_done[index])

        return result_list

    finally:
        if pbar is not None:
            pbar.close()


class ProgramRunner(Serializable):
    """Base class for crash-recoverable program runners bundling config fields
    and a template-method `run()` for checkpoint/resume/progress.

    Subclasses implement hook methods to define their specific work logic.
    Whether a call resumes a prior run is inferred entirely from
    `item_checkpoint_dir`'s own on-disk state (see `run()`) -- there is no
    separate `resume` flag to pass.
    """

    _SERIALIZE_ATTRS = [
        "parallel_strategy",
        "item_checkpoint_dir",
        "force_resume",
        "checkpoint_batch_size",
        "shot_checkpoint_dir",
        "lazy_loading_enabled",
    ]

    def __init__(
        self,
        parallel_strategy: ParallelStrategy | None = None,
        item_checkpoint_dir: str | Path | None = None,
        force_resume: bool = False,
        checkpoint_batch_size: int | None = None,
        shot_checkpoint_dir: str | Path | None = None,
        lazy_loading_enabled: bool = True,
    ):
        self.parallel_strategy = parallel_strategy
        self.item_checkpoint_dir = (
            Path(item_checkpoint_dir)
            if item_checkpoint_dir is not None
            else None
        )
        self.force_resume = force_resume
        self.checkpoint_batch_size = checkpoint_batch_size
        self.shot_checkpoint_dir = (
            Path(shot_checkpoint_dir)
            if shot_checkpoint_dir is not None
            else None
        )
        self.lazy_loading_enabled = lazy_loading_enabled
        self._validate_checkpoint_kwargs()

    def _get_encoding_attr(
        self, attr: str, ignore_no_serialize_flags: bool = False
    ) -> Any:
        """Convert Path objects to strings for serialization."""
        if attr in ("item_checkpoint_dir", "shot_checkpoint_dir"):
            val = getattr(self, attr)
            return str(val) if val is not None else None
        return super()._get_encoding_attr(attr, ignore_no_serialize_flags)

    @classmethod
    def _from_decoded_attrs(cls, attr_dict: dict[str, Any]) -> "ProgramRunner":
        """Reconstruct from decoded attributes, converting strings back to Paths."""
        # Convert path strings back to Path objects
        if attr_dict.get("item_checkpoint_dir") is not None:
            attr_dict["item_checkpoint_dir"] = Path(
                attr_dict["item_checkpoint_dir"]
            )
        if attr_dict.get("shot_checkpoint_dir") is not None:
            attr_dict["shot_checkpoint_dir"] = Path(
                attr_dict["shot_checkpoint_dir"]
            )
        return super()._from_decoded_attrs(attr_dict)

    def _validate_checkpoint_kwargs(self) -> None:
        """Validate that checkpoint_batch_size requires shot_checkpoint_dir."""
        if (
            self.checkpoint_batch_size is not None
            and self.shot_checkpoint_dir is None
        ):
            raise ValueError(
                "checkpoint_batch_size requires shot_checkpoint_dir to be set"
            )

    def run(self) -> Any:
        """Run the program with checkpoint/resume support.

        Whether to resume is always inferred from `item_checkpoint_dir`'s
        own on-disk state, never from a caller-supplied flag: no content
        means a fresh run; a `runner.h5` (this method's own crash-
        recovery snapshot, written here at call-start, before any work is
        dispatched) means continuing that prior run, subject to a mismatch
        check against the stored config (`force_resume` bypasses a real
        mismatch); unrelated content with no `runner.h5` always raises,
        since there's nothing to safely continue from. A crash mid-run can
        be recovered from via
        `type(self).read(item_checkpoint_dir / "runner.h5").run()`.
        """
        resuming = False
        if self.item_checkpoint_dir is not None:
            has_content = self.item_checkpoint_dir.exists() and any(
                self.item_checkpoint_dir.iterdir()
            )
            runner_path = self.item_checkpoint_dir / "runner.h5"
            if has_content:
                if not runner_path.exists():
                    raise FileExistsError(
                        f"{self.item_checkpoint_dir} exists with content "
                        "that isn't a recognized checkpoint (no runner.h5)."
                    )
                stored = type(self).read(runner_path)
                mismatches = [
                    f
                    for f in self._mismatch_check_fields()
                    if getattr(self, f) != getattr(stored, f)
                ]
                if mismatches and not self.force_resume:
                    raise ValueError(
                        f"Cannot resume: stored config differs in "
                        f"{', '.join(mismatches)}. Pass force_resume=True to resume anyway."
                    )
                resuming = True
            self.item_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.write(runner_path)
            # The directory now always has at least runner.h5 in it
            # (just written above, or already present from a prior/
            # recovered run) -- the checks above already establish it's
            # safe to proceed, so tell run_checkpointed_items to trust
            # this directory's own journal state unconditionally, rather
            # than tripping its own pre-existing-content guard on the
            # runner.h5 file this method itself just wrote.
            resuming = True

        result_list = run_checkpointed_items(
            items=self._get_items(),
            process_item=self._process_item_fn(),
            parallel_strategy=self.parallel_strategy,
            desc=self._desc(),
            static_kwargs=self._static_kwargs(),
            item_checkpoint_dir=self.item_checkpoint_dir,
            resume=resuming,
            item_key_fn=self._item_key_fn(),
            on_item_done=self._make_on_item_done(),
        )
        return self._finalize(result_list)

    # Hook methods -- subclasses implement these
    def _get_items(self) -> Sequence:
        """Return list of items to process."""
        raise NotImplementedError

    def _process_item_fn(self) -> Callable:
        """Return a plain top-level function reference for process_item."""
        raise NotImplementedError

    def _static_kwargs(self) -> dict[str, Any]:
        """Return dict of static kwargs to pass to process_item."""
        raise NotImplementedError

    def _make_on_item_done(self) -> Callable[[int, Any, Any], None] | None:
        """Return a closure for on_item_done callback, or None."""
        raise NotImplementedError

    def _finalize(self, result_list: list) -> Any:
        """Return final result from result_list."""
        raise NotImplementedError

    def _item_key_fn(self) -> Callable[[Any], str] | None:
        """Return item_key_fn or None (uses plain position if None)."""
        return None

    def _desc(self) -> str:
        """Return description string for progress bar."""
        return "Processing items"

    def _mismatch_check_fields(self) -> list[str]:
        """Return list of field names to compare for resume mismatch check."""
        return []


def _load_or_init_index_map(
    checkpoint_dir: Path,
) -> dict[str, int]:
    """Load index_map.json if it exists, else return empty dict."""
    index_map_path = checkpoint_dir / "index_map.json"
    if index_map_path.exists():
        with open(index_map_path) as f:
            return json.load(f)
    return {}


def _assign_indices_with_keys(
    items: Sequence[T],
    item_key_fn: Callable[[T], str],
    index_map: dict[str, int],
) -> list[tuple[int, T]]:
    """Assign indices to items using item_key_fn, reusing/extending index_map."""
    items_with_index = []
    for item in items:
        key = item_key_fn(item)
        if key not in index_map:
            index_map[key] = len(index_map)
        index = index_map[key]
        items_with_index.append((index, item))
    return items_with_index


def _save_index_map_atomic(
    checkpoint_dir: Path, index_map: dict[str, int]
) -> None:
    """Atomically save index_map to checkpoint_dir/index_map.json."""
    index_map_path = checkpoint_dir / "index_map.json"
    # Write to a temp file in the same directory, then rename atomically
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=checkpoint_dir,
        delete=False,
        suffix=".tmp",
    ) as f:
        json.dump(index_map, f)
        temp_path = f.name

    try:
        os.replace(temp_path, index_map_path)
    except Exception:
        os.unlink(temp_path)
        raise


def _read_journal_files(checkpoint_dir: Path) -> dict[int, Any]:
    """Read all journal_*.jsonl files and return {index: result} dict."""
    done: dict[int, Any] = {}
    for journal_file in sorted(checkpoint_dir.glob("journal_*.jsonl")):
        with open(journal_file) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                index = entry["index"]
                result_b64 = entry["result"]
                result = pickle.loads(base64.b64decode(result_b64))
                done[index] = result
    return done


def _worker_id() -> str:
    """Get hostname_pid worker identifier."""
    return f"{socket.gethostname()}_{os.getpid()}"


def _serialize_result(result: Any) -> str:
    """Serialize result to base64-encoded pickle string."""
    return base64.b64encode(pickle.dumps(result)).decode("ascii")


def _run_serial(
    remaining: list[tuple[int, T]],
    process_item: Callable[..., Any],
    static_kwargs: dict[str, Any],
    item_checkpoint_dir: Path | None,
    on_item_done: Callable[[int, T, Any], None] | None,
    parallel_strategy: ParallelStrategy | None,
    pbar: Any,
) -> dict[int, Any]:
    """Execute remaining items serially. Returns {index: result} for in-memory results."""
    shot_executor = resolve_shot_executor(
        parallel_strategy.shot_executor
        if parallel_strategy is not None
        else None
    )
    results_dict: dict[int, Any] = {}

    # Open journal file for serial execution (self-journaling)
    journal_file = None
    if item_checkpoint_dir is not None:
        journal_path = item_checkpoint_dir / f"journal_{_worker_id()}.jsonl"
        journal_file = open(journal_path, "a")

    try:
        for index, item in remaining:
            result = process_item(
                item,
                index,
                shot_executor=shot_executor,
                **static_kwargs,
            )

            results_dict[index] = result

            if journal_file is not None:
                entry = {
                    "index": index,
                    "result": _serialize_result(result),
                }
                journal_file.write(json.dumps(entry) + "\n")
                journal_file.flush()

            if on_item_done is not None:
                on_item_done(index, item, result)

            if pbar is not None:
                pbar.update(1)
    finally:
        if journal_file is not None:
            journal_file.close()

    return results_dict


def _run_parallel(
    remaining: list[tuple[int, T]],
    process_item: Callable[..., Any],
    static_kwargs: dict[str, Any],
    item_checkpoint_dir: Path | None,
    on_item_done: Callable[[int, T, Any], None] | None,
    items_with_index: list[tuple[int, T]],
    parallel_strategy: ParallelStrategy,
    pbar: Any,
    poll_interval: float,
    already_done_indices: Iterable[int] | None = None,
) -> dict[int, Any]:
    """Execute remaining items in parallel with checkpointing and polling.

    Returns {index: result} for in-memory results (when no checkpointing).
    """
    # Build a mapping of index -> item for use in on_poll callback
    items_map = {index: item for index, item in items_with_index}

    # Track observed indices to avoid double-counting, seeding with already-done indices
    observed_indices: set[int] = set(already_done_indices or [])

    def on_poll() -> None:
        """Poll journal files and call on_item_done for new results."""
        if item_checkpoint_dir is None:
            return

        done = _read_journal_files(item_checkpoint_dir)
        for index in sorted(done.keys()):
            if index not in observed_indices:
                observed_indices.add(index)
                result = done[index]
                if index in items_map:
                    item = items_map[index]
                    if on_item_done is not None:
                        on_item_done(index, item, result)
                    if pbar is not None:
                        pbar.update(1)

    # Each chunk worker resolves this itself; only the raw value is forwarded here.
    shot_executor = (
        parallel_strategy.shot_executor
        if parallel_strategy is not None
        else None
    )

    # Make chunks and dispatch
    chunks = parallel_strategy.make_chunks(remaining)
    worker = functools.partial(
        _generic_chunk_worker,
        process_item=process_item,
        static_kwargs=static_kwargs,
        item_checkpoint_dir=item_checkpoint_dir,
        shot_executor=shot_executor,
    )

    # Dispatch with on_poll callback
    on_poll_callback = None
    if (
        item_checkpoint_dir is not None
        or on_item_done is not None
        or pbar is not None
    ):
        on_poll_callback = on_poll

    chunk_results_list = parallel_strategy.dispatch(
        worker,
        chunks,
        desc="Processing chunks",
        on_poll=on_poll_callback,
        poll_interval=poll_interval,
    )

    # Final poll to catch any remaining items
    if on_poll_callback is not None:
        on_poll()

    # Build newly_computed dict from chunk results (fallback/safety net when no checkpointing)
    newly_computed: dict[int, Any] = {}
    for chunk_results in chunk_results_list:
        for index, result in chunk_results:
            newly_computed[index] = result

    # For any items not already observed via journal polling (i.e., when item_checkpoint_dir
    # is None), invoke on_item_done now so callers can collect results via the callback
    for index, result in newly_computed.items():
        if index not in observed_indices:
            observed_indices.add(index)
            if index in items_map:
                item = items_map[index]
                if on_item_done is not None:
                    on_item_done(index, item, result)
                if pbar is not None:
                    pbar.update(1)

    return newly_computed


def _generic_chunk_worker(
    chunk: list[tuple[int, T]],
    process_item: Callable[..., Any],
    static_kwargs: dict[str, Any],
    item_checkpoint_dir: Path | None,
    shot_executor: Any,
) -> list[tuple[int, Any]]:
    """Worker function for parallel execution of a chunk.

    Returns list of (index, result) tuples.
    """
    pin_worker_threads()
    shot_executor = resolve_shot_executor(shot_executor)

    # Open journal file for this worker
    journal_file = None
    if item_checkpoint_dir is not None:
        journal_path = item_checkpoint_dir / f"journal_{_worker_id()}.jsonl"
        journal_file = open(journal_path, "a")

    results = []
    try:
        for index, item in chunk:
            result = process_item(
                item,
                index,
                shot_executor=shot_executor,
                **static_kwargs,
            )

            if journal_file is not None:
                entry = {
                    "index": index,
                    "result": _serialize_result(result),
                }
                journal_file.write(json.dumps(entry) + "\n")
                journal_file.flush()

            results.append((index, result))
    finally:
        if journal_file is not None:
            journal_file.close()

    return results
