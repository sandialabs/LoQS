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

import copy
import functools
import h5py
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from tqdm import tqdm

from loqs.internal import worker_id
from loqs.internal.serializable import Serializable
from loqs.internal.streamingmerge import (
    merge_dict_attr,
    iter_dict_attr_entries,
)
from loqs.tools.paralleltools import (
    ParallelStrategy,
    resolve_shot_executor,
    pin_worker_threads,
)

T = TypeVar("T")
R = TypeVar("R")


def _resolve_items_with_index(
    items: Sequence[T],
    precomputed_indices: Sequence[int] | None,
) -> list[tuple[int, T]]:
    """Pair each item with its index: caller-precomputed (a
    `MultiProgramRunner`-driven call) if given, else plain position."""
    if precomputed_indices is not None:
        return list(zip(precomputed_indices, items))
    return [(i, item) for i, item in enumerate(items)]


class MultiProgramRunner(Serializable):
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
        "index_map",
        "_reduced_results",
        "keep_shot_results",
        "_program_results",
        "poll_interval",
        "show_progress",
    ]

    _NO_COLLAPSE_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {"_reduced_results", "_program_results"}
    )

    def __init__(
        self,
        parallel_strategy: ParallelStrategy | None = None,
        item_checkpoint_dir: str | Path | None = None,
        force_resume: bool = False,
        checkpoint_batch_size: int | None = None,
        shot_checkpoint_dir: str | Path | None = None,
        lazy_loading_enabled: bool = True,
        index_map: dict[str, int] | None = None,
        keep_shot_results: bool = False,
        poll_interval: float = 1.0,
        show_progress: bool = True,
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
        self.index_map = index_map
        self._reduced_results: dict[int, Any] = {}
        self.keep_shot_results = keep_shot_results
        self._program_results: dict[int, Any] = {}
        self.poll_interval = poll_interval
        self.show_progress = show_progress
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
    def _from_decoded_attrs(
        cls, attr_dict: dict[str, Any]
    ) -> "MultiProgramRunner":
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
        # Extract internal fields that are not constructor parameters
        # (must be set directly on the instance, not passed to __init__)
        index_map = attr_dict.pop("index_map", None)
        reduced_results = attr_dict.pop("_reduced_results", None)
        program_results = attr_dict.pop("_program_results", None)
        # Reconstruct with constructor parameters only
        obj = super()._from_decoded_attrs(attr_dict)
        # Restore internal state directly on the instance
        obj.index_map = index_map
        obj._reduced_results = (
            reduced_results if reduced_results is not None else {}
        )
        obj._program_results = (
            program_results if program_results is not None else {}
        )
        return obj

    def _validate_checkpoint_kwargs(self) -> None:
        """Validate checkpoint-related configuration constraints."""
        if (
            self.checkpoint_batch_size is not None
            and self.shot_checkpoint_dir is None
        ):
            raise ValueError(
                "checkpoint_batch_size requires shot_checkpoint_dir to be set"
            )
        if self.keep_shot_results and self.item_checkpoint_dir is None:
            raise ValueError(
                "keep_shot_results requires item_checkpoint_dir to be set"
            )
        if self.keep_shot_results and self.checkpoint_batch_size is None:
            raise ValueError(
                "keep_shot_results requires checkpoint_batch_size (and "
                "shot_checkpoint_dir) to be set, so kept results are read "
                "back from each item's own on-disk shot checkpoint rather "
                "than held fully in memory for every item at once"
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
        stored = None
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
            # runner.h5 now always exists, so tell _run_dispatch
            # to trust this directory's state rather than tripping its own
            # pre-existing-content guard on the file just written above.
            resuming = True

        # Seed _reduced_results from prior run if resuming, to avoid duplicates
        # when replaying already-done items (merge_dict_attr is append-only)
        if stored is not None:
            self._reduced_results = dict(stored._reduced_results)

        # Pre-assign indices via item_key_fn, adopting the persisted map
        # from `stored` first so a fresh resumed instance doesn't reassign
        # indices out from under already-checkpointed work.
        precomputed_indices = None
        if self._item_key_fn() is not None:
            key_fn = self._item_key_fn()
            if self.index_map is None:
                self.index_map = (
                    dict(stored.index_map)
                    if stored is not None and stored.index_map is not None
                    else {}
                )
            items_with_index = _assign_indices_with_keys(
                self._get_items(), key_fn, self.index_map
            )
            precomputed_indices = [idx for idx, _ in items_with_index]
            # Update runner.h5 with the now-populated index_map
            if self.item_checkpoint_dir is not None:
                self.write(self.item_checkpoint_dir / "runner.h5")

        result_list = self._run_dispatch(
            items=self._get_items(),
            precomputed_indices=precomputed_indices,
            resuming=resuming,
        )
        return self._finalize(result_list)

    def _run_dispatch(
        self,
        items: Sequence[T],
        precomputed_indices: Sequence[int] | None,
        resuming: bool,
    ) -> list:
        """Dispatch item processing with checkpoint/resume/progress tracking:
        item indexing, prior-progress replay, serial/parallel dispatch, and
        final result assembly in original item order."""
        # Item indexing/identity
        items_with_index = _resolve_items_with_index(
            items, precomputed_indices
        )

        # Read prior progress
        done: dict[int, Any] = {}
        if self.item_checkpoint_dir is not None:
            done = _read_worker_files(self.item_checkpoint_dir)
            if not resuming:
                done = {}

        # Compute on_item_done callback once and reuse it throughout
        on_item_done = self._make_on_item_done()

        # Replay already-done items
        for index, item in items_with_index:
            if index in done and on_item_done is not None:
                on_item_done(index, item, done[index])

        # Determine remaining work
        remaining = [
            (index, item)
            for index, item in items_with_index
            if index not in done
        ]

        # Progress bar setup
        pbar = None
        shots_pbar = None
        if self.show_progress:
            pbar = tqdm(total=len(items), initial=len(done), desc=self._desc())

        # Shots bar: parallel dispatch + checkpointing only, gated on
        # self.show_progress like the items pbar above.
        num_shots_for_progress = self._num_shots_for_progress()
        is_parallel_dispatch = (
            self.parallel_strategy is not None
            and self.parallel_strategy.is_chunked
        )
        show_shots_bar = (
            self.show_progress
            and is_parallel_dispatch
            and self.checkpoint_batch_size is not None
            and self.shot_checkpoint_dir is not None
            and num_shots_for_progress is not None
        )

        if show_shots_bar:
            shots_pbar = tqdm(
                total=len(items) * num_shots_for_progress,
                initial=len(done) * num_shots_for_progress,
                desc="Shots",
            )

        # One-time notice if shot progress can't be shown, gated on
        # self.show_progress like the bar itself above.
        if (
            self.show_progress
            and num_shots_for_progress is not None
            and not show_shots_bar
            and is_parallel_dispatch
            and (
                self.checkpoint_batch_size is None
                or self.shot_checkpoint_dir is None
            )
        ):
            print(
                "Shot-level progress reporting requires checkpoint_batch_size"
                " and shot_checkpoint_dir to be set; showing item-level"
                " progress only."
            )

        try:
            # Dispatch
            newly_computed: dict[int, Any] = {}
            if not remaining:
                # Nothing to do, skip dispatch
                pass
            elif (
                self.parallel_strategy is None
                or not self.parallel_strategy.is_chunked
            ):
                # Serial execution
                newly_computed = _run_serial(
                    remaining,
                    self._process_item_fn(),
                    self._static_kwargs() or {},
                    self.item_checkpoint_dir,
                    on_item_done,
                    self.parallel_strategy,
                    pbar,
                    self.keep_shot_results,
                    self._shot_checkpoint_subdir,
                )
            else:
                # Parallel execution: create a snapshot for pickling
                # (same pattern as NoiseSweepRunner._static_kwargs)
                runner_snapshot = copy.copy(self)
                runner_snapshot.parallel_strategy = None
                newly_computed = _run_parallel(
                    remaining,
                    self._process_item_fn(),
                    self._static_kwargs() or {},
                    self.item_checkpoint_dir,
                    on_item_done,
                    items_with_index,
                    self.parallel_strategy,
                    pbar,
                    self.poll_interval,
                    done.keys(),
                    self.keep_shot_results,
                    runner_snapshot._shot_checkpoint_subdir,
                    shots_pbar,
                    num_shots_for_progress,
                )

            # Final assembly - read authoritative results from worker files if available
            if self.item_checkpoint_dir is not None:
                final_done = _read_worker_files(
                    self.item_checkpoint_dir, attr_name="results"
                )
                # Consolidate _program_results if keep_shot_results is enabled
                if self.keep_shot_results:
                    _consolidate_program_results(self.item_checkpoint_dir)
                    if self.lazy_loading_enabled:
                        runner_path = self.item_checkpoint_dir / "runner.h5"
                        self._program_results = {
                            index: self._make_lazy_program_results(
                                runner_path, index
                            )
                            for index in final_done.keys()
                        }
                    else:
                        self._program_results = _read_worker_files(
                            self.item_checkpoint_dir,
                            attr_name="_program_results",
                        )
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
            if shots_pbar is not None:
                shots_pbar.close()

    def _make_lazy_program_results(self, runner_file: Path, index: int) -> Any:
        """Construct a lazy-loading ProgramResults for a nested entry.

        Creates a ProgramResults configured to load shots from the runner.h5's
        own `_program_results` dict attribute at the given index, with
        lazy_loading_enabled=True.

        Parameters
        ----------
        runner_file : Path
            Path to the runner.h5 file.
        index : int
            Integer key in runner.h5's `_program_results` dict attribute.

        Returns
        -------
        ProgramResults
            A new ProgramResults configured for nested lazy loading.
        """
        from loqs.core.programresults import ProgramResults

        pr = ProgramResults(lazy_loading_enabled=True)
        pr._set_nested_shot_source(runner_file, index)
        # Set checkpoint_dir so nested loading works
        if self.item_checkpoint_dir is not None:
            pr._checkpoint_dir = self.item_checkpoint_dir
        return pr

    def _merge_reduced_result(self, index: int, value: Any) -> None:
        """Merge one item's reduced result into `_reduced_results`, persisted
        incrementally into `runner.h5` via the streaming-merge primitives.

        If the index is already present in `self._reduced_results`, returns
        immediately without writing to disk (avoids duplicate entries when
        replaying already-done items on resume, since merge_dict_attr is
        append-only). Otherwise, updates in memory and persists to disk.
        """
        # Check if already present to avoid duplicates on resume
        if index in self._reduced_results:
            return

        # Update in-memory dict first
        self._reduced_results[index] = value

        # Persist to disk if checkpointing is enabled
        if self.item_checkpoint_dir is None:
            return

        runner_path = self.item_checkpoint_dir / "runner.h5"
        with h5py.File(runner_path, "a") as f:
            obj_grp = _get_runner_object_group(f)
            merge_dict_attr(
                obj_grp,
                "_reduced_results",
                [(index, value)],
                key_use_dataset=True,
                value_use_dataset=False,
            )

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

    def _shot_checkpoint_subdir(self, index: int) -> Path | None:
        """Return the checkpoint directory for a specific item's shots, or None.

        Default implementation returns None (no per-item shot subdirectories).
        Subclasses can override to provide item-specific checkpoint paths.

        Parameters
        ----------
        index : int
            The item index being processed.

        Returns
        -------
        Path | None
            A checkpoint directory for this item's shots, or None.
        """
        return None

    def _num_shots_for_progress(self) -> int | None:
        """Return the number of shots per item for progress reporting, or None.

        Default implementation returns `self.num_shots` if it exists, None
        otherwise. This supports fine-grained shot-level progress tracking
        in the dispatch progress bar when parallel dispatch with checkpointing
        is enabled.

        Returns
        -------
        int | None
            The number of shots per item, or None if not applicable.
        """
        return getattr(self, "num_shots", None)


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


def _read_worker_files(
    checkpoint_dir: Path, attr_name: str = "results"
) -> dict[int, Any]:
    """Read all worker_*_runner.h5 files and return {index: result} dict.

    Reads the specified attribute from each worker file and merges them into
    a single dict. Transient HDF5 lock conflicts are silently skipped (those
    worker files will be retried on the next poll tick or final assembly pass).

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing worker_*_runner.h5 files.
    attr_name : str, optional
        Name of the dict attribute to read from each worker file.
        Default is "results".

    Returns
    -------
    dict[int, Any]
        Merged {index: result} dict from all worker files.
    """
    done: dict[int, Any] = {}
    for worker_file in sorted(checkpoint_dir.glob("worker_*_runner.h5")):
        try:
            with h5py.File(worker_file, "r") as f:
                for key, value in iter_dict_attr_entries(f, attr_name):
                    done[key] = value
        except (BlockingIOError, OSError):
            # Transient lock conflict (e.g., concurrent writer opening/closing
            # the file) -- skip this file for now, it will be retried
            continue
    return done


def _get_runner_object_group(f: h5py.File) -> h5py.Group:
    """Navigate to the MultiProgramRunner object group inside runner.h5."""
    root = f[next(iter(f.keys()))]
    if len(root.keys()) > 0:
        first_child = root[next(iter(root.keys()))]
        if (
            isinstance(first_child, h5py.Group)
            and first_child.attrs.get("encode_type") == "Serializable"
        ):
            return first_child
    return root


def _consolidate_program_results(checkpoint_dir: Path) -> None:
    """Consolidate _program_results from all worker files into runner.h5.

    Streams one item at a time from worker files into runner.h5's object group,
    keeping memory bounded. Skips indices already present in runner.h5's
    _program_results attribute to avoid duplicate entries on resume.
    """
    runner_path = checkpoint_dir / "runner.h5"
    if not runner_path.exists():
        return

    with h5py.File(runner_path, "a") as out_f:
        if len(out_f.keys()) == 0:
            return
        out_root = _get_runner_object_group(out_f)
        existing_keys: set[int] = set()
        if "_program_results" in out_root:
            pr_grp = out_root["_program_results"]
            if (
                "dict" in pr_grp
                and "keys" in pr_grp["dict"]
                and "iterable" in pr_grp["dict/keys"]
            ):
                keys_iterable = pr_grp["dict/keys/iterable"]
                storage_format = keys_iterable.attrs.get(
                    "storage_format", "groups"
                )
                from loqs.internal.streamingmerge import _read_iterable_side

                existing_keys = set(
                    _read_iterable_side(
                        keys_iterable, storage_format, decode_cache=None
                    )
                )

        for worker_file in sorted(checkpoint_dir.glob("worker_*_runner.h5")):
            try:
                with h5py.File(worker_file, "r") as in_f:
                    if "_program_results" in in_f:
                        for key, pr in iter_dict_attr_entries(
                            in_f, "_program_results"
                        ):
                            if key in existing_keys:
                                continue
                            merge_dict_attr(
                                out_root,
                                "_program_results",
                                [(key, pr)],
                                encode_cache={},
                                key_use_dataset=True,
                                value_use_dataset=False,
                            )
                            existing_keys.add(key)
            except (BlockingIOError, OSError):
                continue


def _write_dict_entry_with_retry(
    worker_file_path: Path,
    attr_name: str,
    index: int,
    value: Any,
    max_retries: int = 5,
) -> None:
    """Write a single entry to a dict attribute in a worker file with retry logic.

    Handles transient HDF5 locking issues via exponential backoff.

    Parameters
    ----------
    worker_file_path : Path
        Path to the worker_*_runner.h5 file.
    attr_name : str
        Name of the dict attribute (e.g., "results", "_program_results").
    index : int
        The key for the entry.
    value : Any
        The value to store.
    max_retries : int, optional
        Maximum number of retry attempts (default 5, ~0.15s total delay).
    """
    import time

    for attempt in range(max_retries):
        try:
            with h5py.File(worker_file_path, "a") as f:
                merge_dict_attr(
                    f,
                    attr_name,
                    [(index, value)],
                    encode_cache={},
                    key_use_dataset=True,
                    value_use_dataset=False,
                )
            break
        except (BlockingIOError, OSError):
            if attempt < max_retries - 1:
                time.sleep(0.01 * (2**attempt))  # Exponential backoff
            else:
                raise


def _write_current_item_index_with_retry(
    worker_file_path: Path,
    index: int,
    max_retries: int = 5,
) -> None:
    """Write current_item_index attribute to a worker file with retry logic.

    Handles transient HDF5 locking issues via exponential backoff.
    Overwrites any prior value.

    Parameters
    ----------
    worker_file_path : Path
        Path to the worker_*_runner.h5 file.
    index : int
        The current item index being processed.
    max_retries : int, optional
        Maximum number of retry attempts (default 5, ~0.15s total delay).
    """
    import time

    for attempt in range(max_retries):
        try:
            with h5py.File(worker_file_path, "a") as f:
                f.attrs["current_item_index"] = index
            break
        except (BlockingIOError, OSError):
            if attempt < max_retries - 1:
                time.sleep(0.01 * (2**attempt))  # Exponential backoff
            else:
                raise


def _resolve_kept_program_results(
    index: int,
    shot_checkpoint_subdir: Callable[[int], Path | None] | None,
    in_memory_pr: Any,
) -> Any:
    """Resolve ProgramResults from checkpoint or in-memory source.

    Tries to load from checkpoint first (if a subdir path is available),
    falls back to the in-memory value from process_item, returns None
    if neither is available.

    Parameters
    ----------
    index : int
        The item index.
    shot_checkpoint_subdir : Callable[[int], Path | None] | None
        Hook method returning per-item checkpoint directory, or None.
    in_memory_pr : Any
        The in-memory ProgramResults from process_item, or None.

    Returns
    -------
    Any
        The resolved ProgramResults, or None.
    """
    pr = None
    # First try to load from checkpoint if a path is available
    if shot_checkpoint_subdir is not None:
        shot_dir = shot_checkpoint_subdir(index)
        if shot_dir is not None:
            from loqs.core.programresults import ProgramResults

            pr = ProgramResults()
            pr.load_checkpoint(checkpoint_dir=shot_dir)
    # Otherwise use the in-memory one from process_item
    if pr is None:
        pr = in_memory_pr
    return pr


def _run_serial(
    remaining: list[tuple[int, T]],
    process_item: Callable[..., Any],
    static_kwargs: dict[str, Any],
    item_checkpoint_dir: Path | None,
    on_item_done: Callable[[int, T, Any], None] | None,
    parallel_strategy: ParallelStrategy | None,
    pbar: Any,
    keep_shot_results: bool = False,
    shot_checkpoint_subdir: Callable[[int], Path | None] | None = None,
) -> dict[int, Any]:
    """Execute remaining items serially. Returns {index: result} for in-memory results."""
    shot_executor = resolve_shot_executor(
        parallel_strategy.shot_executor
        if parallel_strategy is not None
        else None
    )
    results_dict: dict[int, Any] = {}

    try:
        for index, item in remaining:
            # Call process_item with keep_shot_results kwarg only if enabled
            extra_kwargs = static_kwargs.copy()
            if keep_shot_results:
                extra_kwargs["keep_shot_results"] = True

            raw_return = process_item(
                item,
                index,
                shot_executor=shot_executor,
                **extra_kwargs,
            )

            # Unpack result and optional ProgramResults based on keep_shot_results
            if keep_shot_results:
                result, in_memory_pr = raw_return
            else:
                result = raw_return

            results_dict[index] = result

            # Checkpoint result to worker file
            if item_checkpoint_dir is not None:
                worker_file_path = (
                    item_checkpoint_dir / f"worker_{worker_id()}_runner.h5"
                )
                _write_dict_entry_with_retry(
                    worker_file_path, "results", index, result
                )

                # If keep_shot_results is enabled, retrieve and write ProgramResults
                if keep_shot_results:
                    pr = _resolve_kept_program_results(
                        index, shot_checkpoint_subdir, in_memory_pr
                    )
                    if pr is not None:
                        _write_dict_entry_with_retry(
                            worker_file_path, "_program_results", index, pr
                        )

            if on_item_done is not None:
                on_item_done(index, item, result)

            if pbar is not None:
                pbar.update(1)
    finally:
        pass

    return results_dict


def _mark_observed_and_notify(
    index: int,
    result: Any,
    observed_indices: set[int],
    items_map: dict[int, Any],
    on_item_done: Callable[[int, Any, Any], None] | None,
    pbar: Any,
) -> None:
    """Record `index` as observed and fire `on_item_done`/`pbar` exactly
    once for it -- shared between live polling and the post-dispatch
    fallback pass, so an index is never double-notified regardless of
    which path first sees it."""
    if index in observed_indices:
        return
    observed_indices.add(index)
    if index in items_map:
        item = items_map[index]
        if on_item_done is not None:
            on_item_done(index, item, result)
        if pbar is not None:
            pbar.update(1)


def _poll_one_worker_file(
    worker_file: Path,
    consumed_count: int,
    observed_indices: set[int],
    items_map: dict[int, Any],
    on_item_done: Callable[[int, Any, Any], None] | None,
    pbar: Any,
) -> int:
    """Read this worker file's entries past `consumed_count`, notifying for
    each one, and return the file's updated consumed count.

    A transient HDF5 lock conflict (e.g. the worker itself mid-write) is
    silently tolerated -- the file is simply retried on the next poll tick,
    at whichever consumed count it last reached here.
    """
    try:
        with h5py.File(worker_file, "r") as f:
            for key, value in iter_dict_attr_entries(
                f, "results", start_index=consumed_count
            ):
                consumed_count += 1
                _mark_observed_and_notify(
                    key, value, observed_indices, items_map, on_item_done, pbar
                )
    except (BlockingIOError, OSError):
        pass
    return consumed_count


def _read_worker_current_indices(checkpoint_dir: Path) -> set[int]:
    """Read current_item_index attributes from all worker_*_runner.h5 files.

    Returns the set of item indices currently being processed by any worker.
    Silently tolerates missing files or transient HDF5 lock conflicts, which
    is appropriate since the set of workers can change mid-dispatch.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing worker_*_runner.h5 files.

    Returns
    -------
    set[int]
        Set of item indices currently being processed.
    """
    in_flight: set[int] = set()
    for worker_file in sorted(checkpoint_dir.glob("worker_*_runner.h5")):
        try:
            with h5py.File(worker_file, "r") as f:
                if "current_item_index" in f.attrs:
                    in_flight.add(int(f.attrs["current_item_index"]))
        except (BlockingIOError, OSError):
            # Transient lock conflict -- skip this file for now
            continue
    return in_flight


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
    keep_shot_results: bool = False,
    shot_checkpoint_subdir: Callable[[int], Path | None] | None = None,
    shots_pbar: Any = None,
    num_shots_for_progress: int | None = None,
) -> dict[int, Any]:
    """Execute remaining items in parallel with checkpointing and polling.

    Returns {index: result} for in-memory results (when no checkpointing).

    Parameters
    ----------
    shots_pbar : Any, optional
        A tqdm progress bar for shot-level progress (only when parallel dispatch
        with checkpointing is enabled). If provided, it is updated during each
        poll tick with the total shots completed so far.
    num_shots_for_progress : int | None, optional
        Number of shots per item (for computing absolute shot totals). Only used
        if shots_pbar is provided.
    """
    # Build a mapping of index -> item for use in on_poll callback
    items_map = {index: item for index, item in items_with_index}

    # Track observed indices to avoid double-counting, seeding with already-done indices
    observed_indices: set[int] = set(already_done_indices or [])

    # Track consumed count per worker file for efficient polling
    consumed_counts: dict[str, int] = {}

    def on_poll() -> None:
        """Poll every worker file and notify for any newly-completed items.

        Also updates the shots progress bar if one is provided.
        """
        if item_checkpoint_dir is None:
            return

        for worker_file in sorted(
            item_checkpoint_dir.glob("worker_*_runner.h5")
        ):
            key = str(worker_file)
            consumed_counts[key] = _poll_one_worker_file(
                worker_file,
                consumed_counts.get(key, 0),
                observed_indices,
                items_map,
                on_item_done,
                pbar,
            )

        # Update shots progress bar if enabled
        if shots_pbar is not None and num_shots_for_progress is not None:
            from loqs.core.programresults import ProgramResults

            # Count items already fully done
            done_items = len(observed_indices)
            total_shots_from_done = done_items * num_shots_for_progress

            # Count shots from in-flight items via their checkpoint directories
            in_flight_items = _read_worker_current_indices(item_checkpoint_dir)
            total_shots_from_inflight = 0
            for item_index in in_flight_items:
                shot_subdir = shot_checkpoint_subdir(item_index)
                if shot_subdir is not None:
                    shots_done = ProgramResults._count_done_shots(shot_subdir)
                    total_shots_from_inflight += shots_done

            # Set absolute total and refresh
            total_shots = total_shots_from_done + total_shots_from_inflight
            shots_pbar.n = total_shots
            shots_pbar.refresh()

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
        keep_shot_results=keep_shot_results,
        shot_checkpoint_subdir=shot_checkpoint_subdir,
    )

    # Dispatch with on_poll callback
    on_poll_callback = None
    if (
        item_checkpoint_dir is not None
        or on_item_done is not None
        or pbar is not None
        or shots_pbar is not None
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

    # For any items not already observed via worker file polling (i.e., when item_checkpoint_dir
    # is None), invoke on_item_done now so callers can collect results via the callback
    for index, result in newly_computed.items():
        _mark_observed_and_notify(
            index, result, observed_indices, items_map, on_item_done, pbar
        )

    return newly_computed


def _generic_chunk_worker(
    chunk: list[tuple[int, T]],
    process_item: Callable[..., Any],
    static_kwargs: dict[str, Any],
    item_checkpoint_dir: Path | None,
    shot_executor: Any,
    keep_shot_results: bool = False,
    shot_checkpoint_subdir: Callable[[int], Path | None] | None = None,
) -> list[tuple[int, Any]]:
    """Worker function for parallel execution of a chunk.

    Returns list of (index, result) tuples.
    """
    pin_worker_threads()
    shot_executor = resolve_shot_executor(shot_executor)

    results = []
    try:
        for index, item in chunk:
            # Write current_item_index to worker file if checkpointing is enabled
            if item_checkpoint_dir is not None:
                worker_file_path = (
                    item_checkpoint_dir / f"worker_{worker_id()}_runner.h5"
                )
                _write_current_item_index_with_retry(worker_file_path, index)

            # Call process_item with keep_shot_results kwarg only if enabled
            extra_kwargs = static_kwargs.copy()
            if keep_shot_results:
                extra_kwargs["keep_shot_results"] = True

            raw_return = process_item(
                item,
                index,
                shot_executor=shot_executor,
                **extra_kwargs,
            )

            # Unpack result and optional ProgramResults based on keep_shot_results
            if keep_shot_results:
                result, in_memory_pr = raw_return
            else:
                result = raw_return

            # Checkpoint result to worker file
            if item_checkpoint_dir is not None:
                worker_file_path = (
                    item_checkpoint_dir / f"worker_{worker_id()}_runner.h5"
                )
                _write_dict_entry_with_retry(
                    worker_file_path, "results", index, result
                )

                # If keep_shot_results is enabled, retrieve and write ProgramResults
                if keep_shot_results:
                    pr = _resolve_kept_program_results(
                        index, shot_checkpoint_subdir, in_memory_pr
                    )
                    if pr is not None:
                        _write_dict_entry_with_retry(
                            worker_file_path, "_program_results", index, pr
                        )

            results.append((index, result))
    finally:
        pass

    return results
