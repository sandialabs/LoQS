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
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from tqdm import tqdm

from loqs.core.historydatacollector import HistoryDataCollector
from loqs.core.programresults import (
    _normalize_decoded_int_keyed_dict,
    _resolve_checkpoint_object_group,
    _reset_empty_groups_format_dict_attr,
)
from loqs.core.quantumprogram import QuantumProgram
from loqs.internal import (
    _retry_hdf5_read,
    _retry_hdf5_write,
    pin_worker_threads,
    worker_id,
)
from loqs.internal.serializable import Serializable, ResolvingDecodeCache
from loqs.internal.streamingmerge import (
    filter_unmerged_dict_attr_entries,
    get_dict_attr_keys,
    merge_dict_attr,
    iter_dict_attr_entries,
    read_checkpoint_dict_attr_union,
    read_checkpoint_dict_attr_union_keys,
)
from loqs.tools.paralleltools import (
    ParallelStrategy,
    resolve_shot_executor,
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


def _verify_final_completeness(
    items_with_index: list[tuple[int, Any]],
    final_done: dict[int, Any],
    program_results: dict[int, Any] | None,
) -> None:
    """Raise a `RuntimeError` naming the first missing index if any item is
    absent from `final_done`, or (when `program_results` is not None) from
    `program_results`. The caller passes `None` for `program_results` when
    shot results aren't kept, or when `force_resume` is bypassing a config
    mismatch under which a resumed item's `_program_results` may never have
    existed.
    """
    for index, _ in items_with_index:
        if index not in final_done:
            raise RuntimeError(f"Item {index} is missing from final results")
    if program_results is not None:
        for index, _ in items_with_index:
            if index not in program_results:
                raise RuntimeError(
                    f"Item {index} is missing from final _program_results"
                )


class MultiProgramRunner(Serializable, Generic[T]):
    """Base class for crash-recoverable program runners bundling config fields
    and a template-method `run()` for checkpoint/resume/progress.

    Subclasses implement hook methods to define their specific work logic.
    Checkpoint/resume behavior is controlled by explicit `checkpoint` and
    `resume` flags, applied against `item_checkpoint_dir`'s own on-disk state
    according to a 4-case state machine (see `run()`). Once `run()` completes,
    `item_wall_clock_times`/`shot_wall_clock_times` are populated as instance
    attributes recording each item's/shot's own wall-clock duration (see their
    own attribute docstrings for the exact mapping).
    """

    items: Sequence[T]

    _SERIALIZE_ATTRS = [
        "parallel_strategy",
        "item_checkpoint_dir",
        "checkpoint",
        "resume",
        "force_resume",
        "shot_checkpoint",
        "shot_checkpoint_dir",
        "lazy_loading",
        "index_map",
        "_reduced_results",
        "keep_shot_results",
        "_program_results",
        "item_wall_clock_times",
        "shot_wall_clock_times",
        "poll_interval",
        "show_progress",
        "runner_filename",
        "results_filename",
        "run_kwargs",
    ]

    # (worker-key, runner-key, value_use_dataset) triples for attributes that
    # must populate on every dispatch, whether or not checkpointing is used.
    _ALWAYS_MERGED_ATTRS: ClassVar[tuple[tuple[str, str, bool], ...]] = (
        ("results", "_reduced_results", False),
        ("item_wall_clock_times", "item_wall_clock_times", True),
        ("shot_wall_clock_times", "shot_wall_clock_times", False),
    )

    # (worker-key, runner-key, value_use_dataset) triples: "results"/
    # "_program_results" keep their worker-vs-runner name split; the two
    # timing attrs share one name. item_wall_clock_times is a plain float
    # (dataset format); shot_wall_clock_times is a nested dict, so it needs
    # the general Serializable dict encoder instead (groups format).
    # "_program_results" is the sole disk-only exception -- it never
    # populates in memory without checkpointing, unlike the always-merged set.
    _STREAMED_DICT_ATTRS: ClassVar[tuple[tuple[str, str, bool], ...]] = (
        _ALWAYS_MERGED_ATTRS
        + (("_program_results", "_program_results", False),)
    )

    _NO_COLLAPSE_ATTRS: ClassVar[frozenset[str]] = frozenset(
        runner_key for _, runner_key, _ in _STREAMED_DICT_ATTRS
    )

    CHKPT_SUBDIR_PREFIX: ClassVar[str] = "item"

    def __init__(
        self,
        parallel_strategy: ParallelStrategy | None = None,
        item_checkpoint_dir: str | Path | None = None,
        checkpoint: bool = False,
        resume: bool = False,
        force_resume: bool = False,
        shot_checkpoint: bool = False,
        shot_checkpoint_dir: str | Path | None = None,
        lazy_loading: bool = True,
        keep_shot_results: bool = False,
        poll_interval: float = 1.0,
        show_progress: bool = True,
        runner_filename: str = "runner.h5",
        results_filename: str = "results.h5",
        run_kwargs: dict[str, Any] | None = None,
    ):
        self.parallel_strategy = parallel_strategy
        self.item_checkpoint_dir = (
            Path(item_checkpoint_dir)
            if item_checkpoint_dir is not None
            else None
        )
        self.checkpoint = checkpoint
        self.resume = resume
        self.force_resume = force_resume
        self.shot_checkpoint = shot_checkpoint
        self.shot_checkpoint_dir = (
            Path(shot_checkpoint_dir)
            if shot_checkpoint_dir is not None
            else None
        )
        self.lazy_loading = lazy_loading
        self.index_map: dict[str, int] | None = None
        self._reduced_results: dict[int, Any] = {}
        self.keep_shot_results = keep_shot_results
        self._program_results: dict[int, Any] = {}
        self.item_wall_clock_times: dict[int, float] = {}
        """Total wall-clock duration for processing each item,
        mapped by item index."""

        self.shot_wall_clock_times: dict[int, dict[int, float]] = {}
        """Wall-clock duration for each shot, nested by item index then shot index."""
        self.poll_interval = poll_interval
        self.show_progress = show_progress
        self.runner_filename = runner_filename
        self.results_filename = results_filename
        self.item_key_fn: Callable[[Any], str] | None = None
        self.run_kwargs = dict(run_kwargs) if run_kwargs is not None else {}
        if "max_frame_limit" not in self.run_kwargs:
            warnings.warn(
                "run_kwargs did not specify 'max_frame_limit'; defaulting to 1_000_000."
            )
            self.run_kwargs["max_frame_limit"] = 1_000_000
        if (
            "checkpoint_dir" in self.run_kwargs
            and self.shot_checkpoint_dir is not None
        ):
            raise ValueError(
                "checkpoint_dir in run_kwargs conflicts with shot_checkpoint_dir; use only one of the two (or leave both unset)."
            )
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
        cls, attr_dict: Mapping[str, Any]
    ) -> "MultiProgramRunner":
        """Reconstruct from decoded attributes, converting strings back to Paths."""
        attr_dict = dict(attr_dict)
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
        item_wall_clock_times = attr_dict.pop("item_wall_clock_times", None)
        shot_wall_clock_times = attr_dict.pop("shot_wall_clock_times", None)
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
        obj.item_wall_clock_times = (
            _normalize_decoded_int_keyed_dict(
                item_wall_clock_times, value_cast=float
            )
            if item_wall_clock_times is not None
            else {}
        )
        obj.shot_wall_clock_times = (
            _normalize_decoded_int_keyed_dict(
                shot_wall_clock_times,
                value_cast=lambda inner: _normalize_decoded_int_keyed_dict(
                    inner, value_cast=float
                ),
            )
            if shot_wall_clock_times is not None
            else {}
        )
        # Auto-set resume=True when deserializing a checkpoint-enabled runner:
        # if checkpoint=True and item_checkpoint_dir exists, we're implicitly resuming
        if obj.checkpoint and obj.item_checkpoint_dir is not None:
            obj.resume = True
        return obj

    def _validate_checkpoint_kwargs(self) -> None:
        """Validate checkpoint-related configuration constraints."""
        # Bijection: checkpoint and item_checkpoint_dir must agree
        if self.checkpoint and self.item_checkpoint_dir is None:
            raise ValueError(
                "checkpoint=True requires item_checkpoint_dir to be set"
            )
        if self.item_checkpoint_dir is not None and not self.checkpoint:
            raise ValueError(
                "item_checkpoint_dir is not None requires checkpoint=True"
            )

        # resume=True requires checkpoint=True
        if self.resume and not self.checkpoint:
            raise ValueError("resume=True requires checkpoint=True")

        # Bijection: shot_checkpoint and shot_checkpoint_dir must agree
        if self.shot_checkpoint and self.shot_checkpoint_dir is None:
            raise ValueError(
                "shot_checkpoint=True requires shot_checkpoint_dir to be set"
            )
        if self.shot_checkpoint_dir is not None and not self.shot_checkpoint:
            raise ValueError(
                "shot_checkpoint_dir is not None requires shot_checkpoint=True"
            )

        if self.keep_shot_results and self.item_checkpoint_dir is None:
            raise ValueError(
                "keep_shot_results requires item_checkpoint_dir to be set"
            )
        if self.keep_shot_results and not self.shot_checkpoint:
            raise ValueError(
                "keep_shot_results requires shot_checkpoint=True (and "
                "shot_checkpoint_dir) to be set, so kept results are read "
                "back from each item's own on-disk shot checkpoint rather "
                "than held fully in memory for every item at once"
            )

    def run(self) -> Any:
        """Run the program with checkpoint/resume support.

        Implements a 4-case state machine based on `checkpoint` and `resume`
        flags and on-disk state (`runner.h5` presence in `item_checkpoint_dir`):

        (a) `checkpoint=True`, `resume=False`, no on-disk state: free to create
            and start; write `runner.h5`.
        (b) `checkpoint=True`, `resume=False`, on-disk state exists: raise
            `ValueError` (never silently overwrite/resume).
        (c) `checkpoint=True`, `resume=True`, on-disk state exists: resume,
            subject to the existing mismatch check (`force_resume` bypasses).
        (d) `checkpoint=True`, `resume=True`, no on-disk state: raise
            `ValueError` (nothing to resume from).

        For both cases (a) and (c), `_reduced_results`, `_program_results`,
        `item_wall_clock_times`, `shot_wall_clock_times`, and `index_map` are
        seeded from stored state before `runner.h5` is written, then
        `runner.h5` is updated with the full prior state. Progress and
        done-detection is driven by `_read_done_union` on subsequent work.
        When checkpointing, `_run_dispatch` populates `item_wall_clock_times`/
        `shot_wall_clock_times` directly from the consolidated on-disk state
        once dispatch completes; without checkpointing, it merges them
        in-memory from each item's own dispatch result instead.
        """
        stored = None
        if self.checkpoint:
            assert (
                self.item_checkpoint_dir is not None
            )  # Validated in _validate_checkpoint_kwargs
            runner_path = self.item_checkpoint_dir / self.runner_filename
            has_content = self.item_checkpoint_dir.exists() and any(
                self.item_checkpoint_dir.iterdir()
            )

            if has_content and not runner_path.exists():
                # Unrelated foreign content, ambiguous state
                raise FileExistsError(
                    f"{self.item_checkpoint_dir} exists with content "
                    f"that isn't a recognized checkpoint (no {self.runner_filename})."
                )

            # Case (b): checkpoint=True, resume=False, but on-disk state exists
            if has_content and not self.resume:
                raise ValueError(
                    f"{self.item_checkpoint_dir} contains an existing checkpoint "
                    f"({self.runner_filename} present). Pass resume=True to continue that run, "
                    "or use a different item_checkpoint_dir for a fresh run."
                )

            # Case (d): checkpoint=True, resume=True, but no on-disk state
            if self.resume and not has_content:
                raise ValueError(
                    f"{self.item_checkpoint_dir} is empty or nonexistent; "
                    "nothing to resume from. Pass resume=False for a fresh run, "
                    "or use a directory containing a prior checkpoint."
                )

            # Cases (a) and (c): valid cases
            if has_content:
                # Case (c): on-disk state exists, resuming
                stored = type(self).read(runner_path)
                if type(stored) is not type(self):
                    raise TypeError(
                        f"Cannot resume: checkpoint at {runner_path} was "
                        f"created by {type(stored).__name__}, not "
                        f"{type(self).__name__}."
                    )
                mismatches = [
                    self._mismatch_field_display_name(f)
                    for f in self._mismatch_check_fields()
                    if getattr(self, f) != getattr(stored, f)
                ]
                if mismatches and not self.force_resume:
                    raise ValueError(
                        f"Cannot resume: stored config differs in "
                        f"{', '.join(mismatches)}. Pass force_resume=True to resume anyway."
                    )

            # Seed internal state from stored before writing runner.h5, so the
            # first write reflects the full prior state, not a later one.
            if stored is not None:
                assert isinstance(stored, MultiProgramRunner)
                self._reduced_results = dict(stored._reduced_results)
                self._program_results = dict(stored._program_results)
                self.item_wall_clock_times = dict(stored.item_wall_clock_times)
                self.shot_wall_clock_times = dict(stored.shot_wall_clock_times)
                if self.index_map is None:
                    self.index_map = (
                        dict(stored.index_map)
                        if stored.index_map is not None
                        else {}
                    )

            # Both cases (a) and (c): write/update runner.h5 with seeded state
            self.item_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.write(runner_path)
            # Reset any empty dict attribute Serializable.write just wrote in
            # the wrong ("groups") format, so the first real merge_dict_attr
            # call for it establishes the correct format instead.
            with h5py.File(runner_path, "a") as f:
                for attr_name in self._NO_COLLAPSE_ATTRS:
                    _reset_empty_groups_format_dict_attr(f, attr_name)

        # Pre-assign indices via item_key_fn, adopting the persisted map
        # (or the now-seeded map) so a fresh resumed instance doesn't reassign
        # indices out from under already-checkpointed work.
        precomputed_indices = None
        key_fn = self.item_key_fn
        if key_fn is not None:
            if self.index_map is None:
                self.index_map = {}
            items_with_index = _assign_indices_with_keys(
                self.items, key_fn, self.index_map
            )
            precomputed_indices = [idx for idx, _ in items_with_index]
            # Update runner.h5 with the now-populated index_map
            if self.item_checkpoint_dir is not None:
                index_map_path = (
                    self.item_checkpoint_dir / self.runner_filename
                )
                self.write(index_map_path)
                # This write may again re-serialize an as-yet-empty dict
                # attribute in the wrong format; reset it the same way.
                with h5py.File(index_map_path, "a") as f:
                    for attr_name in self._NO_COLLAPSE_ATTRS:
                        _reset_empty_groups_format_dict_attr(f, attr_name)

        self._run_dispatch(
            precomputed_indices=precomputed_indices,
        )
        return self.build_output(self._ordered_reduced_results())

    def _run_dispatch(
        self,
        precomputed_indices: Sequence[int] | None,
    ) -> None:
        """Dispatch item processing with checkpoint/resume/progress tracking:
        item indexing, prior-progress replay, serial/parallel dispatch, and
        final completion verification in original item order."""
        # Item indexing/identity
        items_with_index = _resolve_items_with_index(
            self.items, precomputed_indices
        )

        # Read prior progress (union of runner.h5 and worker files)
        done: dict[int, Any] = {}
        if self.item_checkpoint_dir is not None:
            done = _read_done_union(
                self.item_checkpoint_dir,
                runner_filename=self.runner_filename,
                attr_name="results",
            )

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
            pbar = tqdm(
                total=len(self.items), initial=len(done), desc=self._desc()
            )

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
            and self.shot_checkpoint
            and self.checkpoint
            and num_shots_for_progress is not None
        )

        if show_shots_bar:
            assert num_shots_for_progress is not None
            shots_pbar = tqdm(
                total=len(self.items) * num_shots_for_progress,
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
        ):
            if not self.shot_checkpoint:
                print(
                    "Shot-level progress reporting requires shot_checkpoint=True"
                    " (and shot_checkpoint_dir) to be set; showing item-level"
                    " progress only."
                )
            elif not self.checkpoint:
                print(
                    "Shot-level progress reporting requires checkpoint=True"
                    " (in addition to shot_checkpoint=True) to be set;"
                    " showing item-level progress only."
                )

        try:
            # Dispatch
            newly_computed: dict[int, dict[str, Any]] = {}
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
                    _shared_item_worker,
                    self._static_kwargs() or {},
                    self.item_checkpoint_dir,
                    on_item_done,
                    self.parallel_strategy,
                    pbar,
                    self.keep_shot_results,
                    self._shot_checkpoint_subdir,
                    self.results_filename,
                )
            else:
                # Parallel execution: create a snapshot for pickling
                runner_snapshot = copy.copy(self)
                runner_snapshot.parallel_strategy = None
                newly_computed = _run_parallel(
                    remaining,
                    _shared_item_worker,
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
                    self.results_filename,
                )

            # Final assembly - consolidate worker files and read union of
            # runner.h5 and any remaining worker files
            if self.item_checkpoint_dir is not None:
                # Always consolidate all worker-file attributes (per
                # _STREAMED_DICT_ATTRS), then delete them once merged
                _consolidate_worker_files(
                    self.item_checkpoint_dir,
                    runner_filename=self.runner_filename,
                    delete_originals=True,
                )
                # Read every always-merged attribute from disk directly into
                # its own runner-side attribute; a lock conflict here
                # propagates, since this one-shot pass has no next poll tick.
                for worker_key, runner_key, _ in self._ALWAYS_MERGED_ATTRS:
                    setattr(
                        self,
                        runner_key,
                        _read_done_union(
                            self.item_checkpoint_dir,
                            runner_filename=self.runner_filename,
                            attr_name=worker_key,
                            retry_on_conflict=True,
                        ),
                    )
                final_done = self._reduced_results
                # Handle _program_results if keep_shot_results is enabled
                if self.keep_shot_results:
                    if self.lazy_loading:
                        runner_path = (
                            self.item_checkpoint_dir / self.runner_filename
                        )
                        program_results_keys = _read_done_union_keys(
                            self.item_checkpoint_dir,
                            runner_filename=self.runner_filename,
                            attr_name="_program_results",
                            retry_on_conflict=True,
                        )
                        self._program_results = {
                            index: self._make_lazy_program_results(
                                runner_path, index
                            )
                            for index in program_results_keys
                        }
                    else:
                        self._program_results = _read_done_union(
                            self.item_checkpoint_dir,
                            runner_filename=self.runner_filename,
                            attr_name="_program_results",
                            retry_on_conflict=True,
                        )
            else:
                # No checkpointing: merge each _ALWAYS_MERGED_ATTRS value out
                # of the aux dicts below, since nothing was written to disk.
                final_done = done.copy()
                final_done.update(newly_computed)
                for _, runner_key, _ in self._ALWAYS_MERGED_ATTRS:
                    getattr(self, runner_key).update(
                        {
                            index: aux[runner_key]
                            for index, aux in final_done.items()
                        }
                    )

            _verify_final_completeness(
                items_with_index,
                final_done,
                (
                    self._program_results
                    if (self.keep_shot_results and not self.force_resume)
                    else None
                ),
            )

        finally:
            if pbar is not None:
                pbar.close()
            if shots_pbar is not None:
                shots_pbar.close()

    def _make_lazy_program_results(self, runner_file: Path, index: int) -> Any:
        """Construct a lazy-loading ProgramResults for a nested entry.

        Creates a ProgramResults configured to load shots from the runner.h5's
        own `_program_results` dict attribute at the given index, with
        lazy_loading=True. `num_shots`/`max_frame_limit` are forwarded from
        this runner (genuine per-runner scalars), and `shot_wall_clock_times`
        is forwarded from this runner's own already-populated per-item timing
        data; `parent_program`/`name` are genuinely per-item, so they aren't
        set here directly -- instead, `_set_nested_shot_source` marks them
        unresolved, and `ProgramResults`' own `name`/`parent_program`
        properties lazily resolve each one from this same nested source the
        first time a caller actually reads it.

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

        pr = ProgramResults(
            num_shots=getattr(self, "num_shots", None),
            max_frame_limit=self.run_kwargs.get("max_frame_limit"),
            shot_wall_clock_times=dict(
                self.shot_wall_clock_times.get(index, {})
            ),
            lazy_loading=True,
        )
        pr._set_nested_shot_source(runner_file, index)
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

        runner_path = self.item_checkpoint_dir / self.runner_filename

        def _write(f: h5py.File) -> None:
            obj_grp = _get_runner_object_group(f)
            merge_dict_attr(
                obj_grp,
                "_reduced_results",
                [(index, value)],
                key_use_dataset=True,
                value_use_dataset=False,
            )

        _retry_hdf5_write(runner_path, _write)

    # Hook methods -- subclasses implement these
    def build_program(self, index: int) -> QuantumProgram:
        """Build a QuantumProgram for the item at the given index."""
        raise NotImplementedError

    def reduce_program_outcomes(self, program_results: Any) -> Any:
        """Reduce program_results to a single outcome value."""
        raise NotImplementedError

    def build_output(self, ordered_results: list[tuple[Any, Any]]) -> Any:
        """Build final output from ordered (item, result) pairs.

        Warns if any result is None (incomplete run), then delegates
        to _build_output for subclass-specific final assembly.
        """
        if any(result is None for _, result in ordered_results):
            warnings.warn(
                "One or more items produced a None result (incomplete run)."
            )
        return self._build_output(ordered_results)

    def _build_output(self, ordered_results: list[tuple[Any, Any]]) -> Any:
        """Subclass-specific final output assembly from ordered results."""
        raise NotImplementedError

    # Internal helpers -- not subclass hooks
    def _ordered_reduced_results(self) -> list[tuple[Any, Any]]:
        """Build a list of (item, result) pairs in original item order."""
        ordered = []
        for pos, item in enumerate(self.items):
            if self.item_key_fn is not None and self.index_map is not None:
                key = self.item_key_fn(item)
                index = self.index_map.get(key, pos)
            else:
                index = pos
            ordered.append((item, self._reduced_results.get(index)))
        return ordered

    def _static_kwargs(self) -> dict[str, Any]:
        """Return dict of static kwargs to pass to _shared_item_worker."""
        runner_snapshot = copy.copy(self)
        runner_snapshot.parallel_strategy = None
        return {
            "build_program": runner_snapshot.build_program,
            "reduce_program_outcomes": runner_snapshot.reduce_program_outcomes,
            "run_kwargs": self.run_kwargs,
            "shot_checkpoint_dir": self.shot_checkpoint_dir,
            "shot_checkpoint": self.shot_checkpoint,
            "chkpt_subdir_prefix": self.CHKPT_SUBDIR_PREFIX,
            "results_filename": self.results_filename,
            "lazy_loading": self.lazy_loading,
            "force_resume": self.force_resume,
            "num_shots": getattr(self, "num_shots", None),
        }

    def _make_on_item_done(self) -> Callable[[int, Any, Any], None] | None:
        """Return a closure for on_item_done callback, or None.

        Default: merges (index, result) into self._reduced_results via
        _merge_reduced_result, ignoring the item itself -- covers every
        current subclass, which all just forward result unchanged.
        """

        def on_item_done(index: int, item: Any, result: Any) -> None:
            self._merge_reduced_result(index, result)

        return on_item_done

    def _desc(self) -> str:
        """Return description string for progress bar."""
        return "Processing items"

    def _mismatch_check_fields(self) -> list[str]:
        """Return list of field names to compare for resume mismatch check."""
        return []

    @property
    def _normalized_collect_shot_data_args(
        self,
    ) -> tuple[HistoryDataCollector, ...]:
        """Canonical form of collect_shot_data_args (a plain sequence of one
        or more collector specs), used only for resume mismatch comparison so
        a differently-spelled-but-equivalent spec doesn't spuriously fail
        resume."""
        raw_args = getattr(self, "collect_shot_data_args", ())
        return tuple(HistoryDataCollector.from_raw(c) for c in raw_args)

    def _mismatch_field_display_name(self, field: str) -> str:
        """Map an internal comparison-only field name (as returned by
        _mismatch_check_fields) to its corresponding public constructor
        parameter name, for a clear resume-mismatch error message."""
        if field == "_normalized_collect_shot_data_args":
            return "collect_shot_data_args"
        return field

    def _shot_checkpoint_subdir(self, index: int) -> Path | None:
        """Return the checkpoint directory for a specific item's shots, or None.

        Guards on `shot_checkpoint` being True and the CHKPT_SUBDIR_PREFIX
        class attribute being non-None; otherwise returns None (no per-item
        shot subdirectories). The bijection validates that shot_checkpoint_dir is
        set whenever shot_checkpoint is True.
        """
        prefix = self.CHKPT_SUBDIR_PREFIX
        if prefix is None or not self.shot_checkpoint:
            return None
        assert self.shot_checkpoint_dir is not None
        return _checkpoint_subdir_for_prefix(
            self.shot_checkpoint_dir, prefix, index
        )

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


def _checkpoint_subdir_for_prefix(
    shot_checkpoint_dir: str | Path, prefix: str, index: int
) -> Path:
    """Build a per-item checkpoint subdirectory path for a given prefix and index."""
    return Path(shot_checkpoint_dir) / f"{prefix}_{index}"


def _is_sweep_callable(value: Any) -> bool:
    """Whether `value` should be treated as a per-item callable rather than a fixed value.

    Plain `callable(value)` is not sufficient on its own: classes are themselves callable in
    Python (calling a class constructs an instance), so a fixed value that happens to be a
    class would otherwise be misclassified as "a callable to invoke with the item."
    """
    return callable(value) and not isinstance(value, type)


def _shared_item_worker(
    item: Any,
    index: int,
    *,
    build_program: Callable[[int], QuantumProgram],
    reduce_program_outcomes: Callable[[Any], Any],
    run_kwargs: dict[str, Any],
    shot_checkpoint_dir: Path | None = None,
    shot_checkpoint: bool = False,
    chkpt_subdir_prefix: str = "item",
    results_filename: str = "results.h5",
    lazy_loading: bool = True,
    force_resume: bool = False,
    num_shots: int | None = None,
    shot_executor: Any = None,
    n_shot_batches: int | None = None,
    keep_shot_results: bool = False,
    **kwargs: Any,
) -> Any:
    """Shared worker function for processing items with program building and reduction.

    Builds a program for the given index, runs it with resolved kwargs, reduces
    the program results, and optionally keeps shot results. Handles checkpoint
    setup when needed. Returns a dict keyed by `"_reduced_results"`,
    `"item_wall_clock_times"` (this entire body's own wall-clock duration),
    and `"shot_wall_clock_times"` (per-shot wall-clock times), plus
    `"program_results"` when `keep_shot_results` is True.
    """
    item_start_time = time.perf_counter()
    program = build_program(index)

    resolved_run_kwargs = {
        key: (value(item) if _is_sweep_callable(value) else value)
        for key, value in run_kwargs.items()
    }
    resolved_run_kwargs.setdefault("verbose", False)
    resolved_run_kwargs["lazy_loading"] = lazy_loading
    resolved_run_kwargs["force_resume"] = force_resume
    resolved_run_kwargs["results_filename"] = results_filename

    if shot_executor is not None:
        resolved_run_kwargs["shot_executor"] = shot_executor
    if n_shot_batches is not None:
        resolved_run_kwargs["n_shot_batches"] = n_shot_batches
    if num_shots is not None and "num_shots" not in resolved_run_kwargs:
        resolved_run_kwargs["num_shots"] = num_shots

    if shot_checkpoint and shot_checkpoint_dir is not None:
        checkpoint_dir = _checkpoint_subdir_for_prefix(
            shot_checkpoint_dir, chkpt_subdir_prefix, index
        )
        resolved_run_kwargs["checkpoint"] = True
        resolved_run_kwargs["checkpoint_dir"] = checkpoint_dir
        resolved_run_kwargs["resume"] = (
            checkpoint_dir / results_filename
        ).exists()
    else:
        resolved_run_kwargs.setdefault("checkpoint", False)
        resolved_run_kwargs.setdefault("resume", False)

    program_results = program.run(**resolved_run_kwargs)
    reduced_result = reduce_program_outcomes(program_results)
    item_wall_clock_time = time.perf_counter() - item_start_time
    # Copy now, since program_results may be discarded below. Falls back to
    # {} for a test double or other stand-in that doesn't carry this attr.
    shot_wall_clock_times = dict(
        getattr(program_results, "shot_wall_clock_times", None) or {}
    )

    aux = {
        "_reduced_results": reduced_result,
        "item_wall_clock_times": item_wall_clock_time,
        "shot_wall_clock_times": shot_wall_clock_times,
    }

    if keep_shot_results:
        aux["program_results"] = program_results
        return aux

    del program, program_results
    return aux


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
    checkpoint_dir: Path,
    attr_name: str = "results",
    retry_on_conflict: bool = False,
) -> dict[int, Any]:
    """Read all worker_*_runner.h5 files and return {index: result} dict.

    Reads the specified attribute from each worker file and merges them into
    a single dict. By default, a transient HDF5 lock conflict on a worker
    file is silently skipped (safe for live polling, since there's a next
    poll tick to retry on). With `retry_on_conflict=True`, the open is
    routed through `_retry_hdf5_read` first, and a lock conflict that
    survives that retry budget propagates instead of being swallowed -- for
    a one-shot final-assembly pass, there is no next tick to catch it.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing worker_*_runner.h5 files.
    attr_name : str, optional
        Name of the dict attribute to read from each worker file.
        Default is "results".
    retry_on_conflict : bool, optional
        If True, retry a transient lock conflict via `_retry_hdf5_read` and
        let a conflict that survives the retry budget propagate rather than
        being silently skipped. Default False (live-polling behavior).

    Returns
    -------
    dict[int, Any]
        Merged {index: result} dict from all worker files.
    """
    return read_checkpoint_dict_attr_union(
        checkpoint_dir,
        None,
        "worker_*_runner.h5",
        attr_name,
        retry_on_conflict=retry_on_conflict,
    )


def _read_done_union(
    checkpoint_dir: Path,
    runner_filename: str = "runner.h5",
    attr_name: str = "results",
    retry_on_conflict: bool = False,
) -> dict[int, Any]:
    """Compute the union of runner.h5's consolidated dict attribute and all
    worker_*_runner.h5 files' matching attributes, via the shared
    read_checkpoint_dict_attr_union primitive.

    Reads from runner.h5's own state first (if it exists), then merges in any
    entries from worker files, with worker file entries taking precedence on
    key collision (same merge semantics as merge_dict_attr).

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing runner.h5 and/or worker_*_runner.h5 files.
    runner_filename : str, optional
        Name of the runner checkpoint file (default "runner.h5").
    attr_name : str, optional
        Name of the dict attribute to read. Maps "results" to "_reduced_results"
        in runner.h5's actual attribute name; other names used as-is.
        Default is "results".
    retry_on_conflict : bool, optional
        Forwarded to `_read_worker_files`. If True, a transient lock conflict
        on the runner.h5 read is also retried via `_retry_hdf5_read`, with a
        conflict surviving that retry budget propagating instead of falling
        back to worker files. Default False (live-polling behavior).

    Returns
    -------
    dict[int, Any]
        Union of all entries from runner.h5 and all worker files.
        Returns empty dict if no checkpoints exist.
    """
    canonical_attr_name = (
        "_reduced_results" if attr_name == "results" else None
    )
    done: dict[int, Any] = read_checkpoint_dict_attr_union(
        checkpoint_dir,
        runner_filename,
        None,
        attr_name,
        canonical_attr_name=canonical_attr_name,
        retry_on_conflict=retry_on_conflict,
    )
    done.update(
        _read_worker_files(
            checkpoint_dir,
            attr_name=attr_name,
            retry_on_conflict=retry_on_conflict,
        )
    )

    return done


def _read_done_union_keys(
    checkpoint_dir: Path,
    runner_filename: str = "runner.h5",
    attr_name: str = "results",
    retry_on_conflict: bool = False,
) -> set[int]:
    """Key-only sibling of `_read_done_union`: returns the union of keys
    present in runner.h5's consolidated dict attribute and all
    worker_*_runner.h5 files' matching attributes, without decoding any
    values. Same "results" -> "_reduced_results" attribute-name mapping and
    `retry_on_conflict` semantics as `_read_done_union`; returns an empty
    set if no checkpoints exist.
    """
    canonical_attr_name = (
        "_reduced_results" if attr_name == "results" else None
    )
    return read_checkpoint_dict_attr_union_keys(
        checkpoint_dir,
        runner_filename,
        "worker_*_runner.h5",
        attr_name,
        canonical_attr_name=canonical_attr_name,
        retry_on_conflict=retry_on_conflict,
    )


def _get_runner_object_group(f: h5py.File) -> h5py.Group:
    """Navigate to the MultiProgramRunner object group inside runner.h5."""
    return _resolve_checkpoint_object_group(f)


def _get_existing_dict_keys(obj_grp: h5py.Group, attr_name: str) -> set[int]:
    """Extract existing keys from a dict attribute in an HDF5 object group."""
    return set(get_dict_attr_keys(obj_grp, attr_name))


def _consolidate_worker_files(
    checkpoint_dir: Path,
    runner_filename: str = "runner.h5",
    delete_originals: bool = True,
) -> None:
    """Consolidate all worker_*_runner.h5 files into runner.h5 and optionally
    delete worker files once merged.

    Streams one item at a time from worker files into runner.h5's object group,
    keeping memory bounded. Merges every attribute in
    `MultiProgramRunner._STREAMED_DICT_ATTRS` ("results" stored as
    "_reduced_results", plus "_program_results", "item_wall_clock_times", and
    "shot_wall_clock_times"). Skips indices already present in runner.h5 to
    avoid duplicate entries on resume. Deletes each worker file only once its
    entries are confirmed merged (crash-safe: a crashed merge self-heals on
    retry since the worker file is still present).

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing worker_*_runner.h5 files.
    runner_filename : str, optional
        Name of the runner.h5 file (default "runner.h5").
    delete_originals : bool, optional
        If True, delete worker files after they are successfully merged
        (default True). If False, leave worker files in place.
    """
    runner_path = checkpoint_dir / runner_filename
    if not runner_path.exists():
        return

    # Track existing keys in runner.h5 to avoid duplicates, one set per
    # runner-side attribute name.
    existing_keys: dict[str, set[int]] = {}
    runner_is_empty = False

    def _read_existing_keys(out_f: h5py.File) -> None:
        nonlocal runner_is_empty, existing_keys
        if len(out_f.keys()) == 0:
            runner_is_empty = True
            return
        out_root = _get_runner_object_group(out_f)
        existing_keys = {
            runner_key: _get_existing_dict_keys(out_root, runner_key)
            for _, runner_key, _ in MultiProgramRunner._STREAMED_DICT_ATTRS
        }

    _retry_hdf5_write(runner_path, _read_existing_keys)
    if runner_is_empty:
        return

    # Consolidate each worker file and delete it once merged. The worker file
    # read (retried) wraps the runner.h5 write (also retried), so an
    # unreadable worker file fails within its own retry budget alone.
    for worker_file in sorted(checkpoint_dir.glob("worker_*_runner.h5")):
        try:

            def _read_and_merge_worker(in_f: h5py.File) -> None:
                def _merge_worker_into_runner(out_f: h5py.File) -> None:
                    out_root = _get_runner_object_group(out_f)
                    # A shared decode_cache is required across every
                    # attribute below so that a Serializable value
                    # referenced by entries in more than one attribute
                    # decodes to the same real object everywhere, rather
                    # than an unresolved DeferredRef past its first
                    # occurrence.
                    decode_cache = ResolvingDecodeCache(
                        root=in_f, format="hdf5"
                    )
                    for (
                        worker_key,
                        runner_key,
                        value_use_dataset,
                    ) in MultiProgramRunner._STREAMED_DICT_ATTRS:
                        entries = filter_unmerged_dict_attr_entries(
                            in_f,
                            worker_key,
                            existing_keys[runner_key],
                            decode_cache=decode_cache,
                        )
                        if entries is not None:
                            merge_dict_attr(
                                out_root,
                                runner_key,
                                entries,
                                encode_cache={},
                                key_use_dataset=True,
                                value_use_dataset=value_use_dataset,
                            )

                _retry_hdf5_write(runner_path, _merge_worker_into_runner)

            _retry_hdf5_read(worker_file, _read_and_merge_worker)

            # Delete worker file once its contents are confirmed merged
            if delete_originals:
                try:
                    worker_file.unlink()
                except OSError:
                    # File already deleted or inaccessible; ignore
                    pass

        except (BlockingIOError, OSError, KeyError):
            # A retry-exhausted lock (indistinguishable from corruption, since
            # both raise plain OSError) or a missing attribute -- best-effort
            # skip; genuinely lost data is caught downstream by run()'s own
            # final-assembly completeness checks.
            continue


def _write_dict_entry_with_retry(
    worker_file_path: Path,
    attr_name: str,
    index: int,
    value: Any,
    value_use_dataset: bool = False,
) -> None:
    """Write a single entry to a dict attribute in a worker file with retry logic.

    Handles transient HDF5 locking issues via exponential backoff, using
    `_retry_hdf5_write`'s own default retry budget.

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
    value_use_dataset : bool, optional
        Forwarded to `merge_dict_attr` (only consulted if this attribute
        doesn't already exist in the worker file). Default False.
    """
    _retry_hdf5_write(
        worker_file_path,
        lambda f: merge_dict_attr(
            f,
            attr_name,
            [(index, value)],
            encode_cache={},
            key_use_dataset=True,
            value_use_dataset=value_use_dataset,
        ),
    )


def _write_current_item_index_with_retry(
    worker_file_path: Path,
    index: int,
) -> None:
    """Write current_item_index attribute to a worker file with retry logic.

    Handles transient HDF5 locking issues via exponential backoff, using
    `_retry_hdf5_write`'s own default retry budget. Overwrites any prior
    value.

    Parameters
    ----------
    worker_file_path : Path
        Path to the worker_*_runner.h5 file.
    index : int
        The current item index being processed.
    """
    _retry_hdf5_write(
        worker_file_path,
        lambda f: f.attrs.__setitem__("current_item_index", index),
    )


def _resolve_kept_program_results(
    index: int,
    shot_checkpoint_subdir: Callable[[int], Path | None] | None,
    in_memory_pr: Any,
    results_filename: str = "results.h5",
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
        Callable returning the per-item checkpoint directory, or None.
    in_memory_pr : Any
        The in-memory ProgramResults from process_item, or None.
    results_filename : str, optional
        Filename for checkpoint loading (default "results.h5").

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

            pr = ProgramResults(results_filename=results_filename)
            pr.load_checkpoint(checkpoint_dir=shot_dir)
    # Otherwise use the in-memory one from process_item; if both are
    # present, backfill any metadata the checkpoint load still left unset.
    if pr is None or not pr.shot_histories:
        pr = in_memory_pr
    elif in_memory_pr is not None:
        if pr.parent_program is None:
            pr.parent_program = in_memory_pr.parent_program
        if pr.name == "(Unnamed program results)":
            pr.name = in_memory_pr.name
        if pr.num_shots is None:
            pr.num_shots = in_memory_pr.num_shots
        if pr.max_frame_limit is None:
            pr.max_frame_limit = in_memory_pr.max_frame_limit
    return pr


def _process_and_checkpoint_item(
    process_item: Callable[..., Any],
    item: Any,
    index: int,
    static_kwargs: dict[str, Any],
    shot_executor: Any,
    n_shot_batches: int | None,
    keep_shot_results: bool,
    shot_checkpoint_subdir: Callable[[int], Path | None] | None,
    item_checkpoint_dir: Path | None,
    results_filename: str = "results.h5",
) -> dict[str, Any]:
    """Call process_item, checkpoint its result to a worker file, and return
    a dict for the caller to thread upward.

    When checkpointing, everything is already durable on disk, so only
    `{"_reduced_results": ...}` is returned (the sole key `on_item_done`
    reads); otherwise the full `_ALWAYS_MERGED_ATTRS` runner-side mapping is
    returned for the no-checkpoint branch's own in-memory merge.
    `process_item`'s own `"program_results"` entry (when `keep_shot_results`)
    is checkpointed to the worker file directly below but excluded either
    way, to avoid forwarding a lazy-loading `ProgramResults`'s open HDF5
    handle across a pickle boundary in the parallel case.

    Parameters
    ----------
    results_filename : str, optional
        Filename for checkpoint loading (default "results.h5").
    """
    # Build kwargs for this item
    extra_kwargs = static_kwargs.copy()
    if keep_shot_results:
        extra_kwargs["keep_shot_results"] = True

    aux = process_item(
        item,
        index,
        shot_executor=shot_executor,
        n_shot_batches=n_shot_batches,
        **extra_kwargs,
    )

    in_memory_pr = aux.get("program_results")

    # Checkpoint result to worker file
    if item_checkpoint_dir is not None:
        worker_file_path = (
            item_checkpoint_dir / f"worker_{worker_id()}_runner.h5"
        )
        # If keep_shot_results is enabled, retrieve ProgramResults
        pr = None
        if keep_shot_results:
            pr = _resolve_kept_program_results(
                index,
                shot_checkpoint_subdir,
                in_memory_pr,
                results_filename=results_filename,
            )
        for (
            worker_key,
            runner_key,
            value_use_dataset,
        ) in MultiProgramRunner._ALWAYS_MERGED_ATTRS:
            _write_dict_entry_with_retry(
                worker_file_path,
                worker_key,
                index,
                aux[runner_key],
                value_use_dataset=value_use_dataset,
            )
        if keep_shot_results and pr is not None:
            _write_dict_entry_with_retry(
                worker_file_path,
                "_program_results",
                index,
                pr,
                value_use_dataset=False,
            )

    if item_checkpoint_dir is not None:
        return {"_reduced_results": aux["_reduced_results"]}
    return {
        runner_key: aux[runner_key]
        for _, runner_key, _ in MultiProgramRunner._ALWAYS_MERGED_ATTRS
    }


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
    results_filename: str = "results.h5",
) -> dict[int, dict[str, Any]]:
    """Execute remaining items serially. Returns {index: aux} for in-memory
    results, where each `aux` is the dict returned by
    `_process_and_checkpoint_item` (reduced result plus timing attributes).

    Parameters
    ----------
    results_filename : str, optional
        Filename for checkpoint loading (default "results.h5").
    """
    shot_executor = resolve_shot_executor(
        parallel_strategy.shot_executor
        if parallel_strategy is not None
        else None
    )
    n_shot_batches = (
        parallel_strategy.n_shot_batches
        if parallel_strategy is not None
        else None
    )
    results_dict: dict[int, dict[str, Any]] = {}

    for index, item in remaining:
        aux = _process_and_checkpoint_item(
            process_item,
            item,
            index,
            static_kwargs,
            shot_executor,
            n_shot_batches,
            keep_shot_results,
            shot_checkpoint_subdir,
            item_checkpoint_dir,
            results_filename=results_filename,
        )

        results_dict[index] = aux

        if on_item_done is not None:
            on_item_done(index, item, aux["_reduced_results"])

        if pbar is not None:
            pbar.update(1)

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
            # A shared decode_cache is required across this loop so that
            # a Serializable value referenced by more than one entry (a
            # ProgramResults sharing a parent QuantumProgram, say) decodes
            # to the same real object everywhere, rather than an
            # unresolved DeferredRef past its first occurrence.
            decode_cache = ResolvingDecodeCache(root=f, format="hdf5")
            for key, value in iter_dict_attr_entries(
                f,
                "results",
                start_index=consumed_count,
                decode_cache=decode_cache,
            ):
                consumed_count += 1
                _mark_observed_and_notify(
                    key, value, observed_indices, items_map, on_item_done, pbar
                )
    except (BlockingIOError, OSError, KeyError):
        # Transient lock conflict, missing attribute, or file corruption;
        # skip this file for now
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
    results_filename: str = "results.h5",
) -> dict[int, dict[str, Any]]:
    """Execute remaining items in parallel with checkpointing and polling.

    Returns {index: aux} for in-memory results (when no checkpointing), where
    each `aux` is the dict returned by `_process_and_checkpoint_item`
    (reduced result plus timing attributes).

    Parameters
    ----------
    shots_pbar : Any, optional
        A tqdm progress bar for shot-level progress (only when parallel dispatch
        with checkpointing is enabled). If provided, it is updated during each
        poll tick with the total shots completed so far.
    num_shots_for_progress : int | None, optional
        Number of shots per item (for computing absolute shot totals). Only used
        if shots_pbar is provided.
    results_filename : str, optional
        Filename for checkpoint loading (default "results.h5").
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
            # Exclude items already in observed_indices to avoid double-counting
            in_flight_items = _read_worker_current_indices(item_checkpoint_dir)
            in_flight_items = in_flight_items - observed_indices
            total_shots_from_inflight = 0
            if shot_checkpoint_subdir is not None:
                for item_index in in_flight_items:
                    shot_subdir = shot_checkpoint_subdir(item_index)
                    if shot_subdir is not None:
                        shots_done = ProgramResults._count_done_shots(
                            shot_subdir, results_filename=results_filename
                        )
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
    n_shot_batches = (
        parallel_strategy.n_shot_batches
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
        n_shot_batches=n_shot_batches,
        keep_shot_results=keep_shot_results,
        shot_checkpoint_subdir=shot_checkpoint_subdir,
        results_filename=results_filename,
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
    newly_computed: dict[int, dict[str, Any]] = {}
    for chunk_results in chunk_results_list:
        for index, aux in chunk_results:
            newly_computed[index] = aux

    # For any items not already observed via worker file polling (i.e., when item_checkpoint_dir
    # is None), invoke on_item_done now so callers can collect results via the callback
    for index, aux in newly_computed.items():
        _mark_observed_and_notify(
            index,
            aux["_reduced_results"],
            observed_indices,
            items_map,
            on_item_done,
            pbar,
        )

    return newly_computed


def _generic_chunk_worker(
    chunk: list[tuple[int, T]],
    process_item: Callable[..., Any],
    static_kwargs: dict[str, Any],
    item_checkpoint_dir: Path | None,
    shot_executor: Any,
    n_shot_batches: int | None = None,
    keep_shot_results: bool = False,
    shot_checkpoint_subdir: Callable[[int], Path | None] | None = None,
    results_filename: str = "results.h5",
) -> list[tuple[int, dict[str, Any]]]:
    """Worker function for parallel execution of a chunk.

    Returns a list of (index, aux) pairs, where each `aux` is the dict
    returned by `_process_and_checkpoint_item` (reduced result plus timing
    attributes).

    Parameters
    ----------
    results_filename : str, optional
        Filename for checkpoint loading (default "results.h5").
    """
    pin_worker_threads()
    shot_executor = resolve_shot_executor(shot_executor)

    results = []
    for index, item in chunk:
        # Write current_item_index to worker file if checkpointing is enabled
        if item_checkpoint_dir is not None:
            worker_file_path = (
                item_checkpoint_dir / f"worker_{worker_id()}_runner.h5"
            )
            _write_current_item_index_with_retry(worker_file_path, index)

        aux = _process_and_checkpoint_item(
            process_item,
            item,
            index,
            static_kwargs,
            shot_executor,
            n_shot_batches,
            keep_shot_results,
            shot_checkpoint_subdir,
            item_checkpoint_dir,
            results_filename=results_filename,
        )

        results.append((index, aux))

    return results
