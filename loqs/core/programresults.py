#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""[](api:ProgramResults) definition."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from typing import Any, ClassVar
from pathlib import Path
import h5py

from loqs.internal import (
    Displayable,
    worker_id as _global_worker_id,
)
from loqs.internal.swmrledger import (
    held_swmr_writer,
    mark_ledger_item_done,
    read_swmr_ledger_done_union,
)
from loqs.core.history import (
    History,
    HistoryLike,
    HistoryCollectDataIndexTypes,
)

# Import QuantumProgram to avoid circular imports - we'll use it in type hints
from typing import TYPE_CHECKING, Iterable

from loqs.internal.checkpointstore import (
    SHOT_LAYOUT,
    BaseCheckpointStore,
    BatchReceipt,
    CheckpointStore,
    _NestedStore,
)
from loqs.internal.streamingmerge import _normalize_decoded_int_keyed_dict

if TYPE_CHECKING:
    from loqs.core.quantumprogram import QuantumProgram


_UNRESOLVED: Any = object()
"""Sentinel distinguishing "not yet lazily resolved" from a real resolved
value that happens to equal the plain default (e.g. `parent_program=None`,
or `name="(Unnamed program results)"` genuinely stored on disk)."""


_SHOT_LEDGER_FIELDS: tuple[str, ...] = ("done", "wall_clock_times")
"""Ledger fields a shot-level ledger tracks."""


def _held_shot_ledger(
    checkpoint_dir: str | Path,
    effective_worker_id: str,
    results_filename: str = "results.h5",
) -> AbstractContextManager[h5py.Group]:
    """Hold this writer's shot-level SWMR ledger in `checkpoint_dir` for a
    `with` block -- see `held_swmr_writer`. Nested holders share one handle,
    closed when the outermost one exits."""
    store = CheckpointStore(
        Path(checkpoint_dir).resolve(),
        SHOT_LAYOUT,
        canonical_filename=results_filename,
    )
    return held_swmr_writer(
        store.ledger_path(effective_worker_id), _SHOT_LEDGER_FIELDS
    )


def _mark_shot_batch_done(
    ledger_group: h5py.Group,
    shot_indices: list[int],
    wall_clock_times: dict[int, float],
) -> None:
    """Mark every shot in `shot_indices` done in the SWMR ledger, then flush
    both ledger datasets once. Callers must only call this after the file
    holding the shots' data is fully written and closed; a shot missing from
    `wall_clock_times` is recorded as `nan`."""
    for shot_index in shot_indices:
        mark_ledger_item_done(
            ledger_group,
            shot_index,
            wall_clock_times.get(shot_index, float("nan")),
            auto_flush=False,
        )
    ledger_group["done"].flush()
    ledger_group["wall_clock_times"].flush()


class ProgramResults(Displayable):
    """A container for the results of a quantum program execution.

    This class stores the shot histories and provides methods for collecting
    and analyzing data from the executed shots. It replaces the direct storage
    of shot histories in QuantumProgram.
    """

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    _SERIALIZE_ATTRS = [
        "shot_histories",
        "shot_wall_clock_times",
        "_unwritten_shots",
        "name",
        "parent_program",
        "num_shots",
        "max_frame_limit",
        "_results_filename",
    ]

    # Each streamed dict attr, paired with its own `value_use_dataset` flag
    # for `merge_dict_attr`: `shot_histories`' `History` values are never
    # native HDF5 scalars (always groups), while `shot_wall_clock_times`'
    # plain `float` values are, so they use the compact dataset format.
    _STREAMED_DICT_ATTRS: ClassVar[tuple[tuple[str, bool], ...]] = (
        ("shot_histories", False),
        ("shot_wall_clock_times", True),
    )

    # The store appends shots by navigating these attrs' raw HDF5 layout;
    # array-free-subtree collapse would break that, so exempt them.
    _NO_COLLAPSE_ATTRS: ClassVar[frozenset[str]] = frozenset(
        name for name, _ in _STREAMED_DICT_ATTRS
    )

    def __init__(
        self,
        shot_histories: dict[int, History] | None = None,
        shot_wall_clock_times: dict[int, float] | None = None,
        name: str = "(Unnamed program results)",
        parent_program: "QuantumProgram | str | Path | None" = None,
        checkpoint_enabled: bool = False,
        checkpoint_dir: str | Path | None = None,
        lazy_loading: bool = True,
        max_memory_shots: int = 100,
        num_shots: int | None = None,
        max_frame_limit: int | None = None,
        results_filename: str = "results.h5",
    ) -> None:
        """
        Parameters
        ----------
        shot_histories:
            A dictionary mapping shot indices to History objects.
            Defaults to None, which initializes an empty dict.

        shot_wall_clock_times:
            A dictionary mapping shot indices to each shot's wall-clock
            execution duration in seconds. Defaults to None, which
            initializes an empty dict.

        name:
            Name for logging

        parent_program:
            Reference to the parent QuantumProgram that generated these results.
            Can be a QuantumProgram object, a filepath string, or None.

        checkpoint_enabled:
            Whether checkpointing is enabled for this ProgramResults.

        checkpoint_dir:
            Directory where checkpoint files (including the parent program file,
            if written) are stored. If None, a default of `./checkpoints` is
            used when a write is actually needed.

        num_shots:
            The total number of shots for this run (for resume detection).

        max_frame_limit:
            The maximum frame limit used in this run (for resume detection).

        results_filename:
            Filename to use for the canonical results checkpoint file.
            Defaults to "results.h5".
        """
        self.shot_histories = (
            shot_histories if shot_histories is not None else {}
        )
        """Record of shot [](api:History) objects, mapped by shot index."""

        self.shot_wall_clock_times = (
            shot_wall_clock_times if shot_wall_clock_times is not None else {}
        )
        """Record of each shot's wall-clock execution duration in seconds,
        mapped by shot index."""

        self._unwritten_shots: set[int] = set()
        """Set of shot indices that have not been written to checkpoint files yet."""

        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        """Directory where checkpoint files are stored."""

        self._nested_store: _NestedStore | None = None
        """A read-only view of this object's entry inside another store's
        `_program_results`, if this is a lazy nested proxy."""

        self._name_value: Any = name
        """Backing field for the `name` property."""

        self.num_shots = num_shots
        """Total number of shots for this run (for resume detection)."""

        self._max_frame_limit_value: int | None = max_frame_limit
        """Backing field for the `max_frame_limit` property."""

        self._parent_program_value: "QuantumProgram | str | Path | None" = (
            parent_program
        )
        """Backing field for the `parent_program` property."""

        self._checkpoint_enabled = checkpoint_enabled
        """Whether checkpointing is enabled."""

        self._lazy_loading = lazy_loading
        """Whether lazy loading is enabled"""

        self._results_filename = results_filename
        """Filename for the canonical results checkpoint file."""

        self._max_memory_shots = max_memory_shots
        """Maximum number of shots to keep loaded."""

        self._memory_cache: dict[int, History] = {}  # Cache for loaded shots
        self._cache_order: list[int] = (
            []
        )  # Track order of cache usage for LRU eviction

        self._checkpoint_encode_cache: dict = {}
        """Persistent `Serializable.encode` cache shared across every `checkpoint()`
        call for this object's lifetime, so an object reused across shots is written
        once and cheaply referenced afterward, instead of re-expanded every time."""

        self._shot_store_cache: CheckpointStore | None = None
        """The store `_shot_store()` hands out. Not serialized."""

        # With checkpointing and a QuantumProgram parent, ensure the canonical
        # file exists and refer to it instead of the program.
        from loqs.core import QuantumProgram

        if checkpoint_enabled and isinstance(parent_program, QuantumProgram):
            self._ensure_snapshot()
            self.parent_program = str(self._shot_store().canonical_path)
            self._checkpoint_encode_cache = (
                self._shot_store().shared_encode_cache()
            )

    def _set_nested_shot_source(self, view: _NestedStore) -> None:
        """Configure this ProgramResults to read from a nested view.

        The view is a store's `nested(index)` over an entry of another
        object's `_program_results`. Shots are read through it, and `name`,
        `parent_program` and `max_frame_limit` resolve from it on first use.

        Parameters
        ----------
        view : _NestedStore
            The read-only nested view of this object's entry.
        """
        self._nested_store = view
        self._name_value = _UNRESOLVED
        self._parent_program_value = _UNRESOLVED
        if self._max_frame_limit_value is None:
            self._max_frame_limit_value = _UNRESOLVED

    @property
    def name(self) -> str:
        """Name for logging. Resolves lazily from the nested source on
        first access if this is a lazy nested-source proxy that hasn't
        resolved it yet."""
        if self._name_value is _UNRESOLVED:
            self._name_value = self._resolve_nested_attr("name")
        return self._name_value

    @name.setter
    def name(self, value: str) -> None:
        self._name_value = value

    @property
    def parent_program(self) -> "QuantumProgram | str | Path | None":
        """Reference to the parent QuantumProgram that generated these
        results. Resolves lazily from the nested source on first access if
        this is a lazy nested-source proxy that hasn't resolved it yet."""
        if self._parent_program_value is _UNRESOLVED:
            self._parent_program_value = self._resolve_nested_attr(
                "parent_program"
            )
        return self._parent_program_value

    @parent_program.setter
    def parent_program(
        self, value: "QuantumProgram | str | Path | None"
    ) -> None:
        self._parent_program_value = value

    @property
    def max_frame_limit(self) -> "int | None":
        """Maximum frame limit for this run (for resume detection). Resolves
        lazily from the nested source on first access if this is a lazy
        nested-source proxy that hasn't resolved it yet."""
        if self._max_frame_limit_value is _UNRESOLVED:
            self._max_frame_limit_value = self._resolve_nested_attr(
                "max_frame_limit"
            )
        return self._max_frame_limit_value

    @max_frame_limit.setter
    def max_frame_limit(self, value: "int | None") -> None:
        self._max_frame_limit_value = value

    def _resolve_nested_attr(self, attr_name: str) -> Any:
        """Read one non-streamed attribute (`name`, `parent_program` or
        `max_frame_limit`) from this proxy's nested entry, without decoding
        its shots. Errors propagate, so a damaged entry is never hidden by
        a default.
        """
        assert self._nested_store is not None
        return self._nested_store.read_attrs((attr_name,))[attr_name]

    def _shot_store(self) -> CheckpointStore:
        """The shot-level checkpoint store for this object's directory and
        canonical filename, rebuilt only when either changes."""
        directory = (
            self._checkpoint_dir
            if self._checkpoint_dir is not None
            else Path("./checkpoints")
        )
        store = getattr(self, "_shot_store_cache", None)
        if (
            store is None
            or store.directory != directory
            or store.layout.canonical_filename != self._results_filename
        ):
            store = CheckpointStore(
                directory,
                SHOT_LAYOUT,
                canonical_filename=self._results_filename,
            )
            self._shot_store_cache = store
        return store

    def _ensure_snapshot(self) -> None:
        """Write the canonical file as a shot-free shell of this object if
        it doesn't exist yet. An existing file is never rewritten, so a
        resuming call keeps the config it was started with."""
        if self._checkpoint_dir is None:
            self._checkpoint_dir = Path("./checkpoints")
        store = self._shot_store()
        if store.canonical_path.exists():
            return
        shell = ProgramResults(
            name=self.name,
            parent_program=self.parent_program,
            num_shots=self.num_shots,
            max_frame_limit=self.max_frame_limit,
            results_filename=self._results_filename,
        )
        store.write_snapshot(shell)

    def add_shot(
        self,
        shot_index: int,
        history: HistoryLike,
        wall_clock_time: float | None = None,
    ) -> None:
        """Add a shot history to the results.

        Parameters
        ----------
        shot_index:
            The index of the shot to add.
        history:
            The History object for the shot.
        wall_clock_time:
            The shot's wall-clock execution duration in seconds. If None
            (the default), no entry is recorded for this shot in
            `shot_wall_clock_times`.
        """
        history = History(history)
        self.shot_histories[shot_index] = history
        self._unwritten_shots.add(shot_index)
        if wall_clock_time is not None:
            self.shot_wall_clock_times[shot_index] = wall_clock_time

    def collect_shot_data(
        self,
        key: str,
        indices: HistoryCollectDataIndexTypes,
        strip_none_entries: bool = False,
        return_counter: bool = False,
        frame_filter: Mapping[str, object] | None = None,
    ) -> list | Counter:
        """Collate frame data over executed shots.

        Parameters
        ----------
        key:
            See `key` in [](api:History.collect_data)

        indices:
            See `indices` in [](api:History.collect_data)

        strip_none_entries:
            See `strip_none_entries` in [](api:History.collect_data)

        return_counter:
            Whether to return using a collections.Counter or not (default).

        frame_filter:
            See `frame_filter` in [](api:History.collect_data)

        Returns
        -------
        list
            List of [](api:History.collect_data) outputs per shot
        """
        data = [
            h.collect_data(
                key,
                indices,
                strip_none_entries=strip_none_entries,
                frame_filter=frame_filter,
            )
            for h in self._iter_histories_in_order()
        ]
        return Counter(data) if return_counter else data

    def _iter_histories_in_order(self) -> Iterator[History]:
        """Yield every available shot's history in ascending shot order.

        The reader (a nested view or the checkpoint directory's store) gives
        the stored shots not in memory in one pass, one decoded value at a
        time; without one only the shots in memory are yielded.
        """
        in_memory = sorted(self.shot_histories)
        reader = self._shot_reader()
        if reader is None:
            yield from (self.shot_histories[i] for i in in_memory)
            return

        stored = reader.iter_items(
            keys=[k for k in reader.keys() if k not in self.shot_histories]
        )
        pending = next(stored, None)
        for idx in in_memory:
            while pending is not None and pending[0] < idx:
                yield pending[1]
                pending = next(stored, None)
            yield self.shot_histories[idx]
        while pending is not None:
            yield pending[1]
            pending = next(stored, None)

    def _shot_reader(self) -> BaseCheckpointStore | None:
        """The store shots are read from: the nested view of a lazy proxy,
        else the shot store when a checkpoint directory is set, else
        `None`."""
        if self._nested_store is not None:
            return self._nested_store
        if self._checkpoint_dir is not None:
            return self._shot_store()
        return None

    def mark_shots_as_written(self, shot_indices: list[int]) -> None:
        """Mark shots as having been written to checkpoint files.

        Parameters
        ----------
        shot_indices:
            List of shot indices to mark as written.
        """
        for shot_index in shot_indices:
            if shot_index in self._unwritten_shots:
                self._unwritten_shots.remove(shot_index)

    def get_unwritten_shots(self) -> list[int]:
        """Get a list of shot indices that have not been written to checkpoint files.

        Returns
        -------
        list
            List of unwritten shot indices.
        """
        return list(self._unwritten_shots)

    @classmethod
    def _from_decoded_attrs(cls, attr_dict) -> "ProgramResults":
        """Create a ProgramResults object from decoded attributes dictionary."""
        # shot_histories may be absent entirely (a checkpoint snapshot
        # written before any shot data existed can have this attribute
        # reset, awaiting the first real write to establish its storage
        # format); treat that the same as an explicitly empty dict.
        shot_histories = _normalize_decoded_int_keyed_dict(
            attr_dict.get("shot_histories") or {}
        )

        # shot_wall_clock_times gets the same string-key-to-int treatment as
        # shot_histories above, plus a value-to-float cast (its values are
        # plain floats rather than Serializable objects already typed).
        shot_wall_clock_times = _normalize_decoded_int_keyed_dict(
            attr_dict.get("shot_wall_clock_times") or {}, value_cast=float
        )

        obj = cls(
            shot_histories=shot_histories,
            shot_wall_clock_times=shot_wall_clock_times,
            name=attr_dict["name"],
            parent_program=attr_dict["parent_program"],
            num_shots=attr_dict.get("num_shots"),
            max_frame_limit=attr_dict.get("max_frame_limit"),
            results_filename=attr_dict.get("_results_filename", "results.h5"),
        )

        # Set internal attributes that aren't in the constructor
        obj._unwritten_shots = attr_dict["_unwritten_shots"]

        return obj

    def checkpoint(
        self,
        checkpoint_dir: str | Path | None = None,
        worker_id: str | None = None,
    ) -> BatchReceipt:
        """Write every currently-unwritten shot to the checkpoint files.

        Shots are marked done in the writer's ledger afterwards.

        Parameters
        ----------
        checkpoint_dir:
            Directory for checkpoint files. If None, uses `./checkpoints`.
        worker_id:
            Identifies the writer. With None (one writer per directory),
            shots are appended to the canonical `results.h5`. With an id,
            they go to that writer's own payload file, which
            `consolidate_checkpoints()` later merges into `results.h5`.
            An id can be reused after its writer was killed, but not while
            that writer is alive.

        Returns
        -------
        BatchReceipt
            The keys written and the file holding them; keys are empty when
            nothing was unwritten.
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path("./checkpoints")
        else:
            checkpoint_dir = Path(checkpoint_dir)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = checkpoint_dir

        shots_to_checkpoint = self.get_unwritten_shots()
        if not shots_to_checkpoint:
            return BatchReceipt((), None)  # Nothing to checkpoint

        effective_worker_id = (
            worker_id if worker_id is not None else _global_worker_id()
        )

        # `shot_histories` always; `shot_wall_clock_times` where present.
        entries: dict[int, dict[str, Any]] = {}
        for shot_index in shots_to_checkpoint:
            entry: dict[str, Any] = {
                "shot_histories": self.shot_histories[shot_index]
            }
            if shot_index in self.shot_wall_clock_times:
                entry["shot_wall_clock_times"] = self.shot_wall_clock_times[
                    shot_index
                ]
            entries[shot_index] = entry

        wall_clock_times = {
            shot_index: self.shot_wall_clock_times[shot_index]
            for shot_index in shots_to_checkpoint
            if shot_index in self.shot_wall_clock_times
        }

        self._ensure_snapshot()
        store = self._shot_store()
        with _held_shot_ledger(
            checkpoint_dir, effective_worker_id, self._results_filename
        ) as ledger_group:
            if worker_id is None:
                receipt = store.put_batch(
                    entries, encode_cache=self._checkpoint_encode_cache
                )
            else:
                receipt = store.put_batch(entries, worker_id=worker_id)
            # Marked done only once the batch is committed.
            _mark_shot_batch_done(
                ledger_group, shots_to_checkpoint, wall_clock_times
            )

        self.mark_shots_as_written(shots_to_checkpoint)

        # Implement lazy loading: remove written shots from memory
        if self._lazy_loading:
            for shot_index in shots_to_checkpoint:
                if shot_index in self.shot_histories:
                    del self.shot_histories[shot_index]
        return receipt

    def mark_shots_checkpointed(self, shot_indices: list[int]) -> None:
        """Record that the given shots are already durably checkpointed
        somewhere else (e.g. by the worker that computed them), without
        writing anything here. Lets a driver's own `ProgramResults` still
        honor `lazy_loading` (bounding its own memory) for shots it
        received from a worker rather than checkpointing itself.

        Parameters
        ----------
        shot_indices:
            Shot indices to mark as already-checkpointed.
        """
        self.mark_shots_as_written(list(shot_indices))
        if self._lazy_loading:
            for shot_index in shot_indices:
                if shot_index in self.shot_histories:
                    del self.shot_histories[shot_index]

    @staticmethod
    def _count_live_done_shots(checkpoint_dir: Path) -> int:
        """Count the shots marked done in a directory's SWMR shot ledgers.

        Safe to poll while writers are running: it opens only
        `worker_*_checkpoint.h5` files, each in SWMR read mode, and never
        `results.h5` or a payload file. Shots that exist only in
        `results.h5` (e.g. from an earlier run) are not counted, and a
        pre-fix data file at the ledger name is skipped.

        Parameters
        ----------
        checkpoint_dir : Path
            Directory holding the shot ledgers.

        Returns
        -------
        int
            Number of unique shot indices marked done across the ledgers.
        """
        store = CheckpointStore(checkpoint_dir, SHOT_LAYOUT)
        return len(read_swmr_ledger_done_union(store.ledger_paths()))

    def load_checkpoint(
        self,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        """Load checkpoint data from disk through the checkpoint store.

        Metadata still at defaults is filled from the canonical file, and
        shots and wall-clock times not already in memory are loaded from it
        and any payloads; loaded shots leave the unwritten set. A missing
        directory or canonical file loads nothing; an unreadable file
        raises `CheckpointReadError`.

        Parameters
        ----------
        checkpoint_dir:
            Directory containing checkpoint files. If None, uses the
            default checkpoint directory.
        """
        if checkpoint_dir is None:
            if self._checkpoint_dir is None:
                checkpoint_dir = Path("./checkpoints")
            else:
                checkpoint_dir = self._checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            return  # No checkpoint directory

        store = CheckpointStore(
            checkpoint_dir,
            SHOT_LAYOUT,
            canonical_filename=self._results_filename,
        )
        if not store.canonical_path.exists():
            return  # No canonical file

        # Restore stored metadata only where self's fields are still unset,
        # never overwriting values a caller set deliberately.
        stored = store.read_attrs(
            ("name", "parent_program", "num_shots", "max_frame_limit")
        )
        if (
            self.name == "(Unnamed program results)"
            and stored["name"] != "(Unnamed program results)"
        ):
            self.name = stored["name"]
        if (
            self.parent_program is None
            and stored["parent_program"] is not None
        ):
            self.parent_program = stored["parent_program"]
        if self.num_shots is None and stored["num_shots"] is not None:
            self.num_shots = stored["num_shots"]
        if (
            self.max_frame_limit is None
            and stored["max_frame_limit"] is not None
        ):
            self.max_frame_limit = stored["max_frame_limit"]

        # Shots and wall-clock times are independent: a shot may have one
        # but not the other. Entries already in memory are not read again.
        new_shots = [k for k in store.keys() if k not in self.shot_histories]
        self._merge_loaded_entries(store.iter_items(keys=new_shots), ())
        new_times = [
            k
            for k in store.keys("shot_wall_clock_times")
            if k not in self.shot_wall_clock_times
        ]
        self._merge_loaded_entries(
            (),
            store.iter_items("shot_wall_clock_times", keys=new_times),
        )

    def _merge_loaded_entries(
        self,
        loaded_histories: Iterable[tuple[Any, Any]],
        loaded_wall_clock_times: Iterable[tuple[Any, Any]],
    ) -> None:
        """Merge loaded `(shot index, value)` entries into memory, keeping
        any entry already present. Keys become `int` and wall-clock values
        `float`, and every loaded shot leaves `_unwritten_shots`."""
        for shot_index, history in loaded_histories:
            shot_index = int(shot_index)
            self.shot_histories.setdefault(shot_index, history)
            self._unwritten_shots.discard(shot_index)
        for shot_index, wall_clock_time in loaded_wall_clock_times:
            self.shot_wall_clock_times.setdefault(
                int(shot_index), float(wall_clock_time)
            )

    def consolidate_checkpoints(
        self,
        checkpoint_dir: str | Path | None = None,
    ) -> Path:
        """Merge all payloads in `checkpoint_dir` into the canonical file.

        The merged payloads and ledgers are then removed. An unreadable
        payload raises `CheckpointReadError` and nothing is deleted. Run
        only when no writer is live.

        Parameters
        ----------
        checkpoint_dir:
            Directory containing checkpoint files to consolidate.

        Returns
        -------
        Path
            Path to the canonical checkpoint file.
        """
        if checkpoint_dir is None:
            if self._checkpoint_dir is None:
                checkpoint_dir = Path("./checkpoints")
            else:
                checkpoint_dir = self._checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = checkpoint_dir
        self._ensure_snapshot()
        store = self._shot_store()
        store.consolidate(encode_cache=self._checkpoint_encode_cache)
        return store.canonical_path

    def get_shot_history(self, shot_index: int) -> History | None:
        """Get a shot history, loading from checkpoint if lazy loading is on.

        Raises `CheckpointReadError` if the store's file for the shot is
        unreadable.

        Parameters
        ----------
        shot_index:
            Index of the shot to retrieve.

        Returns
        -------
        History | None
            The requested History object, or None if not found.
        """
        # Check if shot is in memory cache first
        if hasattr(self, "_lazy_loading") and self._lazy_loading:
            if shot_index in self._memory_cache:
                # Move to end of cache order (most recently used)
                self._cache_order.remove(shot_index)
                self._cache_order.append(shot_index)
                return self._memory_cache[shot_index]

            # Check if shot is in shot_histories (for non-checkpointed runs)
            if shot_index in self.shot_histories:
                return self.shot_histories[shot_index]

            # Not cached: load from the reader (a missing shot is None; a
            # corrupt file raises `CheckpointReadError`).
            # Known and deferred: right after `collect_shot_data` this read
            # is about 1 ms slower per call than before the checkpoint store
            # (cold and repeated reads aren't). Probably the collect leaves
            # no decoded shots for it to hit; keeping them would cost memory.
            loaded = False
            reader = self._shot_reader()
            if reader is not None:
                try:
                    history = reader.get(shot_index)
                except KeyError:
                    pass
                else:
                    self._memory_cache[shot_index] = history
                    loaded = True
            if loaded:
                # Successfully loaded, move to end of cache order
                self._cache_order.append(shot_index)

                # Check if we need to evict from cache
                if len(self._cache_order) > self._max_memory_shots:
                    oldest_shot = self._cache_order.pop(0)
                    del self._memory_cache[oldest_shot]

                return self._memory_cache[shot_index]

            return None
        else:
            # Normal mode: check if shot is in memory
            if shot_index not in self.shot_histories:
                return None

            return self.shot_histories[shot_index]
