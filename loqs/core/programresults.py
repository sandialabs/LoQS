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
from collections.abc import Mapping
from typing import ClassVar
from pathlib import Path
import h5py
import numpy as np
from datetime import datetime
import uuid

from loqs.internal import Displayable, Serializable
from loqs.internal.serializable import ResolvingDecodeCache
from loqs.core.history import (
    History,
    HistoryLike,
    HistoryCollectDataIndexTypes,
)
from loqs.core import Frame

# Import QuantumProgram to avoid circular imports - we'll use it in type hints
from typing import TYPE_CHECKING, Iterable

from loqs.internal.encoder.hdf5encoder import HDF5Encoder
from loqs.internal.streamingmerge import (
    merge_dict_attr,
    iter_dict_attr_entries,
    get_dict_attr_value,
    get_dict_attr_group,
    get_dict_attr_keys,
)

if TYPE_CHECKING:
    from loqs.core.quantumprogram import QuantumProgram


def _resolve_checkpoint_object_group(parent_group: h5py.Group) -> h5py.Group:
    """Navigate from a checkpoint file (or subgroup) to its actual object's encoded group.

    Handles both file layouts:
    - Bootstrap path (_write_shot_entries): Top-level is directly the object group
      (Serializable_1 with encode_type).
    - Fresh envelope path (_write_results_snapshot_if_fresh): Top-level is a
      wrapper (/root with version attr), contains the object group.

    Starting from parent_group, descends through single-child wrapper groups
    (those with no encode_type attribute and only a "version" attribute) until
    reaching a group that either has encode_type or has more than one child.

    Parameters
    ----------
    parent_group : h5py.Group
        The file or subgroup to resolve.

    Returns
    -------
    h5py.Group
        The actual object's encoded group (should have encode_type attribute).
    """
    current = parent_group
    while len(current.keys()) == 1:
        # Only descend if current has no encode_type (not an actual object yet)
        if "encode_type" in current.attrs:
            return current
        # Descend to the single child
        first_child_name = next(iter(current.keys()))
        first_child = current[first_child_name]
        if isinstance(first_child, h5py.Group):
            current = first_child
        else:
            # Child is a dataset, stop here
            return current
    # Stop if we have multiple children or if encode_type is present
    return current


class ProgramResults(Displayable):
    """A container for the results of a quantum program execution.

    This class stores the shot histories and provides methods for collecting
    and analyzing data from the executed shots. It replaces the direct storage
    of shot histories in QuantumProgram.
    """

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    _SERIALIZE_ATTRS = [
        "shot_histories",
        "_unwritten_shots",
        "name",
        "parent_program",
        "num_shots",
        "max_frame_limit",
        "_results_filename",
    ]

    # `_write_shot_entries`/`merge_dict_attr` navigate directly into this
    # attr's raw HDF5 structure (dict -> keys/values -> iterable, one group
    # per shot) to append new shots cheaply, bypassing the normal recursive
    # decode entirely -- HDF5's array-free-subtree collapse would silently
    # break that navigation whenever a batch of shots happens to have no
    # array anywhere in it (e.g. an all-classical program with no quantum
    # state), so this attr is exempted from collapse.
    _NO_COLLAPSE_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {"shot_histories"}
    )

    def __init__(
        self,
        shot_histories: dict[int, History] | None = None,
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

        self._unwritten_shots = set()
        """Set of shot indices that have not been written to checkpoint files yet."""

        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        """Directory where checkpoint files are stored."""

        self._worker_id = None
        """Which writer's checkpoint file this object last read/wrote --
        `None` for the un-suffixed `results.h5` (also what
        `consolidate_checkpoints` itself writes), or a `hostname_pid`-style
        string identifying one specific writer's own file."""

        self._nested_source_file: Path | None = None
        """File path containing a nested group source (an entry inside
        another object's _program_results dict attribute), if this
        ProgramResults was constructed to load shots from a nested source."""

        self._nested_source_index: int | None = None
        """Integer key of this ProgramResults inside a parent's _program_results
        dict attribute, if loaded from a nested source."""

        self.name = name
        """Name for logging"""

        self.num_shots = num_shots
        """Total number of shots for this run (for resume detection)."""

        self.max_frame_limit = max_frame_limit
        """Maximum frame limit for this run (for resume detection)."""

        self.parent_program = parent_program
        """Reference to the parent QuantumProgram that generated these results."""

        self._checkpoint_enabled = checkpoint_enabled
        """Whether checkpointing is enabled."""

        self._lazy_loading = lazy_loading
        """Whether lazy loading is enabled"""

        self._results_filename = results_filename
        """Filename for the canonical results checkpoint file."""

        self._max_memory_shots = max_memory_shots
        """Maximum number of shots to keep loaded."""

        self._memory_cache = {}  # Cache for loaded shots
        self._cache_order = []  # Track order of cache usage for LRU eviction

        self._checkpoint_encode_cache: dict = {}
        """Persistent `Serializable.encode` cache shared across every `checkpoint()`
        call for this object's lifetime, so an object reused across shots is written
        once and cheaply referenced afterward, instead of re-expanded every time."""

        self._checkpoint_decode_cache: ResolvingDecodeCache = ResolvingDecodeCache(
            root=None, format="hdf5"
        )
        """Persistent `Serializable.decode` cache shared across lazy shot loading
        calls; `_root` is re-pointed at each freshly-opened file handle."""

        # If checkpointing is enabled and parent_program is a QuantumProgram object,
        # write results.h5 (this whole ProgramResults object) if it doesn't already
        # exist, then use it instead of parent_program for cache building
        from loqs.core import QuantumProgram

        if checkpoint_enabled and isinstance(parent_program, QuantumProgram):
            self._write_results_snapshot_if_fresh()
            # Build encode_cache by decoding the written results and reversing cache mapping
            self._build_encode_cache_from_parent_program()

    def _set_nested_shot_source(self, source_file: Path, index: int) -> None:
        """Configure this ProgramResults to load shots from a nested source.

        Sets up internal fields to point to an entry inside another object's
        own `_program_results` dict attribute, rather than a standalone
        checkpoint file.

        Parameters
        ----------
        source_file : Path
            Path to the HDF5 file containing the parent object.
        index : int
            Integer key of this ProgramResults inside the parent's
            `_program_results` dict attribute.
        """
        self._nested_source_file = Path(source_file)
        self._nested_source_index = index

    def _write_results_snapshot_if_fresh(self) -> None:
        """Write the entire ProgramResults (including nested parent_program) to
        results.h5 only if that file doesn't already exist. The written
        shot_histories attribute (empty dict initially) is already in the state
        needed for merge_dict_attr to extend it later. Then update
        self.parent_program to point to results.h5 (as a string path).
        """
        # Set default checkpoint_dir if needed
        if self._checkpoint_dir is None:
            self._checkpoint_dir = Path("./checkpoints")

        # Ensure checkpoint directory exists
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        results_path = self._checkpoint_dir / self._results_filename

        # Write only if results.h5 doesn't exist yet (this is what makes
        # a resuming call not re-derive a fresh config from a possibly-different self)
        if not results_path.exists():
            from loqs.internal.serializable import Serializable

            Serializable.write(self, results_path, format="hdf5")

        # Always reassign parent_program to the results.h5 path (whether or not
        # we just wrote it), so _build_encode_cache_from_parent_program works
        # identically on both fresh and resuming calls
        self.parent_program = str(results_path)

    def _build_encode_cache_from_parent_program(self) -> None:
        """Build an encode_cache by decoding results.h5 as a ProgramResults
        (the nested QuantumProgram decodes normally underneath) and
        reversing its own cache mapping."""
        if not isinstance(self.parent_program, (str, Path)):
            return

        try:
            from loqs.internal.serializable import Serializable

            decode_cache = {}
            ProgramResults.read(self.parent_program, decode_cache=decode_cache)

            # Decode cache is cache_id to object
            # Encode cache is id(object) to cache_id
            self._checkpoint_encode_cache = {
                id(v): k for k, v in decode_cache.items()
            }
        except Exception:
            # If there's any error reading the results or building the cache,
            # just continue without the cache - it's not critical for functionality
            pass

    def add_shot(self, shot_index: int, history: HistoryLike) -> None:
        """Add a shot history to the results.

        Parameters
        ----------
        shot_index:
            The index of the shot to add.
        history:
            The History object for the shot.
        """
        history = History(history)
        self.shot_histories[shot_index] = history
        self._unwritten_shots.add(shot_index)

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
        if self.shot_histories:
            histories = list(self.shot_histories.values())
        elif hasattr(self, "_lazy_loading") and self._lazy_loading:
            shot_indices = self._get_available_shot_indices()
            loaded = [self.get_shot_history(idx) for idx in shot_indices]
            histories = [h for h in loaded if h is not None]
        else:
            histories = []

        data = [
            h.collect_data(
                key,
                indices,
                strip_none_entries=strip_none_entries,
                frame_filter=frame_filter,
            )
            for h in histories
        ]
        return Counter(data) if return_counter else data

    def _get_available_shot_indices(self) -> list[int]:
        """Return the available shot indices for lazy loading: checked first
        against a nested `runner.h5` shot source, then a standalone
        checkpoint directory, falling back to `range(num_shots)` (or
        whatever's already in the in-memory cache) if neither is present.
        """
        from loqs.internal.streamingmerge import get_dict_attr_keys

        if (
            self._nested_source_file is not None
            and self._nested_source_file.exists()
        ):
            try:
                with h5py.File(self._nested_source_file, "r") as f:
                    source_group = self._resolve_shot_source_group(f)
                    if source_group is not None:
                        keys = get_dict_attr_keys(
                            source_group, "shot_histories"
                        )
                        if keys:
                            return keys
            except (KeyError, OSError):
                pass

        if self._checkpoint_dir is not None and self._checkpoint_dir.exists():
            results_file = self._checkpoint_dir / self._results_filename
            if results_file.exists():
                try:
                    with h5py.File(results_file, "r") as f:
                        keys = get_dict_attr_keys(f, "shot_histories")
                        if keys:
                            return keys
                except (KeyError, OSError):
                    pass

        if hasattr(self, "num_shots") and self.num_shots is not None:
            return list(range(self.num_shots))

        return sorted(self._memory_cache.keys())

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
        # Handle shot_histories: convert string keys back to integers if needed
        shot_histories = attr_dict["shot_histories"]
        if shot_histories and all(
            isinstance(k, str) and k.isdigit() for k in shot_histories.keys()
        ):
            # Convert string keys to integers
            shot_histories = {int(k): v for k, v in shot_histories.items()}

        obj = cls(
            shot_histories=shot_histories,
            name=attr_dict["name"],
            parent_program=attr_dict["parent_program"],
            num_shots=attr_dict.get("num_shots"),
            max_frame_limit=attr_dict.get("max_frame_limit"),
        )

        # Set internal attributes that aren't in the constructor
        obj._unwritten_shots = attr_dict["_unwritten_shots"]

        return obj

    def checkpoint(
        self,
        checkpoint_dir: str | Path | None = None,
        worker_id: str | None = None,
    ) -> None:
        """Write every currently-unwritten shot to this writer's own checkpoint file.

        Always flushes everything `get_unwritten_shots()` currently reports
        -- there is no separate "batch index" to compute or track. How
        often this gets called (once per shot, once per batch of several)
        is entirely up to the caller; a `QuantumProgram.run()` worker
        processing `checkpoint_batch_size` shots at a time calls this once
        per batch, which is what actually controls durability granularity
        vs. I/O overhead.

        Parameters
        ----------
        checkpoint_dir:
            Directory to store checkpoint files. If None, uses `./checkpoints`.
        worker_id:
            A string identifying which physical writer this file belongs to
            (e.g. `f"{socket.gethostname()}_{os.getpid()}"`), so multiple
            concurrent writers never open the same file. If None, writes
            directly to the un-suffixed `results.h5` -- the same filename
            `consolidate_checkpoints` writes its own merged output under, so
            a single-writer caller can skip both worker identification and
            consolidation entirely.
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path("./checkpoints")
        else:
            checkpoint_dir = Path(checkpoint_dir)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = checkpoint_dir
        self._worker_id = worker_id

        shots_to_checkpoint = self.get_unwritten_shots()
        if not shots_to_checkpoint:
            return  # Nothing to checkpoint

        if worker_id is not None:
            filename = checkpoint_dir / f"worker_{worker_id}_checkpoint.h5"
        else:
            filename = checkpoint_dir / self._results_filename

        self._write_checkpoint_file(filename, shots_to_checkpoint)
        self.mark_shots_as_written(shots_to_checkpoint)

        # Implement lazy loading: remove written shots from memory
        if self._lazy_loading:
            for shot_index in shots_to_checkpoint:
                if shot_index in self.shot_histories:
                    del self.shot_histories[shot_index]

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

    def _write_checkpoint_file(
        self,
        filename: Path,
        shot_indices: list[int],
    ) -> None:
        """Write checkpoint data to an HDF5 file using standard Serializable encoding.

        Parameters
        ----------
        filename:
            Path to the checkpoint file.
        shot_indices:
            List of shot indices to write to the checkpoint.
        """
        # Prepare data to write - create a dict of unwritten shots
        unwritten_shot_histories = {}
        for shot_index in shot_indices:
            if shot_index in self.shot_histories:
                unwritten_shot_histories[shot_index] = self.shot_histories[
                    shot_index
                ]

        if not unwritten_shot_histories:
            return  # No data to write

        # Write to HDF5 file using standard Serializable encoding
        with h5py.File(
            filename, "a"
        ) as f:  # 'a' mode allows appending to existing files
            self._write_shot_entries(f, unwritten_shot_histories.items())

    def _write_shot_entries(
        self, h5_file: h5py.File, entries: Iterable[tuple[int, History]]
    ) -> None:
        """Stream shot entries into an HDF5 file's shot_histories dict
        attribute, creating it fresh or extending it, without ever
        materializing more than one entry in memory at a time.

        Parameters
        ----------
        h5_file:
            Open HDF5 file object in 'a' mode.
        entries:
            An iterable of (shot_index: int, history: History) pairs to append.
        """
        if len(h5_file.keys()) == 0:
            # Bootstrap an empty ProgramResults shell, then drop the empty
            # shot_histories skeleton it leaves behind -- the generic
            # encoder always writes an empty dict attribute in "groups"
            # format, which would otherwise prevent the merge_dict_attr
            # call below from picking "dataset" format for shot-index keys.
            Serializable.encode(
                ProgramResults(shot_histories={}),
                format="hdf5",
                h5_group=h5_file,
                encode_cache=self._checkpoint_encode_cache,
            )
            root_group = _resolve_checkpoint_object_group(h5_file)
            if "shot_histories" in root_group:
                del root_group["shot_histories"]

        # Resolve to the actual object group, handling both fresh-envelope
        # and bootstrap-created file layouts.
        root_group = _resolve_checkpoint_object_group(h5_file)

        # Shot-index keys stay a compact dataset (plain ints); History
        # values are never native scalars, so always end up as groups.
        merge_dict_attr(
            root_group,
            "shot_histories",
            entries,
            encode_cache=self._checkpoint_encode_cache,
            key_use_dataset=True,
            value_use_dataset=False,
        )

    @staticmethod
    def _load_done_shots(
        checkpoint_dir: Path, results_filename: str = "results.h5"
    ) -> dict[int, HistoryLike]:
        """Scan checkpoint_dir for results.h5 and every worker_*_checkpoint.h5,
        decode each, and return the union of their shot_histories.
        Explicitly skips *.tmp files (stale leftovers from a crash).

        Parameters
        ----------
        checkpoint_dir:
            Directory to scan for checkpoint files.
        results_filename:
            Filename for the canonical results checkpoint file.
            Defaults to "results.h5".

        Returns
        -------
        dict[int, HistoryLike]
            Union of all shot_histories found, mapping shot index to History.
            Returns empty dict if no checkpoints exist yet.
        """
        done = {}

        # First, read results.h5 if it exists
        results_file = checkpoint_dir / results_filename
        if results_file.exists():
            try:
                with h5py.File(results_file, "r") as f:
                    loaded = Serializable.decode(f, format="hdf5")
                    if (
                        isinstance(loaded, ProgramResults)
                        and loaded.shot_histories
                    ):
                        done.update(loaded.shot_histories)
            except Exception:
                pass  # Skip if we can't read this file

        # Then, read every worker_*_checkpoint.h5 (sorted, no .tmp files)
        worker_files = sorted(
            f
            for f in checkpoint_dir.glob("worker_*_checkpoint.h5")
            if not f.name.endswith(".tmp")
        )
        for worker_file in worker_files:
            try:
                with h5py.File(worker_file, "r") as f:
                    loaded = Serializable.decode(f, format="hdf5")
                    if (
                        isinstance(loaded, ProgramResults)
                        and loaded.shot_histories
                    ):
                        done.update(loaded.shot_histories)
            except Exception:
                pass  # Skip if we can't read this file

        return done

    @staticmethod
    def _count_done_shots(
        checkpoint_dir: Path, results_filename: str = "results.h5"
    ) -> int:
        """Count the number of unique shot indices in checkpoint files.

        Scans checkpoint_dir for results.h5 and every worker_*_checkpoint.h5,
        counts the union of their shot_histories keys, and returns the total count.
        Explicitly skips *.tmp files (stale leftovers from a crash).

        Uses get_dict_attr_keys to read only the keys without decoding any
        History values, making this cheap even for workers with many shots.

        Parameters
        ----------
        checkpoint_dir : Path
            Directory to scan for checkpoint files.
        results_filename:
            Filename for the canonical results checkpoint file.
            Defaults to "results.h5".

        Returns
        -------
        int
            Number of unique shot indices found across all checkpoint files.
            Returns 0 if no checkpoints exist yet.
        """
        checkpoint_dir = Path(checkpoint_dir)
        done_indices: set[int] = set()

        # First, count indices from results.h5 if it exists
        results_file = checkpoint_dir / results_filename
        if results_file.exists():
            try:
                with h5py.File(results_file, "r") as f:
                    keys = get_dict_attr_keys(f, "shot_histories")
                    done_indices.update(keys)
            except Exception:
                pass  # Skip if we can't read this file

        # Then, count indices from every worker_*_checkpoint.h5 (sorted, no .tmp)
        worker_files = sorted(
            f
            for f in checkpoint_dir.glob("worker_*_checkpoint.h5")
            if not f.name.endswith(".tmp")
        )
        for worker_file in worker_files:
            try:
                with h5py.File(worker_file, "r") as f:
                    keys = get_dict_attr_keys(f, "shot_histories")
                    done_indices.update(keys)
            except Exception:
                pass  # Skip if we can't read this file

        return len(done_indices)

    def load_checkpoint(
        self,
        checkpoint_dir: str | Path | None = None,
        worker_id: str | None = None,
    ) -> None:
        """Load checkpoint data from disk.

        Parameters
        ----------
        checkpoint_dir:
            Directory containing checkpoint files. If None, uses the default checkpoint directory.
        worker_id:
            Which writer's checkpoint file to load. If None (the common
            case -- also what `consolidate_checkpoints` writes its own
            merged output under), loads the un-suffixed `results.h5`.
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

        if worker_id is not None:
            pattern = f"worker_{worker_id}_checkpoint.h5"
        else:
            pattern = self._results_filename

        checkpoint_file = checkpoint_dir / pattern
        if checkpoint_file.exists():
            self._load_single_checkpoint_file(checkpoint_file)

    def _load_single_checkpoint_file(self, filename: Path) -> None:
        """Load data from a single checkpoint file using standard Serializable decoding.

        Parameters
        ----------
        filename:
            Path to the checkpoint file to load.
        """
        with h5py.File(filename, "r") as f:
            # Use standard Serializable decoding to load the ProgramResults
            loaded_results = Serializable.decode(f, format="hdf5")
            assert isinstance(loaded_results, ProgramResults)

            # Merge the loaded shot histories into our current results
            if loaded_results.shot_histories:
                # Merge shot histories, keeping track of which shots are already checkpointed
                for (
                    shot_index,
                    history,
                ) in loaded_results.shot_histories.items():
                    # Only add shots that we don't already have in memory
                    if shot_index not in self.shot_histories:
                        self.shot_histories[shot_index] = history
                    # Don't add to unwritten_shots since it's already checkpointed
                    if shot_index in self._unwritten_shots:
                        self._unwritten_shots.remove(shot_index)

    def consolidate_checkpoints(
        self,
        checkpoint_dir: str | Path | None = None,
        output_file: str | Path | None = None,
        delete_originals: bool = True,
    ) -> Path:
        """Merge every per-worker checkpoint file in `checkpoint_dir` directly
        into the output file via streaming-safe, entry-by-entry merging.

        Decodes and writes shots one at a time via `iter_dict_attr_entries`,
        so peak memory stays bounded to a single shot regardless of how many
        workers or shots are involved. Deletes each worker file only once its
        own entries are confirmed merged, so a crash mid-consolidation
        self-heals on retry (the worker file is still present, gets re-merged,
        and deduplication against already-merged keys ensures no duplicates).

        Parameters
        ----------
        checkpoint_dir:
            Directory containing checkpoint files to consolidate.
        output_file:
            Path for the consolidated output file. If None, writes to
            `checkpoint_dir / "results.h5"` -- the same filename
            `checkpoint(worker_id=None)` itself writes to, so a single-writer
            run and a many-writer run both end up readable via
            `load_checkpoint(worker_id=None)`.
        delete_originals:
            Whether to delete the original per-worker checkpoint files
            after consolidation.

        Returns
        -------
        Path
            Path to the consolidated checkpoint file.
        """
        if checkpoint_dir is None:
            if self._checkpoint_dir is None:
                checkpoint_dir = Path("./checkpoints")
            else:
                checkpoint_dir = self._checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if output_file is None:
            output_file = checkpoint_dir / self._results_filename
        else:
            output_file = Path(output_file)

        worker_files = sorted(checkpoint_dir.glob("worker_*_checkpoint.h5"))

        # Merge each worker file directly into output_file (created if
        # missing), deduplicating against already-merged keys so a retry
        # never double-counts.
        with h5py.File(output_file, "a") as out_f:
            for worker_file in worker_files:
                # Read already-merged keys from output_file so we can skip
                # duplicates and avoid corrupt entries on a retry
                already_merged = set()
                try:
                    keys = get_dict_attr_keys(out_f, "shot_histories")
                    already_merged.update(keys)
                except (KeyError, Exception):
                    # output_file might not have shot_histories yet on first call
                    pass

                # Stream only new entries from this worker file
                self._merge_worker_into_output(
                    worker_file, out_f, already_merged
                )

                # Only delete the worker file once its entries are confirmed
                # merged and present in output_file
                if delete_originals:
                    worker_file.unlink()

        # Ensure output_file has valid structure even if no workers were present
        if (
            not output_file.exists()
            or len(h5py.File(output_file, "r").keys()) == 0
        ):
            with h5py.File(output_file, "a") as out_f:
                if len(out_f.keys()) == 0:
                    self._write_shot_entries(out_f, iter(()))

        return output_file

    def _merge_worker_into_output(
        self,
        worker_file: Path,
        out_h5_file: h5py.File,
        already_merged: set[int],
    ) -> None:
        """Merge one worker's checkpoint file into the output file, skipping
        any shot indices already present in the output (deduplication safety
        for crash-recovery retries).

        Streams one shot at a time via iter_dict_attr_entries, so peak
        memory is bounded to a single shot.

        Parameters
        ----------
        worker_file:
            Path to the worker checkpoint file to read.
        out_h5_file:
            Already-open, writable HDF5 file to merge this worker's shots into.
        already_merged:
            Set of shot indices already present in out_h5_file's
            shot_histories. Any entry from the worker file whose key is in
            this set is skipped.
        """
        with h5py.File(worker_file, "r") as in_f:
            if len(in_f.keys()) == 0:
                return

            in_root_group = _resolve_checkpoint_object_group(in_f)
            # Fresh decode_cache per worker file, matching this file's own scope --
            # not a cache shared across separate worker files.
            # Use a generator expression (not a list comprehension) to filter lazily,
            # so entries are decoded and written one at a time, with peak memory
            # bounded to a single shot even when filtering duplicates.
            entries = (
                (key, value)
                for key, value in iter_dict_attr_entries(
                    in_root_group, "shot_histories", decode_cache={}
                )
                if key not in already_merged
            )
            # Consumed fully here, while `in_f` is still open.
            self._write_shot_entries(out_h5_file, entries)

    def _stream_checkpoint_file_into(
        self, filename: Path, out_h5_file
    ) -> None:
        """Stream one worker's checkpoint file into the output file, decoding
        and writing its shots one at a time so peak memory stays bounded to
        a single shot rather than the whole file's shot_histories dict.

        Parameters
        ----------
        filename:
            Path to the worker checkpoint file to read.
        out_h5_file:
            Already-open, writable HDF5 file/group to merge this worker's
            shots into.
        """
        with h5py.File(filename, "r") as in_f:
            if len(in_f.keys()) == 0:
                return

            in_root_group = _resolve_checkpoint_object_group(in_f)
            # Fresh decode_cache per file, matching this file's own scope --
            # not a cache shared across separate worker files.
            entries = iter_dict_attr_entries(
                in_root_group, "shot_histories", decode_cache={}
            )
            # Consumed fully here, while `in_f` is still open.
            self._write_shot_entries(out_h5_file, entries)

    def get_shot_history(self, shot_index: int) -> History | None:
        """Get a shot history, potentially loading from checkpoint if lazy loading is enabled.

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

            # If not in cache, try to load from checkpoint
            if (
                self._checkpoint_dir is not None
                and self._load_shot_from_checkpoint(shot_index)
            ):
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

    def _resolve_shot_source_group(self, f: h5py.File) -> h5py.Group | None:
        """Resolve the source group for shot loading (nested or standalone).

        If this ProgramResults was configured with a nested source via
        `_set_nested_shot_source`, navigates to and returns the nested group.
        Otherwise, returns the file's own top-level root group.

        Parameters
        ----------
        f : h5py.File
            An open HDF5 file (the caller already has it open).

        Returns
        -------
        h5py.Group | None
            The appropriate source group, or None if nested source is not
            properly configured.
        """
        if (
            self._nested_source_file is not None
            and self._nested_source_index is not None
        ):
            # get_dict_attr_group already navigates past any wrapper levels
            # (a plain worker scratch file's _program_results sits directly
            # at file root; a consolidated runner.h5's sits nested under its
            # own Serializable-encoded object group) via its own internal
            # `_resolve_dict_target_group` call, so no manual descent here.
            try:
                source_group = get_dict_attr_group(
                    f, "_program_results", self._nested_source_index
                )
                return source_group
            except (KeyError, TypeError):
                return None
        else:
            # Return the file's own top-level root group, handling both
            # fresh-envelope and bootstrap-created file layouts.
            if len(f.keys()) == 0:
                return None
            return _resolve_checkpoint_object_group(f)

    def _load_shot_from_checkpoint(self, shot_index: int) -> bool:
        """Load a specific shot from checkpoint files.

        Parameters
        ----------
        shot_index:
            Index of the shot to load.

        Returns
        -------
        bool
            True if shot was successfully loaded, False otherwise.
        """
        if self._nested_source_file is None:
            # Standalone file mode
            if (
                self._checkpoint_dir is None
                or not self._checkpoint_dir.exists()
            ):
                return False

            # Try to find the shot in this writer's own checkpoint file
            if self._worker_id is not None:
                checkpoint_file = (
                    self._checkpoint_dir
                    / f"worker_{self._worker_id}_checkpoint.h5"
                )
            else:
                checkpoint_file = self._checkpoint_dir / self._results_filename

            if not checkpoint_file.exists():
                return False

            return self._load_shot_from_single_file(
                checkpoint_file, shot_index
            )
        else:
            # Nested source mode
            return self._load_shot_from_single_file(
                self._nested_source_file, shot_index
            )

    def _load_shot_from_single_file(
        self, filename: Path, shot_index: int
    ) -> bool:
        """Load a shot from a checkpoint file without materializing the full object.

        Supports both nested and standalone sources. Opens the file, resolves
        the appropriate source group, then uses get_dict_attr_value to fetch
        only the requested shot's History from the shot_histories dict attribute,
        without decoding any other shots.

        Parameters
        ----------
        filename:
            Path to the checkpoint file.
        shot_index:
            Index of the shot to load.

        Returns
        -------
        bool
            True if shot was successfully loaded, False otherwise.
        """
        try:
            with h5py.File(filename, "r") as f:
                source_group = self._resolve_shot_source_group(f)
                if source_group is None:
                    return False

                # If source_group is from a nested dict entry, it contains
                # the raw Serializable-encoded wrapper, so we need to unwrap it
                # to get to the actual ProgramResults attributes
                if self._nested_source_file is not None:
                    # Unwrap the Serializable wrapper group
                    if len(source_group) == 0:
                        return False
                    actual_group = source_group[
                        next(iter(source_group.keys()))
                    ]
                else:
                    actual_group = source_group

                # Re-point the persistent decode cache at this freshly-opened
                # file handle (the previous one, if any, is already closed).
                decode_cache = getattr(self, "_checkpoint_decode_cache", None)
                if isinstance(decode_cache, ResolvingDecodeCache):
                    decode_cache._root = f
                else:
                    decode_cache = ResolvingDecodeCache(root=f, format="hdf5")
                    self._checkpoint_decode_cache = decode_cache

                # Use get_dict_attr_value to fetch only this one shot
                # without decoding all others
                try:
                    history = get_dict_attr_value(
                        actual_group,
                        "shot_histories",
                        shot_index,
                        decode_cache=decode_cache,
                    )
                except KeyError:
                    return False

                if self._lazy_loading:
                    self._memory_cache[shot_index] = history
                else:
                    self.shot_histories[shot_index] = history
                    # Remove from unwritten_shots since it's already checkpointed
                    if shot_index in self._unwritten_shots:
                        self._unwritten_shots.remove(shot_index)

                return True
        except (OSError, ValueError):
            return False

        return False
