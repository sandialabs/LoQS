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

from loqs.internal import Displayable, Serializable
from loqs.core.history import (
    History,
    HistoryLike,
    HistoryCollectDataIndexTypes,
)
from loqs.core import Frame

# Import QuantumProgram to avoid circular imports - we'll use it in type hints
from typing import TYPE_CHECKING

from loqs.internal.encoder.hdf5encoder import HDF5Encoder

if TYPE_CHECKING:
    from loqs.core.quantumprogram import QuantumProgram


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
    ]

    # `_merge_into_existing_checkpoint`/`_merge_iterable` navigate directly
    # into this attr's raw HDF5 structure (dict -> keys/values -> iterable,
    # one group per shot) to append new shots cheaply, bypassing the normal
    # recursive decode entirely -- HDF5's array-free-subtree collapse would
    # silently break that navigation whenever a batch of shots happens to
    # have no array anywhere in it (e.g. an all-classical program with no
    # quantum state), so this attr is exempted from collapse.
    _NO_COLLAPSE_ATTRS: ClassVar[frozenset[str]] = frozenset({"shot_histories"})

    def __init__(
        self,
        shot_histories: dict[int, History] | None = None,
        name: str = "(Unnamed program results)",
        parent_program: "QuantumProgram | str | Path | None" = None,
        checkpoint_enabled: bool = False,
        lazy_loading_enabled: bool = True,
        max_memory_shots: int = 100,
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
        """
        self.shot_histories = (
            shot_histories if shot_histories is not None else {}
        )
        """Record of shot [](api:History) objects, mapped by shot index."""

        self._unwritten_shots = set()
        """Set of shot indices that have not been written to checkpoint files yet."""

        self._checkpoint_dir = None
        """Directory where checkpoint files are stored."""

        self._worker_id = None
        """Which writer's checkpoint file this object last read/wrote --
        `None` for the un-suffixed `checkpoint.h5` (also what
        `consolidate_checkpoints` itself writes), or a `hostname_pid`-style
        string identifying one specific writer's own file."""

        self.name = name
        """Name for logging"""

        self.parent_program = parent_program
        """Reference to the parent QuantumProgram that generated these results."""

        self._checkpoint_enabled = checkpoint_enabled
        """Whether checkpointing is enabled."""

        self._lazy_loading_enabled = lazy_loading_enabled
        """Whether lazy loading is enabled"""

        self._max_memory_shots = max_memory_shots
        """Maximum number of shots to keep loaded."""

        self._memory_cache = {}  # Cache for loaded shots
        self._cache_order = []  # Track order of cache usage for LRU eviction

        self._checkpoint_encode_cache: dict = {}
        """Persistent `Serializable.encode` cache shared across every `checkpoint()`
        call for this object's lifetime, so an object reused across shots is written
        once and cheaply referenced afterward, instead of re-expanded every time."""

        # If checkpointing is enabled and parent_program is a QuantumProgram object,
        # we need to write it to file and store the filename instead
        from loqs.core import QuantumProgram

        if checkpoint_enabled and isinstance(parent_program, QuantumProgram):
            self._write_parent_program_to_file(parent_program)
            # Build encode_cache by decoding the written program and reversing cache mapping
            self._build_encode_cache_from_parent_program()

    def _write_parent_program_to_file(self, program) -> None:
        """Write the parent QuantumProgram to file and store the filename.

        Parameters
        ----------
        program:
            The QuantumProgram object to write to file.
        """
        # Create a temporary directory for the program file if checkpoint_dir is not set
        if self._checkpoint_dir is None:
            self._checkpoint_dir = Path("./checkpoints")
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique filename for the program
        program_filename = (
            self._checkpoint_dir
            / f"parent_program_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        )

        # Write the program to file
        program.write(program_filename, format="hdf5")

        # Store the filename instead of the program object
        self.parent_program = str(program_filename)

    def _build_encode_cache_from_parent_program(self) -> None:
        """Build an encode_cache by decoding the written parent program and reversing cache mapping."""
        if not isinstance(self.parent_program, (str, Path)):
            return

        try:
            # Import QuantumProgram here to avoid circular imports
            from loqs.core.quantumprogram import QuantumProgram

            # Read the parent program from file just to build the decode cache
            decode_cache = {}
            QuantumProgram.read(self.parent_program, decode_cache=decode_cache)

            # Decode cache is cache_id to object
            # Encode cache is id(object) to cache_id
            self._encode_cache = {id(v): k for k, v in decode_cache.items()}
        except Exception:
            # If there's any error reading the program or building the cache,
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
        data = [
            h.collect_data(
                key,
                indices,
                strip_none_entries=strip_none_entries,
                frame_filter=frame_filter,
            )
            for h in self.shot_histories.values()
        ]
        return Counter(data) if return_counter else data

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
            directly to the un-suffixed `checkpoint.h5` -- the same filename
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
            filename = checkpoint_dir / "checkpoint.h5"

        self._write_checkpoint_file(filename, shots_to_checkpoint)
        self.mark_shots_as_written(shots_to_checkpoint)

        # Implement lazy loading: remove written shots from memory
        if self._lazy_loading_enabled:
            for shot_index in shots_to_checkpoint:
                if shot_index in self.shot_histories:
                    del self.shot_histories[shot_index]

    def mark_shots_checkpointed(self, shot_indices: list[int]) -> None:
        """Record that the given shots are already durably checkpointed
        somewhere else (e.g. by the worker that computed them), without
        writing anything here. Lets a driver's own `ProgramResults` still
        honor `lazy_loading_enabled` (bounding its own memory) for shots it
        received from a worker rather than checkpointing itself.

        Parameters
        ----------
        shot_indices:
            Shot indices to mark as already-checkpointed.
        """
        self.mark_shots_as_written(list(shot_indices))
        if self._lazy_loading_enabled:
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
            self._update_single_file_checkpoint(f, unwritten_shot_histories)

    def _update_single_file_checkpoint(
        self, h5_file, unwritten_shot_histories: dict[int, History]
    ) -> None:
        """Update a single-file checkpoint by merging new shots into existing HDF5 structure.

        This method navigates to the correct HDF5 groups and adds new item_groups
        for the unwritten shots using Serializable.encode().

        Parameters
        ----------
        h5_file:
            Open HDF5 file object.
        unwritten_shot_histories:
            Dictionary mapping shot indices to History objects to be written.
        """
        # Check if this is a new file or existing checkpoint
        if len(h5_file.keys()) == 0:
            # New file - write the full ProgramResults structure
            self._write_full_checkpoint_structure(
                h5_file, unwritten_shot_histories
            )
        else:
            # Existing file - find the shot_histories group and add new entries
            self._merge_into_existing_checkpoint(
                h5_file, unwritten_shot_histories
            )

    def _merge_into_existing_checkpoint(
        self, h5_file, unwritten_shot_histories: dict[int, History]
    ) -> None:
        """Merge new shots into an existing checkpoint file.

        This method handles both dataset-based and group-based storage formats:
        - If iterable contains datasets, extend the datasets
        - If iterable contains groups, add individual entries

        Parameters
        ----------
        h5_file:
            Open HDF5 file object.
        unwritten_shot_histories:
            Dictionary mapping shot indices to History objects to be written.
        """
        from loqs.internal.serializable import Serializable

        # Find the root group (should be the only one at root level)
        if len(h5_file.keys()) != 1:
            raise ValueError(
                "Invalid checkpoint file structure - expected single root group"
            )

        root_group_name = list(h5_file.keys())[0]
        root_group = h5_file[root_group_name]

        # Navigate to shot_histories group
        if "shot_histories" not in root_group:
            raise ValueError(
                "Invalid checkpoint file structure - missing shot_histories group"
            )

        shot_histories_group = root_group["shot_histories"]

        # Navigate to the dict group
        if "dict" not in shot_histories_group:
            raise ValueError(
                "Invalid checkpoint file structure - missing dict group in shot_histories"
            )

        dict_group = shot_histories_group["dict"]

        # Navigate to keys and values iterable groups
        if "keys" not in dict_group or "values" not in dict_group:
            raise ValueError(
                "Invalid checkpoint file structure - missing keys/values groups in dict"
            )

        keys_group = dict_group["keys"]
        values_group = dict_group["values"]

        # Navigate to the iterable groups within keys and values
        if "iterable" not in keys_group or "iterable" not in values_group:
            raise ValueError(
                "Invalid checkpoint file structure - missing iterable groups"
            )

        keys_iterable_group = keys_group["iterable"]
        values_iterable_group = values_group["iterable"]

        def _merge_iterable(group, new_data):
            if group.attrs["storage_format"] == "dataset":
                ds = group["data"]
                if ds.maxshape[0] == None:
                    # This is an extendable dataset, slice in new data
                    current_length = len(ds)

                    # Check if we need to resize the dataset first
                    new_size = current_length + len(new_data)
                    if new_size > len(ds):
                        ds.resize((new_size,))

                    ds[current_length : current_length + len(new_data)] = (
                        new_data[:]
                    )
                else:
                    # Not extendable, do a full load and rewrite as extendable
                    all_data = list(ds) + new_data
                    del group["data"]
                    HDF5Encoder._encode_iterable_dataset(
                        group, all_data, extendable_dataset=True
                    )
            else:
                # This is groups format, just add groups
                next_index = len(group.keys())

                for i in new_data:
                    # track_order=True for consistent group storage; encode_cache
                    # reuses this object's persistent cache so a shared object is a
                    # cheap reference here, not re-expanded from scratch.
                    item_group = group.create_group(
                        str(next_index), track_order=True
                    )
                    Serializable.encode(
                        i,
                        format="hdf5",
                        h5_group=item_group,
                        encode_cache=self._checkpoint_encode_cache,
                    )
                    next_index += 1

        _merge_iterable(
            keys_iterable_group, list(unwritten_shot_histories.keys())
        )
        _merge_iterable(
            values_iterable_group, list(unwritten_shot_histories.values())
        )

    def _write_full_checkpoint_structure(
        self, h5_file, shot_histories: dict[int, History]
    ) -> None:
        """Write a complete ProgramResults structure to HDF5 using standard encoding.

        Parameters
        ----------
        h5_file:
            Open HDF5 file object.
        shot_histories:
            Dictionary mapping shot indices to History objects.
        """
        # Create a temporary ProgramResults object with just the shot_histories
        # This will use the standard Serializable encoding
        temp_results = ProgramResults(shot_histories=shot_histories)

        # Reuses this object's own persistent encode_cache (no reset_encode_id),
        # so a shared object stays cheaply referenced across checkpoint() calls.
        Serializable.encode(
            temp_results,
            format="hdf5",
            h5_group=h5_file,
            encode_cache=self._checkpoint_encode_cache,
        )

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
            merged output under), loads the un-suffixed `checkpoint.h5`.
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
            pattern = "checkpoint.h5"

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
        """Merge every per-worker checkpoint file in `checkpoint_dir` into one file.

        Streams one worker's file in at a time -- decoding it, merging just
        its shots into the output, then letting it be garbage collected
        before moving to the next -- rather than loading every worker's
        shots into `self.shot_histories` up front and writing once at the
        end. This bounds peak memory to roughly one worker's own share of
        the total result, not the whole result, regardless of how many
        workers or shots are involved. Writes to a temporary file first and
        renames it into place only once complete, so a crash partway
        through never leaves a corrupt or partial `output_file` behind.

        Parameters
        ----------
        checkpoint_dir:
            Directory containing checkpoint files to consolidate.
        output_file:
            Path for the consolidated output file. If None, writes to the
            un-suffixed `checkpoint_dir / "checkpoint.h5"` -- the same
            filename `checkpoint(worker_id=None)` itself writes to, so a
            single-writer run and a many-writer run both end up readable
            via `load_checkpoint(worker_id=None)`.
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
            output_file = checkpoint_dir / "checkpoint.h5"
        else:
            output_file = Path(output_file)

        worker_files = sorted(checkpoint_dir.glob("worker_*_checkpoint.h5"))

        # Write to a fresh temp file, not `output_file` directly -- avoids
        # ever reading and writing the same file at once (possible if a
        # caller passes an `output_file` that coincides with one of the
        # globbed worker files) and keeps `output_file` atomic: it only
        # ever appears once fully written.
        tmp_output_file = output_file.with_name(output_file.name + ".tmp")
        if tmp_output_file.exists():
            tmp_output_file.unlink()

        with h5py.File(tmp_output_file, "a") as out_f:
            for worker_file in worker_files:
                self._stream_checkpoint_file_into(worker_file, out_f)
            if len(out_f.keys()) == 0:
                # No worker files (or all were empty) -- still leave behind
                # a validly-decodable, if empty, ProgramResults structure
                # rather than a bare, structure-less HDF5 file.
                self._write_full_checkpoint_structure(out_f, {})

        tmp_output_file.replace(output_file)

        if delete_originals:
            for worker_file in worker_files:
                worker_file.unlink()

        return output_file

    def _stream_checkpoint_file_into(
        self, filename: Path, out_h5_file
    ) -> None:
        """Read one worker's checkpoint file into a throwaway `ProgramResults`,
        merge just its shots into `out_h5_file`, then let it be garbage
        collected -- the actual "streaming" step `consolidate_checkpoints`
        relies on to keep peak memory bounded to one worker's share of the
        total result at a time.

        Parameters
        ----------
        filename:
            Path to the worker checkpoint file to read.
        out_h5_file:
            Already-open, writable HDF5 file/group to merge this worker's
            shots into.
        """
        with h5py.File(filename, "r") as in_f:
            loaded = Serializable.decode(in_f, format="hdf5")
        assert isinstance(loaded, ProgramResults)
        if loaded.shot_histories:
            self._update_single_file_checkpoint(
                out_h5_file, loaded.shot_histories
            )

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
        if (
            hasattr(self, "_lazy_loading_enabled")
            and self._lazy_loading_enabled
        ):
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
            if shot_index in self.shot_histories:
                return self.shot_histories[shot_index]

            # If not in memory and we have checkpoint files, try to load
            if (
                self._checkpoint_dir is not None
                and shot_index in self._unwritten_shots
            ):
                # Load from checkpoint and add to shot_histories
                if self._load_shot_from_checkpoint(shot_index):
                    return self.shot_histories[shot_index]

            return None

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
        if self._checkpoint_dir is None or not self._checkpoint_dir.exists():
            return False

        # Try to find the shot in this writer's own checkpoint file
        if self._worker_id is not None:
            checkpoint_file = (
                self._checkpoint_dir
                / f"worker_{self._worker_id}_checkpoint.h5"
            )
        else:
            checkpoint_file = self._checkpoint_dir / "checkpoint.h5"

        if not checkpoint_file.exists():
            return False

        return self._load_shot_from_single_file(checkpoint_file, shot_index)

    def _load_shot_from_single_file(
        self, filename: Path, shot_index: int
    ) -> bool:
        """Load a shot from a checkpoint file using standard Serializable decoding.

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
                # Load the full ProgramResults from the checkpoint
                loaded_results = Serializable.decode(f, format="hdf5")
                assert isinstance(loaded_results, ProgramResults)

                # Check if the shot exists in the loaded results
                if shot_index in loaded_results.shot_histories:
                    history = loaded_results.shot_histories[shot_index]

                    if self._lazy_loading_enabled:
                        self._memory_cache[shot_index] = history
                    else:
                        self.shot_histories[shot_index] = history
                        # Remove from unwritten_shots since it's already checkpointed
                        if shot_index in self._unwritten_shots:
                            self._unwritten_shots.remove(shot_index)

                    return True
        except (KeyError, OSError, ValueError):
            return False

        return False
