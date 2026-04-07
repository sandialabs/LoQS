#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`ProgramResults` definition.
"""

from __future__ import annotations

from collections import Counter
from typing import ClassVar
from pathlib import Path
import h5py
import numpy as np
from datetime import datetime

from loqs.internal import Displayable, Serializable
from loqs.core.history import (
    History,
    HistoryCastableTypes,
    HistoryCollectDataIndexTypes,
)
from loqs.core import Frame

# Import QuantumProgram to avoid circular imports - we'll use it in type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loqs.core.quantumprogram import QuantumProgram


class ProgramResults(Displayable):
    """A container for the results of a quantum program execution.

    This class stores the shot histories and provides methods for collecting
    and analyzing data from the executed shots. It replaces the direct storage
    of shot histories in QuantumProgram.
    """

    CACHE_ON_SERIALIZE: ClassVar[bool] = True

    SERIALIZE_ATTRS = [
        "shot_histories",
        "_unwritten_shots",
        "name",
        "parent_program",
    ]

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
        """Record of shot :class:`.History` objects, mapped by shot index."""

        self._unwritten_shots = set()
        """Set of shot indices that have not been written to checkpoint files yet."""

        self._checkpoint_dir = None
        """Directory where checkpoint files are stored."""

        self._checkpoint_strategy = None
        """Current checkpointing strategy being used."""

        self._worker_id = None
        """Worker ID for parallel checkpointing."""

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

    def add_shot(self, shot_index: int, history: HistoryCastableTypes) -> None:
        """Add a shot history to the results.

        Parameters
        ----------
        shot_index:
            The index of the shot to add.
        history:
            The History object for the shot.
        """
        history = History.cast(history)
        self.shot_histories[shot_index] = history
        self._unwritten_shots.add(shot_index)

    def collect_shot_data(
        self,
        key: str,
        indices: HistoryCollectDataIndexTypes,
        strip_none_entries: bool = False,
        return_counter: bool = False,
    ) -> list | Counter:
        """Collate frame data over executed shots.

        Parameters
        ----------
        key:
            See ``key`` in :meth:`.History.collect_data`

        indices:
            See ``indices`` in :meth:`.History.collect_data`

        strip_none_entries:
            See ``strip_none_entries`` in :meth:`.History.collect_data`

        return_counter:
            Whether to return using a collections.Counter or not (default).

        Returns
        -------
        list
            List of :meth:`.History.collect_data` outputs per shot
        """
        data = [
            h.collect_data(key, indices, strip_none_entries)
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
    def from_decoded_attrs(cls, attr_dict) -> "ProgramResults":
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
        strategy: str = "single_file",
        batch_size: int = 1,
        current_batch_index: int = 1,
        worker_id: int | None = None,
    ) -> None:
        """Checkpoint the program results to disk.

        Parameters
        ----------
        checkpoint_dir:
            Directory to store checkpoint files. If None, uses a temporary directory.
        strategy:
            Checkpointing strategy. Options are "single_file" (all shots in one file)
            or "per_batch" (one file per batch).
        batch_size:
            Number of shots per batch.
        current_batch_index:
            Index of the current batch being checkpointed.
        worker_id:
            Worker ID for parallel checkpointing. If None, assumes single worker.
        """
        # Validate strategy early
        if strategy not in ("single_file", "per_batch"):
            raise ValueError(f"Unknown checkpoint strategy: {strategy}")

        # Set up checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path("./checkpoints")
        else:
            checkpoint_dir = Path(checkpoint_dir)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_strategy = strategy
        self._worker_id = worker_id

        # Get unwritten shots
        unwritten_shots = self.get_unwritten_shots()
        if not unwritten_shots:
            return  # Nothing to checkpoint

        # Determine which shots to checkpoint based on batch
        # For single_file strategy, checkpoint all shots that belong to batches <= current_batch_index
        # For per_batch strategy, checkpoint only shots that belong to current_batch_index
        if batch_size == 1 and current_batch_index == 1:
            # Special case: when using default batching (batch_size=1, current_batch_index=1),
            # checkpoint all unwritten shots to avoid confusion
            shots_to_checkpoint = list(unwritten_shots)
        else:
            # Normal batching logic
            shots_to_checkpoint = []
            for shot_index in unwritten_shots:
                # Use original batch calculation: (shot_index + 1) // batch_size
                batch_idx = (shot_index + 1) // batch_size

                if strategy == "single_file":
                    # For single_file strategy, checkpoint all shots with 1 <= batch_idx <= current_batch_index
                    # This allows "late" shots that belong to previous batches to be checkpointed
                    # but excludes shots with batch_idx = 0
                    if batch_idx >= 1 and batch_idx <= current_batch_index:
                        shots_to_checkpoint.append(shot_index)
                else:  # per_batch strategy
                    # For per_batch strategy, only checkpoint shots that match current_batch_index exactly
                    if batch_idx == current_batch_index:
                        shots_to_checkpoint.append(shot_index)

            # If no shots were selected for this batch, but we have unwritten shots,
            # checkpoint all unwritten shots as a fallback
            if not shots_to_checkpoint and unwritten_shots:
                shots_to_checkpoint = list(unwritten_shots)

        if not shots_to_checkpoint:
            return  # No shots in this batch

        # Create checkpoint filename
        if strategy == "single_file":
            if worker_id is not None:
                filename = checkpoint_dir / f"worker_{worker_id}_checkpoint.h5"
            else:
                filename = checkpoint_dir / "checkpoint.h5"
        elif strategy == "per_batch":
            if worker_id is not None:
                filename = (
                    checkpoint_dir
                    / f"worker_{worker_id}_batch_{current_batch_index}.h5"
                )
            else:
                filename = checkpoint_dir / f"batch_{current_batch_index}.h5"
        else:
            raise ValueError(f"Unknown checkpoint strategy: {strategy}")

        # Write checkpoint
        self._write_checkpoint_file(filename, shots_to_checkpoint, strategy)

        # Mark shots as written
        self.mark_shots_as_written(shots_to_checkpoint)

        # Implement lazy loading: remove written shots from memory
        if self._lazy_loading_enabled:
            for shot_index in shots_to_checkpoint:
                if shot_index in self.shot_histories:
                    del self.shot_histories[shot_index]

    def _write_checkpoint_file(
        self,
        filename: Path,
        shot_indices: list[int],
        strategy: str,
    ) -> None:
        """Write checkpoint data to an HDF5 file using standard Serializable encoding.

        Parameters
        ----------
        filename:
            Path to the checkpoint file.
        shot_indices:
            List of shot indices to write to the checkpoint.
        strategy:
            Checkpointing strategy being used.
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
            if strategy == "single_file":
                # For single file strategy, we need to merge into existing structure
                self._update_single_file_checkpoint(
                    f, unwritten_shot_histories
                )
            else:
                # For per_batch strategy, create a new file or overwrite
                # Write all shots using standard encoding
                self._write_per_batch_checkpoint(f, unwritten_shot_histories)

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
        - If iterable contains datasets (modern format), extend the datasets
        - If iterable contains groups (legacy format), add individual entries

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

        # Check if we have dataset-based storage (modern format)
        if "data" in keys_iterable_group:
            # Dataset-based storage - extend existing datasets
            existing_keys = list(keys_iterable_group["data"][()])
            
            # Load existing values from groups
            existing_values = []
            for i in range(len(existing_keys)):
                if str(i) in values_iterable_group:
                    value_group = values_iterable_group[str(i)]
                    existing_value = Serializable.decode(value_group, format="hdf5")
                    existing_values.append(existing_value)
            
            # Merge existing and new shot histories
            existing_shot_histories = dict(zip(existing_keys, existing_values))
            merged_shot_histories = {**existing_shot_histories, **unwritten_shot_histories}
            
            # Delete old data
            del keys_iterable_group["data"]
            for key in list(values_iterable_group.keys()):
                del values_iterable_group[key]
            
            # Write merged data
            merged_keys = list(merged_shot_histories.keys())
            keys_iterable_group.create_dataset("data", data=merged_keys)
            
            for i, (shot_index, history) in enumerate(merged_shot_histories.items()):
                value_item_group = values_iterable_group.create_group(str(i))
                Serializable.encode(history, format="hdf5", h5_group=value_item_group)
            
        else:
            # Group-based storage - add individual entries (legacy format)
            current_keys = list(keys_iterable_group.keys())
            current_values = list(values_iterable_group.keys())

            if len(current_keys) != len(current_values):
                raise ValueError(
                    "Invalid checkpoint file - keys and values have different lengths"
                )

            next_index = len(current_keys)

            # Add each new shot to the checkpoint
            for shot_index, history in unwritten_shot_histories.items():
                # Add key to keys iterable using proper integer index
                key_item_group = keys_iterable_group.create_group(str(next_index))
                Serializable.encode(
                    shot_index, format="hdf5", h5_group=key_item_group
                )

                # Add value (History) to values iterable using proper integer index
                value_item_group = values_iterable_group.create_group(
                    str(next_index)
                )
                Serializable.encode(
                    history, format="hdf5", h5_group=value_item_group
                )

                next_index += 1

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

        # Encode the ProgramResults object to HDF5
        Serializable.encode(temp_results, format="hdf5", h5_group=h5_file)

    def _write_per_batch_checkpoint(
        self, h5_file, shot_histories: dict[int, History]
    ) -> None:
        """Write a per-batch checkpoint file using standard encoding.

        Parameters
        ----------
        h5_file:
            Open HDF5 file object.
        shot_histories:
            Dictionary mapping shot indices to History objects.
        """
        # For per-batch strategy, write the full ProgramResults structure
        # This creates a standalone checkpoint file
        self._write_full_checkpoint_structure(h5_file, shot_histories)

    def load_checkpoint(
        self,
        checkpoint_dir: str | Path | None = None,
        strategy: str = "single_file",
        worker_id: int | None = None,
    ) -> None:
        """Load checkpoint data from disk.

        Parameters
        ----------
        checkpoint_dir:
            Directory containing checkpoint files. If None, uses the default checkpoint directory.
        strategy:
            Checkpointing strategy that was used. Options are "single_file" or "per_batch".
        worker_id:
            Worker ID for parallel checkpointing. If None, assumes single worker.
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

        # Find checkpoint files based on strategy
        if strategy == "single_file":
            if worker_id is not None:
                pattern = f"worker_{worker_id}_checkpoint.h5"
            else:
                pattern = "checkpoint.h5"

            checkpoint_file = checkpoint_dir / pattern
            if checkpoint_file.exists():
                self._load_single_checkpoint_file(checkpoint_file)

        elif strategy == "per_batch":
            if worker_id is not None:
                pattern = f"worker_{worker_id}_batch_*.h5"
            else:
                pattern = "batch_*.h5"

            # Load all batch files
            for batch_file in checkpoint_dir.glob(pattern):
                self._load_single_checkpoint_file(batch_file)

        else:
            raise ValueError(f"Unknown checkpoint strategy: {strategy}")

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
        strategy: str = "single_file",
    ) -> Path:
        """Consolidate multiple checkpoint files into a single file.

        Parameters
        ----------
        checkpoint_dir:
            Directory containing checkpoint files to consolidate.
        output_file:
            Path for the consolidated output file. If None, creates a file in checkpoint_dir.
        delete_originals:
            Whether to delete original checkpoint files after consolidation.
        strategy:
            The checkpointing strategy that was used.

        Returns
        -------
        Path
            Path to the consolidated checkpoint file.
        """
        # Validate strategy early
        if strategy not in ("single_file", "per_batch"):
            raise ValueError(f"Unknown checkpoint strategy: {strategy}")

        if checkpoint_dir is None:
            if self._checkpoint_dir is None:
                checkpoint_dir = Path("./checkpoints")
            else:
                checkpoint_dir = self._checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)

        if output_file is None:
            output_file = (
                checkpoint_dir
                / f"consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            )
        else:
            output_file = Path(output_file)

        # Load all checkpoint files into self
        if strategy == "single_file":
            # Look for worker checkpoint files
            for worker_file in checkpoint_dir.glob("worker_*_checkpoint.h5"):
                self._load_single_checkpoint_file(worker_file)

            # Look for main checkpoint file
            main_file = checkpoint_dir / "checkpoint.h5"
            if main_file.exists():
                self._load_single_checkpoint_file(main_file)

        elif strategy == "per_batch":
            # Look for all batch files
            for batch_file in checkpoint_dir.glob("worker_*_batch_*.h5"):
                self._load_single_checkpoint_file(batch_file)

            for batch_file in checkpoint_dir.glob("batch_*.h5"):
                self._load_single_checkpoint_file(batch_file)

        else:
            raise ValueError(f"Unknown checkpoint strategy: {strategy}")

        # Write consolidated data to new file
        self.write(output_file)

        # Optionally delete original files
        if delete_originals:
            if strategy == "single_file":
                for worker_file in checkpoint_dir.glob(
                    "worker_*_checkpoint.h5"
                ):
                    worker_file.unlink()
                main_file = checkpoint_dir / "checkpoint.h5"
                if main_file.exists():
                    main_file.unlink()

            elif strategy == "per_batch":
                for batch_file in checkpoint_dir.glob("worker_*_batch_*.h5"):
                    batch_file.unlink()
                for batch_file in checkpoint_dir.glob("batch_*.h5"):
                    batch_file.unlink()

        return output_file

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

        # Try to find the shot in any checkpoint file
        success = False

        # Look for single file checkpoints
        if self._checkpoint_strategy == "single_file":
            if self._worker_id is not None:
                checkpoint_file = (
                    self._checkpoint_dir
                    / f"worker_{self._worker_id}_checkpoint.h5"
                )
            else:
                checkpoint_file = self._checkpoint_dir / "checkpoint.h5"

            if checkpoint_file.exists():
                success = self._load_shot_from_single_file(
                    checkpoint_file, shot_index
                )

        # Look for per-batch checkpoints
        elif self._checkpoint_strategy == "per_batch":
            if self._worker_id is not None:
                pattern = f"worker_{self._worker_id}_batch_*.h5"
            else:
                pattern = "batch_*.h5"

            # Check all batch files
            for batch_file in self._checkpoint_dir.glob(pattern):
                if self._load_shot_from_batch_file(batch_file, shot_index):
                    success = True
                    break

        return success

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

    def _load_shot_from_batch_file(
        self, filename: Path, shot_index: int
    ) -> bool:
        """Load a shot from a batch checkpoint file using standard Serializable decoding.

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
        # For batch files, we use the same loading method as single files
        # since they both use the standard Serializable encoding
        return self._load_shot_from_single_file(filename, shot_index)
